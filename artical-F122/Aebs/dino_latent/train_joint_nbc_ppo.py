"""PPO + signed NBC joint pilot in the EXISTING train-image replay simulator.

Real on-policy sampling in this replay environment, not online CARLA images.
Nearest-image pose mismatch is recorded, not claimed to be covered. Safety
scores are detached and weighted by action likelihood ratios (no dynamics
gradient). PPO update follows SB3 2.4.x, with a PPO-Barrier-inspired cost term.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

from Aebs.dino_latent.diagnose_expanded_failures import load_fixed_inputs
from Aebs.dino_latent.diagnose_environment_barrier import write_json
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.run_expanded_ppo import AppearanceEnv
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.train_latent_ppo import evaluate
from Aebs.dino_latent.train_image_transition_nbc import SignedNBC, barrier_losses, region_grids
from Aebs.dino_latent.nbc_retention import retention_scores, retention_failures, guarded_nbc_update
from Aebs.dino_latent.nbc_multistep import check_transition_chain,multistep_scores


class LoggedReplay(AppearanceEnv):
    """Same train-only image lookup/physics/reward; reset in the task init set."""
    def __init__(self, representation, data, latent, rows, vt_scale, appearance=None):
        super().__init__(representation, data, latent, rows, "image_nearest", appearance)
        self.vt_scale = vt_scale
        self.episode_id = 0

    def _latent(self, distance_norm):
        pool = self.pools[self.appearance]
        d = distance_norm*self.distance_scale_m
        i = pool[np.argmin(abs(self.image_distances_m[pool]-d))]
        self.last_image_row = int(self.source_indices[i])
        self.last_gap = float(abs(self.image_distances_m[i]-d))
        self.lookup_errors_m.append(self.last_gap)
        return self.image_latents[i].astype(np.float32)

    def physical_vt(self):
        return np.array([self.base.state[0]*self.distance_scale_m/self.vt_scale, self.base.state[1]],dtype=np.float32)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self,seed=seed)
        self.episode_id += 1
        if self.fixed_appearance is None:
            names=sorted(self.pools)
            self.appearance=names[int(self.np_random.integers(len(names)))]
        obs=self.reset_to(float(self.np_random.uniform(15.,16.)),float(self.np_random.uniform(2.5,3.)))
        return obs, {}

    def step(self, action):
        before=self.physical_vt()
        row,gap=self.last_image_row,self.last_gap
        obs,reward,terminated,truncated,info=super().step(action)
        info.update(physical_state_vt=before,physical_next_vt=self.physical_vt(),
                    image_row=row,lookup_error_m=gap,appearance=self.appearance,
                    episode_id=self.episode_id,terminated=terminated,truncated=truncated)
        return obs,reward,terminated,truncated,info


class SafeSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    safety_cost: torch.Tensor


class SafetyBuffer(RolloutBuffer):
    def reset(self):
        super().reset()
        if self.n_envs != 1: raise ValueError("pilot supports one chronologically logged environment")
        self.physical_x=np.zeros((self.buffer_size,2),np.float32)
        self.physical_y=np.zeros_like(self.physical_x)
        self.safety_cost=np.zeros(self.buffer_size,np.float32)
        self.executed_actions=np.zeros((self.buffer_size,1),np.float32)
        self.raw_rewards=np.zeros(self.buffer_size,np.float32)
        self.episode_ids=np.zeros(self.buffer_size,np.int64)
        self.image_rows=np.zeros(self.buffer_size,np.int64)
        self.lookup_gaps=np.zeros(self.buffer_size,np.float32)
        self.done_flags=np.zeros((self.buffer_size,2),bool)
        self.appearances=np.empty(self.buffer_size,dtype="U24")

    def _get_samples(self, batch_inds, env=None):
        return SafeSamples(*super()._get_samples(batch_inds,env),self.to_torch(self.safety_cost[batch_inds]))


def safety_surrogate(log_prob, old_log_prob, detached_cost):
    return (torch.exp(log_prob-old_log_prob)*detached_cost.detach()).mean()


class TrajectoryLogger(BaseCallback):
    def _on_rollout_start(self):
        self.directory=self.model.archive_dir/("rollout_%03d" % (len(self.model.joint_history)+1))
        self.directory.mkdir(exist_ok=False)
        torch.save(self.model.policy.state_dict(),self.directory/"sampling_policy.pt")

    def _on_step(self):
        b=self.model.rollout_buffer
        i=b.pos
        info=self.locals["infos"][0]
        b.physical_x[i]=info["physical_state_vt"]
        b.physical_y[i]=info["physical_next_vt"] # terminal state, NOT autoreset observation
        b.executed_actions[i]=self.locals["clipped_actions"][0]
        b.raw_rewards[i]=self.locals["rewards"][0]
        b.episode_ids[i]=info["episode_id"]
        b.image_rows[i]=info["image_row"]
        b.lookup_gaps[i]=info["lookup_error_m"]
        b.done_flags[i]=[info["terminated"],info["truncated"]]
        b.appearances[i]=info["appearance"]
        return True

    def _on_rollout_end(self):
        b=self.model.rollout_buffer
        check_transition_chain(b.physical_x,b.physical_y,b.episode_ids,b.done_flags[:,0],b.done_flags[:,1])
        with torch.no_grad():
            c=self.model.nbc(torch.from_numpy(b.physical_x))
            cn=self.model.nbc(torch.from_numpy(b.physical_y))
            raw_single=F.relu(cn-.9*c+.01).numpy()
            scores,horizons,counts=multistep_scores(c,cn,b.episode_ids,b.done_flags[:,0],b.done_flags[:,1],steps=self.model.barrier_steps)
            raw=scores.numpy()
        self.model.latest_window_diagnostic=dict(barrier_steps=self.model.barrier_steps,valid_windows_per_horizon=counts,
            full_horizon_starts=int((horizons==self.model.barrier_steps).sum()),partial_starts=int((horizons<self.model.barrier_steps).sum()),
            single_positive=int((raw_single>0).sum()),multi_positive=int((raw>0).sum()),
            additional_positive=int(((raw_single<=0)&(raw>0)).sum()))
        b.safety_cost=((raw-raw.mean())/max(1e-4,float(raw.std()))).astype(np.float32)
        torch.save(self.model.nbc.state_dict(),self.directory/"scoring_nbc.pt")
        np.savez_compressed(self.directory/"trajectory.npz",observations=b.observations.copy(),
            sampled_actions=b.actions.copy(),executed_actions=b.executed_actions,
            old_log_prob=b.log_probs.copy(),physical_x=b.physical_x,physical_y=b.physical_y,
            raw_rewards=b.raw_rewards,ppo_bootstrapped_rewards=b.rewards.copy(),
            terminated=b.done_flags[:,0],truncated=b.done_flags[:,1],episode_ids=b.episode_ids,
            episode_starts=b.episode_starts.copy(),image_rows=b.image_rows,appearances=b.appearances,
            lookup_gap_m=b.lookup_gaps,raw_safety_violation=raw,centered_safety_cost=b.safety_cost,
            single_step_safety_violation=raw_single,valid_barrier_horizons=horizons.numpy(),
            returns=b.returns.copy(),advantages=b.advantages.copy())
        self.model.latest_rollout_dir=self.directory
        self.model.latest_raw_cost=raw.copy()


class JointNBCPPO(PPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params()+["nbc","nbc_optimizer","archive_dir","latest_rollout_dir","latest_raw_cost",
            "old_nbc_x","old_nbc_y","guard_x","guard_y","guard_initial","guard_unsafe","retention_generator"]

    def attach_nbc(self, nbc, archive_dir, coefficient=.05):
        self.nbc=nbc
        self.nbc_optimizer=torch.optim.Adam(nbc.parameters(),lr=3e-4)
        self.archive_dir=Path(archive_dir)
        self.archive_dir.mkdir(exist_ok=False)
        self.safety_coefficient=coefficient
        self.joint_history=[]
        self.safety_feedback=True
        self.retention_enabled=False
        self.barrier_steps=1

    def attach_retention(self, train_pairs, scale):
        with np.load(train_pairs,allow_pickle=False) as data:
            self.old_nbc_x=torch.from_numpy(data["state"].copy())
            self.old_nbc_y=torch.from_numpy(data["image_next_state"].copy())
        if (self.old_nbc_x.shape!=self.old_nbc_y.shape or
            not torch.isfinite(self.old_nbc_x).all() or not torch.isfinite(self.old_nbc_y).all()):
            raise ValueError("invalid old train transitions")
        # A separate generator leaves the PPO sampling/shuffling RNG untouched.
        self.retention_generator=torch.Generator().manual_seed(7)
        ids=torch.randperm(len(self.old_nbc_x),generator=self.retention_generator)[:16384]
        self.guard_x,self.guard_y=self.old_nbc_x[ids],self.old_nbc_y[ids]
        self.guard_initial,self.guard_unsafe=region_grids(scale,81,torch.device("cpu"))
        self.retention_reference=retention_scores(self.nbc,self.guard_x,self.guard_y,self.guard_initial,self.guard_unsafe)
        if retention_failures(self.retention_reference,self.retention_reference):
            raise ValueError("initial NBC fails train-only regional retention margin gate")
        self.retention_enabled=True
        np.savez_compressed(self.archive_dir.parent/"retention_train_guard.npz",train_indices=ids.numpy())

    def train(self):
        # SB3 PPO clipped reward surrogate + detached likelihood-weighted cost.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        b=self.rollout_buffer
        with torch.no_grad():
            _,lp,_=self.policy.evaluate_actions(torch.as_tensor(b.observations.reshape(-1,33)),torch.as_tensor(b.actions.reshape(-1,1)))
        logp_error=float(np.max(abs(lp.numpy()-b.log_probs.reshape(-1))))
        if logp_error>1e-4: raise ValueError("sampling policy/old log-prob mismatch")
        self.safety_coefficient=(min(1.,self.safety_coefficient+.05*float(self.latest_raw_cost.mean()))
                                 if self.safety_feedback else 0.)
        nu=self.safety_coefficient
        policy_before={k:v.clone() for k,v in self.policy.state_dict().items()}
        mean_params=list(self.policy.mlp_extractor.policy_net.parameters())+list(self.policy.action_net.parameters())
        maximum_safety_grad=0.
        losses, kls=[],[]
        stop=False
        for epoch in range(self.n_epochs):
            for batch in b.get(self.batch_size):
                values,log_prob,entropy=self.policy.evaluate_actions(batch.observations,batch.actions)
                advantages=batch.advantages
                advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
                ratio=torch.exp(log_prob-batch.old_log_prob)
                reward_loss=-torch.min(advantages*ratio,advantages*ratio.clamp(.8,1.2)).mean()
                cost=safety_surrogate(log_prob,batch.old_log_prob,batch.safety_cost)
                loss=(reward_loss+nu*cost)/(1.+nu)+self.vf_coef*F.mse_loss(values.flatten(),batch.returns)-self.ent_coef*entropy.mean()
                with torch.no_grad():
                    lr=log_prob-batch.old_log_prob
                    kl=float((torch.exp(lr)-1.-lr).mean())
                kls.append(kl)
                if self.target_kl and kl>1.5*self.target_kl:
                    stop=True
                    break
                if not losses:
                    grads=torch.autograd.grad(nu*cost/(1.+nu),mean_params,retain_graph=True,allow_unused=True)
                    maximum_safety_grad=float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
                self.policy.optimizer.step()
                losses.append(float(loss.detach()))
            self._n_updates+=1
            if stop: break
        # Update NBC only after scoring/updating policy: it cannot erase its own
        # risk signal before that signal reaches the PPO actor.
        x,y=torch.from_numpy(b.physical_x),torch.from_numpy(b.physical_y)
        scale=float(self.nbc_scale)
        initial,unsafe=region_grids(scale,32,torch.device("cpu"))
        retention=None
        if self.retention_enabled:
            retention=guarded_nbc_update(self.nbc,self.nbc_optimizer,x,y,self.old_nbc_x,self.old_nbc_y,
                initial,unsafe,self.guard_x,self.guard_y,self.guard_initial,self.guard_unsafe,
                self.retention_reference,self.retention_generator,barrier_steps=self.barrier_steps,
                episode_ids=b.episode_ids,terminated=b.done_flags[:,0],truncated=b.done_flags[:,1])
            nbc_loss=retention["optimized_loss"]
        else:
            self.nbc.train()
            nbc_loss=0.
            for _ in range(10):
                self.nbc_optimizer.zero_grad(set_to_none=True)
                loss,_=barrier_losses(self.nbc,x,y,initial,unsafe,.1,.01)
                if self.barrier_steps>1:
                    loss=(F.relu(self.nbc(initial)+.01).mean()+F.relu(.01-self.nbc(unsafe)).mean()+
                          multistep_scores(self.nbc(x),self.nbc(y),b.episode_ids,b.done_flags[:,0],b.done_flags[:,1],steps=self.barrier_steps)[0].mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nbc.parameters(),1.)
                self.nbc_optimizer.step()
                nbc_loss=float(loss.detach())
        self.nbc.eval()
        drift=max(float((v-policy_before[k]).abs().max()) for k,v in self.policy.state_dict().items() if k.startswith(("mlp_extractor.policy_net.","action_net.")))
        row=dict(rollout=len(self.joint_history)+1,timesteps=int(self.num_timesteps),
            sampling_log_prob_max_error=logp_error,raw_safety_violation_mean=float(self.latest_raw_cost.mean()),
            positive_safety_samples=int((self.latest_raw_cost>0).sum()),safety_coefficient=nu,
            actor_safety_gradient_norm=maximum_safety_grad,actor_parameter_max_drift=drift,
            optimizer_steps=len(losses),approx_kl_max=max(kls),nbc_training_loss=nbc_loss,
            lookup_gap_mean_m=float(b.lookup_gaps.mean()),lookup_gap_max_m=float(b.lookup_gaps.max()),
            terminated_episodes=int(b.done_flags[:,0].sum()),timeout_episodes=int(b.done_flags[:,1].sum()),
            safety_feedback=self.safety_feedback,nbc_retention=retention,
            multistep_windows=self.latest_window_diagnostic)
        self.joint_history.append(row)
        write_json(self.latest_rollout_dir/"update.json",row)
        write_json(self.archive_dir.parent/"joint_history.json",self.joint_history)
        print("PPO + NBC joint update",row,flush=True)


def matched_nbc(policy, critic, fixed, q_model, rep_scale, vt_scale, split, critic_label="frozen_initial_nbc"):
    ids=np.array([i for i,r in enumerate(fixed["rows"]) if r["split"]==split])
    image_ids=np.repeat(ids,60)
    d=fixed["distances"][image_ids]
    v=np.tile(np.linspace(.05,3.,60,dtype=np.float32),len(ids))
    keep=(d>6.)&(d<=16.)
    image_ids,d,v=image_ids[keep],d[keep],v[keep]
    violations=escapes=unsafe_i=unsafe_q=0
    mae=maximum=0.
    for start in range(0,len(d),4096):
        ds,vs=d[start:start+4096],v[start:start+4096]
        with torch.no_grad():
            zi=torch.from_numpy(fixed["latents"][image_ids[start:start+4096]])
            zq=q_model(torch.from_numpy(ds[:,None]/rep_scale))
            speed=torch.from_numpy(vs[:,None])
            oi,oq=torch.cat((zi,speed),1),torch.cat((zq,speed),1)
            ai=policy.predict(oi.numpy(),deterministic=True)[0].flatten()
            aq=policy.predict(oq.numpy(),deterministic=True)[0].flatten()
            xi=np.column_stack((ds/vt_scale,vs)).astype(np.float32)
            yi=np.column_stack(((ds-.05*vs)/vt_scale,np.clip(vs-.05*ai,0,3))).astype(np.float32)
            yq=np.column_stack(((ds-.05*vs)/vt_scale,np.clip(vs-.05*aq,0,3))).astype(np.float32)
            c=critic(torch.from_numpy(xi)).numpy()
            cn=critic(torch.from_numpy(yi)).numpy()
        violations+=int(((.9*c-cn)<-1e-6).sum())
        escapes+=int(((c<=0)&(cn>0)).sum())
        unsafe_i+=int(((yi[:,0]*vt_scale<=6)&(yi[:,1]>.5)).sum())
        unsafe_q+=int(((yq[:,0]*vt_scale<=6)&(yq[:,1]>.5)).sum())
        gap=abs(ai-aq)
        mae+=float(gap.sum()); maximum=max(maximum,float(gap.max()))
    result=dict(pairs=len(d),
                image_next_unsafe=unsafe_i,q_next_unsafe=unsafe_q,dual_path_action_mae=mae/len(d),dual_path_action_max=maximum,
                nearest_lookup_used=False,task_episode_guarantee=False)
    result[critic_label+"_violations"]=violations
    result[critic_label+"_escapes"]=escapes
    return result


def audit_saved_run(directory, fixed, saved):
    """Check stored transitions and FINAL barrier without further training."""
    checkpoint=torch.load(directory/"joint_nbc.pt",map_location="cpu")
    scale=checkpoint["vt_distance_scale_m"]
    critic=SignedNBC(scale,checkpoint["hidden"])
    critic.load_state_dict(checkpoint["state_dict"],strict=True)
    critic.eval()
    initial,unsafe=region_grids(scale,41,torch.device("cpu"))
    with torch.no_grad():
        ci,cu=critic(initial),critic(unsafe)
    rows=[];outcomes=Counter()
    for path in sorted((directory/"rollouts").glob("*/trajectory.npz")):
        with np.load(path) as z:
            x,y=z["physical_x"],z["physical_y"]
            action=z["executed_actions"].reshape(-1)
            expected=np.column_stack((x[:,0]-.05*x[:,1]/scale,np.clip(x[:,1]-.05*action,0,3)))
            same=z["episode_ids"][1:]==z["episode_ids"][:-1]
            done=z["terminated"]|z["truncated"]
            for i in np.where(done)[0]:
                d,v=float(y[i,0]*scale),float(y[i,1])
                outcome=("timeout" if z["truncated"][i] else "unsafe" if d<=6 and v>.5
                         else "success" if 5<=d<=6 and v<=.5 else "stopped_safe" if v<=0 else "out_of_domain")
                outcomes[outcome]+=1
            rows.append(dict(rollout=path.parent.name,steps=len(x),
                physics_max_error=float(abs(expected-y).max()),
                same_episode_continuity_max_error=float(abs(y[:-1][same]-x[1:][same]).max()) if same.any() else 0.,
                terminal_samples=int(done.sum()),lookup_gap_max_m=float(z["lookup_gap_m"].max()),
                raw_actions_outside_bounds=int((abs(z["sampled_actions"])>3).sum())))
    matched=matched_nbc(PPO.load(directory/"joint_ppo.zip",device="cpu"),critic,fixed,saved.policy.q_model,
                        saved.policy.distance_scale_m,scale,"validation",critic_label="final_nbc")
    return dict(scope="saved-run development audit; no additional training; no certificate",
        initial_positive_violations=int((ci>0).sum()),initial_max=float(ci.max()),
        unsafe_nonpositive_violations=int((cu<=0).sum()),unsafe_min=float(cu.min()),
        region_grid_size=41,matched_final_candidate=matched,
        rollouts=rows,sampled_training_terminal_outcomes=dict(outcomes),
        policy_sha256=sha256(directory/"joint_ppo.zip"),nbc_sha256=sha256(directory/"joint_nbc.pt"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path("results/dino_expanded_v1")
    parser.add_argument("--cache-dir",type=Path,default=root/"02_dino_h5")
    parser.add_argument("--representation",type=Path,default=root/"08_R2_group_alignment/dino_safety_latent.pt")
    parser.add_argument("--ppo-dir",type=Path,default=root/"09_R2_ppo")
    parser.add_argument("--spvc-dir",type=Path,default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--final-recheck",type=Path,default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    parser.add_argument("--nbc",type=Path,default=Path("results/environment_nbc_20261014_001/01_signed_nbc_mean/nbc.pt"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--timesteps",type=int,default=8192)
    parser.add_argument("--eval-grid",type=int,default=3)
    parser.add_argument("--nbc-retention",action="store_true",help="train-only old-transition replay and guarded NBC updates")
    parser.add_argument("--diagnostic-dir",type=Path,default=Path("results/environment_nbc_20261013_001/01_frozen_diagnostic"))
    parser.add_argument("--safety-feedback",choices=("on","off"),default="on",help="off: reward-only PPO control, NBC kept as a shadow learner")
    parser.add_argument("--barrier-steps",type=int,choices=(1,5),default=1,help="1 reproduces single-step; 5 uses masked 1..5-step feedback and NBC loss")
    parser.add_argument("--audit-only",action="store_true",help="audit an existing completed run; no training or overwritten metrics")
    args=parser.parse_args()
    if (args.output_dir.exists() and not args.audit_only) or args.timesteps<2048 or args.timesteps%2048 or args.eval_grid<2:
        parser.error("new output; timesteps positive multiples of 2048; eval grid>=2")
    torch.set_num_threads(1)
    started=time.monotonic()
    fixed=load_fixed_inputs(args,parser)
    base,saved,_,identity=load_spvc_pair(args.representation,fixed["controller_path"],args.spvc_dir,args.final_recheck)
    scale=saved.policy.state_distance_scale_m
    if args.audit_only:
        target=args.output_dir/"postrun_audit.json"
        report=json.loads((args.output_dir/"metrics.json").read_text())
        if target.exists() or report["status"] not in ("joint_replay_pilot_complete","reward_only_ppo_control_complete"):
            parser.error("audit requires completed run and a new postrun_audit.json")
        expected=report["source_hashes"]
        paths=dict(actor=args.spvc_dir/"latent_ppo_spvc.pt",old_sbc=args.spvc_dir/"sbc.pt",representation=args.representation,
                   nbc=args.nbc,base_ppo=fixed["controller_path"],cache=fixed["cache_path"],features=fixed["features_path"])
        if "retention_train_pairs" in expected:
            paths.update(retention_train_pairs=args.diagnostic_dir/"train_pairs.npz",retention_diagnostic=args.diagnostic_dir/"metrics.json")
        if {k:sha256(p) for k,p in paths.items()}!=expected:
            raise ValueError("audit source identity changed")
        audit=audit_saved_run(args.output_dir,fixed,saved)
        write_json(target,audit)
        print(json.dumps(audit,indent=2),flush=True)
        print("audit:",target,flush=True)
        return
    nbc_checkpoint=torch.load(args.nbc,map_location="cpu")
    if nbc_checkpoint["source_hashes"]["actor"]!=sha256(args.spvc_dir/"latent_ppo_spvc.pt") or nbc_checkpoint["source_hashes"]["representation"]!=sha256(args.representation) or nbc_checkpoint["vt_distance_scale_m"]!=scale:
        raise ValueError("NBC source actor/representation/coordinates mismatch")
    critic=SignedNBC(scale,nbc_checkpoint["hidden"])
    critic.load_state_dict(nbc_checkpoint["state_dict"],strict=True)
    critic.eval()
    def make_env(appearance=None):
        return LoggedReplay(args.representation,fixed["data_path"],fixed["latent_path"],fixed["rows"],scale,appearance)
    base.policy.mlp_extractor.policy_net.load_state_dict(saved.policy.controller_net.mlp_extractor.state_dict(),strict=True)
    base.policy.action_net.load_state_dict(saved.policy.controller_net.action_net.state_dict(),strict=True)
    base.policy.set_training_mode(False)
    probe=np.column_stack((fixed["latents"][:64],np.ones(64,dtype=np.float32)))
    mean_error=float(abs(base.predict(probe,deterministic=True)[0]-saved.predict(probe)[0]).max())
    if mean_error>1e-6: raise ValueError("initial full PPO does not reproduce final SPVC mean")
    model=JointNBCPPO("MlpPolicy",make_env(),learning_rate=1e-5,n_steps=2048,batch_size=256,n_epochs=4,
        gamma=.99,gae_lambda=.95,ent_coef=0.,target_kl=.01,seed=7,device="cpu",verbose=0,
        policy_kwargs=copy.deepcopy(base.policy_kwargs),rollout_buffer_class=SafetyBuffer)
    model.policy.load_state_dict(base.policy.state_dict(),strict=True)
    source=dict(actor=args.spvc_dir/"latent_ppo_spvc.pt",old_sbc=args.spvc_dir/"sbc.pt",representation=args.representation,
                nbc=args.nbc,base_ppo=fixed["controller_path"],cache=fixed["cache_path"],features=fixed["features_path"])
    if args.nbc_retention:
        source.update(retention_train_pairs=args.diagnostic_dir/"train_pairs.npz",retention_diagnostic=args.diagnostic_dir/"metrics.json")
        if sha256(source["retention_train_pairs"])!=nbc_checkpoint["source_hashes"]["train_pairs"] or sha256(source["retention_diagnostic"])!=nbc_checkpoint["source_hashes"]["diagnostic"]:
            raise ValueError("NBC retention data differs from pretraining source")
    hashes={k:sha256(p) for k,p in source.items()}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    model.attach_nbc(copy.deepcopy(critic),args.output_dir/"rollouts")
    model.nbc_scale=scale
    model.safety_feedback=args.safety_feedback=="on"
    model.barrier_steps=args.barrier_steps
    if args.nbc_retention:
        model.attach_retention(source["retention_train_pairs"],scale)
    report=dict(experiment="DINO_train_image_replay_PPO_NBC_joint_pilot_v1",status="running",source_hashes=hashes,
        source_identity=identity,initial_mean_equivalence_error=mean_error,initial_exploration_std=float(base.policy.log_std.exp().mean()),
        policy_initialization="final SPVC mean; inherited base PPO value/exploration variance; fresh optimizer, not a full PPO resume",
        scope="on-policy joint learning in train-image nearest-distance replay + analytic AEBS; NOT online CARLA",
        representation_frozen=True,q_frozen=True,test_used=False,cp_used=False,qp_used=False,
        safety_condition="signed NBC lambda=.1 epsilon=.01; %d-step discounted hinges" % args.barrier_steps,
        safety_actor_loss="likelihood-ratio weighted detached globally centered/scaled hinge; adaptive nonnegative coefficient, clipped to 1",
        old_log_prob_action="sampled unclipped Gaussian action; executed clipped action recorded separately",
        nbc_update_order="score with pre-update NBC; update PPO; then fit NBC on recorded transitions (10 steps)",
        actor_lr=1e-5,nbc_lr=3e-4,timesteps=args.timesteps,seed=7,multi_step=args.barrier_steps>1,
        barrier_steps=args.barrier_steps,multi_step_discount=.99,safety_certificate=False,
        boundary_handling="actual terminal successor included; no crossing episodes/rollouts; available-horizon weight normalization",
        safety_feedback=args.safety_feedback,nbc_retention=args.nbc_retention,
        retention_reference=model.retention_reference if args.nbc_retention else None,
        retention_spec="2048 old train transitions/step; balanced new/old invariant hinge; mean+max region hinge; fixed 16384 train guard and 81x81 region gate; restore NBC+Adam on rejection" if args.nbc_retention else None,
        script_sha256=sha256(Path(__file__)),retention_script_sha256=sha256(Path(__file__).with_name("nbc_retention.py")),
        multistep_script_sha256=sha256(Path(__file__).with_name("nbc_multistep.py")),
        evaluation={"before":{},"after":{}},matched={},joint_history=[])
    write_json(args.output_dir/"metrics.json",report)
    try:
        appearances=sorted(make_env().pools)
        print("Evaluating unchanged baseline: 12 train appearances; exact-image validation; no test",flush=True)
        for name in appearances:
            report["evaluation"]["before"][name]=evaluate(base,make_env(name),args.eval_grid)
        report["matched"]["before_validation"]=matched_nbc(base,critic,fixed,saved.policy.q_model,saved.policy.distance_scale_m,scale,"validation")
        torch.save(model.policy.state_dict(),args.output_dir/"initial_full_policy.pt")
        print("Starting 8192-step joint pilot" if args.timesteps==8192 else "Starting joint pilot: %d steps" % args.timesteps,flush=True)
        model.learn(total_timesteps=args.timesteps,callback=TrajectoryLogger())
        model.save(args.output_dir/"joint_ppo.zip")
        accepted_updates=(sum(r["nbc_retention"]["accepted"] for r in model.joint_history)
                          if model.retention_enabled else len(model.joint_history))
        torch.save(dict(state_dict=model.nbc.state_dict(),hidden=nbc_checkpoint["hidden"],vt_distance_scale_m=scale,
                        policy_sha256=sha256(args.output_dir/"joint_ppo.zip"),source_hashes=hashes,
                        accepted_nbc_updates=accepted_updates,retention_enabled=model.retention_enabled,
                        note=("All NBC proposals rejected; pretraining NBC weights retained, not adapted to final policy"
                              if not accepted_updates else "NBC includes accepted updates on pre-policy-update samples; not a certified final policy/barrier pair")),args.output_dir/"joint_nbc.pt")
        report["accepted_nbc_updates"]=accepted_updates
        final=PPO.load(args.output_dir/"joint_ppo.zip",device="cpu")
        print("Evaluating saved final policy against the same starts/appearances and frozen initial NBC",flush=True)
        for name in appearances:
            report["evaluation"]["after"][name]=evaluate(final,make_env(name),args.eval_grid)
        report["matched"]["after_validation"]=matched_nbc(final,critic,fixed,saved.policy.q_model,saved.policy.distance_scale_m,scale,"validation")
        report["joint_history"]=model.joint_history
        report["safety_actor_gradient_active"]=any(r["actor_safety_gradient_norm"]>1e-8 for r in model.joint_history)
        report["actor_updated"]=any(r["actor_parameter_max_drift"]>0 for r in model.joint_history)
        report["source_files_unchanged"]={k:sha256(p) for k,p in source.items()}==hashes
        if not report["source_files_unchanged"]: raise ValueError("baseline source changed")
        def aggregate(results):
            counts=Counter()
            n=0;ret=steps=0.
            for r in results.values():
                n+=r["episodes"];counts.update(r["counts"])
                ret+=r["mean_return"]*r["episodes"];steps+=r["mean_steps"]*r["episodes"]
            return dict(episodes=n,counts=dict(counts),success_rate=counts["success"]/n,unsafe_rate=counts["unsafe"]/n,
                        timeout_rate=counts["timeout"]/n,mean_return=ret/n,mean_steps=steps/n)
        report["aggregate"]={k:aggregate(v) for k,v in report["evaluation"].items()}
        report["postrun_audit"]=audit_saved_run(args.output_dir,fixed,saved)
        write_json(args.output_dir/"postrun_audit.json",report["postrun_audit"])
        if not model.safety_feedback:
            report["status"]="reward_only_ppo_control_complete"
        else:
            report["status"]="joint_replay_pilot_complete" if report["safety_actor_gradient_active"] and report["actor_updated"] else "joint_pilot_no_active_safety_update"
    except BaseException as exc:
        report.update(status="failed",error=repr(exc))
        raise
    finally:
        report["runtime_seconds"]=time.monotonic()-started
        write_json(args.output_dir/"metrics.json",report)
    print("[PPO + NBC joint replay pilot, NOT a safety certificate]")
    print(report["status"], "safety actor gradient active:",report["safety_actor_gradient_active"])
    print("rollouts:",report["aggregate"],flush=True)
    print("exact-image matched:",report["matched"],flush=True)
    print("final candidate audit:",report["postrun_audit"],flush=True)
    print("metrics:",args.output_dir/"metrics.json",flush=True)


if __name__=="__main__": main()
