"""PPO synthesis, no runtime planner/filter. Broad resets are training only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Aebs.lane_keeping.policy import export_actor,load_actor
from .env import RobotEnv
from . import core


@torch.no_grad()
def evaluate(actor,radius=0.,n=300,seed=711):
    rng=np.random.default_rng(seed);x=core.sample_initial(rng,n)
    states=[x.copy()];actions=[]
    for _ in range(core.HORIZON):
        obs=core.normalize(x)+rng.uniform(-radius,radius,(n,3))
        u=actor(torch.tensor(obs,dtype=torch.float32)).numpy()
        x=core.dynamics(x,u);states.append(x.copy());actions.append(u)
    states=np.array(states);actions=np.array(actions)
    bad=core.unsafe(states).any(0)
    reached=core.goal(states).any(0)
    held=core.goal(states[-10:]).all(0)&(((actions[-10:,:,0]+1)*core.V_MAX/2).max(0)<.08)
    return dict(episodes=n,unsafe=int(bad.sum()),reached=int((reached&~bad).sum()),
                reached_and_held=int((held&~bad).sum()),radius=radius,
                mean_final_distance=float(np.linalg.norm(states[-1,:,:2]-core.GOAL,axis=1).mean())),states,actions


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300000)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--radius',type=float,default=0.)
    p.add_argument('--resume',type=Path)
    p.add_argument('--init-policy',type=Path,help='portable policy/value weights; new optimizer, mutually exclusive with resume')
    args=p.parse_args(argv)
    if args.resume and args.init_policy:raise ValueError('choose resume or init-policy')
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    env=DummyVecEnv([lambda i=i:RobotEnv(args.seed+i,args.radius,True) for i in range(8)])
    if args.resume:model=PPO.load(args.resume,env=env,device='cpu')
    else:model=PPO('MlpPolicy',env,seed=args.seed,n_steps=256,batch_size=512,n_epochs=10,
                   learning_rate=3e-4,gamma=.99,gae_lambda=.95,ent_coef=.002,verbose=1,
                   policy_kwargs=dict(net_arch=[64,64],activation_fn=torch.nn.Tanh),device='cpu')
    if args.init_policy:model.policy.load_state_dict(torch.load(args.init_policy,map_location='cpu',weights_only=False))
    model.learn(total_timesteps=args.steps)
    model.save(args.output_dir/'ppo_model.zip');actor=export_actor(model,args.output_dir/'actor.pt')
    torch.save(model.policy.state_dict(),args.output_dir/'policy_weights.pt')
    records={}
    for radius in sorted({0.,args.radius,.03,.06}):
        report,states,actions=evaluate(actor,radius)
        records[str(radius)]=report
        np.savez_compressed(args.output_dir/f'rollout_{radius}.npz',states=states,actions=actions)
    (args.output_dir/'report.json').write_text(json.dumps(records,indent=2))
    (args.output_dir/'config.json').write_text(json.dumps(dict(task=core.config(),seed=args.seed,
             timesteps_requested=args.steps,training_radius=args.radius,curriculum=True,
             reward='offline grid distance potential, never supplied to deployment controller',
             resume=str(args.resume),init_policy=str(args.init_policy)),indent=2))
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()
