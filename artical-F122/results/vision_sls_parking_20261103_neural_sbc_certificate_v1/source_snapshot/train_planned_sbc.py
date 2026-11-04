"""Train and diagnose time-dependent SBC candidates; sampled checks only."""
import argparse
import json
from pathlib import Path
import time
import shutil
import numpy as np
import torch
from .core import INITIAL_STATES, nominal_observation, constraints, STATE_LOW, STATE_HIGH
from .planning_feedback import ReferenceBank, PlannedController
from .planned_transition import PlannedTransition
from .planned_env import evaluate_planned, perturbed_initial_state
from .planned_sbc import PlannedBarrier, risk, finite_horizon_barrier
from .perception_error import radius
from .train_migration import load_ppo
from .train_ppo import sha


def initial_samples(bank, policy, count, seed, error_model=None):
    rng = np.random.default_rng(seed); states, refs = [], []
    for n in range(count):
        j = n % 15
        x = INITIAL_STATES[j].copy() if n < 15 else perturbed_initial_state(j, rng, .04)
        y = nominal_observation(x)
        if error_model is not None and n >= 15:
            y += rng.uniform(-1, 1, 3)*radius(x[None], error_model)[0]
        controller = PlannedController(bank, policy, mode='residual')
        controller.reset(y, bank.goals[j]); states.append(np.r_[x, controller.estimate]); refs.append(controller.reference)
    return torch.tensor(np.asarray(states), dtype=torch.float32), torch.tensor(refs)


def training_pairs(bank, transition, nominal, count, seed, error_model=None, teacher=False):
    rng = np.random.default_rng(seed); rows = []
    widths = np.array([.08, .08, .08, .08]*2)
    with torch.no_grad():
        for j in range(15):
            for t in range(30):
                z = nominal[j, t]+rng.uniform(-1, 1, (count, 8))*widths
                z[0] = nominal[j, t]
                # The stopped process imposes drift only while currently safe.
                valid = np.array([not constraints(s[:4])['collision'] and constraints(s[:4])['in_bounds'] for s in z])
                z = z[valid]
                if not len(z):
                    continue
                error = np.zeros((len(z), 3))
                if error_model is not None:
                    lo, hi = np.asarray(error_model['low']), np.asarray(error_model['high'])
                    valid = ((z[:, :3] >= lo) & (z[:, :3] <= hi)).all(1)
                    z = z[valid]
                    if not len(z):
                        continue
                    error = rng.choice([-1., 1.], (len(z), 3))*radius(z[:, :4], error_model)
                s = torch.tensor(z, dtype=torch.float64)
                nxt = transition(s, j, t, torch.tensor(error))
                target = finite_horizon_barrier(transition, s.clone(), j, t) if teacher else risk(s)
                rows.append((s.float(), nxt.float(), torch.full((len(z),), j, dtype=torch.long),
                             torch.full((len(z),), t, dtype=torch.long), target.float()))
    return tuple(torch.cat([r[k] for r in rows]) for k in range(5))


@torch.no_grad()
def diagnostics(barrier, pairs, initial, initial_refs, nominal, transition):
    s, nxt, j, t, _ = pairs
    iv = barrier(initial, initial_refs, torch.zeros_like(initial_refs)).flatten()
    gaps = (barrier(nxt, j, t+1)-barrier(s, j, t)).flatten()
    rng = np.random.default_rng(99031)
    # Include the entire physical box plus outside points; direct geometry
    # identifies the unsafe subset, without assumptions on observer memory.
    x = rng.uniform(STATE_LOW-.3, STATE_HIGH+.3, (12000, 4))
    unsafe = np.array([constraints(q)['collision'] or not constraints(q)['in_bounds'] for q in x])
    z = torch.tensor(np.c_[x[unsafe], rng.uniform(STATE_LOW-1, STATE_HIGH+1, (unsafe.sum(), 4))], dtype=torch.float32)
    jr = torch.tensor(rng.integers(15, size=len(z))); tr = torch.tensor(rng.integers(31, size=len(z)))
    uv = barrier(z, jr, tr).flatten()
    nominal_values = []
    for ref in range(15):
        states = torch.tensor(nominal[ref], dtype=torch.float32)
        nominal_values.append(barrier(states, torch.full((31,),ref), torch.arange(31)).flatten().numpy())
    values = np.asarray(nominal_values); ng = np.diff(values, axis=1)
    return dict(initial_count=len(iv), initial_max=float(iv.max()), initial_violations=int((iv>1).sum()),
                unsafe_count=len(uv), unsafe_min=float(uv.min()), unsafe_violations=int((uv<20).sum()),
                drift_count=len(gaps), drift_violations_strict=int((gaps>0).sum()),
                drift_violations_above_1e_6=int((gaps>1e-6).sum()), max_drift=float(gaps.max()),
                nominal_replay_drift_violations=int((ng>0).sum()), max_nominal_replay_drift=float(ng.max()),
                continuous_state_verification=False, safety_certificate=False), gaps.numpy(), values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--epochs', type=int, default=6000); p.add_argument('--error-model')
    a = p.parse_args(); root, out = Path(a.candidate), Path(a.output_dir)
    if out.exists() or a.epochs < 1:
        p.error('new output directory and positive epochs required')
    torch.set_num_threads(1); torch.manual_seed(821)
    bank = ReferenceBank(root/'references.npz'); policy = load_ppo(root/'ppo.zip',device='cpu')
    transition = PlannedTransition(bank, policy).eval().requires_grad_(False)
    error_model = json.loads(Path(a.error_model).read_text()) if a.error_model else None
    if error_model and error_model['provenance']['projection_sha256'] != sha(root/'projection.pt'):
        raise ValueError('CP/projection mismatch')
    out.mkdir(parents=True); started = time.monotonic()
    cfg = dict(vars(a), ppo_sha256=sha(root/'ppo.zip'), reference_sha256=sha(root/'references.npz'),
               projection_sha256=sha(root/'projection.pt'),
               error_model_sha256=sha(a.error_model) if a.error_model else None,
               continuous_state=8, discrete_state='reference index (15) and time (31)',
               neural_input=24, network=[24,64,64,1], initial_threshold=1, unsafe_threshold=20,
               architecture='geometric unsafe-risk + Softplus correction + .01*(30-time)',
               initial_sampling='nominal and +/- .04 physical perturbations; CP initial errors if enabled',
               drift_sampling='independent +/- .08 neighborhoods in physical state and observer memory',
               error_sampling='one random CP-box corner per sampled current state' if error_model else 'zero',
               initial_reference_selection_included=True, process_noise='zero', policy_updated=False,
               safety_certificate=False, continuous_state_verification=False, certificate_probability=None)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ('planned_sbc.py','train_planned_sbc.py','planned_transition.py'):
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    _, data = evaluate_planned(bank, policy)
    nominal = np.concatenate([data['states'],data['estimates']],axis=-1)
    print('building sampled dynamics and deterministic lookahead teacher',flush=True)
    train = training_pairs(bank,transition,nominal,24,822,error_model,teacher=error_model is None)
    valid = training_pairs(bank,transition,nominal,12,823,error_model)
    initial, refs = initial_samples(bank,policy,960,824,error_model)
    audit_initial, audit_refs = initial_samples(bank,policy,1920,825,error_model)
    barrier = PlannedBarrier(bank); opt = torch.optim.Adam(barrier.parameters(),lr=.001)
    s,nxt,j,t,teacher = train; history=[]; best=None; selected=0
    hard=torch.nonzero(teacher.flatten()>0,as_tuple=False).flatten()
    teacher=teacher+.01*(30-t[:,None])
    for epoch in range(a.epochs):
        ids=torch.randint(len(s),(384,)); ids0=torch.randint(len(initial),(128,))
        if len(hard):
            ids=torch.cat([ids,hard[torch.randint(len(hard),(128,))]])
        cur=barrier(s[ids],j[ids],t[ids]); after=barrier(nxt[ids],j[ids],t[ids]+1)
        iv=barrier(initial[ids0],refs[ids0],torch.zeros_like(refs[ids0]))
        initial_loss=torch.relu(iv-.5).square().mean()/400
        drift_loss=torch.relu(after-cur).square().mean()/400
        teacher_loss=(torch.log1p(cur)-torch.log1p(teacher[ids])).square().mean() if error_model is None else cur.mean()*0
        loss=20*initial_loss+10*drift_loss+.02*teacher_loss
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(barrier.parameters(),10);opt.step()
        if (epoch+1)%100==0:
            with torch.no_grad():
                violations=(barrier(nxt,j,t+1)-barrier(s,j,t)).flatten()>0
                hard=torch.nonzero(violations | (teacher.flatten()>.01*(30-t)+1e-5),as_tuple=False).flatten()
        if (epoch+1)%500==0 or epoch+1==a.epochs:
            metrics,_,_=diagnostics(barrier,valid,audit_initial,audit_refs,nominal,transition)
            score=(metrics['initial_violations'],metrics['drift_violations_strict'],metrics['max_drift'])
            if best is None or score < best:
                best=score;selected=epoch+1;torch.save(barrier.state_dict(),out/'sbc.pt')
            row=dict(epoch=epoch+1,loss=float(loss.detach()),initial=metrics['initial_violations'],
                     drift=metrics['drift_violations_strict'],max_drift=metrics['max_drift'],selected_epoch=selected)
            print(json.dumps(row),flush=True);history.append(row)
    torch.save(barrier.state_dict(),out/'last_sbc.pt')
    barrier.load_state_dict(torch.load(out/'sbc.pt',map_location='cpu'))
    selected_metrics,gaps,values=diagnostics(barrier,valid,audit_initial,audit_refs,nominal,transition)
    # Separate sampling seed not used for checkpoint selection.
    test = training_pairs(bank,transition,nominal,24,992001,error_model)
    test_initial,test_refs=initial_samples(bank,policy,1920,992002,error_model)
    test_metrics,_,_=diagnostics(barrier,test,test_initial,test_refs,nominal,transition)
    lookahead=[]
    with torch.no_grad():
        for ref in range(15):
            s0=torch.tensor(nominal[ref,:1],dtype=torch.float64)
            lookahead.append(float(finite_horizon_barrier(transition,s0,ref,0)))
    report=dict(config=cfg,selected_epoch=selected,selection=selected_metrics,new_seed_audit=test_metrics,
                deterministic_lookahead_initial_values=lookahead,
                deterministic_lookahead_scope='nominal dynamics only; initial values numerically evaluated; no image-error certificate',
                history=history,runtime_seconds=time.monotonic()-started,
                ppo_unchanged=sha(root/'ppo.zip')==cfg['ppo_sha256'],
                safety_certificate=False,certificate_probability=None)
    np.savez_compressed(out/'diagnostics.npz',states=valid[0].numpy(),successors=valid[1].numpy(),
                        references=valid[2].numpy(),times=valid[3].numpy(),gaps=gaps,nominal_values=values)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(selected=selected_metrics,new_seed=test_metrics),indent=2),flush=True)


if __name__ == '__main__':
    main()
