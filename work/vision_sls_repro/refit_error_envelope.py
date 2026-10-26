"""Reproduce the author's 250-point MILP using the native Gurobi API.

The installed Ubuntu-20-compatible Drake wheel has no Gurobi backend, so this
script transcribes the same linear constraints to gurobipy. No supplied
checkpoint or controller envelope is overwritten.
"""
import argparse
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
args.source = args.source.resolve()
args.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.source))
import numpy as np
import torch
import yaml
from dotmap import DotMap
import gurobipy as gp
from matplotlib import pyplot as plt
from dyn.unicycle_of_perception import Unicycle_OF
from perception_utils.networks import SupervisedDinoObservabilityLargeUnicycle

torch.set_num_threads(4)
experiment = args.source / 'expe/parking_lot'
config = DotMap(yaml.safe_load((experiment / 'data/learn_model_error_observability.yaml').read_text()))
states, features = torch.load(experiment / config.dataset.load_filename, map_location='cpu')
states, features = states.float(), features.float()
if states.shape[1] == 3:
    states = torch.cat([states, torch.zeros((len(states), 1))], dim=1)
net = SupervisedDinoObservabilityLargeUnicycle(config.model)
net.load_state_dict(torch.load(experiment / config.train.load_ckpt, map_location='cpu'))
net.eval()
C = net.C / net.C.norm(dim=1, keepdim=True)
assert torch.all(C[:, 3] == 0), 'Dummy-speed convention needs reassessment'
m = Unicycle_OF()
assert np.allclose(C.detach().numpy(), m.C, atol=1e-6)
with torch.no_grad():
    residual_vectors = net(features) - states @ C.T
    residual = residual_vectors.abs().amax(dim=1).numpy()
    design = m.mons_fn(states[:, :2]).numpy().astype(float)
assert design.shape == (len(states), 10)
# Check symbolic monomial ordering against the explicitly coded saved envelope.
saved_values = np.maximum(design @ m.theta_poly[:, 0], m.theta_bias)
explicit = np.array([m.get_disturbance(x - np.array([3, 0, 0, 0])) for x in states[:100].numpy()])
assert np.allclose(saved_values[:100], explicit, atol=1e-6)
train = np.random.default_rng(args.seed).choice(len(states), 250, replace=False)
validation = np.setdiff1d(np.arange(len(states)), train)
X, r = design[train], residual[train].astype(float)
N = len(train)
M = 1.001 * r.max()
started = time.perf_counter()
model = gp.Model('vision_sls_original_envelope')
model.Params.TimeLimit = 180
model.Params.Threads = 4
model.Params.Seed = args.seed
model.Params.LogFile = str(args.output / 'gurobi.log')
beta = model.addMVar(10, lb=-.5, ub=.5, name='theta')
bias = model.addVar(lb=0, ub=.04, name='theta_bias')
z = model.addMVar(N, vtype=gp.GRB.BINARY, name='z')
q = model.addMVar(N, vtype=gp.GRB.BINARY, name='q')
poly = X @ beta
model.addConstr(poly >= r - M*z - M*q)
model.addConstr(bias >= r - M*(1-z) - M*q)
model.addConstr(q.sum() == (1-.97)*z.sum())
model.setObjective(poly.sum() + 5*N*bias, gp.GRB.MINIMIZE)
model.optimize()
elapsed = time.perf_counter() - started
assert model.SolCount > 0, f'No feasible calibration solution: status {model.Status}'
coeff, floor = beta.X.copy(), float(bias.X)
zv, qv = z.X.copy(), q.X.copy()
train_poly = X @ coeff
violations = [np.max(r-M*zv-M*qv-train_poly),
              np.max(r-M*(1-zv)-M*qv-floor),
              abs(qv.sum()-.03*zv.sum())]
assert max(violations) < 1e-5, violations
bound = np.maximum(design @ coeff, floor)
np.savez_compressed(args.output / 'refitted_envelope.npz',
                    theta_poly=coeff, theta_bias=np.array([floor]), degree=3, nx=2,
                    z_val=zv, q_val=qv, M=M, train_indices=train,
                    validation_indices=validation, states=states.numpy(),
                    residual_vectors=residual_vectors.detach().numpy(),
                    residual_norm=residual, fitted_bounds=bound)
summary = {
    'solver': 'gurobipy native API, same MILP as original relax_poly=True branch',
    'gurobi_version': gp.gurobi.version(), 'seed': args.seed,
    'dataset_samples': len(states), 'fit_samples': N,
    'heldout_samples': len(validation), 'degree': 3, 'threshold_keep': .97,
    'dummy_speed': 'zero; C fourth column verified exactly zero',
    'M': float(M), 'b_min': floor, 'z_sum': float(zv.sum()), 'q_sum': float(qv.sum()),
    'status': int(model.Status), 'objective': model.ObjVal, 'MIP_gap': model.MIPGap,
    'wall_seconds': elapsed, 'max_fitting_constraint_residual': float(max(violations)),
    'fit_absolute_residual_coverage': float(np.mean(residual[train] <= bound[train]+1e-8)),
    'heldout_absolute_residual_coverage': float(np.mean(residual[validation] <= bound[validation]+1e-8)),
    'provided_envelope_heldout_coverage': float(np.mean(residual[validation] <= saved_values[validation]+1e-8)),
    'provided_envelope_provenance': 'Saved fit may have used these data; not an independent test.',
    'limitations': ['No retraining of DINO or the projection network.',
                    'This re-fit is diagnostic; the visual controller uses the unchanged supplied envelope.',
                    'Coverage is empirical and not a deterministic safety certificate.'],
}
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(residual[validation], bound[validation], s=7, alpha=.35, label='Held-out samples')
limit=max(float(residual[validation].max()),float(bound[validation].max()))
ax.plot([0,limit],[0,limit],color='black',linestyle='--',label='Bound = actual residual')
ax.set(xlabel='Actual maximum absolute observation residual', ylabel='Fitted envelope',
       title='VISION-SLS envelope calibration (250 fitting points)')
ax.legend()
fig.tight_layout()
fig.savefig(args.output / 'calibration_coverage.png', dpi=160)
print(json.dumps(summary, indent=2), flush=True)
