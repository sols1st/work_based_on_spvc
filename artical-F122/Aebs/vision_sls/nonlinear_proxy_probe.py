"""Fit a smooth state-to-observation proxy to a FROZEN RGB visual head.

This is a static proxy ablation, not a new closed-loop CP/SBC certificate.
The deployed decoder is unchanged; its nominal bias must not be discarded.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class ResidualProxy(nn.Module):
    """h(x)=Cx+r(x_d); r is smooth with a bounded derivative.

    Speed dependence remains explicit and linear under the static-image law.
    Clipping the product of spectral norms bounds |r'| globally in real arithmetic.
    """
    def __init__(self, C, hidden=16, slope_budget=.5):
        super().__init__()
        self.register_buffer('C', torch.as_tensor(C, dtype=torch.float32).reshape(1, 2))
        if abs(float(self.C[0, 0])) <= slope_budget:
            raise ValueError('residual slope must be smaller than nominal distance slope')
        self.slope_budget = slope_budget
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return state @ self.C.T + self.net(state[:, :1])

    @torch.no_grad()
    def slope_bound(self):
        return math.prod(float(torch.linalg.matrix_norm(m.weight, ord=2))
                         for m in self.net if isinstance(m, nn.Linear))

    @torch.no_grad()
    def constrain(self):
        bound = self.slope_bound()
        if bound > self.slope_budget:
            self.net[-1].weight.mul_(self.slope_budget / bound)


def stats(error):
    e = np.asarray(error, dtype=float).reshape(-1)
    return dict(n=len(e), mae_m=float(abs(e).mean()), rmse_m=float(np.sqrt((e*e).mean())),
                p95_m=float(np.quantile(abs(e), .95)), p99_m=float(np.quantile(abs(e), .99)),
                max_m=float(abs(e).max()), bias_m=float(e.mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seeds', type=int, nargs='+', default=[151, 152, 153])
    a = p.parse_args()
    if a.steps < 1:
        raise ValueError('positive steps required')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source = json.loads((a.source/'protocol.json').read_text())
    report = json.loads((a.source/'report.json').read_text())
    data = np.load(a.source/'observations.npz', allow_pickle=False)
    scale = source['visual_distance_scale_m']
    C = np.asarray(report['C_final'], dtype=np.float32)
    tr = np.asarray(source['train_rows'], dtype=int)
    va = np.asarray(source['internal_validation_rows'], dtype=int)
    assert not set(tr) & set(va)
    assert np.all(data['split'][np.r_[tr, va]] == 'train')
    masks = dict(development=va, calibration=np.flatnonzero(data['split']=='calibration'),
                 test=np.flatnonzero(data['split']=='test'))
    assert all(len(x) for x in masks.values())
    d = data['distance_m'].astype(np.float32)
    target = torch.tensor(data['observation'], dtype=torch.float32).reshape(-1, 1)
    # Replicated speeds are assumptions of the static library, not measured labels.
    speeds = torch.tensor([0., 1.5, 3.])
    x = torch.tensor(d/scale).reshape(-1, 1)
    state_ref = torch.cat([x, torch.full_like(x, 1.5)], dim=1)
    nominal = (state_ref @ torch.tensor(C).T).numpy().flatten()
    latent = target.numpy().flatten()
    conversion = scale/float(C[0, 0])
    true_error = conversion*(latent-nominal)
    protocol = dict(format='frozen_rgb_nonlinear_proxy_probe_v1', steps=a.steps, seeds=a.seeds,
        batch_size=256, lr=.001, proxy='Cx + MLP(distance/scale); 1-16-16-1 Tanh',
        residual_slope_cap=.5, C=C.tolist(), visual_distance_scale_m=scale,
        synthetic_speeds=speeds.tolist(), train_rows=tr.tolist(), validation_rows=va.tolist(),
        selection='minimum development MSE, evaluated every 100 updates and at final update',
        frozen_visual=True, frozen_controller=True, reference_speed=1.5,
        bindings={n:digest(a.source/n) for n in ['protocol.json','report.json','observations.npz','joint_observation.pt']},
        source_sha256=digest(__file__),
        scope='Static frozen-head proxy fitting. Historical held-out images, not fresh blind test. No trajectory CP or SBC.',
        budget_note='MLP seeds have equal updates; affine baseline uses exact least squares, not matched compute.')
    write(a.output/'protocol.json', protocol)
    rows = []

    def evaluate(name, h, extra):
        residual = conversion*(latent-h)
        nominal_bias = conversion*(h-nominal)
        assert np.allclose(residual+nominal_bias, true_error, atol=3e-6)
        result = dict(method=name, **extra, splits={})
        for split, ids in masks.items():
            result['splits'][split] = dict(proxy_residual=stats(residual[ids]),
                nominal_decoder_bias=stats(nominal_bias[ids]),
                unchanged_total_observation_error=stats(true_error[ids]))
        cal = np.sort(abs(residual[masks['calibration']]))
        rank = math.ceil((len(cal)+1)*.95)
        q = float(cal[rank-1]) if rank <= len(cal) else None
        result['static_quantile_diagnostic'] = dict(radius_m_equivalent=q, n=len(cal),rank=rank,
            test_fraction=float((abs(residual[masks['test']])<=q).mean()) if q is not None else None,
            scope='Frame residual diagnostic; distance-bin split does not establish exchangeable trajectory CP.')
        np.savez_compressed(a.output/(name+'_predictions.npz'), nominal_latent=h,
            proxy_residual_m=residual,nominal_bias_m=nominal_bias,total_error_m=true_error)
        rows.append(result)

    evaluate('linear_Cx', nominal, {})
    # A bias/slope correction should not be attributed to nonlinear expressivity.
    design = np.c_[d/scale, np.ones(len(d))]
    coeff = np.linalg.lstsq(design[tr], (latent-nominal)[tr], rcond=None)[0]
    evaluate('affine_residual', nominal+design@coeff, dict(coefficients=coeff.tolist()))
    for seed in a.seeds:
        torch.manual_seed(seed)
        model = ResidualProxy(C)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        history, best = [], None
        for step in range(1, a.steps+1):
            ids = tr[torch.randint(len(tr), (256,)).numpy()]
            distances = x[ids].repeat_interleave(3, dim=0)
            states = torch.cat([distances, speeds.repeat(len(ids))[:, None]], dim=1)
            truth = target[ids].repeat_interleave(3, dim=0)
            loss = (model(states)-truth).square().mean()
            opt.zero_grad(); loss.backward(); opt.step(); model.constrain()
            if step % 100 == 0 or step == a.steps:
                with torch.no_grad():
                    dev = float((model(state_ref[va])-target[va]).square().mean())
                history.append(dict(step=step, train_loss=float(loss.detach()), development_mse=dev))
                if best is None or dev < best[0]:
                    best = (dev, copy.deepcopy(model.state_dict()), step)
        model.load_state_dict(best[1]); model.eval()
        with torch.no_grad():
            h = model(state_ref).numpy().flatten()
        torch.save(dict(format='aebs_residual_proxy_probe_v1', state_dict=model.state_dict(),
            hidden=16,slope_budget=.5,source_bindings=protocol['bindings'],
            scale=scale,seed=seed,best_step=best[2]), a.output/f'proxy_{seed}.pt')
        write(a.output/f'history_{seed}.json', history)
        evaluate(f'nonlinear_seed_{seed}', h, dict(best_step=best[2],
            residual_derivative_bound=model.slope_bound(),
            minimum_nominal_distance_slope=float(C[0,0])-model.slope_bound()))
        print(json.dumps(rows[-1]), flush=True)
    write(a.output/'report.json', dict(protocol=protocol, methods=rows,
        safety_status='not_certified: nonlinear nominal model not wired into SBC verifier',
        interpretation='Only surrogate matching changes; visual estimates/actions/total error are identical.'))


if __name__ == '__main__':
    main()
