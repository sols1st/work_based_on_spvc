"""Separate reference-policy bias from observation-latent sensitivity."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

from .benefit_native_spvt import StateDecoder, SEEDS, target
from .proxy_architecture_compare import sha, write


def main():
    torch.set_num_threads(1)
    upstream = Path('/Users/johann/project/SPVT')
    root = Path('results/benefit_native_spvt_20261122')
    output = Path('results/comparison_design_review_20261123')
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('original_spvt_models', upstream/'model/model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator_path = upstream/'model/pretrained_ckpts/mlp_generator.pth'
    generator = module.MLPGenerator().eval().requires_grad_(False)
    generator.load_state_dict(torch.load(generator_path, map_location='cpu')['generator_state_dict'])
    models = {}
    sources = [generator_path]
    pid = upstream/'model/pretrained_ckpts/pid_controller.pth'
    model = module.E2ENetworkBN()
    model.load_state_dict(torch.load(pid, map_location='cpu'))
    models['spvt_pid'] = model
    sources.append(pid)
    path = root/'baseline_checkpoints/checkpoint_epoch_100.pth'
    combined = module.get_model()
    combined.load_state_dict(torch.load(path, map_location='cpu')['model_state_dict'])
    assert all(torch.equal(v, combined.generator.state_dict()[k]) for k, v in generator.state_dict().items())
    models['spvt_epoch100'] = combined.controller
    sources.append(path)
    for seed in SEEDS:
        for kind in ('same_target_e2e', 'current_lowdim'):
            path = root/f'{kind}_{seed}.pt'
            model = StateDecoder() if kind == 'current_lowdim' else module.E2ENetworkBN()
            model.load_state_dict(torch.load(path, map_location='cpu')['state_dict'])
            models[f'{kind}_{seed}'] = model
            sources.append(path)
    rng = np.random.default_rng(202611239501)
    state = torch.tensor(rng.uniform([-60,-7],[60,7],(10000,2)),dtype=torch.float32)
    z = torch.tensor(rng.uniform(-.8,.8,(10000,10)),dtype=torch.float32)
    with torch.no_grad():
        zero = generator(torch.zeros_like(z),state)
        observed = generator(z,state)
        reference = target(state).numpy()*30
    results = {}
    arrays = {'state_cm_degrees':state.numpy(), 'latent':z.numpy(), 'reference_degrees':reference}
    for key, model in models.items():
        model.eval().requires_grad_(False)
        with torch.no_grad():
            def action(obs):
                return target(model(obs)) if key.startswith('current_lowdim') else model(obs).flatten()
            nominal = action(zero).numpy()*30
            actual = action(observed).numpy()*30
        bias = nominal-reference
        sensitivity = actual-nominal
        total = actual-reference
        np.testing.assert_allclose(total,bias+sensitivity,atol=1e-5)
        results[key] = dict(reference_mae_deg=float(abs(total).mean()),
            nominal_policy_bias_mae_deg=float(abs(bias).mean()),
            latent_sensitivity_mae_deg=float(abs(sensitivity).mean()),
            latent_sensitivity_p95_deg=float(np.quantile(abs(sensitivity),.95)),
            nominal_policy_bias_p95_deg=float(np.quantile(abs(bias),.95)))
        arrays[key+'_actual_deg'] = actual
        arrays[key+'_nominal_deg'] = nominal
    np.savez_compressed(output/'same_state_diagnostic.npz',**arrays)
    write(output/'reference_bias.json',dict(seed=202611239501,n_states=10000,results=results,
          sources={str(p):sha(p) for p in sources},source_sha256=sha(__file__),
          scope='Common-state static counterfactual audit; latent sensitivity is action change when z changes from zero to a random value, not a trajectory CP guarantee.'))
    print(results)


if __name__ == '__main__':
    main()
