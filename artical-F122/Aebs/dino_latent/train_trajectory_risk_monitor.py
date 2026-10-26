"""Fit a trajectory-risk monitor on DEVELOPMENT library trajectories only.

This is a separate branch, not a newly verified stochastic supermartingale.
The PPO, q and original legacy SBC stay untouched. Calibration/evaluation
episodes and CP scores are never inputs to this training program.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development-dir", type=Path, default=Path("results/native_R2_sbc_risk_cp_20261020_v1"))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() or min(a.steps, a.batch_size) < 1: p.error("New output and positive training budget required")
    torch.set_num_threads(1); torch.manual_seed(7)
    config_path = a.development_dir/"config.json"
    config = json.loads(config_path.read_text())
    source_path = a.development_dir/"development_rollouts.npz"
    data = np.load(source_path)
    if config["seeds"]["development"] != 7101 or config["model_training"] is not False:
        raise ValueError("Expected first frozen development library batch")
    if set(data["outcomes"].tolist()) != {"success"}: raise ValueError("Unsafe development occupancy must not be labelled safe")
    valid = data["valid"]
    current = data["states_m_mps"][:, :-1][valid]
    successor = data["states_m_mps"][:, 1:][valid]
    if not np.isfinite(current).all() or not np.isfinite(successor).all(): raise ValueError("Invalid development states")
    source_hashes = {str(source_path): sha256(source_path), str(config_path): sha256(config_path)}
    controller_path = next(Path(k) for k in config["source_sha256"] if k.endswith("/native_ppo.zip"))
    native_path = next(Path(k) for k in config["source_sha256"] if k.endswith("/historical_R2.pt"))
    for path in (controller_path, native_path):
        if sha256(path) != config["source_sha256"][str(path)]: raise ValueError("Frozen model identity changed")
        source_hashes[str(path)] = sha256(path)
    source_hashes[str(Path(__file__))] = sha256(Path(__file__))
    scale = float(Aebs(.05).std1)
    x, y = torch.from_numpy(current.copy()), torch.from_numpy(successor.copy())
    x[:, 0] /= scale; y[:, 0] /= scale
    model = MLP([2, 16, 8, 1], activation="tanh", square_output=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    rng = np.random.default_rng(7)
    a.output_dir.mkdir(parents=True)
    save_config = dict(experiment="native_R2_development_occupancy_risk_monitor_v1", seed=7,
                       steps=a.steps, batch_size=a.batch_size, lr=.003, network=[2,16,8,1],
                       training_trajectories=len(valid), training_transitions=len(x),
                       loss="20*mean(B_safe)+20*mean(relu(1.2-B_unsafe)^2)+10*mean(relu(B_init-.01)^2)+10*mean(relu(B_next-B_current))",
                       calibration_or_evaluation_loaded=False, PPO_q_updated=False,
                       original_SBC_overwritten=False, classical_SBC_verified=False,
                       source_sha256=source_hashes)
    (a.output_dir/"config.json").write_text(json.dumps(save_config, indent=2))
    history = []; start = time.monotonic()
    for step in range(1, a.steps+1):
        ids = rng.integers(0, len(x), a.batch_size)
        unsafe = torch.from_numpy(np.column_stack((rng.uniform(5,6,a.batch_size)/scale,
                                                  rng.uniform(.500001,3,a.batch_size))).astype(np.float32))
        init = torch.from_numpy(np.column_stack((rng.uniform(15,16,256)/scale,
                                                rng.uniform(2.5,3,256))).astype(np.float32))
        before, after = model(x[ids]), model(y[ids])
        occupancy = (before.mean()+after.mean())/2
        unsafe_loss = F.relu(1.2-model(unsafe)).square().mean()
        initial_loss = F.relu(model(init)-.01).square().mean()
        increase = F.relu(after-before).mean()
        loss = 20*occupancy+20*unsafe_loss+10*initial_loss+10*increase
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step % 500 == 0:
            row = dict(step=step, loss=float(loss), safe_occupancy=float(occupancy),
                       unsafe_loss=float(unsafe_loss), positive_increment=float(increase))
            history.append(row); print("[Development risk monitor]", row, flush=True)
    model.eval().requires_grad_(False)
    torch.save(model.state_dict(), a.output_dir/"risk_monitor.pt")
    with torch.no_grad():
        safe_values = model(torch.cat((x,y))).numpy().reshape(-1)
        d,v = np.meshgrid(np.linspace(5,6,100), np.linspace(.500001,3,100), indexing="ij")
        unsafe_grid = torch.tensor(np.column_stack((d.reshape(-1)/scale,v.reshape(-1))),dtype=torch.float32)
        unsafe_values = model(unsafe_grid).numpy().reshape(-1)
    for path,digest in source_hashes.items():
        if sha256(Path(path)) != digest: raise ValueError("Development source changed")
    metrics = dict(**save_config, status="development_risk_monitor_trained_not_a_classical_SBC_certificate",
                   monitor_sha256=sha256(a.output_dir/"risk_monitor.pt"),
                   controller_sha256=sha256(controller_path), native_checkpoint_sha256=sha256(native_path),
                   normalization=dict(lower=0., upper=1.),
                   safe_occupancy_max=float(safe_values.max()), safe_occupancy_mean=float(safe_values.mean()),
                   unsafe_grid_min=float(unsafe_values.min()), unsafe_grid_below_one=int((unsafe_values < 1).sum()),
                   unsafe_grid_points=len(unsafe_values), continuous_unsafe_bound_verified=False,
                   runtime_seconds=time.monotonic()-start)
    (a.output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False))
    (a.output_dir/"history.json").write_text(json.dumps(history,indent=2,allow_nan=False))
    print("[Risk monitor training finished]", metrics["safe_occupancy_max"], metrics["unsafe_grid_min"], flush=True)
    print("metrics:",a.output_dir/"metrics.json",flush=True)


if __name__ == "__main__": main()
