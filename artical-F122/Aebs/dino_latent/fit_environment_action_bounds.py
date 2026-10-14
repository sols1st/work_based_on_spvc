"""Fit train-only state-dependent image-minus-q action bounds.

Development diagnostic only: no CP calibration, frozen-policy modification,
test data, rollout or SBC re-verification. Validation never selects a model.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from Aebs.dino_latent.check_environment_action_envelope import group_actions
from Aebs.dino_latent.diagnose_environment_barrier import write_json
from Aebs.dino_latent.prepare_expanded_data import sha256


class ActionErrorBounds(nn.Module):
    """Physical (distance m, speed m/s) -> ordered error bounds including zero."""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.register_buffer("input_center", torch.tensor([11., 1.5]))
        self.register_buffer("input_scale", torch.tensor([5., 1.5]))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -4.)

    def forward(self, states):
        radius = 6.*torch.sigmoid(self.net((states-self.input_center)/self.input_scale))
        return torch.stack((-radius[:, 0], radius[:, 1]), dim=1)


def envelope_targets(stream):
    states, fields, appearances = group_actions(stream)
    _, nominal, _ = group_actions(stream, "nominal")
    q = nominal["action_low"]
    targets = np.column_stack((fields["action_low"]-q, fields["action_high"]-q))
    if not np.isfinite(targets).all() or np.any(targets[:, 0] > 0) or np.any(targets[:, 1] < 0):
        raise ValueError("invalid grouped action error targets")
    return states.astype(np.float32), targets.astype(np.float32), appearances


def bounds_loss(prediction, targets):
    under = torch.stack(((prediction[:, 0]-targets[:, 0]).relu(),
                         (targets[:, 1]-prediction[:, 1]).relu()), dim=1)
    fit = (prediction-targets).square().mean()
    missed = under.square().mean()
    worst = under.square().max()
    width = (prediction[:, 1]-prediction[:, 0]).mean()
    return fit+20.*missed+.1*worst+.001*width


def evaluate_summary(states, errors, prediction, appearances, band, nominal, tolerance=1e-6):
    low, high = prediction[:, 0], prediction[:, 1]
    outside = np.maximum(np.maximum(low-errors, errors-high), 0.)
    miss = outside > tolerance
    widths = high-low
    clipped_width = np.clip(nominal+high, -3., 3.)-np.clip(nominal+low, -3., 3.)
    def part(mask):
        n = int(mask.sum())
        if not n:
            return {"pairs": 0}
        x = states[mask]
        _, inverse = np.unique(x, axis=0, return_inverse=True)
        state_miss = np.zeros(int(inverse.max())+1, dtype=bool)
        np.logical_or.at(state_miss, inverse, miss[mask])
        return dict(pairs=n, outside_count=int(miss[mask].sum()), coverage=float(1.-miss[mask].mean()),
                    unique_states=len(state_miss), states_with_any_miss=int(state_miss.sum()),
                    all_observed_appearance_state_coverage=float(1.-state_miss.mean()),
                    mean_error_interval_width=float(widths[mask].mean()),
                    p95_error_interval_width=float(np.quantile(widths[mask], .95)),
                    max_error_interval_width=float(widths[mask].max()),
                    mean_clipped_action_width=float(clipped_width[mask].mean()),
                    max_outside_error=float(outside[mask].max()))
    all_mask = np.ones(len(errors), dtype=bool)
    by_appearance = {str(a): part(appearances == a) for a in sorted(set(appearances.tolist()))}
    distances = {"%.1f-%.1f" % (lo, hi): part((states[:, 0] >= lo) & (states[:, 0] < hi))
                 for lo, hi in ((6.,8.), (8.,10.), (10.,12.), (12.,14.), (14.,16.01))}
    speeds = {"%.1f-%.2f" % (lo, hi): part((states[:, 1] >= lo) & (states[:, 1] < hi))
              for lo, hi in ((0.,.5), (.5,1.), (1.,2.), (2.,2.5), (2.5,3.01))}
    worst = np.argsort(-outside, kind="stable")[:10]
    cases = [dict(distance_m=float(states[i, 0]), speed_mps=float(states[i, 1]),
                  appearance=str(appearances[i]), action_error=float(errors[i]),
                  predicted_error_bounds=prediction[i].tolist(), outside_error=float(outside[i]),
                  legacy_band=bool(band[i])) for i in worst if miss[i]]
    return dict(all_ongoing=part(all_mask), legacy_band=part(band), by_appearance=by_appearance,
                by_distance=distances, by_speed=speeds, worst_cases=cases,
                measurement_tolerance=tolerance, tolerance_is_not_a_safety_allowance=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=Path("results/environment_nbc_20261013_001/01_frozen_diagnostic"))
    parser.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() or args.epochs < 1 or args.batch_size < 1 or args.hidden < 2:
        parser.error("new output directory and positive training sizes required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.monotonic()
    diagnostic_path = args.diagnostic_dir/"metrics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    if diagnostic.get("status") != "environment_barrier_diagnostic_complete" or diagnostic.get("source_files_unchanged") is not True or diagnostic.get("test_used") is not False:
        raise ValueError("not a completed development diagnostic")
    source = {"actor": args.spvc_dir/"latent_ppo_spvc.pt", "barrier": args.spvc_dir/"sbc.pt", "representation": args.representation}
    frozen_hashes = {k: sha256(p) for k, p in source.items()}
    if any(frozen_hashes[k] != diagnostic["source_hashes"][k] for k in source):
        raise ValueError("frozen models no longer match stored action data")
    train_path, validation_path = [args.diagnostic_dir/(s+"_pairs.npz") for s in ("train", "validation")]
    pair_hashes = {"train": sha256(train_path), "validation": sha256(validation_path)}
    # Only train arrays are read before fitting. Validation is evaluated once at the end.
    with np.load(train_path, allow_pickle=False) as stream:
        train_states, targets, appearance_set = envelope_targets(stream)
        train_distances = np.unique(stream["distance_m"])
        train_rows = np.unique(stream["image_row"])
    device = torch.device(args.device)
    x = torch.from_numpy(train_states).to(device)
    y = torch.from_numpy(targets).to(device)
    model = ActionErrorBounds(args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    report = dict(experiment="train_only_state_dependent_action_error_bounds_v1", status="running",
                  training_scope="new offline action-error bounds model only; DINO/q/PPO/SBC frozen",
                  test_used=False, formal_calibration=False, rollout_run=False, sbc_reverified=False,
                  source_hashes=frozen_hashes, pair_hashes=pair_hashes, diagnostic_sha256=sha256(diagnostic_path),
                  script_sha256=sha256(Path(__file__)), training_config=vars(args).copy(),
                  target="per exact training state, lower/upper hull of image_action-q_action including zero",
                  interval_formula="[a_q+lower(d,v), a_q+upper(d,v)] clipped to [-3,3]",
                  model_inputs=["distance m", "speed m/s"], model_selection="final preset epoch; no validation selection",
                  validation_role="reused development split, not independent calibration",
                  loss="endpoint MSE + 20*undercoverage MSE + 0.1*batch maximum undercoverage squared + 0.001*width",
                  train_unique_states=len(train_states), appearances=appearance_set, history=[], results={})
    report["training_config"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["training_config"].items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"metrics.json", report)
    try:
        for epoch in range(1, args.epochs+1):
            model.train()
            permutation = torch.randperm(len(x), device=device)
            total = 0.
            for start in range(0, len(x), args.batch_size):
                ids = permutation[start:start+args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = bounds_loss(model(x[ids]), y[ids])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach())*len(ids)
            if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    pred = model(x)
                    outside = torch.maximum((pred[:, 0]-y[:, 0]).relu(), (y[:, 1]-pred[:, 1]).relu())
                    row = dict(epoch=epoch, train_loss=total/len(x),
                               train_states_with_target_miss=int((outside>1e-6).sum()),
                               max_target_miss=float(outside.max()), mean_error_width=float((pred[:, 1]-pred[:, 0]).mean()))
                report["history"].append(row)
                print("action bounds", row, flush=True)
                write_json(args.output_dir/"metrics.json", report)
        model.eval()
        checkpoint_path = args.output_dir/"action_error_bounds.pt"
        torch.save(dict(state_dict=model.cpu().state_dict(), hidden=args.hidden,
                        input_coordinates=["distance m", "speed m/s"], frozen_model_hashes=frozen_hashes,
                        empirical_development_only=True), checkpoint_path)
        # Evaluate what was saved, not a subsequently changed in-memory model.
        saved = torch.load(checkpoint_path, map_location=device)
        model = ActionErrorBounds(saved["hidden"]).to(device).eval()
        model.load_state_dict(saved["state_dict"], strict=True)
        with torch.no_grad():
            reloaded = model(x).cpu().numpy()
        np.savez_compressed(args.output_dir/"train_state_bounds.npz", states=train_states,
                            targets=targets, predicted_bounds=reloaded)
        for split, path in (("train", train_path), ("validation", validation_path)):
            with np.load(path, allow_pickle=False) as stream:
                states = np.column_stack((stream["distance_m"], stream["speed_mps"])).astype(np.float32)
                errors = stream["image_action"]-stream["q_action"]
                appearances, band, nominal = stream["appearance"], stream["fixed_legacy_band"], stream["q_action"]
                if split == "validation":
                    if np.intersect1d(train_distances, np.unique(stream["distance_m"])).size or np.intersect1d(train_rows, np.unique(stream["image_row"])).size:
                        raise ValueError("training/validation overlap")
                prediction = np.empty((len(states), 2), dtype=np.float32)
                with torch.no_grad():
                    for start in range(0, len(states), args.batch_size):
                        prediction[start:start+args.batch_size] = model(torch.from_numpy(states[start:start+args.batch_size]).to(device)).cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError("nonfinite predicted bounds")
                result = evaluate_summary(states, errors, prediction, appearances, band, nominal)
                # A deliberately wide train-only constant baseline, not a contract.
                global_bounds = np.array([targets[:, 0].min(), targets[:, 1].max()], dtype=np.float32)
                result["global_train_hull_baseline"] = evaluate_summary(states, errors,
                    np.broadcast_to(global_bounds, prediction.shape), appearances, band, nominal)["all_ongoing"]
                result["global_train_hull_bounds"] = global_bounds.tolist()
                report["results"][split] = result
                np.savez_compressed(args.output_dir/(split+"_predictions.npz"), predicted_error_bounds=prediction)
                print(split, "development coverage", result["all_ongoing"], flush=True)
                print(split, "legacy band", result["legacy_band"], flush=True)
                write_json(args.output_dir/"metrics.json", report)
        report["frozen_models_unchanged"] = {k: sha256(p) for k, p in source.items()} == frozen_hashes
        report["pair_data_unchanged"] = {"train": sha256(train_path), "validation": sha256(validation_path)} == pair_hashes
        if not report["frozen_models_unchanged"] or not report["pair_data_unchanged"]:
            raise ValueError("source changed during fit")
        report.update(status="development_only_action_bounds_fitted", checkpoint=str(checkpoint_path),
                      checkpoint_sha256=sha256(checkpoint_path), training_validation_disjoint=True,
                      safety_claim=False, independent_coverage_claim=False)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["runtime_seconds"] = time.monotonic()-started
        write_json(args.output_dir/"metrics.json", report)
    print("\n[Train-only action-error bounds: development only]")
    print("status:", report["status"])
    print("No CP calibration, PPO/SBC changes, test or safety certificate.")
    print("metrics:", args.output_dir/"metrics.json", flush=True)


if __name__ == "__main__":
    main()
