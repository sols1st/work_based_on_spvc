"""Run the minimal AEBS improvement pipeline and write a single summary."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from Aebs.mvp.robust_sbc import train_barrier
from Aebs.semantic.safety_filter import load_controller
from Aebs.uncertainty.collect import collect_deterministic_process_uncertainty, collect_uncertainty


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def run_command(command: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def prepare_semantic(config: Dict, output_dir: Path) -> Dict:
    semantic_dir = output_dir / "01_semantic"
    source_dir = Path(config["semantic_source_dir"])
    checkpoint = semantic_dir / "semantic_encoder.pt"
    metrics_path = semantic_dir / "metrics.json"
    copied_checkpoint = copy_if_exists(source_dir / "semantic_encoder.pt", checkpoint)
    copied_metrics = copy_if_exists(source_dir / "metrics.json", metrics_path)
    copy_if_exists(source_dir / "split_indices.npz", semantic_dir / "split_indices.npz")
    if copied_checkpoint and copied_metrics:
        metrics = load_json(metrics_path)
        metrics["mvp_status"] = "reused_existing_seed7_checkpoint"
        save_json(metrics_path, metrics)
        return {
            "status": "done",
            "checkpoint": str(checkpoint),
            "metrics": metrics,
        }

    command = [
        sys.executable,
        "-u",
        "-m",
        "Aebs.semantic.train",
        "--data",
        config["data"],
        "--output-dir",
        str(semantic_dir),
        "--seed",
        str(config["seed"]),
        "--alpha",
        "0.05",
    ]
    run_command(command, semantic_dir / "train.log")
    return {
        "status": "done",
        "checkpoint": str(checkpoint),
        "metrics": load_json(metrics_path),
    }


def run_controller(config: Dict, output_dir: Path, semantic_checkpoint: str) -> Dict:
    controller_dir = output_dir / "02_controller"
    metrics_path = controller_dir / "metrics.json"
    checkpoint = controller_dir / "robust_controller.zip"
    if metrics_path.exists() and checkpoint.exists():
        return {
            "status": "done",
            "checkpoint": str(checkpoint),
            "metrics": load_json(metrics_path),
        }
    settings = config["controller"]
    command = [
        sys.executable,
        "-u",
        "-m",
        "Aebs.semantic.robust_controller",
        "--semantic-checkpoint",
        semantic_checkpoint,
        "--controller",
        config["baseline_controller"],
        "--data",
        config["data"],
        "--output-dir",
        str(controller_dir),
        "--seed",
        str(config["seed"]),
        "--timesteps",
        str(settings["timesteps"]),
        "--eval-episodes",
        str(settings["eval_episodes"]),
        "--learning-rate",
        str(settings["learning_rate"]),
        "--distill-epochs",
        str(settings["distill_epochs"]),
        "--distill-learning-rate",
        str(settings["distill_learning_rate"]),
        "--distill-batch-size",
        str(settings["distill_batch_size"]),
    ]
    run_command(command, controller_dir / "train.log")
    return {
        "status": "done",
        "checkpoint": str(checkpoint),
        "metrics": load_json(metrics_path),
    }


def run_uncertainty(config: Dict, output_dir: Path, semantic_checkpoint: str, controller_checkpoint: str) -> Dict:
    settings = config["uncertainty"]
    uncertainty_dir = output_dir / settings.get("output_name", "03_uncertainty")
    metrics_path = uncertainty_dir / "metrics.json"
    model_path = uncertainty_dir / "state_dependent_uncertainty.npz"
    if metrics_path.exists() and model_path.exists():
        return {
            "status": "done",
            "model": str(model_path),
            "metrics": load_json(metrics_path),
        }
    source = settings.get("source", "legacy_semantic_action_delta")
    if source == "deterministic_process_residual":
        metrics = collect_deterministic_process_uncertainty(
            config["data"],
            uncertainty_dir,
            int(config["seed"]),
            int(settings["bins"]),
            int(settings["states_per_cell"]),
        )
        return {
            "status": "done",
            "model": str(model_path),
            "metrics": metrics,
        }
    if source != "legacy_semantic_action_delta":
        raise ValueError(f"unknown uncertainty source: {source}")
    controller = load_controller(controller_checkpoint)
    metrics = collect_uncertainty(
        controller,
        semantic_checkpoint,
        config["data"],
        uncertainty_dir,
        int(config["seed"]),
        int(settings["bins"]),
        int(settings["states_per_cell"]),
        int(settings["errors_per_state"]),
        float(settings["alpha"]),
    )
    return {
        "status": "done",
        "model": str(model_path),
        "metrics": metrics,
    }


def run_robust_sbc(config: Dict, output_dir: Path, semantic_checkpoint: str, uncertainty_path: str, robust_controller: str) -> Dict:
    settings = config["robust_sbc"]
    sbc_dir = output_dir / settings.get("output_name", "04_robust_sbc")
    metrics_path = sbc_dir / "metrics.json"
    checkpoint = sbc_dir / "barrier.pt"
    if metrics_path.exists() and checkpoint.exists():
        return {
            "status": "done",
            "checkpoint": str(checkpoint),
            "metrics": load_json(metrics_path),
        }
    controller_name = settings["controller"]
    if controller_name == "baseline":
        controller_path = config["baseline_controller"]
    elif controller_name == "robust":
        controller_path = robust_controller
    elif controller_name == "safety_filter":
        controller_path = config["safety_filter_controller"]
    else:
        raise ValueError(f"unknown robust_sbc controller: {controller_name}")
    metrics = train_barrier(
        semantic_checkpoint,
        uncertainty_path,
        controller_path,
        config["data"],
        sbc_dir,
        int(config["seed"]),
        int(settings["epochs"]),
        int(settings["train_states"]),
        int(settings["batch_size"]),
        float(settings["learning_rate"]),
        float(settings["epsilon"]),
        int(settings["grid_size"]),
        float(settings.get("decrease_weight", 100.0)),
        float(settings.get("init_weight", 1.0)),
        float(settings.get("unsafe_weight", 1.0)),
        bool(settings.get("square_output", False)),
        int(settings.get("region_warmup_epochs", 25)),
        float(settings.get("warmup_decrease_weight", 0.0)),
        float(settings.get("init_target", 1.0)),
        float(settings.get("unsafe_target", 10.0)),
        bool(settings.get("include_verifier_grid_train", False)),
        float(settings.get("max_decrease_weight", 25.0)),
        float(settings.get("topk_decrease_weight", 0.0)),
        float(settings.get("topk_decrease_fraction", 0.1)),
        str(settings.get("initial_barrier", "")),
        float(settings.get("terminal_speed_threshold", 0.0)),
        int(settings.get("hard_case_count", 2048)),
        float(settings.get("hard_case_distance_m", 13.35)),
        float(settings.get("hard_case_speed", 3.0)),
        float(settings.get("hard_case_distance_radius_m", 0.8)),
        float(settings.get("hard_case_speed_radius", 0.2)),
        float(settings.get("region_topk_fraction", 0.1)),
        float(settings.get("goal_distance_m", 6.0)),
        float(settings.get("goal_speed", 0.5)),
        float(settings.get("unsafe_distance_low_m", 5.0)),
        float(settings.get("unsafe_distance_high_m", 6.0)),
        float(settings.get("unsafe_speed", 0.5)),
        float(settings.get("goal_weight", 10.0)),
        float(settings.get("goal_target", 1.0)),
        bool(settings.get("use_recoverable_domain", False)),
        float(settings.get("max_braking", 3.0)),
        bool(settings.get("enforce_goal_constraint", False)),
    )
    return {
        "status": "done",
        "checkpoint": str(checkpoint),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Aebs/mvp/config.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_json(Path(args.config))
    output_dir = Path(config["output_dir"])
    if args.force and output_dir.exists():
        raise SystemExit("--force is intentionally disabled for the MVP runner; move results aside manually if needed.")

    started = time.time()
    save_json(output_dir / "config.json", config)
    summary = {
        "experiment": "aebs_mvp_minimal",
        "seed": config["seed"],
        "started_at_unix": started,
        "steps": {},
        "resolved": [],
        "unresolved": [],
    }

    semantic = prepare_semantic(config, output_dir)
    summary["steps"]["01_semantic"] = semantic
    summary["resolved"].append("Step 0/1: unified config and semantic checkpoint reuse are working.")

    controller = run_controller(config, output_dir, semantic["checkpoint"])
    summary["steps"]["02_controller"] = controller
    summary["resolved"].append("Step 2: controller training now uses exact/uniform/endpoint semantic mixture plus light imitation distillation.")

    uncertainty = run_uncertainty(config, output_dir, semantic["checkpoint"], controller["checkpoint"])
    summary["steps"]["03_uncertainty"] = uncertainty
    summary["resolved"].append("Step 3: 4x4 state-dependent disturbance boxes are generated through one shared API.")

    robust_sbc = run_robust_sbc(
        config,
        output_dir,
        semantic["checkpoint"],
        uncertainty["model"],
        controller["checkpoint"],
    )
    summary["steps"]["04_robust_sbc"] = robust_sbc
    summary["resolved"].append("Step 4/5: a minimal barrier trainer and fixed-grid robust verifier now run against the semantic contract and uncertainty boxes.")

    summary["unresolved"].extend(
        [
            "The robust controller fine-tuning result is worse than the baseline and should not be used as the final policy.",
            "The first robust SBC verifier result must be inspected before claiming any certificate.",
            "State discretization and statistical margins are still conservative placeholders in the MVP verifier.",
        ]
    )
    summary["runtime_seconds"] = float(time.time() - started)
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
