"""Extract a compact record from the verbose original SafePVC log."""

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    iterations = [int(value) for value in re.findall(r"#### Iteration (\d+)", text)]
    probabilities = [float(value) for value in re.findall(r"at least ([0-9.]+)%", text)]
    violations = [int(value) for value in re.findall(r"^violations=(\d+)$", text, flags=re.MULTILINE)]
    runtimes = [float(value) for value in re.findall(r"'runtime': ([0-9.eE+-]+)", text)]
    result = {
        "experiment": "baseline_original",
        "completed_iterations": len(iterations),
        "last_iteration": max(iterations) if iterations else None,
        "best_reported_safety_probability": max(probabilities) / 100.0 if probabilities else 0.0,
        "reported_probability_count": len(probabilities),
        "minimum_hard_violations": min(violations) if violations else None,
        "final_hard_violations": violations[-1] if violations else None,
        "runtime_seconds": max(runtimes) if runtimes else None,
        "terminated_by_iteration_limit": "Iteration limit reached" in text,
        "caveats": [
            "The original code is not seeded, so this is not a deterministic baseline.",
            "The logged safety bound uses a hard-coded discretization factor and an empirical uniform disturbance model.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
