"""Explore the shared gap between the passing and failing registered stress levels."""
import argparse
import json
from pathlib import Path

import torch

from ..lane_keeping import core
from ..lane_keeping.learned_sbc import load_sbc_checkpoint, sha256
from ..lane_keeping.policy import load_actor
from ..lane_keeping.robust_trajectory_cp import write_json
from .audit import dynamic
from .collect import ARMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    coarse = json.loads((root / "verification.json").read_text())
    contracts = json.loads((root / "contracts.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    config = protocol["audit"]
    assert all(coarse["runs"][a]["0.15"]["status"] == "pass" and
               coarse["runs"][a]["0.2"]["status"] == "witness" for a in ARMS)
    package = root / "inputs/package"
    for name, key in (("actor", "actor_sha256"), ("sbc", "sbc_sha256")):
        assert sha256(package / f"{name}.pt") == protocol["binding"][key]
    assert sha256(root / "contracts.json") == coarse["contracts_sha256"]
    assert sha256(Path(__file__).with_name("audit.py")) == coarse["audit_source"]
    torch.set_num_threads(2)
    actor = load_actor(package / "actor.pt").double().requires_grad_(False)
    barrier = load_sbc_checkpoint(package / "sbc.pt")[0].double().requires_grad_(False)
    # Monotonicity in additional disturbance means that a newly found witness
    # remains valid at every larger radius. Unknown cells do not imply failure.
    levels = [i / 1000 for i in range(155, 200, 5)]
    record = dict(coarse=sha256(root / "verification.json"),
                  source=sha256(Path(__file__)), levels=levels, arms={})
    for arm in ARMS:
        rows = {}
        max_pass, first_witness = 0.15, 0.20
        for level in levels:
            by_curve = {}
            for curve in core.KAPPAS:
                outcome = dynamic(actor, barrier, curve, contracts["arms"][arm], level, config)
                by_curve[curve] = outcome
                if outcome["status"] == "witness":
                    break
            flags = [row["status"] for row in by_curve.values()]
            status = ("witness" if "witness" in flags else
                      "unresolved" if "unresolved" in flags else "pass")
            rows[level] = dict(status=status, curvatures=by_curve)
            print(arm, level, status, flush=True)
            if status == "pass":
                max_pass = level
            elif status == "witness":
                first_witness = level
                break
        record["arms"][arm] = dict(largest_pass=max_pass, first_witness=first_witness, runs=rows)
        write_json(root / "refinement.json", record)


if __name__ == "__main__":
    main()
