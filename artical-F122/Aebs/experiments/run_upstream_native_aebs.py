"""Run the upstream AEBS VT implementation, binding every reported score to weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=float) + "\n")


def run(args):
    root = args.upstream.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    sys.path[:0] = [str(root / "auto_LiRPA"), str(root)]
    from Aebs.system.env import Aebs
    from Aebs.VT.loop import Loop
    from Aebs.VT.train import VTLearner
    from Aebs.VT.verify import VTVerifier
    import auto_LiRPA
    paths = [
        "Aebs/VT/loop.py", "Aebs/VT/train.py", "Aebs/VT/verify.py", "Aebs/VT/utils.py",
        "Aebs/system/env.py", "Combined_network/model.py", "cGAN/taxi_models_and_data.py",
        "Aebs/controller/state_net_trained.pth", "Aebs/data/Downsampled.h5",
        "Aebs/controller/best_model/best_model.zip", "Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth",
    ]
    # Upstream saves CUDA-tagged tensors. Remap only their storage when running on CPU.
    if not torch.cuda.is_available():
        load = torch.load
        def cpu_load(*items, **kw):
            kw.setdefault("map_location", "cpu")
            return load(*items, **kw)
        torch.load = cpu_load
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    log = dict(upstream="QLong01/artical", commit="6279872ed7708cd103c0801d758d2a95c203ce27",
               arm=args.arm, seed=args.seed, threads=args.threads, device="cuda" if torch.cuda.is_available() else "cpu",
               torch_version=torch.__version__, cuda=torch.version.cuda, source={p:digest(root/p) for p in paths},
               auto_lirpa_path=str(Path(auto_LiRPA.__file__).resolve()),
               criterion="native Loop.run and VTVerifier; best score among accepted, separated candidates",
               rounds="source default iterations 0 through 100 inclusive, wall time 3600s",
               note="fixed skips the controller update; alternating uses upstream training unmodified; checkpoint snapshots before policy update")
    write(out / "protocol.json", log)
    env = Aebs(.05)
    learner = VTLearner([2,16,8,1], env, 2., 4., .1, 1., .95, square_l_output=True)
    verifier = VTVerifier(learner, env, learner.create_bounded_module(learner.l_model),
                          batch_size=2048, reach_prob=.9, fail_check_fast=True)
    loop = Loop(learner, verifier, env, jitter_grid=env.space_split, soft_constraint=False)
    original_train = loop.train
    original_check = verifier.check_dec_cond
    rows = []
    verdict = {}
    best = -np.inf
    def check(k):
        result = original_check(k)
        verdict.update(accepted=bool(result[0]), hard_violations=int(result[1]))
        return result
    def save_pair(name):
        torch.save(dict(policy=learner.p_net.state_dict(), barrier=learner.l_model.state_dict(),
                        iteration=loop.iter), out / (name + ".pt"))
    def train(which, epochs=10, violation_buffer=None):
        nonlocal best
        if which == "p":
            row = dict(iteration=loop.iter, **verdict)
            if row["accepted"]:
                for key in ("ub_init", "lb_unsafe", "domain_min"):
                    row[key] = float(loop.info[key])
                if row["lb_unsafe"] >= row["ub_init"]:
                    row["score"] = 1. - (row["ub_init"]-row["domain_min"])/(row["lb_unsafe"]-row["domain_min"])
                    if row["score"] > best:
                        save_pair("best")
                        best = row["score"]
            rows.append(row)
            with (out / "rounds.jsonl").open("a") as fp:
                fp.write(json.dumps(row, allow_nan=False) + "\n")
            if loop.iter == 100:
                save_pair("final_verified")
            if args.arm == "fixed":
                return
        return original_train(which, epochs, violation_buffer)
    verifier.check_dec_cond = check
    loop.train = train
    loop.run(3600)
    scores = [r for r in rows if "score" in r]
    result = dict(rounds=len(rows), author_accepted=sum(r["accepted"] for r in rows),
                  scored=len(scores), best=max(scores,key=lambda r:r["score"]) if scores else None,
                  final=rows[-1] if rows else None,
                  weights={p.name:digest(p) for p in sorted(out.glob("*.pt"))})
    write(out / "summary.json", result)
    print("NATIVE_SUMMARY", result, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("upstream", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--arm", choices=["fixed","alternating"], required=True)
    cli.add_argument("--seed", type=int, default=202611171)
    cli.add_argument("--threads", type=int, default=4)
    run(cli.parse_args())
