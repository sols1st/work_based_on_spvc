"""Cross the newly synthesized upstream controller and release robust controller on the same camera replay."""
import argparse
from pathlib import Path

import numpy as np
import torch

from Aebs.experiments.native_spvc_camera_cross import Models, read, write, collect, tensor
from Aebs.vision_sls.run import sha256
from Combined_network.model import CombinedPolicyNetwork


class PairedActors(Models):
    def __init__(self, checkpoint):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        backbone = self.ppo.policy.mlp_extractor.policy_net
        self.trained = CombinedPolicyNetwork(backbone, self.ppo.policy.action_net)
        weights = {key[len("controller_net."):]:v for key,v in saved["policy"].items()
                   if key.startswith("controller_net.")}
        self.trained.load_state_dict(weights)
        self.trained.eval()

    def action(self, actor, observed, speed):
        if actor == "native_trained":
            inputs = tensor(np.column_stack([observed / self.original_scale,speed]))
            return self.trained(inputs).flatten().clamp(-3,3).numpy()
        return super().action(actor,observed,speed)


@torch.inference_mode()
def run(args):
    out = args.native / "camera_cross"
    out.mkdir(exist_ok=False)
    bank = args.bank / "full_release_bank"
    frozen = args.bank / "fixed_tube_2000"
    ckpt = args.native/"alternating/best.pt"
    assert read(args.native/"alternating/summary.json")["weights"]["best.pt"] == sha256(ckpt)
    protocol = read(frozen/"protocol.json")
    write(out/"protocol.json",dict(arms=["native_trained", "robust_actor"],
        inputs=["rgb_native","rgb_dino"], source=sha256(Path(__file__)), native=sha256(ckpt),
        bank=sha256(bank/"estimates.npz"), original_protocol=sha256(frozen/"protocol.json"),
        n=protocol["n"], seed=protocol["seed"], batch=protocol["batch"],
        common="identical fixed dense camera law and underlying initial and appearance draws",
        primary="crossed empirical finite-horizon stopping outcomes; no stochastic certification claim on this camera law"))
    torch.set_num_threads(2)
    models = PairedActors(ckpt)
    with np.load(bank/"estimates.npz") as arr:
        data = {front:[(arr["distance_m"][arr["appearance"]==i],arr[field][arr["appearance"]==i])
                       for i in range(12)] for front,field in [("rgb_native","native_m"),("rgb_dino","dino_m")]}
    # Keep numerical batch shapes, ordering, sample initial states, and labels identical to the prior common-actor run.
    with np.load(frozen/"rgb_dino.npz") as same:
        initial, appearance = same["initial"],same["appearance"]
    batch=protocol["batch"]
    results={}
    for actor in ("native_trained","robust_actor"):
        for name in data:
            rows=[collect(models,data,name,actor,initial[b:b+batch],appearance[b:b+batch])
                  for b in range(0,len(initial),batch)]
            records={key:np.concatenate([r[key] for r in rows]) for key in rows[0]}
            key=actor+"__"+name
            np.savez_compressed(out/(key+".npz"),**records)
            if actor=="robust_actor":
                with np.load(frozen/(name+".npz")) as reference:
                    for item in ("outcomes","max_truth_m","appearance","initial","steps"):
                        np.testing.assert_array_equal(records[item],reference[item])
            outcomes=dict(zip(*[v.tolist() for v in np.unique(records["outcomes"],return_counts=True)]))
            results[key]=dict(outcomes=outcomes,median_peak_m=float(np.median(records["max_truth_m"])),
                              in_common_tube=int(np.sum(records["max_truth_m"]<=.45)))
    write(out/"summary.json",results)
    print(results,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("native",type=Path)
    cli.add_argument("bank",type=Path)
    run(cli.parse_args())
