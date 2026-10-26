"""Post-hoc diagnostic only: separate same-distance appearance spread from q fit.

Empirical group centroids use audit labels ONLY to explain residuals; they are
not new predictors and are not used to select or update any saved model.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch

from Aebs.dino_latent.models import SafetyProjection, PhysicalToLatent
from Aebs.dino_latent.benchmark_static_latents import FeatureAutoencoder
from Aebs.dino_latent.prepare_expanded_data import sha256


def decompose(image, proxy, groups):
    image,proxy,groups=map(np.asarray,(image,proxy,groups))
    if image.shape!=proxy.shape or image.ndim!=2 or len(groups)!=len(image): raise ValueError("Unpaired arrays")
    centroids=np.empty_like(image,dtype=np.float64)
    for g in np.unique(groups):
        keep=groups==g
        if np.max(np.abs(proxy[keep]-proxy[keep][0]))>1e-5: raise ValueError("q not constant within exact-distance group")
        centroids[keep]=image[keep].mean(0)
    mse=float(np.mean((image-proxy)**2))
    appearance=float(np.mean((image-centroids)**2))
    fit=float(np.mean((centroids-proxy)**2))
    centered=float(np.mean((image-image.mean(0))**2))
    return dict(pair_mse=mse,same_distance_appearance_mse=appearance,centroid_q_fit_mse=fit,
        identity_abs_error=abs(mse-appearance-fit),
        appearance_fraction_of_pair_mse=appearance/mse if mse else None,
        appearance_fraction_of_centered_image_variance=appearance/centered if centered else None,
        interpretation="finite-sample group decomposition, not a population lower bound or coverage guarantee")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    a=p.parse_args(); out=a.directory/"decomposition.json"
    if out.exists(): p.error("Diagnostic output already exists")
    config=json.loads((a.directory/"config.json").read_text())
    for f,digest in config["source_sha256"].items():
        if sha256(Path(f))!=digest: raise ValueError("Frozen source changed")
    source_paths=[Path(f) for f in config["source_sha256"]]
    manifest=next(f for f in source_paths if f.name=="cache_manifest.json")
    feature=next(f for f in source_paths if f.name=="features.h5")
    representation=next(f for f in source_paths if f.name=="dino_safety_latent.pt")
    records=json.loads(manifest.read_text())["records"]
    ids=json.loads((a.directory/"split.json").read_text())["internal_audit"]
    groups=np.array([records[i]["group_id"] for i in ids])
    with h5py.File(feature,"r") as f: raw=torch.from_numpy(f["features"][:][ids])
    rep=torch.load(representation,map_location="cpu")
    x=(raw-rep["feature_mean"])/rep["feature_std"]
    d=torch.tensor([records[i]["distance_m"] for i in ids],dtype=torch.float32)[:,None]/float(rep["distance_scale_m"])
    projection=SafetyProjection(); projection.load_state_dict(rep["projection_state_dict"]); projection.eval()
    original_q=PhysicalToLatent(); original_q.load_state_dict(rep["physical_to_latent_state_dict"]); original_q.eval()
    coordinates=torch.load(a.directory/"pca_coordinates.pt",map_location="cpu")
    weights={name:sha256(a.directory/(name+".pt")) for name in ("r2_frozen_refit_q","pca32","feature_ae32","feature_vae32")}
    results={}
    torch.set_num_threads(1)
    with torch.no_grad():
        native_i=projection(x); native_q=original_q(d)
        results["original_R2"]=dict(native=decompose(native_i.numpy(),native_q.numpy(),groups))
        for name in weights:
            state=torch.load(a.directory/(name+".pt"),map_location="cpu")
            q=PhysicalToLatent(); q.load_state_dict(state["q_state_dict"]); q.eval()
            predicted=q(d)*state["latent_std"]+state["latent_mean"]
            if name=="r2_frozen_refit_q":
                encoded=projection(x); common_i=encoded; common_q=predicted
            elif name=="pca32":
                basis,mean=coordinates["basis"],coordinates["mean"]
                encoded=(x-mean)@basis
                common_i=projection(encoded@basis.T+mean); common_q=projection(predicted@basis.T+mean)
            else:
                codec=FeatureAutoencoder(name=="feature_vae32")
                codec.load_state_dict(state["codec_state_dict"]); codec.eval()
                encoded=codec.encode(x)[0]
                common_i=projection(codec.decoder(encoded)); common_q=projection(codec.decoder(predicted))
            results[name]=dict(native=decompose(encoded.numpy(),predicted.numpy(),groups),
                               common=decompose(common_i.numpy(),common_q.numpy(),groups))
    for name,digest in weights.items():
        if sha256(a.directory/(name+".pt"))!=digest: raise ValueError("Model changed")
    output=dict(status="posthoc_same_distance_decomposition_complete_no_training",images=len(ids),
        exact_distance_groups=len(np.unique(groups)),models_unchanged=True,model_sha256=weights,results=results)
    out.write_text(json.dumps(output,indent=2,allow_nan=False))
    for name,r in results.items(): print(name,r["native"],flush=True)
    print("diagnostic:",out,flush=True)


if __name__=="__main__": main()
