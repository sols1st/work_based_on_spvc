"""Reload the final actor and recompute held-out observations from real images."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .deploy import VisionSLSController
from .run import ROOT, evaluate, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,default=ROOT/'results/vision_sls_spvc')
    parser.add_argument('--certificate-dir',type=Path,help='Enable the saved CP-NCBF action filter')
    parser.add_argument('--data',type=Path,default=ROOT/'Aebs/data/Downsampled.h5')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    root = args.artifact_dir
    checkpoint = torch.load(root/'observation.pt',map_location='cpu',weights_only=False)
    if sha256(args.data) != checkpoint['provenance']['data_sha256']:
        parser.error('evaluation data do not match the saved split')
    if args.certificate_dir:
        from .deploy_cp import VisionSLSCPController
        controller = VisionSLSCPController(root,args.certificate_dir,args.device)
    else:
        controller = VisionSLSController(root,args.device)
    ids = np.load(root/'splits.npz')['test']
    with h5py.File(args.data,'r') as data:
        images = torch.tensor(np.asarray(data['X_train'])[ids])
        distances = np.asarray(data['y_train']).reshape(-1)[ids]
    with torch.no_grad():
        observations = torch.cat([controller.encoder(batch.to(args.device)).cpu()
                                  for batch in images.split(16)])
        image_actions = controller(images[:4].to(args.device),
                                   torch.full((4,),2.75,device=args.device)).cpu()
    policy = controller.policy.cpu()
    expected = policy.forward_observation(observations[:4],torch.full((4,),2.75))
    if not torch.allclose(image_actions,expected,atol=2e-4,rtol=2e-4):
        raise RuntimeError('reloaded image actor and observation actor disagree')
    result = dict(saved_actor_image_reload_passed=True,
                  real_test_images=len(ids),image_actions=image_actions.tolist(),
                  action_path_max_gap=float((image_actions-expected).abs().max()),
                  nominal=evaluate(policy,checkpoint['distance_scale_m']),
                  heldout_image_replay=evaluate(policy,checkpoint['distance_scale_m'],distances,
                                               observations.numpy().reshape(-1)),
                  live_carla=False,image_closed_loop_certified=False)
    result['cp_ncbf_filter_enabled'] = args.certificate_dir is not None
    if not args.certificate_dir and (root/'summary.json').exists():
        summary=json.loads((root/'summary.json').read_text())
        if summary.get('status')=='continuous_state_numerical_interval_checks_passed':
            from .package_sbc import verified_bundle_status
            result['sbc_verification']=verified_bundle_status(root)
            result['image_closed_loop_certified_scope']='unseen images/live CARLA; distinct from the verified fixed replay model'
    output_root = args.certificate_dir if args.certificate_dir else root
    (output_root/'evaluation_reload.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
