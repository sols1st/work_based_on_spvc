"""Fingerprint the registered protocol, adapters, and unchanged original core."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

root=Path(__file__).resolve().parent
source=root/'source'
def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            digest.update(chunk)
    return digest.hexdigest()
prior=json.loads(Path('/root/work_based_on_spvc/vision_sls_repro_20261024/results/provenance.json').read_text())
core=['expe/parking_lot/main_parking_lot.py',
      'expe/parking_lot/run_parking_lot_dino.py','dyn/unicycle.py','dyn/unicycle_of_perception.py',
      'solver/SCP_OF_SLS.py','solver/fast_OF_SLS.py','solver/parallel_riccati.py']
current={name:sha(source/name) for name in core}
assert all(current[name]==prior['source_python_sha256'][name] for name in core)
assets={name:sha(source/name) for name in
        dict(prior['provided_data_sha256'],**prior['dynamics_model_npz_sha256'])}
assert all(assets[name]==expected for name,expected in
           dict(prior['provided_data_sha256'],**prior['dynamics_model_npz_sha256']).items())
weights_hash=sha(root/'assets/dinov2_vitg14_pretrain.pth')
assert weights_hash==prior['weights_sha256']
result={'host':platform.node(),'python':sys.version,'task_root':str(root),
        'original_core_unchanged':True,'original_core_sha256':current,
        'source_revision':prior['upstream_source_revision'],
        'protocol_sha256':sha(root/'protocol.json'),
        'adapter_sha256':{p.name:sha(p) for p in sorted(root.glob('*.py'))},
        'run_script_sha256':sha(root/'run_all.sh'),
        'weights_sha256':weights_hash,'original_assets_sha256':assets,
        'versions':{k:importlib.metadata.version(k) for k in
                    ['torch','torchvision','numpy','scipy','casadi','numba','drake','pybullet']}}
for name in ['calibration','baseline','direct_cp']:
    p=root/'results'/name/'diagnostic_traces.npz'
    if p.exists(): result[name+'_traces_sha256']=sha(p)
(root/'results/provenance.json').write_text(json.dumps(result,indent=2)+'\n')
print('Protocol and original-core checks recorded.')
