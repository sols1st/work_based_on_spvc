"""Record the host and file identities used by the quadrotor reproduction."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

root = Path(__file__).resolve().parent
source = root/'source'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
v = {
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'ssh_endpoint': 'root@10.11.154.191:20020', 'hostname': platform.node(),
    'task_root': str(root), 'python_executable': sys.executable, 'python': sys.version,
    'environment': 'Reuses the parking-lot task venv via symlink; no package changes.',
    'source_revision': '89b5896a326a57dc8bd7edf7e93f13105a47cdd4',
    'dinov2_source_revision': '7764ea0f912e53c92e82eb78a2a1631e92725fc8',
    'weights_url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
    'weights_bytes': (root/'assets/dinov2_vitl14_pretrain.pth').stat().st_size,
    'weights_sha256': (root/'assets/dinov2_vitl14_pretrain.sha256').read_text().strip(),
    'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version',
                                    '--format=csv,noheader'],text=True).strip(),
    'versions': {k: importlib.metadata.version(k) for k in
                 ['torch','torchvision','numpy','scipy','drake','casadi','numba','pybullet','gym','pillow']},
    'source_python_sha256': {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))},
    'quad_model_sha256': {str(p.relative_to(source)):sha(p) for p in
                          sorted((source/'expe/quad_perception/data').glob('*.pt'))},
    'envelope_sha256': sha(source/'dyn/theta_lp_large_new2_val.pt'),
    'adapter_scripts_sha256': {p.name:sha(p) for p in sorted(root.glob('*.py'))},
    'run_script_sha256': sha(root/'run_all.sh'),
}
assert v['weights_bytes']==1217586395
assert v['weights_sha256']=='d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'
(root/'results/provenance.json').write_text(json.dumps(v,indent=2)+'\n')
print('Recorded provenance for',v['hostname'])
