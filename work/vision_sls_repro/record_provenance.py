"""Record source fingerprints and host details for this reproduction."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

root = Path(__file__).resolve().parent
source = root / 'source'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'ssh_endpoint': 'root@10.11.154.191:20020',
    'hostname': platform.node(),
    'platform': platform.platform(),
    'python': sys.version,
    'task_root': str(root),
    'python_executable': sys.executable,
    'environment': 'Task-local venv borrowing preinstalled torch via system-site-packages.',
    'upstream_source_revision': '89b5896a326a57dc8bd7edf7e93f13105a47cdd4',
    'dinov2_source_revision': '7764ea0f912e53c92e82eb78a2a1631e92725fc8',
    'weights_url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth',
    'weights_bytes': (root / 'assets/dinov2_vitg14_pretrain.pth').stat().st_size,
    'weights_sha256': (root / 'assets/dinov2_vitg14_pretrain.sha256').read_text().strip(),
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
                                    '--format=csv,noheader'], text=True).strip(),
    'versions': {name: importlib.metadata.version(name) for name in
                 ['torch', 'torchvision', 'numpy', 'scipy', 'drake', 'casadi',
                  'numba', 'pybullet', 'gym', 'gurobipy', 'pillow', 'matplotlib']},
    'source_python_sha256': {
        str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*.py'))
        if '__pycache__' not in p.parts
    },
    'provided_data_sha256': {
        str(p.relative_to(source)): sha(p)
        for p in sorted((source / 'expe/parking_lot/data').iterdir())
        if p.suffix in {'.pt', '.yaml'}
    },
    'dynamics_model_npz_sha256': {
        str(p.relative_to(source)): sha(p) for p in sorted((source / 'dyn').rglob('*.npz'))
    },
    'adaptation_scripts_sha256': {p.name: sha(p) for p in sorted(root.glob('*.py'))},
    'run_script_sha256': sha(root / 'run_all.sh'),
}
assert manifest['weights_sha256'] == 'baf8467e50af277596bbbafa06887c177ee899ab46033649c383577d7e9309d3'
(root / 'results/provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Recorded provenance for', manifest['hostname'])
