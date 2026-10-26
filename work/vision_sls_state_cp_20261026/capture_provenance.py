"""Save reproducible source, script, protocol, environment and result identities."""
import argparse,hashlib,json,platform,subprocess,sys
from pathlib import Path
import importlib.metadata as md
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
root=p.parse_args().root.resolve();source=(root/'source').resolve()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def git(*args):
    r=subprocess.run(['git','-C',str(source),*args],capture_output=True,text=True)
    return {'returncode':r.returncode,'output':r.stdout.strip(),'stderr':r.stderr.strip()}
reference=root.parent/('vision_sls_repro_20261024' if 'parking' in root.name else 'vision_sls_quad_20261024')/'results/provenance.json'
prior=json.loads(reference.read_text())
current={str(f.relative_to(source)):sha(f) for f in sorted(source.rglob('*.py')) if '.git' not in f.parts}
changed={k:{'previous':v,'current':current.get(k)} for k,v in prior['source_python_sha256'].items() if current.get(k)!=v}
assert not changed,changed
data={'server':platform.node(),'python':sys.version,'platform':platform.platform(),
 'source_resolved':str(source),'source_revision':prior.get('upstream_source_revision',prior.get('source_revision')),
 'source_revision_evidence':'copied source tree, verified every Python hash against original reproduction provenance',
 'prior_provenance_path':str(reference),'prior_provenance_sha256':sha(reference),
 'all_previously_recorded_source_python_hashes_equal':True,
 'original_asset_provenance':{k:prior.get(k) for k in ['dinov2_source_revision','weights_url','weights_bytes','weights_sha256']},
 'source_python_sha256':current,
 'experiment_script_sha256':{f.name:sha(f) for f in sorted(root.glob('*.py'))},
 'protocol_sha256':sha(root/'protocol.json'),
 'packages':{x:md.version(x) for x in ['numpy','scipy','torch','casadi','gurobipy','drake','pybullet']},
 'result_sha256':{str(f.relative_to(root/'results')):sha(f) for f in sorted((root/'results').rglob('*'))
                  if f.is_file() and f.suffix in ['.json','.npz'] and f.name!='provenance.json'}}
(root/'results/provenance.json').write_text(json.dumps(data,indent=2)+'\n')
print('Saved provenance for '+str(root))
