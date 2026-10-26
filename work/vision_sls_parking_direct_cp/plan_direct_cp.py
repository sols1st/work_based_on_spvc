"""Execute original parking-lot planning with the directly calibrated CP model."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--calibration',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.source,args.output=args.source.resolve(),args.output.resolve()
cal=json.loads(args.calibration.read_text())
assert cal['old_envelope_used'] is False
args.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(args.source))
from direct_cp_model import install_direct_cp
Model=install_direct_cp(args.source,cal['radius'])
m=Model()
assert not hasattr(m,'theta_poly') and not hasattr(m,'theta_bias')
assert m.direct_cp_radius==cal['radius']
os.chdir(args.output)
started=time.perf_counter()
runpy.run_path(str(args.source/'expe/parking_lot/main_parking_lot.py'),run_name='__main__')
elapsed=time.perf_counter()-started
sys.argv=['check_plan.py','--source',str(args.source),'--output',str(args.output),
          '--plan',str(args.output/'data/unicycle_of_perception_0.npz')]
runpy.run_path(str(Path(__file__).resolve().parent/'check_plan.py'),run_name='__main__')
record={'radius':cal['radius'],'old_envelope_loaded':False,'planning_wall_seconds':elapsed,
        'model':'F=radius*I_3; original dynamics, C, costs and constraints'}
(args.output/'direct_cp_plan.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
