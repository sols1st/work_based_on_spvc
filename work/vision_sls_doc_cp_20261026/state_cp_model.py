"""Install the document's box using its proven outer Euclidean-ball enclosure."""
import ast,json
from pathlib import Path
import numpy as np
import casadi as ca
from doc_cp_features import box_bound

def install_state_cp(source:Path,model_path:Path):
    model=json.loads(model_path.read_text());scene=model['scene']
    assert np.isfinite(model['cp_multiplier']) and model['cp_multiplier']>0
    if scene=='parking':
        from dyn.unicycle import Unicycle as Parent
        from dyn.unicycle_of_perception import Unicycle_OF as Model
        file=source/'dyn/unicycle_of_perception.py';nx,ny=4,3
    else:
        from dyn.quad_10d import Quad10d as Parent
        from dyn.quad_10d_of_perception import Quad10d_OF as Model
        file=source/'dyn/quad_10d_of_perception.py';nx,ny=10,5
    nodes=[n for n in ast.walk(ast.parse(file.read_text())) if isinstance(n,ast.Assign) and
      any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='C' for t in n.targets)]
    assert len(nodes)==1;C=np.asarray(ast.literal_eval(nodes[0].value.args[0]),float)
    x=ca.MX.sym('state',nx);f=x[[0,1]] if scene=='parking' else x[[0,1,2,6,7]]
    z=(f-ca.DM(model['feature_mean']))/ca.DM(model['feature_scale'])
    terms=[]
    for row in model['powers']:
        term=1
        for j,power in enumerate(row):term=term*z[j]**power
        terms.append(term)
    b0=ca.dot(ca.DM(model['coefficients']),ca.vertcat(*terms))
    box=model['cp_multiplier']*ca.fmax(model['epsilon'],b0);ball=np.sqrt(ny)*box
    boxfn=ca.Function('doc_cp_box',[x],[box]);ballfn=ca.Function('doc_cp_ball',[x],[ball])
    def init(self):
        Parent.__init__(self);self.ny=self.nv=ny;self.C=C.copy()
        self.F=float(ballfn(np.zeros(nx)))*np.eye(ny)
        self.doc_cp_box_fn=boxfn;self.state_cp_bound_fn=ballfn;self.state_cp_metadata=model
        if scene=='quad':self.use_perception_model=True
    def scalar(x):return ballfn(x) if isinstance(x,(ca.MX,ca.SX)) else float(ballfn(x))
    def get_disturbance(self,x,scale=1.):
        val=scalar(x)*scale
        if scene=='parking':return val
        return ca.repmat(val,ny,1) if isinstance(val,(ca.MX,ca.SX)) else np.full(ny,val)
    def measurement(self,x,v):return self.C@x+scalar(x)*v
    def plot(self,ax):
        xx,yy=np.meshgrid(np.linspace(-3.5,.5,160),np.linspace(-2,2,160))
        s=np.column_stack((xx.ravel(),yy.ravel(),np.zeros((xx.size,2))))
        return ax.contourf(xx,yy,box_bound(s,model).reshape(xx.shape),levels=20,cmap='Greys')
    Model.__init__=init;Model.get_disturbance=get_disturbance;Model.measurement=measurement
    if scene=='parking':Model.plot_disturbance_map=plot
    return Model
