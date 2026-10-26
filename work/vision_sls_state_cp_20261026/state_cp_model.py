"""Install a genuinely state-dependent CP model without loading old parameters."""
import ast,json
from pathlib import Path
import numpy as np
import casadi as ca
from state_cp_features import bound

def install_state_cp(source:Path,model_path:Path):
    model=json.loads(model_path.read_text());scene=model['scene']
    if scene=='parking':
        from dyn.unicycle import Unicycle as Parent
        from dyn.unicycle_of_perception import Unicycle_OF as Model
        file=source/'dyn/unicycle_of_perception.py';nx,ny=4,3
    else:
        from dyn.quad_10d import Quad10d as Parent
        from dyn.quad_10d_of_perception import Quad10d_OF as Model
        file=source/'dyn/quad_10d_of_perception.py';nx,ny=10,5
    nodes=[n for n in ast.walk(ast.parse(file.read_text())) if isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and
               t.value.id=='self' and t.attr=='C' for t in n.targets)]
    assert len(nodes)==1
    C=np.asarray(ast.literal_eval(nodes[0].value.args[0]),dtype=float)
    x=ca.MX.sym('state',nx)
    f=ca.vertcat(x[0],x[1],ca.sin(x[2]),ca.cos(x[2])) if scene=='parking' else x[[0,1,2,6,7]]
    z=(f-ca.DM(model['feature_mean']))/ca.DM(model['feature_scale'])
    d=f.shape[0]
    terms=ca.vertcat(1,*[z[j] for j in range(d)],*[z[i]*z[j] for i in range(d) for j in range(i,d)])
    g=model['target_scale']*ca.dot(ca.DM(model['coefficients']),terms)
    b=ca.fmax(model['floor'],g+model['cp_offset'])
    bfn=ca.Function('state_cp_bound',[x],[b])
    def init(self):
        Parent.__init__(self)
        self.ny=self.nv=ny;self.C=C.copy()
        self.F=float(bfn(np.zeros(nx)))*np.eye(ny)
        self.state_cp_metadata=model;self.state_cp_bound_fn=bfn
        if scene=='quad':self.use_perception_model=True
    def scalar(x):
        if isinstance(x,(ca.MX,ca.SX)):return bfn(x)
        return float(bfn(x))
    def get_disturbance(self,x,scale=1.):
        v=scalar(x)*scale
        if scene=='parking':return v
        return ca.repmat(v,ny,1) if isinstance(v,(ca.MX,ca.SX)) else np.full(ny,v)
    def measurement(self,x,v):return self.C@x+scalar(x)*v
    def plot_disturbance_map(self,ax):
        xx,yy=np.meshgrid(np.linspace(-3.5,.5,160),np.linspace(-2,2,160))
        states=np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,np.pi/2),np.zeros(xx.size)))
        return ax.contourf(xx,yy,bound(states,model).reshape(xx.shape),levels=20,cmap='Greys')
    Model.__init__=init;Model.get_disturbance=get_disturbance;Model.measurement=measurement
    if scene=='parking':Model.plot_disturbance_map=plot_disturbance_map
    return Model
