"""Direct CP observation ball for the parking-lot system; no old bound loaded."""
import ast
from pathlib import Path
import numpy as np
import torch

def install_direct_cp(source: Path,radius: float):
    assert np.isfinite(radius) and radius>0
    from dyn.unicycle import Unicycle
    from dyn.unicycle_of_perception import Unicycle_OF
    tree=ast.parse((source/'dyn/unicycle_of_perception.py').read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and
               t.value.id=='self' and t.attr=='C' for t in n.targets)]
    assert len(nodes)==1
    C=np.array(ast.literal_eval(nodes[0].value.args[0]),dtype=float)
    assert C.shape==(3,4)
    def init(self):
        Unicycle.__init__(self)
        self.ny=self.nv=3
        self.C=C.copy()
        self.F=radius*np.eye(3)
        self.direct_cp_radius=radius
    def get_disturbance(self,x,scale=1.):
        return radius*scale
    def measurement(self,x,v):
        return self.C@x+self.F@v
    def cov_fn(self,x,scale=1.):
        if isinstance(x,torch.Tensor):
            return torch.full((x.shape[0],1),radius*scale,device=x.device,dtype=x.dtype)
        return np.full((len(x),1),radius*scale)
    Unicycle_OF.__init__=init
    Unicycle_OF.get_disturbance=get_disturbance
    Unicycle_OF.measurement=measurement
    Unicycle_OF.cov_fn=cov_fn
    return Unicycle_OF
