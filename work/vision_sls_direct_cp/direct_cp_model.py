"""Replace only the quadrotor observation-error model by a direct CP ball.

No polynomial coefficients, original bias, or old error envelope are loaded.
Dynamics, constraints and learned C are retained from the original source.
"""
import ast
from pathlib import Path
import numpy as np
import casadi as ca

def install_direct_cp(source: Path, radius: float):
    assert np.isfinite(radius) and radius>0
    from dyn.quad_10d import Quad10d
    from dyn.quad_10d_of_perception import Quad10d_OF
    tree=ast.parse((source/'dyn/quad_10d_of_perception.py').read_text())
    candidates=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and
                    t.value.id=='self' and t.attr=='C' for t in n.targets)]
    assert len(candidates)==1
    call=candidates[0].value
    assert isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='array'
    C=np.array(ast.literal_eval(call.args[0]),dtype=float)
    assert C.shape==(5,10)
    def init(self):
        Quad10d.__init__(self)
        self.ny=self.nv=5
        self.C=C.copy()
        self.F=radius*np.eye(5)
        self.use_perception_model=True
        self.direct_cp_radius=radius
    def get_disturbance(self,x,scale=1.):
        return np.full(5,radius*scale)
    def measurement(self,x,v):
        return self.C@x+self.F@v
    Quad10d_OF.__init__=init
    Quad10d_OF.get_disturbance=get_disturbance
    Quad10d_OF.measurement=measurement
    return Quad10d_OF
