"""Record QP curvature; optionally convexify only the local quadratic surrogate."""
import json
from pathlib import Path
import numpy as np
import casadi as ca

def install_qp_audit(path:Path,convexify=False):
    from solver.qp import QP
    original=QP.solve
    records=[]
    def solve(self,*args,**kwargs):
        H=np.asarray(2*self.H_mat+self.q_cost_quad,dtype=float)
        H=(H+H.T)/2
        values,vectors=np.linalg.eigh(H)
        record={'qp_index':len(records),'min_eigenvalue':float(values[0]),
                'max_eigenvalue':float(values[-1]),'convexification_applied':False}
        old_quad=self.q_cost_quad
        if convexify and values[0]<-1e-7:
            fixed=(vectors*np.maximum(values,1e-6))@vectors.T
            self.q_cost_quad=ca.DM(fixed-2*np.asarray(self.H_mat))
            record.update(convexification_applied=True,
                          hessian_change_frobenius=float(np.linalg.norm(fixed-H)))
        try:
            result=original(self,*args,**kwargs)
            record['solver_success']=bool(result.get('success',False))
            try:record['solver_status']=str(self.solver_qp.stats().get('return_status'))
            except Exception:pass
            return result
        finally:
            self.q_cost_quad=old_quad
            records.append(record)
            path.write_text(json.dumps({'convexify':convexify,'records':records},indent=2)+'\n')
    QP.solve=solve
