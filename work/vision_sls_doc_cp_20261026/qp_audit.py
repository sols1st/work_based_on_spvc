"""Record QP curvature; optionally convexify only the local quadratic surrogate."""
import json
from pathlib import Path
import numpy as np
import casadi as ca

def install_qp_audit(path:Path,convexify=False,relative_floor=1e-8,normalize_objective=False):
    from solver.qp import QP
    original=QP.solve
    records=[]
    def solve(self,*args,**kwargs):
        H=np.asarray(2*self.H_mat+self.q_cost_quad,dtype=float)
        H=(H+H.T)/2
        values,vectors=np.linalg.eigh(H)
        record={'qp_index':len(records),'min_eigenvalue':float(values[0]),
                'max_eigenvalue':float(values[-1]),'convexification_applied':False}
        old_quad=self.q_cost_quad;old_H=self.H_mat;old_linear=self.q_cost_lin
        if convexify and values[0]<-1e-7:
            eigenvalue_floor=max(1e-6,relative_floor*float(np.max(np.abs(values))))
            fixed=(vectors*np.maximum(values,eigenvalue_floor))@vectors.T
            self.q_cost_quad=ca.DM(fixed-2*np.asarray(self.H_mat))
            record.update(convexification_applied=True,
                          eigenvalue_floor=eigenvalue_floor,
                          hessian_change_frobenius=float(np.linalg.norm(fixed-H)))
        objective_scale=1.
        if normalize_objective:
            effective=np.asarray(2*self.H_mat+self.q_cost_quad,dtype=float)
            objective_scale=max(1.,float(np.max(np.abs(effective))))
            self.H_mat=self.H_mat/objective_scale
            self.q_cost_quad=self.q_cost_quad/objective_scale
            self.q_cost_lin=self.q_cost_lin/objective_scale
        record['objective_normalization_factor']=objective_scale
        try:
            result=original(self,*args,**kwargs)
            if result.get('success',False) and normalize_objective:
                for key in ['dual_vec','dual_mu','dual_mu_f','cost']:
                    if key in result:result[key]=result[key]*objective_scale
            record['solver_success']=bool(result.get('success',False))
            try:record['solver_status']=str(self.solver_qp.stats().get('return_status'))
            except Exception:pass
            return result
        finally:
            self.q_cost_quad=old_quad;self.H_mat=old_H;self.q_cost_lin=old_linear
            records.append(record)
            path.write_text(json.dumps({'convexify':convexify,'records':records},indent=2)+'\n')
    QP.solve=solve
