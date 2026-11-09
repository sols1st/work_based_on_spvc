import json,math
import numpy as np
from scipy.optimize import linprog

CAL='results/lane_trajectory_cp_20261107_v1/calibration.npz'
HOLD='results/lane_trajectory_cp_20261107_v1/holdout.npz'
cal=np.load(CAL); hold=np.load(HOLD)
# split calibration: fit 99, CP-cal 100, holdout 100
fit=np.arange(0,99); qidx=np.arange(99,199)
states=cal['states'][:,:80]; errors=np.abs(cal['errors'])
res=np.maximum(errors[:,:,0],errors[:,:,1])

def b1(t,deg=2):
    t=np.asarray(t); out=[]
    for i in range(deg+1): out.append(math.comb(deg,i)*t**i*(1-t)**(deg-i))
    return np.stack(out,1)
def basis(e,t,deg=2):
    be=b1(e,deg); bt=b1(t,deg)
    return (be[:,:,None]*bt[:,None,:]).reshape(len(e),-1)

fit_e=states[fit,:,0].reshape(-1); fit_t=states[fit,:,1].reshape(-1); fit_r=res[fit].reshape(-1)
B=basis(fit_e,fit_t,2); n=len(fit_r); p=B.shape[1]
cost=np.r_[B.mean(0),np.full(n,20./n)]
A=np.column_stack([-B,-np.eye(n)])
sol=linprog(cost,A_ub=A,b_ub=-fit_r,bounds=[(0.01,None)]*p+[(0,None)]*n,method='highs')
print('fit',sol.success,sol.fun)
c=sol.x[:p]
def score(idx, data, states_all):
    e=states_all[idx,:,0]; t=states_all[idx,:,1]
    b=basis(e.reshape(-1),t.reshape(-1),2)@c; b=b.reshape(len(idx),-1)
    return np.max(res[idx]/b,axis=1), b
s_cal,b_cal=score(qidx,cal,states); s_hold,b_hold=score(np.arange(100),hold,hold['states'][:,:80])
q_state=np.quantile(s_cal,.95); q_const=np.quantile(np.max(np.abs(cal['errors'][qidx]).max(2),1),.95)
print('q_state',q_state,'q_const',q_const)
for name,s,q,b in [('state_cal',s_cal,q_state,b_cal),('state_hold',s_hold,q_state,b_hold)]:
    print(name,'coverage',np.mean(s<=q),'radius mean',q*b.mean(),'max',q*b.max())
print('const cal coverage',np.mean(np.max(np.abs(cal['errors'][qidx]).max(2),1)<=q_const),'hold',np.mean(np.max(np.abs(hold['errors']).max(2),1)<=q_const))
print('published .090093 hold coverage',np.mean(np.max(np.abs(hold['errors']).max(2),1)<=.0900931580))
print('coeff',c.tolist())
json.dump(dict(coeff=c.tolist(),q_state=float(q_state),q_const=float(q_const),
 state_cal_coverage=float(np.mean(s_cal<=q_state)),state_hold_coverage=float(np.mean(s_hold<=q_state)),
 state_cal_radius_mean=float(q_state*b_cal.mean()),state_hold_radius_mean=float(q_state*b_hold.mean()),
 state_cal_radius_max=float(q_state*b_cal.max()),state_hold_radius_max=float(q_state*b_hold.max()),
 const_cal_coverage=float(np.mean(np.max(np.abs(cal['errors'][qidx]).max(2),1)<=q_const)),
 const_hold_coverage=float(np.mean(np.max(np.abs(hold['errors']).max(2),1)<=q_const)),
 const_radius=float(q_const)),open('/tmp/lane_state_cp.json','w'),indent=2)
