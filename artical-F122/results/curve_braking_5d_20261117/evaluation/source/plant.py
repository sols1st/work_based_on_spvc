"""Five-state discrete Frenet bicycle. State (e, psi, d, v, delta), m/rad/s."""
import numpy as np

DT = .2
L = 2.6
VMAX = 2.6
DMAX = .07
KAPPAS = np.array([-.02, .02])
HALF_WIDTH, HALF_LENGTH, ROAD_HALF = .70, 1.60, 1.75
INIT_LO = np.array([-.08,-.028,5.30,2.10,-.035])
INIT_HI = np.array([ .08, .028,5.90,2.40, .035])
SCALE = np.array([.65,.19,2.])
CENTER = np.array([0.,0.,4.])
HORIZON = 14
TARGET = (3.60,4.40)  # final centre station distance to obstacle rear

def normalize(s):
    return (s[..., :3]-CENTER)/SCALE

def step(x,k,actions):
    """Rate commands: longitudinal acceleration and steering change per metre."""
    e,psi,d,v,delta=x.T
    accel,steer=actions.T
    parked=v==0
    speed=np.clip(v+DT*accel,0,VMAX)
    ds=DT*speed
    advance=ds*np.cos(psi)/(1-k*e)
    new=np.column_stack((e+ds*np.sin(psi), psi+ds*np.tan(delta)/L-k*advance,
                         d-advance,speed,np.clip(delta+ds*steer,-DMAX,DMAX)))
    return np.where(parked[:,None],x,new)

def margins(x,k):
    """Conservative body and rear-face separating clearances (positive = safe).
    The signed-distance Hessian for a circle has norm <= k/(1-k*offset).
    The Frenet station gradient has norm <=1/(1-k*offset).
    """
    rho=np.hypot(HALF_LENGTH,HALF_WIDTH)
    offset=np.abs(x[...,0])+rho
    den=1-np.abs(k)*offset
    lat=(ROAD_HALF-np.abs(x[...,0])-HALF_WIDTH-HALF_LENGTH*np.abs(np.sin(x[...,1]))
         -.5*np.abs(k)*rho**2/den)
    gap=x[...,2]-rho/den
    return np.where(den>0,lat,-np.inf),np.where(den>0,gap,-np.inf)
