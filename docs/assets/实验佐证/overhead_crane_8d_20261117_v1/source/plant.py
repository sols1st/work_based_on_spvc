"""Two horizontal crane axes with an acceleration-coupled suspended payload."""
import numpy as np

DT=.15
GRAVITY_OVER_LENGTH=4.
DAMPING=.16
HORIZON=40
INITIAL=(.70,.09,.13,.12)
CART_NOISE=.025
LOAD_NOISE=.04
AXIS_ACCEL=3.8
TOTAL_ACCEL=6.
# p, v, s, r = trolley position and speed, payload relative displacement and speed.
LIMITS={'trolley_position':1.45,'trolley_speed':1.8,'sway':.65,'sway_rate':1.5,
        'payload_position':1.55,'payload_diagonal':2.6}
TARGET={'payload_position':.09,'trolley_speed':.12,'sway':.055,'sway_rate':.12}
DWELL=4


def coefficients(dt, gravity, damping):
    # Semi-implicit Euler. Accepts rational as well as float coefficients.
    a=[[1,dt,0,0],[0,1,0,0],[0,0,1-dt*dt*gravity,dt*(1-dt*damping)],
       [0,0,-dt*gravity,1-dt*damping]]
    drive=[dt*dt,dt,-dt*dt,-dt]
    gust=[0,0,dt*dt,dt]
    return a,drive,gust


def matrices():
    return tuple(np.asarray(m,dtype=float) for m in coefficients(DT,GRAVITY_OVER_LENGTH,DAMPING))


def scenarios(n,seed):
    rng=np.random.default_rng(seed)
    starts=rng.uniform(-1,1,(n,2,4))*INITIAL
    starts[:,:,0]=rng.choice([-1,1],(n,2))*rng.uniform(.34,INITIAL[0],(n,2))
    # A repeatable bias and gust per axis, sampled independently across trajectories.
    span=np.array([CART_NOISE,LOAD_NOISE])
    noise=(rng.uniform(-.6,.6,(n,1,2,2))+rng.uniform(-.4,.4,(n,HORIZON,2,2)))*span
    return starts,noise,rng.integers(0,3,n)


def observation(state):
    return np.concatenate([state[...,0],state[...,2]/.5],axis=-1)


def control(z,state,gains):
    p,s=z[...,:2],.5*z[...,2:]
    k=np.asarray(gains)
    raw=-(k[0]*p+k[1]*state[...,1]+k[2]*s+k[3]*state[...,3])
    bounded=np.clip(raw,-AXIS_ACCEL,AXIS_ACCEL)
    return bounded/np.maximum(1,np.abs(bounded).sum(axis=-1,keepdims=True)/TOTAL_ACCEL)


def step(state,action,noise):
    a,b,g=matrices()
    return state@a.T + (action+noise[...,0])[...,None]*b + noise[...,1,None]*g


def metrics(states):
    p,v,s,r=np.moveaxis(states,-1,0)
    load=p+s
    checks={'trolley_position':p,'trolley_speed':v,'sway':s,'sway_rate':r,'payload_position':load,
            'payload_diagonal':load.sum(axis=-1,keepdims=True)}
    safe=np.logical_and.reduce([np.abs(checks[name]).max(axis=(1,2))<bound for name,bound in LIMITS.items()])
    dwell=np.logical_and.reduce([np.abs(checks[name][:,-DWELL:]).max(axis=(1,2))<=bound
                                for name,bound in TARGET.items()])
    dist=np.linalg.norm(load[:,-1],axis=1)
    return dict(n=len(states),safe=int(safe.sum()),settled=int((dwell&safe).sum()),
                mean_final_load_m=float(dist.mean()),
                mean_last_second_sway_m=float(np.linalg.norm(s[:,-7:],axis=-1).mean()))
