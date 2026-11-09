"""Registered discrete unicycle task; no hidden planner or action filter."""
import numpy as np
import torch

DT=.1
HORIZON=120
SCALE=np.array([3.,2.,1.2])
CENTER=np.array([3.,2.,0.])
LOW=np.array([.35,.35,-1.2])
HIGH=np.array([5.65,3.65,1.2])
INITIAL_LOW=np.array([.8,1.8,-.1])
INITIAL_HIGH=np.array([1.2,2.2,.1])
GOAL=np.array([5.,2.])
GOAL_RADIUS=.3
OBSTACLE=np.array([3.,2.])
OBSTACLE_RADIUS=.4
ROBOT_RADIUS=.25
SAFETY_MARGIN=.1
EXCLUSION_RADIUS=OBSTACLE_RADIUS+ROBOT_RADIUS+SAFETY_MARGIN
V_MAX=.6
OMEGA_MAX=1.


def normalize(x):return (np.asarray(x)-CENTER)/SCALE
def physical(x):return np.asarray(x)*SCALE+CENTER


def dynamics(states,actions):
    x=np.asarray(states,dtype=np.float64)
    a=np.clip(np.asarray(actions,dtype=np.float64),-1,1)
    v=(a[...,0]+1)*V_MAX/2;w=a[...,1]*OMEGA_MAX
    return x+DT*np.stack([v*np.cos(x[...,2]),v*np.sin(x[...,2]),w],axis=-1)


def torch_dynamics(states,actions):
    scale=states.new_tensor(SCALE);center=states.new_tensor(CENTER)
    x=states*scale+center;a=actions.clamp(-1,1)
    v=(a[...,0]+1)*V_MAX/2;w=a[...,1]*OMEGA_MAX
    delta=torch.stack([v*x[...,2].cos(),v*x[...,2].sin(),w],-1)*DT
    return states+delta/scale


def unsafe(states):
    x=np.asarray(states)
    return ((x<LOW).any(axis=-1)|(x>HIGH).any(axis=-1)|
            (np.linalg.norm(x[...,:2]-OBSTACLE,axis=-1)<=EXCLUSION_RADIUS))


def goal(states):
    return np.linalg.norm(np.asarray(states)[...,:2]-GOAL,axis=-1)<=GOAL_RADIUS


def sample_initial(rng,n):return rng.uniform(INITIAL_LOW,INITIAL_HIGH,(n,3))


def config():
    return dict(state=['x','y','theta'],visual_observation_dimension=3,action=['linear_velocity','angular_velocity'],
        dt=DT,horizon=HORIZON,scale=SCALE.tolist(),center=CENTER.tolist(),
        safe_low=LOW.tolist(),safe_high=HIGH.tolist(),initial_low=INITIAL_LOW.tolist(),
        initial_high=INITIAL_HIGH.tolist(),goal=GOAL.tolist(),goal_radius=GOAL_RADIUS,
        obstacle=OBSTACLE.tolist(),obstacle_radius=OBSTACLE_RADIUS,robot_radius=ROBOT_RADIUS,
        safety_margin=SAFETY_MARGIN,exclusion_radius=EXCLUSION_RADIUS,
        v_max=V_MAX,omega_max=OMEGA_MAX,process_noise=0,
        safety_scope='discrete sample instants; heading operating range is part of safety; velocity commands assumed executed',
        online_filter=False,goal_absorbing=False)
