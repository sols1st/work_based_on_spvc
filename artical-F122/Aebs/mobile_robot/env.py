"""Goal-reaching task. Reward potential is offline training shaping only."""
import heapq
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from . import core


class DistancePotential:
    def __init__(self):
        self.step=.1;self.dist=np.full((61,41),np.inf)
        xs,ys=np.meshgrid(np.arange(61)*self.step,np.arange(41)*self.step,indexing='ij')
        free=((xs>=.35)&(xs<=5.65)&(ys>=.35)&(ys<=3.65)&
              ((xs-3)**2+(ys-2)**2> .95**2))
        self.dist[50,20]=0;queue=[(0.,50,20)]
        while queue:
            value,i,j=heapq.heappop(queue)
            if value!=self.dist[i,j]:continue
            for di,dj in [(a,b) for a in [-1,0,1] for b in [-1,0,1] if a or b]:
                ni,nj=i+di,j+dj
                if 0<=ni<61 and 0<=nj<41 and free[ni,nj]:
                    new=value+self.step*np.hypot(di,dj)
                    if new<self.dist[ni,nj]:self.dist[ni,nj]=new;heapq.heappush(queue,(new,ni,nj))
        # Smooth, finite values near obstacle/border for reward only.
        from scipy.ndimage import distance_transform_edt
        bad=~np.isfinite(self.dist)
        _,idx=distance_transform_edt(bad,return_indices=True)
        self.dist[bad]=self.dist[tuple(idx[:,bad])]+.2

    def __call__(self,state):
        x,y=np.clip(np.asarray(state)[:2]/self.step,[0,0],[59.999,39.999])
        i,j=int(x),int(y);a,b=x-i,y-j
        return ((1-a)*(1-b)*self.dist[i,j]+a*(1-b)*self.dist[i+1,j]
                +(1-a)*b*self.dist[i,j+1]+a*b*self.dist[i+1,j+1])


POTENTIAL=DistancePotential()


class RobotEnv(gym.Env):
    def __init__(self,seed=0,radius=0.,curriculum=False):
        self.observation_space=spaces.Box(-2.,2.,(3,),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,(2,),dtype=np.float32)
        self.rng=np.random.default_rng(seed);self.radius=radius;self.curriculum=curriculum
        self.steps=0;self.state=core.INITIAL_LOW.copy();self.previous_action=np.zeros(2)

    def observation(self):
        return (core.normalize(self.state)+self.rng.uniform(-self.radius,self.radius,3)).astype('float32')

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.rng=np.random.default_rng(seed)
        self.state=core.sample_initial(self.rng,1)[0]
        if self.curriculum and self.rng.random()<.5:
            # Training-only broad reset; final evaluation retains original initial box.
            while True:
                x=self.rng.uniform([.5,.5,-.5],[5.4,3.5,.5])
                if not core.unsafe(x) and np.linalg.norm(x[:2]-core.OBSTACLE)>.95:break
            self.state=x
        self.steps=0;self.previous_action=np.array([-1.,0.])
        return self.observation(),{}

    def step(self,action):
        action=np.clip(np.asarray(action).reshape(2),-1,1)
        old=self.state.copy();self.state=core.dynamics(old,action);self.steps+=1
        bad=bool(core.unsafe(self.state));at_goal=bool(core.goal(self.state))
        v=(action[0]+1)*core.V_MAX/2
        reward=15*(POTENTIAL(old)-POTENTIAL(self.state))-.03
        reward-=.01*action[1]**2+.01*np.square(action-self.previous_action).sum()
        if at_goal:reward+=1.-2*v-.05*abs(self.state[2])
        if bad:reward-=20
        self.previous_action=action
        return self.observation(),float(reward),bad,self.steps>=core.HORIZON,dict(unsafe=bad,at_goal=at_goal)
