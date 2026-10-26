"""Advisor document's polynomial scale and multiplicative CP box."""
import itertools
import numpy as np

def features(states,scene):
    x=np.asarray(states,float)
    return x[:,:2] if scene=='parking' else x[:,[0,1,2,6,7]]

def powers(d,degree):
    answer=[(0,)*d]
    for order in range(1,degree+1):
        for indices in itertools.combinations_with_replacement(range(d),order):
            row=[0]*d
            for j in indices:row[j]+=1
            answer.append(tuple(row))
    return answer

def design(states,model):
    z=(features(states,model['scene'])-model['feature_mean'])/model['feature_scale']
    return np.prod(z[:,None,:]**np.asarray(model['powers'])[None,:,:],axis=2)

def envelope(states,model):return design(states,model)@np.asarray(model['coefficients'])
def positive_scale(states,model):return np.maximum(model['epsilon'],envelope(states,model))
def box_bound(states,model):return model['cp_multiplier']*positive_scale(states,model)
def ball_bound(states,model):return np.sqrt(model['ny'])*box_bound(states,model)
