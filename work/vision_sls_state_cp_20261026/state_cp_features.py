"""New state features and error predictor, independent of the old envelope."""
import numpy as np

def features(states,scene):
    states=np.asarray(states,dtype=float)
    if scene=='parking':
        return np.column_stack((states[:,0],states[:,1],np.sin(states[:,2]),np.cos(states[:,2])))
    return states[:,[0,1,2,6,7]]

def design(states,model):
    z=(features(states,model['scene'])-model['feature_mean'])/model['feature_scale']
    d=z.shape[1]
    return np.column_stack([np.ones(len(z))]+[z[:,j] for j in range(d)]+
                           [z[:,i]*z[:,j] for i in range(d) for j in range(i,d)])

def predict(states,model):
    return model['target_scale']*(design(states,model)@np.asarray(model['coefficients']))

def bound(states,model):
    return np.maximum(model['floor'],predict(states,model)+model['cp_offset'])
