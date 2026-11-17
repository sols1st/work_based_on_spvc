"""Learned 3-D observation proxies, rate-limited controller and 5-D ReLU barrier."""
import torch
from torch import nn
from torch.nn import functional as F
from . import plant

class Proxy(nn.Module):
    def __init__(self,kind,n):
        super().__init__()
        self.kind=kind
        self.head=nn.Linear(n,3)
        self.C=nn.Parameter(torch.eye(3))
        self.bias=nn.Parameter(torch.zeros(3),requires_grad=(kind!='linear'))
        self.nets=nn.ModuleList([nn.Sequential(nn.Linear(2,12),nn.Tanh(),nn.Linear(12,1),nn.Tanh()) for _ in range(3)] if kind=='neural' else [])
        for net in self.nets:
            nn.init.zeros_(net[2].weight);nn.init.zeros_(net[2].bias)
    def matrix(self):
        return F.normalize(self.C,dim=1)
    def coupling(self,z,inverse):
        cols=list(z.split(1,dim=1))
        for i in (reversed(range(len(self.nets))) if inverse else range(len(self.nets))):
            others=torch.cat([cols[j] for j in range(3) if j!=i],dim=1)
            cols[i]=cols[i]+(-.25 if inverse else .25)*self.nets[i](others)
        return torch.cat(cols,dim=1)
    def nominal(self,x):
        return self.coupling(x@self.matrix().T+self.bias,False)
    def decode(self,y):
        return torch.linalg.solve(self.matrix(),(self.coupling(y,True)-self.bias).T).T

class Actor(nn.Module):
    """When observed rear-face station <=7.50 m, command at least 1.20 m/s² brake."""
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,obs,v,delta,k):
        inputs=torch.cat([obs,(v/2.6)[:,None],(delta/.07)[:,None],(k/.02)[:,None]],1)
        logits=self.net(inputs)
        measured_d=4+2*obs[:,2]
        floor=(1.20-.8*F.relu(measured_d-7.50)).clamp(min=-.4)
        brake=floor+(3.5-floor)*logits[:,0].sigmoid()
        return torch.stack([-brake,.16*logits[:,1].tanh()],dim=1)

class Barrier(nn.Module):
    """Structured nonnegative ReLU network. Fixed hinge bank encodes braking reserve;
    fitted positive feature weights multiply five-state inputs; max is ReLU-realizable.
    """
    def __init__(self):
        super().__init__()
        self.free=nn.Parameter(torch.zeros(4))
        self.register_buffer('knots',torch.arange(12,dtype=torch.float64)*.24)
    def coefficients(self):
        return self.free.new_tensor([.195,.054,.025,1.018])+self.free.sigmoid()*self.free.new_tensor([.02,.012,.045,.035])
    def reserve(self,v):
        return plant.DT*F.relu(v[:,None]-self.knots.to(v)).sum(1)
    def forward(self,x):
        e,psi,d,v,delta=x.T
        ae,at,beta,ad=self.coefficients();reserve=self.reserve(v)
        facets=torch.stack([(e.abs()+ae*reserve-.65)/.028,
                 (psi.abs()+beta*delta.abs()+at*reserve-.19)/.014,
                 (1.92+ad*reserve-d)/.18,(d-6.15)/.15],1)
        return 20*F.relu(1+facets.amax(1))
