"""Scalar discrete CBF-QP, using image observation and measured speed only.

The feasible set is a braking interval because the structured NCBF is monotone
in speed. Bisection projects onto that interval to numerical tolerance, solving
the scalar minimum-intervention problem min (a-a_ref)^2.
Infeasibility is reported explicitly; maximum braking is NOT called certified.
"""
import torch
from torch import nn
import math


class CPNCBFFilter(nn.Module):
    def __init__(self,base_policy,barrier,coefficients,q_hat,decay=.05,numerical_margin=1e-5):
        super().__init__()
        self.base_policy=base_policy
        self.barrier=barrier
        self.register_buffer('C',torch.tensor([[1.,0.]]))
        self.register_buffer('coefficients',torch.as_tensor(coefficients,dtype=torch.float32))
        self.q_hat=float(q_hat);self.decay=float(decay)
        self.numerical_margin=float(numerical_margin)
        self.distance_scale_m=base_policy.distance_scale_m
        degree=len(coefficients)-1
        derivative_bound=self.q_hat*degree/11*max(abs(float(coefficients[i+1])-float(coefficients[i])) for i in range(degree))
        if not math.isfinite(self.q_hat) or self.q_hat<=0 or derivative_bound>=1:
            raise ValueError('monotone inverse requires finite CP radius and |b_prime|<1')
        self.last_info={}

    def radius(self,distance):
        t=((distance-5)/11).clamp(0,1); degree=len(self.coefficients)-1
        return self.q_hat*sum(self.coefficients[k]*math.comb(degree,k)*t**k*(1-t)**(degree-k) for k in range(degree+1))

    def distance_interval(self,observation_m):
        # d+b(d) and d-b(d) are strictly increasing by checked derivative bound.
        a=torch.full_like(observation_m,5.);b=torch.full_like(observation_m,16.)
        domain_ok=(observation_m>=a-self.radius(a))&(observation_m<=b+self.radius(b))
        for _ in range(28):
            middle=(a+b)/2; left=middle+self.radius(middle)<observation_m
            a=torch.where(left,middle,a);b=torch.where(left,b,middle)
        lower=a  # outward side of the bracket
        a=torch.full_like(observation_m,5.);b=torch.full_like(observation_m,16.)
        for _ in range(28):
            middle=(a+b)/2; left=middle-self.radius(middle)<observation_m
            a=torch.where(left,middle,a);b=torch.where(left,b,middle)
        return lower,b,domain_ok

    def constraint(self,distance,speed,action):
        state=torch.stack([distance,speed],-1)
        nxt=torch.stack([distance-.05*speed,(speed-.05*action).clamp(0,3)],-1)
        # No true-state goal termination is used by the deployed filter.
        return self.barrier(nxt)-(1-self.decay)*self.barrier(state)

    def required_action(self,distance,speed):
        lower=torch.full_like(speed,-3.);upper=torch.full_like(speed,3.)
        feasible=self.constraint(distance,speed,upper)>=self.numerical_margin
        for _ in range(28):
            middle=(lower+upper)/2
            ok=self.constraint(distance,speed,middle)>=self.numerical_margin
            lower=torch.where(ok,lower,middle);upper=torch.where(ok,middle,upper)
        return upper,feasible

    @torch.no_grad()
    def forward_observation(self,observation,speed):
        speed=speed.reshape(-1)
        estimate=observation.reshape(-1)*self.distance_scale_m
        distance_low,distance_high,domain_ok=self.distance_interval(estimate)
        required,feasible=self.required_action(distance_low,speed)
        nominal=self.base_policy.forward_observation(observation,speed).reshape(-1)
        action=torch.maximum(nominal,required).clamp(-3,3)
        self.last_info=dict(feasible=feasible,observation_in_contract_domain=domain_ok,
                            distance_low=distance_low,distance_high=distance_high,
                            intervention=(action-nominal).abs()>1e-5)
        return action[:,None]
