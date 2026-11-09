"""Finite-sample conformal tools and a positive state-dependent error scale."""
import math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta


def cp_quantile(scores, alpha=.05, confidence_failure=.01):
    values=np.asarray(scores,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('finite nonempty 1-D scores required')
    if not 0<alpha<1 or not 0<confidence_failure<1:
        raise ValueError('alpha and beta must lie in (0,1)')
    n=len(values); rank=math.ceil((n+1)*(1-alpha))
    q=float(np.partition(values,rank-1)[rank-1]) if rank<=n else float('inf')
    # With ties, this is a conservative lower tolerance bound, not an equality.
    coverage=float(beta.ppf(confidence_failure,rank,n+1-rank)) if rank<=n else 1.
    return dict(q_hat=q,rank=rank,n=n,alpha=alpha,marginal_target=1-alpha,
                confidence=1-confidence_failure,coverage_lower_at_confidence=coverage,
                guarantee_scope='one exchangeable draw, fixed score function; not trajectory safety')


def basis(distance,degree=3):
    d=np.asarray(distance,dtype=np.float64).reshape(-1)
    if not np.isfinite(d).all() or np.any((d<5-1e-5)|(d>16+1e-5)):
        raise ValueError('distance outside registered [5,16] m domain')
    t=np.clip((d-5)/11,0,1)
    return np.column_stack([math.comb(degree,k)*t**k*(1-t)**(degree-k) for k in range(degree+1)])


def fit_scale(distance,residual,degree=3,slack_penalty=20.,floor=.01):
    """VISION-SLS Eq.31 with gamma=0 and nonnegative Bernstein coefficients.

    Nonnegative basis and coefficients guarantee positivity over [5,16], not
    just at training points. Units here are metres, not normalized latent units.
    """
    m=basis(distance,degree); r=np.asarray(residual,dtype=np.float64)
    if r.shape!=(len(m),) or not np.isfinite(r).all() or (r<0).any() or floor<=0:
        raise ValueError('nonnegative paired residuals and positive floor required')
    n=len(r)
    cost=np.r_[m.mean(0),np.full(n,slack_penalty/n)]
    constraints=np.column_stack([-m,-np.eye(n)])
    fit=linprog(cost,A_ub=constraints,b_ub=-r,bounds=[(floor,None)]*m.shape[1]+[(0,None)]*n,method='highs')
    if not fit.success: raise RuntimeError(fit.message)
    return fit.x[:m.shape[1]],dict(objective=float(fit.fun),degree=degree,floor_m=floor,
                                  gamma=0.,slack_penalty=slack_penalty,solver='HiGHS',
                                  positivity='nonnegative Bernstein basis on [5,16]')


def envelope(distance,coefficients,q_hat=1.):
    coefficients=np.asarray(coefficients,dtype=np.float64)
    if coefficients.ndim!=1 or len(coefficients)<2 or not np.isfinite(coefficients).all() or (coefficients<=0).any():
        raise ValueError('positive finite Bernstein coefficients required')
    if not np.isfinite(q_hat) or q_hat<=0:
        raise ValueError('nonfinite/vacuous or nonpositive CP radius')
    return q_hat*(basis(distance,len(coefficients)-1)@coefficients)
