"""Absolute, signed and threshold-tail diagnostics. No coverage guarantee."""
import numpy as np


def tail_summary(values, thresholds=(), unit="m/s"):
    value=np.asarray(values,dtype=np.float64).reshape(-1)
    if not len(value) or not np.isfinite(value).all(): raise ValueError("Need finite nonempty residuals")
    absolute=np.abs(value)
    tails={}
    for threshold in thresholds:
        if not np.isfinite(threshold) or threshold<0: raise ValueError("Invalid threshold")
        key="%.12g"%threshold
        above=absolute>threshold
        tails[key]=dict(count_above=int(above.sum()),denominator=len(value),fraction_above=float(above.mean()),
            positive_above=int((value>threshold).sum()),negative_below=int((value < -threshold).sum()),
            within_or_equal_count=int((~above).sum()))
    return dict(count=len(value),unit=unit,bias=float(value.mean()),mae=float(absolute.mean()),
        rmse=float(np.sqrt(np.mean(value**2))),p95_abs=float(np.quantile(absolute,.95)),
        p99_abs=float(np.quantile(absolute,.99)),max_abs=float(absolute.max()),
        signed_min=float(value.min()),signed_max=float(value.max()),thresholds=tails)


def trajectory_tail_summary(residuals,trajectory_index,trajectory_count,thresholds=()):
    value=np.asarray(residuals,dtype=np.float64)
    groups=np.asarray(trajectory_index,dtype=np.int64)
    if value.ndim!=1 or groups.shape!=value.shape or np.any(groups<0) or np.any(groups>=trajectory_count):
        raise ValueError("Need one signed residual and valid trajectory index per step")
    maxima=np.full(trajectory_count,np.nan)
    for group in np.unique(groups): maxima[group]=np.max(np.abs(value[groups==group]))
    keep=np.isfinite(maxima)
    if not keep.any(): raise ValueError("No trajectories checked")
    result=tail_summary(maxima[keep],thresholds)
    result.update(trajectories_checked=int(keep.sum()),trajectories_absent=int((~keep).sum()),
        score="maximum absolute per-step residual in each trajectory; empirical diagnostic, not new CP")
    return result,maxima


def align_rows(requested,available):
    available=np.asarray(available); requested=np.asarray(requested)
    if len(np.unique(available))!=len(available) or len(np.unique(requested))!=len(requested):
        raise ValueError("Duplicate feature rows")
    lookup={int(v):i for i,v in enumerate(available)}
    try: return np.array([lookup[int(v)] for v in requested],dtype=np.int64)
    except KeyError: raise ValueError("Required row missing from paired cache")
