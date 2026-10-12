"""KM plug-in RMST and fixed-time survival inference."""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
Z=norm.ppf(.975)
def wald(value, variance):

    if not np.isfinite([value, variance]).all() or variance < 0:

        return dict(estimate=value, se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)

    se = math.sqrt(variance)

    return dict(estimate=value, se=se, ci_low=value - Z * se, ci_high=value + Z * se, p=float(2 * norm.sf(abs(value / se))) if se else np.nan)

def km_summary(time, event, tau):

    """Exact KM step integral and its Greenwood plug-in variance.



    Var(RMST) = sum_j [integral(t_j,tau) S(u)du]^2 d_j/[Y_j(Y_j-d_j)].

    A terminal all-fail event has zero post-event tail area and contributes 0.

    Beyond last follow-up with positive survival, return UNESTIMABLE, not a

    last-value extrapolation. Same-time events occur before censor removal.

    """

    (t, e) = (np.asarray(time, float), np.asarray(event, int))

    if len(t) == 0 or not np.isfinite(t).all() or (t < 0).any() or (not np.isin(e, [0, 1]).all()):

        raise ValueError('Invalid survival input')

    order = np.argsort(t, kind='stable')

    (t, e) = (t[order], e[order])

    (u, start, counts) = np.unique(t, return_index=True, return_counts=True)

    deaths = np.add.reduceat(e, start)

    risk = len(t) - start

    mask = (deaths > 0) & (u <= tau)

    (et, dd, yy) = (u[mask], deaths[mask], risk[mask])

    survival = np.cumprod(1.0 - dd / yy)

    end_s = float(survival[-1]) if len(survival) else 1.0

    meta = dict(n=len(t), events=int(e.sum()), at_risk=int((t >= tau).sum()), events_by_tau=int(((t <= tau) & (e == 1)).sum()), censored_before_tau=int(((t < tau) & (e == 0)).sum()), max_followup=float(t.max()), status='SUCCESS', reason='')

    if t.max() < tau and end_s > 0:

        return dict(**{**meta, 'status': 'UNESTIMABLE', 'reason': 'Positive KM tail ends before horizon; no extrapolation'}, survival=np.nan, survival_var=np.nan, rmst=np.nan, rmst_var=np.nan)

    dt = np.diff(np.r_[0.0, et, tau])

    segment_area = np.r_[1.0, survival] * dt

    rmst = float(segment_area.sum())

    tail = np.cumsum(segment_area[:0:-1])[::-1]

    green = np.divide(dd, yy * (yy - dd), out=np.zeros(len(dd), float), where=yy > dd)

    vr = float(np.dot(tail * tail, green))

    vs = float(end_s * end_s * green.sum()) if end_s else 0.0

    return dict(**meta, survival=end_s, survival_var=vs, rmst=rmst, rmst_var=vr)
