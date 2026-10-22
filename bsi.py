"""Partition-conditional Boltzmann--Shannon Index; base-two JSD.

Covariance extents are sqrt(det(sample covariance)), NOT raw SVD products.
No automatic truncation, regularization, or silent removal of zero cells.
"""
import numpy as np
from scipy.special import rel_entr, logsumexp
from scipy.spatial import ConvexHull


def distributions_bsi(p, q):
    """Return 1-JSD_2 for normalized, finite, nonnegative vectors.

    Zero probability cells contribute zero to their own KL term. A cell where
    exactly one distribution is positive must still contribute to the other.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.ndim != 1 or p.size == 0 or p.shape != q.shape:
        raise ValueError('p and q must be nonempty vectors of equal length')
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError('Probabilities must be finite')
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError('Probabilities must be nonnegative')
    if not np.isclose(p.sum(), 1, atol=1e-12, rtol=1e-10) or not np.isclose(q.sum(), 1, atol=1e-12, rtol=1e-10):
        raise ValueError('Probabilities must sum to one')
    m = (p + q) / 2
    divergence = (rel_entr(p, m).sum() + rel_entr(q, m).sum()) / (2*np.log(2))
    return float(np.clip(1-divergence, 0, 1))


def from_extents(counts, extents):
    """Evaluate supplied nonnegative counts and extents on a common label set."""
    counts, extents = np.asarray(counts, float), np.asarray(extents, float)
    if counts.ndim != 1 or counts.size == 0 or counts.shape != extents.shape:
        raise ValueError('Counts and extents must be nonempty matching vectors')
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(extents)) or np.any(counts < 0) or np.any(extents < 0):
        raise ValueError('Counts and extents must be finite and nonnegative')
    if counts.sum() <= 0 or extents.sum() <= 0:
        raise ValueError('Both totals must be positive')
    p, q = counts/counts.sum(), extents/extents.sum()
    return {'bsi': distributions_bsi(p,q), 'p': p, 'q': q}


def labeled_bsi(X, labels, extent='covariance'):
    """Estimate extents for nonempty labels in a common ambient dimension.

    Covariance mode rejects fewer than d+1 samples or numerical rank deficiency.
    Hull mode is a sensitivity analysis, not a general support-volume estimator.
    """
    X, labels = np.asarray(X, float), np.asarray(labels)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or labels.ndim != 1 or len(labels) != len(X):
        raise ValueError('X must be an N by d matrix with N labels')
    if not np.all(np.isfinite(X)):
        raise ValueError('X must contain finite observations')
    if extent not in ('covariance', 'hull'):
        raise ValueError('Unknown extent estimator')
    states, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    d = X.shape[1]
    logs = []
    for k, n in enumerate(counts):
        if n <= d:
            raise ValueError(f'State {states[k]} needs at least d+1 observations')
        Z = X[inverse == k] - X[inverse == k].mean(axis=0)
        s = np.linalg.svd(Z, compute_uv=False)
        if s[-1] <= np.finfo(float).eps * max(Z.shape)*s[0]:
            raise ValueError(f'State {states[k]} is numerically rank deficient')
        if extent == 'covariance':
            logs.append(np.log(s).sum() - d/2*np.log(n-1))
        else:
            volume = np.ptp(Z[:,0]) if d == 1 else ConvexHull(Z).volume
            logs.append(np.log(volume))
    logs = np.asarray(logs)
    p, q = counts/counts.sum(), np.exp(logs-logsumexp(logs))
    return {'bsi': distributions_bsi(p,q), 'p':p, 'q':q, 'labels':states,
            'counts':counts, 'log_extents':logs, 'estimator':extent}
