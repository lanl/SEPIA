import numpy as np

def mvnormal_lpdf(x, mean, cov):
    """
    Multivariate Normal log density function.

    mean: mean.
    cov: covariance.

    Usage:
    >>> mvnormal_lpdf(np.random.randn(4), np.zeros(4), np.eye(4))
    """
    n = cov.shape[0]
    d = x - mean
    lpdf = -0.5 * (d @ np.linalg.inv(cov) @ d)
    lpdf += -0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1])
    return lpdf
