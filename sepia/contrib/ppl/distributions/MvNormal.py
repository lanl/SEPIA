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

class MvNormal:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        assert self.mean.shape[0] == self.cov.shape[0] == self.cov.shape[1]

    def logpdf(self, x):
        return mvnormal_lpdf(x, mean=self.mean, cov=self.cov)

    def sample(self, rng=np.random):
        return rng.multivariate_normal(mean=self.mean, cov=self.cov)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)
