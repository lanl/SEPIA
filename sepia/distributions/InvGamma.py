import numpy as np
from numpy import math
from .bijectors import Log

def invgamma_lpdf(x, shape, scale):
    """
    Inverse-Gamma log density function.

    shape: shape (> 0)
    scale: scale (> 0)

    Note that the mean under this parameterization is `scale / (shape - 1)`.

    Usage:
    >>> invgamma_lpdf(3, shape=2, scale=1)
    """
    const = np.log(scale) * shape - math.lgamma(shape)
    return const + (-shape - 1) * np.log(x) - scale / x


class InvGamma(SepiaDistribution):
    def __init__(self, shapeparam, scale):
        self.shape = np.broadcast(shapeparam, scale).shape
        self.lower = np.full(self.shape, 0)
        self.upper = np.full(self.shape, np.inf)
        self.params = (self.shapeparam, self.scale)
        self.shapeparam = np.full(self.shape, shapeparam)
        self.scale = np.full(self.shape, scale)
        self.bijector = Log(shape=self.shape)

    def in_support(self, x):
        out_of_support = np.any(x < 0)
        return not out_of_support

    def logpdf(self, x):
        if self.in_support(x):
            return invgamma_lpdf(x, shape=self.shapeparam, scale=self.scale)
        else:
            return -np.NINF

    def sample(self, rng=np.random):
        """Return a random sample from the uniform distribution."""
        return 1 / rng.gamma(self.shapeparam, 1 / self.scale)
