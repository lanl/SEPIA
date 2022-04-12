from functools import cached_property
import numpy as np
from .bijectors import Log
from .AbstractDistribution import AbstractDistribution
from scipy.special import gammaln

def inversegamma_lpdf(x, shape, scale):
    """
    Inverse-Gamma log density function.

    shape: shape (> 0)
    scale: scale (> 0)

    Note that the mean under this parameterization is `scale / (shape - 1)`.

    Usage:
    >>> inversegamma_lpdf(3, shape=2, scale=1)
    """
    const = np.log(scale) * shape - gammaln(shape)

    return const + (-shape - 1) * np.log(x) - scale / x


class InverseGamma(AbstractDistribution):
    def __init__(self, shapeparam, scale):
        self.shape = np.broadcast(shapeparam, scale).shape
        self.shapeparam = shapeparam
        self.scale = scale

    @cached_property
    def params(self):
        return (self.shapeparam, self.scale)

    @cached_property
    def lower(self):
        return np.full(self.shape, 0)

    @cached_property
    def upper(self):
        return np.full(self.shape, np.inf)

    @cached_property
    def bijector(self):
        return Log(shape=self.shape)

    def in_support(self, x):
        return x > 0

    def _logpdf(self, x):
        return inversegamma_lpdf(x, shape=self.shapeparam, scale=self.scale),

    def sample(self, rng=np.random):
        """Return a random sample from the Inverse-Gamma distribution."""
        return 1 / rng.gamma(self.shapeparam, 1 / self.scale)
