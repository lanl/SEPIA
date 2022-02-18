from functools import cached_property
import numpy as np
from numpy import math
from .bijectors import Log
from .AbstractDistribution import AbstractDistribution

def gamma_lpdf(x, shape, scale):
    """
    Gamma log density function.

    shape: shape (> 0)
    scale: scale (> 0)

    Note that the mean under this parameterization is `shape * scale`.

    Usage:
    >>> gamma_lpdf(3, shape=2, scale=1)
    """
    const = -math.lgamma(shape) - shape * np.log(scale)
    return const + (shape - 1) * np.log(x) - x / scale

class Gamma(AbstractDistribution):
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
        return gamma_lpdf(x, shape=self.shapeparam, scale=self.scale)

    def sample(self, rng=np.random):
        """Return a random sample from the Gamma distribution."""
        return rng.gamma(self.shapeparam, self.scale)
