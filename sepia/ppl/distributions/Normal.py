from functools import cached_property
import numpy as np
from .bijectors import Identity
from .AbstractDistribution import AbstractDistribution

def normal_lpdf(x, loc=0, scale=1):
    """
    Gaussian log density function.

    loc: location (defulats to 0).
    scale: scale (> 0, defulats to 1).

    Usage:
    >>> normal_lpdf(3, loc=1, scale=2)
    """
    z = (x - loc) / scale
    const = -0.5 * np.log(2 * np.pi) - np.log(scale)
    return -(z * z) / 2 + const


class Normal(AbstractDistribution):
    def __init__(self, loc, scale):
        self.shape = np.broadcast(loc, scale).shape
        self.loc = loc
        self.scale = scale

    @cached_property
    def params(self):
        return (self.loc, self.scale)

    @cached_property
    def lower(self):
        return np.full(self.shape, np.NINF)

    @cached_property
    def upper(self):
        return np.full(self.shape, np.inf)

    @cached_property
    def bijector(self):
        return Identity(shape=self.shape)

    def in_support(self, x):
        return np.full(np.shape(x), True)

    def _logpdf(self, x):
        return normal_lpdf(x, loc=self.loc, scale=self.scale)

    def sample(self, rng=np.random):
        """Return a random sample from the Normal distribution."""
        return rng.normal(self.loc, self.scale)
