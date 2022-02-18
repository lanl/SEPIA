from functools import cached_property
import numpy as np
from .bijectors import Logit
from .AbstractDistribution import AbstractDistribution

def uniform_lpdf(x, lower, upper):
    """
    Gamma log density function.

    lower: lower bound.
    upper: upper bound.

    Usage:
    >>> uniform_lpdf(3.5, lower=2, upper=4)
    """
    in_support = (x >= lower) & (x <= upper)
    return np.where(in_support, -np.log(upper - lower), np.NINF)


class Uniform(AbstractDistribution):
    def __init__(self, lower=0, upper=1):
        self.shape = np.broadcast(lower, upper).shape
        self.lower = lower
        self.upper = upper

    @cached_property
    def params(self):
        return (self.lower, self.upper)

    @cached_property
    def bijector(self):
        return Logit(lower=self.lower, upper=self.upper)

    def in_support(self, x):
        return (x > 0) & (x < 1)

    def _logpdf(self, x):
        return -np.log(self.upper - self.lower)

    def sample(self, rng=np.random):
        """Return a random sample from the uniform distribution."""
        return rng.uniform(self.lower, self.upper)
