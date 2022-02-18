from functools import cached_property
import numpy as np
from numpy import math
from .bijectors import Logit
from .AbstractDistribution import AbstractDistribution

def beta_lpdf(x, a, b):
    """
    Beta log density function.
    a: first concentration parameter (>0).
    b: second concentration parameter (>0).
    Usage:
    >>> beta_lpdf(0.6, a=3, b=7)
    """
    const = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return const + (a - 1) * np.log(x) + (b - 1) * np.log1p(-x)

class Beta(AbstractDistribution):
    def __init__(self, a, b):
        self.shape = np.broadcast(a, b).shape
        self.a = a
        self.b = b

    @cached_property
    def params(self):
        return (self.a, self.b)

    @cached_property
    def lower(self):
        return np.full(self.shape, 0)

    @cached_property
    def upper(self):
        return np.full(self.shape, 1)

    @cached_property
    def bijector(self):
        return Logit(lower=self.lower, upper=self.upper)

    def in_support(self, x):
        return (x > 0) & (x < 1)

    def _logpdf(self, x):
        return beta_lpdf(x, a=self.a, b=self.b)

    def sample(self, rng=np.random):
        """Return a random sample from the beta distribution."""
        return rng.beta(self.a, self.b)
