import numpy as np
from numpy import math
from .bijectors import Logit

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

class Beta(SepiaDistribution):
    def __init__(self, a, b):
        self.shape = np.broadcast(a, b).shape
        self.a = np.full(self.shape, a)
        self.b = np.full(self.shape, b)
        self.lower = np.full(self.shape, 0)
        self.upper = np.full(self.shape, 1)
        self.params = (self.a, self.b)
        self.bijector = Logit(lower=self.lower, upper=self.upper)

    def logpdf(self, x):
        if self.in_support(x):
            return beta_lpdf(x, a=self.a, b=self.b)
        else:
            return np.NINF

    def sample(self, rng=np.random):
        """Return a random sample from the beta distribution."""
        return rng.beta(self.a, self.b)
