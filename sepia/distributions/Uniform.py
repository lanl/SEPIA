import numpy as np
from .bijectors import Logit

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


class Uniform(SepiaDistribution):
    def __init__(self, lower=0, upper=1):
        self.shape = np.broadcast(lower, upper).shape
        self.lower = np.full(self.shape, lower)
        self.upper = np.full(self.shape, upper)
        self.params = (self.lower, self.upper)
        self.bijector = Logit(lower=self.lower, upper=self.upper)

    def logpdf(self, x):
        if self.in_support(x):
            return -np.log(upper - lower)
        else:
            return -np.NINF

    def sample(self, rng=np.random):
        """Return a random sample from the uniform distribution."""
        return rng.uniform(self.lower, self.upper)
