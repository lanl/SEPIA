import numpy as np
from .bijectors import Identity

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


class Normal(SepiaDistribution):
    def __init__(self, loc, scale):
        self.shape = np.broadcast(loc, scale).shape
        self.lower = np.full(self.shape, np.NINF)
        self.upper = np.full(self.shape, np.inf)
        self.params = (self.loc, self.scale)
        self.loc = np.full(self.shape, loc)
        self.scale = np.full(self.shape, scale)
        self.bijector = Identity(shape=self.shape)

    def in_support(self, x):
        return True

    def logpdf(self, x):
        return normal_lpdf(x, loc=self.loc, scale=self.scale)

    def sample(self, rng=np.random):
        """Return a random sample from the Normal distribution."""
        return rng.normal(self.loc, self.scale)
