from abc import ABC, abstractmethod
import numpy as np

def expit(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    return np.log(p) - np.log1p(-p)

class Bijector(ABC):
    def __init__(self, shape=None):
        if shape is None:
            self.shape = ()

    @abstractmethod
    def transform(self, x):
        # This converts the parameter into the real space.
        pass

    @abstractmethod
    def inv_transform(self, x):
        # This converts the parameter from the real space.
        pass

    @abstractmethod
    def logdetjac(self, real_x):
        pass

    def logpdf_plus_logdetjac(self, logpdf, real_x):
        """
        logpdf plus the log absolute determinant of the jacobian, evaluated at
        parameter on the real space.
        """
        x = self.inv_transform(real_x)
        return logpdf(x) + self.logdetjac(real_x)


class Identity(Bijector):
    def __init__(self, shape=None):
        super().__init__(shape=shape)

    def transform(self, x):
        return x

    def inv_transform(self, x):
        return x

    def logdetjac(self, real_x):
        return np.zeros_like(real_x)


class Log(Bijector):
    def __init__(self, shape=None):
        super().__init__(shape=shape)

    def transform(self, x):
        return np.log(x)

    def inv_transform(self, x):
        return np.exp(x)

    def logdetjac(self, real_x):
        return real_x


class Logit(Bijector):
    def __init__(self, lower, upper):
        self.shape = np.broadcast(lower, upper).shape
        self.lower = np.full(self.shape, lower)
        self.upper = np.full(self.shape, upper)

    def transform(self, x):
        u = (x - self.lower) / (self.upper - self.lower)
        return logit(u)

    def inv_transform(self, x):
        return expit(x) * (self.upper - self.lower) + self.lower

    def logdetjac(self, real_x):
        return real_x - 2 * np.log1p(np.exp(real_x))
