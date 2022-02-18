from abc import ABC, abstractmethod
import numpy as np

class AbstractDistribution(ABC):
    """
    Abstract class for univariate continuous distributions.
    """
    # NOTE:
    # requires properties:
    # - self.lower
    # - self.upper
    # - self.bijector

    @abstractmethod
    def _logpdf(self, x):
        """Evaluate and return the log density at `x`."""
        pass

    def logpdf(self, x):
        return np.where(
            self.in_support(x),
            self._logpdf(x),
            np.NINF
        )

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def support(self):
        """Return the support."""
        return (self.lower(), self.upper())

    def in_support(self, x):
        """Return `True` if `x` is within the support. Otherwise, return `False`."""
        return (x > self.lower) & (x < self.upper)

    @abstractmethod
    def sample(self, rng=np.random):
        """Return a random sample from the distribution."""
        pass

    # NOTE: I think the following methods could be helpful if we consider using a multivariate
    # Gaussian proposal for parameters in their unconstrained space. This would involve/require
    # transforming priors so that they are unconstrained. (e.g. If X ~ Gamma(a, b), then propose a
    # new value for log(x), and add the appropriate log absolute Jacobian.)
    def to_unconstrained_space(self, constrained_x):
        """
        Transform `constrained_x` to unconstrained space.
        """
        return self.bijector.transform(constrained_x)

    def to_constrained_space(self, unconstrained_x):
        """Transform `unconstrained_x` with support on the real line / ball into its support."""
        return self.bijector.inv_transform(unconstrained_x)

    def log_abs_det_jacobian(self, unconstrained_x):
        """
        Compute the log absolute determinant of the jacobian given the parameter's
        constrained and unconstrained values.
        """
        return self.bijector.logdetjac(unconstrained_x)

    def logpdf_plus_log_abs_det_jacobian(self, unconstrained_x, constrained_x=None):
        """
        Compute the log density plus the log absolute value of the determinant of the jacobian
        given a parameter's constrained and unconstrained values.
        """
        if constrained_x is None:
            constrained_x = self.to_constrained_space(unconstrained_x)
        return self._logpdf(constrained_x) + self.log_abs_det_jacobian(unconstrained_x)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)
