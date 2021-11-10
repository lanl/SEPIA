from abc import ABC, abstractmethod
import numpy as np

class SepiaDistribution(ABC):
    @property
    @abstractmethod
    def bijector(self):
        """Bijector for distribution."""
        pass

    @abstractmethod
    def logpdf(self, x):
        """Evaluate and return the log density at `x`."""
        pass

    @abstractmethod
    def support(self):
        """Return the support."""
        pass

    @abstractmethod
    def in_support(self, x):
        """Return `True` if `x` is within the support. Otherwise, return `False`."""
        pass

    @abstractmethod
    def sample(self, rng=np.random):
        """Return a random sample from the distribution."""
        pass

    # NOTE: I think the following methods could be helpful if we consider using a multivariate
    # Gaussian proposal for parameters in their unconstrained space. This would involve/require
    # transforming priors so that they are unconstrained. (e.g. If X ~ Gamma(a, b), then propose a
    # new value for log(x), and add the appropriate log absolute Jacobian.)

    def to_constrained_space(self, unconstrained_x):
        """Transform `unconstrained_x` with support on the real line / ball into its support."""
        return self.bijector.inv_transfrom(unconstrained_x)

    def log_abs_det_jacobian(self, unconstrained_x):
        """
        Compute the log absolute determinant of the jacobian given the parameter's
        constrained and unconstrained values.
        """
        return self.bijector.logdetjac(unconstrained_x)

    def logpdf_plus_log_abs_det_jacobian(self, unconstrained_x):
        """
        Compute the log density plus the log absolute value of the determinant of the jacobian
        given a parameter's constrained and unconstrained values.
        """
        x = self.to_constrained_space(unconstrained_x)
        return self.logpdf(x) + self.log_abs_det_jacobian(self, unconstrained_x)
