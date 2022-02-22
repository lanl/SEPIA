import numpy as np
from scipy.spatial import distance

from . import ppl
from .ppl import distributions as dist
from .ppl.inference.diagnostics import ess
from .ppl.inference import MCMC, MvARWM, Shaper

def sqexpkernel(X, length_scale, process_sd):
    """
    Squared exponential covariance kernel.

    ```math
    k(a, b) = process_sd ^ 2 * exp(-sum_{k=1}^K (a_{i,k} - b_{j, k})^2 / (2 * length_scale))
    ```

    :param numpy.ndarray X: Matrix of GP inputs at which distances are computed.
    :param float length_scale: a positive scalar length scale.
    :param float process_sd: square root of the process variance. The process
        variance is the variance when distance is 0.
    """
    D2 = distance.cdist(X, X, metric="sqeuclidean")
    return (process_sd ** 2) * np.exp(-D2 / length_scale ** 2)

# TODO: Think about these.
def make_default_priors(theta_dim):
    return dict(
        length_scale = dist.Gamma(2, 1.5),  # should be smooth
        process_sd = dist.Gamma(0.01, 1),
        t = dist.Uniform(np.zeros(theta_dim), 1),
        lam = dist.Gamma(5, 1/5),
    )

class NoEmuCalibModel(ppl.AbstractModel):
    """
    Emulator-free Calibration Model.
    """
    def model(self, y, xs, eta, W, theta_dim, num_basis, priors, D=None):
        """
        :param np.ndarray y: concatenated vector of (observed) responses.
        :param [np.ndarray xs]: list of vectors.
        :param function(x, t) eta: a function (computer model) which takes
            inputs x (vector) and parameters t (vector) and returns a matrix of
            outputs, where the rows have the same index as the elements in y.
        :param W: observation coavariance, up to a constant.
        :param int theta_dim: number of phyiscal parameter to calibrate.
        :param num_basis: Number of bases.
        :param dict priors: Priors for length_scale, process_sd, t, and lam.
        :param matrix or None D: Discrepancy basis.
        """
        if D is not None:
            # GP covariance for discrepancy.
            length_scale = self.rv("length_scale", priors["length_scale"])  # should be smooth.
            process_sd = self.rv("process_sd", priors["process_sd"])  # should encourage little discrepancy. 
            Sigma = self.transform(
                "Sigma",
                np.kron(
                    sqexpkernel(xs, length_scale=length_scale, process_sd=process_sd),
                    np.eye(num_basis)
                )
            )

        # theta, parameter to calibrate.
        t = self.rv("t", priors["t"])

        # Marginal covariance.
        # TODO: wrap the covariances so they can be inverted more efficiently,
        # e.g., using Woodbury.
        lam = self.rv("lam", priors["lam"])

        if D is not None:
            marg_cov = self.transform(
                "marg_cov",
                D @ Sigma @ D.T + lam ** 2 * W
            )
        else:
            marg_cov = self.transform("marg_cov", lam ** 2 * W)

        # Likelihood.
        y = self.rv(
            "y",
            dist.MvNormal(
                np.concatenate([eta(x, t) for x in xs]).flatten(),
                marg_cov
            ),
            obs=y
        )

def make_model_data(y, xs, eta, W, theta_dim, num_basis, priors=None, D=None):
    if priors is None:
        priors = make_default_priors(theta_dim=theta_dim)

    return dict(y=y, xs=xs, eta=eta, W=W, theta_dim=theta_dim,
                num_basis=num_basis, D=D, priors=priors)

def do_mcmc(model, data, num_samples: int, burn: int, window=None, thinning: int=1, seed=None, init_state=None):
    if seed is not None:
        np.random.seed(1)

    if init_state is None:
        init_state = model.prior_predictive(**data)

    bijector = model.make_bijector(**data)
    shaper = Shaper(init_state)

    def logprob(vec):
        return model.logpdf(shaper.unvec(vec), **data, biject=True)

    if window is None:
        window = np.array([int(burn/2), burn, np.inf])

    kernel = MvARWM(logprob=logprob, window=window)
    mcmc = MCMC(kernel, np.random.randn(shaper.dim), shaper=shaper, bijector=bijector)
    samples = mcmc.fit(num_samples, burn, thinning)

    return dict(mcmc=mcmc, samples=samples, init_state=init_state,
                kernel=kernel, window=window, burn=burn,
                num_samples=num_samples, thinning=thinning)