import numpy as np
from scipy.spatial import distance
from scipy.linalg import block_diag

from . import ppl
from .ppl import distributions as dist
from .ppl.inference.diagnostics import ess
from .ppl.inference import MCMC, MvARWM, Shaper
from .ppl.util.pbar import pbrange

def radial_basis(X, knots, sd):
    """
    X: points to evaluate kernel (n x q)
    knots: kernel locations (m x q)
    sd: kernel width

	Return: an nxm matrix of Gaussian density evaluations at X with centers at
	knots and provided sd.
    """
    diff = X[..., None] - knots.T[None, ...]  # n x q x m
    ss = np.sum(diff ** 2, axis=1)  # n x m
    v = sd ** 2
    return np.exp(-ss / (2 * v)) / np.sqrt(2 * np.pi) / sd

class DBasis:
    def __init__(self, S, knots=None, num_basis=None, seed=None,
                 normalize=True, basis=radial_basis, bias=True,
                 **kwargs):
        """
        S: List of indexing points represented by list of np arrays. Each array need not have same
        length. Each element represents an observation.
        """
        self.S = S
        self.dim = S[0].shape[1]
        assert np.all([s.shape[1] == self.dim for s in self.S])
        self.num_basis = num_basis
        self.knots = knots
        self.normalize = normalize
        self.basis = basis
        self.bias = bias
        self.kwargs = kwargs
        self.D = None

        if self.num_basis is None:
            self.num_basis = self.knots.shape[0]
        elif self.knots is None:
            if self.dim == 1:
                self.knots = np.linspace(0, 1, self.num_basis)[:, None]
            else:
                np.random.seed(seed)
                # NOTE: Latin hypercube?
                self.knots = np.random.rand(self.num_basis, self.dim)

        Bs = [self.basis(s, self.knots, **self.kwargs) for s in self.S]
        if self.normalize:
           Bmax = max([np.abs(B).max() for B in Bs])
           Bs = [B / Bmax for B in Bs]

        if self.bias:
            Bs = [np.hstack([
                np.ones([B.shape[0], 1]),
                B
            ]) for B in Bs]
            self.num_basis += 1

        self.D = block_diag(*Bs)

        if self.normalize:
            self.D /= np.abs(self.D).max()

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
        length_scale = dist.Gamma(100, 1/100),  # large -> encourage smoothness
        process_sd = dist.Gamma(1, 1/10),  # small -> encourage little discrepancy
        t = dist.Uniform(np.zeros(theta_dim), 1),
        lam = dist.Gamma(5, 1/5),
    )

class NoEmuCalibModel(ppl.AbstractModel):
    """
    Emulator-free Calibration Model.
    """
    def model(self, y, X, eta, W, theta_dim, priors, Dbasis=None):
        """
        :param np.ndarray y: concatenated vector of (observed) responses.
        :param np.ndarray X: matrix of physics model inputs to eta.
        :param function(x, t) eta: a function (computer model) which takes
            inputs x (vector) and parameters t (vector) and returns a matrix of
            outputs, where the rows have the same index as the elements in y.
        :param W: observation coavariance, up to a constant.
        :param int theta_dim: number of phyiscal parameter to calibrate.
        :param dict priors: Priors for length_scale, process_sd, t, and lam.
        :param matrix or None D: Discrepancy basis.
        """
        if Dbasis is not None:
            D = Dbasis.D
            num_basis = Dbasis.num_basis

            # GP covariance for discrepancy.
            length_scale = self.rv("length_scale", priors["length_scale"])  # should be smooth.
            process_sd = self.rv("process_sd", priors["process_sd"])  # should encourage little discrepancy. 
            Sigma = self.transform(
                "Sigma",
                np.kron(
                    sqexpkernel(X, length_scale=length_scale, process_sd=process_sd),
                    np.eye(num_basis)
                )
            )

        # theta, parameter to calibrate.
        t = self.rv("t", priors["t"])

        # Marginal covariance.
        # TODO: wrap the covariances so they can be inverted more efficiently,
        # e.g., using Woodbury.
        lam = self.rv("lam", priors["lam"])

        if Dbasis is not None:
            marg_cov = self.transform(
                "marg_cov",
                D @ Sigma @ D.T + lam ** 2 * W
            )
        else:
            marg_cov = self.transform("marg_cov", lam ** 2 * W)

        # Likelihood.
        # TODO: Implement a custom likelihood which accepts D, Sigma, lam, and
        # W, and does Sherman-Woodbury. Might be able to cache inverse of W by
        # memoizing.
        y = self.rv(
            "y",
            dist.MvNormal(
                np.concatenate([eta(x, t) for x in X]).flatten(),
                marg_cov
            ),
            obs=y
        )

def make_model_data(y, X, eta, W, theta_dim, priors=None, Dbasis=None):
    if priors is None:
        priors = make_default_priors(theta_dim=theta_dim)

    return dict(y=y, X=X, eta=eta, W=W, theta_dim=theta_dim,
                Dbasis=Dbasis, priors=priors)

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

def posterior_predictive_nodelta(Xnew, data, post_samples, indexing_points):
    pass


def posterior_predictive(Xnew, Wnew, data, post_samples, indexing_points):
    Dbasis = data["Dbasis"]

    if Dbasis is None:
        return posterior_predictive_nodelta(Xnew, data, post_samples, indexing_points)
    else:
        Dbasis = DBasis(
            S=indexing_points + Dbasis.S, # new indexing points and old indexing points.
            knots=Dbasis.knots,
            basis=Dbasis.basis, normalize=Dbasis.normalize,
            bias=Dbasis.bias, **Dbasis.kwargs
        )
        D = Dbasis.D
        num_basis = Dbasis.num_basis

    eta = data['eta']
    def _mean_fn(X, t):
        return np.concatenate([
            eta(x, t) for x in X
        ])

    # NOTE: This is not used if discrepancy is not included.
    def get_post(i):
        def cov_fn(X):
            Sigma = np.kron(
                sqexpkernel(
                    X,
                    length_scale=post_samples['length_scale'][i],
                    process_sd=post_samples['process_sd'][i],
                ),
                np.eye(num_basis)
            )
            # print((D.shape, Sigma.shape))
            return D @ Sigma @ D.T

        def mean_fn(X):
            return _mean_fn(X, post_samples['t'][i])

        gp = dist.GP(cov_fn=cov_fn, mean_fn=mean_fn)
        cov_obs = post_samples['lam'][i] ** 2 * data['W']
        post = gp.posterior(X=data['X'], y=data['y'], Xnew=Xnew, cov_obs=cov_obs)
        return np.random.normal(post.mean, np.sqrt(np.diag(post.cov)))

    num_mcmc_samples = len(post_samples[list(post_samples.keys())[0]])

    post_mean_fn = np.stack([get_post(i) for i in pbrange(num_mcmc_samples)])
    post_delta = np.stack([
        post_mean_fn[i] - _mean_fn(Xnew, t)
        for i, t in enumerate(post_samples['t'])
    ])

    L = np.linalg.cholesky(Wnew)
    post_predictive = np.stack([
        # Use the cholesky for this.
        (L * post_samples['lam'][i]) @ np.random.randn(L.shape[0]) + post_mean_fn[i]
        # NOTE: Same as.
        # dist.MvNormal(post_mean_fn[i], post_samples['lam'][i] ** 2 * Wnew).sample()
        for i in pbrange(post_mean_fn.shape[0])
    ])

    return dict(mean_fn=post_mean_fn, delta=post_delta, predictive=post_predictive)
