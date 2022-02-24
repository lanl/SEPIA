import numpy as np
from .MvNormal import MvNormal

class GP:
    def __init__(self, cov_fn, mean_fn=None):
        """
        cov_fn: covariance function. Takes a matrix X and returns the MvNormal covariance matrix.
        mean_fn: mean function. Takes a matrix X and returns the MvNormal mean.
        """
        self.cov_fn = cov_fn
        self.mean_fn = mean_fn

    def compute_mean(self, X):
        # Helper to compute mean at indexing points.
        # If mean_fn is not provided, return 0.
        if self.mean_fn is None:
            return np.zeros(X.shape[0])
        else:
            return self.mean_fn(X)

    def __call__(self, X, nugget=0):
        """
        When an instantiated GP is called, with argument X, a MvNormal is returned.
        """
        cov = self.cov_fn(X)
        self.dim = X.shape[0]
        mean = self.compute_mean(X)
        return MvNormal(mean=mean, cov=cov + nugget)

    def posterior(self, X, y, Xnew, cov_obs=0):
        """
        Compute posterior distribution of y at new locations Xnew, given data (X, y).
        A covariance matrix (cov_obs) for the observations can be provided. This
        will be added to the covariance matrix for the observations only.
        The result is a MvNormal with the correct posterior mean and covariance.
        """
        dim = y.shape[0]
        X_all = np.concatenate([Xnew, X])
        K_all = self.cov_fn(X_all)
        K_old = K_all[-dim:, -dim:] + cov_obs
        K_new = K_all[:-dim, :-dim]
        G = K_all[-dim:, :-dim]
        C = np.linalg.solve(K_old, G).T
        mu = self.compute_mean(X_all)
        mu_new = mu[:-dim]
        mu_old = mu[-dim:]

        return MvNormal(
            mean=mu_new + C @ (y - mu_old),
            cov=K_new - C @ G
        )
