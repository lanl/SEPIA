import numpy as np
from scipy.spatial import distance

import ppl
import ppl.distributions as dist

def sqexpkernel(X, length_scale, process_sd):
    D2 = distance.cdist(X, X, metric="sqeuclidean")
    return (process_sd ** 2) * np.exp(-D2 / length_scale ** 2)

class NoEmuCalibModel(ppl.AbstractModel):
    def model(self, y, xs, eta, W, theta_dim, num_basis, D=None):
        if D is not None:
            # GP covariance for discrepancy.
            length_scale = self.rv("length_scale", dist.Gamma(2, 1.5))  # should be smooth.
            process_sd = self.rv("process_sd", dist.Gamma(0.01, 1))  # should encourage little discrepancy. 
            Sigma = self.transform(
                "Sigma",
                np.kron(
                    sqexpkernel(xs, length_scale=length_scale, process_sd=process_sd),
                    np.eye(num_basis)
                )
            )

        # theta, parameter to calibrate.
        t = self.rv("t", dist.Uniform(np.zeros(theta_dim), 1))

        # Marginal covariance.
        # TODO: wrap the covariances so they can be inverted more efficiently,
        # e.g., using Woodbury.
        lam = self.rv("lam", dist.Gamma(5, 1/5))

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
