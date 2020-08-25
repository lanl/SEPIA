import unittest
import numpy as np

import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model

np.random.seed(42)

class SepiaModelSetupRaggedOutputTestCase(unittest.TestCase):

    def setUp(self, m=100, n=1, nt_sim=50, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
        self.data_dict = generate_data.generate_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs,
                                                                  n_theta=n_theta, n_basis=n_basis,
                                                                  sig_n=sig_n, seed=seed)

    def test_multivariate_sim_and_obs_ragged_setup(self):
        m = 700  # number of simulated observations
        p = 3  # dimension of x (simulation inputs)
        ell_sim = 1000  # dimension of y output sim
        pu = 3  # number of PCs
        q = 2  # dimension of t (extra sim inputs)
        n = 5  # number of observed observations

        ell_obs = np.random.randint(100, 600, n)

        y_ind_sim = np.linspace(0, 100, ell_sim)
        K_true_sim = np.vstack([0.5 * (np.sin(y_ind_sim) + 1), np.square(-y_ind_sim + 50) / 2500, y_ind_sim / 100])
        y_sim = np.transpose(np.log(1 + y_ind_sim)[:, None] + np.dot(K_true_sim.T, 2 * np.array([1, 0.5, 0.2])[:,
                                                                                       None] * np.random.normal(0, 1, (
        pu, m))))
        x_sim = 0.5 * np.random.uniform(-1, 3, (m, p))
        t = np.random.uniform(-10, 10, (m, q))

        y_ind_obs = [np.linspace(0, 100, ell_obs[i]) + np.random.uniform(-3, 3, ell_obs[i]) for i in
                     range(len(ell_obs))]
        for yi in y_ind_obs:
            yi[yi < 0] = 0
        K_true_obs = [np.vstack([0.5 * (np.sin(yi) + 1), np.square(-yi + 50) / 2500, yi / 100]) for yi in y_ind_obs]
        y_obs = [10 + np.squeeze(np.log(1 + y_ind_obs[i])[:, None] +
                                 np.dot(K_true_obs[i].T,
                                        2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0, 1, (pu, 1)))) for i
                 in range(len(y_ind_obs))]
        x_obs = 0.5 * np.random.uniform(-1, 3, (n, p))

        d = SepiaData(x_sim=x_sim, y_sim=y_sim, t_sim=t, y_ind_sim=y_ind_sim, x_obs=x_obs, y_obs=y_obs,
                      y_ind_obs=y_ind_obs)

        model = setup_model(d)



