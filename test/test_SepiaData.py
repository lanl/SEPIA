import unittest
import numpy as np

from sepia.SepiaData import SepiaData

np.random.seed(42)

class SepiaDataTestCase(unittest.TestCase):

    def test_univariate_sim_only_x_only(self):
        """
        Tests setup for univariate sim only where we only use an x input, not t.
        """
        m = 700  # number of simulated observations
        p = 3    # dimension of x (simulation inputs)

        x = 0.5 * np.random.uniform(-1, 3, (m, p))
        y = 5 * np.random.normal(0, 1, m) + 2
        d = SepiaData(x_sim=x, y_sim=y, t_sim=None)

        print('Testing univariate sim-only SepiaData...')
        print(d)
        self.assertTrue(d.obs_data is None)
        self.assertTrue(d.sim_only)
        self.assertTrue(d.scalar_out)

        d.transform_xt()
        self.assertTrue(np.all(np.min(d.sim_data.x_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.x_trans, 0) == 1))

        d.transform_xt(-10, 10)
        self.assertTrue(np.all(np.min(d.sim_data.x_trans, 0) == -10))
        self.assertTrue(np.all(np.max(d.sim_data.x_trans, 0) == 10))

        d.standardize_y(center=False, scale=False)
        self.assertEqual(d.sim_data.orig_y_sd, 1)
        self.assertEqual(d.sim_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.sim_data.y, d.sim_data.y_std))

        d.standardize_y(scale='columnwise')
        self.assertTrue(np.allclose(d.sim_data.orig_y_sd, 5, rtol=0.1))
        #self.assertTrue(np.allclose(d.sim_data.orig_y_mean, 2, rtol=0.1))
        self.assertTrue(np.allclose(np.mean(d.sim_data.y_std, 0), 0, rtol=0.1))
        self.assertTrue(np.allclose(np.std(d.sim_data.y_std, 0), 1, rtol=0.1))
        self.assertTrue(d.sim_data.y.shape == d.sim_data.y_std.shape)

        d.create_K_basis(10)
        d.create_D_basis()

    def test_univariate_sim_only_t_only(self):
        """
        Tests setup for univariate sim only where we only use a t input, not x; x is set up as a dummy internally.
        """
        m = 700  # number of simulated observations
        p = 3    # dimension of x (simulation inputs)

        t = 0.5 * np.random.uniform(-1, 3, (m, p))
        y = 5 * np.random.normal(0, 1, m) + 2
        d = SepiaData(x_sim=None, y_sim=y, t_sim=t)

        print('Testing univariate sim-only SepiaData...')
        print(d)
        self.assertTrue(d.obs_data is None)
        self.assertTrue(d.sim_only)
        self.assertTrue(d.scalar_out)

        d.transform_xt()
        self.assertTrue(np.all(d.sim_data.x_trans == 0.5))

        d.standardize_y(center=False, scale=False)
        self.assertEqual(d.sim_data.orig_y_sd, 1)
        self.assertEqual(d.sim_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.sim_data.y, d.sim_data.y_std))

        d.standardize_y(scale='columnwise')
        self.assertTrue(np.allclose(d.sim_data.orig_y_sd, 5, rtol=0.1))
        self.assertTrue(np.allclose(d.sim_data.orig_y_mean, 2, rtol=0.1))
        self.assertTrue(np.allclose(np.mean(d.sim_data.y_std, 0), 0, rtol=0.1))
        self.assertTrue(np.allclose(np.std(d.sim_data.y_std, 0), 1, rtol=0.1))
        self.assertTrue(d.sim_data.y.shape == d.sim_data.y_std.shape)

        d.create_K_basis(10)
        d.create_D_basis()

    def test_multivariate_sim_only_x_only(self):
        """
        Tests setup for multivariate sim only where we only use an x input, not t.
        """
        m = 700     # number of simulated observations
        p = 3       # dimension of x (simulation inputs)
        ell = 1000  # dimension of y output
        pu = 3      # number of PCs

        y_ind = np.linspace(0, 100, ell)
        K_true = np.vstack([0.5*(np.sin(y_ind) + 1), np.square(-y_ind + 50)/2500, y_ind/100])
        y = np.transpose(np.log(1 + y_ind)[:, None] + np.dot(K_true.T, 2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0, 1, (pu, m))))
        x = 0.5 * np.random.uniform(-1, 3, (m, p))
        d = SepiaData(x_sim=x, y_sim=y, t_sim=None, y_ind_sim=y_ind)

        print('Testing multivariate sim-only SepiaData...')
        print(d)
        self.assertTrue(d.obs_data is None)
        self.assertTrue(d.sim_only)
        self.assertTrue(not d.scalar_out)

        d.transform_xt()
        self.assertTrue(np.all(np.min(d.sim_data.x_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.x_trans, 0) == 1))

        d.transform_xt(-10, 10)
        self.assertTrue(np.all(np.min(d.sim_data.x_trans, 0) == -10))
        self.assertTrue(np.all(np.max(d.sim_data.x_trans, 0) == 10))

        d.standardize_y(center=False, scale=False)
        self.assertEqual(d.sim_data.orig_y_sd, 1)
        self.assertEqual(d.sim_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.sim_data.y, d.sim_data.y_std))

        d.standardize_y(scale='columnwise')
        self.assertTrue(np.allclose(d.sim_data.orig_y_mean, np.log(1 + y_ind), rtol=0.1, atol=0.5))
        self.assertTrue(np.allclose(np.std(d.sim_data.y_std, 0), 1, rtol=0.1))
        self.assertTrue(np.allclose(np.mean(d.sim_data.y_std, 0), 0, rtol=0.1))
        self.assertTrue(d.sim_data.y.shape == d.sim_data.y_std.shape)

        d.create_K_basis(3)
        self.assertTrue(d.sim_data.K.shape == (pu, ell))
        d.create_D_basis()
        print(d)

    def test_multivariate_sim_only_t_only(self):
        """
        Tests setup for multivariate sim only where we only use a t input, not x (dummy x is set up).
        """
        m = 700  # number of simulated observations
        p = 3  # dimension of t (simulation inputs)
        ell = 1000  # dimension of y output
        pu = 3  # number of PCs

        y_ind = np.linspace(0, 100, ell)
        K_true = np.vstack([0.5 * (np.sin(y_ind) + 1), np.square(-y_ind + 50) / 2500, y_ind / 100])
        y = np.transpose(np.log(1 + y_ind)[:, None] + np.dot(K_true.T,
                                                             2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0,
                                                                                                                     1,
                                                                                                                     (
                                                                                                                     pu,
                                                                                                                     m))))
        t = 0.5 * np.random.uniform(-1, 3, (m, p))
        d = SepiaData(x_sim=None, y_sim=y, t_sim=t, y_ind_sim=y_ind)

        print('Testing multivariate sim-only SepiaData...')
        print(d)
        self.assertTrue(d.obs_data is None)
        self.assertTrue(d.sim_only)
        self.assertTrue(not d.scalar_out)

        d.transform_xt()

        d.standardize_y(center=False, scale=False)
        self.assertEqual(d.sim_data.orig_y_sd, 1)
        self.assertEqual(d.sim_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.sim_data.y, d.sim_data.y_std))

        d.standardize_y(scale='columnwise')
        self.assertTrue(np.allclose(d.sim_data.orig_y_mean, np.log(1 + y_ind), rtol=0.1, atol=0.5))
        self.assertTrue(np.allclose(np.std(d.sim_data.y_std, 0), 1, rtol=0.1))
        self.assertTrue(np.allclose(np.mean(d.sim_data.y_std, 0), 0, rtol=0.1))
        self.assertTrue(d.sim_data.y.shape == d.sim_data.y_std.shape)

        d.create_K_basis(3)
        self.assertTrue(d.sim_data.K.shape == (pu, ell))
        d.create_D_basis()
        print(d)

    def test_univariate_sim_and_obs(self):
        """
        Tests univiariate sim and obs where we pass in both x and t.
        """
        m = 700  # number of simulated observations
        p = 3    # dimension of x (sim/obs inputs)
        q = 2    # dimension of t (extra sim inputs)
        n = 5    # number of observed observations

        x_sim = np.random.uniform(-1, 3, (m, p))
        t = np.random.uniform(-10, 10, (m, q))
        x_obs = np.random.uniform(-1.5, 3.5, (n, p))
        y_sim = 5 * np.random.normal(0, 1, m) + 2
        y_obs = 5 * np.random.normal(0, 1, n) + 1
        d = SepiaData(x_sim=x_sim, y_sim=y_sim, t_sim=t, x_obs=x_obs, y_obs=y_obs)

        print('Testing univariate sim and obs SepiaData...')
        print(d)
        self.assertTrue(d.obs_data is not None)
        self.assertTrue(not d.sim_only)
        self.assertTrue(d.scalar_out)

        d.transform_xt()
        self.assertTrue(np.all(np.min(d.sim_data.x_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.x_trans, 0) == 1))
        self.assertTrue(np.all(np.min(d.sim_data.t_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.t_trans, 0) == 1))

        d.transform_xt(-10, 10)
        self.assertTrue(np.all(np.min(d.sim_data.x_trans, 0) == -10))
        self.assertTrue(np.all(np.max(d.sim_data.x_trans, 0) == 10))
        self.assertTrue(np.all(np.min(d.sim_data.t_trans, 0) == -10))
        self.assertTrue(np.all(np.max(d.sim_data.t_trans, 0) == 10))

        d.standardize_y(center=False, scale=False)
        self.assertEqual(d.sim_data.orig_y_sd, 1)
        self.assertEqual(d.sim_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.sim_data.y, d.sim_data.y_std))
        self.assertEqual(d.obs_data.orig_y_sd, 1)
        self.assertEqual(d.obs_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.obs_data.y, d.obs_data.y_std))

        d.standardize_y(scale='columnwise')
        self.assertTrue(np.allclose(d.sim_data.orig_y_sd, 5, rtol=0.1))
        self.assertTrue(np.allclose(d.sim_data.orig_y_mean, 2, rtol=0.1))
        self.assertTrue(np.allclose(np.mean(d.sim_data.y_std, 0), 0, rtol=0.1))
        self.assertTrue(np.allclose(np.std(d.sim_data.y_std, 0), 1, rtol=0.1))
        self.assertTrue(d.sim_data.y.shape == d.sim_data.y_std.shape)
        self.assertTrue(np.allclose(d.obs_data.orig_y_sd, 5, rtol=0.1))
        self.assertTrue(np.allclose(d.obs_data.orig_y_mean, 2, rtol=0.1))
        self.assertTrue(d.obs_data.y.shape == d.obs_data.y_std.shape)

        d.create_K_basis(10)
        d.create_D_basis()

    def test_univariate_sim_and_obs_no_x(self):
        """
        Tests univiariate sim and obs where we pass in t, but not x (it will create dummy x).
        """
        m = 1000  # number of simulated observations
        q = 2  # dimension of t (extra sim inputs)
        n = 5  # number of observed observations

        t = np.random.uniform(-10, 10, (m, q))
        y_sim = 5 * np.random.normal(0, 1, m) + 2
        y_obs = 5 * np.random.normal(0, 1, n) + 1
        d = SepiaData(y_sim=y_sim, t_sim=t, y_obs=y_obs)

        print('Testing univariate sim and obs SepiaData...')
        print(d)
        self.assertTrue(d.obs_data is not None)
        self.assertTrue(not d.sim_only)
        self.assertTrue(d.scalar_out)

        d.transform_xt()
        self.assertTrue(np.all(np.min(d.sim_data.t_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.t_trans, 0) == 1))

        d.standardize_y(center=False, scale=False)
        self.assertEqual(d.sim_data.orig_y_sd, 1)
        self.assertEqual(d.sim_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.sim_data.y, d.sim_data.y_std))
        self.assertEqual(d.obs_data.orig_y_sd, 1)
        self.assertEqual(d.obs_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.obs_data.y, d.obs_data.y_std))

        d.standardize_y(scale='columnwise')
        self.assertTrue(np.allclose(d.sim_data.orig_y_sd, 5, rtol=0.1))
        self.assertTrue(np.allclose(d.sim_data.orig_y_mean, 2, rtol=0.1))
        self.assertTrue(np.allclose(np.mean(d.sim_data.y_std, 0), 0, rtol=0.1))
        self.assertTrue(np.allclose(np.std(d.sim_data.y_std, 0), 1, rtol=0.1))
        self.assertTrue(d.sim_data.y.shape == d.sim_data.y_std.shape)
        self.assertTrue(np.allclose(d.obs_data.orig_y_sd, 5, rtol=0.1))
        self.assertTrue(np.allclose(d.obs_data.orig_y_mean, 2, rtol=0.1))
        self.assertTrue(d.obs_data.y.shape == d.obs_data.y_std.shape)

        d.create_K_basis(10)
        d.create_D_basis()

    def test_multivariate_sim_and_obs(self):
        """
        Tests multivariate sim and obs where we pass in x and t.
        """
        m = 700         # number of simulated observations
        p = 3           # dimension of x (simulation inputs)
        ell_sim = 1000  # dimension of y output sim
        ell_obs = 258   # dimension of y output obs
        pu = 3          # number of PCs
        q = 2           # dimension of t (extra sim inputs)
        n = 5           # number of observed observations

        y_ind_sim = np.linspace(0, 100, ell_sim)
        K_true_sim = np.vstack([0.5*(np.sin(y_ind_sim) + 1), np.square(-y_ind_sim + 50)/2500, y_ind_sim/100])
        y_sim = np.transpose(np.log(1 + y_ind_sim)[:, None] + np.dot(K_true_sim.T, 2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0, 1, (pu, m))))
        x_sim = 0.5 * np.random.uniform(-1, 3, (m, p))
        t = np.random.uniform(-10, 10, (m, q))

        y_ind_obs = np.linspace(0, 100, ell_obs) + np.random.uniform(-3, 3, ell_obs)
        y_ind_obs[y_ind_obs < 0] = 0
        K_true_obs = np.vstack([0.5*(np.sin(y_ind_obs) + 1), np.square(-y_ind_obs + 50)/2500, y_ind_obs/100])
        y_obs = 10 + np.transpose(np.log(1 + y_ind_obs)[:, None] + np.dot(K_true_obs.T, 2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0, 1, (pu, n))))
        x_obs = 0.5 * np.random.uniform(-1, 3, (n, p))

        d = SepiaData(x_sim=x_sim, y_sim=y_sim, t_sim=t, y_ind_sim=y_ind_sim, x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)

        print('Testing multivariate sim and obs SepiaData...')
        print(d)
        self.assertTrue(d.obs_data is not None)
        self.assertTrue(not d.sim_only)
        self.assertTrue(not d.scalar_out)

        d.transform_xt()
        self.assertTrue(np.all(np.min(d.sim_data.x_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.x_trans, 0) == 1))
        self.assertTrue(np.all(np.min(d.sim_data.t_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.t_trans, 0) == 1))

        d.standardize_y(center=False, scale=False)
        self.assertEqual(d.sim_data.orig_y_sd, 1)
        self.assertEqual(d.sim_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.sim_data.y, d.sim_data.y_std))
        self.assertEqual(d.obs_data.orig_y_sd, 1)
        self.assertEqual(d.obs_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.obs_data.y, d.obs_data.y_std))

        d.standardize_y(scale='columnwise')
        self.assertTrue(np.allclose(d.sim_data.orig_y_mean, np.log(1 + y_ind_sim), rtol=0.1, atol=0.5))
        self.assertTrue(np.allclose(np.mean(d.sim_data.y_std, 0), 0, rtol=0.1))
        self.assertTrue(np.allclose(np.std(d.sim_data.y_std, 0), 1, rtol=0.1))
        self.assertTrue(d.sim_data.y.shape == d.sim_data.y_std.shape)
        self.assertTrue(np.allclose(d.obs_data.orig_y_mean, np.log(1 + y_ind_obs), rtol=0.1, atol=0.5))
        self.assertTrue(d.obs_data.y.shape == d.obs_data.y_std.shape)

        d.create_K_basis(3)
        self.assertTrue(d.sim_data.K.shape == (pu, ell_sim))
        self.assertTrue(d.obs_data.K.shape == (pu, ell_obs))
        d.create_D_basis()
        self.assertTrue(d.obs_data.D.shape == (1, ell_obs))
        print(d)

    def test_multivariate_sim_and_obs_no_x(self):
        """
        Tests multivariate sim and obs where we pass in t but not x (x is a dummy variable).
        """
        m = 700         # number of simulated observations
        ell_sim = 1000  # dimension of y output sim
        ell_obs = 258   # dimension of y output obs
        pu = 3          # number of PCs
        q = 2           # dimension of t (extra sim inputs)
        n = 5           # number of observed observations

        y_ind_sim = np.linspace(0, 100, ell_sim)
        K_true_sim = np.vstack([0.5*(np.sin(y_ind_sim) + 1), np.square(-y_ind_sim + 50)/2500, y_ind_sim/100])
        y_sim = np.transpose(np.log(1 + y_ind_sim)[:, None] + np.dot(K_true_sim.T, 2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0, 1, (pu, m))))
        t = np.random.uniform(-10, 10, (m, q))

        y_ind_obs = np.linspace(0, 100, ell_obs) + np.random.uniform(-3, 3, ell_obs)
        y_ind_obs[y_ind_obs < 0] = 0
        K_true_obs = np.vstack([0.5*(np.sin(y_ind_obs) + 1), np.square(-y_ind_obs + 50)/2500, y_ind_obs/100])
        y_obs = 10 + np.transpose(np.log(1 + y_ind_obs)[:, None] + np.dot(K_true_obs.T, 2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0, 1, (pu, n))))

        d = SepiaData(y_sim=y_sim, t_sim=t, y_ind_sim=y_ind_sim, y_obs=y_obs, y_ind_obs=y_ind_obs)

        print('Testing multivariate sim and obs SepiaData...')
        print(d)
        self.assertTrue(d.obs_data is not None)
        self.assertTrue(not d.sim_only)
        self.assertTrue(not d.scalar_out)

        d.transform_xt()
        self.assertTrue(np.all(d.sim_data.x_trans == 0.5))
        self.assertTrue(np.all(d.obs_data.x_trans == 0.5))
        self.assertTrue(np.all(np.min(d.sim_data.t_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.t_trans, 0) == 1))

        d.standardize_y(center=False, scale=False)
        self.assertEqual(d.sim_data.orig_y_sd, 1)
        self.assertEqual(d.sim_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.sim_data.y, d.sim_data.y_std))
        self.assertEqual(d.obs_data.orig_y_sd, 1)
        self.assertEqual(d.obs_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.obs_data.y, d.obs_data.y_std))

        d.standardize_y(scale='columnwise')
        self.assertTrue(np.allclose(d.sim_data.orig_y_mean, np.log(1 + y_ind_sim), rtol=0.1, atol=0.5))
        self.assertTrue(np.allclose(np.mean(d.sim_data.y_std, 0), 0, rtol=0.1))
        self.assertTrue(np.allclose(np.std(d.sim_data.y_std, 0), 1, rtol=0.1))
        self.assertTrue(d.sim_data.y.shape == d.sim_data.y_std.shape)
        self.assertTrue(np.allclose(d.obs_data.orig_y_mean, np.log(1 + y_ind_obs), rtol=0.1, atol=0.5))
        self.assertTrue(d.obs_data.y.shape == d.obs_data.y_std.shape)

        d.create_K_basis(3)
        self.assertTrue(d.sim_data.K.shape == (pu, ell_sim))
        self.assertTrue(d.obs_data.K.shape == (pu, ell_obs))
        d.create_D_basis()
        self.assertTrue(d.obs_data.D.shape == (1, ell_obs))
        print(d)

    def test_multivariate_sim_and_obs_ragged(self):
        """
        Tests multivariate sim and obs where we pass in x and t but obs is ragged.
        """
        m = 700         # number of simulated observations
        p = 3           # dimension of x (simulation inputs)
        ell_sim = 1000  # dimension of y output sim
        pu = 3          # number of PCs
        q = 2           # dimension of t (extra sim inputs)
        n = 5           # number of observed observations

        ell_obs = np.random.randint(100, 600, n)

        y_ind_sim = np.linspace(0, 100, ell_sim)
        K_true_sim = np.vstack([0.5*(np.sin(y_ind_sim) + 1), np.square(-y_ind_sim + 50)/2500, y_ind_sim/100])
        y_sim = np.transpose(np.log(1 + y_ind_sim)[:, None] + np.dot(K_true_sim.T, 2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0, 1, (pu, m))))
        x_sim = 0.5 * np.random.uniform(-1, 3, (m, p))
        t = np.random.uniform(-10, 10, (m, q))

        y_ind_obs = [np.linspace(0, 100, ell_obs[i]) + np.random.uniform(-3, 3, ell_obs[i]) for i in range(len(ell_obs))]
        for yi in y_ind_obs:
            yi[yi < 0] = 0
        K_true_obs = [np.vstack([0.5*(np.sin(yi) + 1), np.square(-yi + 50)/2500, yi/100]) for yi in y_ind_obs]
        y_obs = [10 + np.squeeze(np.log(1 + y_ind_obs[i])[:, None] +
                                   np.dot(K_true_obs[i].T, 2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0, 1, (pu, 1)))) for i in range(len(y_ind_obs))]
        x_obs = 0.5 * np.random.uniform(-1, 3, (n, p))

        d = SepiaData(x_sim=x_sim, y_sim=y_sim, t_sim=t, y_ind_sim=y_ind_sim, x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)

        print('Testing multivariate sim and obs SepiaData...')
        print(d)
        self.assertTrue(d.obs_data is not None)
        self.assertTrue(not d.sim_only)
        self.assertTrue(not d.scalar_out)

        d.transform_xt()
        self.assertTrue(np.all(np.min(d.sim_data.x_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.x_trans, 0) == 1))
        self.assertTrue(np.all(np.min(d.sim_data.t_trans, 0) == 0))
        self.assertTrue(np.all(np.max(d.sim_data.t_trans, 0) == 1))

        d.standardize_y(center=False, scale=False)
        self.assertEqual(d.sim_data.orig_y_sd, 1)
        self.assertEqual(d.sim_data.orig_y_mean, 0)
        self.assertTrue(np.allclose(d.sim_data.y, d.sim_data.y_std))
        #self.assertEqual(d.obs_data.orig_y_sd, 1)
        #self.assertEqual(d.obs_data.orig_y_mean, 0)
        #self.assertTrue(np.allclose(d.obs_data.y, d.obs_data.y_std))

        d.standardize_y(scale='columnwise')
        self.assertTrue(np.allclose(d.sim_data.orig_y_mean, np.log(1 + y_ind_sim), rtol=0.1, atol=0.5))
        self.assertTrue(np.allclose(np.mean(d.sim_data.y_std, 0), 0, rtol=0.1))
        self.assertTrue(np.allclose(np.std(d.sim_data.y_std, 0), 1, rtol=0.1))
        self.assertTrue(d.sim_data.y.shape == d.sim_data.y_std.shape)
        #self.assertTrue(np.allclose(d.obs_data.orig_y_mean, np.log(1 + y_ind_obs), rtol=0.1, atol=0.5))
        #self.assertTrue(d.obs_data.y.shape == d.obs_data.y_std.shape)

        d.create_K_basis(3)
        self.assertTrue(d.sim_data.K.shape == (pu, ell_sim))
        #self.assertTrue(d.obs_data.K.shape == (pu, ell_obs))
        d.create_D_basis()
        #self.assertTrue(d.obs_data.D.shape == (1, ell_obs))
        print(d)

    def test_univariate_sim_only_x_only_cat_ind(self):
        """
        Tests setup for univariate sim only where we only use an x input, not t, and use x_cat_ind
        """
        m = 200  # number of simulated observations
        p = 3    # dimension of x (simulation inputs)

        x = np.concatenate([0.5 * np.random.uniform(-1, 3, (m, p-1)), np.random.choice(range(1,5), (m, 1), replace=True)], axis=1)
        y = 5 * np.random.normal(0, 1, m) + 2
        x_cat_ind = [0, 0, 4]
        d = SepiaData(x_sim=x, y_sim=y, x_cat_ind=x_cat_ind)

        print('Testing univariate sim-only SepiaData...')
        print(d)

        d.transform_xt()
        self.assertTrue(np.allclose(d.sim_data.x_trans[:, 2], x[:, 2]))
        self.assertEqual(np.min(d.sim_data.x_trans[:, 2]), 1)
        self.assertEqual(np.max(d.sim_data.x_trans[:, 2]), 4)

    def test_univariate_sim_only_t_only_cat_ind(self):
        """
        Tests setup for univariate sim only where we only use a t input, not x; x is set up as a dummy internally.
        Use t_cat_ind.
        """
        m = 200  # number of simulated observations
        p = 3    # dimension of x (simulation inputs)

        t = np.concatenate([0.5 * np.random.uniform(-1, 3, (m, p-1)), np.random.choice(range(1,5), (m, 1), replace=True)], axis=1)
        y = 5 * np.random.normal(0, 1, m) + 2
        t_cat_ind = [0, 0, 4]
        d = SepiaData(x_sim=None, y_sim=y, t_sim=t, t_cat_ind=t_cat_ind)

        print('Testing univariate sim-only SepiaData...')
        print(d)

        d.transform_xt()
        self.assertTrue(np.allclose(d.sim_data.t_trans[:, 2], t[:, 2]))
        self.assertEqual(np.min(d.sim_data.t_trans[:, 2]), 1)
        self.assertEqual(np.max(d.sim_data.t_trans[:, 2]), 4)




