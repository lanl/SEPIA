import unittest
import numpy as np
import copy

from sepia.Examples import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model

np.random.seed(42)

class SepiaModelSetupMultiOutputTestCase(unittest.TestCase):

    def setUp(self, m=100, n=1, nt_sim=50, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
        self.data_dict = generate_data.generate_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs,
                                                                  n_theta=n_theta, n_basis=n_basis,
                                                                  sig_n=sig_n, seed=seed)

    def test_multivariate_sim_only_setup(self):
        """
        Tests setup for multivariate sim only model
        """

        d = SepiaData(t_sim=self.data_dict['t_sim'], y_sim=self.data_dict['y_sim'], y_ind_sim=self.data_dict['y_ind_sim'])
        print('Testing multivariate sim-only SepiaModelSetup...', flush=True)
        print(d, flush=True)

        # Try it without doing standardization/transform to be sure it doesn't break
        model_notrans = setup_model(copy.deepcopy(d))

        # Do explicit transformation
        d.transform_xt()
        d.standardize_y()
        model = setup_model(d)

        # Check that either way gives same transformation
        self.assertTrue(np.allclose(model_notrans.data.sim_data.orig_y_mean, model.data.sim_data.orig_y_mean))
        self.assertTrue(np.allclose(model_notrans.data.sim_data.orig_y_sd, model.data.sim_data.orig_y_sd))
        self.assertTrue(np.allclose(model_notrans.data.sim_data.y_std, model.data.sim_data.y_std))
        self.assertTrue(np.allclose(model_notrans.data.sim_data.t_trans, model.data.sim_data.t_trans))

        # Check model components are set up as expected
        self.assertTrue(not model.num.scalar_out)
        self.assertTrue(model.num.sim_only)
        self.assertTrue(model.num.m == 100)
        self.assertTrue(model.num.n == 0)
        self.assertTrue(model.num.p == 1)
        self.assertTrue(model.num.q == 3)
        self.assertTrue(model.num.pu == 5)
        self.assertTrue(model.num.pv == 0)
        #self.assertTrue(np.allclose(model.num.w, model.data.sim_data.y_std)) # TODO compute projection

        # Check parameter setup -- betaU
        betaU = model.params.betaU
        self.assertTrue(betaU.val_shape == (model.num.q + model.num.p, model.num.pu))
        self.assertTrue(betaU.prior.dist == 'Beta')
        self.assertTrue(betaU.mcmc.stepType == 'BetaRho')

        # Check parameter setup -- lamUz
        lamUz = model.params.lamUz
        self.assertTrue(lamUz.val_shape == (1, model.num.pu))
        self.assertTrue(lamUz.prior.dist == 'Gamma')
        self.assertTrue(lamUz.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamWOs
        lamWOs = model.params.lamWOs
        self.assertTrue(lamWOs.val_shape == (1, 1))
        self.assertTrue(lamWOs.prior.dist == 'Gamma')
        self.assertTrue(lamWOs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamWs
        lamWs = model.params.lamWs
        self.assertTrue(lamWs.val_shape == (1, model.num.pu))
        self.assertTrue(lamWs.prior.dist == 'Gamma')
        self.assertTrue(lamWs.mcmc.stepType == 'PropMH')

        mcmc_list_names = [p.name for p in model.params.mcmcList]
        self.assertTrue(set(mcmc_list_names) == set(['betaU', 'lamUz', 'lamWOs', 'lamWs']))


    def test_multivariate_sim_and_obs_setup_noD(self):
        """
        Tests setup for multivariate sim and obs model with no D
        """

        d = SepiaData(t_sim=self.data_dict['t_sim'], y_sim=self.data_dict['y_sim'], y_ind_sim=self.data_dict['y_ind_sim'],
                      y_obs=self.data_dict['y_obs'], y_ind_obs=self.data_dict['y_ind_obs'])
        print('Testing multivariate sim and obs SepiaModelSetup...', flush=True)
        print(d, flush=True)

        # Try it without doing standardization/transform/pc basis to be sure it doesn't break
        model_notrans = setup_model(copy.deepcopy(d))

        # Do explicit transformation
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(n_pc=5)
        model = setup_model(d)

        # Check that either way gives same transformation
        self.assertTrue(np.allclose(model_notrans.data.sim_data.orig_y_mean, model.data.sim_data.orig_y_mean))
        self.assertTrue(np.allclose(model_notrans.data.sim_data.orig_y_sd, model.data.sim_data.orig_y_sd))
        self.assertTrue(np.allclose(model_notrans.data.sim_data.y_std, model.data.sim_data.y_std))
        self.assertTrue(np.allclose(model_notrans.data.sim_data.t_trans, model.data.sim_data.t_trans))
        self.assertTrue(np.allclose(model_notrans.data.obs_data.orig_y_mean, model.data.obs_data.orig_y_mean))
        self.assertTrue(np.allclose(model_notrans.data.obs_data.orig_y_sd, model.data.obs_data.orig_y_sd))
        self.assertTrue(np.allclose(model_notrans.data.obs_data.y_std, model.data.obs_data.y_std))

        # Check model components are set up as expected
        self.assertTrue(not model.num.scalar_out)
        self.assertTrue(not model.num.sim_only)
        self.assertTrue(model.num.m == 100)
        self.assertTrue(model.num.n == 1)
        self.assertTrue(model.num.p == 1)
        self.assertTrue(model.num.q == 3)
        self.assertTrue(model.num.pu == 5)
        self.assertTrue(model.num.pv == 0)
        #self.assertTrue(np.allclose(model.num.w, model.data.sim_data.y_std)) # TODO compute projection
        #self.assertTrue(np.allclose(model.num.u, model.data.obs_data.y_std)) # TODO compute projection

        # Check parameter setup -- betaU
        betaU = model.params.betaU
        self.assertTrue(betaU.val_shape == (model.num.q + model.num.p, model.num.pu))
        self.assertTrue(betaU.prior.dist == 'Beta')
        self.assertTrue(betaU.mcmc.stepType == 'BetaRho')

        # Check parameter setup -- lamUz
        lamUz = model.params.lamUz
        self.assertTrue(lamUz.val_shape == (1, model.num.pu))
        self.assertTrue(lamUz.prior.dist == 'Gamma')
        self.assertTrue(lamUz.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamWOs
        lamWOs = model.params.lamWOs
        self.assertTrue(lamWOs.val_shape == (1, 1))
        self.assertTrue(lamWOs.prior.dist == 'Gamma')
        self.assertTrue(lamWOs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamWs
        lamWs = model.params.lamWs
        self.assertTrue(lamWs.val_shape == (1, model.num.pu))
        self.assertTrue(lamWs.prior.dist == 'Gamma')
        self.assertTrue(lamWs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamOs
        lamOs = model.params.lamOs
        self.assertTrue(lamOs.val_shape == (1, 1))
        self.assertTrue(lamOs.prior.dist == 'Gamma')
        self.assertTrue(lamOs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- theta
        theta = model.params.theta
        self.assertTrue(theta.val_shape == (1, model.num.q))
        self.assertTrue(theta.prior.dist == 'Normal')
        self.assertTrue(theta.mcmc.stepType == 'Uniform')
        self.assertTrue(np.allclose(theta.orig_range[0], 0))
        self.assertTrue(np.allclose(theta.orig_range[1], 1))

        mcmc_list_names = [p.name for p in model.params.mcmcList]
        self.assertTrue(set(mcmc_list_names) == set(['betaU', 'lamUz', 'lamWOs', 'lamWs', 'lamOs', 'theta']))


    def test_multivariate_sim_and_obs_setup(self):
        """
        Tests setup for multivariate sim and obs model with D
        """

        d = SepiaData(t_sim=self.data_dict['t_sim'], y_sim=self.data_dict['y_sim'], y_ind_sim=self.data_dict['y_ind_sim'],
                      y_obs=self.data_dict['y_obs'], y_ind_obs=self.data_dict['y_ind_obs'])
        print('Testing multivariate sim and obs SepiaModelSetup with discrep...', flush=True)
        print(d, flush=True)

        # Try it without doing standardization/transform/pc basis to be sure it doesn't break
        model_notrans = setup_model(copy.deepcopy(d))

        # Do explicit transformation
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(n_pc=5)
        d.create_D_basis(type='constant')
        d.create_D_basis(type='linear')
        custom_D = np.vstack([np.ones(d.obs_data.y.shape[1]), d.obs_data.y_ind])
        d.create_D_basis(D_obs=custom_D)
        model = setup_model(d)

        # Check that either way gives same transformation
        self.assertTrue(np.allclose(model_notrans.data.sim_data.orig_y_mean, model.data.sim_data.orig_y_mean))
        self.assertTrue(np.allclose(model_notrans.data.sim_data.orig_y_sd, model.data.sim_data.orig_y_sd))
        self.assertTrue(np.allclose(model_notrans.data.sim_data.y_std, model.data.sim_data.y_std))
        self.assertTrue(np.allclose(model_notrans.data.sim_data.t_trans, model.data.sim_data.t_trans))
        self.assertTrue(np.allclose(model_notrans.data.obs_data.orig_y_mean, model.data.obs_data.orig_y_mean))
        self.assertTrue(np.allclose(model_notrans.data.obs_data.orig_y_sd, model.data.obs_data.orig_y_sd))
        self.assertTrue(np.allclose(model_notrans.data.obs_data.y_std, model.data.obs_data.y_std))

        # Check model components are set up as expected
        self.assertTrue(not model.num.scalar_out)
        self.assertTrue(not model.num.sim_only)
        self.assertTrue(model.num.m == 100)
        self.assertTrue(model.num.n == 1)
        self.assertTrue(model.num.p == 1)
        self.assertTrue(model.num.q == 3)
        self.assertTrue(model.num.pu == 5)
        self.assertTrue(model.num.pv == 2)
        #self.assertTrue(np.allclose(model.num.w, model.data.sim_data.y_std)) # TODO compute projection
        #self.assertTrue(np.allclose(model.num.u, model.data.obs_data.y_std)) # TODO compute projection
        # self.assertTrue(np.allclose(model.num.v, model.data.obs_data.y_std)) # TODO compute projection

        # Check parameter setup -- betaU
        betaU = model.params.betaU
        self.assertTrue(betaU.val_shape == (model.num.q + model.num.p, model.num.pu))
        self.assertTrue(betaU.prior.dist == 'Beta')
        self.assertTrue(betaU.mcmc.stepType == 'BetaRho')

        # Check parameter setup -- betaV
        betaV = model.params.betaV
        self.assertTrue(betaV.val_shape == (1, 1))
        self.assertTrue(betaV.prior.dist == 'Beta')
        self.assertTrue(betaV.mcmc.stepType == 'BetaRho')

        # Check parameter setup -- lamUz
        lamUz = model.params.lamUz
        self.assertTrue(lamUz.val_shape == (1, model.num.pu))
        self.assertTrue(lamUz.prior.dist == 'Gamma')
        self.assertTrue(lamUz.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamUz
        lamVz = model.params.lamVz
        self.assertTrue(lamVz.val_shape == (1, 1))
        self.assertTrue(lamVz.prior.dist == 'Gamma')
        self.assertTrue(lamVz.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamWOs
        lamWOs = model.params.lamWOs
        self.assertTrue(lamWOs.val_shape == (1, 1))
        self.assertTrue(lamWOs.prior.dist == 'Gamma')
        self.assertTrue(lamWOs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamWs
        lamWs = model.params.lamWs
        self.assertTrue(lamWs.val_shape == (1, model.num.pu))
        self.assertTrue(lamWs.prior.dist == 'Gamma')
        self.assertTrue(lamWs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamOs
        lamOs = model.params.lamOs
        self.assertTrue(lamOs.val_shape == (1, 1))
        self.assertTrue(lamOs.prior.dist == 'Gamma')
        self.assertTrue(lamOs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- theta
        theta = model.params.theta
        self.assertTrue(theta.val_shape == (1, model.num.q))
        self.assertTrue(theta.prior.dist == 'Normal')
        self.assertTrue(theta.mcmc.stepType == 'Uniform')
        self.assertTrue(np.allclose(theta.orig_range[0], 0))
        self.assertTrue(np.allclose(theta.orig_range[1], 1))

        mcmc_list_names = [p.name for p in model.params.mcmcList]
        self.assertTrue(set(mcmc_list_names) == set(['betaU', 'betaV', 'lamUz', 'lamVz', 'lamWOs', 'lamWs', 'lamOs', 'theta']))

    def test_multivariate_sim_and_obs_lamVzGroups_setup(self):
        """
        Tests setup for multivariate sim and obs model with D and lamVzGroups
        """

        d = SepiaData(t_sim=self.data_dict['t_sim'], y_sim=self.data_dict['y_sim'], y_ind_sim=self.data_dict['y_ind_sim'],
                      y_obs=self.data_dict['y_obs'], y_ind_obs=self.data_dict['y_ind_obs'])
        print('Testing multivariate sim and obs SepiaModelSetup with discrep...', flush=True)
        print(d, flush=True)

        # Do explicit transformation
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(n_pc=5)
        custom_D = np.vstack([np.ones(d.obs_data.y.shape[1]), d.obs_data.y_ind, d.obs_data.y_ind**2])
        d.create_D_basis(D_obs=custom_D)
        lamVzGroup = [0, 1, 1]
        model = setup_model(d, lamVzGroup=lamVzGroup)

        # Check model components are set up as expected
        self.assertTrue(not model.num.scalar_out)
        self.assertTrue(not model.num.sim_only)
        self.assertTrue(model.num.m == 100)
        self.assertTrue(model.num.n == 1)
        self.assertTrue(model.num.p == 1)
        self.assertTrue(model.num.q == 3)
        self.assertTrue(model.num.pu == 5)
        self.assertTrue(model.num.pv == 3)

        # Check parameter setup -- betaU
        betaU = model.params.betaU
        self.assertTrue(betaU.val_shape == (model.num.q + model.num.p, model.num.pu))
        self.assertTrue(betaU.prior.dist == 'Beta')
        self.assertTrue(betaU.mcmc.stepType == 'BetaRho')

        # Check parameter setup -- betaV
        betaV = model.params.betaV
        self.assertTrue(betaV.val_shape == (1, 2))
        self.assertTrue(betaV.prior.dist == 'Beta')
        self.assertTrue(betaV.mcmc.stepType == 'BetaRho')

        # Check parameter setup -- lamUz
        lamUz = model.params.lamUz
        self.assertTrue(lamUz.val_shape == (1, model.num.pu))
        self.assertTrue(lamUz.prior.dist == 'Gamma')
        self.assertTrue(lamUz.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamUz
        lamVz = model.params.lamVz
        self.assertTrue(lamVz.val_shape == (1, 2))
        self.assertTrue(lamVz.prior.dist == 'Gamma')
        self.assertTrue(lamVz.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamWOs
        lamWOs = model.params.lamWOs
        self.assertTrue(lamWOs.val_shape == (1, 1))
        self.assertTrue(lamWOs.prior.dist == 'Gamma')
        self.assertTrue(lamWOs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamWs
        lamWs = model.params.lamWs
        self.assertTrue(lamWs.val_shape == (1, model.num.pu))
        self.assertTrue(lamWs.prior.dist == 'Gamma')
        self.assertTrue(lamWs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- lamOs
        lamOs = model.params.lamOs
        self.assertTrue(lamOs.val_shape == (1, 1))
        self.assertTrue(lamOs.prior.dist == 'Gamma')
        self.assertTrue(lamOs.mcmc.stepType == 'PropMH')

        # Check parameter setup -- theta
        theta = model.params.theta
        self.assertTrue(theta.val_shape == (1, model.num.q))
        self.assertTrue(theta.prior.dist == 'Normal')
        self.assertTrue(theta.mcmc.stepType == 'Uniform')
        self.assertTrue(np.allclose(theta.orig_range[0], 0))
        self.assertTrue(np.allclose(theta.orig_range[1], 1))

        mcmc_list_names = [p.name for p in model.params.mcmcList]
        self.assertTrue(set(mcmc_list_names) == set(['betaU', 'betaV', 'lamUz', 'lamVz', 'lamWOs', 'lamWs', 'lamOs', 'theta']))



    # Moved to separate test case
    # def test_multivariate_sim_and_obs_ragged_setup(self):
    #     m = 700  # number of simulated observations
    #     p = 3  # dimension of x (simulation inputs)
    #     ell_sim = 1000  # dimension of y output sim
    #     pu = 3  # number of PCs
    #     q = 2  # dimension of t (extra sim inputs)
    #     n = 5  # number of observed observations
    #
    #     ell_obs = np.random.randint(100, 600, n)
    #
    #     y_ind_sim = np.linspace(0, 100, ell_sim)
    #     K_true_sim = np.vstack([0.5 * (np.sin(y_ind_sim) + 1), np.square(-y_ind_sim + 50) / 2500, y_ind_sim / 100])
    #     y_sim = np.transpose(np.log(1 + y_ind_sim)[:, None] + np.dot(K_true_sim.T, 2 * np.array([1, 0.5, 0.2])[:,
    #                                                                                    None] * np.random.normal(0, 1, (
    #     pu, m))))
    #     x_sim = 0.5 * np.random.uniform(-1, 3, (m, p))
    #     t = np.random.uniform(-10, 10, (m, q))
    #
    #     y_ind_obs = [np.linspace(0, 100, ell_obs[i]) + np.random.uniform(-3, 3, ell_obs[i]) for i in
    #                  range(len(ell_obs))]
    #     for yi in y_ind_obs:
    #         yi[yi < 0] = 0
    #     K_true_obs = [np.vstack([0.5 * (np.sin(yi) + 1), np.square(-yi + 50) / 2500, yi / 100]) for yi in y_ind_obs]
    #     y_obs = [10 + np.squeeze(np.log(1 + y_ind_obs[i])[:, None] +
    #                              np.dot(K_true_obs[i].T,
    #                                     2 * np.array([1, 0.5, 0.2])[:, None] * np.random.normal(0, 1, (pu, 1)))) for i
    #              in range(len(y_ind_obs))]
    #     x_obs = 0.5 * np.random.uniform(-1, 3, (n, p))
    #
    #     d = SepiaData(x_sim=x_sim, y_sim=y_sim, t_sim=t, y_ind_sim=y_ind_sim, x_obs=x_obs, y_obs=y_obs,
    #                   y_ind_obs=y_ind_obs)
    #
    #     model = setup_model(d)



