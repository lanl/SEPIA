import unittest
import numpy as np

import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel

np.random.seed(42)

class SepiaSaveLoadTestCase(unittest.TestCase):

    def setUp(self, m=100, n=1, nt_sim=50, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
        multi_data_dict = generate_data.generate_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs,
                                                                   n_theta=n_theta, n_basis=n_basis,
                                                                   sig_n=sig_n, seed=seed)
        univ_data_dict = generate_data.generate_univ_sim_and_obs(m=m, n=n, sig_n=sig_n, seed=seed)

        d = SepiaData(t_sim=univ_data_dict['t_sim'], y_sim=univ_data_dict['y_sim'])
        d.transform_xt()
        d.standardize_y()
        self.univ_sim_only_data = d

        d = SepiaData(t_sim=univ_data_dict['t_sim'], y_sim=univ_data_dict['y_sim'], y_obs=univ_data_dict['y_obs'])
        d.transform_xt()
        d.standardize_y()
        self.univ_sim_and_obs_data = d

        d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                      y_ind_sim=multi_data_dict['y_ind_sim'])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        self.multi_sim_only_data = d

        d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                      y_ind_sim=multi_data_dict['y_ind_sim'], y_obs=multi_data_dict['y_obs'],
                      y_ind_obs=multi_data_dict['y_ind_obs'])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        self.multi_sim_and_obs_noD_data = d

        d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                      y_ind_sim=multi_data_dict['y_ind_sim'], y_obs=multi_data_dict['y_obs'],
                      y_ind_obs=multi_data_dict['y_ind_obs'])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        d.create_D_basis('linear')
        self.multi_sim_and_obs_data = d

    def test_univariate_sim_only_saveload(self):
        """
        Tests save/load for univariate sim only model
        """

        print('Testing univariate sim-only save/load...', flush=True)

        data = self.univ_sim_only_data

        model = SepiaModel(data)
        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)
        model.save_model_info(overwrite=True)

        new_model = SepiaModel(data)
        new_model.restore_model_info()

        # check num stuff
        self.assertEqual(model.num.logLik, new_model.num.logLik)
        self.assertTrue(np.allclose(model.num.uw, new_model.num.uw))
        self.assertTrue(np.allclose(model.num.SigWl, new_model.num.SigWl))

        # check param stuff
        for p1, p2 in zip(model.params, new_model.params):
            self.assertTrue(p1.name == p2.name)
            self.assertTrue(np.allclose(p1.fixed, p2.fixed))
            self.assertTrue(np.allclose(p1.val, p2.val))
            self.assertTrue(np.allclose(np.array(p1.mcmc.draws), np.array(p2.mcmc.draws)))
            if p1.name != 'logPost':
                self.assertTrue(np.allclose(p1.mcmc.stepParam, p2.mcmc.stepParam))
                self.assertTrue(p1.mcmc.stepType == p2.mcmc.stepType)
                self.assertTrue(np.allclose(np.array(p1.prior.bounds), np.array(p2.prior.bounds)))
                self.assertTrue(np.allclose(np.array(p1.prior.params), np.array(p2.prior.params)))
                self.assertTrue(p1.prior.dist == p2.prior.dist)

        new_model.do_mcmc(10)


    def test_univariate_sim_and_obs_saveload(self):
        """
        Tests save/load for univariate sim and obs model
        """

        print('Testing univariate sim and obs save/load...', flush=True)

        data = self.univ_sim_and_obs_data

        model = SepiaModel(data)
        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)
        model.save_model_info(overwrite=True)

        new_model = SepiaModel(data)
        new_model.restore_model_info()

        # check num stuff
        self.assertEqual(model.num.logLik, new_model.num.logLik)
        self.assertTrue(np.allclose(model.num.uw, new_model.num.uw))
        self.assertTrue(np.allclose(model.num.SigWl, new_model.num.SigWl))

        # check param stuff
        for p1, p2 in zip(model.params, new_model.params):
            self.assertTrue(p1.name == p2.name)
            self.assertTrue(np.allclose(p1.fixed, p2.fixed))
            self.assertTrue(np.allclose(p1.val, p2.val))
            self.assertTrue(np.allclose(np.array(p1.mcmc.draws), np.array(p2.mcmc.draws)))
            if p1.name != 'logPost':
                self.assertTrue(np.allclose(p1.mcmc.stepParam, p2.mcmc.stepParam))
                self.assertTrue(p1.mcmc.stepType == p2.mcmc.stepType)
                self.assertTrue(np.allclose(np.array(p1.prior.bounds), np.array(p2.prior.bounds)))
                self.assertTrue(np.allclose(np.array(p1.prior.params), np.array(p2.prior.params)))
                self.assertTrue(p1.prior.dist == p2.prior.dist)

        new_model.do_mcmc(10)

    def test_multivariate_sim_only_saveload(self):
        """
        Tests save/load for multivariate sim only model
        """

        print('Testing multivariate sim-only save/load...', flush=True)

        data = self.multi_sim_only_data

        model = SepiaModel(data)
        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)
        model.save_model_info(overwrite=True)

        new_model = SepiaModel(data)
        new_model.restore_model_info()

        # check num stuff
        self.assertEqual(model.num.logLik, new_model.num.logLik)
        self.assertTrue(np.allclose(model.num.vuw, new_model.num.vuw))
        self.assertTrue(np.allclose(model.num.SigWl, new_model.num.SigWl))

        # check param stuff
        for p1, p2 in zip(model.params, new_model.params):
            self.assertTrue(p1.name == p2.name)
            self.assertTrue(np.allclose(p1.fixed, p2.fixed))
            self.assertTrue(np.allclose(p1.val, p2.val))
            self.assertTrue(np.allclose(np.array(p1.mcmc.draws), np.array(p2.mcmc.draws)))
            if p1.name != 'logPost':
                self.assertTrue(np.allclose(p1.mcmc.stepParam, p2.mcmc.stepParam))
                self.assertTrue(p1.mcmc.stepType == p2.mcmc.stepType)
                self.assertTrue(np.allclose(np.array(p1.prior.bounds), np.array(p2.prior.bounds)))
                self.assertTrue(np.allclose(np.array(p1.prior.params), np.array(p2.prior.params)))
                self.assertTrue(p1.prior.dist == p2.prior.dist)

        new_model.do_mcmc(10)

    def test_multivariate_sim_and_obs_saveload(self):
        """
        Tests save/load for multivariate sim and obs model
        """

        print('Testing multivariate sim and obs save/load...', flush=True)

        data = self.multi_sim_and_obs_data

        model = SepiaModel(data)
        model.tune_step_sizes(20, 5)
        model.do_mcmc(20)
        model.save_model_info(overwrite=True)

        new_model = SepiaModel(data)
        new_model.restore_model_info()

        # check num stuff
        self.assertEqual(model.num.logLik, new_model.num.logLik)
        self.assertTrue(np.allclose(model.num.vuw, new_model.num.vuw))
        self.assertTrue(np.allclose(model.num.SigWl, new_model.num.SigWl))

        # check param stuff
        for p1, p2 in zip(model.params, new_model.params):
            self.assertTrue(p1.name == p2.name)
            self.assertTrue(np.allclose(p1.fixed, p2.fixed))
            self.assertTrue(np.allclose(p1.val, p2.val))
            self.assertTrue(np.allclose(np.array(p1.mcmc.draws), np.array(p2.mcmc.draws)))
            if p1.name != 'logPost':
                self.assertTrue(np.allclose(p1.mcmc.stepParam, p2.mcmc.stepParam))
                self.assertTrue(p1.mcmc.stepType == p2.mcmc.stepType)
                self.assertTrue(np.allclose(np.array(p1.prior.bounds), np.array(p2.prior.bounds)))
                self.assertTrue(np.allclose(np.array(p1.prior.params), np.array(p2.prior.params)))
                self.assertTrue(p1.prior.dist == p2.prior.dist)

        new_model.do_mcmc(10)

    def test_multivariate_sim_and_obs_noD_saveload(self):
        """
        Tests save/load for multivariate sim and obs model no discrep
        """

        print('Testing multivariate sim and obs no discrep save/load...', flush=True)

        data = self.multi_sim_and_obs_noD_data

        model = SepiaModel(data)
        model.tune_step_sizes(20, 5)
        model.do_mcmc(20)
        model.save_model_info(overwrite=True)

        new_model = SepiaModel(data)
        new_model.restore_model_info()

        # check num stuff
        self.assertEqual(model.num.logLik, new_model.num.logLik)
        self.assertTrue(np.allclose(model.num.vuw, new_model.num.vuw))
        self.assertTrue(np.allclose(model.num.SigWl, new_model.num.SigWl))

        # check param stuff
        for p1, p2 in zip(model.params, new_model.params):
            self.assertTrue(p1.name == p2.name)
            self.assertTrue(np.allclose(p1.fixed, p2.fixed))
            self.assertTrue(np.allclose(p1.val, p2.val))
            self.assertTrue(np.allclose(np.array(p1.mcmc.draws), np.array(p2.mcmc.draws)))
            if p1.name != 'logPost':
                self.assertTrue(np.allclose(p1.mcmc.stepParam, p2.mcmc.stepParam))
                self.assertTrue(p1.mcmc.stepType == p2.mcmc.stepType)
                self.assertTrue(np.allclose(np.array(p1.prior.bounds), np.array(p2.prior.bounds)))
                self.assertTrue(np.allclose(np.array(p1.prior.params), np.array(p2.prior.params)))
                self.assertTrue(p1.prior.dist == p2.prior.dist)

        new_model.do_mcmc(10)





