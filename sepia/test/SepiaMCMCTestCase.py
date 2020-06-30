import unittest
import numpy as np
import copy

from sepia.Examples import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model

np.random.seed(42)

class SepiaMCMCTestCase(unittest.TestCase):

    def setUp(self, m=100, n=1, nt_sim=50, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
        multi_data_dict = generate_data.generate_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs,
                                                                   n_theta=n_theta, n_basis=n_basis,
                                                                   sig_n=sig_n, seed=seed)
        univ_data_dict = generate_data.generate_univ_sim_and_obs(m=m, n=n, sig_n=sig_n, seed=seed)

        d = SepiaData(t_sim=univ_data_dict['t_sim'], y_sim=univ_data_dict['y_sim'])
        d.transform_xt()
        d.standardize_y()
        self.univ_sim_only_model = setup_model(d)

        d = SepiaData(t_sim=univ_data_dict['t_sim'], y_sim=univ_data_dict['y_sim'], y_obs=univ_data_dict['y_obs'])
        d.transform_xt()
        d.standardize_y()
        self.univ_sim_and_obs_model = setup_model(d)

        d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                      y_ind_sim=multi_data_dict['y_ind_sim'])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        self.multi_sim_only_model = setup_model(d)

        d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                      y_ind_sim=multi_data_dict['y_ind_sim'], y_obs=multi_data_dict['y_obs'],
                      y_ind_obs=multi_data_dict['y_ind_obs'])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        self.multi_sim_and_obs_noD_model = setup_model(d)

        d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                      y_ind_sim=multi_data_dict['y_ind_sim'], y_obs=multi_data_dict['y_obs'],
                      y_ind_obs=multi_data_dict['y_ind_obs'])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        d.create_D_basis('linear')
        self.multi_sim_and_obs_model = setup_model(d)

    def test_univariate_sim_only_mcmc(self):
        """
        Tests MCMC for univariate sim only model
        """

        print('Testing univariate sim-only SepiaMCMC...', flush=True)

        model = self.univ_sim_only_model

        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)
        samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}
        samples_dict = {p.name: p.mcmc_to_array(trim=10, flat=True) for p in model.params.mcmcList}
        samples_dict = {p.name: p.mcmc_to_array(sampleset=np.arange(50)) for p in model.params.mcmcList}

    def test_univariate_sim_and_obs_mcmc(self):
        """
        Tests mcmc for univariate sim and obs model
        """

        print('Testing univariate sim and obs SepiaMCMC...', flush=True)

        model = self.univ_sim_and_obs_model

        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)
        samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}

    def test_multivariate_sim_only_mcmc(self):
        """
        Tests MCMC for multivariate sim only model
        """

        print('Testing multivariate sim-only SepiaMCMC...', flush=True)

        model = self.multi_sim_only_model

        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)
        samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}

    def test_multivariate_sim_and_obs_mcmc(self):
        """
        Tests mcmc for multivariate sim and obs model
        """

        print('Testing multivariate sim and obs SepiaMCMC...', flush=True)

        model = self.multi_sim_and_obs_model

        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)
        samples_dict = {p.name: p.mcmc_to_array(untransform_theta=True) for p in model.params.mcmcList}

    def test_multivariate_sim_and_obs_noD_mcmc(self):
        """
        Tests mcmc for multivariate sim and obs model no discrep
        """

        print('Testing multivariate sim and obs no discrep SepiaMCMC...', flush=True)

        model = self.multi_sim_and_obs_noD_model

        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)
        samples_dict = {p.name: p.mcmc_to_array(untransform_theta=True) for p in model.params.mcmcList}





