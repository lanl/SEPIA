import unittest
import numpy as np

import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
from sepia.SepiaPredict import SepiaXvalEmulatorPrediction

np.random.seed(42)

class SepiaFullPredictionTestCase(unittest.TestCase):

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

    def test_univariate_fullpred(self):
        """
        Tests full prediction for univariate sim and obs model
        """

        print('Testing univariate sim-only SepiaFullPrediction...', flush=True)

        model = self.univ_sim_and_obs_model

        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)

        # TODO replace with FullPrediction calls
        pred_samples = model.get_samples(numsamples=7)


    def test_multivariate_fullpred(self):
        """
        Tests full prediction for multivariate sim only model
        """

        print('Testing multivariate sim-only SepiaFullPrediction...', flush=True)

        model = self.multi_sim_and_obs_model

        model.tune_step_sizes(10, 10)
        model.do_mcmc(10)

        # TODO replace with FullPrediction calls
        pred_samples = model.get_samples(numsamples=7)



