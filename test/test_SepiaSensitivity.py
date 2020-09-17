import unittest
import numpy as np

import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaSensitivity import sensitivity

np.random.seed(42)

class SepiaSensitivityTestCase(unittest.TestCase):

    def setUp(self, m=20, n=1, nt_sim=30, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
        multi_data_dict = generate_data.generate_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs,
                                                                   n_theta=n_theta, n_basis=n_basis,
                                                                   sig_n=sig_n, seed=seed)
        univ_data_dict = generate_data.generate_univ_sim_and_obs(m=m, n=n, sig_n=sig_n, seed=seed)

        d = SepiaData(t_sim=univ_data_dict['t_sim'], y_sim=univ_data_dict['y_sim'])
        d.transform_xt()
        d.standardize_y()
        self.univ_sim_only_model = SepiaModel(d)

        d = SepiaData(t_sim=univ_data_dict['t_sim'], y_sim=univ_data_dict['y_sim'], y_obs=univ_data_dict['y_obs'])
        d.transform_xt()
        d.standardize_y()
        self.univ_sim_and_obs_model = SepiaModel(d)

        d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                      y_ind_sim=multi_data_dict['y_ind_sim'])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        self.multi_sim_only_model = SepiaModel(d)

        t = np.concatenate([multi_data_dict['t_sim'], np.random.choice(range(1,5), (m, 1), replace=True)], axis=1)
        d = SepiaData(t_sim=t, y_sim=multi_data_dict['y_sim'], y_ind_sim=multi_data_dict['y_ind_sim'], t_cat_ind=[0,0,0,4])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        self.multi_sim_only_catind_model = SepiaModel(d)


    def test_univariate_sim_only_sensitivity(self):
        """
        Tests sensitivity for univariate sim only model
        """

        print('Testing univariate sim-only SepiaSensitivity...', flush=True)

        model = self.univ_sim_only_model

        model.do_mcmc(20)
        samples_dict = model.get_samples(20)
        sens = sensitivity(model, jelist=[(0, 1)])
        sens = sensitivity(model, varlist='all')
        sens = sensitivity(model, samples_dict)
        sens = sensitivity(model)
        sens = sensitivity(model, ngrid=10)
        sens = sensitivity(model, option='median')
        sens = sensitivity(model, option='samples')
        sens = sensitivity(model, option=model.get_samples(numsamples=10, flat=True))


    def test_multivariate_sim_only_sensitivity(self):
        """
        Tests sensitivity for multivariate sim only model
        """

        print('Testing multivariate sim-only SepiaSensitivity...', flush=True)

        model = self.multi_sim_only_model

        model.do_mcmc(20)
        samples_dict = model.get_samples(20)
        sens = sensitivity(model, jelist=[(0, 1), (0, 2)])
        sens = sensitivity(model, varlist='all')
        sens = sensitivity(model, samples_dict)
        sens = sensitivity(model)
        sens = sensitivity(model, varlist=[(0, 1), (0, 2)])
        sens = sensitivity(model, option='median')
        sens = sensitivity(model, option='samples')
        sens = sensitivity(model, option=model.get_samples(numsamples=10, flat=True))

    def test_multivariate_sim_only_catind_sensitivity(self):
        """
        Tests sensitivity for multivariate sim only model with cat_ind
        """

        print('Testing multivariate sim-only cat ind SepiaSensitivity...', flush=True)

        model = self.multi_sim_only_catind_model

        model.do_mcmc(20)
        samples_dict = model.get_samples(20)
        sens = sensitivity(model, jelist=[(0, 1), (0, 2)])
        sens = sensitivity(model, varlist='all')
        sens = sensitivity(model, samples_dict)
        sens = sensitivity(model)
        sens = sensitivity(model, varlist=[(0, 1), (0, 2)])
        sens = sensitivity(model, option='median')
        sens = sensitivity(model, option='samples')
        sens = sensitivity(model, option=model.get_samples(numsamples=10, flat=True))




