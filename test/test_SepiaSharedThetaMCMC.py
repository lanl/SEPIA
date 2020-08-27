import unittest
import numpy as np

import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaSharedThetaModels import SepiaSharedThetaModels

np.random.seed(42)

class SepiaSharedThetaMCMCTestCase(unittest.TestCase):

    def setUp(self, m=100, n=1, nt_sim=50, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
        n_shared = 3
        self.shared_idx = np.array([[1, 1, 1], [2, -1, 2]])
        multi_data_list = []
        univ_data_list = []
        for si in range(n_shared):
            multi_data_dict = generate_data.generate_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs,
                                                                       n_theta=n_theta, n_basis=n_basis,
                                                                       sig_n=sig_n, seed=seed)
            univ_data_dict = generate_data.generate_univ_sim_and_obs(m=m, n=n, sig_n=sig_n, seed=seed)

            d = SepiaData(t_sim=univ_data_dict['t_sim'], y_sim=univ_data_dict['y_sim'], y_obs=univ_data_dict['y_obs'])
            d.transform_xt()
            d.standardize_y()
            univ_data_list.append(d)

            d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                          y_ind_sim=multi_data_dict['y_ind_sim'], y_obs=multi_data_dict['y_obs'],
                          y_ind_obs=multi_data_dict['y_ind_obs'])
            d.transform_xt()
            d.standardize_y()
            d.create_K_basis(5)
            d.create_D_basis('constant')
            multi_data_list.append(d)

        self.univ_model_list = [SepiaModel(d) for d in univ_data_list]
        self.multi_model_list = [SepiaModel(d) for d in multi_data_list]


    def test_univariate_sim_and_obs_shared_theta_mcmc(self):
        """
        Tests mcmc for univariate sim and obs model with shared thetas TODO this breaks, univ models have one t but shared idx has three, should add error check to setup
        """

        print('Testing univariate sim and obs shared theta SepiaMCMC...', flush=True)

        model = SepiaSharedThetaModels(model_list=self.univ_model_list, shared_theta_inds=self.shared_idx)

        model.do_mcmc(100)
        samples_dicts = [{p.name: p.mcmc_to_array() for p in model.params.mcmcList} for model in self.univ_model_list]

    def test_multivariate_sim_and_obs_shared_theta_mcmc(self):
        """
        Tests mcmc for multivariate sim and obs model with shared thetas
        """

        print('Testing multivariate sim and obs shared theta SepiaMCMC...', flush=True)

        model = SepiaSharedThetaModels(model_list=self.multi_model_list, shared_theta_inds=self.shared_idx)

        model.do_mcmc(100)
        samples_dicts = [{p.name: p.mcmc_to_array() for p in model.params.mcmcList} for model in self.multi_model_list]





