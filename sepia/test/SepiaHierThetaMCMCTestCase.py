import unittest
import numpy as np
import copy

from sepia.Examples import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
from sepia.SepiaHierarchicalThetaModels import SepiaHierarchicalThetaModels

np.random.seed(42)

class SepiaHierThetaMCMCTestCase(unittest.TestCase):

    def setUp(self, m=100, n=1, nt_sim=50, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
        n_hier = 3
        self.hier_idx = np.array([[0, 0, 0]])
        #self.hier_idx = np.array([[1, 1, 1], [2, -1, 2]]) # TODO this fails for multivariate; cant use for univariate now
        multi_data_list = []
        univ_data_list = []
        for si in range(n_hier):
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

        self.univ_model_list = [setup_model(d) for d in univ_data_list]
        self.multi_model_list = [setup_model(d) for d in multi_data_list]

    # TODO: univariate test fails! It's because univ setup only has one theta...
    def test_univariate_sim_and_obs_shared_theta_mcmc(self):
        """
        Tests mcmc for univariate sim and obs model with hier thetas
        """

        print('Testing univariate sim and obs hier theta SepiaMCMC...', flush=True)

        model = SepiaHierarchicalThetaModels(model_list=self.univ_model_list, hier_theta_inds=self.hier_idx)

        model.do_mcmc(100)

    def test_multivariate_sim_and_obs_shared_theta_mcmc(self):
        """
        Tests mcmc for multivariate sim and obs model with hier thetas
        """

        print('Testing multivariate sim and obs hier theta SepiaMCMC...', flush=True)

        model = SepiaHierarchicalThetaModels(model_list=self.multi_model_list, hier_theta_inds=self.hier_idx)

        model.do_mcmc(100)





