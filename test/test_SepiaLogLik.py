import unittest
import numpy as np

import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel

np.random.seed(42)

class SepiaLogLikTestCase(unittest.TestCase):

    def setUp(self, m=100, n=1, nt_sim=50, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
        self.multi_data_dict = generate_data.generate_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs,
                                                                        n_theta=n_theta, n_basis=n_basis,
                                                                        sig_n=sig_n, seed=seed)
        self.univ_data_dict = generate_data.generate_univ_sim_and_obs(m=m, n=n, sig_n=sig_n, seed=seed)

    def test_univariate_sim_only_lik(self):
        """
        Tests log lik for univariate sim only model
        """

        d = SepiaData(t_sim=self.univ_data_dict['t_sim'], y_sim=self.univ_data_dict['y_sim'])
        print('Testing univariate sim-only SepiaLogLik...', flush=True)
        print(d, flush=True)

        d.transform_xt()
        d.standardize_y()
        model = SepiaModel(d)

        model.logLik()

        for param in model.params.mcmcList:
            for cindex in range(int(np.prod(param.val_shape))):
                model.logLik(cvar=param.name, cindex=cindex)


    def test_univariate_sim_and_obs_lik(self):
        """
        Tests log lik for univariate sim and obs model
        """

        d = SepiaData(t_sim=self.univ_data_dict['t_sim'], y_sim=self.univ_data_dict['y_sim'],
                      y_obs=self.univ_data_dict['y_obs'])
        print('Testing univariate sim and obs SepiaLogLik...', flush=True)
        print(d, flush=True)

        # Do explicit transformation
        d.transform_xt()
        d.standardize_y()
        model = SepiaModel(d)

        model.logLik()

        for param in model.params.mcmcList:
            for cindex in range(int(np.prod(param.val_shape))):
                model.logLik(cvar=param.name, cindex=cindex)


    def test_multivariate_sim_only_lik(self):
        """
        Tests log lik for multivariate sim only model
        """

        d = SepiaData(t_sim=self.multi_data_dict['t_sim'], y_sim=self.multi_data_dict['y_sim'],
                      y_ind_sim=self.multi_data_dict['y_ind_sim'])
        print('Testing multivariate sim-only SepiaLogLik...', flush=True)
        print(d, flush=True)

        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        model = SepiaModel(d)

        model.logLik()

        for param in model.params.mcmcList:
            for cindex in range(int(np.prod(param.val_shape))):
                model.logLik(cvar=param.name, cindex=cindex)


    def test_multivariate_sim_and_obs_lik(self):
        """
        Tests log lik for multivariate sim and obs model
        """

        d = SepiaData(t_sim=self.multi_data_dict['t_sim'], y_sim=self.multi_data_dict['y_sim'],
                      y_ind_sim=self.multi_data_dict['y_ind_sim'], y_obs=self.multi_data_dict['y_obs'],
                      y_ind_obs=self.multi_data_dict['y_ind_obs'])
        print('Testing multivariate sim-only SepiaLogLik...', flush=True)
        print(d, flush=True)

        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        d.create_D_basis('linear')
        model = SepiaModel(d)

        model.logLik()

        for param in model.params.mcmcList:
            for cindex in range(int(np.prod(param.val_shape))):
                model.logLik(cvar=param.name, cindex=cindex)


    def test_multivariate_sim_and_obs_noD_lik(self):
        """
        Tests log lik for multivariate sim and obs model no discrep
        """

        d = SepiaData(t_sim=self.multi_data_dict['t_sim'], y_sim=self.multi_data_dict['y_sim'],
                      y_ind_sim=self.multi_data_dict['y_ind_sim'], y_obs=self.multi_data_dict['y_obs'],
                      y_ind_obs=self.multi_data_dict['y_ind_obs'])
        print('Testing multivariate sim-only SepiaLogLik...', flush=True)
        print(d, flush=True)

        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        model = SepiaModel(d)

        model.logLik()

        for param in model.params.mcmcList:
            for cindex in range(int(np.prod(param.val_shape))):
                model.logLik(cvar=param.name, cindex=cindex)





