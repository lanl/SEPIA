import unittest
import numpy as np

import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaPredict import SepiaXvalEmulatorPrediction, SepiaEmulatorPrediction, SepiaFullPrediction

np.random.seed(42)

class SepiaPredictionTestCase(unittest.TestCase):

    def setUp(self, m=100, n=1, nt_sim=50, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
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

        d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                      y_ind_sim=multi_data_dict['y_ind_sim'], y_obs=multi_data_dict['y_obs'],
                      y_ind_obs=multi_data_dict['y_ind_obs'])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        self.multi_sim_and_obs_noD_model = SepiaModel(d)

        d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
                      y_ind_sim=multi_data_dict['y_ind_sim'], y_obs=multi_data_dict['y_obs'],
                      y_ind_obs=multi_data_dict['y_ind_obs'])
        d.transform_xt()
        d.standardize_y()
        d.create_K_basis(5)
        d.create_D_basis('linear')
        self.multi_sim_and_obs_model = SepiaModel(d)


    def test_univariate_sim_only_pred(self):
        """
        Tests pred for univariate sim only model
        """

        print('Testing univariate sim-only Sepia prediction...', flush=True)

        model = self.univ_sim_only_model

        model.do_mcmc(50)
        samples = model.get_samples(numsamples=5)
        pred = SepiaEmulatorPrediction(x_pred=model.data.sim_data.x, t_pred=model.data.sim_data.t,
                                       samples=samples, model=model)
        pred.get_w()
        pred.get_y()
        cvpred = SepiaXvalEmulatorPrediction(samples=samples, model=model)
        cvpred.get_w()
        cvpred.get_y()


    def test_univariate_sim_and_obs_pred(self):
        """
        Tests pred for univariate sim and obs model
        """

        print('Testing univariate sim and obs Sepia prediction...', flush=True)

        model = self.univ_sim_and_obs_model

        model.do_mcmc(50)
        samples = model.get_samples(numsamples=5)
        pred = SepiaEmulatorPrediction(x_pred=model.data.sim_data.x, t_pred=model.data.sim_data.t,
                                       samples=samples, model=model)
        pred.get_w()
        pred.get_y()
        cvpred = SepiaXvalEmulatorPrediction(samples=samples, model=model)
        cvpred.get_w()
        cvpred.get_y()

        pred = SepiaFullPrediction(x_pred=model.data.sim_data.x, t_pred=model.data.sim_data.t,
                                       samples=samples, model=model)
        pred.get_u_v()
        pred.get_ysim()
        pred.get_ysim(as_obs=True)

    def test_multivariate_sim_only_pred(self):
        """
        Tests pred for multivariate sim only model
        """

        print('Testing multivariate sim-only Sepia prediction...', flush=True)

        model = self.multi_sim_only_model

        model.do_mcmc(50)
        samples = model.get_samples(numsamples=5)
        pred = SepiaEmulatorPrediction(x_pred=model.data.sim_data.x, t_pred=model.data.sim_data.t,
                                       samples=samples, model=model)
        pred.get_w()
        pred.get_y()
        cvpred = SepiaXvalEmulatorPrediction(samples=samples, model=model)
        cvpred.get_w()
        cvpred.get_y()

    def test_multivariate_sim_and_obs_pred(self):
        """
        Tests pred for multivariate sim and obs model
        """

        print('Testing multivariate sim and obs Sepia prediction...', flush=True)

        model = self.multi_sim_and_obs_model

        model.do_mcmc(50)
        samples = model.get_samples(numsamples=5)
        pred = SepiaEmulatorPrediction(x_pred=model.data.sim_data.x, t_pred=model.data.sim_data.t,
                                       samples=samples, model=model)
        pred.get_w()
        pred.get_y()
        cvpred = SepiaXvalEmulatorPrediction(samples=samples, model=model)
        cvpred.get_w()
        cvpred.get_y()

        pred = SepiaFullPrediction(x_pred=model.data.sim_data.x, t_pred=model.data.sim_data.t,
                                       samples=samples, model=model)
        pred.get_u_v()
        pred.get_ysim()
        pred.get_ysim(as_obs=True)

    def test_multivariate_sim_and_obs_noD_pred(self):
        """
        Tests pred for multivariate sim and obs model no discrep
        """

        print('Testing multivariate sim and obs no discrep Sepia prediction...', flush=True)

        model = self.multi_sim_and_obs_noD_model

        model.do_mcmc(50)
        samples = model.get_samples(numsamples=5)
        pred = SepiaEmulatorPrediction(x_pred=model.data.sim_data.x, t_pred=model.data.sim_data.t,
                                       samples=samples, model=model)
        pred.get_w()
        pred.get_y()
        cvpred = SepiaXvalEmulatorPrediction(samples=samples, model=model)
        cvpred.get_w()
        cvpred.get_y()

        pred = SepiaFullPrediction(x_pred=model.data.sim_data.x, t_pred=model.data.sim_data.t,
                                       samples=samples, model=model)
        pred.get_u_v()
        pred.get_ysim()
        pred.get_ysim(as_obs=True)



