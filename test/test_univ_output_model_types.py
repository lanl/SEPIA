import unittest
import numpy as np

import matplotlib
matplotlib.use('Agg')

from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaPlot import *
from sepia.SepiaPredict import *

np.random.seed(42)

# Check that all model types can be set up and various functions can be run without errors.
# Does not assess output values, more to ensure users won't run into errors for using a specific model type.

def call_data_methods(data):
    data.transform_xt()
    data.standardize_y()
    # Note: we are calling stuff that doesn't make sense for univ to make sure it doesn't break
    data.create_K_basis(2)
    data.create_D_basis('linear')
    data.plot_K_basis()
    data.plot_K_weights()
    data.plot_u_w_pairs()
    data.plot_K_residuals()
    data.plot_data()

def call_model_methods(model):
    model.log_prior()
    model.logLik()
    model.logPost()
    model.print_prior_info()
    model.print_value_info()
    model.print_mcmc_info()
    model.tune_step_sizes(10, 10)
    model.do_mcmc(100)

def call_plot_functions(model):
    samples = model.get_samples()
    theta_pairs(samples)
    mcmc_trace(samples)
    param_stats(samples)
    rho_box_plots(model)
    plot_acf(model, 10)


class TestUnivOutputModelTypes(unittest.TestCase):

    def test_sim_only_x_only(self):
        m = 20
        x = np.random.uniform(-1, 3, (m, 3))
        y = np.random.normal(size=(m, 1))

        data = SepiaData(x_sim=x, y_sim=y)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(x_pred=x, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

    def test_sim_only_x_only_cat(self):
        m = 20
        x = np.concatenate([np.random.uniform(-1, 3, (m, 3)), 1+np.random.choice(3,size=(m, 1))], axis=1)
        y = np.random.normal(size=(m, 1))
        cat_inds = [0, 0, 0, 3]

        data = SepiaData(x_sim=x, y_sim=y, x_cat_ind=cat_inds)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(x_pred=x, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

    def test_sim_only_t_only(self):
        m = 20
        t = np.random.uniform(-1, 3, (m, 3))
        y = np.random.normal(size=(m, 1))

        data = SepiaData(t_sim=t, y_sim=y)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

    def test_sim_only_t_only_cat(self):
        m = 20
        t = np.concatenate([np.random.uniform(-1, 3, (m, 3)), 1+np.random.choice(3,size=(m, 1))], axis=1)
        y = np.random.normal(size=(m, 1))
        cat_inds = [0, 0, 0, 3]

        data = SepiaData(t_sim=t, y_sim=y, t_cat_ind=cat_inds)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

    def test_sim_only_x_and_t(self):
        m = 20
        x = np.random.uniform(-1, 3, (m, 3))
        t = np.random.uniform(-1, 3, (m, 2))
        y = np.random.normal(size=(m, 1))

        data = SepiaData(x_sim=x, t_sim=t, y_sim=y)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(x_pred=x, t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

    def test_sim_only_x_and_t_cat(self):
        m = 20
        x = np.concatenate([np.random.uniform(-1, 3, (m, 3)), 1+np.random.choice(3,size=(m, 1))], axis=1)
        t = np.concatenate([np.random.uniform(-1, 3, (m, 2)), 1 + np.random.choice(4, size=(m, 1))], axis=1)
        y = np.random.normal(size=(m, 1))
        x_cat_ind = [0, 0, 0, 3]
        t_cat_ind = [0, 0, 4]

        data = SepiaData(x_sim=x, t_sim=t, y_sim=y, x_cat_ind=x_cat_ind, t_cat_ind=t_cat_ind)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(x_pred=x, t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

    def test_sim_and_obs_t_only(self):
        m = 20
        n = 10
        t = np.random.uniform(-1, 3, (m, 3))
        y = np.random.normal(size=(m, 1))
        y2 = np.random.normal(size=(n, 1))

        data = SepiaData(t_sim=t, y_sim=y, y_obs=y2)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

        pred = SepiaFullPrediction(t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_u_v()
        pred.get_mu_sigma()
        pred.get_ysim()
        pred.get_ysim(as_obs=True)
        pred.get_ysim(as_obs=True, std=True)
        pred.get_yobs()
        pred.get_yobs(as_obs=True)
        pred.get_yobs(as_obs=True, std=True)

    def test_sim_and_obs_t_only_cat(self):
        m = 20
        n = 10
        t = np.concatenate([np.random.uniform(-1, 3, (m, 3)), 1+np.random.choice(3,size=(m, 1))], axis=1)
        y = np.random.normal(size=(m, 1))
        y2 = np.random.normal(size=(n, 1))
        cat_inds = [0, 0, 0, 3]

        data = SepiaData(t_sim=t, y_sim=y, t_cat_ind=cat_inds, y_obs=y2)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

        pred = SepiaFullPrediction(t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_u_v()
        pred.get_mu_sigma()
        pred.get_ysim()
        pred.get_ysim(as_obs=True)
        pred.get_ysim(as_obs=True, std=True)
        pred.get_yobs()
        pred.get_yobs(as_obs=True)
        pred.get_yobs(as_obs=True, std=True)

    def test_sim_and_obs_x_and_t(self):
        m = 20
        n = 10
        x = np.random.uniform(-1, 3, (m, 3))
        x2 = np.random.uniform(-1, 3, (n, 3))
        t = np.random.uniform(-1, 3, (m, 2))
        y = np.random.normal(size=(m, 1))
        y2 = np.random.normal(size=(n, 1))

        data = SepiaData(x_sim=x, t_sim=t, y_sim=y, x_obs=x2, y_obs=y2)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(x_pred=x, t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

        pred = SepiaFullPrediction(x_pred=x, t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_u_v()
        pred.get_mu_sigma()
        pred.get_ysim()
        pred.get_ysim(as_obs=True)
        pred.get_ysim(as_obs=True, std=True)
        pred.get_yobs()
        pred.get_yobs(as_obs=True)
        pred.get_yobs(as_obs=True, std=True)

    def test_sim_and_obs_x_and_t_cat(self):
        m = 20
        n = 10
        x = np.concatenate([np.random.uniform(-1, 3, (m, 3)), 1 + np.random.choice(3,size=(m, 1))], axis=1)
        x2 = np.concatenate([np.random.uniform(-1, 3, (n, 3)), 1 + np.random.choice(3, size=(n, 1))], axis=1)
        t = np.concatenate([np.random.uniform(-1, 3, (m, 2)), 1 + np.random.choice(4, size=(m, 1))], axis=1)
        y = np.random.normal(size=(m, 1))
        y2 = np.random.normal(size=(n, 1))
        x_cat_ind = [0, 0, 0, 3]
        t_cat_ind = [0, 0, 4]

        data = SepiaData(x_sim=x, t_sim=t, y_sim=y, x_obs=x2, y_obs=y2, x_cat_ind=x_cat_ind, t_cat_ind=t_cat_ind)
        call_data_methods(data)

        model = SepiaModel(data)
        call_model_methods(model)
        call_plot_functions(model)

        samples = model.get_samples()
        pred = SepiaEmulatorPrediction(x_pred=x, t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_w()
        pred.get_y()
        pred.get_y(std=True)
        pred.get_mu_sigma()

        xpred = SepiaXvalEmulatorPrediction(samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        xpred.get_w()
        xpred.get_y()
        xpred.get_y(std=True)
        xpred.get_mu_sigma()

        pred = SepiaFullPrediction(x_pred=x, t_pred=t, samples=samples, model=model, addResidVar=True, storeMuSigma=True)
        pred.get_u_v()
        pred.get_mu_sigma()
        pred.get_ysim()
        pred.get_ysim(as_obs=True)
        pred.get_ysim(as_obs=True, std=True)
        pred.get_yobs()
        pred.get_yobs(as_obs=True)
        pred.get_yobs(as_obs=True, std=True)

