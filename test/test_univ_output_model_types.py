import unittest
import numpy as np



from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaPlot import *
from sepia.SepiaPredict import *

np.random.seed(42)

# Check that all model types can be set up and various functions can be run without errors - does not assess output values

class TestModelTypes(unittest.TestCase):

    def test_univariate_sim_only_x_only(self):
        m = 20
        x = np.random.uniform(-1, 3, (m, 3))
        y = np.random.normal(size=(m, 1))

        data = SepiaData(x_sim=x, y_sim=y)
        data.transform_xt()
        data.standardize_y()

        model = SepiaModel(data)
        model.log_prior()
        model.logLik()
        model.logPost()
        model.print_prior_info()
        model.print_value_info()
        model.print_mcmc_info()
        model.tune_step_sizes(10, 10)
        model.do_mcmc(50)

        samples = model.get_samples()

        mcmc_trace(samples)
        param_stats(samples)

        pred = SepiaEmulatorPrediction(x_pred=x, samples=samples, model=model)
        pred.get_w()
        pred.get_y()
        pred.get_mu_sigma()

        # TODO fails with x only
        #xpred = SepiaXvalEmulatorPrediction(model=model)
        #pred.get_w()
        #pred.get_y()
        #pred.get_mu_sigma()

    def test_univariate_sim_only_x_only_cat(self):
        m = 20
        x = np.concatenate([np.random.uniform(-1, 3, (m, 3)), 1+np.random.choice(3,size=(m, 1))], axis=1)
        y = np.random.normal(size=(m, 1))
        cat_inds = [0, 0, 0, 3]

        data = SepiaData(x_sim=x, y_sim=y, x_cat_ind=cat_inds)
        data.transform_xt()
        data.standardize_y()

        model = SepiaModel(data)
        model.log_prior()
        model.logLik()
        model.logPost()
        model.print_prior_info()
        model.print_value_info()
        model.print_mcmc_info()
        model.tune_step_sizes(10, 10)
        model.do_mcmc(50)

        samples = model.get_samples()

        mcmc_trace(samples)
        param_stats(samples)

        pred = SepiaEmulatorPrediction(x_pred=x, samples=samples, model=model)
        pred.get_w()
        pred.get_y()
        pred.get_mu_sigma()

        # TODO fails with x only
        #xpred = SepiaXvalEmulatorPrediction(model=model)
        #pred.get_w()
        #pred.get_y()
        #pred.get_mu_sigma()

    def test_univariate_sim_only_t_only(self):
        m = 20
        x = np.random.uniform(-1, 3, (m, 3))
        y = np.random.normal(size=(m, 1))

        data = SepiaData(x_sim=x, y_sim=y)
        data.transform_xt()
        data.standardize_y()

        model = SepiaModel(data)
        model.log_prior()
        model.logLik()
        model.logPost()
        model.print_prior_info()
        model.print_value_info()
        model.print_mcmc_info()
        model.tune_step_sizes(10, 10)
        model.do_mcmc(50)

        samples = model.get_samples()

        mcmc_trace(samples)
        param_stats(samples)

        pred = SepiaEmulatorPrediction(x_pred=x, samples=samples, model=model)
        pred.get_w()
        pred.get_y()
        pred.get_mu_sigma()

        # TODO fails with x only
        #xpred = SepiaXvalEmulatorPrediction(model=model)
        #pred.get_w()
        #pred.get_y()
        #pred.get_mu_sigma()

