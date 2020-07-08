import unittest
import numpy as np
import scipy.io
from time import time
import scipy.stats
import os

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
import matlab.engine
from sepia.SepiaPredict import SepiaEmulatorPrediction, SepiaFullPrediction
from setup_test_cases import *

"""
NOTE: requires matlab.engine.

To install at command line:
> source activate <sepia conda env name>
> cd <matlabroot>/extern/engines/python
> python setup.py install
"""

class SepiaPredictTestCase(unittest.TestCase):
    """
    Checks Prediction results between matlab and python.
    Run files in matlab/ dir to generate data prior to running these tests.
    """

    def test_predict_univ_sim_only(self):

        print('starting test_predict_univ_sim_only', flush=True)

        show_figs = True
        seed = 42.
        n_mcmc = 1000
        m = 300
        n_pred = 5

        # call function to do matlab setup/sampling
        model, matlab_output = setup_univ_sim_only(m=m, seed=seed, n_mcmc=n_mcmc, n_pred=n_pred)
        mcmc_time_mat = matlab_output['mcmc_time']
        mcmc_mat = matlab_output['mcmc']
        mcmc_mat = {k: np.array(mcmc_mat[k]) for k in mcmc_mat.keys()}

        # do python sampling
        np.random.seed(int(seed))
        t_start = time()
        model.do_mcmc(n_mcmc)
        t_end = time()

        print('Python mcmc time %0.3g s' % (t_end - t_start), flush=True)
        print('Matlab mcmc time %0.3g s' % mcmc_time_mat, flush=True)

        samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}
        samples_dict['logPost'] = np.array(model.params.lp.mcmc.draws).reshape((-1, 1))

        np.random.seed(int(seed))
        psamps = model.get_samples(0, sampleset=range(n_pred), flat=True)
        pred = SepiaEmulatorPrediction(np.array([0.5, 0.5]).reshape(1,2), psamps, model, storeMuSigma=True)
        print('Samples are:')
        print(pred.w.squeeze())
        print('Matlab Samples are:')
        print(np.array(matlab_output['pred_w']).squeeze())

        print('Mu are:')
        print(pred.mu.squeeze())
        print('Matlab Mu are:')
        print(np.array(matlab_output['pred_Myhat']).squeeze())

        print('Sigma are:')
        print(pred.sigma.squeeze())
        print('Matlab Sigma are:')
        print(np.array(matlab_output['pred_Syhat']).squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_w']).squeeze(), pred.w.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_Myhat']).squeeze(), pred.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_Syhat']).squeeze(), pred.sigma.squeeze()))

        # Prediction with multiple realizations
        np.random.seed(42)
        sampleset = np.arange(100, 1001, 100)-1
        samples = model.get_samples(sampleset=sampleset)
        nq = 10
        x_pred = np.ones((nq, 1))
        t_pred = np.linspace(0, 1, nq).reshape()
        pred_plot = wPred(x_pred=x_pred, theta_pred=t_pred, samples=samples, model=model)

        print('pred_plot_w are:')
        print(pred_plot.w.squeeze()[0,:])
        print('Matlab pred_plot_w are:')
        print(np.array(matlab_output['pred_plot_w']).squeeze()[0,:])

        if show_figs:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(model.data.sim_data.t_trans, model.data.sim_data.y_std)
            plt.plot(np.tile(t, (len(sampleset), 1)), np.squeeze(pred_plot.w), '.')
            plt.show()

        print('Checking predicted realizations for plotting...')
        # Apparently numerics come into play here, need to turn down the rtol on 'close'
        self.assertTrue(np.allclose(np.array(matlab_output['pred_plot_w']).squeeze(), pred_plot.w.squeeze(),rtol=1e-1))

        print('Done.')

    def test_predict_univ_sim_and_obs(self):

        print('starting test_predict_univ_sim_and_obs', flush=True)

        show_figs = True
        seed = 42.
        n_mcmc = 1000
        m = 300
        n = 10
        n_pred = 5

        # call function to do matlab setup/sampling
        model, matlab_output = setup_univ_sim_and_obs(m=m, n=n, seed=seed, n_mcmc=n_mcmc, n_pred=n_pred)
        mcmc_time_mat = matlab_output['mcmc_time']
        mcmc_mat = matlab_output['mcmc']
        mcmc_mat = {k: np.array(mcmc_mat[k]) for k in mcmc_mat.keys()}

        # do python sampling
        np.random.seed(int(seed))
        t_start = time()
        model.do_mcmc(n_mcmc)
        t_end = time()

        print('Python mcmc time %0.3g s' % (t_end - t_start), flush=True)
        print('Matlab mcmc time %0.3g s' % mcmc_time_mat, flush=True)

        samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}
        samples_dict['logPost'] = np.array(model.params.lp.mcmc.draws).reshape((-1, 1))

        np.random.seed(int(seed))
        psamps = model.get_samples(0, sampleset=range(n_pred), flat=True)
        pred = wPred([0.5], psamps, model.num, model.data, returnMuSigma=True)
        print('Samples are:')
        print(pred.w.squeeze())
        print('Matlab Samples are:')
        print(np.array(matlab_output['pred_w']).squeeze())

        print('Mu are:')
        print(pred.mu.squeeze())
        print('Matlab Mu are:')
        print(np.array(matlab_output['pred_Myhat']).squeeze())

        print('Sigma are:')
        print(pred.sigma.squeeze())
        print('Matlab Sigma are:')
        print(np.array(matlab_output['pred_Syhat']).squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_w']).squeeze(), pred.w.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_Myhat']).squeeze(), pred.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_Syhat']).squeeze(), pred.sigma.squeeze()))

        print('Done.')


    def test_predict_multi_sim_only(self):

        print('starting test_predict_multi_sim_only', flush=True)

        # Note: some stuff hardcoded in matlab, might break if some of these are changed
        n_pred = 5
        seed = 42.
        n_mcmc = 30
        m = 20
        nt = 10
        n_pc = 4
        nx = 2

        # TODO diagnose why we need fix_K, make sure not a PCA error somewhere, see if can check abs(weights)
        model, matlab_output = setup_multi_sim_only(m=m, nt=nt, nx=nx, n_pc=n_pc, seed=seed, n_mcmc=n_mcmc, n_pred=n_pred, fix_K=True)
        mcmc_time_mat = matlab_output['mcmc_time']
        mcmc_mat = matlab_output['mcmc']
        mcmc_mat = {k: np.array(mcmc_mat[k]) for k in mcmc_mat.keys()}

        # do python sampling
        np.random.seed(int(seed))
        t_start = time()
        model.do_mcmc(n_mcmc)
        t_end = time()

        print('Python mcmc time %0.3g s' % (t_end - t_start), flush=True)
        print('Matlab mcmc time %0.3g s' % mcmc_time_mat, flush=True)

        samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}
        samples_dict['logPost'] = np.array(model.params.lp.mcmc.draws).reshape((-1, 1))

        self.assertTrue(set(samples_dict.keys()) == set(mcmc_mat.keys()))

        np.random.seed(int(seed))
        psamps = model.get_samples(0, sampleset=range(n_pred), flat=True)
        pred = wPred([0.5, 0.5], psamps, model.num, model.data, returnMuSigma=True)
        print('Samples are:')
        print(pred.w.squeeze())
        print('Matlab Samples are:')
        print(np.array(matlab_output['pred_w']).squeeze())

        print('Mu are:')
        print(pred.mu.squeeze())
        print('Matlab Mu are:')
        print(np.array(matlab_output['pred_Myhat']))

        print('Sigma are:')
        print(pred.sigma.squeeze().squeeze().reshape(n_pred * n_pc, n_pc).T)
        print('Matlab Sigma are:')
        print(np.array(matlab_output['pred_Syhat']).squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_w']).squeeze(), pred.w.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_Myhat']).squeeze(), pred.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_Syhat']).squeeze(),
                                    pred.sigma.squeeze().reshape(n_pred * n_pc, n_pc).T))

        np.random.seed(int(seed))
        pred_arv = wPred([0.5, 0.5], psamps, model.num, model.data,
                         addResidVar=True, returnMuSigma=True)
        print('Add Residual Variance test')
        print('Samples are:')
        print(pred_arv.w.squeeze())
        print('Matlab Samples are:')
        print(np.array(matlab_output['pred_arv_w']).squeeze())

        print('Mu are:')
        print(pred_arv.mu.squeeze())
        print('Matlab Mu are:')
        print(np.array(matlab_output['pred_arv_Myhat']))

        print('Sigma are:')
        print(pred_arv.sigma.squeeze().squeeze().reshape(n_pred * n_pc, n_pc).T)
        print('Matlab Sigma are:')
        print(np.array(matlab_output['pred_arv_Syhat']).squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_arv_w']).squeeze(), pred_arv.w.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_arv_Myhat']).squeeze(), pred_arv.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_arv_Syhat']).squeeze(),
                                    pred_arv.sigma.squeeze().reshape(n_pred * n_pc, n_pc).T))

        print('Done.')


    def test_predict_multi_sim_and_obs(self):

        print('starting test_predict_multi_sim_and_obs', flush=True)

        # Note: some stuff hardcoded in matlab, might break if some of these are changed
        seed = 42.
        n_mcmc = 20
        m = 200
        n = 20
        nt_sim = 75
        nt_obs = 50
        n_pc = 5  # must be smaller than nt
        nx = 3
        noise_sd = 0.1
        n_pred = 5
        model, matlab_output = setup_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd,
                                                       nx=nx, n_pc=n_pc, seed=seed, n_lik=0, n_mcmc=n_mcmc, n_pred=n_pred, fix_K=True)
        mcmc_time_mat = matlab_output['mcmc_time']
        mcmc_mat = matlab_output['mcmc']
        mcmc_mat = {k: np.array(mcmc_mat[k]) for k in mcmc_mat.keys()}

        # do python sampling
        np.random.seed(int(seed))
        t_start = time()
        model.do_mcmc(n_mcmc)
        t_end = time()

        print('Python mcmc time %0.3g s' % (t_end - t_start), flush=True)
        print('Matlab mcmc time %0.3g s' % mcmc_time_mat, flush=True)

        samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}
        samples_dict['logPost'] = np.array(model.params.lp.mcmc.draws).reshape((-1, 1))

        self.assertTrue(set(samples_dict.keys()) == set(mcmc_mat.keys()))

        np.random.seed(int(seed))
        psamps = model.get_samples(0, sampleset=range(5), flat=True)
        pred = wPred([0.5], psamps, model.num, model.data, returnMuSigma=True)
        print('Samples are:')
        print(pred.w.squeeze())
        print('Matlab Samples are:')
        print(np.array(matlab_output['pred_w']).squeeze())

        print('Mu are:')
        print(pred.mu.squeeze())
        print('Matlab Mu are:')
        print(np.array(matlab_output['pred_Myhat']))

        print('Sigma are:')
        print(pred.sigma.squeeze().reshape(n_pred * n_pc, n_pc).T)
        print('Matlab Sigma are:')
        print(np.array(matlab_output['pred_Syhat']).squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_w']).squeeze(), pred.w.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_Myhat']).squeeze(), pred.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred_Syhat']).squeeze(),
                                    pred.sigma.squeeze().reshape(n_pred * n_pc, n_pc).T))

        np.random.seed(int(seed))
        psamps = model.get_samples(0, sampleset=[0, n_pred-1], flat=True)
        pred = uvPred([0.5], psamps, model.num, model.data, returnMuSigma=True, useAltW=True)
        #pred = uvPred([0.5], psamps, model.num, model.data, returnMuSigma=True)
        print('Samples of u are:')
        print(pred.u.squeeze())
        print('Matlab Samples of u are:')
        print(np.array(matlab_output['pred2_u']).squeeze())

        print('Samples of v are:')
        print(pred.v.squeeze())
        print('Matlab Samples of v are:')
        print(np.array(matlab_output['pred2_v']).squeeze())

        print('Mu are:')
        print(pred.mu.squeeze())
        print('Matlab Mu are:')
        print(np.array(matlab_output['pred2_Myhat']))

        print('Sigma are:')
        print(pred.sigma.squeeze().reshape(2 * (n_pc + 1), n_pc + 1).T)
        print('Matlab Sigma are:')
        print(np.array(matlab_output['pred2_Syhat']).squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred2_u']).squeeze(), pred.u.squeeze()))
        self.assertTrue(np.allclose(np.array(matlab_output['pred2_v']).squeeze(), pred.v.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred2_Myhat']).squeeze(), pred.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(np.array(matlab_output['pred2_Syhat']).squeeze(),
                                    pred.sigma.squeeze().reshape(2 * (n_pc + 1), n_pc + 1).T))

        print('Done.')

