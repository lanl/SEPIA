import unittest
import numpy as np
import scipy.io
from time import time
import scipy.stats
import os

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
import matlab.engine
from sepia.SepiaPredict import wPred, uvPred

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

        np.random.seed(42)

        show_figs = True

        # Open data from matlab
        script_path = os.path.dirname(os.path.realpath(__file__))
        mat_fn='%s/data/univ_sim_only_mcmc_test.mat' % script_path
        if os.path.isfile(mat_fn):
            # if the matlab data is already in place, just load that
            print('Found matfile, loading from univ_sim_only_mcmc_test.mat \n')
            matfile=scipy.io.loadmat(mat_fn)
        else:
            print('Generating matfile univ_sim_only_mcmc_test.mat \n')
            # Run matlab code, then open data from matlab
            try:
                eng = matlab.engine.start_matlab()
                eng.cd(script_path)
                eng.addpath('matlab/', nargout=0)
                eng.univ_sim_only_mcmc_test(nargout=0)
                eng.quit()
                matfile = scipy.io.loadmat('mat_fn')
            except Exception as e:
                print(e)
                print('make sure matlab.engine installed')

        y = matfile['y']
        x = matfile['x']
        t = matfile['t']
        data = SepiaData(x_sim=x, t_sim=t, y_sim=y)
        data.standardize_y()
        data.transform_xt()
        print(data)
        model = setup_model(data)
        nsamp = int(matfile['nsamp'])
        nburn = int(matfile['nburn'])
        t_start = time()
        model.do_mcmc(nburn+nsamp)
        t_end = time()
        print('Python mcmc time %0.3g s' % (t_end - t_start))
        print('Matlab mcmc time %0.3g s' % matfile['mcmc_time'])
        # Creates dict with each sampled variable name as key, array of samples (nsamp, ...) as value
        samples = model.get_samples(nburn)
        log_post = np.array(model.params.lp.mcmc.draws)

        np.random.seed(42)
        psamps = model.get_samples(0, sampleset=[0, 1, 2, 3, 4], flat=True)
        pred = wPred([0.5, 0.5], psamps, model.num, model.data, returnMuSigma=True)
        print('Samples are:')
        print(pred.w.squeeze())
        print('Matlab Samples are:')
        print(matfile['pred_w'].squeeze())

        print('Mu are:')
        print(pred.mu.squeeze())
        print('Matlab Mu are:')
        print(matfile['pred_Myhat'].squeeze())

        print('Sigma are:')
        print(pred.sigma.squeeze())
        print('Matlab Sigma are:')
        print(matfile['pred_Syhat'].squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(matfile['pred_w'].squeeze(), pred.w.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(matfile['pred_Myhat'].squeeze(), pred.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(matfile['pred_Syhat'].squeeze(), pred.sigma.squeeze()))

        # Prediction with multiple realizations
        np.random.seed(42)
        sampleset = np.arange(100, 1001, 100)-1
        samples = model.get_samples(sampleset=sampleset)
        nq = 10
        t = np.linspace(0, 1, nq)
        xpred = np.column_stack((np.ones((nq, 1)) * 0.5, t))
        pred_plot = wPred(xpred, samples, model.num, model.data)

        print('pred_plot_w are:')
        print(pred_plot.w.squeeze()[0,:])
        print('Matlab pred_plot_w are:')
        print(matfile['pred_plot_w'].squeeze()[0,:])

        print('Checking predicted realizations for plotting...')
        # Apparently numerics come into play here, need to turn down the rtol on 'close'
        self.assertTrue(np.allclose(matfile['pred_plot_w'].squeeze(), pred_plot.w.squeeze(),rtol=1e-3))

        print('Done.')

        if show_figs:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(data.sim_data.t_trans, data.sim_data.y_std)
            plt.plot(np.tile(t, (len(sampleset), 1)), np.squeeze(pred_plot.w), '.')
            plt.show()

    def test_predict_multi_sim_only(self):

        show_figs = True
        exclude_burnin = True
        n_pc = 2
        seed = 42
        lamWOs_init = 50000.  # use 0 to use default lamWOs initial value
        list_to_sample = [1, 2, 3, 4]  # 1-based indexing for matlab; order is [betaU, lamUz, lamWs, lamWOs]
        nsamp = 100
        nburn = 10
        # Open data from matlab
        script_path = os.path.dirname(os.path.realpath(__file__))
        mat_fn = '%s/data/multi_sim_only_mcmc_test.mat' % script_path
        if os.path.isfile(mat_fn):
            # if the matlab data is already in place, just load that
            print('Found matfile, loading from multi_sim_only_mcmc_test.mat \n')
            matfile = scipy.io.loadmat(mat_fn)
        else:
            print('Generating matfile multi_sim_only_mcmc_test.mat \n')
            # Run matlab code, then open data from matlab
            list_to_sample = [1, 2, 3, 4] # 1-based indexing for matlab; order is [betaU, lamUz, lamWs, lamWOs]
            script_path = os.path.dirname(os.path.realpath(__file__))

            # Run matlab code, then open data from matlab
            try:
                eng = matlab.engine.start_matlab()
                eng.cd(script_path)
                eng.addpath('matlab/', nargout=0)
                eng.multi_sim_only_mcmc_test(nsamp, nburn, list_to_sample, seed, lamWOs_init, n_pc, nargout=0)
                eng.quit()
                matfile = scipy.io.loadmat('%s/data/multi_sim_only_mcmc_test.mat' % script_path)
            except Exception as e:
                print(e)
                print('make sure matlab.engine installed')

        y = matfile['y'].T
        y_ind = matfile['y_ind'].T
        x = matfile['x']
        data = SepiaData(x_sim=x[:, 0][:, None], t_sim=x[:, 1][:, None], y_sim=y, y_ind_sim=y_ind)
        data.standardize_y()
        data.transform_xt()
        data.create_K_basis(n_pc=n_pc)
        print(data)

        np.random.seed(int(seed))
        model = setup_model(data)
        if lamWOs_init > 0:
            model.params.lamWOs.val = np.array([[lamWOs_init]])
        model.params.mcmcList = [model.params.mcmcList[i-1] for i in list_to_sample]
        t_start = time()
        model.do_mcmc(nburn+nsamp)
        t_end = time()
        print('Python mcmc time %0.3g s' % (t_end - t_start))
        print('Matlab mcmc time %0.3g s' % matfile['mcmc_time'])

        np.random.seed(seed)
        psamps = model.get_samples(0, sampleset=[0, 1, 2, 3, 4], flat=True)
        pred = wPred([0.5, 0.5], psamps, model.num, model.data, returnMuSigma=True)
        print('Samples are:')
        print(pred.w.squeeze())
        print('Matlab Samples are:')
        print(matfile['pred_w'].squeeze())

        print('Mu are:')
        print(pred.mu.squeeze())
        print('Matlab Mu are:')
        print(matfile['pred_Myhat'])

        print('Sigma are:')
        print(pred.sigma.squeeze().squeeze().reshape(10,2).T)
        print('Matlab Sigma are:')
        print(matfile['pred_Syhat'].squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(matfile['pred_w'].squeeze(), pred.w.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(matfile['pred_Myhat'].squeeze(), pred.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(matfile['pred_Syhat'].squeeze(),
                                    pred.sigma.squeeze().reshape(10,2).T))

        pred_arv = wPred([0.5, 0.5], psamps, model.num, model.data,
                         addResidVar=True, returnMuSigma=True)
        print('Add Residual Variance test')
        print('Samples are:')
        print(pred_arv.w.squeeze())
        print('Matlab Samples are:')
        print(matfile['pred_arv_w'].squeeze())

        print('Mu are:')
        print(pred_arv.mu.squeeze())
        print('Matlab Mu are:')
        print(matfile['pred_arv_Myhat'])

        print('Sigma are:')
        print(pred_arv.sigma.squeeze().squeeze().reshape(10,2).T)
        print('Matlab Sigma are:')
        print(matfile['pred_arv_Syhat'].squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(matfile['pred_arv_w'].squeeze(), pred_arv.w.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(matfile['pred_arv_Myhat'].squeeze(), pred_arv.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(matfile['pred_arv_Syhat'].squeeze(),
                                    pred_arv.sigma.squeeze().reshape(10,2).T))


        print('Done.')

    def test_predict_sim_from_multi_obs(self):

        show_figs = True
        exclude_burnin = True
        n_pc = 2
        seed = 42.
        lamWOs_init = 50000.  # use 0 to use default lamWOs initial value
        nsamp = 100
        nburn = 0
        # Open data from matlab
        script_path = os.path.dirname(os.path.realpath(__file__))
        mat_fn = '%s/data/multi_sim_and_obs_mcmc_test.mat' % script_path
        if os.path.isfile(mat_fn):
            # if the matlab data is already in place, just load that
            print('Found matfile, loading from multi_sim_and_obs_mcmc_test.mat \n')
            matfile = scipy.io.loadmat(mat_fn)
        else:
            print('Generating matfile multi_sim_and_obs_mcmc_test.mat \n')
            # Run matlab code, then open data from matlab
            script_path = os.path.dirname(os.path.realpath(__file__))

            # Run matlab code, then open data from matlab
            try:
                eng = matlab.engine.start_matlab()
                eng.cd(script_path)
                eng.addpath('matlab/', nargout=0)
                eng.multi_sim_and_obs_mcmc_test(nsamp, nburn, seed, lamWOs_init, n_pc, 0, nargout=0)
                eng.quit()
                matfile = scipy.io.loadmat('%s/data/multi_sim_and_obs_mcmc_test.mat' % script_path)
            except Exception as e:
                print(e)
                print('make sure matlab.engine installed')

        nburn=int(matfile['nburn'])
        nsamp=int(matfile['nsamp'])

        y_sim = matfile['y'].T
        y_ind_sim = matfile['y_ind'].squeeze()
        xt_sim = matfile['x']
        y_obs = matfile['y_obs']
        y_ind_obs = matfile['y_ind_obs'].squeeze()
        x_obs = matfile['x_obs']
        data = SepiaData(x_sim=xt_sim[:, 0][:, None], t_sim=xt_sim[:, 1][:, None], y_sim=y_sim, y_ind_sim=y_ind_sim,
                         x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)
        data.standardize_y()
        data.transform_xt()
        data.create_K_basis(n_pc=n_pc)
        data.create_D_basis(D=matfile['Dobs'].T)
        print(data)

        np.random.seed(int(seed))
        model = setup_model(data)
        if lamWOs_init > 0:
            model.params.lamWOs.val = np.array([[lamWOs_init]])

        t_start = time()
        model.do_mcmc(nburn+nsamp)
        t_end = time()
        print('Python mcmc time %0.3g s' % (t_end - t_start))
        print('Matlab mcmc time %0.3g s' % matfile['mcmc_time'])

        np.random.seed(int(seed))
        psamps = model.get_samples(0, sampleset=[0, 1, 2, 3, 4], flat=True)
        pred = wPred([0.5], psamps, model.num, model.data, returnMuSigma=True)
        print('Samples are:')
        print(pred.w.squeeze())
        print('Matlab Samples are:')
        print(matfile['pred_w'].squeeze())

        print('Mu are:')
        print(pred.mu.squeeze())
        print('Matlab Mu are:')
        print(matfile['pred_Myhat'])

        print('Sigma are:')
        print(pred.sigma.squeeze().squeeze().reshape(10,2).T)
        print('Matlab Sigma are:')
        print(matfile['pred_Syhat'].squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(matfile['pred_w'].squeeze(), pred.w.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(matfile['pred_Myhat'].squeeze(), pred.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(matfile['pred_Syhat'].squeeze(),
                                    pred.sigma.squeeze().reshape(10,2).T))

        print('Done.')

    def test_predict_uv_from_multi_obs(self):

        show_figs = True
        exclude_burnin = True
        n_pc = 2
        seed = 42.
        lamWOs_init = 50000.  # use 0 to use default lamWOs initial value
        nsamp = 100
        nburn = 0
        # Open data from matlab
        script_path = os.path.dirname(os.path.realpath(__file__))
        mat_fn = '%s/data/multi_sim_and_obs_mcmc_test.mat' % script_path
        if os.path.isfile(mat_fn):
            # if the matlab data is already in place, just load that
            print('Found matfile, loading from multi_sim_and_obs_mcmc_test.mat \n')
            matfile = scipy.io.loadmat(mat_fn)
        else:
            print('Generating matfile multi_sim_and_obs_mcmc_test.mat \n')
            # Run matlab code, then open data from matlab
            script_path = os.path.dirname(os.path.realpath(__file__))

            # Run matlab code, then open data from matlab
            try:
                eng = matlab.engine.start_matlab()
                eng.cd(script_path)
                eng.addpath('matlab/', nargout=0)
                eng.multi_sim_and_obs_mcmc_test(nsamp, nburn, seed, lamWOs_init, n_pc, 0, nargout=0)
                eng.quit()
                matfile = scipy.io.loadmat('%s/data/multi_sim_and_obs_mcmc_test.mat' % script_path)
            except Exception as e:
                print(e)
                print('make sure matlab.engine installed')

        nburn=int(matfile['nburn'])
        nsamp=int(matfile['nsamp'])

        y_sim = matfile['y'].T
        y_ind_sim = matfile['y_ind'].squeeze()
        xt_sim = matfile['x']
        y_obs = matfile['y_obs']
        y_ind_obs = matfile['y_ind_obs'].squeeze()
        x_obs = matfile['x_obs']
        data = SepiaData(x_sim=xt_sim[:, 0][:, None], t_sim=xt_sim[:, 1][:, None], y_sim=y_sim, y_ind_sim=y_ind_sim,
                         x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)
        data.standardize_y()
        data.transform_xt()
        data.create_K_basis(n_pc=n_pc)
        data.create_D_basis(D=matfile['Dobs'].T)
        print(data)

        np.random.seed(int(seed))
        model = setup_model(data)
        if lamWOs_init > 0:
            model.params.lamWOs.val = np.array([[lamWOs_init]])

        t_start = time()
        model.do_mcmc(nburn+nsamp)
        t_end = time()
        print('Python mcmc time %0.3g s' % (t_end - t_start))
        print('Matlab mcmc time %0.3g s' % matfile['mcmc_time'])

        np.random.seed(int(seed))
        psamps = model.get_samples(0, sampleset=[0, 4], flat=True)
        #pred = uvPred([0.5], psamps, model.num, model.data, returnMuSigma=True, useAltW=True)
        pred = uvPred([0.5], psamps, model.num, model.data, returnMuSigma=True)
        print('Samples of u are:')
        print(pred.u.squeeze())
        print('Matlab Samples of u are:')
        print(matfile['pred2_u'].squeeze())

        print('Samples of v are:')
        print(pred.v.squeeze())
        print('Matlab Samples of v are:')
        print(matfile['pred2_v'].squeeze())

        print('Mu are:')
        print(pred.mu.squeeze())
        print('Matlab Mu are:')
        print(matfile['pred2_Myhat'])

        print('Sigma are:')
        print(pred.sigma.squeeze().reshape(14,7).T)
        print('Matlab Sigma are:')
        print(matfile['pred2_Syhat'].squeeze())

        print('Checking predicted realizations...')
        self.assertTrue(np.allclose(matfile['pred2_u'].squeeze(), pred.u.squeeze()))
        self.assertTrue(np.allclose(matfile['pred2_v'].squeeze(), pred.v.squeeze()))
        print('Checking predicted means...')
        self.assertTrue(np.allclose(matfile['pred2_Myhat'].squeeze(), pred.mu.squeeze()))
        print('Checking predicted sigmas...')
        self.assertTrue(np.allclose(matfile['pred2_Syhat'].squeeze(),
                                    pred.sigma.squeeze().reshape(14,7).T))

        print('Done.')
