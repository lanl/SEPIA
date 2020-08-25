import unittest
import numpy as np
import scipy.io
from time import time
import scipy.stats
import os
import matlab.engine

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
from setup_test_cases import *
from sepia.SepiaSharedThetaModels import SepiaSharedThetaModels


"""
NOTE: requires matlab.engine.

To install at command line:
> source activate <sepia conda env name>
> cd <matlabroot>/extern/engines/python
> python setup.py install
"""

class SepiaSharedThetaTestCase(unittest.TestCase):
    """
    Checks MCMC results between matlab and python for shared theta model.
    Run files in matlab/ dir to generate data prior to running these tests.
    """



    def test_mcmc_sharedtheta_multi_sim_and_obs(self):

        print('starting test_mcmc_multi_sim_and_obs', flush=True)

        show_figs = True
        seed = 42.
        n_mcmc = 100
        m = 200
        n = 20
        nt_sim = 75
        nt_obs = 50
        n_pc = 5 # must be smaller than nt
        nx = 3
        noise_sd = 0.1
        n_shared = 3
        clist = [[1, 1, 1]] # one-indexed for matlab
        model_list, matlab_output = setup_multi_sim_and_obs_sharedtheta(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd,
                                                                        nx=nx, n_pc=n_pc, seed=seed, n_lik=1, n_mcmc=n_mcmc,
                                                                        n_shared=n_shared, clist=clist)
        mcmc_time_mat = matlab_output['mcmc_time']
        mcmc_mat = matlab_output['mcmc']
        mcmc_mat = {k: np.array(mcmc_mat[k]) for k in mcmc_mat.keys()}

        # Setup shared model object
        shared_model = SepiaSharedThetaModels(model_list=model_list, shared_theta_inds=np.array(clist)-1)

        ll = shared_model.logLik()

        print('matlab linked ll %0.5g' % matlab_output['ll'])
        print('python linked ll %0.5g' % sum(ll))

        # do python sampling
        np.random.seed(int(seed))
        t_start = time()
        shared_model.do_mcmc(n_mcmc)
        t_end = time()

        print('Python mcmc time %0.3g s' % (t_end - t_start), flush=True)
        print('Matlab mcmc time %0.3g s' % mcmc_time_mat, flush=True)

        samples_dicts = [{p.name: p.mcmc_to_array() for p in model.params.mcmcList} for model in model_list]
        for i in range(n_shared):
            samples_dicts[i]['logPost'] = np.array(model_list[i].params.lp.mcmc.draws).reshape((-1, 1))

        if show_figs:
            import matplotlib.pyplot as plt
            for i, k in enumerate(samples_dicts[0].keys()):
                for mi in range(n_shared):
                    param_shape = samples_dicts[mi][k].shape[1]
                    if param_shape >= 5:
                        ncol = 5
                        nrow = int(np.ceil(param_shape / ncol))
                    else:
                        ncol = param_shape
                        nrow = 1
                    plt.figure(i, figsize=[ncol * 3, nrow * 3])
                    for j in range(param_shape):
                        plt.subplot(nrow, ncol, j + 1)
                        plt.hist(samples_dicts[mi][k][:, j], alpha=0.5)
                        plt.hist(mcmc_mat[k][:, j, mi], alpha=0.5)
                        plt.xlabel(k)
                    plt.legend(['python', 'matlab'])
                    plt.show()

        for i in range(n_shared):
            for k in samples_dicts[i].keys():
                self.assertTrue(np.allclose(np.mean(samples_dicts[i][k], 0), np.mean(mcmc_mat[k][:, :, i], 0)))
                self.assertTrue(np.allclose(np.std(samples_dicts[i][k], 0), np.std(mcmc_mat[k][:, :, i], 0)))



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SepiaSharedThetaTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)