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


"""
NOTE: requires matlab.engine.

To install at command line:
> source activate <sepia conda env name>
> cd <matlabroot>/extern/engines/python
> python setup.py install
"""

class SepiaMCMCTestCase(unittest.TestCase):
    """
    Checks MCMC results between matlab and python.
    Run files in matlab/ dir to generate data prior to running these tests.
    """

    def test_mcmc_univ_sim_only(self):
        print('starting test_mcmc_univ_sim_only', flush=True)

        show_figs = True
        seed = 42.
        n_mcmc = 100
        m = 300

        # call function to do matlab setup/sampling
        model, matlab_output = setup_univ_sim_only(m=m, seed=seed, n_mcmc=n_mcmc)
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

        if show_figs:
            import matplotlib.pyplot as plt
            for i, k in enumerate(samples_dict.keys()):

                param_shape = samples_dict[k].shape[1]
                plt.figure(i)
                for j in range(param_shape):
                    plt.subplot(1, param_shape, j+1)
                    plt.hist(samples_dict[k][:, j], alpha=0.5)
                    plt.hist(mcmc_mat[k][:, j], alpha=0.5)
                    plt.xlabel(k)
                plt.legend(['python', 'matlab'])
                plt.show()

        for k in samples_dict.keys():
            self.assertTrue(np.allclose(np.mean(samples_dict[k], 0), np.mean(mcmc_mat[k], 0)))
            self.assertTrue(np.allclose(np.std(samples_dict[k], 0), np.std(mcmc_mat[k], 0)))

    def test_mcmc_univ_sim_and_obs(self):
        print('starting test_mcmc_univ_sim_and_obs', flush=True)

        show_figs = 1
        seed = 42.
        n_mcmc = 100
        m = 100
        n = 10
        model, matlab_output = setup_univ_sim_and_obs(m=m, n=n, seed=seed, n_mcmc=n_mcmc)
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

        if show_figs:
            import matplotlib.pyplot as plt
            for i, k in enumerate(samples_dict.keys()):

                param_shape = samples_dict[k].shape[1]
                plt.figure(i)
                for j in range(param_shape):
                    plt.subplot(1, param_shape, j + 1)
                    plt.hist(samples_dict[k][:, j], alpha=0.5)
                    plt.hist(mcmc_mat[k][:, j], alpha=0.5)
                    plt.xlabel(k)
                plt.legend(['python', 'matlab'])
                plt.show()

        for k in samples_dict.keys():
            self.assertTrue(np.allclose(np.mean(samples_dict[k], 0), np.mean(mcmc_mat[k], 0)))
            self.assertTrue(np.allclose(np.std(samples_dict[k], 0), np.std(mcmc_mat[k], 0)))

    def test_mcmc_multi_sim_only(self):

        print('starting test_mcmc_multi_sim_only', flush=True)

        show_figs = True
        seed = 42.
        n_mcmc = 30
        m = 20
        nt = 10
        n_pc = 4
        nx = 5

        model, matlab_output = setup_multi_sim_only(m=m, nt=nt, nx=nx, n_pc=n_pc, seed=seed, n_mcmc=n_mcmc)
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

        if show_figs:
            import matplotlib.pyplot as plt
            for i, k in enumerate(samples_dict.keys()):
                param_shape = samples_dict[k].shape[1]
                if param_shape >= 5:
                    ncol = 5
                    nrow = int(np.ceil(param_shape / ncol))
                else:
                    ncol = param_shape
                    nrow = 1
                plt.figure(i)
                for j in range(param_shape):
                    plt.subplot(nrow, ncol, j + 1)
                    plt.hist(samples_dict[k][:, j], alpha=0.5)
                    plt.hist(mcmc_mat[k][:, j], alpha=0.5)
                    plt.xlabel(k)
                plt.legend(['python', 'matlab'])
                plt.show()

        for k in samples_dict.keys():
            self.assertTrue(np.allclose(np.mean(samples_dict[k], 0), np.mean(mcmc_mat[k], 0)))
            self.assertTrue(np.allclose(np.std(samples_dict[k], 0), np.std(mcmc_mat[k], 0)))

    def test_mcmc_multi_sim_and_obs(self):

        print('starting test_mcmc_multi_sim_and_obs', flush=True)

        show_figs = True
        seed = 42.
        n_mcmc = 20
        m = 200
        n = 20
        nt_sim = 75
        nt_obs = 50
        n_pc = 5 # must be smaller than nt
        nx = 3
        noise_sd = 0.1
        model, matlab_output = setup_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd,
                                                       nx=nx, n_pc=n_pc, seed=seed, n_lik=0, n_mcmc=n_mcmc)
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

        if show_figs:
            import matplotlib.pyplot as plt
            for i, k in enumerate(samples_dict.keys()):
                param_shape = samples_dict[k].shape[1]
                if param_shape >= 5:
                    ncol = 5
                    nrow = int(np.ceil(param_shape / ncol))
                else:
                    ncol = param_shape
                    nrow = 1
                plt.figure(i)
                for j in range(param_shape):
                    plt.subplot(nrow, ncol, j + 1)
                    plt.hist(samples_dict[k][:, j], alpha=0.5)
                    plt.hist(mcmc_mat[k][:, j], alpha=0.5)
                    plt.xlabel(k)
                plt.legend(['python', 'matlab'])
                plt.show()

        for k in samples_dict.keys():
            self.assertTrue(np.allclose(np.mean(samples_dict[k], 0), np.mean(mcmc_mat[k], 0)))
            self.assertTrue(np.allclose(np.std(samples_dict[k], 0), np.std(mcmc_mat[k], 0)))

    def test_mcmc_multi_sim_and_obs_noD(self):

        print('starting test_mcmc_multi_sim_and_obs_noD', flush=True)

        show_figs = True
        seed = 42.
        n_mcmc = 20
        m = 200
        n = 20
        nt_sim = 75
        nt_obs = 50
        n_pc = 5 # must be smaller than nt
        nx = 3
        noise_sd = 0.1
        model, matlab_output = setup_multi_sim_and_obs_noD(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd,
                                                       nx=nx, n_pc=n_pc, seed=seed, n_lik=0, n_mcmc=n_mcmc)
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

        if show_figs:
            import matplotlib.pyplot as plt
            for i, k in enumerate(samples_dict.keys()):
                param_shape = samples_dict[k].shape[1]
                if param_shape >= 5:
                    ncol = 5
                    nrow = int(np.ceil(param_shape / ncol))
                else:
                    ncol = param_shape
                    nrow = 1
                plt.figure(i)
                for j in range(param_shape):
                    plt.subplot(nrow, ncol, j + 1)
                    plt.hist(samples_dict[k][:, j], alpha=0.5)
                    plt.hist(mcmc_mat[k][:, j], alpha=0.5)
                    plt.xlabel(k)
                plt.legend(['python', 'matlab'])
                plt.show()

        for k in samples_dict.keys():
            self.assertTrue(np.allclose(np.mean(samples_dict[k], 0), np.mean(mcmc_mat[k], 0)))
            self.assertTrue(np.allclose(np.std(samples_dict[k], 0), np.std(mcmc_mat[k], 0)))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SepiaMCMCTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)