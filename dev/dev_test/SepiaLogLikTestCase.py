import unittest
import numpy as np
import scipy.io
from time import time
from tqdm import tqdm
import matlab.engine

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
from setup_test_cases import *

import os

script_path = os.path.dirname(os.path.realpath(__file__))

np.random.seed(42)

"""
NOTE: requires matlab.engine.

To install at command line:
> source activate <sepia conda env name>
> cd <matlabroot>/extern/engines/python
> python setup.py install
"""


class SepiaLogLikTestCase(unittest.TestCase):
    """
    Checks likelihood calculation between matlab and python.
    Run files in matlab/ dir to generate data prior to running these tests.
    """

    def test_log_lik_univ_sim_only(self):
        print('starting test_log_lik_univ_sim_only', flush=True)
        seed = 42.
        n_lik = 100
        m = 1500
        model, matlab_output = setup_univ_sim_only(m=m, seed=seed, n_lik=n_lik)
        np.random.seed(int(seed))
        ll = model.logLik()
        ll_mat = matlab_output['ll']
        ll_time_mat = matlab_output['ll_time']
        t1 = time()
        for _ in range(n_lik):
            _ = model.logLik()
        print('Univariate sim-only:', flush=True)
        print('python ll %0.4g' % ll)
        print('matlab ll %0.4g' % ll_mat)
        print('python ll time per %d calls %0.3g' % (n_lik, (time() - t1)), flush=True)
        print('matlab ll time per %d calls %0.3g' % (n_lik, ll_time_mat), flush=True)
        self.assertTrue(np.allclose(ll, ll_mat))
        print('finished test_log_lik_univ_sim_only', flush=True)


    def test_log_lik_univ_sim_and_obs(self):
        print('starting test_log_lik_univ_sim_and_obs', flush=True)
        seed = 42.
        n_lik_calls = 100
        m = 100
        n = 10
        model, matlab_output = setup_univ_sim_and_obs(m=m, n=n, seed=seed, n_lik=n_lik_calls)
        np.random.seed(int(seed))
        ll = model.logLik()
        ll_mat = matlab_output['ll']
        ll_time_mat = matlab_output['ll_time']
        t1 = time()
        for _ in range(n_lik_calls):
            _ = model.logLik()
        print('Univariate sim and obs:', flush=True)
        print('python ll %0.4g' % ll)
        print('matlab ll %0.4g' % ll_mat)
        print('python ll time per 1000 calls %0.3g' % (time() - t1), flush=True)
        print('matlab ll time per 1000 calls %0.3g' % ll_time_mat, flush=True)
        self.assertTrue(np.allclose(ll, ll_mat))
        print('finished test_log_lik_univ_sim_and_obs', flush=True)

    def test_log_lik_multi_sim_only(self):
        print('starting test_log_lik_multi_sim_only', flush=True)
        seed = 42.
        n_lik = 100
        m = 500
        nt = 75
        n_pc = 15 # must be smaller than nt
        nx = 10
        model, matlab_output = setup_multi_sim_only(m=m, nt=nt, nx=nx, n_pc=n_pc, seed=seed, n_lik=n_lik, n_mcmc=0)
        ll = model.logLik()
        ll_mat = matlab_output['ll']
        ll_time_mat = matlab_output['ll_time']
        t1 = time()
        for _ in range(n_lik):
            _ = model.logLik()
        print('Multivariate sim-only:', flush=True)
        print('python ll %0.4g' % ll)
        print('matlab ll %0.4g' % ll_mat)
        print('python ll time per %d calls %0.3g' % (n_lik, (time() - t1)), flush=True)
        print('matlab ll time per %d calls %0.3g' % (n_lik, ll_time_mat), flush=True)
        self.assertTrue(np.allclose(ll, ll_mat))
        print('finished test_log_lik_multi_sim_only', flush=True)


    def test_log_lik_multi_sim_and_obs(self):
        print('starting test_log_lik_multi_sim_and_obs', flush=True)
        seed = 42.
        n_lik = 100
        m = 200
        n = 20
        nt_sim = 75
        nt_obs = 50
        n_pc = 15 # must be smaller than nt
        nx = 10
        noise_sd = 0.1
        model, matlab_output = setup_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd,
                                                       nx=nx, n_pc=n_pc, seed=seed, n_lik=n_lik, n_mcmc=0)
        ll = model.logLik()
        ll_mat = matlab_output['ll']
        ll_time_mat = matlab_output['ll_time']
        t1 = time()
        for _ in range(n_lik):
            _ = model.logLik()
        print('Multivariate sim and obs:', flush=True)
        print('python ll %0.4g' % ll)
        print('matlab ll %0.4g' % ll_mat)
        print('python ll time per %d calls %0.3g' % (n_lik, (time() - t1)), flush=True)
        print('matlab ll time per %d calls %0.3g' % (n_lik, ll_time_mat), flush=True)
        self.assertTrue(np.allclose(ll, ll_mat))
        print('finished test_log_lik_multi_sim_and_obs', flush=True)


    def test_log_lik_multi_sim_and_obs_noD(self):
        print('starting test_log_lik_multi_sim_and_obs_noD', flush=True)
        seed = 42.
        n_lik = 100
        m = 200
        n = 20
        nt_sim = 75
        nt_obs = 50
        n_pc = 15 # must be smaller than nt
        nx = 10
        noise_sd = 0.1
        model, matlab_output = setup_multi_sim_and_obs_noD(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd,
                                                       nx=nx, n_pc=n_pc, seed=seed, n_lik=n_lik, n_mcmc=0)
        ll = model.logLik()
        ll_mat = matlab_output['ll']
        ll_time_mat = matlab_output['ll_time']
        t1 = time()
        for _ in range(n_lik):
            _ = model.logLik()
        print('Multivariate sim and obs:', flush=True)
        print('python ll %0.4g' % ll)
        print('matlab ll %0.4g' % ll_mat)
        print('python ll time per %d calls %0.3g' % (n_lik, (time() - t1)), flush=True)
        print('matlab ll time per %d calls %0.3g' % (n_lik, ll_time_mat), flush=True)
        self.assertTrue(np.allclose(ll, ll_mat))
        print('finished test_log_lik_multi_sim_and_obs_noD', flush=True)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SepiaLogLikTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)