import unittest
import numpy as np

from sepia.SepiaSensitivity import sensitivity
from setup_test_cases import *


"""
NOTE: requires matlab.engine.

To install at command line:
> source activate <sepia conda env name>
> cd <matlabroot>/extern/engines/python
> python setup.py install
"""

class SepiaSensitivityTestCase(unittest.TestCase):
    """
    Checks Sensitivity analysis results between matlab and python.
    Run files in matlab/ dir to generate data prior to running these tests.
    """

    def test_sens_univ_sim_only(self):
        print('starting test_sens_univ_sim_only', flush=True)

        show_figs = True
        seed = 42.
        n_mcmc = 30
        m = 100

        # call function to do matlab setup/sampling
        model, matlab_output = setup_univ_sim_only(m=m, seed=seed, n_mcmc=n_mcmc, sens=1)
        mcmc_time_mat = matlab_output['mcmc_time']
        mcmc_mat = matlab_output['mcmc']
        mcmc_mat = {k: np.array(mcmc_mat[k]) for k in mcmc_mat.keys()}

        # do python sampling
        np.random.seed(int(seed))
        model.do_mcmc(n_mcmc)

        sme_mat = np.array(matlab_output['smePm'])
        ste_mat = np.array(matlab_output['stePm'])

        # TODO sensitivity fails due to no K
        # sa = sensitivity(model)
        # sme_py = sa.sme
        # ste_py = sa.ste

    def test_sens_multi_sim_only(self):
        print('starting test_sens_multi_sim_only', flush=True)

        # Matlab debug call: setup_multi_sim_only(20, 20, 5, 10, 42., 0, 30, 0, 1)

        show_figs = True
        seed = 42.
        n_mcmc = 30
        m = 20

        # call function to do matlab setup/sampling
        model, matlab_output = setup_multi_sim_only(m=m, seed=seed, n_mcmc=n_mcmc, sens=1)
        mcmc_time_mat = matlab_output['mcmc_time']
        mcmc_mat = matlab_output['mcmc']
        mcmc_mat = {k: np.array(mcmc_mat[k]) for k in mcmc_mat.keys()}

        # do python sampling
        np.random.seed(int(seed))
        model.do_mcmc(n_mcmc)

        sme_mat = np.array(matlab_output['smePm'])
        ste_mat = np.array(matlab_output['stePm'])

        sa = sensitivity(model)
        sme_py = sa['smePm']
        ste_py = sa['stePm']

        self.assertTrue(np.allclose(sme_py, sme_mat))
        self.assertTrue(np.allclose(ste_py, ste_mat))



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SepiaSensitivityTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)