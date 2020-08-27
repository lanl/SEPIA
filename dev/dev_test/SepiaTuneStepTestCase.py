import unittest
import numpy as np

from setup_test_cases import *


"""
NOTE: requires matlab.engine.

To install at command line:
> source activate <sepia conda env name>
> cd <matlabroot>/extern/engines/python
> python setup.py install
"""


class SepiaTuneStepTestCase(unittest.TestCase):
    """
    Checks step size tuning results between matlab and python.
    Run files in matlab/ dir to generate data prior to running these tests.
    """

    def test_tune_step_univ_sim_only(self):
        print('starting test_mcmc_univ_sim_only', flush=True)

        show_figs = True
        seed = 42.
        n_mcmc = 100
        m = 300
        n_lev = 15
        n_burn = 100

        # call function to do matlab setup/sampling
        model, matlab_output = setup_univ_sim_only(m=m, seed=seed, n_lev=n_lev, n_burn=n_burn)
        rhoUwidth = matlab_output['rhoUwidth']
        lamUzwidth = matlab_output['lamUzwidth']
        lamWswidth = matlab_output['lamWswidth']
        lamWOswidth = matlab_output['lamWOswidth']

        # do python step size
        np.random.seed(int(seed))
        res = model.tune_step_sizes(n_burn, n_lev, diagnostics=True)

        print('python betaUwidth')
        print(model.params.betaU.mcmc.stepParam)
        print('matlab betaUwidth')
        print(rhoUwidth)

        print('python lamUzwidth')
        print(model.params.lamUz.mcmc.stepParam)
        print('matlab lamUzwidth')
        print(lamUzwidth)

        print('python lamWswidth')
        print(model.params.lamWs.mcmc.stepParam)
        print('matlab lamWswidth')
        print(lamWswidth)

        print('python lamWOswidth')
        print(model.params.lamWOs.mcmc.stepParam)
        print('matlab lamWOswidth')
        print(lamWOswidth)

        # Broken
        # step = {k: np.array(res[0][k]).squeeze() for k in res[0].keys()}
        # acc = {k: np.array(res[1][k]).squeeze() for k in res[1].keys()}
        #
        # target_logit = np.log(1 / (np.exp(1) - 1))
        #
        # for k in step.keys():
        #     mean_acc = np.mean(acc[k], 1)
        #     plt.plot(np.log(step[k]), np.log(mean_acc / (1 - mean_acc)))
        #     plt.axhline(y=target_logit)
        #     plt.title(k)
        #     plt.xlabel('log step size')
        #     plt.ylabel('logit acc rate')
        #     plt.show()




        print('fin')



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SepiaTuneStepTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)