
import numpy as np
import os
import matlab.engine

from sepia.SepiaDistCov import SepiaDistCov
from sepia.util import timeit


script_path = os.path.dirname(os.path.realpath(__file__))

pu = 30
n1 = 700
n2 = 500
lams = 10.
lamz = 1.
beta = np.exp(-0.25 * np.linspace(0, 1, pu))[:, None]
X1 = np.random.uniform(0, 1, (n1, pu))
X2 = np.random.uniform(0, 1, (n2, pu))
nreps = 100

print('\nMATLAB\n')
try:
    eng = matlab.engine.start_matlab()
    eng.cd(script_path)
    eng.addpath('matlab/', nargout=0)
    eng.profile_dist_cov(n1, n2, lams, lamz, pu, nreps, nargout=0)
    eng.quit()
except Exception as e:
    print(e)
    print('make sure matlab.engine installed')

@timeit
def init_distcov_square():
    for i in range(nreps):
        _ = SepiaDistCov(X1)

sd = SepiaDistCov(X1)
@timeit
def calc_cov_square():
    for i in range(nreps):
        _ = sd.compute_cov_mat(beta, lamz, lams)

@timeit
def init_distcov_rect():
    for i in range(nreps):
        _ = SepiaDistCov(X1, X2)

sd = SepiaDistCov(X1, X2)
@timeit
def calc_cov_rect():
    for i in range(nreps):
        _ = sd.compute_cov_mat(beta, lamz, lams)

print('\nPYTHON\n')

print('create square dist obj x%d' % nreps)
init_distcov_square()

print('calc square cov x%d' % nreps)
calc_cov_square()

print('create rect dist obj x%d' % nreps)
init_distcov_rect()

print('calc rect cov x%d' % nreps)
calc_cov_rect()
