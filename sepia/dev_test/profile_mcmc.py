import numpy as np
import scipy.io
import scipy.stats
import os

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
from sepia.util import timeit
import matlab.engine

"""
NOTE: requires matlab.engine.

To install at command line:
> source activate <sepia conda env name>
> cd <matlabroot>/extern/engines/python
> python setup.py install
"""

# profiling mcmc
n_pc = 2
nsamp = 100  # 1000
nburn = 0  # 100
seed = 42.
lamWOs_init = 50000.  # use 0 to use default lamWOs initial value

np.random.seed(int(seed))
script_path = os.path.dirname(os.path.realpath(__file__))

# Run matlab code, then open data from matlab
try:
    eng = matlab.engine.start_matlab()
    eng.cd(script_path)
    eng.addpath('matlab/', nargout=0)
    eng.multi_sim_and_obs_mcmc_test(nsamp, nburn, seed, lamWOs_init, n_pc, nargout=0)
    eng.quit()
    matfile = scipy.io.loadmat('%s/data/multi_sim_and_obs_mcmc_test.mat' % script_path)
except Exception as e:
    print(e)
    print('make sure matlab.engine installed')

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
data.create_K_basis(n_pc=2)
data.create_D_basis(D=matfile['Dobs'].T)
print(data)

model = setup_model(data)

nsamp = int(matfile['nsamp'])
nburn = int(matfile['nburn'])

@timeit
def run_mcmc():
    model.do_mcmc(nburn + nsamp)

print('Python mcmc time:')
run_mcmc()

# import cProfile
# cProfile.run('run_mcmc()', 'mcmc.profile')
#
# import os
# os.system('snakeviz mcmc.profile')