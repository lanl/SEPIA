import numpy as np
import scipy.io
import matlab.engine

from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from test.util import timeit
from sepia.SepiaPredict import wPred, uvPred

import os
script_path = os.path.dirname(os.path.realpath(__file__))

### Univ -- need univ with sim and obs to test uvPred



### Multi
np.random.seed(42)

n_pc = 2
seed = 42
lamWOs_init = 50000.  # use 0 to use default lamWOs initial value
nsamp = 100
nburn = 0
# Open data from matlab
mat_fn = '%s/data/multi_sim_and_obs_mcmc_test.mat' % script_path
if os.path.isfile(mat_fn):
    # if the matlab data is already in place, just load that
    print('Found matfile, loading from multi_sim_and_obs_mcmc_test.mat \n')
    matfile = scipy.io.loadmat(mat_fn)
else:
    print('Generating matfile multi_sim_and_obs_mcmc_test.mat \n')
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

nburn = int(matfile['nburn'])
nsamp = int(matfile['nsamp'])

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
data.create_D_basis(D_obs=matfile['Dobs'].T)
print(data)

np.random.seed(int(seed))
model = SepiaModel(data)
if lamWOs_init > 0:
    model.params.lamWOs.val = np.array([[lamWOs_init]])

model.do_mcmc(nburn + nsamp)
np.random.seed(seed)
psamps = model.get_samples(sampleset=range(50), flat=True)

@timeit
def do_wPred():
    for _ in range(10):
        pred = wPred([0.5], psamps, model.num, model.data, returnMuSigma=True, addResidVar=True, returnRlz=True)

@timeit
def do_uvPred():
    for _ in range(50):
        pred = uvPred([0.5], psamps, model.num, model.data, returnMuSigma=True, useAltW=False)

print('\nuvPred x10')
do_uvPred()

print('wPred x10')
do_wPred()

