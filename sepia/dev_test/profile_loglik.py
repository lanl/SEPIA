import numpy as np
import scipy.io
import matlab.engine

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
from sepia.util import timeit

import os
script_path = os.path.dirname(os.path.realpath(__file__))

np.random.seed(42)

# Open data from matlab
try:
    eng = matlab.engine.start_matlab()
    eng.addpath('matlab/', nargout=0)
    eng.multi_sim_and_obs_lik_test(nargout=0)
    eng.quit()
    matfile = scipy.io.loadmat('%s/data/multi_sim_and_obs_lik_test.mat' % script_path)
except Exception as e:
    print(e)
    print('make sure matlab.engine installed')
y_sim = matfile['y_sim'].T
y_ind_sim = matfile['y_ind_sim'].squeeze()
x_sim = matfile['x_sim']
y_obs = matfile['y_obs']
x_obs = matfile['x_obs']
y_ind_obs = matfile['y_ind_obs'].squeeze()
data = SepiaData(x_sim=x_sim[:, 0][:, None], t_sim=x_sim[:, 1][:, None], y_sim=y_sim, y_ind_sim=y_ind_sim,
                 x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)
data.standardize_y()
data.transform_xt()
data.create_K_basis(n_pc=2)
data.create_D_basis(D_sim=matfile['Dsim'].T)
model = setup_model(data)

@timeit
def do_loglik():
    for _ in range(1000):
        ll = model.logLik()
