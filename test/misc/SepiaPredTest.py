#%%
import numpy as np
import scipy.io
from time import time
import scipy.stats
import os

import matlab.engine

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
from sepia.SepiaPredict import wPred
from sepia.SepiaPredict import uvPred


#%%
'''
Warning: this was test code almost guaranteed now not to run
but possibly good for refactoring
'''


#%% Get a model and set of posterior samples.

np.random.seed(42)

#os.chdir('..')
test_path = os.getcwd()

# Open data from matlab
#script_path = os.path.dirname(os.path.realpath(__file__))
# Run matlab code, then open data from matlab
regen=False
if regen: # put in a 'regen' flag to allow (non-) regeneration of matlab results
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(script_path)
        eng.addpath('matlab/', nargout=0)
        eng.univ_sim_only_mcmc_test(nargout=0)
        eng.quit()
    except Exception as e:
        print(e)
        print('make sure matlab.engine installed')

multi=True
if not multi:
    matfile = scipy.io.loadmat('%s/data/univ_sim_only_mcmc_test.mat' % test_path)
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
    #nsamp=20
else:
    n_pc = 2
    lamWOs_init = 50000.  # use 0 to use default lamWOs initial value

    matfile = scipy.io.loadmat('%s/data/multi_sim_only_mcmc_test.mat' % test_path)
    y = matfile['y'].T
    y_ind = matfile['y_ind'].T
    x = matfile['x']
    data = SepiaData(x_sim=x[:, 0][:, None], t_sim=x[:, 1][:, None], y_sim=y, y_ind_sim=y_ind)
    data.standardize_y()
    data.transform_xt()
    data.create_K_basis(n_pc=n_pc)
    print(data)
    model = setup_model(data)
    nsamp = int(matfile['nsamp'])
    nburn = int(matfile['nburn'])
    if lamWOs_init > 0:
        model.params.lamWOs.val = np.array([[lamWOs_init]])

t_start = time()
model.do_mcmc(nburn+nsamp)
t_end = time()


#%% Test Predict
np.random.seed(42)
z=model.get_samples(0,sampleset=[1,2,3,4,5],flat=True)
#pred=wPred([0.5,0.5],z,model.num,model.data,returnMuSigma=True)
#pred=wPred(np.array([0.5,0.5,0.6,0.6]).reshape(2,2),z,model.num,model.data,returnMuSigma=True)
pred=wPred([0.5,0.5],z,model.num,model.data,returnMuSigma=True)
print('Samples are:')
print(pred.w)
print('Mu are:')
print(pred.mu)
print('Sigma are:')
print(pred.sigma)

'''
# Matlab code:
rng(42,'twister');
pred=gPredict([0.5,0.5],p.pvals(1:5),p.model,p.data,'returnMuSigma',1);
pred.w, pred.Myhat,pred.Syhat
'''


#%% Plot Predict
#import importlib
#importlib.reload(sepia.SepiaPredict)
np.random.seed(42)

import matplotlib.pyplot as plt

'''
%matlab code
rng(42,'twister')
paramout.model.debugRands=true
nq=10; xpred= [0.5*ones(10,1) linspace(0,1,10)'];
pred=gPredict(xpred,paramout.pvals(100:100:1000),paramout.model,paramout.data)
squeeze(pred.w)
'''

if not multi:
    # Creates dict with each sampled variable name as key, array of samples (nsamp, ...) as value
    sampleset=np.arange(100,1000,100)
    samples=model.get_samples(sampleset=sampleset)

    nq=10
    t=np.linspace(0,1,nq)
    xpred=np.column_stack((np.ones((nq,1))*0.5,t))
    pred=wPred(xpred,samples,model.num,model.data)

    #print(pred.w[0:10])

    plt.figure()
    plt.plot(data.sim_data.t_trans,data.sim_data.y_std)
    plt.plot(np.tile(t,(len(sampleset),1)),np.squeeze(pred.w),'.')
    plt.show()

#%% Obs model run
# Run in sepia/test
os.chdir('sepia/test')

import numpy as np
import scipy.io
from time import time
import scipy.stats
import os

import matlab.engine

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
from sepia.SepiaPredict import wPred
from sepia.SepiaPredict import uvPred



show_figs = True
exclude_burnin = True
n_pc = 2
seed = 42
lamWOs_init = 50000.  # use 0 to use default lamWOs initial value
nsamp = 100
nburn = 0

test_path = os.getcwd()

# Open data from matlab
#script_path = os.path.dirname(os.path.realpath(__file__))
# Run matlab code, then open data from matlab
regen=False
if regen: # put in a 'regen' flag to allow (non-) regeneration of matlab results
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(script_path)
        eng.addpath('matlab/', nargout=0)
        eng.multi_sim_and_obs_mcmc_test(nsamp, nburn, seed, lamWOs_init, n_pc, nargout=0)
        eng.quit()
    except Exception as e:
        print(e)
        print('make sure matlab.engine installed')

np.random.seed(42)
matfile = scipy.io.loadmat('%s/data/multi_sim_and_obs_mcmc_test.mat' % test_path)
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
data.create_D_basis(D=matfile['Dobs'].T)
print(data)

np.random.seed(int(seed))
model = setup_model(data)
if lamWOs_init > 0:
    model.params.lamWOs.val = np.array([[lamWOs_init]])

t_start = time()
model.do_mcmc(nburn + nsamp)
t_end = time()
print('Python mcmc time %0.3g s' % (t_end - t_start))
print('Matlab mcmc time %0.3g s' % matfile['mcmc_time'])

#%%
np.random.seed(seed)
psamps = model.get_samples(0, sampleset=[0, 1, 2, 3, 4], flat=True)
pred = uvPred([0.5], psamps, model.num, model.data, returnMuSigma=True)
