"""
Sets up test cases for comparing lik/mcmc/stepsize/predict between python and matlab.
"""

import numpy as np
import matlab.engine

from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel

import os
import sys
#root_path = os.path.dirname(sys.modules['__main__'].__file__)
root_path = os.path.dirname(os.path.realpath(__file__))


def setup_univ_sim_only(m=300, seed=42., n_lik=0, n_mcmc=0, n_pred=0, n_lev=0, n_burn=0, sens=0):
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(root_path)
        eng.addpath('matlab/', nargout=0)
        res = eng.setup_univ_sim_only(m, seed, n_lik, n_mcmc, n_pred, n_lev, n_burn, sens, nargout=1)
        eng.quit()
    except Exception as e:
        print(e)
        print('Matlab error; make sure matlab.engine installed, check Matlab code for errors.')
    y = np.array(res['y'], dtype=float)
    xt = np.array(res['xt'], dtype=float)
    data = SepiaData(x_sim=xt[:, 0][:, None], t_sim=xt[:, 1][:, None], y_sim=y)
    print(data)
    data.standardize_y()
    data.transform_xt()
    model = SepiaModel(data)
    return model, res


def setup_univ_sim_and_obs(m=100, n=50, seed=42., n_lik=0, n_mcmc=0, n_pred=0):
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(root_path)
        eng.addpath('matlab/', nargout=0)
        res = eng.setup_univ_sim_and_obs(m, n, seed, n_lik, n_mcmc, n_pred, nargout=1)
        eng.quit()
    except Exception as e:
        print(e)
        print('Matlab error; make sure matlab.engine installed, check Matlab code for errors.')
    y = np.array(res['y'], dtype=float)
    xt = np.array(res['xt'], dtype=float)
    y_obs = np.array(res['y_obs'], dtype=float)
    x_obs = np.array(res['x_obs'], dtype=float).reshape((n, 1))
    data = SepiaData(x_sim=xt[:, 0][:, None], t_sim=xt[:, 1][:, None], y_sim=y, x_obs=x_obs, y_obs=y_obs)
    data.standardize_y()
    data.transform_xt()
    print(data)
    model = SepiaModel(data)
    return model, res


def setup_multi_sim_only(m=300, nt=20, nx=5, n_pc=10, seed=42., n_lik=0, n_mcmc=0, n_pred=0, fix_K=False, sens=0):
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(root_path)
        eng.addpath('matlab/', nargout=0)
        res = eng.setup_multi_sim_only(m, nt, nx, n_pc, seed, n_lik, n_mcmc, n_pred, sens, nargout=1)
        eng.quit()
    except Exception as e:
        print(e)
        print('Matlab error; make sure matlab.engine installed, check Matlab code for errors.')
    y = np.array(res['y'], dtype=float)
    y_ind = np.array(res['y_ind'], dtype=float).squeeze()
    xt = np.array(res['xt'], dtype=float)
    data = SepiaData(x_sim=xt[:, 0][:, None], t_sim=xt[:, 1:], y_sim=y, y_ind_sim=y_ind)
    data.standardize_y()
    data.transform_xt()
    data.create_K_basis(n_pc)
    if fix_K:
        data.sim_data.K = np.array(res['K']).T
    print(data)
    model = SepiaModel(data)
    return model, res


def setup_multi_sim_and_obs(m=100, n=10, nt_sim=20, nt_obs=15, noise_sd=0.1, nx=5, n_pc=10, seed=42., n_lik=0, n_mcmc=0, n_pred=0, fix_K=False):
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(root_path)
        eng.addpath('matlab/', nargout=0)
        res = eng.setup_multi_sim_and_obs(m, n, nt_sim, nt_obs, noise_sd, nx, n_pc, seed, n_lik, n_mcmc, n_pred, nargout=1)
        eng.quit()
    except Exception as e:
        print(e)
        print('Matlab error; make sure matlab.engine installed, check Matlab code for errors.')
    y = np.array(res['y'], dtype=float)
    y_ind = np.array(res['y_ind'], dtype=float).squeeze()
    xt = np.array(res['xt'], dtype=float)
    y_obs = np.array(res['y_obs'], dtype=float)
    y_ind_obs = np.array(res['y_ind_obs'], dtype=float).squeeze()
    x_obs = np.array(res['x_obs'], dtype=float)
    data = SepiaData(x_sim=xt[:, 0][:, None], t_sim=xt[:, 1:], y_sim=y, y_ind_sim=y_ind,
                     x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)
    data.standardize_y()
    data.transform_xt()
    if fix_K: # means use the K from matlab - avoid issues with positive/negative component ambiguity
        data.create_K_basis(n_pc, K=np.array(res['K']).T )
    else:
        data.create_K_basis(n_pc)
    data.create_D_basis('constant')
    print(data)
    model = SepiaModel(data)
    return model, res


def setup_multi_sim_and_obs_noD(m=100, n=10, nt_sim=20, nt_obs=15, noise_sd=0.1, nx=5, n_pc=10, seed=42., n_lik=0, n_mcmc=0):
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(root_path)
        eng.addpath('matlab/', nargout=0)
        res = eng.setup_multi_sim_and_obs_noD(m, n, nt_sim, nt_obs, noise_sd, nx, n_pc, seed, n_lik, n_mcmc, nargout=1)
        eng.quit()
    except Exception as e:
        print(e)
        print('Matlab error; make sure matlab.engine installed, check Matlab code for errors.')
    y = np.array(res['y'], dtype=float)
    y_ind = np.array(res['y_ind'], dtype=float).squeeze()
    xt = np.array(res['xt'], dtype=float)
    y_obs = np.array(res['y_obs'], dtype=float)
    y_ind_obs = np.array(res['y_ind_obs'], dtype=float).squeeze()
    x_obs = np.array(res['x_obs'], dtype=float)
    data = SepiaData(x_sim=xt[:, 0][:, None], t_sim=xt[:, 1:], y_sim=y, y_ind_sim=y_ind,
                     x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)
    data.standardize_y()
    data.transform_xt()
    data.create_K_basis(n_pc)
    print(data)
    model = SepiaModel(data)
    return model, res

def setup_multi_sim_and_obs_sharedtheta(m=100, n=10, nt_sim=20, nt_obs=15, noise_sd=0.1, nx=5, n_pc=10, seed=42., n_lik=0,
                                        n_mcmc=0, n_pred=0, n_shared=2, clist=[], fix_K=False):
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(root_path)
        eng.addpath('matlab/', nargout=0)
        res = eng.setup_multi_sim_and_obs_sharedtheta(m, n, nt_sim, nt_obs, noise_sd, nx, n_pc, seed, n_lik, n_mcmc, n_pred,
                                                      n_shared, matlab.double(clist), nargout=1)
        eng.quit()
    except Exception as e:
        print(e)
        print('Matlab error; make sure matlab.engine installed, check Matlab code for errors.')
    y = np.array(res['y'], dtype=float) # (m, nt_sim, n_shared)
    y_ind = np.array(res['y_ind'], dtype=float).squeeze() # (nt_sim, n_shared)
    xt = np.array(res['xt'], dtype=float) # (m, nx, n_shared)
    y_obs = np.array(res['y_obs'], dtype=float) # (n, nt_sim, n_shared)
    y_ind_obs = np.array(res['y_ind_obs'], dtype=float).squeeze() # (nt_obs, n_shared)
    x_obs = np.array(res['x_obs'], dtype=float) # (n, 1, n_shared)
    model_list = []
    for i in range(n_shared):
        data = SepiaData(x_sim=xt[:, 0, i][:, None], t_sim=xt[:, 1:, i], y_sim=y[:, :, i], y_ind_sim=y_ind[:, i],
                         x_obs=x_obs[:, :, i], y_obs=y_obs[:, :, i], y_ind_obs=y_ind_obs[:, i])
        data.standardize_y()
        data.transform_xt()
        data.create_K_basis(n_pc)
        model = SepiaModel(data)
        model_list.append(model)
    return model_list, res

def setup_multi_sim_and_obs_hiertheta(m=100, n=10, nt_sim=20, nt_obs=15, noise_sd=0.1, nx=5, n_pc=10, seed=42., n_lik=0,
                                        n_mcmc=0, n_pred=0, n_shared=2, fix_K=False):
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(root_path)
        eng.addpath('matlab/', nargout=0)
        res = eng.setup_multi_sim_and_obs_hiertheta(m, n, nt_sim, nt_obs, noise_sd, nx, n_pc, seed, n_lik, n_mcmc, n_pred,
                                                      n_shared, nargout=1)
        eng.quit()
    except Exception as e:
        print(e)
        print('Matlab error; make sure matlab.engine installed, check Matlab code for errors.')
    y = np.array(res['y'], dtype=float) # (m, nt_sim, n_shared)
    y_ind = np.array(res['y_ind'], dtype=float).squeeze() # (nt_sim, n_shared)
    xt = np.array(res['xt'], dtype=float) # (m, nx, n_shared)
    y_obs = np.array(res['y_obs'], dtype=float) # (n, nt_sim, n_shared)
    y_ind_obs = np.array(res['y_ind_obs'], dtype=float).squeeze() # (nt_obs, n_shared)
    x_obs = np.array(res['x_obs'], dtype=float) # (n, 1, n_shared)
    model_list = []
    for i in range(n_shared):
        data = SepiaData(x_sim=xt[:, 0, i][:, None], t_sim=xt[:, 1:, i], y_sim=y[:, :, i], y_ind_sim=y_ind[:, i],
                         x_obs=x_obs[:, :, i], y_obs=y_obs[:, :, i], y_ind_obs=y_ind_obs[:, i])
        data.standardize_y()
        data.transform_xt()
        data.create_K_basis(n_pc)
        model = SepiaModel(data)
        model_list.append(model)
    return model_list, res

def setup_neddermeyer(seed=42.,n_mcmc=100,sens=1,n_burn=0,n_lev=0):
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(root_path)
        eng.addpath('matlab/', nargout=0)
        #dataStruct = eng.neddeg(0, nargout=1)
        res = eng.setup_neddermeyer(seed,n_mcmc,sens,n_burn,n_lev)
        eng.quit()
    except Exception as e:
        print(e)
        print('Matlab error; make sure matlab.engine installed, check Matlab code for errors.')
    
    # get python model
    import pickle
    data = pickle.load(open('../../examples/Neddermeyer/pkls/nedderData.pkl','rb'))
    model=SepiaModel(data)
    
    return model, res

if __name__ == '__main__':
    setup_multi_sim_and_obs_sharedtheta(n_mcmc=20, clist=[[1, 1], [2, 2]])
    setup_multi_sim_only()
    setup_univ_sim_only()
    setup_univ_sim_and_obs()
    setup_neddermeyer()


