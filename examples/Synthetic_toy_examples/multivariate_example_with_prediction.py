#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from generate_data import generate_multi_sim_and_obs

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData

# #%load_ext autoreload
# #%autoreload 2

#%%

seed = 42   # random seed
m = 100     # number of simulated observations
n = 1       # number of observed data
sig_n = 0.01 # observation noise SD

data_dict = generate_multi_sim_and_obs(m=m, n=n, sig_n=sig_n, seed=seed)

#%%

data = SepiaData(t_sim=data_dict['t_sim'], y_sim=data_dict['y_sim'], y_ind_sim=data_dict['y_ind_sim'],
                 y_obs=data_dict['y_obs'], y_ind_obs=data_dict['y_ind_obs'])

print(data)

plt.plot(data.sim_data.y_ind, data.sim_data.y.T)
plt.plot(data.obs_data.y_ind, data.obs_data.y.T, 'k.', linewidth=3)
plt.title('Synthetic data (obs. in black)')
plt.xlabel('y index')
plt.ylabel('y')
plt.show()

#%%

data.transform_xt()
data.standardize_y(scale='columnwise')
data.create_K_basis(5)
data.create_D_basis(D_type='linear')

print(data)

#%%

model = SepiaModel(data)

#%%

cachefile_name='multivariate_example_with_prediction.pkl'
import os.path
import pickle

use_save_file=False

if use_save_file and os.path.isfile(cachefile_name):
   model=pickle.load(open(cachefile_name, "rb" ))
else:
    model.tune_step_sizes(50, 20)
    model.do_mcmc(1000)
    if use_save_file:
        pickle.dump(model, open( cachefile_name, "w+b" ))

#%%

# Extract MCMC samples into dictionary with parameter names
samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}

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
        plt.hist(samples_dict[k][:, j])
        plt.xlabel(k)
    plt.show()


#%% Emulator-only predictions

model.verbose=True

from sepia.SepiaPredict import SepiaEmulatorPrediction
# make t_pred most like the calibrated thetas for comparison to below
t_pred=np.tile(np.array([1,0,1]).reshape(1,3),(9,1)) 
pred_samples=model.get_samples(numsamples=7)
pred=SepiaEmulatorPrediction(samples=pred_samples, model=model, t_pred=t_pred)

predw=pred.get_w()
plt.figure()
plt.plot(np.mean(predw,0).T)
plt.xlabel('PCA weights w')
plt.ylabel('w value for each predicted x \n mean over samples')
plt.show()

predystd=pred.get_y(std=True)
plt.figure()
plt.plot(model.data.sim_data.y_ind,np.mean(predystd,0).T)
plt.xlabel('native data space y')
plt.ylabel('standardized predicted y for each predicted x \n mean over samples')
plt.show()

predy=pred.get_y()
plt.figure()
plt.plot(model.data.sim_data.y_ind,np.mean(predy,0).T)
plt.xlabel('native data space y ')
plt.ylabel('native (original) predicted value for each x_pred \n mean over samples')
plt.show()

predy=pred.get_y()
plt.figure()
plt.plot(model.data.sim_data.y_ind,np.mean(predy[:,4,:],0).T)
plt.plot(model.data.sim_data.y_ind,np.quantile(predy[:,4,:],[0.05, 0.95],axis=0).T,'k:')
plt.xlabel('native data space y ')
plt.ylabel('native (original) mean and (0.05,0.95) quantile \n for one x prediction')
plt.show()



#%% Full model predictions

from sepia.SepiaPredict import SepiaFullPrediction
pred_samples=model.get_samples(numsamples=7)
pred=SepiaFullPrediction(samples=pred_samples, model=model, t_pred=t_pred)

plt.figure()
predu, predv = pred.get_u_v()
plt.subplot(2,1,1)
plt.plot(np.mean(predu,0).T)
plt.xlabel('PCA weights u')
plt.ylabel('u value for each predicted x \n mean over samples')
plt.subplot(2,1,2)
plt.plot(np.mean(predv,0).T,'*')
plt.xlabel('PCA weight v (scalar)')
plt.ylabel('v value for each predicted x \n mean over samples')
plt.show()

predysimstd=pred.get_ysim(std=True)
plt.figure()
plt.plot(model.data.sim_data.y_ind,np.mean(predysimstd,0).T)
plt.xlabel('native data space y')
plt.ylabel('standardized predicted ysim for each predicted x \n mean over samples')
plt.show()

preddstd=pred.get_discrepancy(std=True,as_obs=True)
predd=pred.get_discrepancy(as_obs=True)
plt.figure()
plt.subplot(2,1,1)
plt.plot(model.data.obs_data.y_ind,np.mean(preddstd,0).T)
plt.xlabel('native data space y \n (\delta is constant in this example)')
plt.ylabel('standardized predicted discrepancy\n mean over samples')
plt.subplot(2,1,2)
plt.plot(model.data.obs_data.y_ind,np.mean(predd,0).T)
plt.xlabel('native data space y \n (\delta is constant in this example)')
plt.ylabel('native predicted discrepancy \n mean over samples')
plt.show()

predyobs=pred.get_yobs(as_obs=True)
plt.figure()
plt.plot(model.data.obs_data.y_ind,np.mean(predyobs,0).T)
plt.xlabel('native data space yobs')
plt.ylabel('native predicted yobs for each predicted x \n mean over samples')
plt.show()


#%%

predyobs=pred.get_yobs(as_obs=True)
predysim=pred.get_ysim()
plt.figure()
plt.plot(model.data.obs_data.y_ind,model.data.obs_data.y.T,'r')
plt.plot(model.data.obs_data.y_ind,np.mean(predyobs[:,4,:],0).T,'k--')
plt.plot(model.data.obs_data.y_ind,np.quantile(predyobs[:,4,:],[0.05, 0.95],axis=0).T,'k:')
plt.plot(model.data.sim_data.y_ind,np.mean(predysim[:,4,:],0).T,'g--')
plt.xlabel('native scale (obs grid)')
plt.ylabel('native scale response')
plt.title('Observation and model prediction')
plt.legend(['obs','pred mean','pred 10/95', '_nolegend_', 'pred sim'])
plt.show()



#%%

data2 = SepiaData(t_sim=data_dict['t_sim'], y_sim=data_dict['y_sim'], y_ind_sim=data_dict['y_ind_sim'])
data2.transform_xt()
data2.standardize_y()
data2.create_K_basis(5)

print(data2)

model2 = SepiaModel(data2)


#%%

model2.tune_step_sizes(10, 20)
model2.do_mcmc(100)


#%% Emulator-only predictions

model2.verbose=True

from sepia.SepiaPredict import SepiaEmulatorPrediction
# make t_pred most like the calibrated thetas for comparison to below
t_pred=np.tile(np.array([1,0,1]).reshape(1,3),(9,1))
pred_samples=model.get_samples(numsamples=7)
pred2=SepiaEmulatorPrediction(samples=pred_samples, model=model2, t_pred=t_pred)

predw=pred2.get_w()
plt.figure()
plt.plot(np.mean(predw,0).T)
plt.xlabel('PCA weights w')
plt.ylabel('w value for each predicted x \n mean over samples')
plt.show()

predystd=pred2.get_y(std=True)
plt.figure()
plt.plot(model.data.sim_data.y_ind,np.mean(predystd,0).T)
plt.xlabel('native data space y')
plt.ylabel('standardized predicted y for each predicted x \n mean over samples')
plt.show()

predy=pred2.get_y()
plt.figure()
plt.plot(model.data.sim_data.y_ind,np.mean(predy,0).T)
plt.xlabel('native data space y ')
plt.ylabel('native (original) predicted value for each x_pred \n mean over samples')
plt.show()

predy=pred2.get_y()
plt.figure()
plt.plot(model.data.sim_data.y_ind,np.mean(predy[:,4,:],0).T)
plt.plot(model.data.sim_data.y_ind,np.quantile(predy[:,4,:],[0.05, 0.95],axis=0).T,'k:')
plt.xlabel('native data space y ')
plt.ylabel('native (original) mean and (0.05,0.95) quantile \n for one x prediction')
plt.show()

