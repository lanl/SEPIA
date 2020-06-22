# Do sepia MCMC on Al-Al data to compare with matlab

import numpy as np
import scipy.io
from time import time
import scipy.stats
import pickle

np.random.seed(42)

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model


matfile = scipy.io.loadmat('data/alal_data.mat')

nmcmc = matfile['nmcmc'].item()
lamWs_a = matfile['lamWs_a'].item()
lamWs_b = matfile['lamWs_b'].item()
lamWs_upper = matfile['lamWs_upper'].item()
n_pc = matfile['pu'].item()
y_sim = matfile['y_sim'].T
y_ind_sim = np.linspace(0, 1, y_sim.shape[1])
xt_sim = matfile['xt_sim']
y_obs = matfile['y_obs'].T
y_ind_obs = y_ind_sim
x_obs = matfile['x_obs']
data = SepiaData(x_sim=xt_sim[:, 0][:, None], t_sim=xt_sim[:, 1:], y_sim=y_sim, y_ind_sim=y_ind_sim,
                 x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)
data.standardize_y()
data.transform_xt()
data.create_K_basis(n_pc=n_pc)
data.create_D_basis(D=matfile['Dobs'].T)
print(data)

model = setup_model(data, Sigy=matfile['Sigy'])
model.params.lamWs.prior.bounds[1] = lamWs_upper
model.params.lamWs.prior.params = [lamWs_a, lamWs_b]

# to compile numba
model.logLik()

print('start mcmc')
t_start = time()
model.do_mcmc(nmcmc)
t_end = time()
print('Python mcmc time %0.5g s' % (t_end - t_start))

mat_mcmc = scipy.io.loadmat('data/alal_result.mat')
print('Matlab mcmc time %0.5g s' % (mat_mcmc['mcmc_time']))

samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}
log_post = np.array(model.params.lp.mcmc.draws)[1:].squeeze()

pickle.dump(samples_dict, open('data/alal_result.pkl', 'wb'))

betaU_matlab = mat_mcmc['betaU_samp']
lamUz_matlab = mat_mcmc['lamUz_samp']
betaV_matlab = mat_mcmc['betaV_samp']
lamVz_matlab = mat_mcmc['lamVz_samp']
lamWs_matlab = mat_mcmc['lamWs_samp']
lamWOs_matlab = mat_mcmc['lamWOs_samp']
lamOs_matlab = mat_mcmc['lamOs_samp']
theta_matlab = mat_mcmc['theta_samp']
log_post_matlab = mat_mcmc['logPost_trace']

import matplotlib.pyplot as plt

if 'betaU' in samples_dict.keys():
    for i in range(betaU_matlab.shape[1]):
        plt.figure()
        plt.hist(samples_dict['betaU'][:, i], alpha=0.5)
        plt.hist(betaU_matlab[:, i], alpha=0.5)
        plt.title('betaU %d' % i)
        plt.legend(['python', 'matlab'])
        plt.savefig('figures/python_mcmc_betaU%d.png' % i)
        plt.close()

if 'betaV' in samples_dict.keys():
    plt.figure(2)
    plt.hist(samples_dict['betaV'], alpha=0.5)
    plt.hist(betaV_matlab, alpha=0.5)
    plt.title('betaV')
    plt.legend(['python', 'matlab'])
    plt.savefig('figures/python_mcmc_betaV.png')
    plt.close()

if 'lamUz' in samples_dict.keys():
    for i in range(lamUz_matlab.shape[1]):
        plt.figure()
        plt.hist(samples_dict['lamUz'][:, i], alpha=0.5)
        plt.hist(lamUz_matlab[:, i], alpha=0.5)
        plt.title('lamUz %d' % i)
        plt.legend(['python', 'matlab'])
        plt.savefig('figures/python_mcmc_lamUz%d.png' % i)
        plt.close()

if 'lamVz' in samples_dict.keys():
    plt.figure()
    plt.hist(samples_dict['lamVz'], alpha=0.5)
    plt.hist(lamVz_matlab, alpha=0.5)
    plt.title('lamVz')
    plt.legend(['python', 'matlab'])
    plt.savefig('figures/python_mcmc_lamVz.png')
    plt.close()

if 'lamWOs' in samples_dict.keys():
    plt.figure()
    plt.hist(samples_dict['lamWOs'], alpha=0.5)
    plt.hist(lamWOs_matlab, alpha=0.5)
    plt.title('lamWOs')
    plt.legend(['python', 'matlab'])
    plt.savefig('figures/python_mcmc_lamWOs.png')
    plt.close()

if 'lamWs' in samples_dict.keys():
    for i in range(lamWs_matlab.shape[1]):
        plt.figure()
        plt.hist(samples_dict['lamWs'][:, i], alpha=0.5)
        plt.hist(lamWs_matlab[:, i], alpha=0.5)
        plt.title('lamWs %d' % i)
        plt.legend(['python', 'matlab'])
        plt.savefig('figures/python_mcmc_lamWs%d.png' % i)
        plt.close()

if 'lamOs' in samples_dict.keys():
    plt.figure()
    plt.hist(samples_dict['lamOs'], alpha=0.5)
    plt.hist(lamOs_matlab, alpha=0.5)
    plt.title('lamOs')
    plt.legend(['python', 'matlab'])
    plt.savefig('figures/python_mcmc_lamOs.png')
    plt.close()

if 'theta' in samples_dict.keys():
    for i in range(theta_matlab.shape[1]):
        plt.figure()
        plt.hist(samples_dict['theta'][:, i], alpha=0.5)
        plt.hist(theta_matlab[:, i], alpha=0.5)
        plt.title('theta %d' % i)
        plt.legend(['python', 'matlab'])
        plt.savefig('figures/python_mcmc_theta%d.png' % i)
        plt.close()

plt.figure()
plt.plot(log_post_matlab, 'b')
plt.plot(log_post, 'r--')
plt.xlabel('draw number')
plt.ylabel('logpost')
plt.legend(['matlab', 'python'])
plt.savefig('figures/python_mcmc_logpost.png')
plt.close()