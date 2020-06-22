import numpy as np
import matplotlib.pyplot as plt

from setup_test_cases import *

# A manual example of a fixed-parameter model to test it out

m = 50
n = 5
nt_sim = 75
nt_obs = 50
n_pc = 5  # must be smaller than nt
nx = 4
noise_sd = 0.1
model, _ = setup_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd, nx=nx, n_pc=n_pc, seed=42.)

# We are going to fix some of the parameters now manually, then run MCMC
model.params.betaU.fixed[0, 0:3] = True
model.params.lamUz.fixed[0, 1:5] = True

model.do_mcmc(100)

samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}
samples_dict['logPost'] = np.array(model.params.lp.mcmc.draws).reshape((-1, 1))

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