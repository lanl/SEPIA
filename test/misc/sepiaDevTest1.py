#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gatt

Test problem for Sepia. Univariate-output, eta-only model with univariate x/t.

"""

#import scipy.io
#matdata=scipy.io.loadmat('refModel.mat')

import matplotlib.pyplot as plt
import numpy as np
from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData

np.random.seed(42)

# Generate data for eta-only univariate-output model (t is a dummy, x is the input index)
m = 20
t = 0.5*np.ones((m, 1))
x = np.linspace(0, 1, m)[:, None]
y = 10 * x**2 * np.cos(10*x)
y = y - np.mean(y)
xp = np.linspace(0, 1, 100)[:, None]
tp = 0.5*np.ones((100, 1))

# Create SepiaData object and standardize y before calling setup_model
data = SepiaData(x_sim=x, t_sim=t, y_sim=y)
print(data)

data.standardize_y()
data.transform_xt()

# Call setup_model to do a bunch of stuff
model = SepiaModel(data)

# To start using the full covariance calculation
#g.fullCov=True

# Test logPost call
print(model.log_prior())
print(model.logPost())

# Sample from post
mcsamp=100
model.do_mcmc(mcsamp)
samples_dict = {p.name: np.array(p.mcmc.draws[1:]) for p in model.params.mcmcList}
samples_dict['lp'] = np.array(model.params.lp.mcmc.draws[1:])

for i, k in enumerate(samples_dict.keys()):
    plt.figure(i+1)
    samples_tmp = samples_dict[k].reshape(mcsamp, -1)
    plt.hist(np.delete(samples_tmp, np.isinf(samples_tmp)))
    plt.title(k)
    plt.show()

# Below: old code, relies on predict working
# print(f'beta={g.params.beta.mcmc.draws[-1]}, lam={g.params.lam.mcmc.draws[-1]}, lp={g.params.lp.mcmc.draws[-1]}')
#
# # For visualization, get predictions on original scale
# ypred = (g.predict(xp, tp, samples=min(mcsamp,100)) + data.sim_data.y_mean) * data.sim_data.y_sd
#
# # plot results
# if True:
#     plt.figure(1)
#     plt.plot(x, y, 'rx')
#     plt.plot(xp, np.mean(ypred, axis=0))
#     plt.plot(xp, np.quantile(ypred, 0.1, axis=0), 'b:')
#     plt.plot(xp, np.quantile(ypred, 0.9, axis=0), 'b:')
#     plt.show()
#
#     plt.figure(2)
#     plt.subplot(311)
#     plt.plot(g.params.beta.mcmc.draws)
#     plt.subplot(312)
#     plt.plot(g.params.lam.mcmc.draws)
#     plt.subplot(313)
#     plt.plot(g.params.lp.mcmc.draws)
#     plt.show()


