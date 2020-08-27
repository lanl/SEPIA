#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nklein

Test problem for Sepia. Univariate-output, sim and obs model with univariate x/t. TODO

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
p = 2
q = 2
t = 0.5*np.ones((m, q))
x = np.tile(np.linspace(0, 1, m), (p, 1)).T
y = 5*np.sin(5*x[:, 0]) - 5*x[:, 1]
y = y - np.mean(y)

# Create SepiaData object and standardize y before calling SepiaModel
data = SepiaData(x_sim=x, t_sim=t, y_sim=y)
data.standardize_y()

# Call SepiaModel to do a bunch of stuff
g = SepiaModel(data)

# Test logPost call
print('log post initial %0.3g' % g.logPost())

# Sample from post
g.do_mcmc(5000)
print(f'beta={g.params.beta.mcmc.draws[-1]}, lam={g.params.lam.mcmc.draws[-1]}, lp={g.params.lp.mcmc.draws[-1]}')

# For visualization, get predictions on original scale (hold second variable constant)
tp = 0.5*np.ones((100, q))
xp1 = np.concatenate([np.linspace(0, 1, 100)[:, None], 0.5*np.ones((100, 1))], axis=1)
ypred1 = (g.predict(xp1, tp, samples=100) + data.sim_data.orig_y_mean) * data.sim_data.orig_y_sd
xp2 = np.concatenate([0.5*np.ones((100, 1)), np.linspace(0, 1, 100)[:, None]], axis=1)
ypred2 = (g.predict(xp2, tp, samples=100) + data.sim_data.orig_y_mean) * data.sim_data.orig_y_sd

# plot results
if True:
    plt.figure(1)
    plt.subplot(121)
    plt.plot(x[:, 0], y, 'rx')
    plt.plot(xp1[:, 0], np.mean(ypred1, axis=0))
    plt.plot(xp1[:, 0], np.quantile(ypred1, 0.1, axis=0), 'b:')
    plt.plot(xp1[:, 0], np.quantile(ypred1, 0.9, axis=0), 'b:')
    plt.subplot(122)
    plt.plot(x[:, 1], y, 'rx')
    plt.plot(xp2[:, 1], np.mean(ypred2, axis=0))
    plt.plot(xp2[:, 1], np.quantile(ypred2, 0.1, axis=0), 'b:')
    plt.plot(xp2[:, 1], np.quantile(ypred2, 0.9, axis=0), 'b:')
    plt.show()

    plt.figure(2)
    plt.subplot(311)
    plt.plot(g.params.beta.mcmc.draws)
    plt.subplot(312)
    plt.plot(g.params.lam.mcmc.draws)
    plt.subplot(313)
    plt.plot(g.params.lp.mcmc.draws)
    plt.show()


