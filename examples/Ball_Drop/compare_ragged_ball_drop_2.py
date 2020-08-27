#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:28:40 2020

@author: granthutchings
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
#import pyDOE
from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
import sepia.SepiaPlot as SepiaPlot
import os

#%% non-ragged data and model setup
datadir = os.getcwd()+"/data/data_ball_drop_2/"
# experimental data
with open(datadir+'fieldDat12x6gparam.txt','r') as f:
    field_data = np.loadtxt(f)
with open(datadir+'desNative4x2Rrho.txt','r') as f:
    x = np.loadtxt(f) # x = {R, rho_ball}
    
x_obs = x
y_obs = np.reshape(field_data[:,4],(4,3)) # observed times, experiments by row
h_obs = field_data[0:3,3]                 # observed heights

print(y_obs)
print(h_obs)

# simulated data
with open(datadir+'desNative80x4Cg.txt','r') as f:
    sim_data = np.loadtxt(f)
x_sim = sim_data[:,0:2] # x = {R, rho_ball}
t_sim = sim_data[:,2:4] # t = {C, g}
with open(datadir+'simHeights101x1','r') as f:
    h_sim = np.loadtxt(f)
with open(datadir+'sims101x80Cg.txt','r') as f:
    y_sim = np.loadtxt(f).T

# create sepia data object
data = SepiaData(x_sim = x_sim, t_sim = t_sim, y_ind_sim = h_sim, y_sim = y_sim,\
                 x_obs = x_obs, y_obs = y_obs, y_ind_obs = h_obs)
data.transform_xt()
data.standardize_y()
data.create_K_basis(3)
data.create_D_basis('linear')
print(data)
model = SepiaModel(data)

#%% Ragged data and model setup
y_obs_ragged = [np.array(field_data[0:3,4]),np.array(field_data[3:6,4]),\
         np.array(field_data[6:9,4]),np.array(field_data[9:,4])]
h_obs_ragged = [np.array(field_data[0:3,3]),np.array(field_data[3:6,3]),\
         np.array(field_data[6:9,3]),np.array(field_data[9:,3])]# observed heights

#y_obs = [np.array(field_data[0:3,4]),np.array(field_data[3:6,4]),\
#         np.array(field_data[[7,9,11],4]),np.array(field_data[12:,4])]
#h_obs = [np.array(field_data[0:3,3]),np.array(field_data[3:6,3]),\
#         np.array(field_data[[7,9,11],3]),np.array(field_data[12:,3])]# observed heights

print(y_obs)
print(h_obs)

# create sepia data object
data_ragged = SepiaData(x_sim = x_sim, t_sim = t_sim, y_ind_sim = h_sim, y_sim = y_sim,\
                 x_obs = x_obs, y_obs = y_obs_ragged, y_ind_obs = h_obs_ragged)
data_ragged.transform_xt()
data_ragged.standardize_y()
data_ragged.create_K_basis(3)
data_ragged.create_D_basis('linear')
print(data_ragged)
model_ragged = SepiaModel(data_ragged)

#%% Check for same model elements
print('logLik',model.logLik(),model_ragged.logLik())
model.logPost()
model_ragged.logPost()
print('logPost',model.logPost(),model_ragged.logPost())
print('logLik',model.num.logLik,model_ragged.num.logLik)

print('vuw',np.all(np.isclose(model.num.vuw,model_ragged.num.vuw)))
print('LamSim',np.all(np.isclose(model.num.LamSim,model_ragged.num.LamSim)))
print('SigObs',np.all(np.isclose(model.num.SigObs,model_ragged.num.SigObs)))

print('SigU',np.all(np.isclose(model.num.SigU,model_ragged.num.SigU)))
print('SigUW',np.all(np.isclose(model.num.SigUW,model_ragged.num.SigUW)))
print('SigV',np.all(np.isclose(model.num.SigV,model_ragged.num.SigV)))
print('SigWi',np.all(np.isclose(model.num.SigWi,model_ragged.num.SigWi)))
print('SigWl',np.all(np.isclose(model.num.SigWl,model_ragged.num.SigWl)))

# check shapes in case ragged is doing wierd things
# print('SigU shapes')
# print(len(model.num.SigU),len(model_ragged.num.SigU))
# print([tmp.shape for tmp in model.num.SigU])
# print([tmp.shape for tmp in model_ragged.num.SigU])

# print('SigUW shapes')
# print(len(model.num.SigUW),len(model_ragged.num.SigUW))
# print([tmp.shape for tmp in model.num.SigUW])
# print([tmp.shape for tmp in model_ragged.num.SigUW])

# print('SigV shapes')
# print(len(model.num.SigV),len(model_ragged.num.SigV))
# print([tmp.shape for tmp in model.num.SigV])
# print([tmp.shape for tmp in model_ragged.num.SigV])

# print('SigWi shapes')
# print(len(model.num.SigWi),len(model_ragged.num.SigWi))
# print([tmp.shape for tmp in model.num.SigWi])
# print([tmp.shape for tmp in model_ragged.num.SigWi])

# print('SigWl shapes')
# print(len(model.num.SigWl),len(model_ragged.num.SigWl))
# print([tmp.shape for tmp in model.num.SigWl])
# print([tmp.shape for tmp in model_ragged.num.SigWl])

print('D_obs',np.all(np.isclose(model.data.obs_data.D,model_ragged.data.obs_data.D[0])))
print('K_obs',np.all(np.isclose(model.data.obs_data.K,model_ragged.data.obs_data.K[0])))
print('D_sim',np.all(np.isclose(model.data.sim_data.D,model_ragged.data.sim_data.D)))
print('K_sim',np.all(np.isclose(model.data.sim_data.K,model_ragged.data.sim_data.K)))

#%% MCMC
#model.tune_step_sizes(50, 20)
np.random.seed(42)
model.do_mcmc(5000)
samples_dict = {p.name: p.mcmc_to_array(trim=1000,untransform_theta=True) for p in model.params.mcmcList}
theta_names = ['C','g']
SepiaPlot.theta_pairs(samples_dict,theta_names)

#model_ragged.tune_step_sizes(50, 20)
np.random.seed(42)
model_ragged.do_mcmc(5000)
samples_dict_ragged = {p.name: p.mcmc_to_array(trim=1000,untransform_theta=True) for p in model_ragged.params.mcmcList}
SepiaPlot.theta_pairs(samples_dict_ragged,theta_names)
