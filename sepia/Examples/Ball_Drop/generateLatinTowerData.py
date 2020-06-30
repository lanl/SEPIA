#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 07:52:25 2020

@author: granthutchings
"""
#%%
import sys
sys.path.append('/Users/granthutchings/opt/anaconda3/lib/python3.7/site-packages')
import numpy as np
import pyDOE # Latin Hypercube
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from invertH import invertHsim, invertHtrue
from sepia.SepiaModelSetup import setup_model
from sepia.SepiaData import SepiaData
from scipy.stats import norm
from importlib import reload

#%%
# directory where data files should be written
datadir = '/Users/granthutchings/Documents/LANL/SEPIA/sepia/Examples/Ball_Drop/data/'

# notes:
# x = R
# theta = C
# y = {h, t}, i.e., pairs of h and t that form a trace when plotted

# imagine the field experiments involve say 4 platforms --> 4 values of h.
# Then for each R, one experiment gives output of 4 h-t pairs (a curve).
# Likewise for the simulator, we have a dense grid of say 100 heights h.
# Then for each setting of {x, theta} = {R, C} we get output of 100 h-t
# pairs.

# I'll make python files to:
# 1. generate the h-t pairs and write them into files. (this file and invertH.py)
# 2. a "runmcmc"-type file that first calls...
# 3. ...a file that reads in the data and packages it appropriately


# generate "field" data and "simulator" data, where the simulator model is
# systematically off from reality.

# true: d2h/dt2 = g - C (dh/dt)^2 / R
# sim:  d2h/dt2 = g - C (dh/dt) / R

# inputs for field experiments: x = R
# inputs for simulator: x = R, theta = C
# We want to calibrate theta in the simulator to match the field data.

# values for generating data
n = 3
n_field_heights = 4
m_heights = 16
h_field = np.linspace(5,20,n_field_heights) # platform heights for the field experiments
h_sim = np.arange(1.5,25,1.5) # grid of heights fed to the simulator
h_dense = np.concatenate((np.arange(0,2,.01),np.arange(2,25,.5))) # a denser grid for drawing the curves
g = 9.8 # gravity
et = 0.01 # observation error on the experimental times
C_true = 0.1 / (4 * np.pi / 3) 
# the coefficient of drag for a smooth sphere is 0.1, and we're
# dividing by 4/3 pi to absorb a constant related to the volume of the
# sphere (not including R)
R = np.array([.1, .2, .4]) # radii of balls to try (in meters)

# get a Latin hypercube sim_designign of 25 points over R_sim, C_sim
m = 25
#sim_design = pyDOE.lhs(2,m)

# Use Kary's sim_designign for testing purposes
sim_design = np.array([
    [0.1239,    0.8024],
    [0.8738,    0.6473],
    [0.6140,    0.3337],
    [0.8833,    0.4783],
    [0.9946,    0.0548],
    [0.1178,    0.9382],
    [0.1805,    0.2411],
    [0.6638,    0.2861],
    [0.2939,    0.1208],
    [0.2451,    0.2397],
    [0.4577,    0.5696],
    [0.4377,    0.8874],
    [0.0737,    0.7384],
    [0.6931,    0.8683],
    [0.4901,    0.7070],
    [0.5953,    0.9828],
    [0.7506,    0.1009],
    [0.7783,    0.4225],
    [0.8333,    0.5318],
    [0.3987,    0.6312],
    [0.2021,    0.4990],
    [0.3495,    0.3680],
    [0.9411,    0.7935],
    [0.0198,    0.0218],
    [0.5440,    0.1925]])
plt.scatter(sim_design[:,0],sim_design[:,1])
plt.show()
# scale the first column to [0,.5] and call it R_sim
# (this inclusim_design our field values, i.e., R \in [0,.5])
# scale the second column to [0.05,.25] and call it Csim
# (likewise, Ctrue \in [0.05, .25])
sim_design[:,0] = sim_design[:,0] * .4 + .05
sim_design[:,1] = sim_design[:,1] * .2 + .05
R_sim = sim_design[:,0]
C_sim = sim_design[:,1]
plt.scatter(R_sim,C_sim)
plt.xlabel("R design points");plt.ylabel("C design points")
plt.title("Simulator Design")
plt.show()

# Generate field data for each R
y_field       = invertHtrue(h_field, g, C_true, R, et) # observed times
y_field_dense = invertHtrue(h_dense, g, C_true, R, et) # dense grid for plots

# imagine that the biggest ball is too big to get to the highest
# platform, so we don't observe data there
#y_field[-1,-1] = np.nan

# Generate simulated data for each (C,R) pair
y_sim       = invertHsim(h_sim,   g, C_sim, R_sim)
y_sim_dense = invertHsim(h_dense, g, C_sim, R_sim)

# NEED TO MAKE THIS WORK FOR MULTIPLE G AS WELL
#g_sim = np.repeat(g,m)
#y_sim       = invertHsim(h_sim,   g_sim, C_sim, R_sim) # this look a long time
#y_sim_dense = invertHsim(h_dense, g_sim, C_sim, R_sim) # this took a really long time

#%%
# create sepia data object
data = SepiaData(x_sim = np.matrix(R_sim).T, t_sim = np.matrix(C_sim).T, y_sim = y_sim, y_ind_sim = h_sim,\
                 x_obs = np.matrix(R).T, y_obs = y_field, y_ind_obs=h_field)
data.transform_xt()
data.standardize_y()
data.create_K_basis(2)
# Generate D_obs matrix
D_grid = h_sim # locations on which the kernels are centered
D_width = 1.5 # width of each kernel
pv = len(D_grid)
D_obs = np.zeros(shape=(len(h_field),pv))
D_sim = np.zeros(shape=(len(h_sim),pv))
# create each kernel
for j in range(pv):
    # create kernel j for each experiment
    D_obs[:,j] = norm.pdf(h_field, D_grid[j], D_width)
    D_sim[:,j] = norm.pdf(h_sim, D_grid[j],D_width)
    
#D_max = np.max(np.matmul(D_sim,D_sim.T))
#D_sim = D_sim / np.sqrt(D_max)
#D_obs = D_obs / np.sqrt(D_obs)

data.create_D_basis(type='linear',D=D_obs.T)

# visualize bases
data.plot_K_basis()
#data.plot_K_weights()
#data.plot_K_residuals()


print(data)

plt.ylim(-.5,2)
plt.plot(h_sim,D_sim)
plt.show()
plt.plot(h_field,D_obs)

#%% Model
model = setup_model(data)

#%% MCMC
model.tune_step_sizes(50, 20)
model.do_mcmc(10000)

#%% Parameters
# Extract MCMC samples into dictionary with parameter names
#samples_dict = {p.name: p.mcmc_to_array(trim=1000, untransform_theta=True) for p in model.params.mcmcList}
samples_dict = {p.name: p.mcmc_to_array(trim=1000,untransform_theta=True) for p in model.params.mcmcList}

# return parameters (C) to native scale
t_min = np.min(C_sim, 0)
t_max = np.max(C_sim, 0)
theta = samples_dict['theta'] * (t_max - t_min) + t_min
plt.hist(theta[:,0])
plt.xlabel("C")

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
        print(k,"- mean:",np.mean(samples_dict[k][:, j]),"std:",np.std(samples_dict[k][:, j]))
    plt.show()

#%% LOO Cross Validation
# =============================================================================
# run_stats = np.zeros((25,2))
# for i in range(m):
#     # delete i'th simulation
#     data = SepiaData(x_sim = np.delete(np.matrix(R_sim).T,i,0),\
#                      t_sim = np.delete(np.matrix(C_sim).T,i,0),\
#                      y_sim = np.delete(y_sim,i,0),y_ind_sim = h_sim,\
#                      x_obs = np.matrix(R).T, y_obs = y_field, y_ind_obs=h_field)
#     data.transform_xt()
#     data.standardize_y()
#     data.create_K_basis(2)
#     # Generate D_obs matrix
#     for j in range(pv):
#         # create kernel j for each experiment
#         D_obs[:,j] = norm.pdf(h_field, D_grid[j], D_width)
#         D_sim[:,j] = norm.pdf(h_sim, D_grid[j],D_width)
#         
#     D_max = np.max(np.matmul(D_sim,D_sim.T))
#     D_sim = D_sim / np.sqrt(D_max)
#     D_obs = D_obs / np.sqrt(D_obs)
#     
#     data.create_D_basis(type='linear',D=D_obs.T)
# 
#     model = setup_model(data)
# 
#     model.tune_step_sizes(50, 20)
#     model.do_mcmc(1000)
#     samples_dict = {p.name: p.mcmc_to_array(trim=100,untransform_theta=True) for p in model.params.mcmcList}
#     t_min = np.min(C_sim, 0)
#     t_max = np.max(C_sim, 0)
#     theta = samples_dict['theta'] * (t_max - t_min) + t_min
#     #plt.hist(theta[:,0])
#     #plt.xlabel("C")
#     run_stats[i,0] = np.mean(theta[:,0])
#     run_stats[i,1] = np.std(theta[:,0])
# =============================================================================


#%% #===================== Plots ===============================#
y_max = max(max(y_field.max(1)),max(y_sim.max(1))) # max of all row maxes for axis limit
# find closest values each R
# ith column of R_nearest_sim_design contains the n_neighbors nearest sim_designign points (by index)
# for ith value of R
n_neighbors = 3
R_nearest_sim_design = np.zeros(shape=(n_neighbors,len(R)),dtype=int)
for i in range(len(R)):
    dist = np.argsort(np.abs(R_sim-R[i]))
    R_nearest_sim_design[:,i] = dist[0:n_neighbors]
    
# Generate plot for each radius
colors = ('r', 'g', 'b')
#fig, axs = plt.subplots(2,2)
#axs = axs.flatten()
fig = plt.figure(figsize=[12,12],constrained_layout=True)
gs = GridSpec(2,2,figure=fig)
axs = np.array([fig.add_subplot(gs[0,0]),\
                fig.add_subplot(gs[0,1]),\
                fig.add_subplot(gs[1,0])])
for i in range(len(R)):
    # axis limits, ticks, and labels
    axs[i].set_xlim([0, 25])
    axs[i].set_ylim([0, y_max+.5])
    axs[i].xaxis.set_ticks(np.arange(0,30,5))
    axs[i].yaxis.set_ticks(np.arange(0,y_max+.5,1))
    axs[i].set_title("Ball Radius {}".format(R[i]),fontweight="bold")
    axs[i].set_xlabel("Height (m)")
    axs[i].set_ylabel("Time (s)")
    
    # simulations - all
    for j in range(m):
        axs[i].plot(h_dense, np.transpose(y_sim_dense)[:,j],color='lightgrey',\
                linestyle="--",label="Simulation runs" if j==0 else "")
    
        # simulations - nearest neighbors
    for j in range(n_neighbors):
        axs[i].plot(h_dense,np.transpose(y_sim_dense)[:,R_nearest_sim_design[j,i]],\
                    linestyle="--",\
                    color=colors[j],label="Nearest Sim {}".format(j+1))
    
    # true data curve and "real data points"
    axs[i].plot(h_dense, y_field_dense[i,:],'k',label="Reality")
    axs[i].plot(h_field, y_field[i,],'ks',label="Field data")
    

    axs[i].legend(loc="lower right")
    

        
    # imbed sim_designign point subplot
    inset_ax = inset_axes(axs[i],width="30%",height="30%",loc="upper left",\
                          borderpad=2.5)
    inset_ax.set_xlabel("R sim_designign values",fontsize=7,labelpad=1)
    inset_ax.set_ylabel("C sim_designign values",fontsize=7)
    inset_ax.xaxis.set_ticks(R)
    inset_ax.yaxis.set_ticks(np.arange(0,.251,.05))
    inset_ax.tick_params(axis='both', which='major', labelsize=7, pad = -5)
    inset_ax.scatter(R_sim,C_sim,s=15, facecolors='none', edgecolors='grey')
    inset_ax.scatter(R_sim[R_nearest_sim_design[:,i]],C_sim[R_nearest_sim_design[:,i]],s=15,\
                     color=colors)
    inset_ax.axvline(x=R[i], ymin=0, ymax=1,color='k',linewidth=.5)
plt.savefig('data/plotAll.png', dpi=300)
plt.show()

#%% #==================== Write data ===========================#
# write the h-t pairs into files
# sim.dat, should be length(hsim) x length(Csim)
with open(datadir+'sim.dat',"w+") as f:
    for line in np.matrix(np.transpose(y_sim)):
        np.savetxt(f, line)

# sim.height, a file with just the heights (same for all sim runs)
with open(datadir+'sim.height',"w+") as f:
    for line in np.matrix(np.transpose(h_sim)):
        np.savetxt(f, line)

# sim.sim_designign, length(Csim) x (num X's + num thetas)
sim_designignout = np.transpose(np.matrix([R_sim, C_sim]))
with open(datadir+'sim.sim_designign',"w+") as f:
    for line in sim_designignout:
        np.savetxt(f, line)

# field.dat, one row per experiment (radius)
with open(datadir+'field.dat',"w+") as f:
    for line in np.matrix(y_field):
        np.savetxt(f, line)

# field.height
with open(datadir+'field.height',"w+") as f:
    for line in np.matrix(h_field):
        np.savetxt(f, line)

# field radii
with open(datadir+'field.radii',"w+") as f:
    for line in np.matrix(R):
        np.savetxt(f, line)
