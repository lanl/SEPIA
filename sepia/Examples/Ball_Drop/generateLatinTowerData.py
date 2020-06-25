#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 07:52:25 2020

@author: granthutchings
"""
import sys
sys.path.append('/Users/granthutchings/opt/anaconda3/lib/python3.7/site-packages')
import numpy as np
import pyDOE # Latin Hypercube
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from invertH import *
import numpy as np

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
n_field_heights = 4
n_sim_heights = 16
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

# get a Latin hypercube design of 25 points over R_sim, C_sim
n_sim = 25
#des = pyDOE.lhs(2,n_sim)

# Use Kary's design for testing purposes
des = np.array([
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
plt.scatter(des[:,0],des[:,1])
plt.show()

# scale the first column to [0,.5] and call it R_sim
# (this includes our field values, i.e., R \in [0,.5])
# scale the second column to [0.05,.25] and call it Csim
# (likewise, Ctrue \in [0.05, .25])
R_sim = des[:,0] * .4 + .05;
C_sim = des[:,1] * .2 + .05;
plt.scatter(R_sim,C_sim)
plt.show()

# Generate field data for each R
t_field       = invertHtrue(h_field, g, C_true, R, et) # observed times
t_field_dense = invertHtrue(h_dense, g, C_true, R, et) # dense grid for plots

# imagine that the biggest ball is too big to get to the highest
# platform, so we don't observe data there
t_field[-1,-1] = np.nan

# Generate simulated data for each (C,R) pair
t_sim       = invertHsim(h_sim,   g, C_sim, R_sim)
t_sim_dense = invertHsim(h_dense, g, C_sim, R_sim)

# NEED TO MAKE THIS WORK FOR MULTIPLE G AS WELL
#g_sim = np.repeat(g,n_sim)
#t_sim       = invertHsim(h_sim,   g_sim, C_sim, R_sim) # this look a long time
#t_sim_dense = invertHsim(h_dense, g_sim, C_sim, R_sim) # this took a really long time


#%% #===================== Plots ===============================#
y_max = max(max(t_field.max(1)),max(t_sim.max(1))) # max of all row maxes for axis limit
# find closest values each R
# ith column of R_nearest_des contains the n_neighbors nearest design points (by index)
# for ith value of R
n_neighbors = 3
R_nearest_des = np.zeros(shape=(n_neighbors,len(R)),dtype=int)
for i in range(len(R)):
    dist = np.argsort(np.abs(R_sim-R[i]))
    R_nearest_des[:,i] = dist[0:n_neighbors]
    
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
    for j in range(n_sim):
        axs[i].plot(h_dense, np.transpose(t_sim_dense)[:,j],color='lightgrey',\
                linestyle="--",label="Simulation runs" if j==0 else "")
    
        # simulations - nearest neighbors
    for j in range(n_neighbors):
        axs[i].plot(h_dense,np.transpose(t_sim_dense)[:,R_nearest_des[j,i]],\
                    linestyle="--",\
                    color=colors[j],label="Nearest Sim {}".format(j+1))
    
    # true data curve and "real data points"
    axs[i].plot(h_dense, t_field_dense[i,:],'k',label="Reality")
    axs[i].plot(h_field, t_field[i,],'ks',label="Field data")
    

    axs[i].legend(loc="lower right")
    

        
    # imbed design point subplot
    inset_ax = inset_axes(axs[i],width="30%",height="30%",loc="upper left",\
                          borderpad=2.5)
    inset_ax.set_xlabel("R design values",fontsize=7,labelpad=1)
    inset_ax.set_ylabel("C design values",fontsize=7)
    inset_ax.xaxis.set_ticks(R)
    inset_ax.yaxis.set_ticks(np.arange(0,.251,.05))
    inset_ax.tick_params(axis='both', which='major', labelsize=7, pad = -5)
    inset_ax.scatter(R_sim,C_sim,s=15, facecolors='none', edgecolors='grey')
    inset_ax.scatter(R_sim[R_nearest_des[:,i]],C_sim[R_nearest_des[:,i]],s=15,\
                     color=colors)
    inset_ax.axvline(x=R[i], ymin=0, ymax=1,color='k',linewidth=.5)
plt.savefig('data/plotAll.png', dpi=300)
plt.show()





#%% #==================== Write data ===========================#
# write the h-t pairs into files
# sim.dat, should be length(hsim) x length(Csim)
with open(datadir+'sim.dat',"w+") as f:
    for line in np.matrix(np.transpose(t_sim)):
        np.savetxt(f, line)

# sim.height, a file with just the heights (same for all sim runs)
with open(datadir+'sim.height',"w+") as f:
    for line in np.matrix(np.transpose(h_sim)):
        np.savetxt(f, line)

# sim.design, length(Csim) x (num X's + num thetas)
designout = np.transpose(np.matrix([R_sim, C_sim]))
with open(datadir+'sim.design',"w+") as f:
    for line in designout:
        np.savetxt(f, line)

# field.dat, one row per experiment (radius)
with open(datadir+'field.dat',"w+") as f:
    for line in np.matrix(t_field):
        np.savetxt(f, line)

# field.height
with open(datadir+'field.height',"w+") as f:
    for line in np.matrix(h_field):
        np.savetxt(f, line)

# field radii
with open(datadir+'field.radii',"w+") as f:
    for line in np.matrix(R):
        np.savetxt(f, line)
