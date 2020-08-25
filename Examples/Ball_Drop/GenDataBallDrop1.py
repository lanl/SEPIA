#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 07:52:25 2020
Generate, Plot, and write all data needed for ball drop example 1
@author: granthutchings
"""
#%% Imports
import numpy as np
#import pyDOE # Latin Hypercube
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from invertH import invertHsim, invertHtrue

#%% notes
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

#%% Compute data
def gen_data(et,plot_design=False,R_new=None,R_design=None,C_design=None):
    n = 3; m = 25
    g = 9.8 # gravity
    C_true = .1 / (4 * np.pi / 3); print('generating data with C = ',C_true)
    n_field_heights = 4
    h_field = np.linspace(5,20,n_field_heights) # platform heights for the field experiments
    h_sim = np.arange(1.5,25,1.5) # grid of heights fed to the simulator
    h_dense = np.concatenate((np.arange(0,2,.01),np.arange(2,25,.5))) # a denser grid for drawing the curves

    # the coefficient of drag for a smooth sphere is 0.1, and we're
    # dividing by 4/3 pi to absorb a constant related to the volume of the
    # sphere (not including R)
    if R_new is None: R = np.array([.1, .2, .4]) # radii of balls to try (in meters)
    else: R = R_new

    # get a Latin hypercube sim_design of m=25 points over R_sim, C_sim
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
 
    # scale the first column to [0,.5] and call it R_sim
    # (this inclusim_design our field values, i.e., R \in [0,.5])
    # scale the second column to [0.05,.25] and call it Csim
    # (likewise, Ctrue \in [0.05, .25])
    sim_design[:,0] = sim_design[:,0] * .4 + .05
    sim_design[:,1] = sim_design[:,1] * .2 + .05
    if R_design is not None: R_sim = R_design
    else: R_sim = sim_design[:,0]
    if C_design is not None: C_sim = C_design
    else: C_sim = sim_design[:,1]
    if plot_design:
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
    
    data_dict = dict([('R',R),('sim_design',np.column_stack((R_sim,C_sim))),\
                      ('n',n),('m',m),('C_true',C_true),\
                      ('h_field',h_field),('h_sim',h_sim),('h_dense',h_dense),\
                      ('y_field',y_field),('y_field_dense',y_field_dense),\
                      ('y_sim',y_sim),('y_sim_dense',y_sim_dense)])
    
    return(data_dict)

#%% #===================== Plots ===============================#
def plot_data(data_dict,inset=True,near_sim=True):
    n = data_dict['n']
    m = data_dict['m']
    y_sim = data_dict['y_sim']
    y_field = data_dict['y_field']
    R = data_dict['R']
    R_sim = data_dict['sim_design'][:,0]
    C_sim = data_dict['sim_design'][:,1]
    h_field = data_dict['h_field']
    h_sim = data_dict['h_sim']
    h_dense = data_dict['h_dense']
    y_field = data_dict['y_field']
    y_field_dense = data_dict['y_field_dense']
    y_sim = data_dict['y_sim']
    y_sim_dense = data_dict['y_sim_dense']
    
    if isinstance(y_field, list): ragged = True
    else: ragged = False
    
    if ragged:
        y_max = max(max(np.array([np.max(k) for k in y_field])),max(y_sim.max(1)))
    else:
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
        axs[i].set_title("Ball Radius {} m".format(R[i]),fontweight="bold")
        axs[i].set_xlabel("Distance (m)")
        axs[i].set_ylabel("Time (s)")

        # simulations - all
        for j in range(m):
            axs[i].plot(h_dense, np.transpose(y_sim_dense)[:,j],color='lightgreen',\
                    label="Simulation runs" if j==0 else "")

        if near_sim:
            # simulations - nearest neighbors
            for j in range(n_neighbors):
                axs[i].plot(h_dense,np.transpose(y_sim_dense)[:,R_nearest_sim_design[j,i]],\
                            linestyle="--",\
                            color=colors[j],label="Nearest Sim {}".format(j+1))

        # true data curve and "real data points"
        axs[i].plot(h_dense, y_field_dense[i,:],'k',label="Reality")
        if ragged:
            axs[i].plot(h_field[i],y_field[i],'ks',label='Reality')
        else:
            axs[i].plot(h_field, y_field[i,],'ks',label="Field data")


        axs[i].legend(loc="lower right")


        if inset:
            # imbed sim_designign point subplot
            inset_ax = inset_axes(axs[i],width="30%",height="30%",loc="upper left",\
                                  borderpad=2.5)
            inset_ax.set_xlabel("R sim_design values",fontsize=7,labelpad=1)
            inset_ax.set_ylabel("C sim_design values",fontsize=7)
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
def write_data(data_dict, datadir = '/Users/granthutchings/Documents/LANL/SEPIA/sepia/Examples/Ball_Drop/data/ball_drop_1'):
    # datadir == directory where data files should be written to or read from

    # sim.dat, should be length(hsim) x length(Csim)
    y_sim = data_dict['y_sim']
    with open(datadir+'sim.dat',"w+") as f:
        for line in np.array(np.transpose(y_sim)):
            np.savetxt(f, line)

    # sim.height, a file with just the heights (same for all sim runs)
    h_sim = data_dict['h_sim']
    with open(datadir+'sim.height',"w+") as f:
        for line in np.array(np.transpose(h_sim)):
            np.savetxt(f, line)

    # sim.sim_designign, length(Csim) x (num X's + num thetas)
    R_sim = data_dict['R_sim']; C_sim = data_dict['C_sim']    
    sim_design = np.transpose(np.array([R_sim, C_sim]))
    with open(datadir+'sim.design',"w+") as f:
        for line in sim_design:
            np.savetxt(f, line)

    # field.dat, one row per experiment (radius)
    y_field = data_dict['y_field']
    with open(datadir+'field.dat',"w+") as f:
        for line in np.array(y_field):
            np.savetxt(f, line)

    # field.height
    h_field = data_dict['h_field']
    with open(datadir+'field.height',"w+") as f:
        for line in np.array(h_field):
            np.savetxt(f, line)

    # field radii
    R = data_dict['R']
    with open(datadir+'field.radii',"w+") as f:
        for line in np.array(R):
            np.savetxt(f, line)
            
#%%
def read_data(datadir = '/Users/granthutchings/Documents/LANL/SEPIA/sepia/Examples/Ball_Drop/data/ball_drop_1'):
    
    with open(datadir+'sim.dat','r') as f:
        y_sim = np.loadtxt(f)
    with open(datadir+'sim.height',"r") as f:
        h_sim = np.loadtxt(f)
    with open(datadir+'sim.design','r') as f:
        sim_design = np.loadtxt(f)
    with open(datadir+'field.dat','r') as f:
        y_field = np.loadtxt(f)
    with open(datadir+'field.height','r') as f:
        h_field = np.loadtxt(f)
    with open(datadir+'field.radii','r') as f:
        R = np.loadtxt(f)
    
    data_dict = dict([('R',R),('sim_design',sim_design),\
                  ('h_field',h_field),('h_sim',h_sim),\
                  ('y_field',y_field),('y_sim',y_sim)])
    
    return(data_dict)
    
        
        
            
    
