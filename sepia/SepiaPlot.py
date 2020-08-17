#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sns.set(style="ticks")

def theta_pairs(samples_dict,design_names=[],native=False,lims=None,theta_ref=None,save=False):
    """
    Create pairs plot of sampled thetas.

    :param samples_dict: dictionary -- samples from model.get_samples()
    :param design_names: list -- names for thetas, optional
    :param native: boolean -- whether to put theta on native scale (note: you may want to pass lims in this case)
    :param lims: list of tuples -- optional, limits for each theta value for plotting; defaults to [0, 1] if native=False
    :param vlines: list -- optional, scalar values to place vlines on diagonal distplots
    """
    if native is False:
        theta = samples_dict['theta']
    else:
        theta = samples_dict['theta_native']
    n_theta = theta.shape[1]
    if native is False and lims is None:
        lims = [(0, 1) for i in range(n_theta)]
    if not design_names:
        for i in range(theta.shape[1]):
            design_names.append('theta_'+str(i+1))
    thin_idx = np.linspace(0,theta.shape[0]-1,1000,dtype=int) # thin to 1000 samples
    theta_df = pd.DataFrame(theta[thin_idx,:],columns=design_names) # take only 1000 samples to dataframe
    theta_df.insert(0,'idx',theta_df.index,allow_duplicates = False)
    if theta_df.shape[1]>2:
        g = sns.PairGrid(theta_df.loc[:, theta_df.columns != 'idx'], diag_sharey=False)
        g.map_upper(sns.scatterplot, palette = 'coolwarm', hue=theta_df['idx'], legend=False)
        g.map_lower(sns.kdeplot, cmap="viridis", shade=True, n_levels=10, shade_lowest=False)
        g.map_diag(sns.distplot, hist=True)
        if lims is not None:
            # Undo sharing of axes
            for i in range(n_theta):
                [g.diag_axes[i].get_shared_x_axes().remove(axis) for axis in g.axes.ravel()]
                for j in range(n_theta):
                    [g.axes[i, j].get_shared_x_axes().remove(axis) for axis in g.axes.ravel()]
                    [g.axes[i, j].get_shared_y_axes().remove(axis) for axis in g.axes.ravel()]
                    [g.axes[i, j].get_shared_x_axes().remove(axis) for axis in g.diag_axes.ravel()]
                    [g.axes[i, j].get_shared_y_axes().remove(axis) for axis in g.diag_axes.ravel()]
            # Set limits
            for i in range(n_theta):
                for j in range(n_theta):
                    if i == j:
                        g.diag_axes[i].set_xlim(xmin=lims[i][0], xmax=lims[i][1])
                        g.axes[i, i].set_xlim(xmin=lims[i][0], xmax=lims[i][1])
                    else:
                        g.axes[i, j].set_xlim(xmin=lims[j][0], xmax=lims[j][1])
                        g.axes[i, j].set_ylim(ymin=lims[i][0], ymax=lims[i][1])
                        
        if theta_ref is not None:
            for i in range(n_theta):
                g.diag_axes[i].vlines(theta_ref[i],ymin=0,ymax=1,transform = g.diag_axes[i].get_xaxis_transform(),color='r')
                for j in range(n_theta):
                    if i>j: # Lower diag contour plots
                        g.axes[i,j].scatter(theta_ref[j], theta_ref[i], marker='o', s=5, color="red")
        if save: 
            plt.tight_layout()
            plt.savefig("theta_pairs.png",dpi=300)
        plt.show()
    else:
        sns.distplot(theta_df.loc[:, theta_df.columns != 'idx'],hist=True,axlabel=design_names[0])
        if save: 
            plt.tight_layout()
            plt.savefig("theta_pairs.png",dpi=300)
        plt.show()
        
def mcmc_trace(samples_dict,theta_names=None,start=0,end=None,n_to_plot=500,by_group=True,max_print=10,save=False):
    """
    Create trace plot of MCMC.

    :param samples_dict: dictionary -- samples from model.get_samples()
    :param theta_names: list -- names for thetas, optional
    :param start: int -- where to start plotting traces (sample index)
    :param end: int -- where to end plotting traces (sample index)
    :param n_to_plot: int -- how many samples to show
    :param by_group: boolean -- whether to group params of the same name
    :param max_print: int -- maximum number of traces to plot
    """
    # trim samples dict
    n_samples = samples_dict['lamUz'].shape[0]
    if n_to_plot>n_samples:
        n_to_plot = n_samples
    # default end
    if end is None:
        end = n_samples-1
    # check start is valid
    if not isinstance(start,int) or start<0 :
        raise TypeError('invalid start index')
    # check end is valid
    if end is not None and (start>end or end<0 or not isinstance(end,int) or end > n_samples):
        raise TypeError('invalid end index')
    # which indices to plot  
    if (end-start) > n_to_plot:
        plot_idx = np.unique(np.linspace(start,end,n_to_plot,dtype=int))
    else:
        plot_idx = np.arange(start,end,1,dtype=int)
    
    if not by_group:
        for i, k in enumerate(samples_dict.keys()):
            if k == 'theta_native':
                continue
            n_row = min(samples_dict[k].shape[1],max_print)
            fig, axs = plt.subplots(n_row,1,sharex=True,figsize=[10,1.5*n_row])
            fig.subplots_adjust(hspace=0)
            if n_row > 1:
                for j in range(n_row):
                    sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,j], palette="tab10", linewidth=.75, ax = axs[j])
                    if i == 0 and theta_names is not None: axs[j].set_ylabel(theta_names[j])
                    else: axs[j].set_ylabel(k+'_'+str(j+1))
                if save: plt.savefig("mcmc_trace.png",dpi=300,bbox_extra_artists=(lgd), bbox_inches='tight')
                plt.show()
            else:
                sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,0], palette="tab10", linewidth=.75, ax = axs)
                if i == 0 and theta_names is not None: axs.set_ylabel(theta_names[0])
                else: axs.set_ylabel(k)
                if save: plt.savefig("mcmc_trace.png",dpi=300,bbox_extra_artists=(lgd), bbox_inches='tight')
                plt.show()
    else:
        lgds = []
        n_axes = len(samples_dict)-1 # wont be plotting theta_native
        fig, axs = plt.subplots(n_axes,1,sharex=True,figsize=[10,1.5*n_axes])
        fig.subplots_adjust(hspace=0)
        for i, k in enumerate(samples_dict.keys()):
            if k == 'theta_native':
                continue
            n_lines = min(samples_dict[k].shape[1],max_print)
            if n_lines > 1:
                for j in range(n_lines):
                    sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,j], palette="tab10", linewidth=.75, ax = axs[i],
                                label= theta_names[j] if (i==0 and theta_names is not None) else k+str(j+1))
                axs[i].set_ylabel(k)
                lgds.append(axs[i].legend(bbox_to_anchor=(1.025, 1), loc='upper left', borderaxespad=0., ncol=int(np.ceil(n_lines/5))))
            else:
                sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,0], palette="tab10", linewidth=.75, ax = axs[i])
                axs[i].set_ylabel(theta_names[0] if (i==0 and theta_names is not None) else k)
        if save: plt.savefig("mcmc_trace.png",dpi=300,bbox_extra_artists=lgds, bbox_inches='tight')
        plt.show()
         
def param_stats(samples_dict,theta_names=None,q1=0.05,q2=0.95,digits=4):
    """
    Compute statistics on the samples.

    :param samples_dict: dictionary -- samples from model.get_samples()
    :param theta_names: list -- names for thetas, optional
    :param q1: float -- lower quantile in [0, 1]
    :param q2: float -- upper quantile in [0, 1]
    :param digits: int -- how many digits to show in output

    """
    # theta_names : list
    # samples_dict : dictionary of samples
    # stats : dataframe with mean and std of all parameters
    n_theta = samples_dict['theta'].shape[1]
    if theta_names is not None and len(theta_names) != n_theta:
        print('theta_names should have',n_theta, 'entries')
        return
    
    mean = []
    sd = []
    keys = []
    q1_list = []
    q2_list = []
    for i, k in enumerate(samples_dict.keys()):
        n_param = samples_dict[k].shape[1]
        for j in range(n_param):
            mean.append(np.round(np.mean(samples_dict[k][:, j]),digits))
            sd.append(np.round(np.std(samples_dict[k][:, j]),digits))
            q1_list.append(np.round(np.quantile(samples_dict[k][:, j],q1),digits))
            q2_list.append(np.round(np.quantile(samples_dict[k][:, j],q2),digits))
            if i==0 and theta_names is not None: keys.append(theta_names[j])
            elif n_param>1: keys.append(k+'_'+str(j+1))
            else: keys.append(k)
    stats = pd.DataFrame({'mean':mean,'sd':sd,'{} quantile'.format(q1):q1_list,\
                          '{} quantile'.format(q2):q2_list},index=keys)
    return(stats)

def rho_box_plots(model,labels=None):
    """
    Show rho box plots. (Transformed betaU, lengthscale)

    :param model: SepiaModel object
    :param labels: list
    """
    samples_dict = {p.name: p.mcmc_to_array(trim=1000) for p in model.params.mcmcList}
    p = model.num.p
    q = model.num.q
    pu = model.num.pu
    bu = samples_dict['betaU']
    ru = np.exp(-bu / 4)
    for i in range(pu):
        r = ru[:, ((p+q)*i):((p+q)*i)+(p+q)]
        plt.boxplot(r)
        if labels is not None: plt.xticks(np.linspace(1,len(labels),len(labels),dtype=int),labels)
        plt.yticks(np.arange(0,1.2,.2))
        plt.ylabel(r'$\rho$')
        plt.title('PC {}'.format(i+1))
        plt.show() 
        
def plot_acf(model,nlags,nburn=0,alpha=.05,save=False):
    """
    Plot autocorrelation function for all parameters theta
    
    :param model: SepiaModel object
    :param nlags: int -- how many lags to compute/plot
    :param nburn: int -- how many samples to burn
    :param alpha: float -- confidence level for acf significance line (0,1)
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in (0,1)')
    # get theta chains
    for p in model.params.mcmcList:
        if p.name == 'theta': 
            chain = p.mcmc_to_array(trim=nburn, flat=True).T
    
    acf = model.acf(chain,nlags,plot=True,save=save,alpha=alpha)
    return acf
