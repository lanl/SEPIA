#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:26:27 2020

@author: granthutchings
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sns.set(style="ticks")

def theta_pairs(samples_dict,design_names=[]):
    theta = samples_dict['theta']  
    if not design_names:
        for i in range(theta.shape[1]):
            design_names.append('theta_'+str(i+1))
            
    thin_idx = np.linspace(0,theta.shape[0]-1,1000,dtype=int) # thin to 1000 samples
    theta_df = pd.DataFrame(theta[thin_idx,:],columns=design_names) # take only 1000 samples to dataframe
    theta_df.insert(0,'idx',theta_df.index,allow_duplicates = False)
    sns.set_palette(sns.color_palette("RdBu_r", theta_df.shape[0]))
    if theta_df.shape[1]>2:
        with sns.plotting_context("notebook"):
            sns.set_palette(sns.color_palette("RdBu_r", theta_df.shape[0]))
            g = sns.PairGrid(theta_df.loc[:, theta_df.columns != 'idx'], diag_sharey=False)
            g.map_upper(sns.scatterplot, hue=theta_df['idx'], legend=False)
            g.map_lower(sns.kdeplot, colors="C0")
            g.map_diag(sns.distplot, hist=True)
            plt.show()
    else:
        sns.distplot(theta_df.loc[:, theta_df.columns != 'idx'],hist=True,axlabel=design_names[0])
        plt.show()
        
def mcmc_trace(samples_dict,theta_names=None,by_group=False,max_print=5):
    if not by_group:
        for i, k in enumerate(samples_dict.keys()):
            n_row = min(samples_dict[k].shape[1],max_print)
            fig, axs = plt.subplots(n_row,1,sharex=True,figsize=[10,1.5*n_row])
            fig.tight_layout()
            if n_row > 1:
                for j in range(n_row):
                    sns.lineplot(data=samples_dict[k][:,j], palette="tab10", linewidth=.75, ax = axs[j])
                    if i == 0 and theta_names is not None: axs[j].set_xlabel(theta_names[j])
                    else: axs[j].set_xlabel(k+'_'+str(j+1))
                plt.show()
            else:
                sns.lineplot(data=samples_dict[k][:,0], palette="tab10", linewidth=.75, ax = axs)
                if i == 0 and theta_names is not None: axs.set_xlabel(theta_names[0])
                else: axs.set_xlabel(k)
                plt.show()
    else:
        for i, k in enumerate(samples_dict.keys()):
            n_row = min(samples_dict[k].shape[1],max_print)
            fig, axs = plt.subplots(1,1,figsize=[10,1.5*n_row])
            if n_row > 1:
                for j in range(n_row):
                    sns.lineplot(data=samples_dict[k][:,j], palette="tab10", linewidth=.75, ax = axs,
                                label=k+str(j+1))
                    if i == 0: axs.set_xlabel('theta')
                axs.legend()
                plt.show()
            else:
                sns.lineplot(data=samples_dict[k][:,0], palette="tab10", linewidth=.75, ax = axs)
                if i == 0: axs.set_xlabel('theta')
                else: axs.set_xlabel(k)
                plt.show()
         
def param_stats(samples_dict,theta_names=None,q1=0.05,q2=0.95,digits=4):
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
    samples_dict = {p.name: p.mcmc_to_array(trim=1000,untransform_theta=True) for p in model.params.mcmcList}
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