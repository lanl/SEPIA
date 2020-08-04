#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sns.set(style="ticks")

def theta_pairs(samples_dict,design_names=[]):
    """
    Create pairs plot of sampled thetas.

    :param samples_dict: dictionary -- samples from model.get_samples()
    :param design_names: list -- names for thetas, optional
    """
    theta = samples_dict['theta']  
    if not design_names:
        for i in range(theta.shape[1]):
            design_names.append('theta_'+str(i+1))
            
    thin_idx = np.linspace(0,theta.shape[0]-1,1000,dtype=int) # thin to 1000 samples
    theta_df = pd.DataFrame(theta[thin_idx,:],columns=design_names) # take only 1000 samples to dataframe
    theta_df.insert(0,'idx',theta_df.index,allow_duplicates = False)
    if theta_df.shape[1]>2:
        with sns.plotting_context("notebook"):
            g = sns.PairGrid(theta_df.loc[:, theta_df.columns != 'idx'], diag_sharey=False)
            g.map_upper(sns.scatterplot, palette = 'coolwarm', hue=theta_df['idx'], legend=False)
            g.map_lower(sns.kdeplot)
            g.map_diag(sns.distplot, hist=True)
            plt.show()
    else:
        sns.distplot(theta_df.loc[:, theta_df.columns != 'idx'],hist=True,axlabel=design_names[0])
        plt.show()
        
def mcmc_trace(samples_dict,theta_names=None,start=0,end=None,n_to_plot=500,by_group=True,max_print=10):
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
    n_samples = samples_dict['theta'].shape[0]
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
            n_row = min(samples_dict[k].shape[1],max_print)
            fig, axs = plt.subplots(n_row,1,sharex=True,figsize=[10,1.5*n_row])
            fig.subplots_adjust(hspace=0)
            if n_row > 1:
                for j in range(n_row):
                    sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,j], palette="tab10", linewidth=.75, ax = axs[j])
                    if i == 0 and theta_names is not None: axs[j].set_ylabel(theta_names[j])
                    else: axs[j].set_ylabel(k+'_'+str(j+1))
                plt.show()
            else:
                sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,0], palette="tab10", linewidth=.75, ax = axs)
                if i == 0 and theta_names is not None: axs.set_ylabel(theta_names[0])
                else: axs.set_ylabel(k)
                plt.show()
    else:
        n_axes = len(samples_dict)
        fig, axs = plt.subplots(n_axes,1,sharex=True,figsize=[10,1.5*n_axes])
        fig.subplots_adjust(hspace=0)
        for i, k in enumerate(samples_dict.keys()):
            n_lines = min(samples_dict[k].shape[1],max_print)
            if n_lines > 1:
                for j in range(n_lines):
                    sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,j], palette="tab10", linewidth=.75, ax = axs[i],
                                label= theta_names[j] if (i==0 and theta_names is not None) else k+str(j+1))
                axs[i].set_ylabel(k)
                axs[i].legend(bbox_to_anchor=(1.025, 1), loc='upper left', borderaxespad=0., ncol=int(np.ceil(n_lines/5)))
            else:
                sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,0], palette="tab10", linewidth=.75, ax = axs[i])
                axs[i].set_ylabel(theta_names[0] if (i==0 and theta_names is not None) else k)
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
        
def plot_acf(chain,nlags,alpha=.05):
    
    # compute autocorrelation
    if np.ndim(chain) == 1:
        chain = chain.reshape(1,chain.shape[0])
    nchains, nobs = chain.shape
    autocorrs = []
    for i in range(nchains):
        chain1 = (chain[i,:] - np.mean(chain[i,:])) / (np.std(chain[i,:]) * nobs)
        chain2 = (chain[i,:] - np.mean(chain[i,:])) / np.std(chain[i,:])
        autocorr = np.correlate(chain1,chain2,mode='full')
        autocorr = autocorr[autocorr.size//2:]
        autocorrs.append(autocorr[0:nlags+1])

    # plot
    fig, ax = plt.subplots()
    lags = np.linspace(0,nlags,nlags+1,dtype=int,endpoint=True)
    for i in range(len(autocorrs)):
        ax.plot(lags,autocorrs[i],'-o',label='theta {}'.format(i+1))
    ax.set_xticks(np.linspace(0,nlags,10,dtype=int,endpoint=True))
    ax.set_yticks(np.linspace(0,1,11,endpoint=True))
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    
    # significance lines
    sigline = stats.norm.ppf(1 - alpha / 2.) / np.sqrt(nobs)
    ax.axhline(-sigline,linestyle='--',c='b'); ax.axhline(sigline,linestyle='--',c='b')
    
    # axis limits
    
    ymin = min(np.min(-sigline),min([np.min(ac) for ac in autocorrs]))
    ymax = max(np.max(sigline),max([np.max(ac) for ac in autocorrs]))
    ax.set_ylim([min(-.1,1.1*ymin),1.1*ymax])
    
    if nchains > 1: plt.legend()
    plt.show()
    
    # output ESS to console
    ess = []
    for i in range(nchains):
        ess.append(ESS(chain[i,:]))
    print('Effective Sample Sizes:',ess)
    print('Total number of samples:',[nobs]*nchains)
    
    return autocorrs, sigline

def ESS(x):
    """ Compute the effective sample size of estimand of interest. Vectorised implementation. """
    if np.ndim(x) == 1:
        x = x.reshape(1,-1)

    m_chains, n_iters = x.shape

    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    post_var = marg_post_var(x)

    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0

        t += 1

    return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

def marg_post_var(x):
    """ Estimate the marginal posterior variance. Vectorised implementation. """
    m_chains, n_iters = x.shape

    # Calculate between-chain variance
    if m_chains > 1:
        B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)
    else:
        B_over_n = 0

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))

    # (over) estimate of variance
    s2 = W * (n_iters - 1) / n_iters + B_over_n

    return s2