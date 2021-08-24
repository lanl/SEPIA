#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

sns.set(style="ticks")

def theta_pairs(samples_dict,design_names=None,native=False,lims=None,theta_ref=None,save=None):
    """
    Create pairs plot of sampled thetas.

    :param dict samples_dict: samples from model.get_samples()
    :param list/NoneType design_names: list of string names for thetas, optional (None will use default names)
    :param bool native: put theta on native scale? (note: you likely want to pass lims in this case)
    :param list lims: list of tuples, limits for each theta value for plotting; defaults to [0, 1] if native=False
    :param list theta_ref: scalar reference values to plot as vlines on histplots and as red dots on bivariate plots
    :param str save: file name to save plot
    :returns: matplotlib figure
    """
    if 'theta' not in samples_dict.keys():
        print('No thetas to plot')
        return
    if native is False:
        theta = samples_dict['theta']
    else:
        theta = samples_dict['theta_native']
    n_samp, n_theta = theta.shape
    if native is False and lims is None:
        lims = [(0, 1) for i in range(n_theta)]
    if isinstance(design_names, list) and len(design_names) != n_theta:
        raise ValueError('Design names wrong length')
    if design_names is None:
        design_names = ['theta_%d' % (i+1) for i in range(n_theta)]
    thin_idx = np.linspace(0,n_samp-1,np.min([n_samp-1, 1000]),dtype=int) # thin to at most 1000 samples
    theta_df = pd.DataFrame(theta[thin_idx,:],columns=design_names) # take only 1000 samples to dataframe
    theta_df.insert(0,'idx',theta_df.index,allow_duplicates = False)
    
    if theta_df.shape[1]>2:
        g = sns.PairGrid(theta_df.loc[:, theta_df.columns != 'idx'], diag_sharey=False);
        g.map_upper(sns.scatterplot, palette = 'coolwarm', hue=theta_df['idx'], legend=False);
        g.map_lower(sns.kdeplot, cmap="viridis", fill=True, thresh=0.05);
        g.map_diag(sns.histplot, kde=True);
        if lims is not None:
            # Undo sharing of axes
            for i in range(n_theta):
                [g.diag_axes[i].get_shared_x_axes().remove(axis) for axis in g.axes.ravel()];
                for j in range(n_theta):
                    [g.axes[i, j].get_shared_x_axes().remove(axis) for axis in g.axes.ravel()];
                    [g.axes[i, j].get_shared_y_axes().remove(axis) for axis in g.axes.ravel()];
                    [g.axes[i, j].get_shared_x_axes().remove(axis) for axis in g.diag_axes.ravel()];
                    [g.axes[i, j].get_shared_y_axes().remove(axis) for axis in g.diag_axes.ravel()];
            # Set limits
            for i in range(n_theta):
                for j in range(n_theta):
                    if i == j:
                        g.diag_axes[i].set_xlim(xmin=lims[i][0], xmax=lims[i][1]);
                        g.axes[i, i].set_xlim(xmin=lims[i][0], xmax=lims[i][1]);
                    else:
                        g.axes[i, j].set_xlim(xmin=lims[j][0], xmax=lims[j][1]);
                        g.axes[i, j].set_ylim(ymin=lims[i][0], ymax=lims[i][1]);
                        
        if theta_ref is not None:
            for i in range(n_theta):
                g.diag_axes[i].vlines(theta_ref[i],ymin=0,ymax=1,transform = g.diag_axes[i].get_xaxis_transform(),color='r');
                for j in range(n_theta):
                    if i>j: # Lower diag contour plots
                        g.axes[i,j].scatter(theta_ref[j], theta_ref[i], marker='o', s=5, color="red");
        if save is not None: 
            plt.tight_layout()
            plt.savefig(save,dpi=300,bbox_inches='tight')
        return g.fig
    else:
        fig,ax=plt.subplots()
        ax.set_xlabel(design_names[0])
        sns.histplot(theta_df.loc[:, theta_df.columns != 'idx'],kde=True,ax=ax)
        if save is not None: 
            plt.tight_layout()
            plt.savefig(save,dpi=300,bbox_inches='tight')
        return fig
        
def mcmc_trace(samples_dict,theta_names=None,start=0,end=None,n_to_plot=500,by_group=True,max_print=10,save=None):
    """
    Create trace plot of MCMC samples.

    :param dict samples_dict: samples from model.get_samples()
    :param list/NoneType theta_names: list of string names for thetas, optional (None will use default names)
    :param int start: where to start plotting traces (sample index)
    :param int/NoneType end: where to end plotting traces (sample index)
    :param int n_to_plot: how many samples to show
    :param bool by_group: group params of the same name onto one axis?
    :param int max_print: maximum number of traces to plot
    :param str save: file name to save plot
    :returns: matplotlib figure
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
        total_plots = 0
        for i,k in enumerate(samples_dict.keys()):
            if k == 'theta_native':
                continue
            total_plots += min(samples_dict[k].shape[1],max_print)
        fig,axs = plt.subplots(total_plots,1,sharex=True,figsize=[10,1.5*total_plots])
        fig.subplots_adjust(hspace=0)
        axs_idx = 0
        for i, k in enumerate(samples_dict.keys()):
            if k == 'theta_native':
                continue
            n_theta = min(samples_dict[k].shape[1],max_print)
            if n_theta > 1:
                for j in range(n_theta):
                    sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,j], palette="tab10", linewidth=.75, ax = axs[axs_idx])
                    if k=='theta' and theta_names is not None: axs[axs_idx].set_ylabel(theta_names[j])
                    else: axs[axs_idx].set_ylabel(k+'_'+str(j+1))
                    axs_idx+=1
            else:
                sns.lineplot(x=plot_idx,y=samples_dict[k][plot_idx,0], palette="tab10", linewidth=.75, ax = axs[axs_idx])
                if k=='theta' and theta_names is not None: axs.set_ylabel(theta_names[0])
                else: axs[axs_idx].set_ylabel(k)
                axs_idx+=1
        if save is not None: plt.savefig(save,dpi=300, bbox_inches='tight')
        return fig
    else:
        lgds = []
        n_axes = len(samples_dict)-1 if 'theta_native' in samples_dict.keys() else len(samples_dict) # dont plot theta_native
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
        if save is not None: plt.savefig(save,dpi=300,bbox_extra_artists=lgds, bbox_inches='tight')
        return fig
         
def param_stats(samples_dict,theta_names=None,q1=0.05,q2=0.95,digits=4):
    """
    Compute statistics on the samples.

    :param dict samples_dict: samples from model.get_samples()
    :param list/NoneType theta_names: list of string names for thetas, optional (None will use default names)
    :param float q1: lower quantile in [0, 1]
    :param float q2: upper quantile in [0, 1]
    :param int digits: how many digits to show in output
    :return: pandas DataFrame containing statistics
    """
    # theta_names : list
    # samples_dict : dictionary of samples
    # stats : dataframe with mean and std of all parameters
    if 'theta' in samples_dict.keys():
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
    return stats

def rho_box_plots(model,labels=None):
    """
    Show rho box plots. (Rho are the transformed betaU parameters, corresponding to GP lengthscales)

    :param sepia.SepiaModel model: SepiaModel object
    :param list/NoneType labels: optional labels to use for box plot
    :return: matplotlib figure
    """
    samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}
    p = model.num.p
    q = model.num.q
    pu = model.num.pu
    bu = samples_dict['betaU']
    ru = np.exp(-bu / 4)
    fig,axs = plt.subplots(nrows=pu,tight_layout=True,figsize=[5,3*pu],squeeze=False)
    for i,ax in enumerate(axs.flatten()):
        r = ru[:, ((p+q)*i):((p+q)*i)+(p+q)]
        ax.boxplot(r)
        if labels is not None: ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(0,1.2,.2))
        ax.set_ylabel(r'$\rho$')
        ax.set_title('PC {}'.format(i+1))
    return fig
        
def plot_acf(model,nlags,nburn=0,alpha=.05,save=None):
    """
    Plot autocorrelation function for all parameters theta.
    
    :param sepia.SepiaModel model: SepiaModel object
    :param int nlags: how many lags to compute/plot
    :param int nburn: how many samples to burn
    :param float alpha: confidence level for acf significance line (0,1)
    :param str save: file name to save figure
    :return: matplotlib figure
    """

    if nlags>model.get_num_samples():
        raise ValueError('plot_acf: must have more samples than requested lag size')

    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in (0,1)')
    if model.num.sim_only:
        print('ACF needs thetas but this is a sim-only model.')
        return
    # get theta chains
    for p in model.params.mcmcList:
        if p.name == 'theta': 
            chain = p.mcmc_to_array(flat=True).T
    
    acf = model.acf(chain,nlags,plot=True,alpha=alpha)
    if save is not None: 
        acf['figure'].savefig(save,dpi=300,bbox_inches='tight')
    return acf

def plot_K_basis(data, max_plots=4, obs=True):
        """
        Plots K basis elements for both sim and obs indices (if applicable). Only applies to multivariate-output models.

        TODO: Lamy should be 1/Sigy_std

        :param int max_plots: maximum number of principal components to plot
        :return: tuple containing matplotlib figure objects: (fig_sim, fig_obs) or just fig_sim if no observed data is present
        """
        # Return early if scalar out or basis not set up
        if data.scalar_out:
            print('Scalar output, no K basis to plot.')
            return
        if data.sim_data.K is None:
            print('K basis not set up, call create_K_basis() first.')
            return
        # Plot sim basis
        pu = data.sim_data.K.shape[0]
        ncol = 5
        nrow = int(np.ceil((min(pu, max_plots) + 1) / ncol)) # add 1 for mean line
        fig_sim, axs_sim = plt.subplots(nrow, ncol, figsize=(12, 2 * nrow))
        fig_sim.tight_layout()
        for i, ax in enumerate(axs_sim.flatten()):
            if i == 0: # plot mean line
                ax.plot(data.sim_data.y_ind, np.mean(data.sim_data.K,axis=0))
                ax.set_title('sim mean')
                ax.set_ylabel('sim K basis')
                ax.set_xlabel('sim y_ind')
            elif i < pu+1:
                ax.plot(data.sim_data.y_ind, data.sim_data.K.T[:,i-1])
                ax.set_title('PC %d' % (i))
                ax.set_xlabel('sim y_ind')
            else:
                ax.axis('off')
        # If obs are present and requested, plot obs basis
        if not data.sim_only and obs:
            if data.ragged_obs:
                pu = np.array([k.shape[0] for k in data.obs_data.K])
                if np.all(pu == pu[0]): pu = pu[0]
                else: raise ValueError('first dimension in lists not equal')
            else:
                pu = data.obs_data.K.shape[0]
            ncol = 5
            nrow = int(np.ceil((min(pu,max_plots) + 1) / ncol)) # add 1 for mean line
            fig_obs, axs_obs = plt.subplots(nrow,ncol,figsize=(12, 2 * nrow))
            fig_obs.tight_layout()
            for i,ax in enumerate(axs_obs.flatten()):
                if i == 0: # plot mean line
                    if data.ragged_obs: ax.plot(data.obs_data.y_ind[i],np.mean(data.obs_data.K[i],axis=0))
                    else: ax.plot(data.obs_data.y_ind, np.mean(data.obs_data.K,axis=0))
                    ax.set_title('obs mean')
                    ax.set_ylabel('obs K basis')
                    ax.set_xlabel('obs y_ind')
                elif i < pu+1:
                    if data.ragged_obs: ax.plot(data.obs_data.y_ind[i],data.obs_data.K[i].T[:,i-1])
                    else: ax.plot(data.obs_data.y_ind, data.obs_data.K.T[:,i-1])
                    ax.set_title('PC %d' % (i))
                    ax.set_xlabel('obs y_ind')
                else:
                    ax.axis('off')
            return(fig_sim,fig_obs)
        else:
            return fig_sim

def plot_K_weights(data, max_u_plot=5):
        """
        Plots K basis weights for both sim and obs data (if applicable). Only applies to multivariate-output models.

        TODO: Lamy should be 1/Sigy_std

        :param int max_u_plot: max number of u's for which to plot vertical line over histogram of w's
        :return: tuple containing matplotlib figure objects: (fig_uw, fig_v) or just fig_uw if no discrepancy is specified
        """
        # Return early if scalar out or basis not set up
        if data.scalar_out:
            print('Scalar output, no K weights to plot.')
            return
        if data.sim_data.K is None:
            print('K basis not set up, call create_K_basis() first.')
            return
        # Compute sim K weights
        pu = data.sim_data.K.shape[0]
        ncol = 5
        nrow = int(np.ceil(pu / ncol))
        w = np.dot(np.linalg.pinv(data.sim_data.K).T, data.sim_data.y_std.T).T

        fig_uw, axs_uw = plt.subplots(nrow,ncol,figsize=(10,2*nrow))
        fig_uw.tight_layout()

        # Compute obs K weights if obs are present
        if not data.sim_only and data.obs_data.K is not None:
            # set pu
            if data.ragged_obs:
                pu = np.array([k.shape[0] for k in data.obs_data.K])
                if np.all(pu == pu[0]): pu = pu[0]
                else: raise ValueError('first dimension in lists not equal')
            else:
                pu = data.obs_data.K.shape[0]
                    
            # No D
            if data.obs_data.D is None:
                pv = 0
                DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                # compute u
                if data.ragged_obs:
                    u = []
                    for i in range(len(data.obs_data.y_ind)):
                        DK = data.obs_data.K[i]
                        Lamy = np.eye(data.obs_data.y_ind[i].shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])
                        u.append(np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std[i].T])).T)
                    u = np.array(u)
                else:
                    DK = data.obs_data.K
                    Lamy = np.eye(data.obs_data.y_ind.shape[0]) # Identity with size len(y_ind) how to do this with ragged?
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    u = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std.T])).T
                            
                nrow = int(np.ceil(pu / ncol))
                if u.shape[1] == w.shape[1]:
                    for i,ax in enumerate(axs_uw.flatten()):
                        if i < w.shape[1]:
                            limit = abs(max(max(w[:,i].min(), w[:,i].max(), key=abs),\
                                            max(u[:,i].min(), u[:,i].max(), key=abs), key=abs))
                            ax.set_xlim([-1.25*limit,1.25*limit])
                            bins_uw = np.linspace(-limit,limit,15,endpoint=True)
                            ax.set_xlabel('PC %d wt' % (i+1))
                            ax.set_xlim([-limit,limit])
                            ax.hist(w[:,i],bins=bins_uw,label='w',density=True)
                            for j in range(min(u.shape[0],max_u_plot)):
                                ax.axvline(u[j,i],color='darkorange',label='u' if j==0 else '_')
                            ax.legend(prop={'size': 6})
                        else:
                            ax.axis('off')
                    return fig_uw
                            
                else: # do u and w independently
                    raise ValueError('u.shape[1] != w.shape[1]')
                                
            else: # D
                if data.ragged_obs:
                    pv = np.array([d.shape[0] for d in data.obs_data.D])
                    if np.all(pv == pv[0]): pv = pv[0]
                    else: raise ValueError('first dimension in lists not equal')
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    u = []
                    v = []
                    for i in range(len(data.obs_data.D)):
                        DK = np.concatenate([data.obs_data.D[i], data.obs_data.K[i]])
                        Lamy = np.eye(data.obs_data.y_ind[i].shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std[i].T]))
                        v.append(vu[:pv].T)
                        u.append(vu[pv:].T)
                    u = np.array(u)
                    v = np.array(v)
                else:
                    pv = data.obs_data.D.shape[0]
                    DK = np.concatenate([data.obs_data.D, data.obs_data.K])  # (pu+pv, ell_obs)
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    Lamy = np.eye(data.obs_data.y_ind.shape[0])
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std.T]))
                    v = vu[:pv, :].T
                    u = vu[pv:, :].T
                            
                if u.shape[1] == w.shape[1]:
                    for i,ax in enumerate(axs_uw.flatten()):
                        if i < w.shape[1]:
                            limit = abs(max(max(w[:,i].min(), w[:,i].max(), key=abs),\
                                                  max(u[:,i].min(), u[:,i].max(), key=abs), key=abs))
                            ax.set_xlim([-1.1*limit,1.1*limit])
                            bins_uw = np.linspace(-limit,limit,15,endpoint=True)
                            ax.set_xlabel('PC %d wt' % (i+1))
                            ax.hist(w[:,i],bins=bins_uw,label='w',density=True)
                            for j in range(min(u.shape[0],max_u_plot)):
                                ax.axvline(u[j,i],color='darkorange',label='u' if j==0 else '_')
                            ax.legend(prop={'size': 6})
                        else:
                            ax.axis('off')
                else: 
                    raise ValueError('u.shape[1] != w.shape[1]')

                # V
                nrow = int(np.ceil(pv / ncol))
                fig_v, axs_v = plt.subplots(nrow,ncol,figsize=(10,2*nrow))
                fig_v.tight_layout()
                for i,ax in enumerate(axs_v.flatten()):
                    if i < v.shape[1]:
                        ax.hist(v[:,i],density=True)
                        ax.set_xlabel('D %d wt : v' % (i+1))
                    else:
                        ax.axis('off')
                return (fig_uw, fig_v)

def plot_u_w_pairs(data, max_plots=5, save=False):
        """
        Plots principal component basis weights for both sim and obs data (if applicable). Only applies to multivariate-output models.

        :param int max_plots: max number of principal components to plot
        :return: matplotlib figure fig_g: seaborn pairs figure
        """
        # Return early if scalar out or basis not set up
        if data.scalar_out:
            print('Scalar output, no K weights to plot.')
            return
        if data.sim_data.K is None:
            print('K basis not set up, call create_K_basis() first.')
            return
        pu = data.sim_data.K.shape[0]
        w = np.dot(np.linalg.pinv(data.sim_data.K).T, data.sim_data.y_std.T).T
                
        if not data.sim_only and data.obs_data.K is not None:
            if data.ragged_obs:
                pu = np.array([k.shape[0] for k in data.obs_data.K])
                if np.all(pu == pu[0]): pu = pu[0]
                else: raise ValueError('first dimension in lists not equal')
            else:
                pu = data.obs_data.K.shape[0]

            # No D
            if data.obs_data.D is None:
                pv = 0
                DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                if data.ragged_obs:
                    u = []
                    for i in range(len(data.obs_data.K)):
                        DK = data.obs_data.K[i]
                        Lamy = np.eye(data.obs_data.y_ind[i].shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        u.append(np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std[i].T])).T)
                    u = np.array(u)
                else:
                    DK = data.obs_data.K
                    Lamy = np.eye(data.obs_data.y_ind.shape[0])
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    u = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std.T])).T
                            
            else: # D
                if data.ragged_obs:
                    pv = np.array([d.shape[0] for d in data.obs_data.D])
                    if np.all(pv == pv[0]): pv = pv[0]
                    else: raise ValueError('first dimension in lists not equal')
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    u = []
                    v = []
                    for i in range(len(data.obs_data.D)):
                        DK = np.concatenate([data.obs_data.D[i], data.obs_data.K[i]])
                        Lamy = np.eye(data.obs_data.y_ind[i].shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std[i].T]))
                        v.append(vu[:pv].T)
                        u.append(vu[pv:].T)
                    u = np.array(u)
                    v = np.array(v)
                else:
                    pv = data.obs_data.D.shape[0]
                    DK = np.concatenate([data.obs_data.D, data.obs_data.K])  # (pu+pv, ell_obs)
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    Lamy = np.eye(data.obs_data.y_ind.shape[0])
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std.T]))
                    v = vu[:pv, :].T
                    u = vu[pv:, :].T

            # change u,w to match max_plots
            if w.shape[1]>max_plots: 
                w = w[:,0:max_plots]
                print('Plotting up to',max_plots,'pairs. Change with parameter \'max_plots\'')
            col_names = []
            for i in range(w.shape[1]): col_names.append('w{}'.format(i+1))
            w_df = pd.DataFrame(data=w,columns=col_names)
            if u.shape[1]>max_plots: u = u[:,0:max_plots]

            lims = max(np.maximum(np.max(np.abs(w),axis=0),np.max(np.abs(u),axis=0))*1.1)
            with sns.plotting_context("notebook", font_scale=1):
                g = sns.PairGrid(w_df)
                g.map_diag(sns.histplot, kde=True)
                g.map_offdiag(sns.scatterplot)
                for i in range(g.axes.shape[1]): # rows
                    for j in range(g.axes.shape[0]): # columns
                        g.axes[i,j].set_xlim(-lims,lims); g.axes[i,j].set_ylim(-lims,lims)
                        if i == j:
                            for k in range(u.shape[0]):
                                g.axes[i,i].axvline(u[k,i],color='darkorange',label='u{}'.format(i+1) if k==0 else "_")
                            g.axes[i,i].legend(facecolor='white')
                        else:
                            g.axes[i,j].scatter(u[:,j],u[:,i],c='darkorange',label='(u{},u{})'.format(j+1,i+1))
                            g.axes[i,j].legend(facecolor='white')
            if save: plt.savefig(save,dpi=300)
            return g.fig

def plot_K_residuals(data):
        """
        Plots residuals after projection to K basis. Only applies to multivariate-output models.
        :return: tuple containing matplotlib figure objects: (fig_u, fig_v) or just fig_noD if no discrepancy is specified
        """
        # Return early if scalar out or basis not set up
        if data.scalar_out:
            print('Scalar output, no K residuals to plot.')
            return
        if data.sim_data.K is None:
            print('K basis not set up, call create_K_basis() first.')
            return
        if not data.sim_only and data.obs_data.K is not None:
            if isinstance(data.obs_data.K, list):
                print('plot_K_residuals cannot yet handle ragged observations')
                return
            pu = data.obs_data.K.shape[0]
            if data.obs_data.D is None:
                pv = 0
                DK = data.obs_data.K
                DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                Lamy = np.eye(data.obs_data.y_ind.shape[0])
                DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                u = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std.T])).T
                proj = np.dot(u, DK)
                resid = data.obs_data.y_std - proj
                
                fig_noD, axs_noD = plt.subplots(1,3,figsize=(4,6))
               
                axs_noD[0].plot(data.obs_data.y_ind, data.obs_data.y_std.squeeze().T)
                axs_noD[0].set_title('obs y_std')
                axs_noD[0].set_xlabel('obs y_ind')
                
                axs_noD[1].plot(data.obs_data.y_ind, proj.squeeze().T)
                axs_noD[1].set_title('obs projection reconstruction')
                axs_noD[1].set_xlabel('obs y_ind')
                
                axs_noD[2].plot(data.obs_data.y_ind, resid.squeeze().T, '-')
                axs_noD[2].set_title('obs projection residual')
                axs_noD[2].set_xlabel('obs y_ind')
                return fig_noD
            else:
                pv = data.obs_data.D.shape[0]
                DK = np.concatenate([data.obs_data.D, data.obs_data.K])  # (pu+pv, ell_obs)
                DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                Lamy = np.eye(data.obs_data.y_ind.shape[0])
                DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, data.obs_data.y_std.T]))
                v = vu[:pv, :].T
                u = vu[pv:, :].T
                ncol = 5
                nrow = int(np.ceil(pu / ncol))
                fig_u,axs_u = plt.subplots(nrow,ncol,figsize=(8, 2 * nrow))
                for i, ax in enumerate(axs_u.flatten()):
                    if i < pu:
                        ax.hist(u[:, i])
                        ax.set_xlabel('PC %d wt' % (i+1))
                    else:
                        ax.axis('off')
                
                nrow = int(np.ceil(pv / ncol))
                fig_v,axs_v = plt.subplots(nrow,ncol,figsize=(8, 2 * nrow))
                for i,ax in enumerate(axs_v.flatten()):
                    if i < pv:
                        ax.hist(v[:, i])
                        ax.set_xlabel('D %d wt' % (i+1))
                    else:
                        ax.axis('off')
                return (fig_u,fig_v)

def plot_data(data,which_x=None,x_min=None,x_max=None,y_min=None,y_max=None,n_neighbors=3,max_sims=50,save=None):
        """
        Plots observed data and simulation runs on the same axis with n_neighbors nearest simulations
        in x-space. Only applies to multivariate-output models with both simulation and observed data.
        
        :param list/NoneType which_x: optionally sets which x_obs indices to plot
        :param float x_min: sets x lower limit on plot
        :param float x_max: sets x upper limit on plot
        :param float y_min: sets y lower limit on plot
        :param float y_max: sets y upper limit on plot
        :param int n_neighbors: sets number of nearest simulations to highlight
        :param int max_sims: sets maximum number of simulation runs to plot
        :return matplotlib figure fig: figure object of plot
        """
        if data.sim_only:
            print('plot_data does not currently work for sim_only models.')
            return
        if data.scalar_out:
            print('plot_data does not currently work for univariate output models.')
            return

        n = data.obs_data.x.shape[0]
        m = data.sim_data.x.shape[0]

        # plot up to 4 input space points
        if n > 4:
            # if no which_x or given which_x is out of bounds
            if which_x is None or (which_x is not None and not np.all(which_x)<n and not np.all(which_x>-1)):
                # choose 4 equally space input points to plot
                which_x = np.linspace(0,n-1,4,dtype=int)
            x_plot = data.obs_data.x[which_x,:]
        else:
            which_x = np.arange(0,n,1,dtype=int)
            x_plot = data.obs_data.x
        n_plots = x_plot.shape[0]

        # get axis limits
        if data.ragged_obs:
            if x_min is None: x_min = min(min([np.amin(k) for k in data.obs_data.y_ind]),np.amin(data.sim_data.y_ind))
            if x_max is None: x_max = max(max([np.amax(k) for k in data.obs_data.y_ind]),np.amax(data.sim_data.y_ind))
            if y_min is None: y_min = min(min([np.amin(k) for k in data.obs_data.y]),np.amin(data.sim_data.y))
            if y_max is None: y_max = max(max([np.amax(k) for k in data.obs_data.y]),np.amax(data.sim_data.y))
        else:
            if x_min is None: x_min = min(np.amin(data.obs_data.y_ind),np.amin(data.sim_data.y_ind))
            if x_max is None: x_max = max(np.amax(data.obs_data.y_ind),np.amax(data.sim_data.y_ind))    
            if y_min is None: y_min = min(np.amin(data.obs_data.y),np.amin(data.sim_data.y))
            if y_max is None: y_max = max(np.amax(data.obs_data.y),np.amax(data.sim_data.y))

        # nearest neighbots
        # find closest sim input points to each x_plot observed input points
        # ith column of near_sim_idx contains the n_neighbors nearest sim_design points (by index)
        # for ith point in x_plot
        near_sim_idx = None
        # this checks that x is not set up as a dummy, if it is, nearest neighbors in x space doesn't mean anything
        if m>2 and not np.all(data.sim_data.x.flatten() == data.sim_data.x.flatten()[0]) and \
               (data.obs_data.x.shape[0]==1 or not np.all(data.obs_data.x.flatten() == data.obs_data.x.flatten()[0])): 
            n_neighbors = min(min(n_neighbors,m),7)
            near_sim_idx = np.zeros(shape=(n_neighbors,n_plots),dtype=int)
            for i in range(n_plots):
                dist = np.argsort(np.linalg.norm(data.sim_data.x-x_plot[i,:],axis=1))
                near_sim_idx[:,i] = dist[0:n_neighbors]

        # Generate plot for each x_plot (x_obs) point
        fig = plt.figure(figsize=[12,12],constrained_layout=True)
        gs = GridSpec(2,2,figure=fig)
        axs = np.array([fig.add_subplot(gs[0,0]),\
                        fig.add_subplot(gs[0,1]),\
                        fig.add_subplot(gs[1,0]),\
                        fig.add_subplot(gs[1,1])])
        for i in range(4):
            if i < n_plots:
                # axis limits, ticks, and labels
                axs[i].set_xlim([x_min, x_max])
                axs[i].set_ylim([y_min, y_max])
                #axs[i].xaxis.set_ticks(np.linspace(x_min,x_max,10,endpoint=True))
                #axs[i].yaxis.set_ticks(np.linspace(y_min,y_max,10,endpoint=True))
                axs[i].set_title("x_obs point {}".format(i+1))
                axs[i].set_xlabel("y_ind (native)")
                axs[i].set_ylabel("y (native)")

                # simulations all
                if m>max_sims:
                    sims_idx = np.linspace(0,m-1,max_sims,dtype=int)
                else:
                    sims_idx = range(m)
                for j in sims_idx:
                    axs[i].plot(data.sim_data.y_ind, np.transpose(data.sim_data.y)[:,j],color='lightgrey',\
                        linestyle="--",label="Simulation runs" if j==0 else "_")

                # simulations - nearest neighbors
                if near_sim_idx is not None:
                    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k') # for nearest sims
                    for j in range(n_neighbors):
                        axs[i].plot(data.sim_data.y_ind,np.transpose(data.sim_data.y)[:,near_sim_idx[j,i]],\
                                linestyle="--",\
                                color=colors[j],label="Nearest Sim {}".format(j+1))

                # true data curve and "real data points"
                if data.ragged_obs:
                    axs[i].plot(data.obs_data.y_ind[i], data.obs_data.y[which_x[i]],'--ko',label="Obs data")
                else:
                    axs[i].plot(data.obs_data.y_ind, data.obs_data.y[which_x[i]],'--ko',label="Obs data")

                # legend
                axs[i].legend()
                #axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            else:
                axs[i].axis('off')
                
        if save is not None: fig.savefig(save,dpi=300,bbox_inches='tight')
        return fig




