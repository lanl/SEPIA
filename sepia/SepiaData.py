#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
sns.set()

from sepia.DataContainer import DataContainer


class SepiaData(object):
    """
    Data object used for SepiaModel, containing potentially both sim_data and obs_data.

    Many arguments are optional, but users should instantiate object with all data needed for the desired model.

    :param x_sim: nparray -- (n, p) matrix of controllable inputs/experimental conditions (optional)
    :param t_sim: nparray -- (n, q) matrix of non-controllable inputs, can be None (at least one of x_sim and t_sim must be provided)
    :param y_sim: nparray -- (n, ell_sim) matrix of outputs (REQUIRED)
    :param y_ind_sim: nparray -- (ell_sim, ) vector of indices for multivariate y, required if ell_sim > 1
    :param x_obs: nparray -- (m, p) matrix of controllable inputs for observation data (optional)
    :param y_obs:  nparray -- (m, ell_obs) matrix of obs outputs, or list length m of 1D arrays (for ragged y_ind_obs)
    :param y_ind_obs: (l_obs, ) vector of indices for multivariate y or list length m of 1D arrays (for ragged y_ind_obs)
    :raises: TypeError if shapes not conformal or required data missing.

    """

    # Creates DataContainer objects internally to store data.
    # Attributes passed to constructor:
    #     x_sim      simulation GP inputs (the ones known for obs as well), (m, p)
    #     t_sim      simulation GP inputs (the ones not known for obs), (m, q)
    #     y_sim      simulation GP outputs (m, ell_sim)
    #     y_ind_sim  y indices for simulation data (needed if ell_sim > 1)
    #     x_obs      (optional) observation GP inputs, (n, p)
    #     y_obs      (optional) observation GP outputs, (n, ell_obs) or list of ragged
    #     y_ind_obs  (optional) y indices for observation data (needed if ell_obs > 1) or list of ragged
    # Attributes set internally based on input data:
    #     sim_only    boolean, whether it's simulation data only or both simulation and observed
    #     scalar_out  boolean, whether GP has scalar output
    def __init__(self, x_sim=None, t_sim=None, y_sim=None, y_ind_sim=None, x_obs=None, y_obs=None, y_ind_obs=None):
        if y_sim is None:
            raise TypeError('y_sim is required to set up model.')
        if x_sim is None and t_sim is None:
            raise TypeError('At least one of x_sim or t_sim is required to set up model.')
        if x_sim is None:
            x_sim = 0.5 * np.ones((t_sim.shape[0], 1)) # sets up dummy x
        self.sim_data = DataContainer(x=x_sim, y=y_sim, t=t_sim, y_ind=y_ind_sim)
        self.ragged_obs = False
        if y_obs is None:
            self.obs_data = None
            self.sim_only = True
        else:
            if x_obs is None:
                x_obs = 0.5 * np.ones((len(y_obs), 1)) # sets up dummy x
            if x_sim.shape[1] != x_obs.shape[1]:
                raise TypeError('x_sim and x_obs do not contain the same number of variables/columns.')
            self.obs_data = DataContainer(x=x_obs, y=y_obs, y_ind=y_ind_obs)
            self.sim_only = False
            if isinstance(y_obs, list):
                self.ragged_obs = True
        if y_ind_sim is not None and y_sim.shape[1] > 1:
            self.scalar_out = False
        else:
            self.scalar_out = True

    # Prints pretty representation of the SepiaData object for users to check their setup.
    def __str__(self):
        res = ''
        res += 'This SepiaData instance implies the following:\n'
        if self.sim_only:
            res += 'This is a simulator (eta)-only model, y dimension %d\n' % self.sim_data.y.shape[1]
            res += 'm  = %5d (number of simulated data)\n' % self.sim_data.x.shape[0]
            res += 'p  = %5d (number of inputs)\n' % self.sim_data.x.shape[1]
            if self.sim_data.t is not None:
                res += 'q  = %5d (number of additional simulation inputs)\n' % self.sim_data.t.shape[1]
            if self.scalar_out:
                res += 'pu =     1 (univariate response dimension)\n'
            elif self.sim_data.K is not None:
                res += 'pu = %5d (transformed response dimension)\n' % self.sim_data.K.shape[0]
            else:
                res += 'pu NOT SET (transformed response dimension); call method create_K_basis \n'
        else:
            if self.ragged_obs:
                res += 'This is a simulator and obs model, sim y dimension %d, obs y dimension ragged\n' % self.sim_data.y.shape[1]
            else:
                res += 'This is a simulator and obs model, sim y dimension %d, obs y dimension %d\n' % (self.sim_data.y.shape[1], self.obs_data.y.shape[1])
            res += 'n  = %5d (number of observed data)\n' % self.obs_data.x.shape[0]
            res += 'm  = %5d (number of simulated data)\n' % self.sim_data.x.shape[0]
            res += 'p  = %5d (number of inputs)\n' % self.sim_data.x.shape[1]
            res += 'q  = %5d (number of additional simulation inputs to calibrate)\n' % self.sim_data.t.shape[1]
            if self.scalar_out:
                res += 'pu =     1 (univariate response dimension)'
            else:
                if self.sim_data.K is not None and self.obs_data.K is not None:
                    res += 'pu = %5d (transformed response dimension)\n' % self.sim_data.K.shape[0]
                else:
                    res += 'pu NOT SET (transformed response dimension); call method create_K_basis\n'
                if self.obs_data.D is not None:
                    if self.ragged_obs:
                        res += 'pv = %5d (transformed discrepancy dimension)\n' % self.obs_data.D[0].shape[0]
                    else:
                        res += 'pv = %5d (transformed discrepancy dimension)\n' % self.obs_data.D.shape[0]
                else:
                    res += 'pv NOT SET (transformed discrepancy dimension); call method create_D_basis\n'
        return res

    def transform_xt(self, xt_min=0.0, xt_max=1.0):
        """
        Transforms sim_data x and t and obs_data x to lie in [xt_min, xt_max], columnwise.

        If min/max of inputs in a column are equal, it does nothing to that column.
        Stores original min/max as orig_t_min/max, orig_x_min/max for untransforming.

        :param xt_min: scalar -- minimum x or t value
        :param xt_max: scalar -- maximum x or t value
        """
        def trans(x, a, b, x_min, x_max):
            a_vec = a * np.ones_like(x_min)
            b_vec = b * np.ones_like(x_min)
            xmm = x_max - x_min
            # If min/max are equal, don't want to transform
            x_min = x_min.copy()
            x_min[xmm == 0] = 0
            a_vec[xmm == 0] = 0
            b_vec[xmm == 0] = 1
            xmm[xmm == 0] = 1
            return (x - x_min) / xmm * (b_vec - a_vec) + a_vec
        self.sim_data.orig_x_min = np.min(self.sim_data.x, 0, keepdims=True)
        self.sim_data.orig_x_max = np.max(self.sim_data.x, 0, keepdims=True)
        self.sim_data.x_trans = trans(self.sim_data.x, xt_min, xt_max, self.sim_data.orig_x_min, self.sim_data.orig_x_max)
        if self.sim_data.t is not None:
            self.sim_data.orig_t_min = np.min(self.sim_data.t, 0, keepdims=True)
            self.sim_data.orig_t_max = np.max(self.sim_data.t, 0, keepdims=True)
            self.sim_data.t_trans = trans(self.sim_data.t, xt_min, xt_max, self.sim_data.orig_t_min, self.sim_data.orig_t_max)
        if not self.sim_only:
            self.obs_data.orig_x_min = self.sim_data.orig_x_min
            self.obs_data.orig_x_max = self.sim_data.orig_x_max
            self.obs_data.orig_t_min = self.sim_data.orig_t_min
            self.obs_data.orig_t_max = self.sim_data.orig_t_max
            self.obs_data.x_trans = trans(self.obs_data.x, xt_min, xt_max, self.obs_data.orig_x_min, self.obs_data.orig_x_max)

    def standardize_y(self, center=True, scale='scalar'):
        """
        Standardizes both sim_data and obs_data GP outputs (y) based on sim_data.y mean/SD.

        Stores orig_y_mean, orig_y_sd for untransforming.

        :param center: boolean -- whether to subtract simulation mean (across observations)
        :param scale: string/boolean -- how to rescale: 'scalar': single SD over all demeaned data,
                                                        'columnwise': SD for each column of demeaned data,
                                                        False: no rescaling
        """
        if center:
            self.sim_data.orig_y_mean = np.mean(self.sim_data.y, 0)
        else:
            self.sim_data.orig_y_mean = 0.
        y_dm = self.sim_data.y - self.sim_data.orig_y_mean
        if scale == 'scalar':
            self.sim_data.orig_y_sd = np.std(y_dm, ddof=1)
        elif scale == 'columnwise':
            self.sim_data.orig_y_sd = np.std(y_dm, ddof=1, axis=0)
        else:
            self.sim_data.orig_y_sd = 1.
        self.sim_data.y_std = y_dm/self.sim_data.orig_y_sd
        if not self.sim_only:
            if not self.scalar_out and not np.isscalar(self.sim_data.orig_y_mean):
                if self.ragged_obs:
                    orig_y_mean = []
                    for i in range(len(self.obs_data.y)):
                        orig_y_mean.append(np.interp(self.obs_data.y_ind[i], self.sim_data.y_ind.squeeze(), self.sim_data.orig_y_mean))
                else:
                    orig_y_mean = np.interp(self.obs_data.y_ind.squeeze(), self.sim_data.y_ind.squeeze(), self.sim_data.orig_y_mean)
                self.obs_data.orig_y_mean = orig_y_mean
            else:
                if self.ragged_obs:
                    self.obs_data.orig_y_mean = [self.sim_data.orig_y_mean for i in range(len(self.obs_data.y))]
                else:
                    self.obs_data.orig_y_mean = self.sim_data.orig_y_mean
            if not self.scalar_out and not np.isscalar(self.sim_data.orig_y_sd):
                if self.ragged_obs:
                    orig_y_sd = []
                    for i in range(len(self.obs_data.y)):
                        orig_y_sd.append(np.interp(self.obs_data.y_ind[i], self.sim_data.y_ind.squeeze(), self.sim_data.orig_y_sd))
                else:
                    orig_y_sd = np.interp(self.obs_data.y_ind, self.sim_data.y_ind, self.sim_data.orig_y_sd)
                self.obs_data.orig_y_sd = orig_y_sd
            else:
                if self.ragged_obs:
                    self.obs_data.orig_y_sd = [self.sim_data.orig_y_sd for i in range(len(self.obs_data.y))]
                else:
                    self.obs_data.orig_y_sd = self.sim_data.orig_y_sd
            if self.ragged_obs:
                self.obs_data.y_std = [(self.obs_data.y[i] - self.obs_data.orig_y_mean[i]) / self.obs_data.orig_y_sd[i] for i in range(len(self.obs_data.y))]
            else:
                self.obs_data.y_std = (self.obs_data.y - self.obs_data.orig_y_mean) / self.obs_data.orig_y_sd

    def create_K_basis(self, n_pc=0.995, K=None):
        """
        Creates K_sim and K_obs using PCA on sim_data.y_std; should be called after standardize_y.

        :param n_pc: float, int -- proportion in [0, 1] of variance, or an integer number of components
        :param K: nparray -- optional, a basis matrix on sim indices of shape (n_basis_elements, ell_sim)
        """
        if self.scalar_out:
            if n_pc == 1:
                print('Scalar output, using pu = 1 basis.')
                self.sim_data.K = np.zeros((n_pc, 1))
                self.scalar_out = False
                return
            else:
                print('Scalar output, no basis used.')
                return
        if K is not None:
            self.sim_data.K = K
        else:
            self.compute_sim_PCA_basis(n_pc)
        # interpolate PC basis to observed, if present
        if not self.sim_only:
            pu = self.sim_data.K.shape[0]
            if self.ragged_obs:
                K_obs = []
                for ki in range(len(self.obs_data.y)):
                    K_obs_tmp = np.zeros((pu, self.obs_data.y_ind[ki].shape[0]))
                    for i in range(pu):
                        K_obs_tmp[i, :] = np.interp(self.obs_data.y_ind[ki], self.sim_data.y_ind, self.sim_data.K[i, :])
                    K_obs.append(K_obs_tmp)
            else:
                K_obs = np.zeros((pu, self.obs_data.y_ind.shape[0]))
                for i in range(pu):
                    K_obs[i, :] = np.interp(self.obs_data.y_ind, self.sim_data.y_ind, self.sim_data.K[i, :])
            self.obs_data.K = K_obs

    def compute_sim_PCA_basis(self, n_pc):
        # Does PCA basis computation on sim_data.y_std attribute, sets K attribute to calculated basis.
        # Used internally by create_K_basis.
        # :param n_pc: int -- number of components or a proportion of variance explained, in [0, 1].
        y_std = self.sim_data.y_std
        if y_std is None:
            print('WARNING: y not standardized, doing default standardization before PCA...')
            self.standardize_y()
        U, s, V = np.linalg.svd(y_std.T, full_matrices=False)
        s2 = np.square(s)
        if n_pc < 1:
            cum_var = s2 / np.sum(s2)
            pu = np.sum(np.cumsum(cum_var) < n_pc) + 1
        else:
            pu = int(n_pc)
        self.sim_data.K = np.transpose(np.dot(U[:, :pu], np.diag(s[:pu])) / np.sqrt(y_std.shape[0]))


    def create_D_basis(self, type='constant', D_obs=None, D_sim=None, norm=True):
        """
        Create D_obs, D_sim discrepancy bases. Can specify a type of default basis (constant/linear) or provide matrices.

        :param type: string -- 'constant' or 'linear' -- optionally sets up default constant or linear D
        :param D_obs: nparray -- a basis matrix on obs indices of shape (n_basis_elements, ell_obs), or list of matrices
                                 for ragged obs; if D is given, type parameter is ignored.
        :param D_sim: nparray -- a basis matrix on sim indices of shape (n_basis_elements, sim_obs); optional, not
                                 needed to fit model, but if missing certain types of predictions are not available.
        :param norm: boolean -- whether to normalize D matrices
        """
        if self.sim_only:
            print('Sim only, skipping discrepancy...')
            return
        if not self.sim_only:
            if D_sim is not None:
                if not D_sim.shape[1] == self.sim_data.y.shape[1]:
                    raise TypeError('D_sim basis shape incorrect; second dim should match ell_sim')
                self.sim_data.D = D_sim

            if D_obs is not None:
                if self.ragged_obs:
                    for i in range(len(D_obs)):
                        if not D_obs[i].shape[1] == self.obs_data.y[i].shape[1]:
                            raise TypeError('D basis shape incorrect; second dim should match ell_obs')
                else:
                    if not D_obs.shape[1] == self.obs_data.y.shape[1]:
                        raise TypeError('D_obs basis shape incorrect; second dim should match ell_obs')
                self.obs_data.D = D_obs
            elif type == 'constant':
                if self.ragged_obs:
                    self.obs_data.D = [np.ones((1, self.obs_data.y[i].shape[0])) for i in range(len(self.obs_data.y))]
                else:
                    self.obs_data.D = np.ones((1, self.obs_data.y.shape[1]))
                self.sim_data.D = np.ones((1, self.sim_data.y.shape[1]))
            elif type == 'linear' and not self.scalar_out:
                self.sim_data.D = np.vstack([np.ones(self.sim_data.y.shape[1]), self.sim_data.y_ind])
                if self.ragged_obs:
                    self.obs_data.D = [np.vstack([np.ones(self.obs_data.y[i].shape[0]), self.obs_data.y_ind[i]]) for i in range(len(self.obs_data.y))]
                else:
                    self.obs_data.D = np.vstack([np.ones(self.obs_data.y.shape[1]), self.obs_data.y_ind])
            # Normalize D to match priors
            if norm:
                if D_sim is not None:
                    norm_scl = np.sqrt(np.max(np.dot(self.sim_data.D, self.sim_data.D.T)))
                    self.sim_data.D /= norm_scl
                    if self.ragged_obs:
                        for i in range(len(self.obs_data.D)):
                            self.obs_data.D[i] /= norm_scl
                    else:
                        self.obs_data.D /= norm_scl
                else:
                    if self.ragged_obs:
                        norm_scl = np.sqrt(np.max(np.dot(self.sim_data.D[0], self.sim_data.D[0].T)))
                        for i in range(len(self.obs_data.D)):
                            self.obs_data.D[i] /= norm_scl
                    else:
                        norm_scl = np.sqrt(np.max(np.dot(self.obs_data.D, self.obs_data.D.T)))
                        self.obs_data.D /= norm_scl

    def plot_K_basis(self, max_plots=4):
        """
        Plots K basis elements for both sim and obs indices (if applicable).

        :param max_plots: int -- maximum number of principal components to plot
        """
        if self.scalar_out:
            print('Scalar output, no K basis to plot.')
        else:
            if not self.sim_data.K is None:
                pu = self.sim_data.K.shape[0]
                ncol = 5
                nrow = int(np.ceil((min(pu,max_plots) + 1) / ncol)) # add 1 for mean line
                fig, axs = plt.subplots(nrow,ncol,figsize=(12, 2 * nrow))
                fig.tight_layout()
                for i,ax in enumerate(axs.flatten()):
                    if i == 0: # plot mean line
                        ax.plot(self.sim_data.y_ind, np.mean(self.sim_data.K,axis=0))
                        ax.set_title('sim mean')
                        ax.set_ylabel('sim K basis')
                        ax.set_xlabel('sim y_ind')
                    elif i < pu+1:
                        ax.plot(self.sim_data.y_ind, self.sim_data.K.T[:,i-1])
                        ax.set_title('PC %d' % (i))
                        ax.set_xlabel('sim y_ind')
                    else:
                        ax.axis('off')          
                plt.show()
                
            if not self.obs_data.K is None:    
                pu = self.obs_data.K.shape[0]
                ncol = 5
                nrow = int(np.ceil((min(pu,max_plots) + 1) / ncol)) # add 1 for mean line
                fig, axs = plt.subplots(nrow,ncol,figsize=(12, 2 * nrow))
                fig.tight_layout()
                for i,ax in enumerate(axs.flatten()):
                    if i == 0: # plot mean line
                        ax.plot(self.obs_data.y_ind, np.mean(self.obs_data.K,axis=0))
                        ax.set_title('obs mean')
                        ax.set_ylabel('obs K basis')
                        ax.set_xlabel('obs y_ind')
                    elif i < pu+1:
                        ax.plot(self.obs_data.y_ind, self.obs_data.K.T[:,i-1])
                        ax.set_title('PC %d' % (i))
                        ax.set_xlabel('obs y_ind')
                    else:
                        ax.axis('off')
                plt.show()
                
            #if not self.obs_data.K is None:
            #    K_obs_mean = np.mean(self.obs_data.K,axis=0)
            #    for i in range(self.obs_data.K.T.shape[1]):
            #        plt.plot(self.obs_data.y_ind, self.obs_data.K.T[:,i], '-o',label='PC {}'.format(i+1))
            #    plt.plot(self.obs_data.y_ind,K_obs_mean,'--o',label='obs mean')
            #    plt.xlabel('obs y_ind')
            #    plt.ylabel('obs K basis')
            #    plt.xticks(self.obs_data.y_ind)
            #    plt.legend()
            #    plt.show()

    def plot_K_weights(self, max_u_plot=5, plot_sep=False):
        """
        Plots K basis weights for both sim and obs data (if applicable).

        :param max_u_plot: int -- optional max number of u's for which to plot vertical line over histogram of w's
        :param plot_sep: bool -- histogram w's and u's seperately
        """
        if self.scalar_out:
            print('Scalar output, no K weights to plot.')
        else:
            if not self.sim_data.K is None:
                pu = self.sim_data.K.shape[0]
                ncol = 5
                nrow = int(np.ceil(pu / ncol))
                w = np.dot(np.linalg.pinv(self.sim_data.K).T, self.sim_data.y_std.T).T

                fig, axs = plt.subplots(nrow,ncol,figsize=(10,2*nrow))
                fig.tight_layout()

                if not self.obs_data.K is None:
                    pu = self.obs_data.K.shape[0]

                    # No D
                    if self.obs_data.D is None:
                        pv = 0
                        DK = self.obs_data.K
                        DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                        Lamy = np.eye(self.obs_data.y_ind.shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        u = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T])).T
                        nrow = int(np.ceil(pu / ncol))
                        if u.shape[1] == w.shape[1] and not plot_sep:
                            for i,ax in enumerate(axs.flatten()):
                                if i < w.shape[1]:
                                    limit = abs(max(max(w[:,i].min(), w[:,i].max(), key=abs),\
                                                  max(u[:,i].min(), u[:,i].max(), key=abs), key=abs))
                                    ax.set_xlim([-1.1*limit,1.1*limit])
                                    bins_uw = np.linspace(-limit,limit,15,endpoint=True)
                                    ax.set_xlabel('PC %d wt' % (i+1))
                                    ax.set_xlim([-limit,limit])
                                    ax.hist(w[:,i],bins=bins_uw,label='w',density=True)
                                    #ax.hist(u[:,i],bins=bins_uw,alpha=.75,color='darkorange',label='u',density=True)
                                    for j in range(min(u.shape[0],max_u_plot)): 
                                        ax.axvline(u[j,i],color='darkorange',label='u' if j==0 else '_')
                                    ax.legend(prop={'size': 6})
                                else:
                                    ax.axis('off')
                            plt.show()
                        else: # do u and w independently
                            # w
                            for i,ax in enumerate(axs.flatten()):
                                if i < w.shape[1]:
                                    w_abs_max = max(w[:,i].min(), w[:,i].max(), key=abs)
                                    ax.set_xlim([-1.1*w_abs_max,1.1*w_abs_max])
                                    ax.set_xlabel('PC %d wt : w' % (i+1))
                                    ax.hist(w[:,i],density=True)
                                else:
                                    ax.axis('off')
                            plt.show()
                            # u
                            pu = self.obs_data.K.shape[0]
                            nrow = int(np.ceil(pu / ncol))
                            fig, axs = plt.subplots(nrow,ncol,figsize=(10,2*nrow))
                            fig.tight_layout()
                            for i,ax in enumerate(axs.flatten()):
                                if i < u.shape[1]:
                                    ax.hist(u[:,i],density=True)
                                    ax.set_xlabel('PC %d wt : u' % (i+1))
                                else:
                                    ax.axis('off')
                            plt.show()
                    else: # D
                        pv = self.obs_data.D.shape[0]
                        DK = np.concatenate([self.obs_data.D, self.obs_data.K])  # (pu+pv, ell_obs)
                        DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                        Lamy = np.eye(self.obs_data.y_ind.shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T]))
                        v = vu[:pv, :].T
                        u = vu[pv:, :].T
                        if u.shape[1] == w.shape[1] and not plot_sep:
                            for i,ax in enumerate(axs.flatten()):
                                if i < w.shape[1]:
                                    limit = abs(max(max(w[:,i].min(), w[:,i].max(), key=abs),\
                                                  max(u[:,i].min(), u[:,i].max(), key=abs), key=abs))
                                    ax.set_xlim([-1.1*limit,1.1*limit])
                                    bins_uw = np.linspace(-limit,limit,15,endpoint=True)
                                    ax.set_xlabel('PC %d wt' % (i+1))
                                    ax.hist(w[:,i],bins=bins_uw,label='w',density=True)
                                    #ax.hist(u[:,i],bins=bins_uw,alpha=.75,color='darkorange',label='u',density=True)
                                    for j in range(min(u.shape[0],max_u_plot)): 
                                        ax.axvline(u[j,i],color='darkorange',label='u' if j==0 else '_')
                                    ax.legend(prop={'size': 6})
                                else:
                                    ax.axis('off')
                            plt.show()

                        else: # do u and w independently
                            # w
                            for i,ax in enumerate(axs.flatten()):
                                if i < w.shape[1]:
                                    w_abs_max = max(w[:,i].min(), w[:,i].max(), key=abs)
                                    ax.set_xlim([-1.1*w_abs_max,1.1*w_abs_max])
                                    ax.set_xlabel('PC %d wt : w' % (i+1))
                                    ax.hist(w[:,i],density=True)
                                else:
                                    ax.axis('off')
                            plt.show()
                            # u
                            pu = self.obs_data.K.shape[0]
                            nrow = int(np.ceil(pu / ncol))
                            fig, axs = plt.subplots(nrow,ncol,figsize=(10,2*nrow))
                            fig.tight_layout()
                            for i,ax in enumerate(axs.flatten()):
                                if i < u.shape[1]:
                                    ax.hist(u[:,i],density=True)
                                    ax.set_xlabel('PC %d wt : u' % (i+1))
                                else:
                                    ax.axis('off')
                            plt.show()

                        # V
                        nrow = int(np.ceil(pv / ncol))
                        fig, axs = plt.subplots(nrow,ncol,figsize=(10,2*nrow))
                        fig.tight_layout()
                        for i,ax in enumerate(axs.flatten()):
                            if i < v.shape[1]:
                                ax.hist(v[:,i],density=True)
                                ax.set_xlabel('D %d wt : v' % (i+1))
                            else:
                                ax.axis('off')
                        plt.show()
    
    def plot_u_w_pairs(self, max_plots=5):
        """
        Plots principal component basis weights for both sim and obs data (if applicable).

        :param max_plots: int -- optional max number of principle components to plot
        """
        if self.scalar_out:
            print('Scalar output, no K weights to plot.')
        else:
            print('Plotting up to',max_plots,'pairs. Change with parameter \'max_plots\'')
            if not self.sim_data.K is None:
                pu = self.sim_data.K.shape[0]
                ncol = 5
                nrow = int(np.ceil(pu / ncol))
                w = np.dot(np.linalg.pinv(self.sim_data.K).T, self.sim_data.y_std.T).T
                
                if not self.obs_data.K is None:
                    pu = self.obs_data.K.shape[0]

                    # No D
                    if self.obs_data.D is None:
                        pv = 0
                        DK = self.obs_data.K
                        DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                        Lamy = np.eye(self.obs_data.y_ind.shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        u = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T])).T
                    else: # D
                        pv = self.obs_data.D.shape[0]
                        DK = np.concatenate([self.obs_data.D, self.obs_data.K])  # (pu+pv, ell_obs)
                        DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                        Lamy = np.eye(self.obs_data.y_ind.shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T]))
                        v = vu[:pv, :].T
                        u = vu[pv:, :].T

                    # change u,w to match max_plots
                    if w.shape[1]>max_plots: w = w[:,0:max_plots]
                    col_names = []
                    for i in range(w.shape[1]): col_names.append('w{}'.format(i+1))
                    w_df = pd.DataFrame(data=w,columns=col_names)
                    if u.shape[1]>max_plots: u = u[:,0:max_plots]
                        
                    lims = max(np.maximum(np.max(np.abs(w),axis=0),np.max(np.abs(u),axis=0))*1.1)
                    with sns.plotting_context("notebook", font_scale=1):
                        g = sns.PairGrid(w_df)
                        g.map_diag(sns.distplot)
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
                    plt.show()

    def plot_K_residuals(self):
        """
        Plots residuals after projection to K basis.

        """
        if self.scalar_out:
            print('Scalar output, no K weights to plot.')
        else:
            if not self.obs_data.K is None:
                pu = self.obs_data.K.shape[0]
                if self.obs_data.D is None:
                    pv = 0
                    DK = self.obs_data.K
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    Lamy = np.eye(self.obs_data.y_ind.shape[0])
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    u = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T])).T
                    proj = np.dot(u, DK)
                    resid = self.obs_data.y_std - proj
                    plt.figure(1, figsize=(4, 6))
                    plt.subplot(311)
                    plt.plot(self.obs_data.y_ind, self.obs_data.y_std.squeeze())
                    plt.title('obs y_std')
                    plt.xlabel('obs y_ind')
                    plt.subplot(312)
                    plt.plot(self.obs_data.y_ind, proj.squeeze())
                    plt.title('obs projection reconstruction')
                    plt.xlabel('obs y_ind')
                    plt.subplot(313)
                    sns.lineplot(x=self.obs_data.y_ind, y=resid.squeeze())
                    plt.title('obs projection residual')
                    plt.xlabel('obs y_ind')
                    plt.show()
                else:
                    pv = self.obs_data.D.shape[0]
                    DK = np.concatenate([self.obs_data.D, self.obs_data.K])  # (pu+pv, ell_obs)
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    Lamy = np.eye(self.obs_data.y_ind.shape[0])
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T]))
                    v = vu[:pv, :].T
                    u = vu[pv:, :].T
                    ncol = 5
                    nrow = np.ceil(pu / ncol)
                    plt.figure(2, figsize=(8, 2 * nrow))
                    for i in range(pu):
                        plt.subplot(nrow, ncol, i+1)
                        plt.hist(u[:, i])
                        plt.xlabel('PC %d wt' % (i+1))
                    plt.show()
                    ncol = 5
                    nrow = np.ceil(pv / ncol)
                    plt.figure(3, figsize=(8, 2 * nrow))
                    for i in range(pu):
                        plt.subplot(nrow, ncol, i+1)
                        plt.hist(v[:, i])
                        plt.xlabel('D %d wt' % (i+1))
                    plt.show()

    def plot_data(self,which_x = [],x_min=None,x_max=None,y_min=None,y_max=None,n_neighbors=3,max_sims=50):
        """
        Plots observed data and simulation runs on the same axis with n_neighbors nearest simulations
        in x-space colored
        
        :param which_x: list -- optionally sets which x_obs indices to plot
        :param x_min: float -- optionally sets x lower limit on plot
        :param x_max: float -- optionally sets x upper limit on plot
        :param y_min: float -- optionally sets y lower limit on plot
        :param y_max: float -- optionally sets y upper limit on plot
        :param n_neighbors: int -- optionally sets number of nearest simulations to highlight
        :param max_sims: int -- optionally sets maximum number of simulation runs to plot
        """
        n = self.obs_data.x.shape[0]
        m = self.sim_data.x.shape[0]

        # plot up to 4 input space points
        if n > 4:
            # if no which_x or given which_x is out of bounds
            if not which_x or (which_x and not np.all(which_x)<n and not np.all(which_x>-1)):
                # choose 4 equally space input points to plot
                which_x = np.linspace(0,n-1,4,dtype=int)
            x_plot = self.obs_data.x[which_x,:]
        else:
            which_x = np.arange(0,n,1,dtype=int)
            x_plot = self.obs_data.x
        n_plots = x_plot.shape[0]

        # get axis limits
        if x_min is None: x_min = min(np.amin(self.obs_data.y_ind),np.amin(self.sim_data.y_ind))
        if x_max is None: x_max = max(np.amax(self.obs_data.y_ind),np.amax(self.sim_data.y_ind))    
        if y_min is None: y_min = min(np.amin(self.obs_data.y),np.amin(self.sim_data.y))
        if y_max is None: y_max = max(np.amax(self.obs_data.y),np.amax(self.sim_data.y))

        # nearest neighbots
        # find closest sim input points to each x_plot observed input points
        # ith column of near_sim_idx contains the n_neighbors nearest sim_design points (by index)
        # for ith point in x_plot
        near_sim_idx = None
        # this checks that x is not set up as a dummy, if it is, nearest neighbors in x space doesn't mean anything
        if m>2 and not np.all(self.sim_data.x.flatten() == self.sim_data.x.flatten()[0]) and \
               (self.obs_data.x.shape[0]==1 or not np.all(self.obs_data.x.flatten() == self.obs_data.x.flatten()[0])): 
            n_neighbors = min(min(n_neighbors,m),7)
            near_sim_idx = np.zeros(shape=(n_neighbors,n_plots),dtype=int)
            for i in range(n_plots):
                dist = np.argsort(np.linalg.norm(self.sim_data.x-x_plot[i,:],axis=1))
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
                    axs[i].plot(self.sim_data.y_ind, np.transpose(self.sim_data.y)[:,j],color='lightgrey',\
                        linestyle="--",label="Simulation runs" if j==0 else "_")

                # simulations - nearest neighbors
                if near_sim_idx is not None:
                    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k') # for nearest sims
                    for j in range(n_neighbors):
                        axs[i].plot(self.sim_data.y_ind,np.transpose(self.sim_data.y)[:,near_sim_idx[j,i]],\
                                linestyle="--",\
                                color=colors[j],label="Nearest Sim {}".format(j+1))

                # true data curve and "real data points"
                axs[i].plot(self.obs_data.y_ind, self.obs_data.y[which_x[i],:],'--ko',label="Obs data")

                # legend
                axs[i].legend()
                #axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            else:
                axs[i].axis('off')
                
        plt.show()  