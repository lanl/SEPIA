#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd

from sepia.DataContainer import DataContainer

sns.set()


class SepiaData(object):
    """
    Data object used for SepiaModel, containing potentially both `sim_data` and `obs_data` objects of type `sepia.DataContainer`.

    :var numpy.ndarray/NoneType x_sim: controllable inputs/experimental conditions, shape (n, p) or None
    :var numpy.ndarray/NoneType t_sim: non-controllable inputs, shape (n, q) or None
    :var numpy.ndarray y_sim: simulation outputs, shape (n, ell_sim)
    :var numpy.ndarray/NoneType y_ind_sim: indices for multivariate y, shape (ell_sim, ), required if ell_sim > 1
    :var numpy.ndarray/NoneType x_obs: controllable inputs for observation data, shape (m, p) or None
    :var numpy.ndarray/list/NoneType y_obs: observed outputs, shape (m, ell_obs), or list length m of 1D arrays (for ragged y_ind_obs), or None
    :var numpy.ndarray/list/NoneType y_ind_obs: vector of indices for multivariate y, shape (l_obs, ), or list length m of 1D arrays (for ragged y_ind_obs), or None
    :var bool sim_only: is it simulation-only data?
    :var bool scalar_out: is the output y scalar?
    :var bool ragged_obs: do the observations have ragged (non-shared) multivariate indices across instances?
    :var numpy.ndarray/list x_cat_ind: indices of x that are categorical (0 = not cat, int > 0 = how many categories)
    :var numpy.ndarray/list t_cat_ind: indices of t that are categorical (0 = not cat, int > 0 = how many categories)
    """

    def __init__(self, x_sim=None, t_sim=None, y_sim=None, y_ind_sim=None, x_obs=None, y_obs=None, y_ind_obs=None,
                 x_cat_ind=None, t_cat_ind=None):
        """
        Create SepiaData object. Many arguments are optional depending on the type of model.
        Users should instantiate with all data needed for the desired model. See documentation pages for more detail.

        :param numpy.ndarray/NoneType x_sim: controllable inputs/experimental conditions, shape (n, p), or None
        :param numpy.ndarray/NoneType t_sim: non-controllable inputs, shape (n, q), or None
        :param numpy.ndarray y_sim: simulation outputs, shape (n, ell_sim)
        :param numpy.ndarray/NoneType y_ind_sim: indices for multivariate y, shape (ell_sim, ), required if ell_sim > 1
        :param numpy.ndarray/NoneType x_obs: controllable inputs for observation data, shape (m, p) or None
        :param numpy.ndarray/list/NoneType y_obs: observed outputs, shape (m, ell_obs), or list length m of 1D arrays (for ragged y_ind_obs), or None
        :param numpy.ndarray/list/NoneType y_ind_obs: vector of indices for multivariate y, shape (l_obs, ), or list length m of 1D arrays (for ragged y_ind_obs), or None
        :param numpy.ndarray/list/NoneType x_cat_ind: indices of x that are categorical (0 = not cat, int > 0 = how many categories), or None
        :param numpy.ndarray/list/NoneType t_cat_ind: indices of t that are categorical (0 = not cat, int > 0 = how many categories), or None
        :raises: TypeError if shapes not conformal or required data missing.

        .. note: At least one of x_sim and t_sim must be provided, and y_sim must always be provided.

        """

        if y_sim is None:
            raise TypeError('y_sim is required to set up model.')
        if x_sim is None and t_sim is None:
            raise TypeError('At least one of x_sim or t_sim is required to set up model.')
        if x_sim is None:
            x_sim = 0.5 * np.ones((t_sim.shape[0], 1)) # sets up dummy x
        self.kron_design=isinstance(x_sim,list)
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
        # Process categorical indices
        if not self.kron_design:
            if x_cat_ind is not None:
                if len(x_cat_ind) != x_sim.shape[1]:
                    raise TypeError('x_cat_ind length should equal p.')
                for i, ci in enumerate(x_cat_ind):
                    if ci > 0 and ci != np.max(x_sim[:, i]):
                        raise TypeError('Nonzero values of x_cat_ind should equal number of categories.')
            else:
                x_cat_ind = np.zeros(x_sim.shape[1])
        else:
            # TODO only consider the case of non-categorical inputs for the kron design for now ...
            x_cat_ind = [np.zeros(x_sim[ii].shape[1]) for ii in range(len(x_sim))]
        self.x_cat_ind = x_cat_ind
        if t_cat_ind is not None:
            if t_sim is None:
                raise TypeError('Cannot use t_cat_ind if t_sim is not provided.')
            if len(t_cat_ind) != t_sim.shape[1]:
                raise TypeError('t_cat_ind length should equal p.')
            for i, ci in enumerate(t_cat_ind):
                if ci > 0 and ci != np.max(t_sim[:, i]):
                    raise TypeError('Nonzero values of t_cat_ind should equal number of categories.')
        else:
            if t_sim is None:
                t_cat_ind = []
            else:
                t_cat_ind = np.zeros(t_sim.shape[1])
        self.t_cat_ind = t_cat_ind

    # Prints pretty representation of the SepiaData object for users to check their setup.
    def __str__(self):
        res = ''
        res += 'This SepiaData instance implies the following:\n'
        if self.sim_only:
            res += 'This is a simulator (eta)-only model, y dimension %d\n' % self.sim_data.y.shape[1]
            if not self.kron_design:
                res += 'm  = %5d (number of simulated data)\n' % self.sim_data.x.shape[0]
                res += 'p  = %5d (number of inputs)\n' % self.sim_data.x.shape[1]
            else:
                res += 'This is a Kronecker separable simulation design with components: \n'
                for ii in range(len(self.sim_data.x)):
                    res += '   x component %d has m = %5d (simulated data design size) \n' % (ii,self.sim_data.x[ii].shape[0])
                    res += '   x component %d has p = %5d (number of inputs) \n' % (ii,self.sim_data.x[ii].shape[1])
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
        # Print info on categorical variables
        if not self.kron_design:
            if np.any(np.array(self.x_cat_ind) > 0):
                res += 'Categorical x input variables:\n'
                for i, ci in enumerate(self.x_cat_ind):
                    if ci > 0:
                        res += 'x index %d with %d categories\n' % (i, ci)
            if np.any(np.array(self.t_cat_ind) > 0):
                res += 'Categorical t input variables:\n'
                for i, ci in enumerate(self.t_cat_ind):
                    if ci > 0:
                        res += 't index %d with %d categories\n' % (i, ci)
        else:
            pass  # TODO OMG....
        return res

    def transform_xt(self, x_notrans=None, t_notrans=None, x=None, t=None):
        """
        Transforms sim_data x and t and obs_data x to lie in [0, 1], columnwise, or applies
        same transformation to new x and t.

        :param list/NoneType x_notrans: column indices of x that should not be transformed or None
        :param list/NoneType t_notrans: column indices of t that should not be transformed or None
        :param numpy.ndarray/NoneType x: new x values to transform to [0, 1] using same rules as original x data or None
        :param numpy.ndarray/NoneType t: new t values to transform to [0, 1] using same rules as original t data or None
        :returns: tuple of x_trans, t_trans if x and t arguments provided; otherwise returns (None, None)

        .. note:: A column is not transformed if min/max of the column values are equal, if the column is categorical,
                  or if the user specifies no transformation using x_notrans or t_notrans arguments.

        """

        # TODO fix this hack for kron development: x scaling needs to handle kron list
        if self.kron_design:
            return

        x_trans, t_trans = None, None
        if x_notrans is None:
            x_notrans = []
        if t_notrans is None:
            t_notrans = []
        if x_notrans is True:
            x_notrans = np.arange(self.sim_data.x.shape[1])

        # Transform x to unit hypercube
        # if not computed, compute orig x min and orig x max, accounting for notrans_x, all equal x, and categorical x
        if self.sim_data.orig_x_min is None or self.sim_data.orig_x_max is None or self.sim_data.x_trans is None:
            nx = self.sim_data.x.shape[1]
            orig_x_min = np.min(self.sim_data.x, 0, keepdims=True)
            orig_x_max = np.max(self.sim_data.x, 0, keepdims=True)
            # If any xmin/xmax are equal, don't transform
            xmm = orig_x_max - orig_x_min
            x_notrans = list(set(x_notrans) | set([i for i in range(nx) if xmm[:, i] == 0]))
            # If there are cat inds, do not transform
            if self.x_cat_ind is not None:
                x_notrans = list(set(x_notrans) | set([i for i in range(nx) if self.x_cat_ind[i] > 0]))
            orig_x_min[:, x_notrans] = 0
            orig_x_max[:, x_notrans] = 1
            self.sim_data.x_trans = (self.sim_data.x - orig_x_min) / (orig_x_max - orig_x_min)
            self.sim_data.orig_x_min = orig_x_min
            self.sim_data.orig_x_max = orig_x_max
            if not self.sim_only:
                self.obs_data.orig_x_min = orig_x_min
                self.obs_data.orig_x_max = orig_x_max
                self.obs_data.x_trans = (self.obs_data.x - orig_x_min) / (orig_x_max - orig_x_min)
        # If a new x was passed in, transform it
        if x is not None:
            x_trans = (x - self.sim_data.orig_x_min) / (self.sim_data.orig_x_max - self.sim_data.orig_x_min)
        # Transform t to unit hypercube
        if self.sim_data.t is not None:
            if t_notrans is True:
                t_notrans = np.arange(self.sim_data.t.shape[1])
            # if not computed, compute orig t min and orig t max, accounting for notrans_t, all equal t, and categorical t
            if self.sim_data.orig_t_min is None or self.sim_data.orig_t_max is None or self.sim_data.t_trans is None:
                nt = self.sim_data.t.shape[1]
                orig_t_min = np.min(self.sim_data.t, 0, keepdims=True)
                orig_t_max = np.max(self.sim_data.t, 0, keepdims=True)
                # If any tmin/tmax are equal, don't transform
                tmm = orig_t_max - orig_t_min
                t_notrans = list(set(t_notrans) | set([i for i in range(nt) if tmm[:, i] == 0]))
                # If there are cat inds, do not transform
                if self.t_cat_ind is not None:
                    t_notrans = list(set(t_notrans) | set([i for i in range(nt) if self.t_cat_ind[i] > 0]))
                orig_t_min[:, t_notrans] = 0
                orig_t_max[:, t_notrans] = 1
                self.sim_data.t_trans = (self.sim_data.t - orig_t_min) / (orig_t_max - orig_t_min)
                self.sim_data.orig_t_min = orig_t_min
                self.sim_data.orig_t_max = orig_t_max
                if not self.sim_only:
                    self.obs_data.orig_t_min = orig_t_min
                    self.obs_data.orig_t_max = orig_t_max
            # If a new t was passed in, transform it
            if t is not None:
                t_trans = (t - self.sim_data.orig_t_min) / (self.sim_data.orig_t_max - self.sim_data.orig_t_min)
        return x_trans, t_trans

    def standardize_y(self, center=True, scale='scalar'):
        """
        Standardizes both `sim_data` and `obs_data` outputs y based on sim_data.y mean/SD.

        :param bool center: subtract simulation mean (across observations)?
        :param string/bool scale: how to rescale: 'scalar': single SD over all demeaned data, 'columnwise': SD for each column of demeaned data, False: no rescaling
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
        Creates `K_sim` and `K_obs` basis functions using PCA on sim_data.y_std, or using given `K_sim` matrix.

        :param float/int n_pc: proportion in [0, 1] of variance, or an integer number of components
        :param numpy.ndarray/None K: a basis matrix on sim indices of shape (n_basis_elements, ell_sim) or None

        .. note:: if standardize_y() method has not been called first, it will be called automatically by this method.
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
        # :param float/int n_pc: number of components or a proportion of variance explained, in [0, 1].
        y_std = self.sim_data.y_std
        if y_std is None:
            print('WARNING: y not standardized, applying default standardization before PCA...')
            self.standardize_y()
        U, s, V = np.linalg.svd(y_std.T, full_matrices=False)
        s2 = np.square(s)
        if n_pc < 1:
            cum_var = s2 / np.sum(s2)
            pu = np.sum(np.cumsum(cum_var) < n_pc) + 1
        else:
            pu = int(n_pc)
        self.sim_data.K = np.transpose(np.dot(U[:, :pu], np.diag(s[:pu])) / np.sqrt(y_std.shape[0]))

    def create_D_basis(self, D_type='constant', D_obs=None, D_sim=None, norm=True):
        """
        Create `D_obs`, `D_sim` discrepancy bases. Can specify a type of default basis (constant/linear) or provide matrices.

        :param string D_type: 'constant' or 'linear' to set up constant or linear D_sim and D_obs
        :param numpy.ndarray/list/NoneType D_obs: a basis matrix on obs indices of shape (n_basis_elements, ell_obs),
                                                  or list of matrices for ragged observations.
        :param numpy.ndarray/NoneType D_sim: a basis matrix on sim indices of shape (n_basis_elements, sim_obs).
        :param bool norm: normalize D basis?

        .. note:: `D_type` parameter is ignored if `D_obs` and `D_sim` are provided.
        """
        # Return early if sim only or univariate output
        if self.sim_only:
            print('Model only has simulation data, skipping discrepancy...')
            return
        if self.scalar_out:
            print('Model has univariate output, skipping discrepancy...')
            return
        # Check if passed in D_sim/D_obs are correct shape and if so, set them into objects
        if D_sim is not None:
            if not D_sim.shape[1] == self.sim_data.y.shape[1]:
                raise TypeError('D_sim basis shape incorrect; second dim should match ell_sim')
            self.sim_data.D = D_sim
        if D_obs is not None:
            if self.ragged_obs:
                for i in range(len(D_obs)):
                    if not D_obs[i].shape[1] == (self.obs_data.y[i].shape[1] if self.obs_data.y[i].ndim == 2 else self.obs_data.y[i].shape[0]):
                        raise TypeError('D basis shape incorrect; second dim should match ell_obs')
            else:
                if not D_obs.shape[1] == self.obs_data.y.shape[1]:
                    raise TypeError('D_obs basis shape incorrect; second dim should match ell_obs')
            self.obs_data.D = D_obs
        elif D_type == 'constant':
            if self.ragged_obs:
                self.obs_data.D = [np.ones((1, self.obs_data.y[i].shape[0])) for i in range(len(self.obs_data.y))]
            else:
                self.obs_data.D = np.ones((1, self.obs_data.y.shape[1]))
            self.sim_data.D = np.ones((1, self.sim_data.y.shape[1]))
        elif D_type == 'linear':
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
                    norm_scl = np.sqrt(np.max(np.dot(self.obs_data.D[0], self.obs_data.D[0].T)))
                    for i in range(len(self.obs_data.D)):
                        self.obs_data.D[i] /= norm_scl
                else:
                    norm_scl = np.sqrt(np.max(np.dot(self.obs_data.D, self.obs_data.D.T)))
                    self.obs_data.D /= norm_scl

    def plot_K_basis(self, max_plots=4, obs=True):
        """
        Plots K basis elements for both sim and obs indices (if applicable). Only applies to multivariate-output models.

        :param int max_plots: maximum number of principal components to plot
        """
        # Return early if scalar out or basis not set up
        if self.scalar_out:
            print('Scalar output, no K basis to plot.')
            return
        if self.sim_data.K is None:
            print('K basis not set up, call create_K_basis() first.')
            return
        # Plot sim basis
        pu = self.sim_data.K.shape[0]
        ncol = 5
        nrow = int(np.ceil((min(pu, max_plots) + 1) / ncol)) # add 1 for mean line
        fig_sim, axs_sim = plt.subplots(nrow, ncol, figsize=(12, 2 * nrow))
        fig_sim.tight_layout()
        for i, ax in enumerate(axs_sim.flatten()):
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
        # If obs are present and requested, plot obs basis
        if not self.sim_only and obs:
            if self.ragged_obs:
                pu = np.array([k.shape[0] for k in self.obs_data.K])
                if np.all(pu == pu[0]): pu = pu[0]
                else: raise ValueError('first dimension in lists not equal')
            else:
                pu = self.obs_data.K.shape[0]
            ncol = 5
            nrow = int(np.ceil((min(pu,max_plots) + 1) / ncol)) # add 1 for mean line
            fig_obs, axs_obs = plt.subplots(nrow,ncol,figsize=(12, 2 * nrow))
            fig_obs.tight_layout()
            for i,ax in enumerate(axs_obs.flatten()):
                if i == 0: # plot mean line
                    if self.ragged_obs: ax.plot(self.obs_data.y_ind[i],np.mean(self.obs_data.K[i],axis=0))
                    else: ax.plot(self.obs_data.y_ind, np.mean(self.obs_data.K,axis=0))
                    ax.set_title('obs mean')
                    ax.set_ylabel('obs K basis')
                    ax.set_xlabel('obs y_ind')
                elif i < pu+1:
                    if self.ragged_obs: ax.plot(self.obs_data.y_ind[i],self.obs_data.K[i].T[:,i-1])
                    else: ax.plot(self.obs_data.y_ind, self.obs_data.K.T[:,i-1])
                    ax.set_title('PC %d' % (i))
                    ax.set_xlabel('obs y_ind')
                else:
                    ax.axis('off')
            return(fig_sim,fig_obs)
        else:
            return fig_sim

    def plot_K_weights(self, max_u_plot=5):
        """
        Plots K basis weights for both sim and obs data (if applicable). Only applies to multivariate-output models.

        :param int max_u_plot: max number of u's for which to plot vertical line over histogram of w's
        """
        # Return early if scalar out or basis not set up
        if self.scalar_out:
            print('Scalar output, no K weights to plot.')
            return
        if self.sim_data.K is None:
            print('K basis not set up, call create_K_basis() first.')
            return
        # Compute sim K weights
        pu = self.sim_data.K.shape[0]
        ncol = 5
        nrow = int(np.ceil(pu / ncol))
        w = np.dot(np.linalg.pinv(self.sim_data.K).T, self.sim_data.y_std.T).T

        fig_uw, axs_uw = plt.subplots(nrow,ncol,figsize=(10,2*nrow))
        fig_uw.tight_layout()

        # Compute obs K weights if obs are present
        if not self.sim_only and self.obs_data.K is not None:
            # set pu
            if self.ragged_obs:
                pu = np.array([k.shape[0] for k in self.obs_data.K])
                if np.all(pu == pu[0]): pu = pu[0]
                else: raise ValueError('first dimension in lists not equal')
            else:
                pu = self.obs_data.K.shape[0]
                    
            # No D
            if self.obs_data.D is None:
                pv = 0
                DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                # compute u
                if self.ragged_obs:
                    u = []
                    for i in range(len(self.obs_data.y_ind)):
                        DK = self.obs_data.K[i]
                        Lamy = np.eye(self.obs_data.y_ind[i].shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])
                        u.append(np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std[i].T])).T)
                    u = np.array(u)
                else:
                    DK = self.obs_data.K
                    Lamy = np.eye(self.obs_data.y_ind.shape[0]) # Identity with size len(y_ind) how to do this with ragged?
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    u = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T])).T
                            
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
                    return fig
                            
                else: # do u and w independently
                    raise ValueError('u.shape[1] != w.shape[1]')
                                
            else: # D
                if self.ragged_obs:
                    pv = np.array([d.shape[0] for d in self.obs_data.D])
                    if np.all(pv == pv[0]): pv = pv[0]
                    else: raise ValueError('first dimension in lists not equal')
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    u = []
                    v = []
                    for i in range(len(self.obs_data.D)):
                        DK = np.concatenate([self.obs_data.D[i], self.obs_data.K[i]])
                        Lamy = np.eye(self.obs_data.y_ind[i].shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std[i].T]))
                        v.append(vu[:pv].T)
                        u.append(vu[pv:].T)
                    u = np.array(u)
                    v = np.array(v)
                else:
                    pv = self.obs_data.D.shape[0]
                    DK = np.concatenate([self.obs_data.D, self.obs_data.K])  # (pu+pv, ell_obs)
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    Lamy = np.eye(self.obs_data.y_ind.shape[0])
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T]))
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

    
    def plot_u_w_pairs(self, max_plots=5, save=False):
        """
        Plots principal component basis weights for both sim and obs data (if applicable). Only applies to multivariate-output models.

        :param int max_plots: max number of principal components to plot
        """
        # Return early if scalar out or basis not set up
        if self.scalar_out:
            print('Scalar output, no K weights to plot.')
            return
        if self.sim_data.K is None:
            print('K basis not set up, call create_K_basis() first.')
            return
        pu = self.sim_data.K.shape[0]
        w = np.dot(np.linalg.pinv(self.sim_data.K).T, self.sim_data.y_std.T).T
                
        if not self.sim_only and self.obs_data.K is not None:
            if self.ragged_obs:
                pu = np.array([k.shape[0] for k in self.obs_data.K])
                if np.all(pu == pu[0]): pu = pu[0]
                else: raise ValueError('first dimension in lists not equal')
            else:
                pu = self.obs_data.K.shape[0]

            # No D
            if self.obs_data.D is None:
                pv = 0
                DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                if self.ragged_obs:
                    u = []
                    for i in range(len(self.obs_data.K)):
                        DK = self.obs_data.K[i]
                        Lamy = np.eye(self.obs_data.y_ind[i].shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        u.append(np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std[i].T])).T)
                    u = np.array(u)
                else:
                    DK = self.obs_data.K
                    Lamy = np.eye(self.obs_data.y_ind.shape[0])
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    u = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T])).T
                            
            else: # D
                if self.ragged_obs:
                    pv = np.array([d.shape[0] for d in self.obs_data.D])
                    if np.all(pv == pv[0]): pv = pv[0]
                    else: raise ValueError('first dimension in lists not equal')
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    u = []
                    v = []
                    for i in range(len(self.obs_data.D)):
                        DK = np.concatenate([self.obs_data.D[i], self.obs_data.K[i]])
                        Lamy = np.eye(self.obs_data.y_ind[i].shape[0])
                        DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                        vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std[i].T]))
                        v.append(vu[:pv].T)
                        u.append(vu[pv:].T)
                    u = np.array(u)
                    v = np.array(v)
                else:
                    pv = self.obs_data.D.shape[0]
                    DK = np.concatenate([self.obs_data.D, self.obs_data.K])  # (pu+pv, ell_obs)
                    DKridge = 1e-6 * np.diag(np.ones(pu + pv))  # (pu+pv, pu+pv)
                    Lamy = np.eye(self.obs_data.y_ind.shape[0])
                    DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])  # (pu+pv, pu+pv)
                    vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, self.obs_data.y_std.T]))
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
                if save: plt.savefig(save,dpi=300)
                return g.fig

    def plot_K_residuals(self):
        """
        Plots residuals after projection to K basis. Only applies to multivariate-output models.

        """
        # Return early if scalar out or basis not set up
        if self.scalar_out:
            print('Scalar output, no K residuals to plot.')
            return
        if self.sim_data.K is None:
            print('K basis not set up, call create_K_basis() first.')
            return
        if not self.sim_only and self.obs_data.K is not None:
            if isinstance(self.obs_data.K, list):
                print('plot_K_residuals cannot yet handle ragged observations')
                return
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
                
                fig_noD, axs_noD = plt.subplots(1,3,figsize=(4,6))
               
                axs_noD[0].plot(self.obs_data.y_ind, self.obs_data.y_std.squeeze().T)
                axs_noD[0].set_title('obs y_std')
                axs_noD[0].set_xlabel('obs y_ind')
                
                axs_noD[1].plot(self.obs_data.y_ind, proj.squeeze().T)
                axs_noD[1].set_title('obs projection reconstruction')
                axs_noD[1].set_xlabel('obs y_ind')
                
                axs_noD[2].plot(self.obs_data.y_ind, resid.squeeze().T, '-')
                axs_noD[2].set_title('obs projection residual')
                axs_noD[2].set_xlabel('obs y_ind')
                return fig_noD
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

    def plot_data(self,which_x=None,x_min=None,x_max=None,y_min=None,y_max=None,n_neighbors=3,max_sims=50,save=None):
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
        """
        if self.sim_only:
            print('plot_data does not currently work for sim_only models.')
            return
        if self.scalar_out:
            print('plot_data does not currently work for univariate output models.')
            return

        n = self.obs_data.x.shape[0]
        m = self.sim_data.x.shape[0]

        # plot up to 4 input space points
        if n > 4:
            # if no which_x or given which_x is out of bounds
            if which_x is None or (which_x is not None and not np.all(which_x)<n and not np.all(which_x>-1)):
                # choose 4 equally space input points to plot
                which_x = np.linspace(0,n-1,4,dtype=int)
            x_plot = self.obs_data.x[which_x,:]
        else:
            which_x = np.arange(0,n,1,dtype=int)
            x_plot = self.obs_data.x
        n_plots = x_plot.shape[0]

        # get axis limits
        if self.ragged_obs:
            if x_min is None: x_min = min(min([np.amin(k) for k in self.obs_data.y_ind]),np.amin(self.sim_data.y_ind))
            if x_max is None: x_max = max(max([np.amax(k) for k in self.obs_data.y_ind]),np.amax(self.sim_data.y_ind))
            if y_min is None: y_min = min(min([np.amin(k) for k in self.obs_data.y]),np.amin(self.sim_data.y))
            if y_max is None: y_max = max(max([np.amax(k) for k in self.obs_data.y]),np.amax(self.sim_data.y))
        else:
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
                if self.ragged_obs:
                    axs[i].plot(self.obs_data.y_ind[i], self.obs_data.y[which_x[i]],'--ko',label="Obs data")
                else:
                    axs[i].plot(self.obs_data.y_ind, self.obs_data.y[which_x[i]],'--ko',label="Obs data")

                # legend
                axs[i].legend()
                #axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            else:
                axs[i].axis('off')
                
        if save is not None: fig.savefig(save,dpi=300,bbox_inches='tight')
        return fig



