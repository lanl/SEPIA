#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import numpy as np
import matplotlib.pyplot as plt
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
    :var numpy.ndarray/list/NoneType xt_sim_sep: for separable design, list of kronecker composable matrices
    :var bool dummy_x: is there a dummy x? (used in problems where no x is provided)
    :var bool sep_design: is there a Kronecker separable design?
    """

    def __init__(self, x_sim=None, t_sim=None, y_sim=None, y_ind_sim=None, x_obs=None, y_obs=None, Sigy=None, y_ind_obs=None,
                 x_cat_ind=None, t_cat_ind=None, xt_sim_sep=None):
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
        :param numpy.ndarray/NoneType Sigy: optional observation covariance matrix (default is identity)
        :param numpy.ndarray/list/NoneType x_cat_ind: indices of x that are categorical (0 = not cat, int > 0 = how many categories), or None
        :param numpy.ndarray/list/NoneType t_cat_ind: indices of t that are categorical (0 = not cat, int > 0 = how many categories), or None
        :param numpy.ndarray/list/NoneType xt_sim_sep: for separable design, list of kronecker composable matrices; it is a list of 2 or
                                                       more design components that, through Kronecker expansion, produce the full input space (`x` and `t`) for the simulations.
        :raises: TypeError if shapes not conformal or required data missing.

        .. note: At least one of x_sim and t_sim must be provided, and y_sim must always be provided.

        """
        self.sep_design = xt_sim_sep is not None
        self.dummy_x = (not self.sep_design and x_sim is None) or \
                       (self.sep_design and y_obs is not None and x_obs is None)
        self.sim_only = y_obs is None

        # Initial Checks
        if y_sim is None:
            raise TypeError('y_sim is required to set up model.')
        if not self.sep_design:
            if y_obs is not None and ((x_obs is None and x_sim is not None) or (x_obs is not None and x_sim is None)):
                raise ValueError('x_sim and x_obs must both be either not None or None (which is the no-x model case)')
            if x_sim is None and t_sim is None:
                raise TypeError('At least one of x_sim or t_sim is required to set up model.')

        if self.dummy_x:
            if y_obs is not None:
                x_obs = 0.5 * np.ones((len(y_obs), 1))  # sets up dummy x_obs
            if not self.sep_design: # set up dummy_x in x_sim, or delays until sep/kron processing just below
                x_sim = 0.5 * np.ones((t_sim.shape[0], 1))

        if self.sep_design:
            if x_sim is not None or t_sim is not None:
                raise ValueError('Cannot specify x_sim or t_sim if separable design is supplied')
            if self.dummy_x: # augment the composed design with dummy_x column
                xt_sim_sep.insert(0,np.array([0.5]).reshape(1,1))
            # Expand out the design from the components by kronecker product into x_sim and t_sim (as needed)
            temp_des=xt_sim_sep[-1]
            for ndes in reversed(xt_sim_sep[:-1]):
                r1,r2=np.meshgrid(np.arange(ndes.shape[0]),np.arange(temp_des.shape[0]))
                temp_des=np.hstack((ndes[r1.reshape(-1,order='F'),:],temp_des[r2.reshape(-1,order='F'),:]))
            # separate the composed design into x and t components
            if self.sim_only: # Emulator-only model
                x_sim=temp_des # the design can only be attributed to x's
            else:   # extract the shape
                p=x_obs.shape[1]
                x_sim=temp_des[:,:p]
                t_sim=temp_des[:,p:]

        # At this point, dummy_x should be place if needed
        # if it's a separable design, that's composed and split into x_sim and t_sim appropriately
        # the separable design components will be used in logLik and predict, nobody else needs to worry about it now
        # (except carrying it along in SetupModel

        self.sim_data = DataContainer(x=x_sim, y=y_sim, t=t_sim, y_ind=y_ind_sim, xt_sep_design=xt_sim_sep)

        self.scalar_out = (self.sim_data.y.shape[1] == 1)

        if self.sim_only:
            self.obs_data = None
        else:
            if x_sim.shape[1] != x_obs.shape[1]:
                raise TypeError('x_sim and x_obs do not contain the same number of variables/columns.')
            self.obs_data = DataContainer(x=x_obs, y=y_obs, y_ind=y_ind_obs, Sigy=Sigy)
            self.sim_only = False
            self.ragged_obs = isinstance(y_obs, list)

        # Set up Sigy
        if not self.sim_only:
            if self.obs_data.Sigy is None:
                if self.ragged_obs:
                    ell_obs = [self.obs_data.y[i].shape for i in range(len(self.obs_data.y))]
                    self.obs_data.Sigy = [np.atleast_2d(np.diag(np.ones(ell_obs[i]))) for i in range(len(ell_obs))]
                else:
                    ell_obs = self.obs_data.y.shape[1]
                    self.obs_data.Sigy = np.diag(np.ones(ell_obs))

        # Process categorical indices
        if x_cat_ind is not None:
            if len(x_cat_ind) != x_sim.shape[1]:
                raise TypeError('x_cat_ind length should equal p.')
            for i, ci in enumerate(x_cat_ind):
                if ci > 0 and ci != np.max(x_sim[:, i]):
                    raise TypeError('Nonzero values of x_cat_ind should equal number of categories.')
        else:
            x_cat_ind = np.zeros(x_sim.shape[1])
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
                    res += 'pv not set, indicating (unusual case of) no discrepancy; call method create_D_basis to fix \n'
        # Info on separable design, if that's in place.
        if self.sep_design:
            res += 'This is a separable simulation design with components: \n'
            for ii in range(len(self.sim_data.xt_sep_design)):
                res += '   x component %d has m = %5d (simulated data design size) \n' % (
                ii, self.sim_data.xt_sep_design[ii].shape[0])
                res += '   x component %d has p = %5d (number of inputs) \n' % (
                ii, self.sim_data.xt_sep_design[ii].shape[1])
        # Print info on categorical variables
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
        return res

    def transform_xt(self, x_notrans=None, t_notrans=None, x_range=None, t_range=None, x=None, t=None):
        """
        Transforms sim_data x and t and obs_data x to lie in [0, 1], columnwise, or applies
        same transformation to new x and t.

        :param list/NoneType x_notrans: column indices of x that should not be transformed or None
        :param list/NoneType t_notrans: column indices of t that should not be transformed or None
        :param numpy.ndarray/NoneType x: new x values to transform to [0, 1] using same rules as original x data or None
        :param numpy.ndarray/NoneType t: new t values to transform to [0, 1] using same rules as original t data or None
        :param numpy.ndarray/NoneType x_range: user specified data ranges, first row is min, second row is max for each variable
        :param numpy.ndarray/NoneType t_range: user specified data ranges, first row is min, second row is max for each variable
        :returns: tuple of x_trans, t_trans if x and t arguments provided; otherwise returns (None, None)

        .. note:: A column is not transformed if min/max of the column values are equal, if the column is categorical,
                  or if the user specifies no transformation using x_notrans or t_notrans arguments.

        """

        x_trans, t_trans = None, None
        if x_notrans is None:
            x_notrans = []
        if t_notrans is None:
            t_notrans = []
        if x_notrans is True:
            x_notrans = np.arange(self.sim_data.x.shape[1])

        # making notes to transform the separable design elements, if needed
        transform_sep = False

        # Transform x to unit hypercube or user-specified ranges
        # if not computed, compute orig x min and orig x max, accounting for notrans_x, all equal x, and categorical x
        if self.sim_data.orig_x_min is None or self.sim_data.orig_x_max is None or self.sim_data.x_trans is None:
            if self.sep_design:
                transform_sep=True
            nx = self.sim_data.x.shape[1]
            if x_range is None:
                orig_x_min = np.min(self.sim_data.x, 0, keepdims=True)
                orig_x_max = np.max(self.sim_data.x, 0, keepdims=True)
            else:
                if x_range.shape[0] !=2:
                    raise ValueError('user-specified ranges are first row min, second row max')
                if x_range.shape[1] != nx:
                    raise ValueError('user-specified ranges must be given for every x variable')
                orig_x_min = x_range[[0],:]
                orig_x_max = x_range[[1],:]
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

        # Transform t to unit hypercube or user-specified ranges
        if self.sim_data.t is not None:
            if t_notrans is True:
                t_notrans = np.arange(self.sim_data.t.shape[1])
            # if not computed, compute orig t min and orig t max, accounting for notrans_t, all equal t, and categorical t
            if self.sim_data.orig_t_min is None or self.sim_data.orig_t_max is None or self.sim_data.t_trans is None:
                nt = self.sim_data.t.shape[1]
                if t_range is None:
                    orig_t_min = np.min(self.sim_data.t, 0, keepdims=True)
                    orig_t_max = np.max(self.sim_data.t, 0, keepdims=True)
                else:
                    if t_range.shape[0] != 2:
                        raise ValueError('user-specified ranges are first row min, second row max')
                    if t_range.shape[1] != nt:
                        raise ValueError('user-specified ranges must be given for every t variable')
                    orig_t_min = t_range[0, :]
                    orig_t_max = t_range[1, :]
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

        if transform_sep:
            self.sim_data.xt_sep_design_orig = self.sim_data.xt_sep_design.copy()
            if self.sim_data.orig_t_min is not None:
                sep_min = np.hstack((self.sim_data.orig_x_min, self.sim_data.orig_t_min))
                sep_max = np.hstack((self.sim_data.orig_x_max, self.sim_data.orig_t_max))
            else:
                sep_min = self.sim_data.orig_x_min
                sep_max = self.sim_data.orig_x_max
            tind=0
            for ii,dele in enumerate(self.sim_data.xt_sep_design):
                dlen = dele.shape[1]
                self.sim_data.xt_sep_design[ii] = \
                    (dele - sep_min[0,tind:tind+dlen]) / (sep_max[0,tind:tind+dlen] - sep_min[0,tind:tind+dlen])
                tind = tind + dlen

        return x_trans, t_trans

    def standardize_y(self, center=True, scale='scalar', y_mean=None, y_sd=None):
        """
        Standardizes both `sim_data` and `obs_data` outputs y based on sim_data.y mean/SD.

        :param bool center: subtract simulation mean (across observations)?
        :param string/bool scale: how to rescale: 'scalar': single SD over all demeaned data, 'columnwise': SD for each column of demeaned data, False: no rescaling
        :param numpy.ndarray/float/NoneType y_mean: y_mean for sim; optional, should match length of y_ind_sim or be scalar
        :param numpy.ndarray/float/NoneType y_sd: y_sd for sim; optional, should match length of y_ind_sim or be scalar
        """
        if center:
            if y_mean is None:
                self.sim_data.orig_y_mean = np.mean(self.sim_data.y, 0)
            else:
                self.sim_data.orig_y_mean = y_mean
        else:
            self.sim_data.orig_y_mean = 0.
        y_dm = self.sim_data.y - self.sim_data.orig_y_mean
        if y_sd is not None:
            self.sim_data.orig_y_sd = y_sd
        else:
            if scale == 'scalar':
                self.sim_data.orig_y_sd = np.std(y_dm, ddof=1)
            elif scale == 'columnwise':
                self.sim_data.orig_y_sd = np.std(y_dm, ddof=1, axis=0)
            elif scale is False:
                self.sim_data.orig_y_sd = 1.
            else:
                raise ValueError('standardize_y: invalid value for scale parameter, allowed are {''scalar'',''columnwise'',False}')
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
            def cov_norm(ysd):
                if np.isscalar(ysd):
                    return ysd**2
                ysd=ysd.reshape((1,-1))
                return(ysd.T @ ysd)
            if self.ragged_obs:
                ty_std=[]; tSigy_std=[]
                for i in range(len(self.obs_data.y)):
                    ty_std.append( (self.obs_data.y[i] - self.obs_data.orig_y_mean[i]) / self.obs_data.orig_y_sd[i] )
                    tSigy_std.append(self.obs_data.Sigy[i] / cov_norm(self.obs_data.orig_y_sd[i]) )
            else:
                ty_std = (self.obs_data.y - self.obs_data.orig_y_mean) / self.obs_data.orig_y_sd
                tSigy_std = self.obs_data.Sigy / cov_norm(self.obs_data.orig_y_sd)
            self.obs_data.y_std = ty_std
            self.obs_data.Sigy_std=tSigy_std

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
            if not isinstance(K,np.ndarray):
                raise TypeError('create_K_basis: K specified must be a numpy ndarray')
            if len(K.shape)!=2 or K.shape[1]!=self.sim_data.y.shape[1]:
                raise ValueError('create_K_basis: must be 2D, and K and y_sim must have the same second dimension')
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
