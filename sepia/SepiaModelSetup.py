#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set up model based on input SepiaData structure.

"""

# Compared to Matlab:
#     - No categorical vars
#     - No KP structure
#     - No lamVzGroups
#     - Obs data assumed to be on same grid
#
# At end should set up:
#     - num object, params incl. priors, bounds, default mcmc objects

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaDistCov import SepiaDistCov

import scipy.linalg
import numpy as np

def setup_model(data, Sigy=None, lamVzGroup=None):
    """
    Sets up SepiaModel object based on SepiaData object.

    :param data: SepiaData object
    :param Sigy: observation covariance matrix
    :param lamVzGroup: groups for discrepancy variance
    :return: instantiated SepiaModel object
    """

    # Check if things transformed, do a default thing, but give warning if not done by user
    if data.sim_data.y_std is None:
        print('Warning: you did not standardize y!')
        print('Continuing with default standardization.')
        print('To customize, call data.standardize_y on your SepiaData object first, or do manual standardization.')
        data.standardize_y(center=True, scale='scalar')

    # TODO this could get printed even if transform_xt was called, in the case that there is no t...
    # Trying this as a quick fix, need to make sure it works.
    #if data.sim_data.x_trans is None or data.sim_data.t_trans is None:
    if data.sim_data.x_trans is None:
        print('Warning: you did not rescale x/t to unit cube!')
        print('Continuing with default transformation to unit cube.')
        print('To customize, call data.transform_xt on your SepiaData object first, or do manual transformation.')
        data.transform_xt()

    # If not scalar out, check that K/D are set appropriately
    if not data.scalar_out:
        if data.sim_data.K is None:
            print('Warning: you did not set up the K basis!')
            print('Continuing with default PCA basis.')
            print('To customize, call data.create_K_basis on your SepiaData object.')
            data.create_K_basis()
        else:
            if not data.sim_only:
                if data.sim_data.K.shape[0] != data.obs_data.K.shape[0]:
                    raise TypeError('Sim and obs K shapes not conformal.')
        if not data.sim_only and data.obs_data.D is None:
            print('Warning: you did not set up a D basis!')
            print('To use one, call data.create_D_basis on your SepiaData object.')

    # Set up model object to modify
    model = SepiaModel()
    model.data = data
    num = model.num
    num.scalar_out = data.scalar_out
    num.sim_only = data.sim_only

    # Local references to data for initialization of model
    sim_data = data.sim_data
    obs_data = data.obs_data

    # Infer p, q, n, m, pu, pv
    m, p = sim_data.x.shape
    if sim_data.t is not None:
        q = sim_data.t.shape[1]
    else:
        q = 0
    ell_sim = sim_data.y.shape[1]
    if data.scalar_out:
        pu = 1
    else:
        pu = sim_data.K.shape[0]
    num.m, num.p, num.q, num.pu = m, p, q, pu
    if not data.sim_only:
        ell_obs = obs_data.y.shape[1] # TODO ragged
        n = obs_data.x.shape[0]
        if data.scalar_out or obs_data.D is None:
            pv = 0
        else:
            pv = obs_data.D.shape[0] # TODO ragged
        num.n, num.pv = n, pv
    else:
        num.n = 0
        num.pv = 0

    # Set up Sigy/Lamy
    if not data.sim_only:
        if Sigy is None:
            Sigy = np.diag(np.ones(ell_obs)) # TODO ragged
        if data.scalar_out:
            Lamy = 1/Sigy
        else:
            Lamy = np.linalg.inv(Sigy)

    # Set up GP inputs/DistCov objects
    if not data.sim_only:
        data.x = obs_data.x_trans
    else:
        data.x = np.array([], dtype=np.float).reshape((0, 1))
    num.x0Dist = SepiaDistCov(data.x)
    if sim_data.t_trans is not None:
        data.zt = np.concatenate([sim_data.x_trans, sim_data.t_trans], axis=1)
    else:
        data.zt = sim_data.x_trans
    num.ztDist = SepiaDistCov(data.zt)

    # TODO not yet handling lamVzGroups; see line 115 in Matlab code
    if not data.sim_only:
        # Check for lamVzGroups
        if lamVzGroup is not None:
            lamVzGnum = np.unique(lamVzGroup).shape[0]
        else:
            lamVzGroup = [0] * pv
            lamVzGnum = 1
        num.lamVzGroup = lamVzGroup
        num.lamVzGnum = lamVzGnum

    # Transform obs data using D, Kobs -> v, u
    if not data.sim_only: # TODO ragged obs
        if data.scalar_out:
            u = obs_data.y_std
            v = np.array([], dtype=float)
        else:
            DKridge = 1e-6 * np.diag(np.ones(pu+pv))       # (pu+pv, pu+pv)
            if obs_data.D is None:
                DK = obs_data.K
            else:
                DK = np.concatenate([obs_data.D, obs_data.K])  # (pu+pv, ell_obs)
            DKprod = np.linalg.multi_dot([DK, Lamy, DK.T]) # (pu+pv, pu+pv)
            vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy, obs_data.y_std.T]))
            v = vu[:pv, :].T
            u = vu[pv:, :].T
        num.u = u.reshape((n*pu, 1), order='F').copy()
        num.v = v.reshape((n*pv, 1), order='F').copy()
    else:
        num.u = np.array([], dtype=np.float).reshape((0, 1))
        num.v = np.array([], dtype=np.float).reshape((0, 1))

    # Transform sim data using Ksim -> w
    if data.scalar_out:
        w = sim_data.y_std
    else:
        w = np.dot(np.linalg.pinv(sim_data.K).T, sim_data.y_std.T).T
    num.w = w.reshape((m*pu, 1), order='F').copy()

    if num.scalar_out:
       num.uw = np.concatenate((num.u, num.w), axis=0)
    else:
       num.vuw = np.concatenate((num.v, num.u, num.w), axis=0)
       num.vu = np.concatenate((num.v, num.u), axis=0)

    # Compute LamSim
    if data.scalar_out:
        LamSim = np.ones(1)
    else:
        LamSim = np.diag(np.dot(sim_data.K, sim_data.K.T))
    num.LamSim = LamSim

    # Compute LamObs
    if not data.sim_only: # TODO ragged obs
        if data.scalar_out:
            LamObs = Lamy * np.diag(np.ones(n))
            rankLO = np.linalg.matrix_rank(LamObs)
        else:
            if obs_data.D is None: # TODO can do something simpler if DK is just K since orthogonal (may apply in a few places with DK)
                DK = obs_data.K
            else:
                DK = np.concatenate([obs_data.D, obs_data.K]) # Note: our DK = DK' in matlab
            DKridge = 1e-6 * np.diag(np.ones(pu + pv))
            DKprod = np.linalg.multi_dot([DK, Lamy, DK.T])
            LamObs = scipy.linalg.block_diag(*[DKprod for i in range(n)])
            rankLO = np.linalg.matrix_rank(LamObs, hermitian=True)  # Same as value in matlab
            LamObs = LamObs + scipy.linalg.block_diag(*[DKridge for i in range(n)])
            # Reorder indices: is currently v1 u1 v2 u2 ... want v1 v2 v3 ... u1 u2 ...
            inew = []
            for kk in range(pv):
                inew += list(range(kk, n * (pu + pv), pu + pv))
            for kk in range(pu):
                inew += list(range(pv + kk, n * (pu + pv), pu + pv))
            inew = np.array(inew)
            LamObs = LamObs[inew[:, None], inew[None, :]]
        SigObs = np.linalg.inv(LamObs + 1e-8*np.diag(np.ones(LamObs.shape[0])))
        num.SigObs = SigObs
    else:
        rankLO = 0

    # TODO Process optional inputs for lamWOs, lamOs
    # Compute prior correction for lamOs, lamWOs
    if not data.sim_only:
        tot_elements = ell_obs * n # TODO need to change if allowing ragged arrays
        lamOs_a_corr = 0.5 * (tot_elements - rankLO)
        lamOs_b_corr = 0
        if not data.scalar_out:
            if obs_data.D is None:
                DK = obs_data.K
            else:
                DK = np.concatenate([obs_data.D, obs_data.K]) # Note: our DK = DK' in matlab
            DKvu = np.dot(DK.T, np.concatenate([v, u], axis=1).T)
            for i in range(n):
                resid = obs_data.y_std[i, :] - DKvu[:, i]
                lamOs_b_corr += 0.5 * np.sum(np.linalg.multi_dot([resid, Lamy, resid]))

    lamWOs_a_corr = 0.5 * (ell_sim - pu) * m
    if data.scalar_out:
        lamWOs_b_corr = 0
    else:
        y_sim_std_hat = np.dot(w, sim_data.K)
        lamWOs_b_corr = 0.5 * np.sum(np.square(sim_data.y_std - y_sim_std_hat)) # same as matlab

    # Set up params/priors
    if model.num.sim_only:
        model.set_params_sim_only(lamWOs_a_corr, lamWOs_b_corr)
    elif pv == 0:
        model.set_params_noD(lamOs_a_corr, lamOs_b_corr, lamWOs_a_corr, lamWOs_b_corr)
    else:
        model.set_params_full(lamOs_a_corr, lamOs_b_corr, lamWOs_a_corr, lamWOs_b_corr)

    # TODO no categorical indicator, or separable cov structure

    # TODO process optional thetaconstraints object

    return model



# #########
# #
# #########
# if __name__ == "__main__":
#
#     n_obs = 5      # number of obs data
#     n_sim = 700    # number of sim data
#     p = 1          # dimension of x (observed inputs)
#     q = 5          # dimension of t (design inputs)
#
#     ### Univariate test case
#     t = np.random.uniform(-20, 20, (n_sim, q))
#     x = 0.5 * np.ones((n_sim, p))
#     y = 5 * np.random.normal(0, 1, n_sim)
#
#     # Sim data only
#     d1 = SepiaData(x_sim=x, y_sim=y, t_sim=t)
#     d1.standardize_y()
#     d1.transform_xt()
#     print(d1)
#
#     print('univariate sim-only test case')
#     model = setup_model(d1)
#     print('')
#
#     # Sim and obs data
#     x_obs = 0.5 * np.ones((n_obs, p))
#     y_obs = 10 + 5 * np.random.normal(0, 1, n_obs)
#     d2 = SepiaData(x_sim=x, y_sim=y, t_sim=t, x_obs=x_obs, y_obs=y_obs)
#     d2.standardize_y()
#     d2.transform_xt()
#     print(d2)
#
#     print('univariate sim and obs test case')
#     model = setup_model(d2)
#     print('')
#
#     ### Multivariate test case
#     ell_sim = 1000 # dimension of multivariate y_sim
#     ell_obs = 258  # dimension of multivariate y_obs
#     t = np.random.uniform(-20, 20, (n_sim, q))
#     x = 0.5 * np.ones((n_sim, p))
#     y_ind = np.linspace(0, 100, ell_sim)
#     y = 20 * np.square((y_ind[None, :] - 50)/75.) + 5 * np.random.normal(0, 1, (n_sim, 1))
#
#     # Sim data only
#     d1 = SepiaData(x_sim=x, y_sim=y, t_sim=t, y_ind_sim=y_ind)
#     d1.standardize_y()
#     d1.transform_xt()
#     d1.create_K_basis(n_pc=3)
#     print(d1)
#     print('multivariate sim-only test case')
#     model = setup_model(d1)
#     print('')
#
#     # Sim and obs data
#     x_obs = 0.5 * np.ones((n_obs, p))
#     y_obs_ind = np.linspace(10, 85, ell_obs)
#     y_obs = 20 * np.square((y_obs_ind[None, :] - 50)/75.) + 10 + 5 * np.random.normal(0, 1, (n_obs, 1))
#     d2 = SepiaData(x_sim=x, y_sim=y, t_sim=t, y_ind_sim=y_ind, x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_obs_ind)
#     d2.standardize_y()
#     d2.transform_xt()
#     d2.create_K_basis(n_pc=15)
#     d2.create_D_basis('constant')
#     print(d2)
#
#     print('multivariate sim and obs test case')
#     model = setup_model(d2)
#     print('')