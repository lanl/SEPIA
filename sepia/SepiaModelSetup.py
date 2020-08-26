#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaDistCov import SepiaDistCov

import scipy.linalg
import numpy as np


# TODO no categorical indicator, or separable cov structure
# TODO process optional thetaconstraints object
def setup_model(data, Sigy=None, lamVzGroup=None, LamSim=None):
    """
    Sets up SepiaModel object based on SepiaData object.

    :param data: SepiaData object
    :param Sigy: observation covariance matrix (default is identity)
    :param lamVzGroup: groups for lamVz (otherwise single lamVz for all D basis functions)
    :return: instantiated SepiaModel object
    """

    # Check if things transformed, do a default thing, but give warning if not done by user
    if data.sim_data.y_std is None:
        print('Warning: you did not standardize y!')
        print('Continuing with default standardization.')
        print('To customize, call data.standardize_y on your SepiaData object first, or do manual standardization.')
        data.standardize_y(center=True, scale='scalar')

    if data.sim_data.x_trans is None and data.sim_data.t_trans is None:
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
                if data.ragged_obs:
                    obs_K_shapes = np.array([k.shape[0] for k in data.obs_data.K])
                    if not np.all(data.sim_data.K.shape[0] == obs_K_shapes):
                        raise ValueError('Sim and obs K shapes not conformal.')
                else:
                    if data.sim_data.K.shape[0] != data.obs_data.K.shape[0]:
                        raise ValueError('Sim and obs K shapes not conformal.')
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
        if data.ragged_obs:
            ell_obs = [obs_data.y[i].shape for i in range(len(obs_data.y))]
        else:
            ell_obs = obs_data.y.shape[1]
        n = obs_data.x.shape[0]
        if data.scalar_out or obs_data.D is None:
            pv = 0
        else:
            if data.ragged_obs:
                pv = obs_data.D[0].shape[0]
            else:
                pv = obs_data.D.shape[0]
        num.n, num.pv = n, pv
    else:
        num.n = 0
        num.pv = 0

    # Set up Sigy/Lamy
    if not data.sim_only:
        if Sigy is None:
            if data.ragged_obs:
                Sigy = [np.atleast_2d(np.diag(np.ones(ell_obs[i]))) for i in range(len(ell_obs))]
            else:
                Sigy = np.diag(np.ones(ell_obs))
        if data.scalar_out:
            Lamy = 1/Sigy
        else:
            if data.ragged_obs:
                Lamy = [np.linalg.inv(Sigy[i]) for i in range(len(Sigy))]
            else:
                Lamy = np.linalg.inv(Sigy)

    # Set up GP inputs/DistCov objects
    if not data.sim_only:
        data.x = obs_data.x_trans
    else:
        data.x = np.array([], dtype=np.float).reshape((0, 1))
    num.x0Dist = SepiaDistCov(data.x, cat_ind=data.x_cat_ind)
    if sim_data.t_trans is not None:
        data.zt = np.concatenate([sim_data.x_trans, sim_data.t_trans], axis=1)
    else:
        data.zt = sim_data.x_trans
    num.ztDist = SepiaDistCov(data.zt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))

    if not data.sim_only:
        # Check for lamVzGroups, validate
        if lamVzGroup is not None:
            lamVzGnum = np.unique(lamVzGroup).shape[0]
        else:
            lamVzGroup = [0] * pv
            lamVzGnum = 1 if pv > 0 else 0
        num.lamVzGroup = lamVzGroup
        num.lamVzGnum = lamVzGnum
        for i in range(lamVzGnum):
            if i not in lamVzGroup:
                raise TypeError('lamVzGroup not correct')

    # Transform obs data using D, Kobs -> v, u
    if not data.sim_only:
        if data.scalar_out:
            u = obs_data.y_std
            v = np.array([], dtype=float)
        else:
            DKridge = 1e-6 * np.diag(np.ones(pu + pv)) # (pu+pv, pu+pv)
            if data.ragged_obs:
                v = []
                u = []
                for i in range(len(data.obs_data.y)):
                    if obs_data.D is None:
                        DK = obs_data.K[i]
                    else:
                        DK = np.concatenate([obs_data.D[i], obs_data.K[i]])  # (pu+pv, ell_obs)
                    DKprod = np.linalg.multi_dot([DK, Lamy[i], DK.T]) # (pu+pv, pu+pv)
                    y_tmp = obs_data.y_std[i][:, None]
                    vu = np.dot(np.linalg.inv(DKprod + DKridge), np.linalg.multi_dot([DK, Lamy[i], y_tmp]))
                    v.append(vu[:pv, :].T)
                    u.append(vu[pv:, :].T)
                v = np.concatenate(v)
                u = np.concatenate(u)
            else:
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
        num.LamSim = np.ones(1)
    elif LamSim is None:
        num.LamSim = np.diag(np.dot(sim_data.K, sim_data.K.T))
    else:
        num.LamSim = LamSim

    # Compute LamObs
    if not data.sim_only:
        if data.scalar_out:
            LamObs = Lamy * np.diag(np.ones(n))
            rankLO = np.linalg.matrix_rank(LamObs)
        else:
            if obs_data.D is None:
                if data.ragged_obs:
                    DK = [obs_data.K[i] for i in range(n)]
                else:
                    DK = obs_data.K
            else:
                if data.ragged_obs:
                    DK = [np.concatenate([obs_data.D[i], obs_data.K[i]]) for i in range(n)]
                else:
                    DK = np.concatenate([obs_data.D, obs_data.K]) # Note: our DK = DK' in matlab
            DKridge = 1e-6 * np.diag(np.ones(pu + pv))
            if data.ragged_obs:
                DKprod = [np.linalg.multi_dot([DK[i], Lamy[i], DK[i].T]) for i in range(n)]
            else:
                DKprod = [np.linalg.multi_dot([DK, Lamy, DK.T]) for i in range(n)]
            if LamSim is None:
                LamObs = scipy.linalg.block_diag(*DKprod)
            else:
                LamObs = np.diag(LamSim)
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

    # Compute prior correction for lamOs, lamWOs
    if not data.sim_only:
        if data.ragged_obs:
            tot_elements = np.sum(ell_obs)
        else:
            tot_elements = ell_obs * n
        lamOs_a_corr = 0.5 * (tot_elements - rankLO)
        lamOs_b_corr = 0
        if not data.scalar_out:
            if obs_data.D is None:
                if data.ragged_obs:
                    DK = [obs_data.K[i] for i in range(n)]
                else:
                    DK = obs_data.K
            else:
                if data.ragged_obs:
                    DK = [np.concatenate([obs_data.D[i], obs_data.K[i]]) for i in range(n)]
                else:
                    DK = np.concatenate([obs_data.D, obs_data.K]) # Note: our DK = DK' in matlab
            if data.ragged_obs:
                DKvu = [np.dot(DK[i].T, np.concatenate([v[None, i, :], u[None, i, :]], axis=1).T) for i in range(n)]
                for i in range(n):
                    resid = obs_data.y_std[i] - DKvu[i].squeeze()
                    lamOs_b_corr += 0.5 * np.sum(np.linalg.multi_dot([resid, Lamy[i], resid]))
            else:
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

    # If cat ind, fix start values, bounds, prior variance for corresponding theta
    for i in range(len(data.t_cat_ind)):
        if data.t_cat_ind[i] > 0:
            model.params.theta.val[:, i] = 1
            model.params.theta.prior.params[1][:, i] = np.inf
            model.params.theta.prior.bounds[0][:, i] = -np.inf
            model.params.theta.prior.bounds[1][:, i] = np.inf

    return model



