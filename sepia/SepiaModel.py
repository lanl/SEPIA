

import numpy as np
from sepia.SepiaParam import SepiaParam, SepiaParamList
from sepia.SepiaDistCov import SepiaDistCov
from sepia.SepiaLogLik import compute_log_lik
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import scipy.linalg
import os
import pickle

from sepia.contrib import emufree_calib_model


class SepiaModelBase:
    """
    SepiaModel class contains data, SepiaParam objects, and precomputed elements for the likelihood.

    :var sepia.SepiaData data: instantiated `sepia.SepiaData` object
    :var sepia.ModelContainer num: `sepia.ModelContainer` for computed numerical elements for use in evaluating likelihood/posterior
    :var sepia.SepiaParamList params: `sepia.SepiaParamList` containing all `SepiaParam` objects for the model and `mcmcList` (references to params involved in MCMC)
    :var bool verbose: print verbose output for this model?
    """

    def __init__(self, data, lamVzGroup=None, theta_fcon=None, theta_init=None, LamSim=None):
        """
        Sets up `SepiaModel` object based on instantiated `SepiaData` object.

        :param sepia.SepiaData data: instantiated `sepia.SepiaData` object with all transformations and basis creation done
        :param numpy.ndarray/list lamVzGroup: indicate groups for lamVz (otherwise single lamVz for all D basis functions)
        :param function/NoneType theta_fcon: constraint function for thetas; should take entire theta array and return True/False for constraint satisfied
        :param numpy.ndarray/NoneType theta_init: if using theta_fcon, should provide theta_init array that satisfies constraints
        :param numpy.ndarray LamSim: option to pass custom `LamSim` (warning: don't use this unless know what you're doing!)
        :return: instantiated `sepia.SepiaModel` object
        :raises ValueError: if `K_sim` and `K_obs` shapes are not conformal
        :raises TypeError: if `lamVzGroup` is not the correct size or specification

        .. note:: if `SepiaData` input transformations have not been done, a warning will be printed and default transformations
                  will be done to `x`, `t`, and `y`. Similarly, a `K` basis will be created if you did not do that first.
                  Since a `D` basis is not required, this is not created automatically, but a warning will be printed in case
                  creating a model without a `D` basis was not intentional.

        """

        self.verbose = False
        self.data = data
        self.num = ModelContainer() # num for numeric state
        self.num.scalar_out = data.scalar_out
        self.num.sim_only = data.sim_only
        self.num.mean_basis = data.mean_basis

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
                print('Warning: No D basis, proceeding with a no-discrepancy model.')
                print('To use a D basis (normally recommended), call data.create_D_basis on your SepiaData object.')

        # Local references to data for initialization of model
        sim_data = data.sim_data
        obs_data = data.obs_data

        # Infer p, q, n, m, pu, pv
        p = sim_data.x.shape[1]
        m=sim_data.y.shape[0]
        if sim_data.t is not None:
            q = sim_data.t.shape[1]
        else:
            q = 0
        ell_sim = sim_data.y.shape[1]
        if data.scalar_out:
            pu = 1
        else:
            pu = sim_data.K.shape[0]
        self.num.m, self.num.p, self.num.q, self.num.pu = m, p, q, pu
        if not data.sim_only:
            if data.ragged_obs:
                ell_obs = [obs_data.y[i].shape for i in range(len(obs_data.y))]
            else:
                ell_obs = obs_data.y.shape[1]
            n = obs_data.x.shape[0]
            if obs_data.D is None:
                pv = 0
            elif data.scalar_out:
                pv = 1
            else:
                if data.ragged_obs:
                    pv = obs_data.D[0].shape[0]
                else:
                    pv = obs_data.D.shape[0]
            self.num.n, self.num.pv = n, pv
        else:
            self.num.n = 0
            self.num.pv = 0

        # Set up GP inputs/DistCov objects
        if not data.sim_only:
            data.x = obs_data.x_trans
            self.num.x0Dist = SepiaDistCov(data.x, cat_ind=data.x_cat_ind)
        else:
            data.x = np.array([], dtype=float).reshape((0, 1))
            self.num.x0Dist = SepiaDistCov(data.x)

        if sim_data.t_trans is not None:
            data.zt = np.concatenate([sim_data.x_trans, sim_data.t_trans], axis=1)
        else:
            data.zt = sim_data.x_trans

        if self.data.sep_design:
            data.ztSep=sim_data.xt_sep_design
            self.num.ztSepDist=[]
            cat_ind = np.concatenate([data.x_cat_ind, data.t_cat_ind])
            tind=0
            for dele in sim_data.xt_sep_design:
                tlen=dele.shape[1]
                self.num.ztSepDist.append(SepiaDistCov(dele,cat_ind=cat_ind[tind:tind+tlen]))
                tind=tind+tlen
            #self.num.ztDist = SepiaDistCov(data.zt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))
        else:
            self.num.ztDist = SepiaDistCov(data.zt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))

        if not data.sim_only:
            # Check for lamVzGroups, validate
            if lamVzGroup is not None:
                lamVzGnum = np.unique(lamVzGroup).shape[0]
            else:
                lamVzGroup = [0] * pv
                lamVzGnum = 1 if pv > 0 else 0
            self.num.lamVzGroup = lamVzGroup
            self.num.lamVzGnum = lamVzGnum
            for i in range(lamVzGnum):
                if i not in lamVzGroup:
                    raise TypeError('lamVzGroup not correct')

        # Transform obs data using D, Kobs -> v, u
        if not data.sim_only:
            # Data observation error precision Lamy will be used for setup
            if data.ragged_obs:
                Lamy = [np.linalg.inv(obs_data.Sigy_std[i]) for i in range(len(obs_data.Sigy_std))]
            else:
                Lamy = np.linalg.inv(obs_data.Sigy_std)

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
            self.num.u = u.reshape((n*pu, 1), order='F').copy()
            self.num.v = v.reshape((n*pv, 1), order='F').copy()
        else:
            self.num.u = np.array([], dtype=float).reshape((0, 1))
            self.num.v = np.array([], dtype=float).reshape((0, 1))

        # Transform sim data using Ksim -> w
        if data.scalar_out:
            w = sim_data.y_std
        else:
            w = np.dot(np.linalg.pinv(sim_data.K).T, sim_data.y_std.T).T
        self.num.w = w.reshape((m * pu, 1), order='F').copy()

        if self.num.scalar_out:
            self.num.uw = np.concatenate((self.num.u, self.num.w), axis=0)
            self.num.vu = self.num.u
        else:
            self.num.vuw = np.concatenate((self.num.v, self.num.u, self.num.w), axis=0)
            self.num.vu = np.concatenate((self.num.v, self.num.u), axis=0)

        # Compute LamSim
        if data.scalar_out:
            self.num.LamSim = np.ones(1)
        elif LamSim is None:
            self.num.LamSim = np.diag(np.dot(sim_data.K, sim_data.K.T))
        else:
            self.num.LamSim = LamSim

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
            self.num.SigObs = SigObs
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
        if self.num.sim_only:
            self.set_params_sim_only(lamWOs_a_corr, lamWOs_b_corr)
        elif pv == 0:
            self.set_params_noD(lamOs_a_corr, lamOs_b_corr, lamWOs_a_corr, lamWOs_b_corr, theta_fcon)
        else:
            self.set_params_full(lamOs_a_corr, lamOs_b_corr, lamWOs_a_corr, lamWOs_b_corr, theta_fcon)

        if self.num.mean_basis:
            self.set_params_mean_basis()

        # theta constraints function
        if theta_fcon is not None:
            if theta_init is None:
                raise ValueError('need to pass theta_init with theta_fcon')
            elif theta_init.shape != self.params.theta.val_shape:
                raise ValueError('theta_init shape should match theta.val_shape')
            else:
                if self.params.theta.prior.obeys_constraint(theta_init):
                    self.params.theta.val = theta_init
                else:
                    raise ValueError('theta_init does not obey theta_fcon constrant')

        # If cat ind, fix start values, bounds, prior variance for corresponding theta
        if not self.num.sim_only:
            for i in range(len(data.t_cat_ind)):
                if data.t_cat_ind[i] > 0:
                    self.params.theta.val[:, i] = 1
                    self.params.theta.prior.params[1][:, i] = np.inf
                    self.params.theta.prior.bounds[0][:, i] = -np.inf
                    self.params.theta.prior.bounds[1][:, i] = np.inf

        # allocate lists for pre-computed components
        self.num.SigWl = [None] * pu
        self.num.SigWi = [None] * pu
        if data.sep_design:
            self.num.V = [None] * pu
            self.num.Dki2 = [None] * pu

    def save_model_info(self, file_name=None, overwrite=False):
        """
        Saves some important model info needed to restore model state into new model.

        :param string/NoneType file_name: optional file name for saving model (with no extension); if None, uses default name.
        :param bool overwrite: okay to overwrite existing file? If False, and file_name exists, raises exception.
        :raises FileExistsError: if file_name already exists and overwrite=False
        """
        if file_name is None:
            file_name = 'saved_model_info'
        samples = self.get_samples(flat=False)
        param_info = {}
        for p in self.params:
            k = p.name
            if k != 'logPost':
                info = {}
                info['fixed'] = p.fixed
                info['prior_bounds'] = p.prior.bounds
                info['prior_fcon'] = p.prior.fcon
                info['prior_dist'] = p.prior.dist
                info['prior_params'] = p.prior.params
                info['mcmc_stepParam'] = p.mcmc.stepParam
                info['mcmc_stepType'] = p.mcmc.stepType
                param_info[k] = info
        save_dict = {'samples': samples, 'param_info':param_info}
        if os.path.exists(file_name) and overwrite is False:
            raise FileExistsError('File %s already exists; specify unique name or use overwrite=True to overwrite.' % file_name)
        with open('%s.pkl' % file_name, 'wb') as f:
            pickle.dump(save_dict, f)
        print('Model saved to %s.pkl' % file_name)

    def restore_model_info(self, file_name=None):
        """
        Restores model state from model info in file.

        :param string/NoneType file_name: file name for model info (with no extension); if None, uses default name.
        """
        print('WARNING: make sure this model was instantiated with the same input data as the model corresonding to this saved model info.')
        if file_name is None:
            file_name = 'saved_model_info'
        with open('%s.pkl' % file_name, 'rb') as f:
            save_dict = pickle.load(f)
        samples = save_dict['samples']
        param_info = save_dict['param_info']
        # put param info back
        for p in self.params:
            k = p.name
            if k != 'logPost':
                info = param_info[k]
                p.fixed = info['fixed']
                p.prior.bounds = info['prior_bounds']
                p.prior.fcon = info['prior_fcon']
                p.prior.dist = info['prior_dist']
                p.prior.params = info['prior_params']
                p.mcmc.stepParam = info['mcmc_stepParam']
                p.mcmc.stepType = info['mcmc_stepType']
        # Put samples and current values back
        for p in self.params:
            p.val = np.take(samples[p.name], -1, axis=0)
            draws = [s for s in samples[p.name]]
            p.mcmc.draws = draws
        # Call logLik to re-instantiate num info
        self.logLik()

    def log_prior(self):
        """
        Evaluates log prior.

        :return: float -- summed log prior over all parameters
        """
        if self.params is None:
            raise Exception("sepia model params must be set up first.")
        lp = 0
        for param in self.params.mcmcList:
            lp_tmp = param.prior.compute_log_prior()
            #print('Prior value for %s: %f'%(param.name, lp_tmp))
            lp += lp_tmp
        #self.num.logPrior=lp # commented out since we don't use this, and it could be problematic to set it each time called?
        return lp

    def logLik(self, cvar='all', cindex=None):
        """
        Compute model log lik with current values of variables.

        :param string cvar: name of variables changed since last call (controls recomputation of num components), or 'all'
        :param int/NoneType cindex: index of flattened cvar that has changed since last call (or None to indicate all indices should be considered changed)
        :return: scalar -- log lik value
        """
        L = compute_log_lik(self, cvar, cindex)
        return L

    def logPost(self, cvar='all', cindex=None):
        """
        Compute model log posterior with current values of variables.

        :param string cvar: name of variables changed since last call (controls recomputation of num components), or 'all'
        :param int/NoneType cindex: index of flattened cvar that has changed since last call (or None to indicate all indices should be considered changed)
        :return: scalar -- log posterior value
        """
        ll = self.logLik(cvar, cindex)
        lp = sum([prm.prior.compute_log_prior() for prm in self.params.mcmcList])
        return ll + lp

    def print_prior_info(self, pnames=None):
        """
        Print some information about the priors.

        :param list/NoneType pnames: list of parameter names to print information about; default is to print all.
        """

        if pnames is None:
            pnames = [p.name for p in self.params]
        for p in self.params:
            if p.name in pnames and p.name != 'logPost':
                print('%s prior distribution: %s' % (p.name, p.prior.dist))
                print('bounds: ')
                print(p.prior.bounds)
                for i in range(len(p.prior.params)):
                    print('prior param %d' % i)
                    print(p.prior.params[i])

    def print_priors_for_mcmc(self):
        """
        Print some information about the parameter priors that are defined for mcmc
        """
        for pm in self.params.mcmcList:
            print(pm.name,':')
            print('    Dist: ',pm.prior.dist)
            for ii in range(pm.val_shape[0]):
                for jj in range(pm.val_shape[1]):
                    print('   Bounds: ', '(', pm.prior.bounds[0][ii, jj], ',' , pm.prior.bounds[1][ii, jj], ')')
            for ii in range(pm.val_shape[0]):
                for jj in range(pm.val_shape[1]):
                    print('   Params: ','(',pm.prior.params[0][ii,jj], ',' , pm.prior.params[1][ii,jj], ')')

    def print_value_info(self, pnames=None):
        """
        Print some information about the parameter values. (Shows initial values if called before MCMC)

        :param list/NoneType pnames: list of parameter names to print information about; default is to print all.
        """

        if pnames is None:
            pnames = [p.name for p in self.params]
        for p in self.params:
            if p.name in pnames:
                print('%s shape (%d, %d):' % (p.name, p.val_shape[0], p.val_shape[1]))
                print('value:')
                print(p.val)
                print('is fixed?:')
                print(p.fixed)

    def print_mcmc_info(self, pnames=None):
        """
        Print some information about the MCMC setup.

        :param list/NoneType pnames: list of parameter names to print information about; default is to print all.
        """

        if pnames is None:
            pnames = [p.name for p in self.params]
        for p in self.params:
            if p.name in pnames:
                print('%s stepType: %s' % (p.name, p.mcmc.stepType))
                print('stepParam:')
                print(p.mcmc.stepParam)

    def do_mcmc(self, nsamp, prog=True, do_propMH=True, no_init=False, seed=None):
        """
        Run MCMC sampling on instantiated `SepiaModel` object.

        :param int nsamp: number of MCMC samples
        :param bool prog: show progress bar?
        :param bool do_propMH: use propMH sampling for variables with 'propMH' stepType?
        :param bool no_init: skip initialization? (only use if model has already been sampled from)

        .. note:: Calling multiple times on the same model appends samples to the existing samples.

        """
        if seed is not None:
            np.random.seed(seed)
        if self.num.auto_stepsize:
            do_propMH = False
        if not no_init:
            self.params.lp.set_val(self.logPost()) # Need to call once with cvar='all' (default) to initialize
        for _ in tqdm(range(nsamp), desc='MCMC sampling', mininterval=0.5, disable=not(prog)):
            self.mcmc_step(do_propMH)

    def clear_samples(self):
        """
        clear the mcmc samples in the model
        :return: no returned value
        """
        for p in self.params.mcmcList:
            p.mcmc.draws=[]
        self.params.lp.mcmc.draws=[]

    def get_num_samples(self):
        """
        Return the number of samples recorded in the model

        :return: number of samples
        """
        return self.params.lp.get_num_samples()
    def get_last_sample_ind(self):
        """
        Return index of the last sample = the number of samples - 1

        :return: number of samples - 1
        """
        return self.get_num_samples() - 1

    def add_samples(self,sdict=None):
        """
        Add samples from the samples_dict to the model.
        Will be particularly useful in parallel chains, to re-integrate samples.

        :param dict sdict: samples dict, as obtained from get_samples()
        :return: no return value; model object will be modified
        """
        if not isinstance(sdict,dict):
            raise TypeError('add_samples: requires a samples dict as input')

        if not ( set(sdict.keys()) - set(['theta_native']) ) == set([p.name for p in self.params.mcmcList]+['logPost']):
            print(( set(sdict.keys()) - set(['theta_native']) ))
            print(set([p.name for p in self.params.mcmcList]+['logPost']))
            raise ValueError('add_samples: samples dict must match fields in model')
        for k in sdict.keys():
            if len(sdict[k])!=len(sdict['logPost']):
                raise ValueError('add_samples: samples dictionary keys passed in have mismatched sample lengths')

        nsamp=len(sdict['logPost'])
        for pf in self.params.mcmcList:
            for ii in range(nsamp):
                pf.mcmc.draws.append(sdict[pf.name][ii,:].reshape(pf.val_shape, order='F'))
        for ii in range(nsamp):
            self.params.lp.mcmc.draws.append(sdict['logPost'][ii,:].reshape((1,1)))
        self.logLik()

    def set_model_to_sample(self,samp=None):
        """
        Add samples from the samples_dict to the model
        Will be particularly useful in parallel chains, to re-integrate samples.

        :param int samp: sample index to set model to. Default will be final stored sample.
        :return: no return value; model object will be modified
        """
        if samp is None:
            samp=self.get_last_sample_ind()
        elif isinstance(samp,int):
            if samp>self.get_last_sample_ind() or samp<0:
                raise ValueError('set_model_to_sample: samp parameter outside of valid sample indices')
        else:
            raise ValueError('set_model_to_sample: samp parameter must be an integer index')
        for p in self.params.mcmcList:
            p.val=p.mcmc.draws[samp]
        self.logLik()

    def get_samples(self, numsamples=None, nburn=0, sampleset=None, flat=True, includelogpost=True, effectivesamples=False, return_sampleset=False):
        """
        Extract MCMC samples into dictionary format. By default, all samples are returned, or samples can be
        subset using in various ways using the optional input arguments.

        :param int/NoneType numsamples: number of samples to include, evenly spaced from first to last
        :param int nburn: number of samples to discard at beginning of chain (default = 0)
        :param list/NoneType sampleset: list of indices of samples to include; if given, numsamples and nburn are ignored.
        :param bool flat: flatten the resulting arrays (for parameters stored as matrices)?
        :param bool includelogpost: include logPost values?
        :param bool effectivesamples: use effective sample size of thetas to subset samples? If True, numsamples and sampleset are ignored.
        :param bool return_sampleset: whether to return the sampleset indices (default False)
        :return: dict -- array of samples for each parameter, keyed by parameter name; optionally, also return sampleset (list)
        :raises: TypeError if no samples exist or nburn inconsistent with number of draws

        .. note:: If `theta` is in the model, will also add key `theta_native` with `theta` rescaled to original range.

        """
        total_samples = self.get_num_samples()
        if total_samples == 0:
            raise TypeError('No MCMC samples; call do_mcmc() first.')
        if sampleset is not None and numsamples is not None:
            print("Warning: set both numsamples and sampleset, defaulting to use sampleset.")
        if sampleset is None:
            if numsamples is None:
                numsamples = total_samples - nburn
            elif (numsamples + nburn) >= total_samples:
                print('numsamples + nburn larger than number of draws; truncating numsamples to number of draws - nburn (%d).' % (total_samples - nburn))
                numsamples = total_samples - nburn
            sampleset = [int(ii) for ii in np.linspace(nburn, total_samples-1, numsamples)]
        else:
            if max(sampleset) > total_samples:
                print('sampleset includes indices larger than number of draws; truncating to valid draws.')
                sampleset = [ii for ii in sampleset if ii < total_samples and ii >= 0]

        if effectivesamples:
            if self.num.sim_only:
                print('Emulator only - ignoring effectivesamples (needs thetas).')
            else:
                # get max theta ess
                for p in self.params.mcmcList:
                    if p.name == 'theta':
                        theta = p.mcmc_to_array(flat=flat).T
                ess_max = 0
                for i in range(theta.shape[0]):
                    # Skip if categorical
                    if self.data.t_cat_ind[i] > 0:
                        continue
                    tmp = self.ESS(theta[i,nburn:])
                    if tmp > ess_max: ess_max = tmp
                # set ss to grab ess number of samples
                sampleset = np.linspace(nburn, theta.shape[1], ess_max, dtype=int, endpoint=False)
                print("Max effective sample size over thetas: {}".format(ess_max))
                print("Total samples: {}".format(theta.shape[1]))

        samples = {p.name: p.mcmc_to_array(sampleset=sampleset, flat=flat, untransform_theta=False) for p in self.params.mcmcList}
        if includelogpost:
            samples['logPost'] = self.params.lp.mcmc_to_array(sampleset=sampleset, flat=flat, untransform_theta=False)
        # Add theta in native space as new key
        if 'theta' in samples.keys():
            samples['theta_native'] = self.params.theta.mcmc_to_array(sampleset=sampleset, flat=flat, untransform_theta=True)
        if return_sampleset:
            return samples, sampleset
        else:
            return samples

    def tune_step_sizes(self, n_burn, n_levels, prog=True, diagnostics=False, update_vals=True, verbose=True):
        """
        Auto-tune step size based on acceptance rate with YADAS approach.

        :param int n_burn: number of samples to draw for each proposed step size
        :param int n_levels: number of levels to propose for each step size
        :param bool prog: show progress bar?
        :param bool verbose: Print before and after info to console, default True
        :param bool diagnostics: return some information on acceptance rates used inside step size tuning?

        .. note:: Does not work for hierarchical or shared theta models.

        """
        if verbose:
            print('Starting tune_step_sizes...')
            print('Default step sizes:')
            for param in self.params.mcmcList:
                print('%s' % param.name)
                print(param.mcmc.stepParam)
        self.num.auto_stepsize = True
        import copy
        # Set up ranges, step sizes
        ex = np.linspace(-(n_levels - 1)/2, (n_levels - 1)/2, n_levels)
        step_sizes = {p.name: [] for p in self.params.mcmcList}
        for lev in range(n_levels):
            for mcmc_param in self.params.mcmcList:
                if ex[lev] <= 0:
                    base = 2.0
                else:
                    if mcmc_param.name == 'theta':
                        base = np.power(20., 2/(n_levels-1))
                    elif mcmc_param.name in ['betaU', 'betaV']:
                        base = np.power(10., 2 / (n_levels - 1))
                    elif mcmc_param.name in ['lamUz', 'lamOs']:
                        base = np.power(100., 2 / (n_levels - 1))
                    else:
                        base = 2.0
                step_sizes[mcmc_param.name].append(mcmc_param.mcmc.stepParam * np.power(base, ex[lev]))
        # Do sampling for each step size, collect acceptance rates/ step sizes into lists
        acc = {k.name: [] for k in self.params.mcmcList}
        mod_tmp = copy.deepcopy(self)
        # initialize model by calling log post
        mod_tmp.logPost()
        for _ in tqdm(range(n_burn), desc='Step size tuning', mininterval=0.5, disable=not(prog)):
            for i, lev in enumerate(range(n_levels)):
                for p in mod_tmp.params.mcmcList:
                    p.mcmc.stepParam = step_sizes[p.name][i].copy()
                mod_tmp.do_mcmc(1, do_propMH=False, no_init=True, prog=False)
        # Get acceptance
        for p in mod_tmp.params.mcmcList:
            acc[p.name].append(np.transpose(p.calc_accept().reshape((n_burn, n_levels, *p.val_shape)), [1,0,2,3]))
        # Compute GLM for each parameter
        logit = np.log(1 / (np.exp(1) - 1))
        for pi, p in enumerate(self.params.mcmcList):
            new_ss = np.zeros_like(p.mcmc.stepParam)
            p_acc = np.array(acc[p.name]) # (n_levels, n_burn, dim of param)
            p_ss = np.array(step_sizes[p.name])   # (n_levels, dim of param)
            for ind in range(int(np.prod(p.val_shape))):  # Loop over each one
                arr_ind = np.unravel_index(ind, p.val_shape, order='F')
                if p.fixed[arr_ind[0], arr_ind[1]]:
                    continue
                p_acc_ind = p_acc[0, :, :, arr_ind[0], arr_ind[1]].squeeze()
                p_ss_ind = p_ss[:, arr_ind[0], arr_ind[1]].squeeze()
                y = np.vstack([np.sum(p_acc_ind, axis=1), n_burn - np.sum(p_acc_ind, axis=1)]).T
                x = np.vstack([np.ones(n_levels), np.log(p_ss_ind)]).T
                glm_model = sm.GLM(y, x, family=sm.families.Binomial())
                try:
                    res = glm_model.fit()
                except Exception:
                    print('problem fitting glm on param %s index %d, reverting to default step size.' % (p.name, ind))
                    new_ss[arr_ind] = p.mcmc.stepParam[arr_ind]
                else:
                    coefs = res.params
                    opt_ss = np.exp((logit-coefs[0])/coefs[1])
                    new_ss[arr_ind] = opt_ss
            p.mcmc.stepParam = new_ss.copy()
            if update_vals:
                p.val = mod_tmp.params.mcmcList[pi].val.copy()
        if verbose:
            print('Done with tune_step_size.')
            print('Selected step sizes:')
            for param in self.params.mcmcList:
                print('%s' % param.name)
                print(param.mcmc.stepParam)
        if diagnostics:
            return step_sizes, acc, mod_tmp

    def set_param(self, pname, fix=None, val=None, bounds=None, prior_dist=None, prior_params=None):
        #
        # Set up parameter properties and.or priors
        #
        # :param pname: pparameter name (required)
        # :param fix: set value to this and fix (no sampling). Can be scalar or val_shape shaped
        # :param val: set value to this. Must be val_shape shaped
        # :param bounds: set bounds to this. Must be list of two val_shape shaped arrays
        # :param prior_dist: set distribution. String
        # :param prior_params: array of distribution parameters
        #
        if pname not in dir(self.params):
            raise ValueError('Model.set_param: parameter not in model list:'+pname )
        prm=getattr(self.params,pname)
        if bounds is not None:
            if len(bounds)!=2:
                raise ValueError('Model.set_param: bounds must be a list of length 2')
            if not ( ( isinstance(bounds[0],np.ndarray) and isinstance(bounds[1],np.ndarray) ) and \
                (bounds[0].shape==bounds[1].shape and bounds[0].shape==prm.val_shape) ) :
                raise ValueError('Model.set_param: bounds must conform to shape of parameter value: ' + 
                                     str(prm.val_shape) )
            prm.prior.bounds=bounds
        if prior_dist is not None:
            if not isinstance(prior_dist,str):
                raise ValueError('Model.set_param: dist is a name string')
            prm.prior.dist=prior_dist
        if prior_params is not None:
            for pp in prior_params:
                if not ( isinstance(pp,np.ndarray) and pp.shape==prm.val_shape ):
                    raise ValueError('Model.set_param: dist params must conform to shape of parameter value: ' + 
                                     str(prm.val_shape) )
            prm.prior.params=prior_params
        if val is not None:
            if not ( isinstance(val,np.ndarray) and fix.shape==prm.val_shape ):
                    raise ValueError('Model.set_param: val assignment must conform to shape of parameter value ' + 
                                     str(prm.val_shape) )
            prm.val=val
            prm.refVal=val
        if fix is not None:
            prm.fixed[:]=True
            prm.prior.bounds[0][:] = -np.inf
            prm.prior.bounds[1][:] = +np.inf
            if isinstance(fix,(float,int)):
                prm.val[:]=fix
            else:
                if not ( isinstance(fix,np.ndarray) and fix.shape==prm.val_shape ):
                        raise ValueError('Model.set_param: fix (value) must conform to shape of parameter value' + 
                                         str(prm.val_shape) )
                prm.val=fix
            prm.refVal=prm.val

    def set_params_mean_basis(self):
        # set up the gamma parameters for the mean basis multiplier
        # start the mean basis weights at the optimal linear fit for the simulations
        gvals=np.linalg.lstsq(self.data.sim_data.H,self.num.w,rcond=None)[0]
        self.params.gamma = SepiaParam(val=gvals, name='gamma', val_shape=(self.data.sim_data.H.shape[1],1), 
                                       dist='Uniform', params=[],bounds=[-np.inf,np.inf],
                                       mcmcStepParam=1, mcmcStepType='Uniform') #mcmcStepType='PropMH')
        self.params.mcmcList.append(self.params.gamma)

    def set_params_sim_only(self, lamWOs_a_corr=0, lamWOs_b_corr=0):
        #
        # Set up parameters and priors for simulation-only model.
        #
        # :param lamWOs_a_corr: prior correction
        # :param lamWOs_b_corr: prior correction
        #
        self.params = SepiaParamList()
        lamWOs_a = 5 + lamWOs_a_corr
        lamWOs_b = 5e-3 + lamWOs_b_corr
        lamWOs_init = np.max([100, lamWOs_a/lamWOs_b])
        if lamWOs_init >= 1e5:
            print('lamWOs initialized outside default bounds [60, 1e5]; setting initial value to 1e5 - 1.')
            lamWOs_init = 1e5-1
        self.params.betaU = SepiaParam(val=0.1, name='betaU', val_shape=(self.num.p + self.num.q, self.num.pu), dist='Beta',
                                       params=[1., 0.1], bounds=[0., np.inf], mcmcStepParam=0.1, mcmcStepType='BetaRho')
        self.params.lamUz = SepiaParam(val=1., name='lamUz', val_shape=(1, self.num.pu), dist='Gamma', params=[5., 5.],
                                       bounds=[0.3, np.inf], mcmcStepParam=5., mcmcStepType='PropMH')
        self.params.lamWOs = SepiaParam(val=lamWOs_init, name='lamWOs', val_shape=(1, 1), dist='Gamma',
                                        params=[lamWOs_a, lamWOs_b], bounds=[60., 1e5], mcmcStepParam=100., mcmcStepType='PropMH')
        self.params.lamWs = SepiaParam(val=1000., name='lamWs', val_shape=(1, self.num.pu), dist='Gamma', params=[3., 3e-3],
                                       bounds=[60., 1e5], mcmcStepParam=100., mcmcStepType='PropMH')
        self.params.mcmcList = [self.params.betaU, self.params.lamUz, self.params.lamWs, self.params.lamWOs]
        # Set up dummy parameter to hold logpost samples
        self.params.lp = SepiaParam(val=-np.inf, name='logPost', dist='Recorder', val_shape=(1, 1))

    def set_params_noD(self, lamOs_a_corr=0, lamOs_b_corr=0, lamWOs_a_corr=0, lamWOs_b_corr=0, theta_fcon=None):
        #
        # Set up parameters and priors for simulation and observed model with no discrepancy.
        #
        # :param lamOs_a_corr: prior correction
        # :param lamOs_b_corr: prior correction
        # :param lamWOs_a_corr: prior correction
        # :param lamWOs_b_corr: prior correction
        #
        self.set_params_sim_only(lamWOs_a_corr, lamWOs_b_corr)
        # Obs part
        lamOs_a = 1 + lamOs_a_corr
        lamOs_b = 1e-3 + lamOs_b_corr
        lamOs_init = np.max([20, lamOs_a/lamOs_b])
        theta_range = [self.data.obs_data.orig_t_min, self.data.obs_data.orig_t_max]
        if np.allclose(theta_range[0], theta_range[1]):
            theta_range = None
        self.params.theta = SepiaParam(val=0.5, name='theta', val_shape=(1, self.num.q), dist='Normal', params=[0.5, 10.],
                                       bounds=[0, 1], mcmcStepParam=0.2, mcmcStepType='Uniform', orig_range=theta_range, theta_fcon=theta_fcon)
        self.params.lamOs = SepiaParam(val=lamOs_init, name='lamOs', val_shape=(1, 1), dist='Gamma',
                                       params=[lamOs_a, lamOs_b], bounds=[0, np.inf], mcmcStepParam=lamOs_init/2, mcmcStepType='PropMH')
        self.params.mcmcList = [self.params.theta, self.params.betaU, self.params.lamUz, self.params.lamWs, self.params.lamWOs, self.params.lamOs]

    def set_params_full(self, lamOs_a_corr=0, lamOs_b_corr=0, lamWOs_a_corr=0, lamWOs_b_corr=0, theta_fcon=None):
        #
        # Set up parameters and priors for simulation and observed model with discrepancy.

        # :param lamOs_a_corr: prior correction
        # :param lamOs_b_corr: prior correction
        # :param lamWOs_a_corr: prior correction
        # :param lamWOs_b_corr: prior correction
        #
        self.set_params_sim_only(lamWOs_a_corr, lamWOs_b_corr)
        # Obs part
        lamOs_a = 1 + lamOs_a_corr
        lamOs_b = 1e-3 + lamOs_b_corr
        lamOs_init = np.max([20, lamOs_a/lamOs_b])
        theta_range = [self.data.obs_data.orig_t_min, self.data.obs_data.orig_t_max]
        if np.allclose(theta_range[0], theta_range[1]):
            theta_range = None
        self.params.theta = SepiaParam(val=0.5, name='theta', val_shape=(1, self.num.q), dist='Normal', params=[0.5, 10.],
                                       bounds=[0, 1], mcmcStepParam=0.2, mcmcStepType='Uniform', orig_range=theta_range, theta_fcon=theta_fcon)
        self.params.betaV = SepiaParam(val=0.1, name='betaV', val_shape=(self.num.p, self.num.lamVzGnum), dist='Beta', params=[1., 0.1],
                                       bounds=[0, np.inf], mcmcStepParam=0.1, mcmcStepType='BetaRho')
        self.params.lamVz = SepiaParam(val=20., name='lamVz', val_shape=(1, self.num.lamVzGnum), dist='Gamma', params=[1., 1e-3],
                                       bounds=[0., np.inf], mcmcStepParam=10., mcmcStepType='PropMH')
        self.params.lamOs = SepiaParam(val=lamOs_init, name='lamOs', val_shape=(1, 1), dist='Gamma',
                                       params=[lamOs_a, lamOs_b], bounds=[0, np.inf], mcmcStepParam=lamOs_init/2, mcmcStepType='PropMH')
        self.params.mcmcList = [self.params.theta, self.params.betaV, self.params.betaU, self.params.lamVz, self.params.lamUz,
                                    self.params.lamWs, self.params.lamWOs,  self.params.lamOs]

    def acf(self,chain,nlags,plot=True, alpha=.05, ESS=True, save=None):
        #
        # Compute autocorrelation function of mcmc chain
        #
        # Usually called by SepiaPlot.plot_acf(), not user
        #
        # compute autocorrelation
        return_dict = {}
        
        nchains, nobs = chain.shape
        autocorrs = []
        for i in range(nchains):
            chain1 = (chain[i,:] - np.mean(chain[i,:])) / (np.std(chain[i,:]) * nobs)
            chain2 = (chain[i,:] - np.mean(chain[i,:])) / np.std(chain[i,:])
            autocorr = np.correlate(chain1,chain2,mode='full')
            autocorr = autocorr[autocorr.size//2:]
            autocorrs.append(autocorr[0:nlags+1])
        return_dict['acf']=autocorr
        sigline = stats.norm.ppf(1 - alpha / 2.) / np.sqrt(nobs)
        return_dict['sigline']=sigline
        # compute ESS and output to console
        if ESS:
            ess = []
            for i in range(nchains):
                ess.append(self.ESS(chain[i,:]))
            return_dict['ess']=ess
        # plot
        if plot:
            fig, ax = plt.subplots()
            lags = np.linspace(0,nlags,nlags+1,dtype=int,endpoint=True)
            for i in range(len(autocorrs)):
                ax.plot(lags,autocorrs[i],'-o',fillstyle='none',label='theta {}'.format(i+1))
            ax.set_xticks(np.linspace(0,nlags,10,dtype=int,endpoint=True))
            ax.set_yticks(np.linspace(0,1,11,endpoint=True))
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')

            # significance lines
            ax.axhline(-sigline,linestyle='--',c='b'); ax.axhline(sigline,linestyle='--',c='b')

            # axis limits

            ymin = min(np.min(-sigline),min([np.min(ac) for ac in autocorrs]))
            ymax = max(np.max(sigline),max([np.max(ac) for ac in autocorrs]))
            ax.set_ylim([min(-.1,1.1*ymin),1.1*ymax])

            if nchains > 1: plt.legend()
            text = []
            text.append(ax.text(ax.get_xlim()[0],1.225,s='Effective Sample Size: {}'.format(ess),fontsize=16))
            text.append(ax.text(ax.get_xlim()[0],1.125,s='Number of Samples:    {}'.format([nobs]*nchains),fontsize=16))
            return_dict['figure']=fig

        return return_dict
    
    def ESS(self,x):
        #
        # Compute the effective sample size of estimand of interest. Vectorised implementation.
        #
        # Not called by user
        #
        if np.ndim(x) == 1:
            x = x.reshape(1,-1)
            
        m_chains, n_iters = x.shape

        variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

        post_var = self.marg_post_var(x)

        if np.isclose(post_var, 0):
            raise ZeroDivisionError('One of the chains has zero variation across samples; check chains/model.')

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

    def marg_post_var(self,x):
        #
        # Estimate the marginal posterior variance. Vectorised implementation.
        #
        # Usually not called by user
        #
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

    def mcmc_step(self, do_propMH=True):
        # Does a single MCMC step; not typically called by users
        # Loop over parameters
        for prm in self.params.mcmcList:
            # Loop over indices within parameter
            for ind in range(int(np.prod(prm.val_shape))):
                arr_ind = np.unravel_index(ind, prm.val_shape, order='F')
                Accept = False
                # Make reference copies in case we reject
                prm.refVal = prm.val.copy()
                self.refNum = self.num.ref_copy(prm.name)
                # Draw candidate
                #  check for categorical theta
                if prm.name == 'theta' and self.data.t_cat_ind[ind] > 0:
                    # Get possible category values, excluding current value
                    cat_vals = [i for i in range(1, self.data.t_cat_ind[ind]+1) if i != prm.val[arr_ind]]
                    # Choose one
                    cand = np.random.choice(cat_vals, 1)
                else:
                    cand = prm.mcmc.draw_candidate(arr_ind, do_propMH)
                # Set value to candidate
                prm.val[arr_ind] = cand
                # print(prm.mcmc.aCorr)
                if not prm.fixed[arr_ind]: # If not supposed to be fixed, check for acceptance
                    if prm.mcmc.aCorr and prm.prior.is_in_bounds() and prm.prior.obeys_constraint():
                        # This logPost uses val which has the candidate modification
                        clp = self.logPost(cvar=prm.name, cindex=ind)
                        if np.log(np.random.uniform()) < (clp - self.params.lp.val + np.log(prm.mcmc.aCorr)):
                            Accept = True
                # print(f'{prm.val:6.2f} : {prm.refVal:6.2f} | {lp:6.2f} : {clp:6.2f} | {np.log(prm.mcmc.aCorr):6.4f} = {Accept}')
                if Accept:
                    # Accept basically does nothing, the now modified val stays there for the next step
                    prm.mcmc.accept()
                    self.params.lp.set_val(clp)
                    # print(f'{clp-lp}')
                else:
                    # Reject sets val back to refVal that was stored at the top
                    prm.mcmc.reject(ind, self)
            prm.mcmc.record()
        self.params.lp.mcmc.record()


# container to group a number of misc. model pre-calculated info
class ModelContainer():
    """
    Internally used to contain numeric elements computed for the model/likelihood evaluations.

    :var bool auto_stepsize: was step size tuning used?
    :var Dki2: for separable input calculations
    :var np.ndarray LamSim: simulation noise precision diagonal entries, length pu
    :var list lamVzGroup: list size pv giving group assignments for each discrepancy basis element; default is [0]*pv
    :var int lamVzGnum: number of groups for discrepancy basis elements (separate lamVz/betaV per group); default is 1
    :var int m: number of simulation data observations
    :var int n: number of observed data observations
    :var int p: number of controllable inputs (known in principle for both sim and obs data)
    :var int pu: number of PC components
    :var int pv: number of discrepancy basis components
    :var int q: number of non-controllable inputs (known in principle only for sim data)
    :var bool scalar_out: is y a scalar output?
    :var np.ndarray SigObs: observation noise variance matrix, size (pu+pv, pu+pv)
    :var numpy.ndarray SigV: part of likelihood calculation
    :var numpy.ndarray SigU: part of likelihood calculation
    :var numpy.ndarray SigWl: part of likelihood calculation
    :var numpy.ndarray SigWi: part of likelihood calculation
    :var numpy.ndarray SigUW: part of likelihood calculation
    :var bool sim_only: is it simulation-only?
    :var np.ndarray u: projection of observations onto K, size (pu, n)
    :var np.ndarray v: projection of observations onto D, size (pv, n)
    :var list V: for separable design calculations
    :var np.ndarray vu: concatenation of u and v reordered for model usage, size (pu+pv, n)
    :var np.ndarray vuw: concatenation/reordering/reshaping of v, u, and w, size (pu*m + pu*n + pv*n)
    :var np.ndarray w: projection of simulations onto K, size (pu*m, 1)
    :var np.ndarray x: unit hypercube inputs corresponding to observed data
    :var sepia.SepiaDistCov x0Dist: sepia.SepiaDistCov for data.x (obs controllable inputs)
    :var sepia.SepiaDistCov xzDist: sepia.SepiaDistCov for x/z (sim and obs controllable inputs)
    :var zt: unit hypercube inputs corresponding to simulated data
    :var sepia.SepiaDistCov ztDist: sepia.SepiaDistCov for z/t (sim controllable/noncontrollable inputs)
    :var sepia.SepiaDistCov ztSepDist: for separable designs

    """

    def __init__(self):
        self.scalar_out = self.sim_only = self.mean_basis = None  # Useful flags
        self.x = self.theta = self.zt = None  # GP inputs
        self.u = self.v = self.w = None  # GP outputs
        self.LamSim = self.LamObs = self.SigObs = None  # Precomputed cov stuff
        self.n = self.m = self.pu = self.pv = self.p = self.q = None  # Dimensions
        self.lamVzGroup = None
        self.lamVzGnum = 1
        #self.x0Dist = self.xzDist = self.xthetaDist = self.ztDist = None  # distances for covariance
        self.SigV = self.SigU = self.SigWl = self.SigWi = self.SigUW = None
        self.V = self.Dki2 = None # this will be used if separable design
        self.auto_stepsize = False

    def ref_copy(self, pname):
        # Makes a copy only of stuff that might change in mcmc depending on parameter name pname
        ref = {}
        ref['logLik'] = self.logLik
        if pname == 'theta':
            ref['xDist'] = np.copy(self.xDist.sqdist)
            ref['xzDist'] = np.copy(self.xzDist.sqdist)
            ref['SigU'] = self.SigU.copy()
            ref['SigUW'] = self.SigUW.copy()
        elif pname == 'betaU':
            ref['xDist'] = np.copy(self.xDist.sqdist)
            ref['SigWl'] = None if self.SigWl is None else self.SigWl.copy()
            ref['V'] = None if self.V is None else self.V.copy()
            ref['Dki2'] = None if self.Dki2 is None else self.Dki2.copy()
            ref['SigWi'] = None if self.SigWi is None else self.SigWi.copy()
            ref['SigU'] = None if self.SigU is None else self.SigU.copy()
            ref['SigUW'] = None if self.SigUW is None else self.SigUW.copy()
        elif pname == 'lamUz':
            ref['SigU'] = None if self.SigU is None else self.SigU.copy()
            ref['SigUW'] = None if self.SigUW is None else self.SigUW.copy()
            ref['SigWl'] = None if self.SigWl is None else self.SigWl.copy()
            ref['V'] = None if self.V is None else self.V.copy()
            ref['Dki2'] = None if self.Dki2 is None else self.Dki2.copy()
            ref['SigWi'] = None if self.SigWi is None else self.SigWi.copy()
        elif pname == 'lamWs':
            ref['SigU'] = None if self.SigU is None else self.SigU.copy()
            ref['SigWl'] = None if self.SigWl is None else self.SigWl.copy()
            ref['V'] = None if self.V is None else self.V.copy()
            ref['Dki2'] = None if self.Dki2 is None else self.Dki2.copy()
            ref['SigWi'] = None if self.SigWi is None else self.SigWi.copy()
        elif pname in ['betaV', 'lamVz']:
            ref['SigV'] = None if self.SigV is None else self.SigV.copy()
        elif pname == 'lamWOs':
            ref['SigWl'] = None if self.SigWl is None else self.SigWl.copy()
            ref['V'] = None if self.V is None else self.V.copy()
            ref['Dki2'] = None if self.Dki2 is None else self.Dki2.copy()
            ref['SigWi'] = None if self.SigWi is None else self.SigWi.copy()
        return ref

    def restore_ref(self, ref):
        # put items from reference dict back into ModelContainer object
        for attr_name in dir(self):
            if not attr_name.startswith("__") and attr_name in ref.keys():
                if attr_name == 'xDist':
                    self.xDist.sqdist = ref[attr_name]
                elif attr_name == 'xzDist':
                    self.xzDist.sqdist = ref[attr_name]
                else:
                    self.__dict__[attr_name] = ref[attr_name]


class SepiaModel(SepiaModelBase):
    def __init__(self, data, lamVzGroup=None, theta_fcon=None, theta_init=None, LamSim=None):
        if data.use_simulator:
            self.data = data
        else:
            super().__init__(
                data, lamVzGroup=lamVzGroup, theta_fcon=theta_fcon, theta_init=theta_init,
                LamSim=LamSim
            )

    def print_priors_for_mcmc(self, *args, **kwargs):
        if self.data.use_simulator:
            NotImplementedError()
        else:
            return super().print_prior_info(*args, **kwargs)

    def tune_step_sizes(self, *args, **kwargs):
        if self.data.use_simulator:
            NotImplementedError()
        else:
            return super().tune_step_sizes(*args, **kwargs)

    def do_mcmc(self, *args, **kwargs):
        if self.data.use_simulator:
            model = emufree_calib_model.NoEmuCalibModel()

            priors = kwargs.pop("priors", None)

            model_data = emufree_calib_model.make_model_data(
                y=np.concatenate(self.data.y_obs),
                xs=self.data.x_obs,
                eta=self.data.eta,
                W=self.data.Sigy,
                theta_dim=self.data.theta_dim,
                D=self.data.D_obs,
                num_basis=self.data.num_basis,
                priors=priors,
            )

            return emufree_calib_model.do_mcmc(
                model=model, data=model_data,
                num_samples=args[0], burn=kwargs.pop("burn", 0),
                thinning=kwargs.pop("thinning", 1),
                window=kwargs.pop("window", None),
                seed=kwargs.pop("seed", None),
                init_state=kwargs.pop("init_state", None),
            )
        else:
            return super().do_mcmc(*args, **kwargs)
