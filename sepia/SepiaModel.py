

import numpy as np
from sepia.SepiaParam import SepiaParam, SepiaParamList
from sepia.SepiaLogLik import compute_log_lik
import statsmodels.api as sm
from tqdm import tqdm

# container to group a number of misc. model pre-calculated info
class ModelContainer():
    """
    Internally used to contain numeric elements computed for the model/likelihood evaluations.

    :var scalar_out: boolean -- is y a scalar or multivariate
    :var sim_only: boolean -- is it simulation-only or is there obs data
    :var x:
    :var theta:
    :var zt:
    :var LamSim:
    :var LamObs:
    :var SigObs:
    :var n:
    :var m:
    :var pu:
    :var pv:
    :var p:
    :var q:
    :var lamVzGroup:
    :var SigV:
    :var SigU:
    :var SigWl:
    :var SigWi:
    :var SigUW:
    :var auto_stepsize:
    """

    def __init__(self):
        self.scalar_out = self.sim_only = None  # Useful flags
        self.x = self.theta = self.zt = None  # GP inputs
        self.u = self.v = self.w = None  # GP outputs
        self.LamSim = self.LamObs = self.SigObs = None  # Precomputed cov stuff
        self.n = self.m = self.pu = self.pv = self.p = self.q = None  # Dimensions
        self.lamVzGroup = None
        #self.x0Dist = self.xzDist = self.xthetaDist = self.ztDist = None  # distances for covariance
        self.SigV = self.SigU = self.SigWl = self.SigWi = self.SigUW = None
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
            ref['SigWi'] = None if self.SigWi is None else self.SigWi.copy()
            ref['SigU'] = None if self.SigU is None else self.SigU.copy()
            ref['SigUW'] = None if self.SigUW is None else self.SigUW.copy()
        elif pname == 'lamUz':
            ref['SigU'] = None if self.SigU is None else self.SigU.copy()
            ref['SigUW'] = None if self.SigUW is None else self.SigUW.copy()
            ref['SigWl'] = None if self.SigWl is None else self.SigWl.copy()
            ref['SigWi'] = None if self.SigWi is None else self.SigWi.copy()
        elif pname == 'lamWs':
            ref['SigU'] = None if self.SigU is None else self.SigU.copy()
            ref['SigWl'] = None if self.SigWl is None else self.SigWl.copy()
            ref['SigWi'] = None if self.SigWi is None else self.SigWi.copy()
        elif pname in ['betaV', 'lamVz']:
            ref['SigV'] = None if self.SigV is None else self.SigV.copy()
        elif pname == 'lamWOs':
            ref['SigWl'] = None if self.SigWl is None else self.SigWl.copy()
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


class SepiaModel:
    """
    Sepia model class contains data, SepiaParam objects, and precomputed elements for the likelihood.

    :var data: SepiaData object
    :var num: ModelContainer for computed numerical elements for use in evaluating likelihood/posterior
    :var params: SepiaParamList containing all SepiaParam objects for the model and mcmcList (references to params involved in MCMC)
    :var verbose: boolean -- whether to print verbose output for this model
    """

    def __init__(self):
        self.verbose = False
        self.data = None            # SepiaData obj for use later
        self.num = ModelContainer() # num for numeric state
        self.params = None

    #def __str__(self):
    #    if self.data is None:
    #        print('SepiaModel is not set up; call setup_model(SepiaData_object)')
    #    else:
    #        print(self.data)

    def get_samples(self, nburn=0, sampleset=False, numsamples=False, flat=True, includelogpost=True, untransform_theta=False):
        """
        Extract MCMC samples into dictionary format. By default, all samples are returned, or samples can be
        subset using nburn/sampleset/numsamples. Provide either sampleset or numsamples, or neither.

        :param nburn: int -- number of samples to discard at beginning of chain
        :param sampleset: list -- indices of samples to include
        :param numsamples: int -- return num_samples of samples, evenly spaced from first to last
        :param flat: bool -- whether to flatten the resulting arrays (for parameters stored as matrices)
        :param includelogpost: bool -- whether to also get samples of log posterior
        :param untransform_theta: bool -- whether or not to untransform theta to original scale (if theta in model)
        :return: dict -- array of samples for each parameter, keyed by parameter name
        :raises: TypeError if no samples exist or nburn inconsistent with number of draws
        """
        total_samples = self.params.lp.get_num_samples()
        if total_samples == 0:
            raise TypeError('No MCMC samples; call do_mcmc() first.')

        if numsamples and sampleset:
            print("warning: set both numsamples and sampleset, defaulting to use sampleset.")

        # By default, use all samples
        ss = np.arange(total_samples)

        # Parse sampleset/numsamples
        if numsamples is not False:
            if numsamples >= total_samples:
                print('numsamples larger than number of draws; truncating to number of draws (%d).' % total_samples)
            else:
                ss = [int(ii) for ii in np.linspace(0, total_samples-1, numsamples)]
        if sampleset is not False:
            if max(sampleset) > total_samples:
                print('sampleset includes indices larger than number of draws; truncating to valid draws.')
            ss = [ii for ii in sampleset if ii < total_samples and ii >= 0]

        plist = self.params.mcmcList
        if includelogpost: plist.append(self.params.lp)
        samples = {p.name: p.mcmc_to_array(trim=nburn, sampleset=ss, flat=flat, untransform_theta=untransform_theta)
                   for p in plist}
        return samples

    def set_params_sim_only(self, lamWOs_a_corr=0, lamWOs_b_corr=0):
        """
        Set up parameters and priors for simulation-only model.

        :param lamWOs_a_corr: prior correction
        :param lamWOs_b_corr: prior correction
        """
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

    def set_params_noD(self, lamOs_a_corr=0, lamOs_b_corr=0, lamWOs_a_corr=0, lamWOs_b_corr=0):
        """
        Set up parameters and priors for simulation and observed model with no discrepancy.

        :param lamOs_a_corr: prior correction
        :param lamOs_b_corr: prior correction
        :param lamWOs_a_corr: prior correction
        :param lamWOs_b_corr: prior correction
        """
        self.set_params_sim_only(lamWOs_a_corr, lamWOs_b_corr)
        # Obs part
        lamOs_a = 1 + lamOs_a_corr
        lamOs_b = 1e-3 + lamOs_b_corr
        lamOs_init = np.max([20, lamOs_a/lamOs_b])
        theta_range = [self.data.obs_data.orig_t_min, self.data.obs_data.orig_t_max]
        if np.allclose(theta_range[0], theta_range[1]):
            theta_range = None
        self.params.theta = SepiaParam(val=0.5, name='theta', val_shape=(1, self.num.q), dist='Normal', params=[0.5, 10.],
                                       bounds=[0, 1], mcmcStepParam=0.2, mcmcStepType='Uniform', orig_range=theta_range)
        self.params.lamOs = SepiaParam(val=lamOs_init, name='lamOs', val_shape=(1, 1), dist='Gamma',
                                       params=[lamOs_a, lamOs_b], bounds=[0, np.inf], mcmcStepParam=lamOs_init/2, mcmcStepType='PropMH')
        self.params.mcmcList = [self.params.theta, self.params.betaU, self.params.lamUz, self.params.lamWs, self.params.lamWOs, self.params.lamOs]

    def set_params_full(self, lamOs_a_corr=0, lamOs_b_corr=0, lamWOs_a_corr=0, lamWOs_b_corr=0):
        """
        Set up parameters and priors for simulation and observed model with discrepancy.

        :param lamOs_a_corr: prior correction
        :param lamOs_b_corr: prior correction
        :param lamWOs_a_corr: prior correction
        :param lamWOs_b_corr: prior correction
        """
        self.set_params_sim_only(lamWOs_a_corr, lamWOs_b_corr)
        # Obs part
        lamOs_a = 1 + lamOs_a_corr
        lamOs_b = 1e-3 + lamOs_b_corr
        lamOs_init = np.max([20, lamOs_a/lamOs_b])
        theta_range = [self.data.obs_data.orig_t_min, self.data.obs_data.orig_t_max]
        if np.allclose(theta_range[0], theta_range[1]):
            theta_range = None
        self.params.theta = SepiaParam(val=0.5, name='theta', val_shape=(1, self.num.q), dist='Normal', params=[0.5, 10.],
                                       bounds=[0, 1], mcmcStepParam=0.2, mcmcStepType='Uniform', orig_range=theta_range)
        self.params.betaV = SepiaParam(val=0.1, name='betaV', val_shape=(self.num.p, self.num.lamVzGnum), dist='Beta', params=[1., 0.1],
                                       bounds=[0, np.inf], mcmcStepParam=0.1, mcmcStepType='BetaRho')
        self.params.lamVz = SepiaParam(val=20., name='lamVz', val_shape=(1, 1), dist='Gamma', params=[1., 1e-3],
                                       bounds=[0., np.inf], mcmcStepParam=10., mcmcStepType='PropMH')
        self.params.lamOs = SepiaParam(val=lamOs_init, name='lamOs', val_shape=(1, 1), dist='Gamma',
                                       params=[lamOs_a, lamOs_b], bounds=[0, np.inf], mcmcStepParam=lamOs_init/2, mcmcStepType='PropMH')
        self.params.mcmcList = [self.params.theta, self.params.betaV, self.params.betaU, self.params.lamVz, self.params.lamUz,
                                    self.params.lamWs, self.params.lamWOs,  self.params.lamOs]

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

    def do_mcmc(self, nsamp, prog=True, do_propMH=True, no_init=False, seed=None):
        """
        Run MCMC sampling on initialized SepiaModel object.

        Note that calling again appends samples to existing samples, so you can run in chunks.

        :param nsamp: float -- number of MCMC samples
        :param prog: bool -- whether to show progress bar
        :param do_propMH: bool -- whether to use propMH sampling for variables with that step type
        :param no_init: bool -- skip initialization (if model has already been sampled; need to initialize on first call)
        """
        if seed is not None:
            np.random.seed(seed)
        if self.num.auto_stepsize:
            do_propMH = False
        if not no_init:
            self.params.lp.set_val(self.logPost()) # Need to call once with cvar='all' (default) to initialize
        #self.params.lp.mcmc.record(self.params.lp.val)
        for _ in tqdm(range(nsamp), desc='MCMC sampling', mininterval=0.5, disable=not(prog)):
            self.mcmc_step(do_propMH)

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
                cand = prm.mcmc.draw_candidate(arr_ind, do_propMH)
                # Set value to candidate
                prm.val[arr_ind] = cand
                # print(prm.mcmc.aCorr)
                if not prm.fixed[arr_ind]:  # If not supposed to be fixed, check for acceptance
                    if prm.mcmc.aCorr and prm.prior.is_in_bounds(prm.val[arr_ind]):
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

    def logLik(self, cvar='all', cindex=None):
        """
        Compute model log lik with current values of variables.

        :param cvar: string -- name of variables changed since last call (controls recomputation of num components), or 'all'
        :param cindex: int -- index of flattened cvar that has changed since last call (or None)
        :return: scalar -- log lik value
        """
        L = compute_log_lik(self, cvar, cindex)
        return L
        
    def logPost(self, cvar='all', cindex=None):
        """
        Compute model log posterior with current values of variables.

        :param cvar: string -- name of variables changed since last call (controls recomputation of num components), or 'all'
        :param cindex: int -- index of flattened cvar that has changed since last call (or None)
        :return: scalar -- log posterior value
        """
        ll = self.logLik(cvar, cindex)
        lp = sum([prm.prior.compute_log_prior() for prm in self.params.mcmcList])
        return ll + lp

    #TODO: does not handle hierModels/tiedThetaModels, passes a sim only univ test pretty well (lamWOs slightly different ss)
    #TODO: set start value to maximum posterior instead of last sample?
    def tune_step_sizes(self, n_burn, n_levels, prog=True, diagnostics=False):
        """ Atuo-tune step size based on acceptance rate with YADAS approach.

        :param n_burn: int -- number of samples for each step size
        :param n_levels: int -- number of levels for step size
        :param prog: bool -- whether to show progress bar
        :param diagnostics: bool -- whether to return some information on acceptance rates used inside step size tuning
        """
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
                p_acc_ind = p_acc[0, :, :, arr_ind[0], arr_ind[1]].squeeze()
                p_ss_ind = p_ss[:, arr_ind[0], arr_ind[1]].squeeze()
                y = np.vstack([np.sum(p_acc_ind, axis=1), n_burn - np.sum(p_acc_ind, axis=1)]).T
                x = np.vstack([np.ones(n_levels), np.log(p_ss_ind)]).T
                glm_model = sm.GLM(y, x, family=sm.families.Binomial())
                res = glm_model.fit()
                coefs = res.params
                opt_ss = np.exp((logit-coefs[0])/coefs[1])
                new_ss[arr_ind] = opt_ss
            p.mcmc.stepParam = new_ss.copy()
            p.val = mod_tmp.params.mcmcList[pi].val.copy()
        print('Done with tune_step_size.')
        print('Selected step sizes:')
        for param in self.params.mcmcList:
            print('%s' % param.name)
            print(param.mcmc.stepParam)
        if diagnostics:
            return step_sizes, acc, mod_tmp


