"""
@author: nklein

Class for holding multiple models with hierarchical distribution on some thetas

"""

import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import copy

from sepia.SepiaParam import SepiaParam, SepiaParamList
from sepia.SepiaLogLik import compute_log_lik
from sepia.SepiaModel import SepiaModel


class SepiaHierarchicalThetaModels:
    """
    Container for multiple models with hierarchical Normal model on selected thetas.

    :param model_list: list of instantiated SepiaModel objects
    :param hier_theta_inds: nparray -- (n_hier_theta, n_models) where each row corresponds to one group of hierarchically
                            modeled thetas, and each column gives the index of the theta within a particular model, with
                            -1 used to indicate no theta from a particular model is part of the hierarchical group.

    """

    def __init__(self, model_list=None, hier_theta_inds=None):
        self.model_list = model_list            # List of instantiated SepiaModel objects
        self.hier_theta_inds = hier_theta_inds  # Matrix (n_hier_theta, n_models) indicating hier indices, -1 means not in a model
        # Example: hier_theta_inds = np.array([(1, 1, 1), (2, -1, 4)) for 3 models, theta index 1 hierarchical across all models,
        #          theta indices 2/4 hierarchical across models 1 and 3 but no corresponding theta in model 2.
        if not hier_theta_inds.shape[1] == len(model_list):
            raise Exception('Number of models does not match provided hierarchical theta lists')
        self.setup_hier_theta()

    def setup_hier_theta(self):
        # sets up bookkeeping to make mcmc loop simpler
        n_hier, n_models = self.hier_theta_inds.shape
        self.n_hier = n_hier
        # Get indices of models for which each parameter is in a hierarchical distn
        to_update = np.zeros_like(self.hier_theta_inds)
        for i in range(n_hier):
            ti_row = self.hier_theta_inds[i, :]
            for j in range(n_models):
                if ti_row[j] > -1:
                    to_update[i, j] = 1
        self.to_update = to_update
        # Set up params/priors for hierarchical distns (TODO need to check defaults; need better UI to let user set up?)
        hier_mu = []
        hier_lambda = []
        hier_delta = []
        for i in range(n_hier):
            # Create sepParams for hierarchical parameters
            # mean
            hier_mu.append(SepiaParam(0.5, 'mu%d' % i, val_shape=(1, 1), dist='Normal', params=[0.5, 10.], bounds=[0, 1],
                                      mcmcStepType='Uniform', mcmcStepParam=0.2))
            # Precision
            hier_lambda.append(SepiaParam(5, 'lam%d' % i, val_shape=(1, 1), dist='Gamma', params=[1., 1e-3], bounds=[0, np.inf],
                                          mcmcStepType='PropMH', mcmcStepParam=100))
            # Lockstep mean update width
            hier_delta.append(SepiaParam(0, 'delta%d' % i, val_shape=(1, 1), dist='Uniform', mcmcStepType='Uniform', mcmcStepParam=0.2))
        self.hier_mu = hier_mu
        self.hier_lambda = hier_lambda
        self.hier_delta = hier_delta
        # Update relevant theta priors
        for i in range(n_hier):
            r = self.hier_theta_inds[i, :]
            for j in range(r.shape[0]):
                if r[j] > -1:
                    self.model_list[j].params.theta.prior.params[0][0, r[j]] = hier_mu[i].val.copy()
                    self.model_list[j].params.theta.prior.params[1][0, r[j]] = np.sqrt(1./hier_lambda[i].val.copy())

    def log_prior(self):
        """
        Evalutes log prior across all models (does not include hierarchical priors)
        :return:
        """
        lp_list = []
        for model in self.model_list:
            lp = model.log_prior()
            #model.num.logPrior = lp # TODO maybe should not set this here
            lp_list.append(lp)
        return lp_list

    def logLik(self, cvar='all', cindex=None):
        L_list = []
        for model in self.model_list:
            L = compute_log_lik(model, cvar, cindex) # note sets num.logLik
            L_list.append(L)
        return L_list
        
    def logPost(self, cvar='all', cindex=None):
        ll_list = self.logLik(cvar, cindex)
        lp_list = self.log_prior()
        # Evaluate hierarchical priors
        mu_prior = 0
        for mu_i in self.hier_mu:
            mu_prior += mu_i.prior.compute_log_prior()
        lam_prior = 0
        for lam_i in self.hier_lambda:
            lam_prior += lam_i.prior.compute_log_prior()
        #print(f'{lp}') 
        return [ll_list[i] + lp_list[i] for i in range(len(ll_list))] + [mu_prior, lam_prior]

    def do_mcmc(self, nsamp, do_propMH=True, prog=True, do_lockstep=True):
        # Initialize all models
        for model in self.model_list:
            model.params.lp.set_val(model.logPost())

        # MCMC loop
        for _ in tqdm(range(nsamp), desc='MCMC sampling', mininterval=0.5, disable=not(prog)):

            ### The usual: sample all non-hierarchical model parameters
            for model in self.model_list:
                model.mcmc_step(do_propMH)

            ### Sampling hierarchical parameters
            for hi in range(self.n_hier):
                theta_inds = self.hier_theta_inds[hi, :]
                n_models = theta_inds.shape[0]
                mu_param = self.hier_mu[hi]
                lam_param = self.hier_lambda[hi]
                self.mcmc_step_mulam(theta_inds, mu_param)
                self.mcmc_step_mulam(theta_inds, lam_param)

                if do_lockstep:
                    ## Lockstep update
                    arr_ind = np.unravel_index(0, self.hier_delta[hi].val_shape, order='F')
                    delta_cand = self.hier_delta[hi].mcmc.draw_candidate(arr_ind, False)
                    mu_cand = delta_cand + mu_param.val[0, 0].copy()
                    # check in bounds for mu; check in bounds for theta (TODO other constraints...)
                    inb = mu_param.prior.is_in_bounds(mu_cand)
                    if inb:
                        for mi in range(n_models):
                            if theta_inds[mi] > -1:
                                theta_param = self.model_list[mi].params.theta
                                inb = inb and (mu_cand > theta_param.prior.bounds[0] and mu_cand < theta_param.prior.bounds[1])
                                # Check if new thetas will be in bounds, too, and don't continue if not
                                tv = theta_param.val[0, theta_inds[mi]]
                                inb = inb and (tv + delta_cand > theta_param.prior.bounds[0] and tv + delta_cand < theta_param.prior.bounds[1])
                    # If in bounds, evaluate draw to decide whether or not to accept
                    if inb:
                        # Store old/current log prior and prior params for thetas in case reject,
                        # put modified mu cand into priors and thetas
                        old_prior = []
                        old_prior_params = []
                        old_lik = []
                        for mi in range(n_models):
                            if theta_inds[mi] > -1:
                                old_prior_params.append(copy.deepcopy(self.model_list[mi].params.theta.prior.params))
                                old_lik.append(self.model_list[mi].logLik('theta'))
                                #old_prior.append(self.model_list[mi].log_prior())
                                self.model_list[mi].params.theta.prior.params[0][0, theta_inds[mi]] = mu_cand
                                self.model_list[mi].params.theta.refVal[0, theta_inds[mi]] = self.model_list[mi].params.theta.val[0, theta_inds[mi]].copy()
                                self.model_list[mi].params.theta.val[0, theta_inds[mi]] = self.model_list[mi].params.theta.val[0, theta_inds[mi]].copy() + delta_cand
                        old_prior.append(self.hier_mu[hi].prior.compute_log_prior())
                        # Put mu candidate into place
                        mu_param.refVal = mu_param.val.copy()
                        mu_param.val[0, 0] = mu_cand
                        # Compute new prior/lik
                        new_prior = []
                        new_lik = []
                        for mi in range(n_models):
                            if theta_inds[mi] > -1:
                                #new_prior.append(self.model_list[mi].log_prior())
                                new_lik.append(self.model_list[mi].logLik('theta'))
                        new_prior.append(self.hier_mu[hi].prior.compute_log_prior())
                        # Calculate acceptance
                        if np.log(np.random.uniform()) < (sum(new_prior) + sum(new_lik) - sum(old_prior) - sum(old_lik)):
                            # Accept: most of work is done, update each model's logpost and update recorded mcmc draw
                            for mi in range(n_models):
                                if theta_inds[mi] > -1:
                                    # TODO: do we store loglik separately? (so don't need to do whole loglik again?)
                                    self.model_list[mi].params.lp.val = self.model_list[mi].logPost('theta')
                                    # Have to overwrite already recorded sample for theta
                                    self.model_list[mi].params.theta.mcmc.draws[_][0, theta_inds[mi]] = self.model_list[mi].params.theta.val[0, theta_inds[mi]].copy()
                        else:
                            # Reject: need to put things back
                            mu_param.val = mu_param.refVal.copy()
                            for mi in range(n_models):
                                if theta_inds[mi] > -1:
                                    self.model_list[mi].params.theta.prior.params = old_prior_params[mi]
                                    self.model_list[mi].params.theta.val[0, theta_inds[mi]] = self.model_list[mi].params.theta.refVal[0, theta_inds[mi]].copy()

                # Record hierarchical model draws
                self.hier_mu[hi].mcmc.record()
                self.hier_lambda[hi].mcmc.record()
                self.hier_delta[hi].mcmc.record()
            # Recalculate and record logPost into each model
            for model in self.model_list:
                lp_tmp = model.logPost()
                model.params.lp.set_val(lp_tmp)
                model.params.lp.mcmc.draws[_] = lp_tmp

    def mcmc_step_mulam(self, theta_inds, hprm):
        n_models = len(self.model_list)
        # draw cand
        arr_ind = np.unravel_index(0, hprm.val_shape, order='F')
        hprm_cand = hprm.mcmc.draw_candidate(arr_ind, True)
        # check in bounds for mu; check in bounds for theta? (TODO other constraints...)
        inb = hprm.prior.is_in_bounds(hprm_cand)
        # If in bounds, evaluate draw to decide whether or not to accept; if not in bounds, nothing changes
        if inb:
            # Store old/current log prior and prior params for thetas in case reject, put cand into theta priors
            old_prior = []
            old_prior_params = []
            for mi in range(n_models):
                old_prior_params.append(copy.deepcopy(self.model_list[mi].params.theta.prior.params))
                if theta_inds[mi] > -1:
                    old_prior.append(self.model_list[mi].log_prior())
                    if 'mu' in hprm.name:
                        self.model_list[mi].params.theta.prior.params[0][0, theta_inds[mi]] = hprm_cand
                    else:
                        self.model_list[mi].params.theta.prior.params[1][0, theta_inds[mi]] = np.sqrt(1./hprm_cand)
            old_prior.append(hprm.prior.compute_log_prior())
            # Put candidate into hier param, copy old value into refVal in case reject
            hprm.refVal = hprm.val.copy()
            hprm.val[0, 0] = hprm_cand
            # Compute new priors
            new_prior = []
            for mi in range(n_models):
                if theta_inds[mi] > -1:
                    new_prior.append(self.model_list[mi].log_prior())
            new_prior.append(hprm.prior.compute_log_prior())
            # Calculate acceptance
            if np.log(np.random.uniform()) < (np.sum(new_prior) - np.sum(old_prior) + np.log(hprm.mcmc.aCorr)):
                # Accept: most of work is done, just update each model's logpost
                for mi in range(n_models):
                    if theta_inds[mi] > -1:
                        # TODO: do we store loglik separately? (so don't need to do whole loglik again?)
                        self.model_list[mi].params.lp.val = self.model_list[mi].logPost('theta')
            else:
                # Reject: need to put things back
                hprm.val = hprm.refVal.copy()
                for mi in range(n_models):
                    self.model_list[mi].params.theta.prior.params = old_prior_params[mi]





# Prelim testing
if __name__ == '__main__':

    from sepia.dev_test.setup_test_cases import *
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    np.random.seed(42)
    #np.seterr(all='raise') # for testing only

    nmcmc = 500

    m = 80
    n = 20
    nt_sim = 75
    nt_obs = 50
    n_pc = 5  # must be smaller than nt
    nx = 6
    noise_sd = 0.01
    model1, _ = setup_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd, nx=nx, n_pc=n_pc,
                                        seed=42.)
    model2, _ = setup_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd, nx=nx, n_pc=n_pc,
                                        seed=1.)
    model3, _ = setup_multi_sim_and_obs(m=m, n=n, nt_sim=nt_sim, nt_obs=nt_obs, noise_sd=noise_sd, nx=nx, n_pc=n_pc,
                                        seed=12.)

    hier_theta_models = SepiaHierarchicalThetaModels([model1, model2, model3], np.array([(0, 0, 0), (1, 1, 1)]))

    print('model 1 ll %0.3g' % model1.logLik())
    print('model 2 ll %0.3g' % model2.logLik())
    print('model 3 ll %0.3g' % model3.logLik())
    print('sum of 3 model ll %0.3g' % (model1.logLik() + model2.logLik() + model3.logLik()))
    print('hier models ll %0.3g' % sum(hier_theta_models.logLik()))

    hier_theta_models.do_mcmc(nmcmc)

    samples_dict1 = {p.name: p.mcmc_to_array() for p in model1.params.mcmcList}
    samples_dict2 = {p.name: p.mcmc_to_array() for p in model2.params.mcmcList}
    samples_dict3 = {p.name: p.mcmc_to_array() for p in model3.params.mcmcList}
    hier_mu_samples_dict = {'mu%d' % i: hier_theta_models.hier_mu[i].mcmc_to_array() for i in range(len(hier_theta_models.hier_mu))}
    hier_lam_samples_dict = {'lam%d' % i: hier_theta_models.hier_lambda[i].mcmc_to_array() for i in range(len(hier_theta_models.hier_lambda))}
    hier_delta_samples_dict = {'delta%d' % i: hier_theta_models.hier_delta[i].mcmc_to_array() for i in range(len(hier_theta_models.hier_delta))}


    for i in range(len(hier_theta_models.hier_mu)):
        print('hier mu%d acc rate %0.3g' % (i, np.mean(hier_theta_models.hier_mu[i].calc_accept())))

    for i in range(len(hier_theta_models.hier_lambda)):
        print('hier lam%d acc rate %0.3g' % (i, np.mean(hier_theta_models.hier_lambda[i].calc_accept())))

    for i in range(len(hier_theta_models.hier_delta)):
        print('hier delta%d acc rate %0.3g' % (i, np.mean(hier_theta_models.hier_delta[i].calc_accept())))

    plt.subplot(221)
    plt.hist(hier_mu_samples_dict['mu0'])
    plt.subplot(222)
    plt.plot(hier_mu_samples_dict['mu0'])
    plt.subplot(223)
    plt.hist(hier_mu_samples_dict['mu1'])
    plt.subplot(224)
    plt.plot(hier_mu_samples_dict['mu1'])
    plt.show()

    plt.subplot(221)
    plt.hist(hier_lam_samples_dict['lam0'])
    plt.subplot(222)
    plt.plot(hier_lam_samples_dict['lam0'])
    plt.subplot(223)
    plt.hist(hier_lam_samples_dict['lam1'])
    plt.subplot(224)
    plt.plot(hier_lam_samples_dict['lam1'])
    plt.show()

    plt.subplot(221)
    plt.hist(hier_delta_samples_dict['delta0'])
    plt.subplot(222)
    plt.plot(hier_delta_samples_dict['delta0'])
    plt.subplot(223)
    plt.hist(hier_delta_samples_dict['delta1'])
    plt.subplot(224)
    plt.plot(hier_delta_samples_dict['delta1'])
    plt.show()

    #plt.subplot(121)
    #plt.hist(hier_lam_samples_dict['lam0'])
    #plt.subplot(122)
    #plt.hist(hier_lam_samples_dict['lam1'])
    #plt.show()

    # Visual comparison
    for i, k in enumerate(samples_dict1.keys()):
        param_shape = samples_dict1[k].shape[1]
        if param_shape >= 5:
            ncol = 5
            nrow = int(np.ceil(param_shape / ncol))
        else:
            ncol = param_shape
            nrow = 1
        plt.figure(i, [ncol * 3, nrow * 3])
        for j in range(param_shape):
            s1 = samples_dict1[k][:, j]
            s2 = samples_dict2[k][:, j]
            s3 = samples_dict3[k][:, j]
            minval = np.min(np.concatenate([s1, s2, s3]))
            maxval = np.max(np.concatenate([s1, s2, s3]))
            #bw = (maxval - minval) / 10
            k1 = gaussian_kde(s1)
            k2 = gaussian_kde(s2)
            k3 = gaussian_kde(s3)
            x = np.linspace(minval, maxval, 500)
            plt.subplot(nrow, ncol, j + 1)
            plt.plot(x, k1(x), 'k')
            plt.plot(x, k2(x), 'r:')
            plt.plot(x, k3(x), 'g--')
            plt.xlabel(k)
        plt.tight_layout()
        plt.show()





