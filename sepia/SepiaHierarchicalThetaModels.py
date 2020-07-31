

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

    :var model_list: list of SepiaModel objects
    :var hier_theta_inds: indices showing which thetas are hierarchically linked, size (n_hier_theta, n_models)

    """

    def __init__(self, model_list=None, hier_theta_inds=None):
        """
        Instantiate hierarchical model container.

        :param model_list: list of instantiated SepiaModel objects
        :param hier_theta_inds: nparray -- (n_hier_theta, n_models) where each row corresponds to one group of hierarchically
                                modeled thetas, and each column gives the index of the theta within a particular model, with
                                -1 used to indicate no theta from a particular model is part of the hierarchical group.

        """
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

    def do_mcmc(self, nsamp, do_propMH=True, prog=True, do_lockstep=True):
        """
        Does MCMC for hierarchical model.

        :param nsamp: int -- how many MCMC samples
        :param do_propMH: boolean -- whether to use propMH sampling for params with stepType propMH
        :param prog: boolean -- whether to show progress bar for sampling
        :param do_lockstep: boolean -- whether to do lockstep update
        """
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




