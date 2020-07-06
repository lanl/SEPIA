"""
@author: nklein

Class for holding multiple models with shared thetas

"""

import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

from sepia.SepiaParam import SepiaParam, SepiaParamList
from sepia.SepiaLogLik import compute_log_lik
from sepia.SepiaModel import SepiaModel


class SepiaSharedThetaModels:
    """
    Container for multiple models with selected thetas shared across models.

    :param model_list: list of instantiated SepiaModel objects
    :param shared_theta_inds: nparray -- (n_shared_theta, n_models) where each row corresponds to one group of shared
                              thetas, and each column gives the index of the theta within a particular model, with
                             -1 used to indicate no theta from a particular model is part of the shared group.

    """

    def __init__(self, model_list=None, shared_theta_inds=None):
        self.model_list = model_list                # List of instantiated SepiaModel objects
        self.shared_theta_inds = shared_theta_inds  # Matrix (n_shared_theta, n_models) indicating shared indices, -1 means not in a model
        # Example: shared_theta_inds = np.array([(1, 1, 1), (2, -1, 4)) for 3 models, theta index 1 tied in all,
        #          theta indices 2/4 tied in models 1 and 3 but no corresponding theta in model 2.
        if not shared_theta_inds.shape[1] == len(model_list):
            raise Exception('Number of models does not match provided shared theta lists')
        self.setup_shared_theta()

    def setup_shared_theta(self):
        # sets up bookkeeping to make mcmc loop simpler
        n_shared, n_models = self.shared_theta_inds.shape
        # Get index of first model containing each shared parameter to designate it as the "sampling model"
        to_sample = np.zeros_like(self.shared_theta_inds)
        for i in range(n_shared):
            ti_row = self.shared_theta_inds[i, :]
            for j in range(n_models):
                if ti_row[j] > -1:
                    break
            to_sample[i, j] = 1
        # Get indices of other models sharing each parameter, not including the "sampling model"
        to_update = np.zeros_like(self.shared_theta_inds)
        for i in range(n_shared):
            ti_row = self.shared_theta_inds[i, :]
            for j in range(n_models):
                if ti_row[j] > -1 and to_sample[i, j] == 0:
                    to_update[i, j] = 1
        self.to_sample = to_sample
        self.to_update = to_update

    def log_prior(self):
        """
        Evalutes log prior across all models, returns list.
        :return:
        """
        lp_list = []
        for model in self.model_list:
            lp = model.log_prior()
            model.num.compute_log_prior = lp
            lp_list.append(lp)
        return lp_list

    def logLik(self, cvar='all', cindex=None):
        """
        Evaluates log lik across all models, returns list.
        :param cvar:
        :param cindex:
        :return:
        """
        L_list = []
        for model in self.model_list:
            L = compute_log_lik(model, cvar, cindex)
            L_list.append(L)
        return L_list
        
    def logPost(self, cvar='all', cindex=None):
        """
        Evaluates log post across all models, returns list.
        :param cvar:
        :param cindex:
        :return:
        """
        ll_list = self.logLik(cvar, cindex)
        lp_list = self.log_prior()
        #print(f'{lp}') 
        return [ll_list[i] + lp_list[i] for i in range(len(ll_list))]

    def do_mcmc(self, nsamp, do_propMH=True, prog=True):
        # Initialize all models
        for model in self.model_list:
            model.params.lp.set_val(model.logPost())
        # Main sampling loop
        for _ in tqdm(range(nsamp), desc='MCMC sampling', mininterval=0.5, disable=not(prog)):
            for mi, model in enumerate(self.model_list):
                # Loop over parameters
                for prm in model.params.mcmcList:
                    # Loop over indices within parameter
                    for ind in range(int(np.prod(prm.val_shape))):
                        arr_ind = np.unravel_index(ind, prm.val_shape, order='F')
                        Accept = False
                        # Make reference copies in this model in case we reject
                        prm.refVal = prm.val.copy()
                        model.refNum = model.num.ref_copy(prm.name)
                        # Check whether this theta ind is shared
                        if prm.name is 'theta' and np.any(ind == self.shared_theta_inds[:, mi]):
                            shr_ind = np.where(self.shared_theta_inds[:, mi] == ind)[0]
                            cand = prm.mcmc.draw_candidate(arr_ind, do_propMH) # to match matlab, we will draw here (may not be used at all)
                            # If this is the model to sample from, do sample and update other models
                            if self.to_sample[shr_ind, mi] == 1:
                                # Draw candidate
                                #cand = prm.mcmc.drawCand(arr_ind, do_propMH)
                                inb = prm.prior.is_in_bounds(cand)
                                other_model_inds = np.where(self.to_update[shr_ind, :].squeeze() == 1)[0]
                                # Make reference copies across other models in case we reject, check if cand in bounds
                                for omi in other_model_inds:
                                    om = self.model_list[omi]
                                    om.params.theta.refVal = om.params.theta.val.copy()
                                    om.refNum = om.num.ref_copy('theta')
                                    inb = inb and om.params.theta.prior.is_in_bounds(cand)
                                # If in bounds, put cand in val and evaluate log lik
                                if inb:
                                    # Get current logpost: only evaluate theta prior on this model!
                                    other_mod_loglik = sum([self.model_list[i].logLik(prm.name) for i in other_model_inds])
                                    lp = model.logPost(prm.name, ind) + other_mod_loglik
                                    # Set candidate into models to get new lp
                                    prm.val[arr_ind] = cand
                                    for omi in other_model_inds:
                                        om = self.model_list[omi]
                                        # Get index of shared theta in other model
                                        om_theta_ind = self.shared_theta_inds[shr_ind, omi][0]
                                        om_arr_ind = np.unravel_index(om_theta_ind, om.params.theta.val_shape, order='F')
                                        # Set candidate
                                        om.params.theta.val[om_arr_ind] = cand
                                    # Get new logpost: only evaluate theta prior on this model!
                                    other_mod_loglik_new = sum([self.model_list[i].logLik(prm.name) for i in other_model_inds])
                                    this_model_logpost_new = model.logPost(prm.name, ind)
                                    clp = this_model_logpost_new + other_mod_loglik_new
                                    if np.log(np.random.uniform()) < (clp - lp):
                                        # Accept: set current log post for each model (need to store log prior?)
                                        model.params.lp.set_val(this_model_logpost_new)
                                        for omi in other_model_inds:
                                            om = self.model_list[omi]
                                            om.params.lp.set_val(om.logPost(prm.name))
                                    else:
                                        # Reject
                                        prm.mcmc.reject(ind, model)
                                        for omi in other_model_inds:
                                            om = self.model_list[omi]
                                            om_theta_ind = self.shared_theta_inds[shr_ind, omi][0]
                                            om.params.theta.mcmc.reject(om_theta_ind, om)
                            # If not sampling this theta in this model, do nothing and continue to next param index
                            else:
                                continue # TODO need to do anything else here? store lp?
                        # If not a shared theta, continue with usual sampling
                        else:
                            # Make reference copies in this model in case we reject
                            prm.refVal = prm.val.copy()
                            model.refNum = model.num.ref_copy(prm.name)
                            lp = model.logPost(prm.name, ind)
                            # Usual sampling for non shared params
                            # Draw candidate
                            cand = prm.mcmc.draw_candidate(arr_ind, do_propMH)
                            # Set value to candidate
                            prm.val[arr_ind] = cand
                            if not prm.fixed[arr_ind]: # If not supposed to be fixed, check for acceptance
                                if prm.mcmc.aCorr and prm.prior.is_in_bounds(prm.val[arr_ind]):
                                    # This logPost uses val which has the candidate modification
                                    clp = model.logPost(prm.name, ind)
                                    if np.log(np.random.uniform()) < (clp - lp + np.log(prm.mcmc.aCorr)):
                                        Accept = True
                            if Accept:
                                # Accept basically does nothing, the now modified val stays there for the next step
                                prm.mcmc.accept()
                                model.params.lp.set_val(clp)
                            else:
                                # Reject sets val back to refVal that was stored at the top
                                prm.mcmc.reject(ind, model)
                    # Record parameter TODO check that thetas always get recorded, even for other models...
                    prm.mcmc.record()
                # Record lp
                model.params.lp.mcmc.record()
                # Record values at end of loop over params for this model
                #for prm in model.params.mcmcList:
                #    prm.mcmc.record()
            # After done with all sampling across models, record the step (need to get all theta modifications first)
            #for model in self.model_list:
            #    for prm in model.params.mcmcList:
            #        prm.mcmc.record()
            #    model.params.lp.mcmc.record()



# Prelim testing
if __name__ == '__main__':

    from sepia.dev_test.setup_test_cases import *
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import copy

    nmcmc = 50

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

    shared_theta_models = SepiaSharedThetaModels([copy.deepcopy(model1), copy.deepcopy(model2), copy.deepcopy(model3)],
                                                 np.array([(1, 1, 2), (-1, 3, 3), (2, 2, -1)]))
    #shared_theta_models = SepiaSharedThetaModels([model1, model2, model3], np.array([(0, 0, 0)]))


    print('model 1 ll %0.3g' % model1.logLik())
    print('model 2 ll %0.3g' % model2.logLik())
    print('model 3 ll %0.3g' % model3.logLik())
    print('shared models ll %0.3g' % sum(shared_theta_models.logLik()))

    np.random.seed(42)
    shared_theta_models.do_mcmc(1000)
    np.random.seed(42)
    model1.do_mcmc(1000)
    np.random.seed(42)
    model2.do_mcmc(1000)
    np.random.seed(42)
    model3.do_mcmc(1000)

    samples_dict1 = {p.name: p.mcmc_to_array() for p in model1.params.mcmcList}
    samples_dict2 = {p.name: p.mcmc_to_array() for p in model2.params.mcmcList}
    samples_dict3 = {p.name: p.mcmc_to_array() for p in model3.params.mcmcList}

    shared_samples_dict1 = {p.name: p.mcmc_to_array() for p in shared_theta_models.model_list[0].params.mcmcList}
    shared_samples_dict2 = {p.name: p.mcmc_to_array() for p in shared_theta_models.model_list[1].params.mcmcList}
    shared_samples_dict3 = {p.name: p.mcmc_to_array() for p in shared_theta_models.model_list[2].params.mcmcList}

    # See if samples differ for shared params
    for i in range(shared_theta_models.shared_theta_inds.shape[0]): # loop over shared params
        tinds = shared_theta_models.shared_theta_inds[i, :]
        s1 = shared_samples_dict1['theta'][:, tinds[0]]
        s2 = shared_samples_dict2['theta'][:, tinds[1]]
        s3 = shared_samples_dict3['theta'][:, tinds[2]]
        plt.plot(s1, 'k')
        plt.plot(s2, 'r:')
        plt.plot(s3, 'g--')
        plt.show()
        if tinds[0] > -1 and tinds[1] > -1:
            assert np.allclose(s1, s2)
        if tinds[0] > -1 and tinds[2] > -1:
            assert np.allclose(s1, s3)
        if tinds[1] > -1 and tinds[2] > -1:
            assert np.allclose(s2, s3)

    # Compare theta samples from individual models vs shared model
    plt.figure(i, [15, 6])
    # Individual model samples
    for j in range(samples_dict1['theta'].shape[1]):
        s1 = samples_dict1['theta'][:, j]
        s2 = samples_dict2['theta'][:, j]
        s3 = samples_dict3['theta'][:, j]
        minval = np.min(np.concatenate([s1, s2, s3]))
        maxval = np.max(np.concatenate([s1, s2, s3]))
        # bw = (maxval - minval) / 10
        k1 = gaussian_kde(s1)
        k2 = gaussian_kde(s2)
        k3 = gaussian_kde(s3)
        x = np.linspace(minval, maxval, 500)
        plt.subplot(2, 5, j + 1)
        plt.plot(x, k1(x), 'k')
        plt.plot(x, k2(x), 'r:')
        plt.plot(x, k3(x), 'g--')
        plt.xlabel('theta %d' % j)
        plt.title('individual')
    # Shared model samples
    for j in range(samples_dict1['theta'].shape[1]):
        s1 = shared_samples_dict1['theta'][:, j]
        s2 = shared_samples_dict2['theta'][:, j]
        s3 = shared_samples_dict3['theta'][:, j]
        minval = np.min(np.concatenate([s1, s2, s3]))
        maxval = np.max(np.concatenate([s1, s2, s3]))
        # bw = (maxval - minval) / 10
        k1 = gaussian_kde(s1)
        k2 = gaussian_kde(s2)
        k3 = gaussian_kde(s3)
        x = np.linspace(minval, maxval, 500)
        plt.subplot(2, 5, 5 + j + 1)
        plt.plot(x, k1(x), 'k')
        plt.plot(x, k2(x), 'r:')
        plt.plot(x, k3(x), 'g--')
        plt.xlabel('theta %d' % j)
        plt.title('shared')
    plt.tight_layout()
    plt.show()

    # Visualize all samples from shared model
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
            s1 = shared_samples_dict1[k][:, j]
            s2 = shared_samples_dict2[k][:, j]
            s3 = shared_samples_dict3[k][:, j]
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



