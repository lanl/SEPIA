
import numpy as np
from tqdm import tqdm


class SepiaSharedThetaModels:
    """
    Container for multiple models with sharing of selected thetas between models.

    :var list model_list: list of instantiated `sepia.SepiaModel` objects
    :var numpy.ndarray shared_theta_inds: indices showing which thetas are shared across models, shape (n_hier_theta, n_models)
    :var int n_hier: number of shared theta groups
    :var int n_models: number of models
    :var numpy.ndarray to_update: 0/1 matrix indicating which variables need to be shared
    :var numpy.ndarray to_sample: 0/1 matrix indicating which variable should be sampled (then copied to others)

    """

    def __init__(self, model_list=None, shared_theta_inds=None):
        """
        Instantiate shared theta model object.

        :param list model_list: list of instantiated `sepia.SepiaModel` objects
        :param numpy.ndarray shared_theta_inds: indices showing which thetas are shared, shape (n_shared_theta, n_models)
        :raises TypeError: if number of models doesn't match `shared_theta_inds` shape or if variable types aren't the same (categorical vs continuous)

        .. note:: In `shared_theta_inds`, each row corresponds to one group of shared thetas, and each
                  column gives the index of the theta within a particular model, with -1 used to indicate no theta
                  from a particular model is part of the shared group.
                  Example: shared_theta_inds = np.array([(1, 1, 1), (2, -1, 4)) for 3 models, theta index 1 shared across all models,
                  theta indices 2/4shared across models 1 and 3 but no corresponding theta in model 2.

        """
        self.model_list = model_list                # List of instantiated SepiaModel objects
        self.shared_theta_inds = shared_theta_inds  # Matrix (n_shared_theta, n_models)
        if not shared_theta_inds.shape[1] == len(model_list):
            raise TypeError('Number of models does not match provided shared theta lists')
        for st in shared_theta_inds:
            t_cat = [model_list[i].data.t_cat_ind[st[i]] for i in range(len(model_list)) if st[i] >= 0]
            if not np.all(t_cat == t_cat[0]):
                raise TypeError('Shared indices must share same t_cat_ind values.')
        self.setup_shared_theta()

    def setup_shared_theta(self):
        # sets up bookkeeping to make mcmc loop simpler
        n_shared, n_models = self.shared_theta_inds.shape
        self.n_shared = n_shared
        self.n_models = n_models
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

    def do_mcmc(self, nsamp, do_propMH=True, prog=True):
        """
        Do MCMC for shared theta model.

        :param int nsamp: number of MCMC samples
        :param bool do_propMH: use propMH sampling for params with stepType propMH?
        :param bool prog: show progress bar for sampling?

        """
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
                        if prm.name == 'theta' and np.any(ind == self.shared_theta_inds[:, mi]):
                            shr_ind = np.where(self.shared_theta_inds[:, mi] == ind)[0]
                            # to match matlab RNG, we will draw here (may not be used at all)
                            #  check for categorical theta
                            if prm.name == 'theta' and model.data.t_cat_ind[ind] > 0:
                                # Get possible category values, excluding current value
                                cat_vals = [i for i in range(1, model.data.t_cat_ind[ind] + 1) if i != prm.val[arr_ind]]
                                # Choose one
                                cand = np.random.choice(cat_vals, 1)
                            else:
                                cand = prm.mcmc.draw_candidate(arr_ind, do_propMH)
                            # If this is the model to sample from, update other models
                            if self.to_sample[shr_ind, mi] == 1:
                                inb = cand > prm.prior.bounds[0][arr_ind] and cand < prm.prior.bounds[1][arr_ind]
                                inb = inb and prm.prior.obeys_constraint(cand) # check theta constraint -- going to assume theta has same constraints in all models?
                                other_model_inds = np.where(self.to_update[shr_ind, :].squeeze() == 1)[0]
                                # Make reference copies across other models in case we reject, check if cand in bounds
                                for omi in other_model_inds:
                                    om = self.model_list[omi]
                                    om.params.theta.refVal = om.params.theta.val.copy()
                                    om.refNum = om.num.ref_copy('theta')
                                    inb = inb and cand > om.params.theta.prior.bounds[0][arr_ind] and cand < om.params.theta.prior.bounds[1][arr_ind]
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
                                continue
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
                                if prm.mcmc.aCorr and prm.prior.is_in_bounds() and prm.prior.obeys_constraint():
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
                    # Record parameter
                    prm.mcmc.record()
                # Record lp
                model.params.lp.mcmc.record()

