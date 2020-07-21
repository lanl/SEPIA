
import numpy as np
from sepia.SepiaPrior import SepiaPrior
from sepia.SepiaMCMC import SepiaMCMC


# This is a list of parameters; MCMC will cycle through these
class SepiaParamList:
    def __init__(self):
        pass

    def __iter__(self):
        for attr in dir(self):
            if not attr.startswith("__"):
                res = self.__getattribute__(attr)
                if isinstance(res, SepiaParam):
                    yield res

    def get_values(self):
        values = {}
        for p in self:
            values[p.name] = p.val
        return values


class SepiaParam:
    """
    Object containing value, shape, name, prior for a single matrix-valued parameter in the model.

    :param val: ndarray -- starting value (can be scalar and will then be expanded to val_shape)
    :param name: string -- parameter name
    :param val_shape: int, tuple -- shape of parameter
    :param dist: string -- prior distribution name ('Normal', 'Gamma', 'Beta', 'Uniform')
    :param params: list -- list of ndarrays, params for prior
    :param bounds: list -- bounds for parameter values (can be inf)
    :param mcmcStepType: string -- step type for MCMC ('Normal', 'Uniform', 'PropMH')
    :param mcmcStepParam: ndarray, scalar -- step size parameter (has val_shape or scalar is expanded to that shape)
    :param orig_range: list, size 2 -- range for untransformed parameter (applicable for theta)
    :raises: Exception if non scalar val doesn't match val_shape
    """

    def __init__(self, val, name, val_shape=1, dist='Normal', params=[], bounds=False, mcmcStepType='Normal',
                 mcmcStepParam=0.1, orig_range=None):
        if np.isscalar(val):
            self.val = val * np.ones(val_shape)
        else:
            if val.shape != val_shape:
                raise Exception('initial non-scalar value does not match shape')
            self.val = val
        self.fixed = np.zeros_like(self.val, dtype=bool)
        self.name = name
        self.val_shape = val_shape
        if dist is not 'Recorder':
            self.refVal = False  # place to put val when mcmc cand eval
            self.prior = SepiaPrior(self, dist=dist, params=params, bounds=bounds)
            if np.isscalar(mcmcStepParam):
                mcmcStepParam = mcmcStepParam * np.ones(val_shape)
            self.mcmc = SepiaMCMC(self, stepType=mcmcStepType, stepParam=mcmcStepParam)
        else:
            self.mcmc = SepiaMCMC(self, stepType='Recorder')
        self.orig_range = orig_range

    def get_num_samples(self):
        return len(self.mcmc.draws)

    def mcmc_to_array(self, trim=0, sampleset=False, flat=True, untransform_theta=False):
        """
        Convert internal representation of MCMC draws to an array.

        :param trim: number of samples to trim from beginning of chain
        :param sampleset: indices of samples to use
        :param flat: whether to flatten size of parameter
        :param untransform_theta: Boolean -- whether or not to untransform theta to original scale
        :return: ndarray -- MCMC draws
        """
        if type(sampleset) is int: sampleset=[sampleset] # it has to be a list
        if sampleset is not False:
            draws = np.array(self.mcmc.draws)[sampleset, :, :]  # (nsamp, p+q, pu)
        else:
            draws = np.array(self.mcmc.draws)[trim:, :, :] # (nsamp, p+q, pu)
        if untransform_theta:
            draws = self.untransform_theta(draws)
        if flat:
            draws_flat = np.zeros((draws.shape[0], draws.shape[1]*draws.shape[2]))
            for samp in range(draws.shape[0]): # TODO I don't think a loop is needed here... just one reshape
                draws_flat[samp, :] = np.ndarray.flatten(draws[samp, :, :], order='F')
            draws = draws_flat
        return draws

    def calc_accept_rate(self):
        tlist=self.calc_accept()
        return np.sum(tlist,axis=0) / tlist.shape[0]

    def calc_accept(self):
        """
        Calculate whether each draw was rejected or accepted.

        :return: ndarray -- 0/1 whether accepted or not
        """
        # Calculate binary accept/reject arrays for each param from mcmc chain
        draws = np.array(self.mcmc.draws)
        acc = np.ones((1, draws.shape[1], draws.shape[2]))
        diff = np.diff(draws, axis=0)
        no_change = np.isclose(diff, 0)
        changed = np.logical_not(no_change).astype(float)
        acc = np.concatenate([acc, changed], axis=0)
        return acc

    def set_val(self, sval):
        """
        Sets val to new value, expanding to val_shape from scalar if needed.

        :param sval: ndarray, scalar -- new value
        """
        if np.isscalar(sval):
            sval = np.array(sval).reshape(self.val_shape)
        self.val = sval

    def untransform_theta(self, sample_array):
        """
        Untransforms an array of theta values using stored t_min and t_max.

        :param sample_array: ndarray -- array of theta samples, (nsamp, theta_size)
        :return: untransformed sample array
        """
        if self.orig_range is not None:
            t_min = self.orig_range[0]
            t_max = self.orig_range[1]
            sample_array = sample_array * (t_max - t_min) + t_min
        return sample_array







