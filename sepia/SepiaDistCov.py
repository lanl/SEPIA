
import numpy as np


class SepiaDistCov(object):
    """
    Class for computing/storing distances and covariance matrix.

    If only one data set is given, the auto-distance is computed, else distance with data2.

    :param data: input data shape (n_samples, _)
    :param data2: optional second data set to compute distances with data

    """

    def __init__(self, data, data2=None, catInd=False):
        if data2 is None:
            self.type = 1
            self.n = data.shape[0]
            self.ind = np.triu_indices(n=self.n, k=1)
            self.sqdist = np.square(data[self.ind[0], :] - data[self.ind[1], :])
        else:
            self.type = 2
            self.n = data.shape[0]
            self.m = data2.shape[0]
            self.ind = np.unravel_index(np.arange(self.n * self.m), (self.n, self.m))
            self.sqdist = np.square(data[self.ind[0], :] - data2[self.ind[1], :])

    def compute_cov_mat(self, beta, lamz, lams=None, verbose=False):
        """
        Computes GP covariance matrix using precomputed distances.

        :param beta: GP lengthscale parameter
        :param lamz: GP marginal precision
        :param lams: nugget precision
        :param verbose: True will print stuff
        :return: computed covariance matrix
        """
        if verbose:
            print('in covMat, beta shape ',beta.shape,
                    ', lamz shape ',lamz.shape,
                    ', sqdist shape ',self.sqdist.shape)
        # lamz is the process precision, nominally 1 for scaled data
        # lams is a 'nugget' or noise variance
        if self.sqdist.shape[0] == 0:
            sqdist_exp_sum = np.array([])
        else:
            sqdist_sum = np.squeeze(self.sqdist @ beta)
            sqdist_exp_sum = np.exp(-sqdist_sum) / lamz
        if self.type == 1:
            cov = np.zeros((self.n, self.n))
            cov[self.ind] = sqdist_exp_sum
            cov = cov + cov.T
            lam = 1/lamz 
            if lams is not None: lam += 1/lams
            np.fill_diagonal(cov, lam)
        elif self.type == 2:
            cov = np.zeros((self.n, self.m))
            cov[self.ind] = sqdist_exp_sum
        return cov

