
import numpy as np

class SepiaDistCov(object):
    """
    Class for computing/storing distances and covariance matrix, typically not called directly by user but used in model.

    :var type: int -- 1: self-distance for single input data, 2: cross-distance for two input data
    :var n: int -- number of observations in data set 1
    :var m: int -- number of observations in data set 2
    :var ind: nparray -- indices into matrix upper triangle
    :var sqdist: nparray -- squared distances for upper triangle

    """

    def __init__(self, data, data2=None, cat_ind=None):
        """
        Instantiate SepiaDistCov.

        If only one data set is given, the auto-distance is computed, else computes distance between data and data2.

        :param data: nparray -- input data, shape (n_samples, _)
        :param data2: nparray -- optional, second data set to compute distances with data
        :param cat_ind: list, 1d array -- optional, cat_ind as passed to sepiadata

        """
        if data2 is None:
            self.type = 1
            self.n = data.shape[0]
            self.p = data.shape[1]
            self.ind = np.triu_indices(n=self.n, k=1)
            if cat_ind is None:
                self.sqdist = np.square(data[self.ind[0], :] - data[self.ind[1], :])
            else:
                is_cat = [ci > 0 for ci in cat_ind]
                not_cat = [ci == 0 for ci in cat_ind]
                data_notcat = data[:, not_cat]
                data_iscat = data[:, is_cat]
                sqdist = np.zeros((self.ind[0].shape[0], data.shape[1]))
                sqdist[:, not_cat] = np.square(data_notcat[self.ind[0], :] - data_notcat[self.ind[1], :])
                sqdist[:, is_cat] = 0.5*(data_iscat[self.ind[0], :] != data_iscat[self.ind[1], :])
                self.sqdist = sqdist
        else:
            self.type = 2
            self.n = data.shape[0]
            self.m = data2.shape[0]
            self.p = data.shape[1]
            if data.shape[1] != data2.shape[1]:
                raise ValueError('SepiaDistCov: p for data and data2 not the same')
            self.ind = np.unravel_index(np.arange(self.n * self.m), (self.n, self.m))
            if cat_ind is None:
                self.sqdist = np.square(data[self.ind[0], :] - data2[self.ind[1], :])
            else:
                is_cat = [ci > 0 for ci in cat_ind]
                not_cat = [ci == 0 for ci in cat_ind]
                data_notcat = data[:, not_cat]
                data_iscat = data[:, is_cat]
                data2_notcat = data2[:, not_cat]
                data2_iscat = data2[:, is_cat]
                sqdist = np.zeros((self.ind[0].shape[0], data.shape[1]))
                sqdist[:, not_cat] = np.square(data_notcat[self.ind[0], :] - data2_notcat[self.ind[1], :])
                sqdist[:, is_cat] = 0.5*(data_iscat[self.ind[0], :] != data2_iscat[self.ind[1], :])
                self.sqdist = sqdist

    def compute_cov_mat(self, beta, lamz, lams=None, verbose=False):
        """
        Computes GP covariance matrix using precomputed distances.

        :param beta: nparray -- GP lengthscale parameters
        :param lamz: nparray -- GP marginal precision parameters
        :param lams: scalar -- nugget precision
        :param verbose: boolean -- whether to print extra information
        :return: computed covariance matrix
        """
        if verbose:
            # TODO: Should be checks whether these conform
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
