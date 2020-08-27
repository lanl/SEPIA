import numpy as np

class DataContainer(object):
    """
    DataContainer serves to contain all data structures for a single data source (simulation or observation).

    This is constructed as part of SepiaData and generally won't be used directly by a user. Instance attributes:

    :var x: nparray or None -- x values, controllable inputs/experimental variables (n, p)
    :var y: nparray -- y values (n, ell)
    :var t: nparray or None -- t values, non-controllable inputs (n, q)
    :var y_ind: nparray or None -- indices for multivariate y outputs (ell, )
    :var K: nparray or list -- (pu, ell) PCA basis, or list of K matrices for ragged observations
    :var D: nparray or list or None -- (pv, ell) discrepancy basis, or list of D matrices for ragged observations
    :var y_sd: nparray -- standard deviation of original y values (may be scalar or array size ell)
    :var y_mean: nparray -- mean of original y values (may be salar or array size ell)
    :var y_std: nparray -- standardized y values (n, ell)
    :var x_trans: nparray -- x values translated to unit hypercube (n, p)
    :var t_trans: nparray -- t values translated to unit hypercube (n, q)
    """

    def __init__(self, x, y, t=None, y_ind=None):
        """
        Initializes DataContainer object.

        :param x: GP inputs (controllable/experimental conditions, ie the ones known for both sim and obs), shape (n, p)
        :param y: GP outputs, shape (n, ell) or list of 1D arrays for ragged observations
        :param t: optional GP inputs (not controllable, ie the ones known only for sim), shape (n, q)
        :param y_ind: optional y indices (needed if ell > 1) or list of 1D arrays for ragged observations
        """
        self.x = x
        self.y = y
        # Parse mandatory inputs (x and y)
        if self.x.shape[0] != len(self.y):
            raise ValueError('Number of observations in x and y must be the same size.')
        # Optional inputs (depending on if sim_only or scalar_out)
        if t is not None and t.shape[0] != self.x.shape[0]:
            raise ValueError('Dimension 0 of x and t must be the same size.')
        if self.y[0].shape[0] > 1 and y_ind is None:
            raise ValueError('y_ind required when y has multivariate output.')
        if y_ind is not None:
            if isinstance(y_ind, list):
                y_shapes = np.array([ytmp.shape for ytmp in self.y])
                y_ind_shapes = np.array([ytmp.shape for ytmp in y_ind])
                if not np.all(y_shapes[:,0] == y_ind_shapes[:,0]):
                    raise ValueError('Dimension 1 of y must match dimension 0 of y_ind.')
            else:
                if self.y.shape[1] != y_ind.shape[0]:
                    raise ValueError('Dimension 1 of y must match dimension 0 of y_ind.')
        self.t = t
        self.y_ind = y_ind
        # Basis and transform stuff initialized to None
        self.K = None
        self.D = None
        self.orig_y_sd = None
        self.orig_y_mean = None
        self.y_std = None
        self.x_trans = None
        self.t_trans = None
        self.orig_t_min = None
        self.orig_t_max = None
        self.orig_x_min = None
        self.orig_x_max = None

    # These make sure x/y/t are 2D no matter what
    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        if x.ndim == 1:
            x = x[:, None]
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        if not isinstance(y, list):
            if y.ndim == 1:
                y = y[:, None]
        self.__y = y

    @property
    def t(self):
        return self.__t

    @t.setter
    def t(self, t):
        if t is not None:
            if t.ndim == 1:
                t = t[:, None]
        self.__t = t