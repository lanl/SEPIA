
import numpy as np


class SepiaPrior:
    """
    Sepia prior class, containing prior distribution name, parameters, and bounds.

    :var parent: SepiaParam -- which parameter this prior corresponds to
    :var dist: string -- which prior distribution, in 'Normal', 'Gamma', 'Beta', 'Uniform'
    :var params: list -- each element is a different parameter to the distribution (ie first is mean, second is SD), can be matrix valued
    :var bounds: list -- bounds for each prior (can be np.inf)
    """

    def __init__(self, parent, dist='Normal', params=None, bounds=None):
        """
        Instantiate SepiaPrior object.

        :param parent: SepiaParam -- parent object
        :param dist: string -- prior distribution name ('Normal', 'Gamma', 'Beta', 'Uniform')
        :param params: list -- each element of list is a different parameter to the distribution, can be ndarray or scalar
        :param bounds: list -- two-element list of bounds for prior, each element can be scalar or list to provide different bounds to each param
        """
        self.parent = parent
        self.dist = dist 
        if not (dist == 'Normal' or dist == 'Gamma' or dist == 'Beta' or dist == 'Uniform'):
            raise Exception('Invalid dist type {dist} in sepPrior')
        # If params given, process into correct format; params input should always be a list, either of arrays or scalars,
        # and is transformed into a list of arrays that match param val_shape
        if isinstance(params, list):
            for i in range(len(params)):
                if np.isscalar(params[i]):
                    params[i] = params[i] * np.ones(parent.val_shape)
                elif params[i].shape != parent.val_shape:
                    raise Exception('initial non-scalar prior param values does not match variable shape')
            self.params = params
        # Set default prior parameters if params not given
        else:
            if dist == 'Normal':
                self.params = [np.zeros(parent.val_shape), np.ones(parent.val_shape)]
            elif dist == 'Gamma':
                self.params = [10 * np.ones(parent.val_shape), 10 * np.ones(parent.val_shape)]
            elif dist == 'Beta':
                self.params = [np.ones(parent.val_shape), np.ones(parent.val_shape)]
            elif dist == 'Uniform':
                self.params = []
        # Set bounds
        if bounds is not None:
            lb = bounds[0] * np.ones(parent.val_shape) if np.isscalar(bounds[0]) else bounds[0]
            ub = bounds[1] * np.ones(parent.val_shape) if np.isscalar(bounds[1]) else bounds[1]
            self.bounds = [lb, ub]
        else:
            if dist == 'Normal':
                self.bounds = [np.zeros(parent.val_shape), np.ones(parent.val_shape)]
            elif dist == 'Gamma':
                self.bounds = [np.zeros(parent.val_shape), 1e6 * np.ones(parent.val_shape)]
            elif dist == 'Beta':
                self.bounds = [np.zeros(parent.val_shape), np.ones(parent.val_shape)]
            elif dist == 'Uniform':
                self.bounds = [np.zeros(parent.val_shape), np.ones(parent.val_shape)]

    def compute_log_prior(self):
        """
        Compute log prior for this particular parent SepiaParam object.

        :return: log prior value
        """
        x = self.parent.val
        lp = 0
        if not self.is_in_bounds():
            lp = -np.inf
        else:
            if self.dist == 'Normal':
                # proportional to log normal density
                lp = -0.5 * np.sum(np.square((x - self.params[0])/self.params[1]))
            elif self.dist == 'Gamma':
                lp = np.sum((self.params[0] - 1) * np.log(x) - self.params[1] * x)
            elif self.dist == 'Beta':
                if self.parent.name in ['betaU', 'betaV']:
                    rho = np.exp(-0.25 * x)
                    rho[rho > 0.999] = 0.999
                    lp = np.sum((self.params[0] - 1) * np.log(rho) + (self.params[1] - 1) * np.log(1 - rho))
                else:
                    lp = np.sum((self.params[0] - 1) * np.log(x) + (self.params[1] - 1) * np.log(1 - x))
        return lp

    def is_in_bounds(self, x=None):
        """
        Check whether value is in bounds. By default, with no x, checks self.parent.val.

        :param: x -- nparray value, with self.parent.val_shape; if none, defaults to self.parent.val
        :return: bool -- True if all in bounds, False otherwise
        """
        if x is None:
            x = self.parent.val
        # check whether bounds are satisfied for all variables
        return True if np.all(np.logical_and(x > self.bounds[0], x < self.bounds[1])) else False
