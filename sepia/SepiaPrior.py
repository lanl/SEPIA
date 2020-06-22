
import numpy as np


class SepiaPrior:
    """
    Sepia prior class, containing prior distribution name, parameters, and bounds.

    :param parent: SepiaParam -- parent object
    :param dist: string -- prior distribution name ('Normal', 'Gamma', 'Beta', 'Uniform')
    :param params: list -- each element of list is a different parameter to the distribution, can be ndarray or scalar
    :param bounds: list -- list of bounds for prior
    """
    def __init__(self, parent, dist='Normal', params=False, bounds=False):
        self.parent = parent
        self.dist = dist 
        if not (dist is 'Normal' or dist is 'Gamma' or  dist is 'Beta' or dist is 'Uniform'):
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
            if dist is 'Normal':
                self.params = [np.zeros(parent.val_shape), np.ones(parent.val_shape)]
            elif dist is 'Gamma':
                self.params = [10 * np.ones(parent.val_shape), 10 * np.ones(parent.val_shape)]
            elif dist is 'Beta':
                self.params = [np.ones(parent.val_shape), np.ones(parent.val_shape)]
            elif dist is 'Uniform':
                self.params = []
        # Set bounds
        if dist is 'Normal':
            self.bounds = bounds if bounds else [0, 1]
        elif dist is 'Gamma':
            self.bounds = bounds if bounds else [0, 1e6]
        elif dist is 'Beta':
            self.bounds = bounds if bounds else [0, 1]
        elif dist is 'Uniform':
            self.bounds = bounds if bounds else [0, 1]

    def compute_log_prior(self):
        """
        Compute log prior for this particular parent SepiaParam object.

        :return: log prior value
        """
        x = self.parent.val
        lp = 0
        if not self.is_in_bounds(x):
            lp = -np.inf
        else:
            if self.dist is 'Normal':
                # proportional to log normal density
                lp = -0.5 * np.sum(np.square((x - self.params[0])/self.params[1]))
            elif self.dist is 'Gamma':
                lp = np.sum((self.params[0] - 1) * np.log(x) - self.params[1] * x)
            elif self.dist is 'Beta':
                if self.parent.name is 'betaU' or 'betaV':
                    rho = np.exp(-0.25 * x)
                    rho[rho > 0.999] = 0.999
                    lp = np.sum((self.params[0] - 1) * np.log(rho) + (self.params[1] - 1) * np.log(1 - rho))
                else:
                    lp = np.sum((self.params[0] - 1) * np.log(x) + (self.params[1] - 1) * np.log(1 - x))
        return lp

    def is_in_bounds(self, x):
        """
        Check whether x is in bounds.

        :param x: scalar, array -- value to check
        :return: bool -- True if all in bounds, False otherwise
        """
        # check whether bounds are satisfied for all variables
        return True if np.all(np.logical_and(x > self.bounds[0], x < self.bounds[1])) else False
