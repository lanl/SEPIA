
import numpy as np


class SepiaMCMC:
    """
    MCMC object to store step type, step parameters, draw candidates, accept/reject, and record draws.

    :var string stepType: MCMC step type in ['PropMH', 'Uniform', 'BetaRho', 'Recorder']
    :var numpy.ndarray/NoneType stepParam: step size params with shape param.val_shape, or None if using 'Recorder'
    :var sepia.SepiaParam parent: instantiated sepia.SepiaParam parameter object that this MCMC object corresponds to
    :var list draws: list of MCMC draws (typically list of arrays, each of shape param.val_shape)
    :var float aCorr: for MH correction
    """

    def __init__(self, parent, stepType='Uniform', stepParam=0.1):
        """
        Initialize MCMC object.

        :param sepia.SepiaParam parent: instantiated sepia.SepiaParam parameter object that this MCMC object corresponds to
        :param string stepType: MCMC step type in 'PropMH', 'Uniform', 'BetaRho', 'Recorder'
        :param numpy.ndarray/NoneType stepParam: step size params with shape param.val_shape, or None if using 'Recorder'

        .. note:: Typically not directly instantiated, but created by `sepia.SepiaModel` constructor.
        """
        self.stepType = stepType
        self.parent = parent
        if self.stepType == 'Recorder':
            self.draws = []
            self.stepParam = None
        else:
            self.stepParam = stepParam
            self.draws = []
            self.aCorr = 1 # placeholder for MH correction

    def draw_candidate(self, arr_ind, do_propMH):
        """
        Draw MCMC candidate for single variable/index.

        :param numpy.ndarray arr_ind: array index for variable of interest within parent sepia.SepiaParam object values
        :param bool do_propMH: do propMH for variables where stepType = 'propMH'?
        :return: scalar candidate value
        :raises ValueError: if stepType is not one of the supported values
        """
        self.aCorr = 1
        if self.stepType == 'Uniform':
            cand = self.parent.val[arr_ind] + self.stepParam[arr_ind] * np.random.uniform(-0.5, 0.5)
        elif self.stepType == 'BetaRho':
            cand = np.exp(-0.25 * self.parent.val[arr_ind]) + self.stepParam[arr_ind] * np.random.uniform(-0.5, 0.5)
            if cand <= 0:
                cand = np.inf
            else:
                cand = -4 * np.log(cand)
        elif self.stepType == 'PropMH':
            if do_propMH:
                cval = self.parent.val[arr_ind]
                w = np.max([1, cval/3])
                dval = cval + w * np.random.uniform(-1, 1)
                w1 = np.max([1, dval/3])
                if cval > (dval+w1):
                    aCorr = False # never will accept in this case
                else:
                    aCorr = w/w1
                cand = dval
                self.aCorr = aCorr
            else:
                cand = self.parent.val[arr_ind] + self.stepParam[arr_ind] * np.random.uniform(-0.5, 0.5)
        else:
            raise ValueError('Unknown stepType')
        return cand

    def reject(self, ind, model):
        """
        Reject candidate by restoring model to previous state.

        :param int ind: scalar (flattened) index of variable
        :param sepia.sepiaModel model: sepia.SepiaModel object containing variable
        """
        # If reject, need to put back refVal into val
        arr_ind = np.unravel_index(ind, self.parent.val_shape, order='F')
        self.parent.val[arr_ind] = self.parent.refVal[arr_ind]
        model.num.restore_ref(model.refNum)

    def accept(self):
        """
        Accept candidate.

        """
        # If accept, new value is already part of val so leave it alone
        pass

    def record(self, x=None):
        """
        Record value into MCMC draws.

        :param numpy.ndarray/NoneType x: optionally, a value to record; otherwise, current parent.val is used.
        """
        if x is None:
            x = self.parent.val.copy()
        self.draws.append(x)
