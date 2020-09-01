
import numpy as np


class SepiaMCMC:
    """
    MCMC object to store step type, step parameters, draw candidates, accept/reject, and record draws.

    :var stepType: string -- MCMC step type in 'Normal', 'PropMH', 'Uniform', 'BetaRho'
    :var stepParam: ndarray -- step size params, or None if using 'recorder'
    :var parent: SepiaParam object -- which parameter this MCMC object corresponds to
    :var draws: list -- list of MCMC draws
    :var aCorr: for MH correction
    """

    def __init__(self, parent, stepType='Normal', stepParam=0.1):
        """
        Initialize MCMC object.

        :param parent: SepiaParam object that this MCMC belongs to
        :param stepType: 'Normal', 'PropMH', 'Uniform', or 'BetaRho'
        :param stepParam: nparray -- step size parameter for each element of matrix-valued parameter
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

        :param arr_ind: array index for variable of interest within parent SepiaParam
        :param do_propMH: flag to do propMH for variables where stepType = 'propMH'
        :return: scalar candidate value
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
            raise Exception('Unknown stepType')
        return cand

    def reject(self, ind, model):
        """
        Reject candidate by restoring model to previous state.

        :param ind: index of variable
        :param model: model object containing variable
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

    def record(self, x=False):
        """
        Record value into MCMC draws.

        :param x: optionally, a value to record; otherwise, current parent.val is used.
        """
        if x is False:
            x = self.parent.val.copy()
        self.draws.append(x)
