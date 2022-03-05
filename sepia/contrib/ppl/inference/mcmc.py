import numpy as np
from copy import copy
from ..util.pbar import pbrange
from .util import beautify_samples

class MCMC:
    def __init__(self, kernel, init_state, shaper=None, bijector=None):
        self.kernel = kernel
        self.state = init_state
        self.samples = []
        self.shaper = shaper
        self.bijector = bijector

    def fit(self, num_samples, burn, thinning=1, keep=None, callback=None,
            pb_args=dict()):
        total_iters = num_samples * thinning + burn
        with pbrange(total_iters, **pb_args) as pbar:
            for i in pbar:
                self.state, accept = self.kernel.step(current_state=self.state)
                if callback is not None:
                    callback(pbar=pbar, kernel=self.kernel, state=self.state,
                             accept=accept, total_iters=total_iters, burn=burn)
                if i + 1 > burn and (i + 1) % thinning == 0:
                    self.samples.append(copy(self.state))


        # Stack parameters, and put into a dictionary if possible.
        if self.shaper is None:
            return np.stack(self.samples)
        else:
            return beautify_samples(self.samples, self.shaper, self.bijector)
