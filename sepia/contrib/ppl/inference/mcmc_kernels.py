import numpy as np
from abc import ABC, abstractmethod


def static_random_walk_step(state, proposal, logprob, rng=np.random):
    cand = proposal(state)
    log_accepctance_prob = logprob(cand) - logprob(state)
    accept = log_accepctance_prob > np.log(rng.uniform())
    new_state = cand if accept else state
    return new_state, accept


class MCMCKernel(ABC):
    def __init__(self, logprob, rng=np.random):
        self.logprob = logprob
        self.rng = rng
        self._accepts = 0

    @abstractmethod
    def step(self, current_state):
        pass


class MvRWM(MCMCKernel):
    def __init__(self, logprob, proposal, rng=np.random):
        super().__init__(logprob=logprob, rng=rng)
        self.proposal = proposal

    def step(self, current_state):
        new_state, accept = static_random_walk_step(
            proposal=self.proposal,
            state=current_state,
            logprob=self.logprob,
            rng=self.rng
        )
        self._accepts += accept
        return new_state, accept


class MvARWM(MvRWM):
    def __init__(self, logprob, rng=np.random, mean=None, cov=None, dim=None,
                 warmup_cov=None, window=[100, np.inf], multiplier=None,
                 beta=0.05):
        self.logprob = logprob
        self.rng = rng
        self._accepts = 0
        self.iter = 1
        self.total_iter = 1
        self.mean = mean
        self.cov = cov
        self.dim = dim
        self.beta = beta
        self.warmup = window[0]
        self.warmup_cov = warmup_cov
        self.window = list(window)
        self.multiplier = multiplier

    def init_stats(self, state):
        if self.mean is None:
            self.mean = state
            self.dim = self.mean.shape[0]
        if self.multiplier is None:
            self.multiplier = 2.38 ** 2 / self.dim
        if self.warmup_cov is None:
            self.warmup_cov = (0.01 / self.dim) * np.eye(self.dim)
        if self.cov is None:
            self.cov = self.warmup_cov + 0

    def update_iter(self):
        if self.total_iter == self.window[0]:
            self.iter = 1
            self._accepts = 0
            self.window.pop(0)

        self.iter += 1
        self.total_iter += 1

    def update_mean(self, state):
        self.mean += (state - self.mean) / self.iter

    def update_cov(self, state):
        d = state - self.mean
        self.cov = (
            self.cov * (self.iter - 1) / self.iter +
            np.outer(d, d) * (self.iter - 1) / self.iter ** 2
        )

    def update_stats(self, state):
        self.update_iter()
        self.update_mean(state)
        self.update_cov(state)

    def proposal(self, state):
        if self.mean is None:
            self.init_stats(state)

        self.update_stats(state)

        if self.beta > self.rng.uniform():
            # Occasionally, use the independent covariance.
            cov = self.warmup_cov
        else:
            # Most of the time, use the "optimal" covariance.
            cov = self.multiplier * self.cov

        return self.rng.multivariate_normal(mean=state, cov=cov)


