import numpy as np
from tqdm import trange
from .util import logsumexp

def concat_list_dict(list_dict):
    out = dict()
    names = list_dict[0].keys()
    for name in names:
        out[name] = np.stack([d[name] for d in list_dict])
    return out


class ImportanceSampling:
    def __init__(self, model):
        self.model = model
        self.particles = []
        self.logprobs = []

    def reset(self):
        self.particles = []
        self.logprobs = []

    # TODO: Preallocate memory?
    def fit(self, num_particles, keep=None, callback=None,
            tqdm_args=dict(), *args, **kwargs):

        with trange(num_particles, **tqdm_args) as pbar:
            for i in pbar:
                particle = self.model.prior_predictive(*args, **kwargs)
                self.particles.append(particle)

                logprob = self.model.logpdf(particle, *args, **kwargs)
                self.logprobs.append(logprob)

                if callback is not None:
                    callback(pbar=pbar, i=i)

    def sample(self, num_samples, rng=np.random, pretty=True, temperature=1):
        lp = np.array(self.logprobs) * temperature
        probs = np.exp(lp - logsumexp(lp))
        samples = rng.choice(self.particles, num_samples, p=probs)
        if pretty:
            return concat_list_dict(samples)
        else:
            return samples
