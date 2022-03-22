import numpy as np

def symmetric(X):
    """
    Converts an almost-symmetric (up to numerical rounding) matrix to a
    symmetric matrix.
    """
    return np.maximum(X, X.transpose())

def beautify_samples(samples, shaper, bijector=None):
    samples_dict = dict()
    param_names = shaper.shapes.keys()
    states = [shaper.unvec(s) for s in samples]
    if bijector is not None:
        states = [bijector(s) for s in states]
    for name in param_names:
        samples_dict[name] = np.stack([s[name] for s in states])
    return samples_dict

def normalize(probs, axis=-1):
    assert np.all(probs >= 0), "probs need to be non-negative!"
    return probs / np.sum(probs, axis=axis)[..., None]

def logsumexp(x):
    max = np.max(x)
    return max + np.log(np.exp(x - max).sum())

def get_num_samples(samples: dict):
    first_param = list(samples.keys())[0]
    return samples[first_param].shape[0]

def tolist(samples: dict):
    """
    Convert a dictionary of samples (each entry being an n-d array) into a list
    of dictionaries.
    """
    n = get_num_samples(samples)
    return [
        {k: v[i] for k, v in samples.items()}
        for i in range(n)
    ]

def trim(samples: dict, burn=0, thin=1):
    n = get_num_samples(samples)
    if 0 < burn < 1:
        burn = int(n * burn)
    return {k: v[burn:n:thin] for k, v in samples.items()}
