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


def gaussian_kernel(X, knots, sd):
    """
    X: points to evaluate kernel (n x q)
    knots: kernel locations (m x q)
    sd: kernel width

	Return: an nxm matrix of Gaussian density evaluations at X with centers at
	knots and provided sd.
    """
    diff = X[..., None] - knots.T[None, ...]  # n x q x m
    ss = np.sum(diff ** 2, axis=1)  # n x m
    v = sd ** 2
    return np.exp(-ss / (2 * v)) / np.sqrt(2 * np.pi) / sd
