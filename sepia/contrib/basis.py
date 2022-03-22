import numpy as np

def radial(X, knots, sd):
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

def constant(x, knots):
    return np.ones([x.shape[0], knots.shape[0]])

def window(x, knots, width):
    diff = x - knots.T
    return np.abs(diff) < width

def hinge(x, knots):
    return np.maximum(x - knots.T, 0)