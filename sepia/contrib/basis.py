import numpy as np

def diff(X, Y):
    """
    X: matrix (n x q)
    Y: matrix (m x q)
    return: matrix of differences (n x q x m)
    """
    return X[..., None] - Y.T[None, ...]

def sqdiff(X, Y):
    """Squared differences."""
    d = diff(X, Y)  # n x q x m
    return np.sum(d ** 2, axis=1)  # n x m

def absdiff(X, Y):
    """Absolute differences."""
    d = diff(X, Y)  # n x q x m
    return np.sum(np.abs(d), axis=1)  # n x m

def euclidean(X, Y):
    """Euclidean distance."""
    return np.sqrt(sqdiff(X, Y))  # n x m

def radial(X, knots, sd):
    """
    X: points to evaluate kernel (n x q)
    knots: kernel locations (m x q)
    sd: kernel width

	Return: an nxm matrix of Gaussian density evaluations at X with centers at
	knots and provided sd.
    """
    ss = sqdiff(X, knots)
    v = sd ** 2
    return np.exp(-ss / (2 * v)) / np.sqrt(2 * np.pi) / sd

def constant(X, knots):
    return np.ones([X.shape[0], knots.shape[0]])

def window(X, knots, width):
    return euclidean(X, knots) < width

def hinge(X, knots):
    assert X.shape[1] == knots.shape[1] == 1, "hinge is only supported for column vectors."
    return np.maximum(X - knots.T, 0)