import numpy as np

def acf(x, lags=None):
    if lags is None:
        lags = range(x.shape[0])
    return np.array([autocorr(x, lag) for lag in lags])

def variogram(x, lag):
    lag = np.abs(lag)
    if lag > 0:
        a, b = x[:-lag], x[lag:]
    else:
        a = b = x
    return np.mean((a - b) ** 2)

def autocorr(x, lag):
    return 1 - variogram(x, lag) / (2 * np.var(x))

def idx_of_first_consequtive_negatives(xs, num_conseq):
    num_conseq_seen = 0
    for i, x in enumerate(xs):
        if x < 0:
            num_conseq_seen += 1
        else:
            num_conseq_seen = 0
        if num_conseq_seen == num_conseq:
            break
    return i - num_conseq + 1

def ess(x):
    n = x.shape[0]
    rho = acf(x, range(1, n))
    idx = idx_of_first_consequtive_negatives(rho, 3)
    return np.minimum(n, n / (1 + 2 * rho[:idx].sum()))

# Another way to define autocorr is `autocov(x, lag) / autocov(x, 0)`.
def autocov(x, lag):
    m = np.mean(x)
    lag = np.abs(lag)
    if lag > 0:
        a, b = x[:-lag], x[lag:]
    else:
        a, b = x, x
    return np.sum((a - m) * (b - m)) / x.shape[0]

def ess_kish(w, log=True):
    """
    Kish's effective sample size. Used for weighted samples. (e.g. importance
    sampling, sequential monte carlo, particle filters.)

    https://en.wikipedia.org/wiki/Effective_sample_size

    If log is True, then the w are log weights. 
    """
    if log:
        return ess_kish(np.exp(w - np.max(w)), log=False)
    else:
        return sum(w) ** 2 / sum(w ** 2)
