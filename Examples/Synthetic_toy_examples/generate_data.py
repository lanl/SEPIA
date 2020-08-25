"""
Some simple cases for generating data for unit tests, demos, etc.

"""

import numpy as np

def generate_univ_sim_and_obs(m=100, n=10, sig_n=0.1, seed=42):
    """
    Generate simple synthetic univariate-output simulation and observation data.

    :param m: scalar -- number of observations in simulation data
    :param n: scalar -- number of observations in observed data
    :param sig_n: scalar -- noise SD for iid observation noise
    :param seed: scalar -- random seed for replicability
    :return: dict -- y_sim, t_sim, y_obs, t_obs
    """
    np.random.seed(seed)

    # Sim data
    t = np.linspace(0, 1, m)[:, None]
    y = 2.5 * np.cos(10 * t)

    # Obs data
    t_obs = np.linspace(0, 1, n)[:, None]
    y_obs = 2.5 * np.cos(10 * t_obs) + sig_n * np.random.normal(size=(n, 1))

    return {'y_sim': y, 't_sim': t, 'y_obs': y_obs, 't_obs': t_obs}

def generate_multi_sim_and_obs(m=100, n=1, nt_sim=50, nt_obs=20, n_theta=3, n_basis=5, sig_n=0.1, seed=42):
    """
    Generate simple synthetic multivariate-output simulation and observation data.

    :param m: scalar -- number of observations in simulation data
    :param n: scalar -- number of observations in observed data
    :param nt_sim: scalar -- dimension of sim output
    :param nt_obs: scalar -- dimension of obs output
    :param n_theta: scalar -- number of thetas
    :param n_basis: scalar -- number of basis elements to use to generate data
    :param sig_n: scalar -- noise SD for iid observation noise
    :param seed: scalar -- random seed for replicability
    :return: dict -- y_sim, t_sim, y_obs, t_obs
    """
    np.random.seed(seed)

    # Sim data
    t_arrs = [np.linspace(0, 1, m) for i in range(n_theta)]
    t_grid = np.meshgrid(*t_arrs)
    t_grid = [tg.flatten() for tg in t_grid]
    t_idx = np.random.permutation(t_grid[0].shape[0])[:m]
    t_sim = np.array([tg[t_idx] for tg in t_grid]).T
    beta = np.abs(0.5 * np.random.uniform(size=n_theta)) + 0.05
    cov = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            cov[i, j] = np.exp(-np.sum(beta * np.square(t_sim[i, :] - t_sim[j, :])))

    chCov = np.linalg.cholesky(cov + 1e-6 * np.eye(m))
    wt_gen = np.dot(chCov, np.random.normal(size=(m, n_basis)))

    y_sim = np.zeros((m, nt_sim))
    y_ind_sim = np.linspace(0, 1, nt_sim)
    for i in range(n_basis):
        for j in range(m):
            y_sim[j, :] = y_sim[j, :] + wt_gen[j, i] * np.cos(np.pi * i * y_ind_sim)

    # Obs data
    t_idx = np.random.permutation(m)[:n]
    t_obs = t_sim[t_idx, :] + 0.01 * np.random.normal(size=(n, n_theta))
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = np.exp(-np.sum(beta * np.square(t_obs[i, :] - t_obs[j, :])))

    chCov = np.linalg.cholesky(cov + 1e-6 * np.eye(n))
    wt_gen = np.dot(chCov, np.random.normal(size=(n, n_basis)))

    y_obs = np.zeros((n, nt_obs))
    y_ind_obs = np.linspace(0, 1, nt_obs)
    for i in range(n_basis):
        for j in range(n):
            y_obs[j, :] = y_obs[j, :] + wt_gen[j, i] * np.cos(np.pi * i * y_ind_obs)
    y_obs = y_obs + sig_n * np.random.normal(size=(n, nt_obs))

    return {'y_sim': y_sim, 'y_ind_sim': y_ind_sim, 't_sim': t_sim,
            'y_obs': y_obs, 'y_ind_obs': y_ind_obs, 't_obs': t_obs}

if __name__ == "__main__":
    generate_multi_sim_and_obs()



