"""
Some simple cases for generating data for unit tests.

"""

import numpy as np
from invertH import invertHtrue, invertHsim

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



def generate_ball_drop_1(et,plot_design=False,R_new=None,R_design=None,C_design=None):
    n = 3; m = 25
    g = 9.8 # gravity
    C_true = .1 / (4 * np.pi / 3); print('generating data with C = ',C_true)
    n_field_heights = 4
    h_field = np.linspace(5,20,n_field_heights) # platform heights for the field experiments
    h_sim = np.arange(1.5,25,1.5) # grid of heights fed to the simulator
    h_dense = np.concatenate((np.arange(0,2,.01),np.arange(2,25,.5))) # a denser grid for drawing the curves

    # the coefficient of drag for a smooth sphere is 0.1, and we're
    # dividing by 4/3 pi to absorb a constant related to the volume of the
    # sphere (not including R)
    if R_new is None: R = np.array([.1, .2, .4]) # radii of balls to try (in meters)
    else: R = R_new

    # get a Latin hypercube sim_design of m=25 points over R_sim, C_sim
    #sim_design = pyDOE.lhs(2,m)

    # Use Kary's sim_designign for testing purposes
    sim_design = np.array([
        [0.1239,    0.8024],
        [0.8738,    0.6473],
        [0.6140,    0.3337],
        [0.8833,    0.4783],
        [0.9946,    0.0548],
        [0.1178,    0.9382],
        [0.1805,    0.2411],
        [0.6638,    0.2861],
        [0.2939,    0.1208],
        [0.2451,    0.2397],
        [0.4577,    0.5696],
        [0.4377,    0.8874],
        [0.0737,    0.7384],
        [0.6931,    0.8683],
        [0.4901,    0.7070],
        [0.5953,    0.9828],
        [0.7506,    0.1009],
        [0.7783,    0.4225],
        [0.8333,    0.5318],
        [0.3987,    0.6312],
        [0.2021,    0.4990],
        [0.3495,    0.3680],
        [0.9411,    0.7935],
        [0.0198,    0.0218],
        [0.5440,    0.1925]])
 
    # scale the first column to [0,.5] and call it R_sim
    # (this inclusim_design our field values, i.e., R \in [0,.5])
    # scale the second column to [0.05,.25] and call it Csim
    # (likewise, Ctrue \in [0.05, .25])
    sim_design[:,0] = sim_design[:,0] * .4 + .05
    sim_design[:,1] = sim_design[:,1] * .2 + .05
    if R_design is not None: R_sim = R_design
    else: R_sim = sim_design[:,0]
    if C_design is not None: C_sim = C_design
    else: C_sim = sim_design[:,1]
    if plot_design:
        plt.scatter(R_sim,C_sim)
        plt.xlabel("R design points");plt.ylabel("C design points")
        plt.title("Simulator Design")
        plt.show()

    # Generate field data for each R
    y_field       = invertHtrue(h_field, g, C_true, R, et) # observed times
    y_field_dense = invertHtrue(h_dense, g, C_true, R, et) # dense grid for plots

    # imagine that the biggest ball is too big to get to the highest
    # platform, so we don't observe data there
    #y_field[-1,-1] = np.nan

    # Generate simulated data for each (C,R) pair
    y_sim       = invertHsim(h_sim,   g, C_sim, R_sim)
    y_sim_dense = invertHsim(h_dense, g, C_sim, R_sim)
    
    data_dict = dict([('R',R),('sim_design',np.column_stack((R_sim,C_sim))),\
                      ('n',n),('m',m),('C_true',C_true),\
                      ('h_field',h_field),('h_sim',h_sim),('h_dense',h_dense),\
                      ('y_field',y_field),('y_field_dense',y_field_dense),\
                      ('y_sim',y_sim),('y_sim_dense',y_sim_dense)])
    
    return(data_dict)



