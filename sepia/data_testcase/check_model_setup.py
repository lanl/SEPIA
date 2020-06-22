# Check data and model setup for real data example

import numpy as np
import scipy.io
import scipy.stats

np.random.seed(42)

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model


matfile = scipy.io.loadmat('data/alal_data.mat')

nmcmc = matfile['nmcmc'].item()
lamWs_a = matfile['lamWs_a'].item()
lamWs_b = matfile['lamWs_b'].item()
lamWs_upper = matfile['lamWs_upper'].item()
n_pc = matfile['pu'].item()
y_sim = matfile['y_sim'].T
y_ind_sim = np.linspace(0, 1, y_sim.shape[1])
xt_sim = matfile['xt_sim']
y_obs = matfile['y_obs'].T
y_ind_obs = y_ind_sim
x_obs = matfile['x_obs']
data = SepiaData(x_sim=xt_sim[:, 0][:, None], t_sim=xt_sim[:, 1:], y_sim=y_sim, y_ind_sim=y_ind_sim,
                 x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)
data.standardize_y()
data.transform_xt()
data.create_K_basis(n_pc=n_pc)
data.create_D_basis(D=matfile['Dobs'].T)
print(data)

import matplotlib.pyplot as plt
plt.plot(data.sim_data.y_std.T)
plt.show()

model = setup_model(data, Sigy=matfile['Sigy'])
model.params.lamWs.prior.bounds[1] = lamWs_upper
model.params.lamWs.prior.params = [lamWs_a, lamWs_b]