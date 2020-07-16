import numpy as np
import pickle

from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model

# Load raw data
sim_s104 = np.genfromtxt('data/features_cdf104S.csv', delimiter=',', skip_header=1)
sim_s105 = np.genfromtxt('data/features_cdf105S.csv', delimiter=',', skip_header=1)
sim_s106 = np.genfromtxt('data/features_cdf106S.csv', delimiter=',', skip_header=1)

obs_s104 = np.genfromtxt('data/features_cdf_obs104S.csv', delimiter=',', skip_header=1)
obs_s105 = np.genfromtxt('data/features_cdf_obs105S.csv', delimiter=',', skip_header=1)
obs_s106 = np.genfromtxt('data/features_cdf_obs106S.csv', delimiter=',', skip_header=1)

design = np.loadtxt('data/Al.trial5.design.txt', skiprows=1)
with open('data/Al.trial5.design.txt', 'r') as f:
    design_names = str.split(f.readline())

# Preprocess data
sel_features = [3, 5, 7, 9]
y_sim = 10000 * np.concatenate([sim_s104[:, sel_features], sim_s105[:, sel_features], sim_s106[:, sel_features]], axis=1)
y_obs = np.concatenate([obs_s104[sel_features], obs_s105[sel_features], obs_s106[sel_features]])[None, :]
n_features = y_obs.shape[1]
y_ind = np.arange(1, n_features+1)

# Set up sepia model
data = SepiaData(t_sim=design, y_sim=y_sim, y_ind_sim=y_ind, y_obs=y_obs, y_ind_obs=y_ind)
data.standardize_y()
data.transform_xt()
data.create_K_basis(n_features)
print(data)

# Setup model
# We have a known observation error
Sigy = np.diag(np.squeeze((0.01 * np.ones(n_features) * y_obs)/data.sim_data.y_sd**2))
model = setup_model(data, Sigy)

# Do mcmc
model.do_mcmc(5000)
samples_dict = {p.name: p.mcmc_to_array(untransform_theta=True) for p in model.params.mcmcList}

with open('data/sepia_mcmc_samples1-5000.pkl', 'wb') as f:
    pickle.dump(samples_dict, f)

with open('data/sepia_model.pkl', 'wb') as f:
    pickle.dump(model, f)