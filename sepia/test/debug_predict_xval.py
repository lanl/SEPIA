import numpy as np

from sepia.Examples import generate_data
from sepia.SepiaData import SepiaData
from sepia.SepiaModelSetup import setup_model
from sepia.SepiaPredict import SepiaXvalEmulatorPrediction, SepiaEmulatorPrediction

multi_data_dict = generate_data.generate_multi_sim_and_obs(m=100, n=1)

d = SepiaData(t_sim=multi_data_dict['t_sim'], y_sim=multi_data_dict['y_sim'],
              y_ind_sim=multi_data_dict['y_ind_sim'])
d.transform_xt()
d.standardize_y()
d.create_K_basis(5)
model = setup_model(d)

model.do_mcmc(100)
samples_dict = {p.name: p.mcmc_to_array() for p in model.params.mcmcList}

pred_samples = model.get_samples(numsamples=7)
x_pred = np.linspace(0,1,9).reshape((9,1))
t_pred=np.tile(np.array([1,0,1]).reshape(1,3),(9,1))
pred = SepiaEmulatorPrediction(x_pred=x_pred, samples=pred_samples, model=model, t_pred=t_pred)
CVpred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model, x_pred=model.data.sim_data.x_trans, t_pred=model.data.sim_data.t_trans)
w_pred = CVpred.get_w()
y_pred_std = CVpred.get_y_standardized()
y_pred_nat = CVpred.get_y_native()