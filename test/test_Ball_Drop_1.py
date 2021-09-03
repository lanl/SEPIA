from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData as SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaPredict import SepiaFullPrediction
from generate_data import generate_ball_drop_1

import numpy as np
from scipy.stats import norm
import unittest

class SepiaBallDrop1TestCase(unittest.TestCase):
    def test_ball_drop_1(self):
        et = 0.01                      # observation error
        data_dict = generate_ball_drop_1(et)

        # field data
        R = data_dict['R']             # radii of balls   .1,.2,.4 (m)
        h_field = data_dict['h_field'] # observed heights 5,10,15,20 (m)
        y_field = data_dict['y_field'] # observed times

        # sim data
        sim_design = data_dict['sim_design']

        R_sim = sim_design[:,0]
        C_sim = sim_design[:,1]
        h_sim = data_dict['h_sim']
        y_sim = data_dict['y_sim']

        data = SepiaData(x_sim = np.reshape(R_sim,(len(R_sim),1)),
                         t_sim = np.reshape(C_sim,(len(C_sim),1)),
                         y_sim = y_sim, y_ind_sim = h_sim,
                         x_obs = np.reshape(R,(len(R),1)), y_obs = y_field, y_ind_obs=h_field)
        data.transform_xt()
        data.standardize_y()
        data.create_K_basis(2)

        # Generate D matrix with normal kernels
        D_grid = h_sim # locations on which the kernels are centered
        D_width = 1.5  # width of each kernel
        pv = len(D_grid)
        D_obs = np.zeros(shape=(data.obs_data.y_ind.shape[0],pv))
        D_sim = np.zeros(shape=(data.sim_data.y_ind.shape[0],pv))
        h_dense = data_dict['h_dense']
        D_dense = np.zeros(shape=(h_dense.shape[0],pv))

        for j in range(pv):
            D_obs[:,j] = norm.pdf(h_field, D_grid[j], D_width)
            D_sim[:,j] = norm.pdf(h_sim, D_grid[j],D_width)
            D_dense[:,j] = norm.pdf(h_dense, D_grid[j],D_width)

        data.create_D_basis(D_obs=D_obs.T,D_sim=D_sim)

        # Model initialization and MCMC
        model = SepiaModel(data)
        model.tune_step_sizes(10, 4)
        model.do_mcmc(50)

        # ----
        # ## Predictions of model output from the full model
        #

        x_pred=data.obs_data.x
        n_pred=x_pred.shape[0]
        pred_samples=model.get_samples(effectivesamples=True)
            
        pred=SepiaEmulatorPrediction(x_pred=x_pred, samples=pred_samples, model=model)#, t_pred=t_pred)
        
        predw=pred.get_w()
        predystd=pred.get_y(std=True)
        predy=pred.get_y()

        pred=SepiaFullPrediction(x_pred=data.obs_data.x, samples=pred_samples, model=model)

        predu, predv = pred.get_u_v()
        predysimstd=pred.get_ysim(std=True)
        preddstd=pred.get_discrepancy(as_obs=True,std=True)
        predd=pred.get_discrepancy(as_obs=True)
        predyobs=pred.get_yobs(as_obs=True)
        predyobs=pred.get_yobs(as_obs=True)
        predysim=pred.get_ysim()