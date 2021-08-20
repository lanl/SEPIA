from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData as SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaPredict import SepiaFullPrediction

import numpy as np
from scipy.stats import norm
import unittest
import os
class SepiaBallDrop2TestCase(unittest.TestCase):
    def test_ball_drop_2(self):        
        datadir='examples/Ball_Drop/data/data_ball_drop_2/' # relative path from test directory
        # experimental data
        with open(datadir+'fieldDat12x6gparam.txt','r') as f:
            field_data = np.loadtxt(f)
        with open(datadir+'desNative4x2Rrho.txt','r') as f:
            x = np.loadtxt(f) # x = {R, rho_ball}

        x_obs = x
        y_obs = np.reshape(field_data[:,4],(4,3)) # observed times, experiments by row
        h_obs = field_data[0:3,3]                 # observed heights

        # simulated data
        with open(datadir+'desNative80x4Cg.txt','r') as f:
            sim_data = np.loadtxt(f)
        x_sim = sim_data[:,0:2] # x = {R, rho_ball}
        t_sim = sim_data[:,2:4] # t = {C, g}
        with open(datadir+'simHeights101x1','r') as f:
            h_sim = np.loadtxt(f)
        with open(datadir+'sims101x80Cg.txt','r') as f:
            y_sim = np.loadtxt(f).T

        # create sepia data object
        data = SepiaData(x_sim = x_sim, t_sim = t_sim, y_ind_sim = h_sim, y_sim = y_sim, 
                         x_obs = x, y_obs = y_obs, y_ind_obs = h_obs)
        data.transform_xt()
        data.standardize_y()

        data.create_K_basis(3)

        # Generate D matrix with normal kernels
        n_kernels = 20
        D_grid = np.linspace(min(h_sim),max(h_sim),n_kernels,dtype=int) # locations on which the kernels are centered
        D_width = max(D_grid)/n_kernels  # width of each kernel
        pv = len(D_grid)
        D_obs = np.zeros(shape=(data.obs_data.y_ind.shape[0],pv))
        D_sim = np.zeros(shape=(data.sim_data.y_ind.shape[0],pv))

        for j in range(pv):
            D_obs[:,j] = norm.pdf(h_obs, D_grid[j], D_width)
            D_sim[:,j] = norm.pdf(h_sim, D_grid[j],D_width)

        data.create_D_basis(D_obs=D_obs.T,D_sim=D_sim.T)

        model = SepiaModel(data)

        model.tune_step_sizes(10, 4)
        model.do_mcmc(50)

        pred_samples=model.get_samples(effectivesamples=True)
        pred=SepiaEmulatorPrediction(x_pred=x, samples=pred_samples, model=model)
        predw=pred.get_w()
        predystd=pred.get_y(std=True)
        predy=pred.get_y()

        # And below are similar predictions for the full models responses joint u and v
        predfull=SepiaFullPrediction(x_pred=x, samples=pred_samples, model=model)

        predysimstd=predfull.get_ysim(std=True)
        preddstd=predfull.get_discrepancy(std=True,as_obs=True)
        predd=predfull.get_discrepancy(as_obs=True)
        predyobs=predfull.get_yobs(as_obs=True)
