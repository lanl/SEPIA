'''
 Gattiker
 
 Testing mean basis code execution (not correctness) 
 
 Plotting is implemented in the Synthetic_toy_examples/mean_basis_cal_pred.py
 counterpart program - the original of these functional tests that can
 be used to examine the response to evaluate correctness of function
 
'''

import unittest

import numpy as np, seaborn as sns, pandas as pd

from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaPredict import SepiaEmulatorPrediction, SepiaFullPrediction
import sepia.SepiaPlot as SP

import matplotlib.pyplot as plt

#
# These functions only pred here, plotting is in the original example file
#

def predPlot(m, title=''):
    sampset=m.get_samples(100,nburn=100)

    xpred=np.linspace(0,10,100).reshape((-1,1))
    pred=SepiaEmulatorPrediction(x_pred=xpred,model=m,samples=sampset)
    yhat=pred.get_y()
    
def predPlotCal(m, title=''):
    sampset=m.get_samples(100,nburn=100)

    tpred=np.linspace(0,10,100).reshape((-1,1))
    pred=SepiaEmulatorPrediction(t_pred=tpred,model=m,samples=sampset)
    yhat=pred.get_y()
    
def predPlotCalx(m, title=''):
    sampset=m.get_samples(100,nburn=100)

    tpred=np.linspace(0,10,100).reshape((-1,1))
    xpred=np.tile([2],tpred.shape)
    pred=SepiaFullPrediction(t_pred=tpred,x_pred=xpred,model=m,samples=sampset)
    yhat=pred.get_yobs()

def predPlotCalx2(m, title=''):
    sampset=m.get_samples(100,nburn=100)

    tpred=np.hstack((np.linspace(0,10,100).reshape((-1,1)),0.5*np.ones((100,1))))
    xpred=np.tile([2],(tpred.shape[0],1))
    pred=SepiaFullPrediction(t_pred=tpred,x_pred=xpred,model=m,samples=sampset)
    yhat=pred.get_yobs()


class TestMeanBasis(unittest.TestCase):

    def setUp(self):
        self.x  = np.array([[1,2,3]]).T
        self.y_lin = np.array([[-1,0,1]]).T
        self.y_nonlin = np.array([[-1,0.5,1]]).T
        self.yobs=np.array([[0.25]])
        self.yobs_sig=np.sqrt( np.array([[0.1]]) )
        
        self.t1 = np.array([[1,2,3,1,2,3,1,2,3]]).T
        self.x1 = np.array([[1,1,1,2,2,2,3,3,3]]).T
        self.y1 = np.array([[-1.5,0,0.5,-1,0.5,1,-0.5,1,1.5]]).T
        self.x1obs=np.array([[2]])
        self.y1obs=np.array([[0.25]])
        self.y1obs_sig=np.sqrt( np.array([[0.1]]) )
        
        self.t2 = np.array([[1,2,3,1,2,3,1,2,3],[3,2,1,3,2,1,3,2,1]]).T
        self.x2 = np.array([[1,1,1,2,2,2,3,3,3]]).T
        self.y2 = np.array([[-1.5,0,0.5,-1,0.5,1,-0.5,1,1.5]]).T
        self.x2obs=np.array([[1],[2],[3]])
        self.y2obs=np.array([[-1],[0.5],[1]])
        self.y2obs_sig=np.sqrt( np.array([[0.1]]) )

    def test_std_em(self):
        # Standard emulator
        data=SepiaData(x_sim=self.x, y_sim=self.y_nonlin)
        data.transform_xt()
        data.standardize_y()
        
        mod=SepiaModel(data)
        
        mod.do_mcmc(500)
        
        predPlot(mod, 'standard')

    def test_std_em_mb(self):
        # Emulator with mean basis
        dataH=SepiaData(x_sim=self.x, y_sim=self.y_nonlin)
        dataH.transform_xt()
        dataH.set_mean_basis('linear')
        dataH.standardize_y()
        
        modH=SepiaModel(dataH)
        
        modH.do_mcmc(500)
        
        predPlot(modH, title = 'linear basis')
        
    def test_cal_dummyx(self):
        # set up calibration, dummy-x
        
        datac=SepiaData(t_sim=self.x, y_sim=self.y_nonlin, y_obs=self.yobs, Sigy=self.yobs_sig)
        datac.transform_xt()
        datac.standardize_y()
        
        modc=SepiaModel(datac)
        
        modc.do_mcmc(500)
        
        predPlotCal(modc, title = 'dummy-x')

    def test_cal_dummyx_mb(self):
        #% calibration, dummy-x, mean basis
        
        datacH=SepiaData(t_sim=self.x, y_sim=self.y_nonlin, y_obs=self.yobs, Sigy=self.yobs_sig)
        datacH.transform_xt()
        datacH.set_mean_basis('linear')
        datacH.standardize_y()
        
        modcH=SepiaModel(datacH)
        
        modcH.do_mcmc(500)
        
        predPlotCal(modcH, title = 'linear basis, dummy-x')

    def test_cal_x(self):
        #% set up calibration, with x
        
        datacx=SepiaData(t_sim=self.t1, x_sim=self.x1, y_sim=self.y1, 
                         x_obs=self.x1obs, y_obs=self.y1obs, Sigy=self.y1obs_sig)
        datacx.transform_xt()
        datacx.standardize_y()
        
        modcx=SepiaModel(datacx)
        
        modcx.do_mcmc(500)
        
        predPlotCalx(modcx, title = 'with x')

    def test_cal_x_mb(self):
        #% set up calibration, with x, mean basis
        
        datacxH=SepiaData(t_sim=self.t1, x_sim=self.x1, y_sim=self.y1, 
                          x_obs=self.x1obs, y_obs=self.y1obs, Sigy=self.y1obs_sig)
        datacxH.transform_xt()
        datacxH.set_mean_basis('linear')
        datacxH.standardize_y()
        
        modcxH=SepiaModel(datacxH)
        
        modcxH.do_mcmc(500)
        
        predPlotCalx(modcxH, title = 'linear basis, with x')
        
    def test_cal_x_aug(self):
        #% set up calibration, with 2Dt multi-x
        
        datacx2=SepiaData(t_sim=self.t2, x_sim=self.x2, y_sim=self.y2, 
                          x_obs=self.x2obs, y_obs=self.y2obs, Sigy=self.y2obs_sig)
        datacx2.transform_xt()
        datacx2.standardize_y()
        
        modcx2=SepiaModel(datacx2)
        
        modcx2.do_mcmc(500)
        
        predPlotCalx2(modcx2, title = 'with 2Dt multi-x')
        
    def test_cal_x_mb_aug(self):
        #% set up calibration, with 2Dt multi-x, mean basis
        
        datacxH2=SepiaData(t_sim=self.t2, x_sim=self.x2, y_sim=self.y2, 
                           x_obs=self.x2obs, y_obs=self.y2obs, Sigy=self.y2obs_sig)
        datacxH2.transform_xt()
        datacxH2.set_mean_basis('linear')
        datacxH2.standardize_y()
        
        modcxH2=SepiaModel(datacxH2)
        
        modcxH2.do_mcmc(500)
        
        predPlotCalx2(modcxH2, title = 'linear basis, with 2Dt multi-x')
