'''
Gattiker:
    Mean basis variations. 
    Migrated this to test_mean_basis.py also
    
    Observe that this is a code blocks rather than a stand-alone script
    
'''

import numpy as np, seaborn as sns, pandas as pd

from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaPredict import SepiaEmulatorPrediction, SepiaFullPrediction
import sepia.SepiaPlot as SP

import matplotlib.pyplot as plt

#% set up data for 3 point emulator, functions

x  = np.array([[1,2,3]]).T
y_lin = np.array([[-1,0,1]]).T
y_nonlin = np.array([[-1,0.5,1]]).T
yobs=np.array([[0.25]])
yobs_sig=np.sqrt( np.array([[0.1]]) )

t1 = np.array([[1,2,3,1,2,3,1,2,3]]).T
x1 = np.array([[1,1,1,2,2,2,3,3,3]]).T
y1 = np.array([[-1.5,0,0.5,-1,0.5,1,-0.5,1,1.5]]).T
x1obs=np.array([[2]])
y1obs=np.array([[0.25]])
y1obs_sig=np.sqrt( np.array([[0.1]]) )

t2 = np.array([[1,2,3,1,2,3,1,2,3],[3,2,1,3,2,1,3,2,1]]).T
x2 = np.array([[1,1,1,2,2,2,3,3,3]]).T
y2 = np.array([[-1.5,0,0.5,-1,0.5,1,-0.5,1,1.5]]).T
x2obs=np.array([[1],[2],[3]])
y2obs=np.array([[-1],[0.5],[1]])
y2obs_sig=np.sqrt( np.array([[0.1]]) )

def predPlot(m, title=''):
    sampset=m.get_samples(500,nburn=100)

    xpred=np.linspace(0,10,100).reshape((-1,1))
    pred=SepiaEmulatorPrediction(x_pred=xpred,model=m,samples=sampset)
    yhat=pred.get_y()
    yhm=np.mean(yhat,axis=0)
    yhs=np.std(yhat,axis=0)
    
    #plt.figure(SP.mcmc_trace(m.get_samples()))
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(xpred,yhat.squeeze().T )
    plt.subplot(2,1,2)
    plt.plot(xpred,np.mean(yhat,axis=0))
    plt.plot(xpred,yhm,'k')
    plt.plot(xpred,yhm-yhs,'k:')
    plt.plot(xpred,yhm+yhs,'k:')
    plt.suptitle('Em-only Model Pred y: ' + title)

def predPlotCal(m, title=''):
    sampset=m.get_samples(500,nburn=100)

    tpred=np.linspace(0,10,100).reshape((-1,1))
    pred=SepiaEmulatorPrediction(t_pred=tpred,model=m,samples=sampset)
    yhat=pred.get_y()
    yhm=np.mean(yhat,axis=0)
    yhs=np.std(yhat,axis=0)
    
    #plt.figure(SP.mcmc_trace(m.get_samples()))
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tpred,yhat.squeeze().T )
    plt.subplot(2,1,2)
    plt.plot(tpred,np.mean(yhat,axis=0))
    plt.plot(tpred,yhm,'k')
    plt.plot(tpred,yhm-yhs,'k:')
    plt.plot(tpred,yhm+yhs,'k:')
    plt.suptitle('Cal Model Emulator Pred y: ' + title)

def predPlotCalx(m, title=''):
    sampset=m.get_samples(500,nburn=100)

    tpred=np.linspace(0,10,100).reshape((-1,1))
    xpred=np.tile([2],tpred.shape)
    pred=SepiaFullPrediction(t_pred=tpred,x_pred=xpred,model=m,samples=sampset)
    yhat=pred.get_yobs()
    yhm=np.mean(yhat,axis=0)
    yhs=np.std(yhat,axis=0)
    
    #plt.figure(SP.mcmc_trace(m.get_samples()))
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tpred,yhat.squeeze().T )
    plt.subplot(2,1,2)
    plt.plot(tpred,np.mean(yhat,axis=0))
    plt.plot(tpred,yhm,'k')
    plt.plot(tpred,yhm-yhs,'k:')
    plt.plot(tpred,yhm+yhs,'k:')
    plt.suptitle('Cal Model Full Pred ysim: ' + title)

def predPlotCalx2(m, title=''):
    sampset=m.get_samples(500,nburn=100)

    tpred=np.hstack((np.linspace(0,10,100).reshape((-1,1)),0.5*np.ones((100,1))))
    xpred=np.tile([2],(tpred.shape[0],1))
    pred=SepiaFullPrediction(t_pred=tpred,x_pred=xpred,model=m,samples=sampset)
    yhat=pred.get_yobs()
    yhm=np.mean(yhat,axis=0)
    yhs=np.std(yhat,axis=0)
    
    #plt.figure(SP.mcmc_trace(m.get_samples()))
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tpred[:,0],yhat.squeeze().T )
    plt.subplot(2,1,2)
    plt.plot(tpred[:,0],np.mean(yhat,axis=0))
    plt.plot(tpred[:,0],yhm,'k')
    plt.plot(tpred[:,0],yhm-yhs,'k:')
    plt.plot(tpred[:,0],yhm+yhs,'k:')
    plt.suptitle('Cal Model Full Pred ysim: ' + title)

#%% Standard emulator
data=SepiaData(x_sim=x, y_sim=y_nonlin)
data.transform_xt()
data.standardize_y()

mod=SepiaModel(data)

mod.do_mcmc(1000)

predPlot(mod, 'standard')


#%% Emulator with mean basis

dataH=SepiaData(x_sim=x, y_sim=y_nonlin)
dataH.transform_xt()
dataH.set_mean_basis('linear')
dataH.standardize_y()

modH=SepiaModel(dataH)

modH.do_mcmc(1000)

predPlot(modH, title = 'linear basis')

#psamp=modH.get_samples(nburn=100)
#sns.pairplot(pd.DataFrame(psamp['gamma']))

#%% set up calibration, dummy-x

datac=SepiaData(t_sim=x, y_sim=y_nonlin, y_obs=yobs, Sigy=yobs_sig)
datac.transform_xt()
datac.standardize_y()

modc=SepiaModel(datac)

modc.do_mcmc(1000)

predPlotCal(modc, title = 'dummy-x')

#%% calibration, dummy-x, mean basis

datacH=SepiaData(t_sim=x, y_sim=y_nonlin, y_obs=yobs, Sigy=yobs_sig)
datacH.transform_xt()
datacH.set_mean_basis('linear')
datacH.standardize_y()

modcH=SepiaModel(datacH)

modcH.do_mcmc(1000)

predPlotCal(modcH, title = 'linear basis, dummy-x')

#%% set up calibration, with x

datacx=SepiaData(t_sim=t1, x_sim=x1, y_sim=y1, x_obs=x1obs, y_obs=y1obs, Sigy=y1obs_sig)
datacx.transform_xt()
datacx.standardize_y()

modcx=SepiaModel(datacx)

modcx.do_mcmc(1000)

predPlotCalx(modcx, title = 'with x')

#%% set up calibration, with x, mean basis

datacxH=SepiaData(t_sim=t1, x_sim=x1, y_sim=y1, x_obs=x1obs, y_obs=y1obs, Sigy=y1obs_sig)
datacxH.transform_xt()
datacxH.set_mean_basis('linear')
datacxH.standardize_y()

modcxH=SepiaModel(datacxH)

modcxH.do_mcmc(1000)

predPlotCalx(modcxH, title = 'linear basis, with x')

#%% set up calibration, with 2Dt multi-x

datacx2=SepiaData(t_sim=t2, x_sim=x2, y_sim=y2, x_obs=x2obs, y_obs=y2obs, Sigy=y2obs_sig)
datacx2.transform_xt()
datacx2.standardize_y()

modcx2=SepiaModel(datacx2)

modcx2.do_mcmc(1000)

predPlotCalx2(modcx2, title = 'with 2Dt multi-x')

#%% set up calibration, with 2Dt multi-x, mean basis

datacxH2=SepiaData(t_sim=t2, x_sim=x2, y_sim=y2, x_obs=x2obs, y_obs=y2obs, Sigy=y2obs_sig)
datacxH2.transform_xt()
datacxH2.set_mean_basis('linear')
datacxH2.standardize_y()

modcxH2=SepiaModel(datacxH2)

modcxH2.do_mcmc(1000)

predPlotCalx2(modcxH2, title = 'linear basis, with 2Dt multi-x')
