from nedderimp import nedderimp
from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData as SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaPredict import SepiaFullPrediction
import numpy as np
from nedderimp import nedderimp
from scipy.interpolate import interp2d
from copy import deepcopy
import unittest


def dist2pi(x1,x2):
    # computes the distance assuming periodicity: 2pi=0
    # x1 and x2 are vectors with common length and values
    # between 0 and 2pi
    d = abs(x1-x2)
    iwrap = d > np.pi
    d[iwrap] = 2*np.pi - d[iwrap]
    return(d)
def dnorm(x,mu,scale):  
    # normal density in 1-d. 
    # It is scaled so that the 1-d integral is 1
    # mu and scale are scalars, x is an array...
    out=np.zeros(len(x))
    u=abs(x-mu)/scale
    out = (1.0/(np.sqrt(2*np.pi)*scale)) * np.exp(-.5 * u**2)
    return(out)

class SepiaNeddermeyerTestCase(unittest.TestCase):
    def test_neddermeyer(self):
        # simulation design
        design = np.array([
            [0.7714,    0.4286,    0.0286],
            [0.3714,    0.1143,    0.7143],
            [0.1714,    0.4571,    0.8857],
            [0.3429,    0.6000,    0.8000],
            [0.8000,    0.6286,    0.4000],
            [0.7429,    0.5429,         0],
            [0.6571,    1.0000,    0.6286],
            [0.2857,         0,    0.4571],
            [0.5143,    0.9429,    0.2286],
            [0.6857,    0.3143,    0.6571],
            [0.8286,    0.2000,    1.0000],
            [0.9714,    0.3429,    0.6000],
            [0.4000,    0.8000,    0.2000],
            [0.5429,    0.2857,    0.2857],
            [0.9143,    0.8857,    0.2571],
            [0.0571,    0.0286,    0.0857],
            [0.1143,    0.5714,    0.7429],
            [0.2000,    0.2286,    0.3714],
            [0.4571,    0.9143,    0.3429],
            [0.6286,    0.7143,    0.6857],
            [     0,    0.8286,    0.9429],
            [0.8857,    0.0857,    0.9714],
            [0.2286,    0.0571,    0.5714],
            [0.7143,    0.1714,    0.8571],
            [0.2571,    0.4857,    0.1429],
            [0.5714,    0.4000,    0.8286],
            [0.9429,    0.6857,    0.4857],
            [0.4857,    0.1429,    0.1143],
            [1.0000,    0.8571,    0.9143],
            [0.6000,    0.6571,    0.5143],
            [0.1429,    0.7429,    0.5429],
            [0.8571,    0.2571,    0.0571],
            [0.3143,    0.3714,    0.4286],
            [0.4286,    0.7714,    0.7714],
            [0.0286,    0.9714,    0.3143],
            [0.0857,    0.5143,    0.1714]])

        # number of experiements and simulations
        n = 3; m = design.shape[0]

        # these parameter values simulate expt 10 in Neddermeyer '43
        # params =             [R1      lam     s         rho   mratio   u0]
        params10 = np.array([1.5*2.54, 2/3,  3e5*6.84e4,  7.5,   .32,   1.65e10])
        paramslhs = np.zeros((m, 6))
        for i in range(m):#
            paramslhs[i,:] = params10*np.array([1, 1, design[i,1]*.2+.9, 1, design[i,0]*.65+.5, design[i,2]*.2+.9])
        # the simulation runs will vary mratio from .32*[.5 to 1.15]
        #                                s      from s0*[.9 to 1.1]
        #                                u0     from u0*[.9 to 1.1]

        nt = 22; nphi = 26
        time = np.c_[np.linspace(0,5.0e-5,nt,endpoint=True)]
        phi = np.linspace(0,1,nphi,endpoint=True) * 2*np.pi;
        rinner = nedderimp(time,params10);
        lam = params10[1]; R1 = params10[0];
        router = np.sqrt(rinner**2 + 1 - lam**2);
        xycirc = np.array([np.cos(phi),np.sin(phi)]).T
        r = nedderimp(time,params10);

        # # Generate simulation data

        yr = np.zeros((m,nt))
        for i in range(m):
            params = paramslhs[i,:]
            yr[i,:] = params[0]*nedderimp(time,params)
            
        y_sim = np.tile(yr,nphi)
        
        # indices
        y_sim_ind_time_phi = np.zeros((22*26,2))
        y_sim_ind_time_phi[:,0] = np.repeat(time,26)
        y_sim_ind_time_phi[:,1] = np.tile(phi,22)
        x_sim = design[:,0].reshape(m,1)
        t_sim = design[:,1:3]

        # # Generate observed data

        phi_obs = np.arange(0,(2*np.pi-.1),(2*np.pi/16))
        n_phi_obs = phi_obs.shape[0]
        time_obs = [np.array([1.5e-5, 2.7e-5, 4.5e-5]),np.array([4.5e-5]),np.array([2.5e-5, 4.5e-5])]
        n_time_obs = [tmp.shape[0] for tmp in time_obs]

        phiknots = np.arange(0,2*np.pi-.1,2*np.pi/8)
        dknots = np.expand_dims(np.array([.04, -.03, .03, -.03, .02, -.03, .03, -.03]),1)*2.5
        pphiknots = len(phiknots)
        Ddelt = np.zeros((phi_obs.shape[0], pphiknots));
        datadelt = np.matmul(Ddelt,dknots)

        # observations
        r_obs = [None]*3; y_obs = [None]*3
        for i in range(3):
            obs_params = deepcopy(params10)
            if i==1: obs_params[4]=.17
            elif i==2: obs_params[4]=.36
            r_obs[i] = np.atleast_2d(obs_params[0]*nedderimp(time_obs[i],obs_params))
            y_obs[i] = np.tile(r_obs[i].T, phi_obs.shape[0]).reshape(n_phi_obs,n_time_obs[i])
            y_obs[i] += np.tile(datadelt, n_time_obs[i])
            y_obs[i] = (y_obs[i] + .01*np.random.normal(size=y_obs[i].shape)).flatten()

        # indices of observations
        x_obs = ((np.array([params10[4], .17, .36])/.32-.5)/.65).reshape(3,1)
        # create y_ind_obs where each row is a (time, angle) pair. Pairs are grouped by time.
        # experiment 1
        y_ind_obs_1 = np.column_stack([
            np.concatenate((np.ones(phi_obs.shape[0])*time_obs[0][0],
                            np.ones(phi_obs.shape[0])*time_obs[0][1],
                            np.ones(phi_obs.shape[0])*time_obs[0][2])),
            np.tile(phi_obs, 3).T
        ])
        # experiment 2
        y_ind_obs_2 = np.column_stack([
            (np.ones(phi_obs.shape[0]) * time_obs[1]).reshape(16,1),
            phi_obs.T
        ])
        # experiment 3
        y_ind_obs_3 = np.column_stack( ( np.concatenate((np.ones(phi_obs.shape[0])*time_obs[2][0],\
                                           np.ones(phi_obs.shape[0])*time_obs[2][1])), np.tile(phi_obs,2).T ) )
        # Store in a list containing all 3 experiments
        y_ind_obs = [y_ind_obs_1, y_ind_obs_2, y_ind_obs_3]
        del y_ind_obs_1, y_ind_obs_2, y_ind_obs_3

        # ### Sepia Data
        data = SepiaData(x_sim = x_sim, t_sim = t_sim, y_sim = y_sim, y_ind_sim = y_sim_ind_time_phi,
                         x_obs = x_obs, y_obs = y_obs, y_ind_obs = y_ind_obs)
        data.transform_xt()

        # ### K basis
        # sim
        ysimmean = np.mean(y_sim,0)
        ysimsd = np.std(y_sim)
        ysimStd = (y_sim - np.tile(ysimmean,m).reshape(y_sim.shape))/ysimsd
        data.sim_data.orig_y_mean = ysimmean
        data.sim_data.orig_y_sd = ysimsd
        data.sim_data.y_std = ysimStd
        # now represent ye images using eof's
        U, s, V = np.linalg.svd(ysimStd.T, full_matrices=False)
        numPC = 3
        data.sim_data.K = U[:,0:numPC]*s[0:numPC]/np.sqrt(m)
        data.sim_data.K = data.sim_data.K.T

        # obs
        phimat = np.repeat(phi,nt).reshape((nt,nphi),order='F')
        timemat = np.repeat(time,nphi).reshape((nt,nphi))
        # compute simulator mean values simdat.ymean interpolated to the data values...
        interp_func = interp2d(phimat[0,:],timemat[:,0],data.sim_data.orig_y_mean.reshape((nt,nphi),order='F'))
        data.obs_data.orig_y_mean = []
        data.obs_data.orig_y_sd = []
        for k in range(n):
            ymk = interp_func(np.unique(data.obs_data.y_ind[k][:,1]),np.unique(data.obs_data.y_ind[k][:,0]))
            data.obs_data.orig_y_mean.append(ymk.flatten())
            data.obs_data.orig_y_sd.append(data.sim_data.orig_y_sd)

        # now compute the centered, scaled observed arrival times yStd
        data.obs_data.y_std = []
        for k in range(n):
            data.obs_data.y_std.append((data.obs_data.y[k] - data.obs_data.orig_y_mean[k])/data.sim_data.orig_y_sd)

        # for now, hack this in - if it used the inbuilt methods it would happen automatically
        tSigy_std=[]
        for i in range(len(data.obs_data.y)):
            tSigy_std.append(np.atleast_2d(np.diag(np.ones(data.obs_data.y[i].shape))))
        data.obs_data.Sigy_std = tSigy_std

        data.obs_data.K = []
        for k in range(n):
            data.obs_data.K.append(np.zeros((data.obs_data.y_std[k].shape[0], numPC)))
            for j in range(numPC):
                f = interp2d(phimat[0,:],timemat[:,0],np.reshape(data.sim_data.K[j,:],(nt,nphi),order='F'))
                data.obs_data.K[k][:,j] = f(data.obs_data.y_ind[k][:,1],data.obs_data.y_ind[k][:,0])[:,0]

        for k in range(n):
            data.obs_data.K[k] = data.obs_data.K[k].T

        ### D basis

        # compute the basis functions for the discrepancy function.
        Dobs = []
        Dsim  = np.zeros((phi.shape[0], pphiknots))
        for k in range(pphiknots):
            Ddelt[:,k] = dnorm(dist2pi(phi_obs,phiknots[k]*np.ones(phi_obs.shape[0])),0,np.pi/8).T
            x = dist2pi(phi,phiknots[k]*np.ones(phi.shape[0])).flatten()
            Dsim[:,k] = dnorm(x=x,mu=0,scale=np.pi/8)

        simdelt = np.matmul(Dsim, dknots)
        timeknots = np.linspace(0, .5, 3) * 1e-4
        ptimeknots = len(timeknots)
        for k in range(n):
            knotlocstime = np.reshape(np.repeat(timeknots, pphiknots), (ptimeknots * pphiknots, 1))
            knotlocsphi = np.expand_dims(np.tile(phiknots, ptimeknots), 1)
            pv = knotlocstime.shape[0]
            Dobs.append(np.zeros((pv, data.obs_data.y_std[k].shape[0])))
            if k == 0: Dsim = np.zeros((pv, data.sim_data.y_std.shape[1]))
            for j in range(pv):
                Dobs[-1][j, :] = dnorm(y_ind_obs[k][:, 0], knotlocstime[j], .25 * 1e-4) * \
                                 dnorm(dist2pi(y_ind_obs[k][:, 1], knotlocsphi[j] * np.ones(y_ind_obs[k][:, 1].shape[0])), 0,
                                       np.pi / 8)
                if k == 0:
                    Dsim[j, :] = dnorm(timemat.flatten('F'), knotlocstime[j], .25 * 1e-4) * \
                                 dnorm(dist2pi(phimat.flatten('F'), knotlocsphi[j] * np.ones(len(phimat.flatten()))), 0,
                                       np.pi / 8)

        data.create_D_basis(D_obs=Dobs, D_sim=Dsim)
    
        print(data)

        model = SepiaModel(data)
        model.tune_step_sizes(10, 4)
        model.do_mcmc(50)

        # Predictions 
        
        samples_dict = model.get_samples()
        
        # get ESS number of samples for preduction
        pred_samples=model.get_samples(nburn=0,effectivesamples=True)
        pred=SepiaEmulatorPrediction(x_pred=x_obs, samples=pred_samples, model=model)

        # Predicted emulator principal component weights for each of the three experimental inputs x

        predw=pred.get_w()
        
        # Predicted Simulator Output

        predy=pred.get_y()

        # Predicted

        predfull = []
        for i in range(3):
            predfull.append(SepiaFullPrediction(x_pred=np.atleast_2d(x_obs[i]), samples=pred_samples, model=model))

        pred_d = []
        pred_sim = []
        pred_obs = []
        for i in range(3):
            pred_d.append(predfull[i].get_discrepancy())
            pred_sim.append(predfull[i].get_ysim())
            pred_obs.append(predfull[i].get_yobs(obs_ref=i))