import unittest

import unittest
import numpy as np
from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaLogLik import compute_log_lik
from sepia.SepiaPredict import SepiaEmulatorPrediction, SepiaFullPrediction

class SepDesSetupLogLik(unittest.TestCase):

    def setUp(self):
        # Set up and check kronecker Emulator-only model
        x1=np.hstack((np.linspace(0,1,3).reshape(-1,1), np.linspace(1,0,3).reshape(-1,1)))
        x2=np.linspace(0,1,4).reshape(-1,1)

        r1,r2=np.meshgrid(range(len(x1)),range(len(x2)))

        self.x_sim=np.hstack( (x1[r1.reshape(-1,order='F'),:], x2[r2.reshape(-1,order='F'),:] ) )
        self.x_sim_kron=[x1,x2]
        self.y_sim=np.column_stack( ((self.x_sim[:,0]/2 + (np.sum(self.x_sim,axis=1)**2)),
                                     np.linspace(0,1,self.x_sim.shape[0]) ) )
        self.y_ind_sim=np.array([0,1])

    def test_emulator_setup_loglik(self):
        # set up regular model
        d=SepiaData(x_sim=self.x_sim,y_sim=self.y_sim,y_ind_sim=np.array([0,1]))
        d.create_K_basis(K=np.eye(2))
        d.transform_xt(x_notrans=True)
        d.standardize_y(scale='columnwise')
        print(d)
        mod=SepiaModel(d)
        print('Emulator model LL=%f \n' % compute_log_lik(mod) )

        # set up kron model
        kd=SepiaData(xt_sim_sep=self.x_sim_kron,y_sim=self.y_sim,y_ind_sim=self.y_ind_sim)
        kd.create_K_basis(K=np.eye(2))
        kd.transform_xt(x_notrans=True)
        kd.standardize_y(scale='columnwise')
        print(kd)
        kmod=SepiaModel(kd)
        print('Emulator Sep model LL=%f \n' % compute_log_lik(kmod) )

        self.assertAlmostEqual(compute_log_lik(mod),compute_log_lik(kmod),places=5)
        pass

    def test_full_setup_loglik(self):
        # Set up and check calibration model
        x_obs=np.ones((3,2)) * np.array([0.5,0.75,0.25]).reshape((-1,1))
        y_obs=np.block([[-0.1,0.1],[-0.2,0.3],[0.1,0]])

        # augment to also test more than scalar dimensions in x and t
        x_sim_cal = np.hstack((0.5*np.ones((self.x_sim.shape[0],1)), self.x_sim[:,:1] ))
        t_sim_cal = self.x_sim[:,1:]
        xt_sim_sep = [np.array(0.5).reshape(1,1)] + self.x_sim_kron

        y_sim_std=(self.y_sim-np.mean(self.y_sim,axis=0).reshape(1,-1)) / np.std(self.y_sim,axis=0,ddof=1).reshape(1,-1)

        dc=SepiaData(x_sim=x_sim_cal, t_sim=t_sim_cal, y_sim=y_sim_std,
                     x_obs=x_obs, y_obs=y_obs, y_ind_sim=self.y_ind_sim, y_ind_obs=self.y_ind_sim)
        dc.create_K_basis(K=np.eye(2))
        dc.create_D_basis(D_sim=np.eye(2),D_obs=np.eye(2))
        dc.transform_xt(x_notrans=True,t_notrans=True)
        dc.standardize_y(y_mean=0,y_sd=1)
        print(dc)
        cmod=SepiaModel(dc)

        print('Calibration model LL=%f'%compute_log_lik(cmod))

        kdc=SepiaData(xt_sim_sep=xt_sim_sep, y_sim=y_sim_std,
                      x_obs=x_obs, y_obs=y_obs, y_ind_sim=self.y_ind_sim, y_ind_obs=self.y_ind_sim)
        kdc.create_K_basis(K=np.eye(2))
        kdc.create_D_basis(D_sim=np.eye(2),D_obs=np.eye(2))
        kdc.transform_xt(x_notrans=True,t_notrans=True)
        kdc.standardize_y(y_mean=0,y_sd=1)
        print(kdc)
        kcmod=SepiaModel(kdc)

        print('Calibration Sep model LL=%f'%compute_log_lik(kcmod))

        self.assertAlmostEqual(compute_log_lik(cmod),compute_log_lik(kcmod), places=5)

        np.random.seed(42)
        cmod.do_mcmc(100)
        csamp=cmod.get_samples(sampleset=[99])
        cpred=SepiaFullPrediction(mode='Sep',model=cmod,samples=csamp, storeMuSigma=True,
                                 x_pred=np.array([0.5,0.5]).reshape((1,-1)))
        print(cpred.get_ysim())
        csm,css=cpred.get_mu_sigma()
        print(csm)
        print(css)

        np.random.seed(42)
        kcmod.do_mcmc(100)
        kcsamp=kcmod.get_samples(sampleset=[99])
        kcpred=SepiaFullPrediction(mode='Sep',model=kcmod,samples=kcsamp, storeMuSigma=True,
                                 x_pred=np.array([0.5,0.5]).reshape((1,-1)))
        print(kcpred.get_ysim())
        kcsm,kcss=kcpred.get_mu_sigma()
        print(kcsm)
        print(kcss)

        print('testing max difference which is %g'%np.max(abs(csm - kcsm)))
        self.assertAlmostEqual(0,np.max(abs(csm-kcsm)))

        print('testing max difference which is %g'%np.max(abs(css - kcss)))
        self.assertAlmostEqual(0,np.max(abs(css-kcss)))

        pass