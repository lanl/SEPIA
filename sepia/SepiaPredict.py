
import scipy.linalg
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from sepia.SepiaDistCov import SepiaDistCov

class SepiaPrediction():
    """
    Base class inherited for predictions. Defines all parameters:

    :param x_pred: (npred x p) matrix, x values for which to predict
    :param samples: sample set, as provided by SepiaModel.get_samples; predict for each sample
    :param model: the SepiaModel object
    :param t_pred: (npred x q) matrix, optional; if present concatenate with x_pred for predictions, \
    otherwise thetas will be taken from theta posterior samples provided. Required for emulator model.
    :param addResidVar: add the posterior residual variability to the samples
    :param storeRlz: make and store a process realizations for each x_pred / sample combination
    :param storeMuSigma: store the mean and sigma for the GP posterior for each x_pred / sample combination
    :param do_call: boolean -- whether to call wPred/uvPred upon initialization

    """

    def __init__(self, x_pred=None, samples=None, model=None, t_pred=None,
                 addResidVar=False, storeRlz=True, storeMuSigma=False, do_call=True):
        """
        Instantiate SepiaPredict object (usually called by subclass init).

        :param x_pred: (npred x p) matrix, x values for which to predict
        :param samples: sample set, as provided by SepiaModel.get_samples; predict for each sample
        :param model: the SepiaModel object
        :param t_pred: (npred x q) matrix, optional; if present concatenate with x_pred for predictions, \
        otherwise thetas will be taken from theta posterior samples provided. Required for emulator model.
        :param addResidVar: add the posterior residual variability to the samples
        :param storeRlz: make and store a process realizations for each x_pred / sample combination
        :param storeMuSigma: store the mean and sigma for the GP posterior for each x_pred / sample combination
        :param do_call: boolean -- whether to call wPred/uvPred upon initialization

        """

        # make a list or scalar into an ndarray
        if not isinstance(x_pred,np.ndarray) or len(x_pred.shape)!=2:
            raise TypeError('x_pred is not a 2D numpy ndarray')
        if t_pred is not None and (not isinstance(t_pred,np.ndarray) or len(t_pred.shape)!=2):
            raise TypeError('t_pred is not a 2D numpy ndarray')

        # Validation of input sizes
        if x_pred.shape[1] != model.num.p:
            raise ValueError('x_pred number of columns %d is not the same as model defined p = %d'%\
                             (x_pred.shape[1],model.num.p))
        if t_pred is not None:
            if t_pred.shape[1] != model.num.q:
                raise ValueError('t_pred number of columns %d is not the same as model defined q = %d'%\
                                 (t_pred.shape[1],model.num.q))
            if x_pred.shape[0] != t_pred.shape[0]:
                raise ValueError('x_pred and t_pred have different number of rows: %d vs %d resp.'%\
                                 (x_pred.shape[0],t_pred.shape[0]))

        self.model=model
        self.xpred=x_pred
        self.t_pred=t_pred
        self.samples=samples
        self.addResidVar=addResidVar
        self.storeRlz=storeRlz
        self.storeMuSigma=storeMuSigma
        self.do_call = do_call
        self.w=[]
        self.u=[]
        self.v=[]
        self.mu=[]
        self.sigma=[]

class SepiaEmulatorPrediction(SepiaPrediction):
    """
    Make predictions of the emulator ('eta') component of the model. This functions with an emulator-only model
    or a full model, but predicts the posterior simulation estimates

    :param all: init parameters are parsed by SepiaPrediction (inherited init)

    Predictions are performed on init and stored in the object for access methods:
    """

    def __init__(self,*args,**kwrds):
        super(SepiaEmulatorPrediction,self).__init__(*args,**kwrds)
        # prediction is samples x prediction points (xpreds) x pu (basis)
        if self.do_call:
            wPred(self)

    def get_w(self):
        '''
        Returns predictions that were made on init

        :return: predictions of w, (#samples x #x_pred x pu) tensor
        '''
        return self.w

    def get_y(self,std=False):
        '''
        Project w through the K basis to provide predictions of y on native (or standardized) scale.
        (standardized refers to the mean=0 and sd=1 standardization process in model setup)

        :return: predictions of y, (#samples x #x_pred x py) tensor
        '''
        if std:
            return self.get_y_standardized()
        else:
            return self.get_y_native()
    def get_y_standardized(self):
        '''
        used by get_y, not called by user
        '''
        if self.model.num.scalar_out:
            return self.w
        else:
            return np.tensordot(self.w,self.model.data.sim_data.K,axes=[[2],[0]])

    def get_y_native(self):
        '''
        used by get_y, not called by user
        '''
        wshape=self.w.shape
        if isinstance(self.model.data.sim_data.orig_y_sd,np.ndarray):
            ysd_inpredshape = np.tile(self.model.data.sim_data.orig_y_sd, (wshape[0], wshape[1], 1))
        else:
            # cheating a bit, if it's scalar it doesn't have to be tiled out
            ysd_inpredshape=self.model.data.sim_data.orig_y_sd
        ymean_inpredshape = np.tile(self.model.data.sim_data.orig_y_mean, (wshape[0], wshape[1], 1))
        return self.get_y_standardized()*ysd_inpredshape+ymean_inpredshape

    def get_mu_sigma(self):
        '''
        Returns the stored (if requested on init) mean (vector) and sigma (matrix) of the posterior process for each sample

        :return: tuple: posterior mean (#samples x #x_pred), sigma (#samples x #x_pred x #x_pred x )
        '''
        return self.mu,self.sigma


class SepiaXvalEmulatorPrediction(SepiaEmulatorPrediction):

    def __init__(self, leave_out_inds=None, model=None, *args, **kwrds):
        import copy
        super(SepiaXvalEmulatorPrediction, self).__init__(do_call=False, x_pred=model.data.sim_data.x_trans,
                                                          t_pred=model.data.sim_data.t_trans, model=model, *args, **kwrds)
        m = self.model.num.m
        pu = self.model.num.pu
        orig_model = copy.deepcopy(self.model)
        # By default, leave out inds is just each simulation in turn; it is a list of lists
        if leave_out_inds is None:
            leave_out_inds = [[i] for i in np.arange(m)]
        w_cv = []
        x_cv = []
        t_cv = []
        for li in tqdm(leave_out_inds, desc='Cross validation...', mininterval=0.5):
            fit_inds = [i for i in np.arange(m) if i not in li]
            sub_model = copy.deepcopy(orig_model)
            # Subset zt to fit inds, update ztDist
            sub_model.data.zt = sub_model.data.zt[fit_inds, :]
            sub_model.num.m = len(fit_inds)
            sub_model.num.ztDist = SepiaDistCov(sub_model.data.zt)
            # Subset x/t to predict inds
            self.xpred = sub_model.data.sim_data.x_trans[li, :]
            self.t_pred = sub_model.data.sim_data.t_trans[li, :]
            # Subset w's -- need to index for each pu
            w_inds = np.zeros(m)
            w_inds[fit_inds] = 1
            w_inds = np.tile(w_inds, pu)
            sub_model.num.w = sub_model.num.w[w_inds == 1, :]
            # Set up sub model and call wPred
            self.model = sub_model
            wPred(self)
            w_cv.append(self.w)
            x_cv.append(self.xpred)
            t_cv.append(self.t_pred)
        self.w = np.concatenate(w_cv, axis=1)
        self.xpred = np.concatenate(x_cv, axis=0)
        self.t_pred = np.concatenate(t_cv, axis=0)
        self.leave_out_inds = leave_out_inds


class SepiaFullPrediction(SepiaPrediction):
    '''
    Make predictions of the full model: both emulator ('eta') and discrepancy ('delta') == (u,v)

    :param all: init parameters are parsed by SepiaPrediction (inherited init)

    Predictions are performed on init and stored in the object for access by methods:
    '''
    def __init__(self,*args,**kwrds):
        super(SepiaFullPrediction,self).__init__(*args,**kwrds)
        # prediction is samples x prediction points x pu or pv (basis)
        uvPred(self)

    def get_u_v(self):
        '''
        Returns predictions that were made on init

        :return: tuple: predictions of u (#samples x #x_pred x pu) , v (#samples x #x_pred x pv)
        '''
        return self.u, self.v

    def get_ysim(self, as_obs=False, std=False, obs_ref=0):
        '''
        Project u through the K basis to provide predictions of ysim on the native scale.
        (native refers to not the mean=0 and sd=1 standardization process in model setup)

        :param as_obs: provide ysim predictions at obs locations (defaults to sim locations)
        :param std: provide ysim predictions on standardized scale (defaults to native scale)
        :param obs_ref: if this is a ragged_obs problem, selects the reference observation index
         to use for transformation parameters; default index 0
        :return: predictions of native ysim, (#samples x #x_pred x py_sim(or py_obs))
        '''
        if std:
            if as_obs:
                if self.model.data.ragged_obs:
                    K = self.model.data.obs_data.K[obs_ref]
                else:
                    K = self.model.data.obs_data.K
                return np.tensordot(self.u,K,axes=[[2],[0]])
            else:
                return np.tensordot(self.u,self.model.data.sim_data.K,axes=[[2],[0]])
        else:  
            if as_obs:
                if self.model.data.ragged_obs:
                    K = self.model.data.obs_data.K[obs_ref]
                else:
                    K = self.model.data.obs_data.K
                ysd_inpredshape, ymean_inpredshape = self.calc_obs_standardizations_inpredshape(obs_ref=obs_ref)
                return np.tensordot(self.u,K,axes=[[2],[0]])*ysd_inpredshape+ymean_inpredshape
            else:
                ysd_inpredshape, ymean_inpredshape = self.calc_sim_standardizations_inpredshape()
                return np.tensordot(self.u,self.model.data.sim_data.K,axes=[[2],[0]])*ysd_inpredshape+ymean_inpredshape
       
    
    def get_discrepancy(self, as_obs=False, std=False, obs_ref=0):
        '''
        return Dsim*v to provide predictions of discrepancy on the native scale at sim locations.
        (native refers to not the sd=1 standardization process in model setup)
        
        :param as_obs: provide discrepancy predictions at obs locations (defaults to sim locations)
        :param std: provide discrepancy predictions on standardized scale (defaults to native scale)
        :param obs_ref: if this is a ragged_obs problem, selects the reference observation index
         to use for transformation parameters; default index 0
        :return: predictions of native discrepancy, (#samples x #x_pred x py_sim(or py_obs))
        '''
        if std:
            if as_obs:
                if self.model.data.ragged_obs:
                    D = self.model.data.obs_data.D[obs_ref]
                else:
                    D = self.model.data.obs_data.D
                return np.tensordot(self.v,D,axes=[[2],[0]]) 
            else:
                return np.tensordot(self.v,self.model.data.sim_data.D.T,axes=[[2],[0]])
        else:
            ysd_inpredshape,_ = self.calc_obs_standardizations_inpredshape(obs_ref=obs_ref)
            if as_obs:  
                if self.model.data.ragged_obs:
                    D = self.model.data.obs_data.D[obs_ref]
                else:
                    D = self.model.data.obs_data.D
                return np.tensordot(self.v,D,axes=[[2],[0]])*ysd_inpredshape
            else:
                return np.tensordot(self.v,self.model.data.sim_data.D.T,axes=[[2],[0]])*ysd_inpredshape

    def get_yobs(self, as_obs=False, std=False, obs_ref=0):
        '''
        return y=Ksim*u+Dsim*v to provide predictions of y on the native scale at sim locations.
        (native refers to not the mean=0 and sd=1 standardization process in model setup)

        :param as_obs: provide discrepancy predictions at obs locations (defaults to sim locations)
        :param std: provide discrepancy predictions on standardized scale (defaults to native scale)
        :param obs_ref: if this is a ragged_obs problem, selects the reference observation index
         to use for transformation parameters; default index 0
        :return: predictions of native y (Emulator+Discrepancy), (#samples x #x_pred x py_sim(or py_obs))
        '''
        return self.get_ysim(as_obs=as_obs,std=std,obs_ref=obs_ref)+self.get_discrepancy(as_obs=as_obs,std=std,obs_ref=obs_ref)
    
    def get_ysim_standardized(self):
        '''
        Project u through the Ksim basis to provide predictions of ysim on the standardized scale.
        (standardized refers to the mean=0 and sd=1 standardization process in model setup)

        :return: predictions of standardized ysim, (#samples x #x_pred x py_sim)
        '''
        return np.tensordot(self.u,self.model.data.sim_data.K,axes=[[2],[0]])

    def get_ysim_asobs_standardized(self,obs_ref=0):
        '''
        Project u through the Kobs basis to provide predictions of ysim on the native scale at obs locations.
        (native refers to not the mean=0 and sd=1 standardization process in model setup)

        :param obs_ref: if this is a ragged_obs problem, selects the reference observation index
          to use for transformation parameters; default index 0
        :return: predictions of native ysim, (#samples x #x_pred x py_obs)
        '''
        if self.model.data.ragged_obs:
            K = self.model.data.obs_data.K[obs_ref]
        else:
            K = self.model.data.obs_data.K
        return np.tensordot(self.u,K,axes=[[2],[0]])

    def get_ysim_native(self):
        '''
        Project u through the Ksim basis to provide predictions of ysim on the native scale.
        (native refers to not the mean=0 and sd=1 standardization process in model setup)

        :return: predictions of native ysim, (#samples x #x_pred x py_sim)
        '''
        ysd_inpredshape, ymean_inpredshape = self.calc_sim_standardizations_inpredshape()
        return self.get_ysim_standardized()*ysd_inpredshape+ymean_inpredshape

    def get_ysim_asobs_native(self,obs_ref=0):
        '''
        Project u through the Kobs basis to provide predictions of ysim on the native scale at obs locations.
        (native refers to not the mean=0 and sd=1 standardization process in model setup)

        :param obs_ref: if this is a ragged_obs problem, selects the reference observation index
          to use for transformation parameters; default index 0
        :return: predictions of native ysim, (#samples x #x_pred x py_obs)
        '''
        ysd_inpredshape, ymean_inpredshape = self.calc_obs_standardizations_inpredshape(obs_ref=obs_ref)
        return self.get_ysim_asobs_standardized(obs_ref=obs_ref)*ysd_inpredshape+ymean_inpredshape
        
    def get_discrepancy_standardized(self, obs_ref=0):
        '''
        return Dobs*v to provide predictions of discrepancy on the standardized scale at obs locations.
        (standardized refers to the sd=1 standardization process in model setup)

        :param obs_ref: if this is a ragged_obs problem, selects the reference observation index
          to use for transformation parameters; default index 0
        :return: predictions of standardized discrepancy, (#samples x #x_pred x py_obs)
        '''
        if self.model.data.ragged_obs:
            D = self.model.data.obs_data.D[obs_ref]
        else:
            D = self.model.data.obs_data.D
        return np.tensordot(self.v,D,axes=[[2],[0]])

    def get_discrepancy_native(self,obs_ref=0):
        '''
        return Dobs*v to provide predictions of discrepancy on the native scale at obs locations.
        (native refers to not the sd=1 standardization process in model setup)

        :param obs_ref: if this is a ragged_obs problem, selects the reference observation index
          to use for transformation parameters; default index 0
        :return: predictions of native discrepancy, (#samples x #x_pred x py_obs)
        '''
        ysd_inpredshape,_ = self.calc_obs_standardizations_inpredshape(obs_ref=obs_ref)
        if self.model.data.ragged_obs:
            D = self.model.data.obs_data.D[obs_ref]
        else:
            D = self.model.data.obs_data.D
        return np.tensordot(self.v,D,axes=[[2],[0]])*ysd_inpredshape
        
    def get_yobs_standardized(self,obs_ref=0):
        '''
        return y=Kobs*u+Dobs*v to provide predictions of y on the standardized scale at obs locations.
        (standardized refers to the mean=0 and sd=1 standardization process in model setup)

        :param obs_ref: if this is a ragged_obs problem, selects the reference observation index
          to use for transformation parameters; default index 0
        :return: predictions of standardized y, (#samples x #x_pred x py_obs)
        '''
        return self.get_ysim_asobs_standardized(obs_ref=obs_ref)+self.get_discrepancy_standardized(obs_ref=obs_ref)

    def get_yobs_native(self, obs_ref=0):
        '''
        return y=Kobs*u+Dobs*v to provide predictions of y on the native scale at obs locations.
        (native refers to not the mean=0 and sd=1 standardization process in model setup)

        :param obs_ref: if this is a ragged_obs problem, selects the reference observation index
          to use for transformation parameters; default index 0
        :return: predictions of native y, (#samples x #x_pred x py_obs)
        '''
        ysd_inpredshape, ymean_inpredshape = self.calc_obs_standardizations_inpredshape(obs_ref=obs_ref)
        return (self.get_ysim_asobs_standardized(obs_ref=obs_ref)+self.get_discrepancy_standardized(obs_ref=obs_ref))*ysd_inpredshape+ymean_inpredshape

    def get_mu_sigma(self):
        '''
        Returns the stored (if requested on init) mean (vector) and sigma (matrix) of the posterior process for each sample

        :return: tuple: posterior mean (#samples x #x_pred), sigma (#samples x #x_pred x #x_pred x )
        '''
        return self.mu,self.sigma

    def calc_sim_standardizations_inpredshape(self):
        # internal function, calculate the ysd and ymean arrays
        # tile out the standardization vectors to the full prediction shape (is this this only way?!?)
        ushape=self.u.shape
        if isinstance(self.model.data.sim_data.orig_y_sd,np.ndarray):
            ysd_inpredshape = np.tile(self.model.data.sim_data.orig_y_sd, (ushape[0], ushape[1], 1))
        else:
            # cheating a bit, if it's scalar it doesn't have to be tiled out
            ysd_inpredshape=self.model.data.sim_data.orig_y_sd
        ymean_inpredshape = np.tile(self.model.data.sim_data.orig_y_mean, (ushape[0], ushape[1], 1))
        return ysd_inpredshape, ymean_inpredshape

    def calc_obs_standardizations_inpredshape(self,obs_ref):
        # internal function, calculate the ysd and ymean arrays
        # tile out the standardization vectors to the full prediction shape (is this this only way?!?)
        if self.model.data.ragged_obs:
            if obs_ref<0 or obs_ref>len(self.model.data.obs_data.orig_y_sd):
                raise ValueError('obs_ref index specified in predictions is not within obs_data size')
            orig_y_sd = self.model.data.obs_data.orig_y_sd[obs_ref]
            orig_y_mean = self.model.data.obs_data.orig_y_mean[obs_ref]
        else:
            orig_y_sd = self.model.data.obs_data.orig_y_sd
            orig_y_mean = self.model.data.obs_data.orig_y_mean
        ushape=self.u.shape
        if isinstance(orig_y_sd,np.ndarray):
            ysd_inpredshape = np.tile(orig_y_sd, (ushape[0], ushape[1], 1))
        else:
            # cheating a bit, if it's scalar it doesn't have to be tiled out
            ysd_inpredshape=orig_y_sd
        ymean_inpredshape = np.tile(orig_y_mean, (ushape[0], ushape[1], 1))
        return ysd_inpredshape, ymean_inpredshape

'''
So much for the sugar, here's the medicine... 
'''

def rmultnormsvd(n,mu,sigma):
    # using this for development, to verify with the same rand stream as matlab
    U, s, V = np.linalg.svd(sigma, full_matrices=False)
    normalrands=norm.ppf(np.random.rand(np.shape(mu)[0],n))
    rnorm=np.tile(mu,(1,n)) + U @ np.diag(np.sqrt(s)) @ normalrands
    return rnorm.squeeze()

def uvPred(pred, useAltW=False):
    # some shorthand references from the pred object
    xpred=pred.xpred
    samples=pred.samples
    num=pred.model.num
    data=pred.model.data
    theta_pred=pred.t_pred

    n=num.n; m=num.m; p=num.p; q=num.q; pu=num.pu; pv=num.pv
    lamVzGnum=num.lamVzGnum; lamVzGroup=num.lamVzGroup

    npred = np.shape(xpred)[0]
    nsamp = samples['lamWs'].shape[0]

    x0Dist = num.x0Dist
    xpred0Dist=SepiaDistCov(xpred)
    xxpred0Dist=SepiaDistCov(data.x,xpred)

    if pred.storeRlz:
        tpred = np.empty((nsamp, npred*(pv+pu) ))
    if pred.storeMuSigma:
        pred.mu=np.empty((nsamp,npred*(pv+pu) ))
        pred.sigma=np.empty((nsamp,npred*(pv+pu),npred*(pv+pu) ))

    for ii in range(nsamp):
        theta = samples['theta'][ii:ii + 1, :]
        betaU = samples['betaU'][ii, :]
        betaU = np.reshape(betaU, (p+q, pu), order='F')
        betaV = samples['betaV'][ii, :]
        betaV = np.reshape(betaV, (p, lamVzGnum), order='F')
        lamUz = samples['lamUz'][ii:ii + 1, :]
        lamVz = samples['lamVz'][ii:ii + 1, :]
        lamWs = samples['lamWs'][ii:ii + 1, :]
        lamWOs = samples['lamWOs'][ii:ii + 1, :]
        lamOs = samples['lamOs'][ii:ii + 1, :]
        
        if theta_pred:
            xpredt = np.concatenate((xpred, theta_pred), axis=1)
        else:
            xpredt = np.concatenate((xpred, np.tile(theta, (npred, 1))), axis=1)


        xtheta=np.concatenate((data.x,np.tile(theta, (n, 1))),axis=1)
        xDist=SepiaDistCov(xtheta)
        xzDist=SepiaDistCov(xtheta,data.zt)
        xpredDist=SepiaDistCov(xpredt)
        xxpredDist=SepiaDistCov(xtheta,xpredt)
        zxpredDist=SepiaDistCov(data.zt,xpredt)

        # SigData
        # Generate the part of the matrix related to the data
        # Four parts to compute: Sig_v, Sig_u, Sig_w, and the Sig_uw crossterm
        vCov=[]
        for jj in range(lamVzGnum):
            vCov.append(x0Dist.compute_cov_mat(betaV[:, jj], lamVz[0,jj]))
        SigV=np.zeros((n*pv,n*pv))
        for jj in range(pv):
            SigV[jj*n:(jj+1)*n,jj*n:(jj+1)*n]=vCov[lamVzGroup[jj]]

        SigU=np.zeros((n*pu,n*pu))
        for jj in range(pu):
            SigU[jj*n:(jj+1)*n,jj*n:(jj+1)*n]=xDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
        np.fill_diagonal(SigU, SigU.diagonal() + np.repeat(np.reciprocal(lamWs), n))

        SigW = np.zeros((m * pu, m * pu))
        for jj in range(pu):
            SigW[jj * m:(jj + 1) * m, jj * m:(jj + 1) * m] = num.ztDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
        np.fill_diagonal(SigW, SigW.diagonal() +
                         np.repeat(np.reciprocal(num.LamSim * lamWOs), m) + np.repeat(np.reciprocal(lamWs), m))


        SigUW=np.zeros((n*pu,m*pu))
        for jj in range(pu):
            SigUW[jj*n:(jj+1)*n,jj*m:(jj+1)*m]=xzDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])

        if num.scalar_out:
            #SigData=[ SigU+SigV +SigObs/lamOs    SigUW; ...
            #          SigUW'                     SigW ];
            SigUpV=SigU+SigV + num.SigObs * 1/lamOs
            SigData=np.block([[SigUpV,SigUW],[SigUW.T,SigW]])
            # Calculate inverse of SigData directly using block stuff TODO use this?
            if useAltW:
                SigWinv = scipy.linalg.inv(SigW)
                SigWinvSigUWT = scipy.linalg.solve(SigW, SigUW.T)
                Atmp = SigUpV - SigUW @ SigWinvSigUWT
                DinvA = scipy.linalg.inv(Atmp)
                DinvB = -scipy.linalg.solve(Atmp, SigWinvSigUWT.T)
                DinvD = SigWinv - DinvB.T @ SigWinvSigUWT.T
                SigDatainv = np.concatenate(
                   (np.concatenate((DinvA, DinvB), axis=1),
                    np.concatenate((DinvB.T, DinvD), axis=1) ), axis=0)
                # Gatt: np.block would be better, like:
                # SigDatainv=np.block([[DinvA,DinvB],[DinvB.T,Dinv.D]])
        else:
            #SigData=[SigV                 0
            #        0                     [ SigU    SigUW; ...
            #                               SigUW'  SigW  ] ];
            #SigData(1:n*(pv+pu),1:n*(pv+pu)) += model.SigObs*1/lamOs;
            SigSubmat=np.block([[SigU,SigUW],[SigUW.T,SigW]])
            # Computing inverse of SigData directly TODO use this?
            if useAltW:
                USigObs = num.SigObs[n*pv:, n*pv:]
                VSigObs = num.SigObs[:n*pv, :n*pv]
                SigWinv = scipy.linalg.inv(SigW)
                Asub = SigU + USigObs*1/lamOs
                SigWinvSigUWT = scipy.linalg.solve(SigW, SigUW.T)
                Atmp = Asub - SigUW @ SigWinvSigUWT
                DinvA = scipy.linalg.inv(Atmp)
                DinvB = -scipy.linalg.solve(Atmp, SigWinvSigUWT.T)
                DinvD = SigWinv - DinvB.T @ SigWinvSigUWT.T
                Dinv = np.concatenate(
                   (np.concatenate((DinvA, DinvB), axis=1),
                    np.concatenate((DinvB.T, DinvD), axis=1) ), axis=0)
                # Gatt: aseemble with np.block as above
            sddim=n*pv + (n+m)*pu
            SigData=np.zeros((sddim,sddim))
            SigData[:n*pv,:n*pv] = SigV
            SigData[n*pv:,n*pv:] = SigSubmat
            SigData[:n*(pv+pu),:n*(pv+pu)] += num.SigObs*1/lamOs
            # TODO use this?
            if useAltW:
                SigDatainv = np.zeros_like(SigData)
                SigDatainv[:n*pv,:n*pv] = scipy.linalg.inv(SigV + VSigObs*1/lamOs)
                SigDatainv[n*pv:,n*pv:] = Dinv

        # SigPred
        # Generate the part of the matrix related to the predictors
        # Parts to compute: Sig_vpred, Sig_upred
        SigVp=np.zeros((npred*pv,npred*pv))
        vpCov=[]
        for jj in range(lamVzGnum):
            vpCov.append(xpred0Dist.compute_cov_mat(betaV[:, jj], lamVz[0,jj]))
        for jj in range(pv):
            SigVp[jj*npred:(jj+1)*npred,jj*npred:(jj+1)*npred]=vpCov[lamVzGroup[jj]]

        SigUp=np.zeros((npred*pu,npred*pu))
        for jj in range(pu):
            SigUp[jj*npred:(jj+1)*npred,jj*npred:(jj+1)*npred] = \
                          xpredDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
        np.fill_diagonal(SigUp, SigUp.diagonal() + np.repeat(np.reciprocal(lamWs), npred))
        if pred.addResidVar:
            np.fill_diagonal(SigUp, SigUp.diagonal() + np.repeat(np.reciprocal(num.LamSim * lamWOs), npred))
        #SigPred=[SigVp 0
        #         0      SigUp  ]
        SigPred=np.zeros( (npred*(pu+pv),npred*(pu+pv)) )
        SigPred[:npred*pv,:npred*pv]=SigVp
        SigPred[npred*pv:,npred*pv:]=SigUp

        # SigCross
        SigVVx=np.zeros((n*pv,npred*pv))
        vvCov=[]
        for jj in range(lamVzGnum):
            vvCov.append(xxpred0Dist.compute_cov_mat(betaV[:, jj], lamVz[0,jj]))
        for jj in range(pv):
            SigVVx[jj*n:(jj+1)*n,jj*npred:(jj+1)*npred]=vvCov[lamVzGroup[jj]]

        SigUUx=np.zeros((n*pu,npred*pu))
        for jj in range(pu):
            SigUUx[jj*n:(jj+1)*n,jj*npred:(jj+1)*npred]=xxpredDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])

        SigWUx=np.zeros((m*pu,npred*pu))
        for jj in range(pu):
            SigWUx[jj*m:(jj+1)*m,jj*npred:(jj+1)*npred]=zxpredDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])

        if num.scalar_out:
            #SigCross=[SigVVx                 SigUUx; ...
            #          zeros(m*pu,npred*pv)   SigWUx];
            SigCross=np.zeros( ( n*pv+m*pu, npred*(pv+pu) )  )
            SigCross[:n*pv,:npred*pv]=SigVVx
            SigCross[n*pv:,:npred*pv]=SigUUx
            SigCross[n*pv:,npred*pv:]=SigWUx
        else:
            #SigCross=[SigVVx                 zeros(n*pv,npred*pu); ...
            #          zeros(n*pu,npred*pv)   SigUUx; ...
            #          zeros(m*pu,npred*pv)   SigWUx];
            SigCross=np.zeros( ( n*pv+(n+m)*pu, npred*(pv+pu) ) )
            SigCross[:n*pv,         :npred*pv]=SigVVx
            SigCross[n*pv:n*(pv+pu),npred*pv:]=SigUUx
            SigCross[n*(pv+pu):,    npred*pv:]=SigWUx

        # Get posterior parameters
        if not useAltW:
            W = scipy.linalg.solve(SigData, SigCross, sym_pos=True)
            if num.scalar_out:
                Myhat = W.T @ num.uw
            else:
                Myhat = W.T @ num.vuw
            Syhat = SigPred - W.T @ SigCross
        else:
        # TODO see if using structure is better/faster
            W = np.zeros((n*pv + (n+m)*pu, npred* (pu + pv)))
            W[:n*pv, :npred*pv] = SigDatainv[:n*pv, :n*pv] @ SigCross[:n*pv, :npred*pv]
            W[n*pv:, npred*pv:] = SigDatainv[n*pv:, n*pv:] @ SigCross[n*pv:, npred*pv:]
            #W = SigDatainv @ SigCross
            #import matplotlib.pyplot as plt
            #W_zeros = W.copy()
            #W_zeros[W == 0] = np.nan
            #plt.imshow(W_zeros, aspect='auto')
            #plt.show()
            # So Wnew gets the zero blocks, W doesn't quite. Also can exploit the zero blocks for the multiplications below, I think.
            # TODO see how to use zeros below, check implementation of SigDatainv
            if num.scalar_out:
                Myhat = W.T @ num.uw
            else:
                #Myhat = W.T @ num.vuw
                Myhat = W[:n*pv, :].T @ num.v + W[n*pv:, :].T @ np.concatenate([num.u, num.w])
            Syhat = SigPred - W.T @ SigCross

        if pred.storeRlz:
            # Record a realization
            tpred[ii, :] = rmultnormsvd(1, Myhat, Syhat)
            # testing speed of built in
            #tpred[ii, :] = np.random.multivariate_normal(Myhat.squeeze(), Syhat)

        if pred.storeMuSigma:
            # add the distribution params to the return
            pred.mu[ii, :] = np.squeeze(Myhat)
            pred.sigma[ii, :, :] = Syhat


    if pred.storeRlz:
        # Reshape the pred matrix to 3D, for each component:
        #  first dim  - (number of realizations [pvals])
        #  second dim  - (number of points [x,theta]s)
        #  third dim - (number of principal components)
        pred.v=np.zeros( (nsamp,npred, pv) )
        pred.u=np.zeros( (nsamp,npred, pu) )
        for ii in range(pv):
          pred.v[:,:,ii]=tpred[:,ii*npred:(ii+1)*npred]
        for ii in range(pu):
          pred.u[:,:,ii]=tpred[:,pv*npred+ii*npred:pv*npred+(ii+1)*npred]

    # and at the end, everything should be stored back in the prediction object.


def wPred(pred):
    # some shorthand references from the pred object
    xpred=pred.xpred
    samples=pred.samples
    num=pred.model.num
    data=pred.model.data
    theta_pred=pred.t_pred

    n=num.n; m=num.m; p=num.p; q=num.q; pu=num.pu

    npred=np.shape(xpred)[0]
    nsamp=samples['lamWs'].shape[0]

    #allocate results containers if needed
    if pred.storeRlz:
        tpred = np.zeros((nsamp, npred * pu))
    if pred.storeMuSigma:
        pred.mu=np.empty((nsamp,npred*pu))
        pred.sigma=np.empty((nsamp,npred*pu,npred*pu))

    for ii in range(nsamp):
        if not num.sim_only:
            theta=samples['theta'][ii:ii+1,:]
        betaU=samples['betaU'][ii,:]
        betaU=np.reshape(betaU,(p+q,pu),order='F')
        lamUz=samples['lamUz'][ii:ii+1,:]
        lamWs=samples['lamWs'][ii:ii+1,:]
        lamWOs=samples['lamWOs'][ii:ii+1,:]

        if theta_pred is not None:
            xpredt = np.concatenate((xpred,theta_pred),axis=1)
        elif not num.sim_only:
            xpredt = np.concatenate( ( xpred,np.tile(theta,(npred, 1)) ),axis=1)
        else:
            xpredt=xpred

        xpredDist=SepiaDistCov(xpredt)
        zxpredDist=SepiaDistCov(data.zt,xpredt)

        Myhat=np.zeros((npred*pu,1))
        Syhat=np.zeros((npred*pu,npred*pu))
        for jj in range(pu):

            SigW = num.ztDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
            np.fill_diagonal(SigW,SigW.diagonal() + 1/(num.LamSim[jj]*lamWOs) + 1/lamWs[0,jj] )

            SigWp = xpredDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
            diagAdd=np.reciprocal(lamWs[0,jj])
            if pred.addResidVar:
                diagAdd += 1/(num.LamSim[jj]*lamWOs)
            np.fill_diagonal(SigWp, SigWp.diagonal() + diagAdd )

            SigWWp = zxpredDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])

            SigData=SigW
            SigPred=SigWp
            SigCross=SigWWp

            # Get posterior parameters
            W=scipy.linalg.solve(SigData,SigCross,sym_pos=True)
            Myhat[jj*npred:(jj+1)*npred] = W.T @ num.w[jj*m:(jj+1)*m,0:1]
            Syhat[jj*npred:(jj+1)*npred,jj*npred:(jj+1)*npred] = SigPred - W.T @ SigCross

        if pred.storeRlz:
          # Record a realization
          tpred[ii,:]=rmultnormsvd(1,Myhat,Syhat)

        if pred.storeMuSigma:
          pred.mu[ii,:]=np.squeeze(Myhat)
          pred.sigma[ii,:,:]=Syhat

    if pred.storeRlz:
        #% Reshape the pred matrix to 3D:
        #%  first dim  - (number of realizations == samples)
        #%  second dim  - (number of prediction points n = number of rows of [x,theta])
        #%  third dim - (number of basis elements in K = pu)
        pred.w=np.zeros((nsamp,npred,pu))
        for ii in range(pu):
            pred.w[:,:,ii]=tpred[:,ii*npred:(ii+1)*npred]

    # and at the end, everything should be stored back in the prediction object.
