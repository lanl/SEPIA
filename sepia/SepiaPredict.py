
import scipy.linalg
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from sepia.SepiaDistCov import SepiaDistCov

class SepiaPrediction():
    """
    Base class inherited for predictions. Contains:

    :var sepia.SepiaModel model: SepiaModel instance
    :var numpy.ndarray xpred: x values for which to predict, shape (npred, p) matrix, on original untransformed scale
    :var numpy.ndarray/NoneType t_pred: t values for which to predict, shape (npred, q) matrix, optional for full model
                                        (if not provided, `theta` values from posterior samples will be used) but required for emulator.
    :var dict samples: from `SepiaModel.get_samples()`
    :var bool addResidVar: add the posterior residual variability to the samples?
    :var bool storeRlz: make and store a process realizations for each x_pred / sample combination?
    :var bool storeMuSigma: store the mean and sigma for the GP posterior for each x_pred / sample combination?
    :var bool do_call: call wPred/uvPred upon initialization?
    :var numpy.ndarray/NoneType w: simulation predictions on PCA weight space, shape (#samples, #x_pred, pu)
    :var numpy.ndarray/NoneType u: observation predictions on PCA weight space, shape (#samples, #x_pred, pu)
    :var numpy.ndarray/NoneType v: observation predictions on D weight space, shape (#samples, #x_pred, pv)
    :var numpy.ndarray/NoneType mu: posterior mean, shape (#samples, #x_pred)
    :var numpy.ndarray/NoneType sigma: posterior sigma, shape (#samples, #x_pred, #x_pred)

    """

    def __init__(self, x_pred=None, samples=None, model=None, t_pred=None,
                 addResidVar=False, storeRlz=True, storeMuSigma=False, do_call=True):
        # """
        # Instantiate SepiaPredict object (usually not called directly, but by subclass __init__).
        #
        # :param numpy.ndarray x_pred: x values for which to predict, shape (npred, p) matrix, on original untransformed scale
        # :param dict samples: from `SepiaModel.get_samples()`
        # :param sepia.SepiaModel model: the SepiaModel object
        # :param numpy.ndarray/NoneType t_pred: t values for which to predict, shape (npred, q) matrix, optional for full model
        #                                 (if not provided, `theta` values from posterior samples will be used) but required for emulator.
        # :param bool addResidVar: add the posterior residual variability to the samples?
        # :param bool storeRlz: make and store a process realizations for each x_pred / sample combination?
        # :param bool storeMuSigma: store the mean and sigma for the GP posterior for each x_pred / sample combination?
        # :param bool do_call: call wPred/uvPred upon initialization?
        # :raises TypeError: if inputs are not expected types
        # :raises ValueError: if inputs are not expected shapes
        #
        # """

        # make a list or scalar into an ndarray
        if not model.data.dummy_x:
            if not isinstance(x_pred,np.ndarray) or len(x_pred.shape)!=2:
                raise TypeError('x_pred is not a 2D numpy ndarray')
        else:
            if x_pred is not None:
                raise TypeError('specified x_pred, but this is a no-x model')
        if t_pred is not None and (not isinstance(t_pred,np.ndarray) or len(t_pred.shape)!=2):
            raise TypeError('t_pred is not a 2D numpy ndarray')

        # Specify the x_pred for a no-x model with a dummy-x field if needed
        if model.data.dummy_x:
            if t_pred is None:
                x_pred=np.array([0.5]).reshape((1,1)) # 1x1 x, to use for predictions of sampled thetas
            else:
                x_pred=0.5*np.ones((t_pred.shape[0],1))

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

        # transform x and/or t from native scale to emulator scale
        if t_pred is None:
            x_pred,_ = model.data.transform_xt(x=x_pred)
        else:
            x_pred,t_pred = model.data.transform_xt(x=x_pred,t=t_pred)

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
    or a full model, but predicts the posterior simulation estimates.
    Predictions are performed on init and stored in the object for access methods.

    """

    def __init__(self,*args,**kwrds):
        """
        Instantiate SepiaEmulatorPrediction object.

        :param numpy.ndarray x_pred: x values for which to predict, shape (npred, p) matrix, on original untransformed scale
        :param dict samples: from `SepiaModel.get_samples()`
        :param sepia.SepiaModel model: the SepiaModel object
        :param numpy.ndarray t_pred: t values for which to predict, shape (npred, q) matrix, required.
        :param bool addResidVar: add the posterior residual variability to the samples?
        :param bool storeRlz: make and store a process realizations for each x_pred / sample combination?
        :param bool storeMuSigma: store the mean and sigma for the GP posterior for each x_pred / sample combination?
        :param bool do_call: call wPred upon initialization?
        :raises TypeError: if inputs are not expected types
        :raises ValueError: if inputs are not expected shapes

        """
        super(SepiaEmulatorPrediction,self).__init__(*args,**kwrds)
        # prediction is samples x prediction points (xpreds) x pu (basis)
        
        if self.model.data.sep_design:
            print('Separable Design model. Separable designs for emulator predictions are not currently implemented')
            print('  Work around is to set this problem up as a full (calibration) model, and use SepiaFullPrediction')
            print('  to get predictions that include those from the emulator. ')
            raise ValueError('Separable design for emulator predictions is not currently implemented')

        if self.do_call:
            wPred(self)

    def get_w(self):
        """
        Returns predictions that were made on init in PCA weight space.

        :return: predictions of w, (#samples x #x_pred x pu) numpy.ndarray
        """
        return self.w

    def get_y(self, std=False):
        """
        Project w through the K basis to provide predictions of y on native (or standardized) scale.
        (standardized refers to the mean=0 and sd=1 standardization process in model setup).

        :param bool std: return standardized (True) or native (default, False) scaling of predictions
        :return: predictions of y, (#samples x #x_pred x py) tensor
        """
        if std:
            return self.get_y_standardized()
        else:
            return self.get_y_native()

    def get_mu_sigma(self):
        """
        Returns the stored (if requested on init) mean (vector) and sigma (matrix) of the posterior process for each sample

        :return: tuple: posterior mean (#samples x #x_pred), sigma (#samples x #x_pred x #x_pred x )
        """
        return self.mu,self.sigma

    def get_y_standardized(self):
        #
        # used by get_y, not called by user
        #
        if self.model.num.scalar_out:
            return self.w
        else:
            return np.tensordot(self.w,self.model.data.sim_data.K,axes=[[2],[0]])

    def get_y_native(self):
        #
        # used by get_y, not called by user
        #
        wshape=self.w.shape
        if isinstance(self.model.data.sim_data.orig_y_sd,np.ndarray):
            ysd_inpredshape = np.tile(self.model.data.sim_data.orig_y_sd, (wshape[0], wshape[1], 1))
        else:
            # cheating a bit, if it's scalar it doesn't have to be tiled out
            ysd_inpredshape=self.model.data.sim_data.orig_y_sd
        ymean_inpredshape = np.tile(self.model.data.sim_data.orig_y_mean, (wshape[0], wshape[1], 1))
        return self.get_y_standardized()*ysd_inpredshape+ymean_inpredshape


class SepiaXvalEmulatorPrediction(SepiaEmulatorPrediction):
    """
    Cross-validated predictions from the emulator.

    """

    def __init__(self, leave_out_inds=None, model=None, *args, **kwrds):
        """
        Instantiate SepiaXvalEmulatorPrediction object.

        :param list/NoneType leave_out_inds: optional, list of lists of indices to leave out in each fold; defaults to leave-one-out
        :param numpy.ndarray x_pred: x values for which to predict, shape (npred, p) matrix, on original untransformed scale
        :param dict samples: from `SepiaModel.get_samples()`
        :param sepia.SepiaModel model: the SepiaModel object
        :param numpy.ndarray t_pred: t values for which to predict, shape (npred, q) matrix, required.
        :param bool addResidVar: add the posterior residual variability to the samples?
        :param bool storeRlz: make and store a process realizations for each x_pred / sample combination?
        :param bool storeMuSigma: store the mean and sigma for the GP posterior for each x_pred / sample combination?
        :param bool do_call: call wPred upon initialization?
        :raises TypeError: if inputs are not expected types
        :raises ValueError: if inputs are not expected shapes

        """
        import copy
        if model.data.dummy_x:
            super(SepiaXvalEmulatorPrediction, self).__init__(do_call=False, t_pred=model.data.sim_data.t_trans, model=model, *args, **kwrds)
        else:
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
            sub_model.num.ztDist = SepiaDistCov(sub_model.data.zt, cat_ind=np.concatenate([sub_model.data.x_cat_ind, sub_model.data.t_cat_ind]))
            # Subset x/t to predict inds (check if None)
            if sub_model.data.sim_data.x_trans is None:
                #self.xpred = np.array([[0.5]])
                self.xpred = None
            else:
                self.xpred = sub_model.data.sim_data.x_trans[li, :]
            if sub_model.data.sim_data.t_trans is None:
                self.t_pred = np.array([[]])
            else:
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
    """
    Make predictions of the full model: both emulator ('eta') and discrepancy ('delta') == (u,v)
    Predictions are performed on init and stored in the object for access by methods.
    """
    def __init__(self,mode='Sep',*args,**kwrds):
        """
        Instantiate SepiaFullPrediction object.

        :param numpy.ndarray x_pred: x values for which to predict, shape (npred, p) matrix, on original untransformed scale
        :param dict samples: from `SepiaModel.get_samples()`
        :param sepia.SepiaModel model: the SepiaModel object
        :param numpy.ndarray t_pred: t values for which to predict, shape (npred, q) matrix, optional (can take from theta samples).
        :param bool addResidVar: add the posterior residual variability to the samples?
        :param bool storeRlz: make and store a process realizations for each x_pred / sample combination?
        :param bool storeMuSigma: store the mean and sigma for the GP posterior for each x_pred / sample combination?
        :raises TypeError: if inputs are not expected types
        :raises ValueError: if inputs are not expected shapes

        """
        super(SepiaFullPrediction,self).__init__(*args,**kwrds)
        # prediction is samples x prediction points x pu or pv (basis)
        # TODO remove notSep option now that Sep is vetted
        if mode=='notSep':
            uvPred(self)
        else:
            uvPredSep(self)


    def get_u_v(self):
        """
        Returns predictions that were made on init

        :return: tuple: predictions of u (#samples x #x_pred x pu) , v (#samples x #x_pred x pv)
        """
        return self.u, self.v

    def get_ysim(self, as_obs=False, std=False, obs_ref=0):
        """
        Project u through the K basis to provide predictions of ysim on the native scale.
        (native refers to not the mean=0 and sd=1 standardization process in model setup)

        :param bool as_obs: provide ysim predictions at obs locations (defaults to sim locations)
        :param bool std: provide ysim predictions on standardized scale (defaults to native scale)
        :param int obs_ref: if this is a ragged_obs problem, selects the reference observation index
         to use for transformation parameters; default index 0
        :return: predictions of native ysim, (#samples x #x_pred x py_sim(or py_obs)) or (#samples x py_sim(or py_obs)) if ragged and obs_ref is specified
        """
        if std:
            if self.model.num.scalar_out:
                return self.u
            else:
                if as_obs:
                        if self.model.data.ragged_obs:
                            K = self.model.data.obs_data.K[obs_ref]
                            return np.tensordot(self.u,K,axes=[[2],[0]])[:,obs_ref,:]
                        else:
                            K = self.model.data.obs_data.K
                            return np.tensordot(self.u,K,axes=[[2],[0]])
                else:
                    return np.tensordot(self.u,self.model.data.sim_data.K,axes=[[2],[0]])
        else:
            if self.model.num.scalar_out:
                return self.u*self.model.data.sim_data.orig_y_sd + self.model.data.sim_data.orig_y_mean
            else:
                if as_obs:
                    if self.model.data.ragged_obs:
                        K = self.model.data.obs_data.K[obs_ref]
                        ysd_inpredshape, ymean_inpredshape = self.calc_obs_standardizations_inpredshape(obs_ref=obs_ref)
                        return (np.tensordot(self.u,K,axes=[[2],[0]])*ysd_inpredshape+ymean_inpredshape)[:,obs_ref,:]
                    else:
                        K = self.model.data.obs_data.K
                        ysd_inpredshape, ymean_inpredshape = self.calc_obs_standardizations_inpredshape(obs_ref=obs_ref)
                        return np.tensordot(self.u,K,axes=[[2],[0]])*ysd_inpredshape+ymean_inpredshape
                else:
                    ysd_inpredshape, ymean_inpredshape = self.calc_sim_standardizations_inpredshape()
                    return np.tensordot(self.u,self.model.data.sim_data.K,axes=[[2],[0]])*ysd_inpredshape+ymean_inpredshape

    def get_discrepancy(self, as_obs=False, std=False, obs_ref=0):
        """
        return Dsim*v to provide predictions of discrepancy on the native scale at sim locations.
        (native refers to not the sd=1 standardization process in model setup)
        
        :param bool as_obs: provide discrepancy predictions at obs locations (defaults to sim locations)
        :param bool std: provide discrepancy predictions on standardized scale (defaults to native scale)
        :param int obs_ref: if this is a ragged_obs problem, selects the reference observation index
         to use for transformation parameters; default index 0
        :return: predictions of native discrepancy, (#samples x #x_pred x py_sim(or py_obs)) or (#samples x py_sim(or py_obs)) if ragged and obs_ref is specified
        """

        if self.model.num.pv==0:  # no-discrepancy model
            raise TypeError('discrepancy requested from a no-discrepancy model')

        if std:
            if as_obs:
                if self.model.data.ragged_obs:
                    D = self.model.data.obs_data.D[obs_ref]
                    return np.tensordot(self.v,D,axes=[[2],[0]])[:,obs_ref,:]
                else:
                    D = self.model.data.obs_data.D
                    return np.tensordot(self.v,D,axes=[[2],[0]]) 
            else:
                return np.tensordot(self.v,self.model.data.sim_data.D,axes=[[2],[0]])
        else:
            ysd_inpredshape,_ = self.calc_obs_standardizations_inpredshape(obs_ref=obs_ref)
            if as_obs:  
                if self.model.data.ragged_obs:
                    D = self.model.data.obs_data.D[obs_ref]
                    return (np.tensordot(self.v,D,axes=[[2],[0]])*ysd_inpredshape)[:,obs_ref,:]
                else:
                    D = self.model.data.obs_data.D
                    return np.tensordot(self.v,D,axes=[[2],[0]])*ysd_inpredshape
            else:
                return np.tensordot(self.v,self.model.data.sim_data.D,axes=[[2],[0]])*ysd_inpredshape

    def get_yobs(self, as_obs=False, std=False, obs_ref=0):
        """
        return y=Ksim*u+Dsim*v to provide predictions of y on the native scale at sim locations.
        (native refers to not the mean=0 and sd=1 standardization process in model setup)

        :param bool as_obs: provide discrepancy predictions at obs locations (defaults to sim locations)
        :param bool std: provide discrepancy predictions on standardized scale (defaults to native scale)
        :param int obs_ref: if this is a ragged_obs problem, selects the reference observation index
         to use for transformation parameters; default index 0
        :return: predictions of native y (Emulator+Discrepancy), (#samples x #x_pred x py_sim(or py_obs)) or (#samples x py_sim(or py_obs)) if ragged and obs_ref is specified
        """

        if self.model.num.pv==0: #means it's a no-discrepancy model
            return self.get_ysim(as_obs=as_obs, std=std, obs_ref=obs_ref)
        else:
            return self.get_ysim(as_obs=as_obs,std=std,obs_ref=obs_ref) + \
                   self.get_discrepancy(as_obs=as_obs,std=std,obs_ref=obs_ref)

    def get_mu_sigma(self):
        """
        Returns the stored (if requested on init) mean (vector) and sigma (matrix) of the posterior process for each sample

        :return: tuple: posterior mean (#samples x #x_pred), sigma (#samples x #x_pred x #x_pred x )
        """
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

# """
# So much for the sugar, here's the medicine...
# """

def rmultnorm(n, mu, sigma, dev=True):

    # the deal with this is to use the same rand stream to generate realizations in the same way
    # gpmsa, for testing purposes. numpy's mvn uses a different stream.
    # I guess there's no reason not to leave it like this - in dev mode?

    if not dev:
        if n!=1:
            raise ValueError('Internal error: native method should have only been asked to produce single realizations')
        return np.random.multivariate_normal(mu.squeeze(), sigma)
    else:
        # development, to verify with the same rand stream as matlab
        U, s, V = np.linalg.svd(sigma, full_matrices=False)
        normalrands=norm.ppf(np.random.rand(np.shape(mu)[0],n))
        rnorm=np.tile(mu,(1,n)) + U @ np.diag(np.sqrt(s)) @ normalrands
        return rnorm.squeeze()

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
        if data.mean_basis is not None:
            gamma=samples['gamma'][ii:ii+1,:].reshape((-1,1))

        if theta_pred is not None:
            xpredt = np.concatenate((xpred,theta_pred),axis=1)
            cat_ind = np.concatenate([data.x_cat_ind, data.t_cat_ind])
        elif not num.sim_only:
            xpredt = np.concatenate( ( xpred,np.tile(theta,(npred, 1)) ),axis=1)
            cat_ind = np.concatenate([data.x_cat_ind, data.t_cat_ind])
        else:
            xpredt=xpred
            cat_ind = data.x_cat_ind

        xpredDist=SepiaDistCov(xpredt, cat_ind=cat_ind)
        zxpredDist=SepiaDistCov(data.zt,xpredt, cat_ind=cat_ind)

        Myhat=np.zeros((npred*pu,1))
        Syhat=np.zeros((npred*pu,npred*pu))
        
        if not data.mean_basis:
            w = num.w
        else:
            w = num.w - data.sim_data.H @ gamma
        
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
            Myhat[jj*npred:(jj+1)*npred] = W.T @ w[jj*m:(jj+1)*m,0:1]
            if data.mean_basis is not None:
                if data.dummy_x or pred.xpred is None:
                    pred_mb_dat=pred.t_pred
                elif pred.t_pred is None:
                    pred_mb_dat=pred.xpred
                else:
                    pred_mb_dat=np.hstack( (pred.xpred,pred.t_pred) )
                H_pred=data.make_mean_basis(pred_mb_dat)
                Myhat[jj*npred:(jj+1)*npred] += H_pred @ gamma
            Syhat[jj*npred:(jj+1)*npred,jj*npred:(jj+1)*npred] = SigPred - W.T @ SigCross

        if pred.storeRlz:
          # Record a realization
          tpred[ii,:]=rmultnorm(1, Myhat, Syhat)

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


#
# This version should not normally be called; should be using uvPredSep
#     - which is the block-wise prediction calculations, supporting the separable design schema
#
def uvPred(pred):
    # some shorthand references from the pred object
    raise RuntimeError('uvPred should not normally be invoked; uvPredSep is the correct method')
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
    xpred0Dist=SepiaDistCov(xpred, cat_ind=data.x_cat_ind)
    xxpred0Dist=SepiaDistCov(data.x, xpred, cat_ind=data.x_cat_ind)

    if pred.storeRlz:
        tpred = np.empty((nsamp, npred*(pv+pu) ))
    if pred.storeMuSigma:
        pred.mu=np.empty((nsamp,npred*(pv+pu) ))
        pred.sigma=np.empty((nsamp,npred*(pv+pu),npred*(pv+pu) ))

    for ii in range(nsamp):
        theta = samples['theta'][ii:ii + 1, :]
        betaU = samples['betaU'][ii, :]
        betaU = np.reshape(betaU, (p+q, pu), order='F')
        if pv>0:
            betaV = samples['betaV'][ii, :]
            betaV = np.reshape(betaV, (p, lamVzGnum), order='F')
            lamVz = samples['lamVz'][ii:ii + 1, :]
            no_D=False
        else:
            no_D=True
        lamUz = samples['lamUz'][ii:ii + 1, :]
        lamWs = samples['lamWs'][ii:ii + 1, :]
        lamWOs = samples['lamWOs'][ii:ii + 1, :]
        lamOs = samples['lamOs'][ii:ii + 1, :]

        if theta_pred is not None:
            xpredt = np.concatenate((xpred, theta_pred), axis=1)
        else:
            xpredt = np.concatenate((xpred, np.tile(theta, (npred, 1))), axis=1)

        xtheta=np.concatenate((data.x,np.tile(theta, (n, 1))),axis=1)
        xDist=SepiaDistCov(xtheta, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))
        xzDist=SepiaDistCov(xtheta,data.zt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))
        xpredDist=SepiaDistCov(xpredt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))
        xxpredDist=SepiaDistCov(xtheta,xpredt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))
        zxpredDist=SepiaDistCov(data.zt,xpredt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))

        # SigData
        # Generate the part of the matrix related to the data
        # Four parts to compute: Sig_v, Sig_u, Sig_w, and the Sig_uw crossterm
        SigV = np.zeros((n * pv, n * pv)) # for no_D model, pv=0
        if not no_D:
            vCov=[]
            for jj in range(lamVzGnum):
                vCov.append(x0Dist.compute_cov_mat(betaV[:, jj], lamVz[0,jj]))
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
            if not no_D:
                SigUplusVpart=SigU+SigV + num.SigObs * 1/lamOs
            else:
                SigUplusVpart = SigU + num.SigObs * 1 / lamOs
            SigData=np.block([[SigUplusVpart,SigUW],[SigUW.T,SigW]])
        else:
            #SigData=[SigV                 0
            #        0                     [ SigU    SigUW; ...
            #                               SigUW'  SigW  ] ];
            #SigData(1:n*(pv+pu),1:n*(pv+pu)) += model.SigObs*1/lamOs;
            SigSubmat=np.block([[SigU,SigUW],[SigUW.T,SigW]])
            sddim=n*pv+(n+m)*pu
            SigData=np.zeros((sddim,sddim))
            SigData[:n*pv,:n*pv] = SigV
            SigData[n*pv:,n*pv:] = SigSubmat
            SigData[:n*(pv+pu),:n*(pv+pu)] += num.SigObs*1/lamOs

        # SigPred
        # Generate the part of the matrix related to the predictors
        # Parts to compute: Sig_vpred, Sig_upred
        SigVp=np.zeros((npred*pv,npred*pv))
        if not no_D:
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
        if not no_D:
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
            if not no_D:
                #SigCross=[SigVVx                 SigUUx; ...
                #          zeros(m*pu,npred*pv)   SigWUx];
                SigCross=np.zeros( ( n*pv+m*pu, npred*(pv+pu) )  )
                SigCross[:n*pv,:npred*pv]=SigVVx
                SigCross[n*pv:,:npred*pv]=SigUUx
                SigCross[n*pv:,npred*pv:]=SigWUx
            else: # no Discrepancy model
                # SigCross=[SigUUx;
                #           SigWUx]
                SigCross=np.vstack((SigUUx,SigWUx))
        else:
            #SigCross=[SigVVx                 zeros(n*pv,npred*pu); ...
            #          zeros(n*pu,npred*pv)   SigUUx; ...
            #          zeros(m*pu,npred*pv)   SigWUx];
            SigCross=np.zeros( ( n*pv+(n+m)*pu, npred*(pv+pu) ) )
            SigCross[:n*pv,         :npred*pv]=SigVVx
            SigCross[n*pv:n*(pv+pu),npred*pv:]=SigUUx
            SigCross[n*(pv+pu):,    npred*pv:]=SigWUx

        # Get posterior parameters
        W = scipy.linalg.solve(SigData, SigCross, sym_pos=True)
        if num.scalar_out:
            Myhat = W.T @ num.uw
        else:
            Myhat = W.T @ num.vuw
        Syhat = SigPred - W.T @ SigCross

        if pred.storeRlz:
            # Record a realization
            tpred[ii, :] = rmultnorm(1, Myhat, Syhat)
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

def uvPredSep(pred):

    # calculate the equivalent quadratic form of kron separable data
    def sepQuadFormCalc(V,zp):
        # calculate right side of the kronecker quadratic form solve
        dlen,mlen=zp.shape
        zpo=np.empty((dlen,mlen))
        for jj in range(mlen):
            zt=zp[:,jj]
            for ii in range(len(V)-1,-1,-1):
                Vsize=V[ii].shape[1]
                zt=scipy.linalg.solve(V[ii],zt.reshape((Vsize,int(dlen/Vsize)),order='F') ).T
            zpo[:,jj]=zt.reshape(-1,order='F')
        return zpo

    def sepCalc(W,S,nugget,m):
      # use kron form to compute:
      #  T = W' * inv( kron(S) + nugget*I ) * W
      #  r = W' * inv( kron(S) + nugget*I ) * m
      # for matrix W, covariance S, scalar nugget, and vector m
      # where S is a cell array of sub-covariances to be composed by kron

        # eigen decomposition of the blocks
        V=[None]*len(S)
        D=[None]*len(S)
        for ii in range(len(S)):
            D[ii], V[ii] = np.linalg.eig(S[ii])

        # compose eigenvalues
        dkron=D[-1]
        for ii in range(len(D)-2,-1,-1):
            dkron=np.kron(D[ii],dkron)

        # inverse sqrt of the D diagonal augmented with the nugget
        Dki2=(1/np.sqrt(dkron + nugget)).reshape((-1,1))

        # Put the parts together for W'inv(S)W
        zp=sepQuadFormCalc(V,W)
        zp2T=zp * np.tile(Dki2,(1,zp.shape[1]) )
        T=zp2T.T @ zp2T

        # Put the parts together for W'inv(S)m
        zp=sepQuadFormCalc(V,m)
        zp2r=zp * Dki2
        r=zp2T.T @ zp2r

        return T,r


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
    xpred0Dist=SepiaDistCov(xpred, cat_ind=data.x_cat_ind)
    xxpred0Dist=SepiaDistCov(data.x, xpred, cat_ind=data.x_cat_ind)
    if data.sep_design:
        ztSep=data.ztSep
        ztSepDist=num.ztSepDist

    if pred.storeRlz:
        tpred = np.empty((nsamp, npred*(pv+pu) ))
    if pred.storeMuSigma:
        pred.mu=np.empty((nsamp,npred*(pv+pu) ))
        pred.sigma=np.empty((nsamp,npred*(pv+pu),npred*(pv+pu) ))

    for ii in range(nsamp):
        theta = samples['theta'][ii:ii + 1, :]
        betaU = samples['betaU'][ii, :]
        betaU = np.reshape(betaU, (p+q, pu), order='F')
        if pv>0:
            betaV = samples['betaV'][ii, :]
            betaV = np.reshape(betaV, (p, lamVzGnum), order='F')
            lamVz = samples['lamVz'][ii:ii + 1, :]
            no_D=False
        else:
            no_D=True
        lamUz = samples['lamUz'][ii:ii + 1, :]
        lamWs = samples['lamWs'][ii:ii + 1, :]
        lamWOs = samples['lamWOs'][ii:ii + 1, :]
        lamOs = samples['lamOs'][ii:ii + 1, :]

        if theta_pred is not None:
            xpredt = np.concatenate((xpred, theta_pred), axis=1)
        else:
            xpredt = np.concatenate((xpred, np.tile(theta, (npred, 1))), axis=1)

        xtheta=np.concatenate((data.x,np.tile(theta, (n, 1))),axis=1)
        xDist=SepiaDistCov(xtheta, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))
        xzDist=SepiaDistCov(xtheta,data.zt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))
        xpredDist=SepiaDistCov(xpredt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))
        xxpredDist=SepiaDistCov(xtheta,xpredt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))
        zxpredDist=SepiaDistCov(data.zt,xpredt, cat_ind=np.concatenate([data.x_cat_ind, data.t_cat_ind]))

        # SigData
        # Generate the part of the matrix related to the data
        # Four parts to compute: Sig_v, Sig_u, Sig_w, and the Sig_uw crossterm
        SigV = np.zeros((n * pv, n * pv)) # for no_D model, pv=0
        if not no_D:
            vCov=[]
            for jj in range(lamVzGnum):
                vCov.append(x0Dist.compute_cov_mat(betaV[:, jj], lamVz[0,jj]))
            for jj in range(pv):
                SigV[jj*n:(jj+1)*n,jj*n:(jj+1)*n]=vCov[lamVzGroup[jj]]

        SigU=np.zeros((n*pu,n*pu))
        for jj in range(pu):
            SigU[jj*n:(jj+1)*n,jj*n:(jj+1)*n]=xDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
        np.fill_diagonal(SigU, SigU.diagonal() + np.repeat(np.reciprocal(lamWs), n))

        #SigW = np.zeros((m * pu, m * pu))
        #for jj in range(pu):
        #    SigW[jj * m:(jj + 1) * m, jj * m:(jj + 1) * m] = num.ztDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
        #np.fill_diagonal(SigW, SigW.diagonal() +
        #                 np.repeat(np.reciprocal(num.LamSim * lamWOs), m) + np.repeat(np.reciprocal(lamWs), m))
        #SigUW=np.zeros((n*pu,m*pu))
        #for jj in range(pu):
        #    SigUW[jj*n:(jj+1)*n,jj*m:(jj+1)*m]=xzDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])

        SigWb = [None]*pu
        if not data.sep_design:
            for jj in range(pu):
                SigWb[jj]= num.ztDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
                # a scalar added to diagonal for each block
                np.fill_diagonal(SigWb[jj], SigWb[jj].diagonal() + 1/(num.LamSim[jj] * lamWOs) + 1/lamWs[0,jj] )
        else:
            SigWbDiag = [None]*pu
            for jj in range(pu):
                segVarStart=0
                SigWb[jj] = [None]*len(ztSep)
                for kk in range(len(ztSep)):
                    segInds = np.arange(segVarStart,segVarStart+ztSepDist[kk].p)
                    segVarStart = segVarStart + ztSepDist[kk].p
                    if kk==0: # count lamUz once while composing the block from sep design
                        SigWb[jj][kk] = ztSepDist[kk].compute_cov_mat(betaU[segInds, jj], lamUz[0, jj])
                    else:
                        SigWb[jj][kk] = ztSepDist[kk].compute_cov_mat(betaU[segInds, jj], 1)
                SigWbDiag[jj]=1/(num.LamSim[jj] * lamWOs) + 1/lamWs[0,jj] # should be scalar

        SigUWb = [None]*pu
        for jj in range(pu):
            SigUWb[jj] = np.vstack( ( np.zeros((jj*n,m)),
                                      xzDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj]),
                                      np.zeros(((pu-jj-1)*n,m)) ) )

        # augmented for separable...
        #if num.scalar_out:
        #    #SigData=[ SigU+SigV +SigObs/lamOs    SigUW; ...
        #    #          SigUW'                     SigW ];
        #    if not no_D:
        #        SigUplusVpart=SigU+SigV + num.SigObs * 1/lamOs
        #    else:
        #        SigUplusVpart = SigU + num.SigObs * 1 / lamOs
        #    SigData=np.block([[SigUplusVpart,SigUW],[SigUW.T,SigW]])
        #else:
        #    #SigData=[SigV                 0
        #    #        0                     [ SigU    SigUW; ...
        #    #                               SigUW'  SigW  ] ];
        #    #SigData(1:n*(pv+pu),1:n*(pv+pu)) += model.SigObs*1/lamOs;
        #    SigSubmat=np.block([[SigU,SigUW],[SigUW.T,SigW]])
        #    sddim=n*pv+(n+m)*pu
        #    SigData=np.zeros((sddim,sddim))
        #    SigData[:n*pv,:n*pv] = SigV
        #    SigData[n*pv:,n*pv:] = SigSubmat
        #    SigData[:n*(pv+pu),:n*(pv+pu)] += num.SigObs*1/lamOs

        # SigPred
        # Generate the part of the matrix related to the predictors
        # Parts to compute: Sig_vpred, Sig_upred
        SigVp=np.zeros((npred*pv,npred*pv))
        if not no_D:
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
        if not no_D:
            vvCov=[]
            for jj in range(lamVzGnum):
                vvCov.append(xxpred0Dist.compute_cov_mat(betaV[:, jj], lamVz[0,jj]))
            for jj in range(pv):
                SigVVx[jj*n:(jj+1)*n,jj*npred:(jj+1)*npred]=vvCov[lamVzGroup[jj]]

        SigUUx=np.zeros((n*pu,npred*pu))
        for jj in range(pu):
            SigUUx[jj*n:(jj+1)*n,jj*npred:(jj+1)*npred]=xxpredDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])

        SigWUxb=[None]*pu
        for jj in range(pu):
            SigWUxb[jj] = np.vstack( ( np.zeros((jj*npred,m)),
                                      zxpredDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj]).T,
                                      np.zeros(((pu-jj-1)*npred,m)) ) )

        #  augmented for separable
        #if num.scalar_out:
        #    if not no_D:
        #        #SigCross=[SigVVx                 SigUUx; ...
        #        #          zeros(m*pu,npred*pv)   SigWUx];
        #        SigCross=np.zeros( ( n*pv+m*pu, npred*(pv+pu) )  )
        #        SigCross[:n*pv,:npred*pv]=SigVVx
        #        SigCross[n*pv:,:npred*pv]=SigUUx
        #        SigCross[n*pv:,npred*pv:]=SigWUx
        #    else: # no Discrepancy model
        #        # SigCross=[SigUUx;
        #        #           SigWUx]
        #        SigCross=np.vstack((SigUUx,SigWUx))
        #else:
        #    #SigCross=[SigVVx                 zeros(n*pv,npred*pu); ...
        #    #          zeros(n*pu,npred*pv)   SigUUx; ...
        #    #          zeros(m*pu,npred*pv)   SigWUx];
        #    SigCross=np.zeros( ( n*pv+(n+m)*pu, npred*(pv+pu) ) )
        #    SigCross[:n*pv,         :npred*pv]=SigVVx
        #    SigCross[n*pv:n*(pv+pu),npred*pv:]=SigUUx
        #    SigCross[n*(pv+pu):,    npred*pv:]=SigWUx

        # Get posterior parameters
        # pre-ksep
        #W = scipy.linalg.solve(SigData, SigCross, sym_pos=True)
        #if num.scalar_out:
        #    Myhat = W.T @ num.uw
        #else:
        #    Myhat = W.T @ num.vuw
        #Syhat = SigPred - W.T @ SigCross

        SigVUo = np.zeros( (n*(pv+pu),n*(pv+pu)) )
        SigVUo[:n*pv,:n*pv]=SigV
        SigVUo[n*pv:,n*pv:]=SigU
        SigVUo = SigVUo + num.SigObs * 1/lamOs

        SigVUx = np.zeros( (n*(pv+pu), npred*(pv+pu)) )
        SigVUx[:n*pv,:npred*pv] = SigVVx
        SigVUx[n*pv:,npred*pv:] = SigUUx

        SignoW = np.block([[SigVUo, SigVUx], [SigVUx.T, SigPred]])

        #TODO: here's where the w is.

        SigWsb=[None]*pu
        mugWsb=[None]*pu
        for jj in range(pu):
            SigWcrossb=np.zeros( ( (n+npred)*(pv+pu), m ) )
            SigWcrossb[n*pv:n*(pv+pu),:]=SigUWb[jj]
            SigWcrossb[(n*(pv+pu)+npred*pv):((n+npred)*(pv+pu)),:]=SigWUxb[jj]
            SigWcrossb=SigWcrossb.T
            if not data.sep_design:
                Tb=scipy.linalg.solve(SigWb[jj],SigWcrossb, sym_pos=True).T
                SigWsb[jj] = Tb @ SigWcrossb
                mugWsb[jj] = (Tb @ num.w[jj*m:(jj+1)*m,0] ).reshape(-1,1)
            else:
                SigWsb[jj],mugWsb[jj]=sepCalc(SigWcrossb,SigWb[jj],SigWbDiag[jj],num.w[jj*m:(jj+1)*m,:])

        SiggWb=SignoW
        mugWb=np.zeros(mugWsb[0].shape)
        for jj in range(pu):
            SiggWb=SiggWb - SigWsb[jj]
            mugWb=mugWb + mugWsb[jj]

        SiggW11=SiggWb[:n*(pv+pu),:n*(pv+pu)]
        SiggW22=SiggWb[n*(pv+pu):,n*(pv+pu):]
        SiggW12=SiggWb[:n*(pv+pu):,n*(pv+pu):]
        mugW1=mugWb[:n*(pv+pu)]
        mugW2=mugWb[n*(pv+pu):]

        #TODO: Here's where u is (in num.vu)

        T=scipy.linalg.solve(SiggW11,SiggW12).T
        Syhat=SiggW22 - T@SiggW12
        Myhat=mugW2 + T @ (num.vu-mugW1)

        if pred.storeRlz:
            # Record a realization
            tpred[ii, :] = rmultnorm(1, Myhat, Syhat)
            # testing speed of built in
            #tpred[ii, :] = np.random.multivariate_normal(Myhat.squeeze(), Syhat)

        if pred.storeMuSigma:
            # add the distribution params to the return
            pred.mu[ii, :] = np.squeeze(Myhat)
            pred.sigma[ii, :, :] = Syhat

    #TODO: Here's where it's added back into u

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

