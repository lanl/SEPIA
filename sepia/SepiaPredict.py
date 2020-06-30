"""
@author: gatt

Sepia Model Predictions

"""
import scipy.linalg
import numpy as np
from scipy.stats import norm
from sepia.SepiaDistCov import SepiaDistCov

# initially a data container
# But eventually this object will do predictions, tranformations, potentially diagnostics
class SepiaEmulatorPrediction():
    def __init__(self, xpred, samples, model, theta_pred=None,
                 addResidVar=False, returnRlz=True, storeMuSigma=False):

        self.model=model
        self.xpred=xpred
        self.theta_pred=theta_pred
        self.samples=samples

        # prediction is samples x pu (basis) x prediction points (xpreds)
        wPred(self, xpred, samples, model.num, model.data, theta_pred,
                       addResidVar, returnRlz, returnMuSigma=storeMuSigma)

    def get_w(self):
        return self.pred
    #def get_y_standardized(self):
    #    return np.tensordot(self.pred.w,self.model.data.
    def get_y_native(self):
        pass

class SepiaFullPrediction():
    def __init__(self, xpred, samples, model, theta_pred=None,
                       addResidVar=False, returnRlz=True, returnMuSigma=False):

        self.pred=uvPred(self, xpred, samples, model.num, model.data, theta_pred,
                    addResidVar, returnRlz, returnMuSigma)



def rmultnormsvd(n,mu,sigma):
    # using this for development, to verify with the same rand stream as matlab
    U, s, V = np.linalg.svd(sigma, full_matrices=False)
    normalrands=norm.ppf(np.random.rand(np.shape(mu)[0],n))
    rnorm=np.tile(mu,(1,n)) + U @ np.diag(np.sqrt(s)) @ normalrands
    return rnorm.squeeze()

def uvPred(pred, xpred, samples, num, data, theta_pred=None,
           addResidVar=False,returnRlz=True,returnMuSigma=False, useAltW=False):

    n=num.n; m=num.m; p=num.p; q=num.q; pu=num.pu; pv=num.pv
    lamVzGnum=num.lamVzGnum; lamVzGroup=num.lamVzGroup

    if type(xpred) == float: xpred = np.reshape(xpred, (1, 1))
    if len(np.shape(xpred)) == 1: xpred = np.reshape(xpred, (1, len(xpred)))
    npred = np.shape(xpred)[0]

    nsamp = samples['lamWs'].shape[0]

    # CHANGED: get x0Dist from num, don't recompute
    x0Dist = num.x0Dist
    xpred0Dist=SepiaDistCov(xpred)
    xxpred0Dist=SepiaDistCov(data.x,xpred)

    if returnRlz:
        tpred = np.empty((nsamp, npred*(pv+pu) ))
    if returnMuSigma:
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
        #ztDist=SepiaDistCov(data.zt) # CHANGED: not being used, don't calculate, or get from num
        xzDist=SepiaDistCov(xtheta,data.zt)
        xpredDist=SepiaDistCov(xpredt)
        xxpredDist=SepiaDistCov(xtheta,xpredt)
        zxpredDist=SepiaDistCov(data.zt,xpredt)

        # SigData
        # Generate the part of the matrix related to the data
        # Four parts to compute: Sig_v, Sig_u, Sig_w, and the Sig_uw crossterm
        vCov=[]
        for jj in range(lamVzGnum):
            vCov.append(x0Dist.compute_cov_mat(betaV[:, jj], lamVz[jj]))
        SigV=np.zeros((n*pv,n*pv))
        for jj in range(pv):
            SigV[jj*n:(jj+1)*n,jj*n:(jj+1)*n]=vCov[lamVzGroup[jj]]

        SigU=np.zeros((n*pu,n*pu))
        for jj in range(pu):
           SigU[jj*n:(jj+1)*n,jj*n:(jj+1)*n]=xDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
        #np.fill_diagonal(SigU, SigU.diagonal() +
        #                 np.kron(np.reciprocal(lamWs),np.ones((1,n))) )
        # CHANGED: use repeat instead of kron
        np.fill_diagonal(SigU, SigU.diagonal() + np.repeat(np.reciprocal(lamWs), n))

        SigW = np.zeros((m * pu, m * pu))
        for jj in range(pu):
            SigW[jj * m:(jj + 1) * m, jj * m:(jj + 1) * m] = num.ztDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
        #np.fill_diagonal(SigW, SigW.diagonal() +
        #                 np.kron(np.reciprocal(num.LamSim * lamWOs), np.ones((1, m))) +
        #                 np.kron(np.reciprocal(lamWs), np.ones((1, m))))
        # CHANGED: use repeat instead of kron
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
            vpCov.append(xpred0Dist.compute_cov_mat(betaV[:, jj], lamVz[jj]))
        for jj in range(pv):
            SigVp[jj*npred:(jj+1)*npred,jj*npred:(jj+1)*npred]=vpCov[lamVzGroup[jj]]

        SigUp=np.zeros((npred*pu,npred*pu))
        for jj in range(pu):
            SigUp[jj*npred:(jj+1)*npred,jj*npred:(jj+1)*npred] = \
                          xpredDist.compute_cov_mat(betaU[:, jj], lamUz[0, jj])
        #np.fill_diagonal(SigUp, SigUp.diagonal() +
        #                   np.kron(np.reciprocal(lamWs), np.ones((1, npred))))
        # CHANGED: use repeat instead of kron
        np.fill_diagonal(SigUp, SigUp.diagonal() + np.repeat(np.reciprocal(lamWs), npred))
        if addResidVar:
            #np.fill_diagonal(SigUp, SigUp.diagonal() +
            #            np.kron(np.reciprocal(num.LamSim * lamWOs), np.ones((1, npred))))
            # CHANGED: use repeat instead of kron
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
            vvCov.append(xxpred0Dist.compute_cov_mat(betaV[:, jj], lamVz[jj]))
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

        if returnRlz:
            # Record a realization
            tpred[ii, :] = rmultnormsvd(1, Myhat, Syhat)
            # testing speed of built in
            #tpred[ii, :] = np.random.multivariate_normal(Myhat.squeeze(), Syhat)

        if returnMuSigma:
            # add the distribution params to the return
            pred.mu[ii, :] = np.squeeze(Myhat)
            pred.sigma[ii, :, :] = Syhat


    if returnRlz:
        # Reshape the pred matrix to 3D, for each component:
        #  first dim  - (number of realizations [pvals])
        #  second dim - (number of principal components)
        #  third dim  - (number of points [x,theta]s)
        pred.v=np.zeros( (nsamp,pv,npred) )
        pred.u=np.zeros( (nsamp,pu,npred) )
        for ii in range(pv):
          pred.v[:,ii,:]=tpred[:,ii*npred:(ii+1)*npred]
        for ii in range(pu):
          pred.u[:,ii,:]=tpred[:,pv*npred+ii*npred:pv*npred+(ii+1)*npred]

    return pred


def wPred(pred, xpred,samples,num,data=None,theta_pred=None,
          addResidVar=False,returnRlz=True,returnMuSigma=False):

    n=num.n; m=num.m; p=num.p; q=num.q; pu=num.pu

    if type(xpred) == float : xpred=np.reshape(xpred,(1,1))
    if len(np.shape(xpred)) ==1 : xpred=np.reshape(xpred,(len(xpred),1))
    npred=np.shape(xpred)[0]

    nsamp=samples['lamWs'].shape[0]

    if returnRlz:
        tpred = np.zeros((nsamp, npred * pu))
    if returnMuSigma:
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

        if not num.sim_only:
            if theta_pred:
                xpredt = np.concatenate((xpred,theta_pred),axis=1)
            else:
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
            if addResidVar:
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

        if returnRlz:
          # Record a realization
          tpred[ii,:]=rmultnormsvd(1,Myhat,Syhat)

        if returnMuSigma:
          # add the distribution params to the return
          pred.mu[ii,:]=np.squeeze(Myhat)
          pred.sigma[ii,:,:]=Syhat

    if returnRlz:
        #% Reshape the pred matrix to 3D:
        #%  first dim  - (number of realizations == samples)
        #%  second dim - (number of basis elements in K = pu)
        #%  third dim  - (number of prediction points n = number of rows of [x,theta])
        pred.w=np.zeros((nsamp,pu,npred))
        for ii in range(pu):
            pred.w[:,ii,:]=tpred[:,ii*npred:(ii+1)*npred]

    return pred

