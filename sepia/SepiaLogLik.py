

import numpy as np
import scipy.linalg

from sepia.SepiaDistCov import SepiaDistCov


def compute_log_lik(g, cvar='all', cindex=None):
    """
    Compute log likelihood for model g. Returns value and also stores it in g.num.logLik.

    :param sepia.SepiaModel g: instantiated `sepia.SepiaModel` object
    :param string cvar: name of changed parameter (used to avoid recomputing things that won't have changed since last call), or 'all'
    :param int/NoneType cindex: index (of flattened parameter array) of changed variable (to avoid recomputing things), or None to assume all indices changed
    :return: numeric log likelihood value
    :raises ValueError: if invalid cvar parameter used
    """

    if g.verbose:
        print('Entering SepiaLogLik')

    def doLogLik(cov, w):
        try:
            chCov = scipy.linalg.cholesky(cov, lower=True)
        except np.linalg.LinAlgError:
            print('chol error')
            return -np.inf
        logDet = np.sum(np.log(np.diag(chCov))) # log sqrt(det)
        if g.verbose:
            print('g.verbose')
            print('in doLogLik chCov shape ', chCov.shape, ' w shape ', w.shape)
        # cho_solve cuts time almost in half compared to lstsq method
        p1 = scipy.linalg.cho_solve((chCov, True), w)
        L = -logDet - 0.5 * np.sum(p1 * w)
        return L

    # calculate the equivalent quadratic form of kron separable data
    def sepQuadFormCalc(V,zp):
        # calculate right side of the kronecker quadratic form solve
        dlen,mlen=zp.shape
        zpo=np.empty((dlen,mlen))
        for jj in range(mlen):
            zt=zp[:,jj]
            for ii in range(len(V)-1,-1,-1):
                Vsize=V[ii].shape[1]
                zt=np.linalg.solve(V[ii],zt.reshape((Vsize,int(dlen/Vsize)),order='F') ).T
            zpo[:,jj]=zt.reshape(-1,order='F')
        return zpo

    def doLogLikSep(Sigma, nugget, data):
        # eigen decomposition of the blocks
        V=[None]*len(Sigma)
        D=[None]*len(Sigma)
        for ii in range(len(Sigma)):
            D[ii], V[ii] = np.linalg.eigh(Sigma[ii])
            #V[ii]=np.flip(V[ii]) # these are needed for detailed numerical comparison to gpmsa of these
            #D[ii]=np.flip(D[ii]) #  (but doesn't lead to the correct final answer overall in python)
        # determinant from eigenvalues
        dkron=D[-1]
        for ii in range(len(D)-2,-1,-1):
            dkron=np.kron(D[ii],dkron)
        logDet=np.sum(np.log(dkron+nugget))

        #Log Likelihood
        zp=sepQuadFormCalc(V,data)
        Dki2=1/np.sqrt(dkron + nugget)
        zp2=zp * Dki2.T
        L=-0.5*logDet-0.5*(zp2.T @ zp2)   # here it is ...

        return L,V,Dki2

    num=g.num  # the model numerical components
    #extract some frequently accessed immuts
    n=num.n; m=num.m; p=num.p; q=num.q
    pu=num.pu; pv=num.pv
    pv=pv # temp to get the unused var lint to stop

    # The precomputation steps
    do_theta = do_betaV = do_lamVz = do_betaU = do_lamUz = do_lamWs = do_lamWOs = False
    if cvar == 'all':
        do_theta = do_betaV = do_lamVz = do_betaU = do_lamUz = do_lamWs = do_lamWOs = True
    elif cvar == 'theta': do_theta = True
    elif cvar == 'betaV': do_betaV = True
    elif cvar == 'lamVz': do_lamVz = True
    elif cvar == 'betaU': do_betaU = True
    elif cvar == 'lamUz': do_lamUz = True
    elif cvar == 'lamWs': do_lamWs = True
    elif cvar == 'lamWOs': do_lamWOs = True
    elif cvar == 'lamOs': pass
    else:
        raise ValueError('Invalid computeLogLik input cvar')

    # These are both related to recalculating x dists, whether with theta or not
    if num.sim_only and cvar == 'all': # calculating everything
        num.xDist = SepiaDistCov(g.data.x, cat_ind=[])
    elif do_theta:  # calculating everything involving theta [note this includes the case 'all']
        xt_tmp = np.concatenate((g.data.x, np.tile(g.params.theta.val, (n, 1))), axis=1)
        num.xDist = SepiaDistCov(xt_tmp, cat_ind=np.concatenate([g.data.x_cat_ind, g.data.t_cat_ind]))
        num.xzDist = SepiaDistCov(xt_tmp, g.data.zt, cat_ind=np.concatenate([g.data.x_cat_ind, g.data.t_cat_ind])) # the connection to theta variables

    # check if we're in a kron separable data definition
    if g.data.sep_design:
        ztSep=g.data.ztSep
        ztSepDist=g.num.ztSepDist
    else:
        ztSep=False

    # % Four parts to compute: Sig_v, Sig_u, Sig_w, and the Sig_uw crossterm

    if do_theta or do_betaU or do_lamUz or do_lamWs:
        SigU = []
        lamUz_val = g.params.lamUz.val
        lamWs_val = g.params.lamWs.val
        betaU_val = g.params.betaU.val
        for jj in range(pu):
            SigU.append(num.xDist.compute_cov_mat(betaU_val[:, jj], lamUz_val[0, jj]))
            np.fill_diagonal(SigU[jj], SigU[jj].diagonal() + 1/lamWs_val[0, jj])
        num.SigU = SigU
    else:
        SigU = num.SigU

    if (do_betaU or do_lamUz or do_lamWs or do_lamWOs):
        if   cvar == 'all': jinds = np.arange(pu)
        elif cvar == 'betaU': jinds = [int(np.ceil( (cindex+1)/(p+q) ) - 1)]
        elif cvar in ['lamUz', 'lamWs']: jinds = [cindex]
        elif cvar == 'lamWOs': jinds = np.arange(pu)

        lamUz_val = g.params.lamUz.val
        betaU_val = g.params.betaU.val
        if ztSep==False:
            ztDistCov = num.ztDist.compute_cov_mat
        LamSim = num.LamSim
        lamWOs_val = g.params.lamWOs.val
        lamWs_val = g.params.lamWs.val
        w = num.w
        for jj in jinds:
            if ztSep==False:  # not kronecker dataset
                cg = ztDistCov(betaU_val[:, jj], lamUz_val[0, jj])
                np.fill_diagonal(cg, cg.diagonal() + 1/(LamSim[jj] * lamWOs_val) + 1/lamWs_val[0, jj])
                # calculate the SigW likelihood for each block
                num.SigWl[jj] = doLogLik(cg, w[jj*m:(jj+1)*m, 0])
                # calculate the SigW inverse for each block
                if n > 0:  # only needed for a calibration model
                    if g.verbose:
                        print('In computeLogLik: shape of cg ', cg.shape)
                    num.SigWi[jj] = np.linalg.inv(cg)
            else: # kronecker dataset, compute as kron'd blocks
                segVarStart=0
                cg=[]
                for ii in range(len(ztSep)):
                    segVarInds=np.arange(segVarStart,segVarStart + ztSepDist[ii].p)
                    segVarStart=segVarStart+ ztSepDist[ii].p
                    if ii==0:  # ensure that lamUz is only counted once
                       cg.append(ztSepDist[ii].compute_cov_mat(betaU_val[segVarInds,jj],lamUz_val[0,jj]))
                    else:
                       cg.append(ztSepDist[ii].compute_cov_mat(betaU_val[segVarInds,jj],1))
                cgNugget=1/(LamSim[jj]*lamWOs_val) + 1/lamWs_val[0,jj]
                num.SigWl[jj], num.V[jj], num.Dki2[jj] = \
                    doLogLikSep(cg,cgNugget,w[jj*m:(jj+1)*m, 0:1])
  
    # The computation is decomposed into the likelihood of W,
    #  and the likelihood of VU|W. 

    # Compute the likelihood of the W part (have already done the blocks)
    LogLikW = np.sum(num.SigWl)

    # only if there are observations do we have the VU|W part. 
    if not num.sim_only:

        # Compute the likelihood of the VU|W
        if do_theta or do_betaU or do_lamUz:
            SigUW=[]
            betaU_val = g.params.betaU.val
            lamUz_val = g.params.lamUz.val
            for jj in range(pu):
                SigUW.append(num.xzDist.compute_cov_mat(betaU_val[:, jj:(jj + 1)], lamUz_val[0, jj]))
            num.SigUW = SigUW
        else:
            SigUW = num.SigUW

        # do these ops, on the block diagonal blocks:
        #    W=SigUW*model.SigWi;
        #    SigUgW=SigU-W*SigUW';
        W = [None]*pu
        SigUgW = [None]*pu
        SigWi = num.SigWi
        for ii in range(pu):
            if ztSep==False:
                W_tmp = SigUW[ii] @ SigWi[ii]
                W[ii] = W_tmp
                SigUgW[ii] = SigU[ii] - W_tmp @ SigUW[ii].T
            else:
                # computation for a kron/separable design
                zp=np.zeros( (m,n) )
                for jj in range(n):
                    zp[:,jj]=np.squeeze(sepQuadFormCalc(num.V[ii],SigUW[ii][jj,:].reshape(-1,1)))
                zp2=zp * num.Dki2[ii].T
                SigUgW[ii]=SigU[ii] - zp2.T @ zp2
                W[ii]=zp2.T
        
        if (do_betaV or do_lamVz) and pv > 0:
            SigV = []
            betaV_val = g.params.betaV.val
            lamVz_val = g.params.lamVz.val
            for jj in range(num.lamVzGnum):
                SigV.append(num.x0Dist.compute_cov_mat(betaV_val[:, jj:(jj + 1)], lamVz_val[0, jj]))
            num.SigV = SigV
        else:
            SigV = num.SigV

        #for scalar output: SigVUgW=[SigV+SigUgW] ...
        #                        + model.SigObs/lamOs;
        #otherwise:         SigVUgW=[SigV             zeros(n*pv,n*pu); ...
        #                            zeros(n*pu,n*pv) SigUgW         ] ...
        #                        + model.SigObs/lamOs;
        SigVUgW = num.SigObs/g.params.lamOs.val
        for ii in range(pv):
            SigVUgW[ii*n:(ii+1)*n, ii*n:(ii+1)*n] = \
                SigVUgW[ii*n:(ii+1)*n, ii*n:(ii+1)*n] + SigV[num.lamVzGroup[ii]]

        if num.scalar_out:
            for ii in range(pu):
                SigVUgW[ii*n:(ii+1)*n, ii*n:(ii+1)*n] = \
                    SigVUgW[ii*n:(ii+1)*n, ii*n:(ii+1)*n] + SigUgW[ii]
        else:
            for ii in range(pu):
                SigVUgW[n*pv+ii*n:n*pv+(ii+1)*n, n*pv+ii*n:n*pv+(ii+1)*n] = \
                    SigVUgW[n*pv+ii*n:n*pv+(ii+1)*n, n*pv+ii*n:n*pv+(ii+1)*n] + SigUgW[ii]

        # do this op: MuVUgW =W*model.w;
        MuVUgW = np.zeros((n*pu, 1))
        w = num.w
        for ii in range(pu):
            if ztSep==False:
                MuVUgW[ii*n:(ii+1)*n, 0] = W[ii] @ w[ii*m:(ii+1)*m, 0]
            else:
                # computation for a kron/separable design
                zp=sepQuadFormCalc(num.V[ii], w[ii*m:(ii+1)*m, 0].reshape(-1,1))
                zp2=zp * num.Dki2[ii].T
                MuVUgW[ii*n:(ii+1)*n, 0]=np.squeeze(W[ii] @ zp2)

        # for scalar output:  MuDiff=   [u] - [MuVUgW]
        # otherwise:          MuDiff= [v;u] - [0;MuVUgW] 
        if num.scalar_out:
            MuDiff = num.u
            MuDiff = MuDiff - MuVUgW
        else:
            #MuDiff=num.vu 
            #print( (MuDiff[pv*n:,0]).shape )
            #MuDiff[pv*n:,0]=MuDiff[pv*n:,0]-MuVUgW
            MuDiff = np.concatenate((num.v, num.u - MuVUgW), axis=0)
            # todo: is this a better operation than add-in place to
            #    the u component of pre-concatenated vu?

        # Now we can get the LL of VU|W
        LogLikVUgW = doLogLik(SigVUgW, MuDiff)

    else: #test on whether we have observations - not sim_only
        LogLikVUgW=0
  
    # Final Answer, LL(VU) = LL(VU|W) + LL(W)
    num.logLik = LogLikVUgW + LogLikW
    
    return num.logLik

