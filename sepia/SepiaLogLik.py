"""
Computing the Log Likelihood for a model.

Handles an indicator of what variable was changed, so that 
partial recomputation and updating of precomputed/saved
components is done where possible.

"""

import numpy as np
import scipy.linalg

from sepia.SepiaDistCov import SepiaDistCov

def compute_log_lik(g, cvar='all', cindex=None):
    """
    Compute log likelihood. Returns value and stores in num.logLik of model object.

    :param g: SepiaModel object
    :param cvar: name of changed variable (to avoid recomputing things)
    :param cindex: index (flattened) of changed variable (to avoid recomputing things)
    :return: numeric log likelihood value
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
    
    num=g.num  # the model numerical components
    #extract some frequently accessed immuts
    n=num.n; m=num.m; p=num.p; q=num.q
    pu=num.pu; pv=num.pv
    pv=pv # temp to get the unused var lint to stop

    # first time checks - allocate etc. 
    if num.SigWl is None: num.SigWl = [None]*pu
    if num.SigWi is None: num.SigWi = [None]*pu

    # The precomputation steps
    do_theta = do_betaV = do_lamVz = do_betaU = do_lamUz = do_lamWs = do_lamWOs = False
    if cvar is 'all':
        do_theta = do_betaV = do_lamVz = do_betaU = do_lamUz = do_lamWs = do_lamWOs = True
    elif cvar is 'theta': do_theta = True
    elif cvar is 'betaV': do_betaV = True
    elif cvar is 'lamVz': do_lamVz = True
    elif cvar is 'betaU': do_betaU = True
    elif cvar is 'lamUz': do_lamUz = True
    elif cvar is 'lamWs': do_lamWs = True
    elif cvar is 'lamWOs': do_lamWOs = True
    elif cvar is 'lamOs': pass
    else: print('Invalid computeLogLik input cvar')

    # These are both related to recalculating x dists, whether with theta or not
    if num.sim_only and cvar is 'all': # calculating everything
        #num.xDist = SepiaDistCov(g.data.x, cat_ind=g.data.x_cat_ind)
        num.xDist = SepiaDistCov(g.data.x, cat_ind=[])
    elif do_theta:  # calculating everything involving theta [note this includes the case 'all']
        xt_tmp = np.concatenate((g.data.x, np.tile(g.params.theta.val, (n, 1))), axis=1)
        num.xDist = SepiaDistCov(xt_tmp, cat_ind=np.concatenate([g.data.x_cat_ind, g.data.t_cat_ind]))
        num.xzDist = SepiaDistCov(xt_tmp, g.data.zt, cat_ind=np.concatenate([g.data.x_cat_ind, g.data.t_cat_ind])) # the connection to theta variables

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
        if   cvar is 'all': jinds = np.arange(pu)
        elif cvar is 'betaU': jinds = [int(np.ceil( (cindex+1)/(p+q) ) - 1)]
        elif cvar in ['lamUz', 'lamWs']: jinds = [cindex]
        elif cvar is 'lamWOs': jinds = np.arange(pu)

        lamUz_val = g.params.lamUz.val
        betaU_val = g.params.betaU.val
        ztDistCov = num.ztDist.compute_cov_mat
        LamSim = num.LamSim
        lamWOs_val = g.params.lamWOs.val
        lamWs_val = g.params.lamWs.val
        w = num.w
        for jj in jinds:
        #if isempty(data.ztSep)
            cg = ztDistCov(betaU_val[:, jj], lamUz_val[0, jj])
            np.fill_diagonal(cg, cg.diagonal() + 1/(LamSim[jj] * lamWOs_val) + 1/lamWs_val[0, jj])
            # calculate the SigW likelihood for each block
            num.SigWl[jj] = doLogLik(cg, w[jj*m:(jj+1)*m, 0])
            # calculate the SigW inverse for each block 
            if n > 0:  # only needed for a calibration model
                if g.verbose:
                    print('In computeLogLik: shape of cg ', cg.shape)
                num.SigWi[jj] = np.linalg.inv(cg)
        #else
        #    % there is a separable design, so compute these as kron'ed blocks
        #    segVarStart=1;
        #    for ii=1:length(data.ztSep)
        #        segVars=segVarStart:(segVarStart + model.ztSepDist{ii}.p-1);
        #        segVarStart=segVarStart+ model.ztSepDist{ii}.p;
        #        if (ii==1)  % ensure that lamUz is only counted once
        #        cg{ii}=gCovMat(model.ztSepDist{ii},betaU(segVars,jj),lamUz(jj));
        #        else
        #        cg{ii}=gCovMat(model.ztSepDist{ii},betaU(segVars,jj),1);
        #    cgNugget=1/(model.LamSim(jj)*lamWOs) + 1/lamWs(jj);
        #    [model.SigWl(jj), model.V(jj).mats, model.Dki2(jj).vec]= ...
        #        doLogLikSep(cg,cgNugget,model.w((jj-1)*m+1:jj*m)); 
  
  
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
            #if isempty(data.ztSep):
            W_tmp = SigUW[ii] @ SigWi[ii]
            W[ii] = W_tmp
            SigUgW[ii] = SigU[ii] - W_tmp @ SigUW[ii].T
            #else
            #    % computation for a kron/separable design
            #    zp=zeros(m,n);
            #    for jj=1:n
            #        zp(:,jj)=sepQuadFormCalc(model.V(ii).mats,SigUW(ii).mat(jj,:)');
            #    end
            #    %zp2=zp .* model.Dki2(ii).vec;
            #    zp2=bsxfun(@times,zp,model.Dki2(ii).vec);
            #    SigUgW(ii).mat=SigU(ii).mat - zp2'*zp2;
            #    W(ii).mat=zp2';
        
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
            # originally in gpmsa: used symetric indices of blkRange=(ii-1)*n+1:ii*n
            SigVUgW[ii*n:(ii+1)*n, ii*n:(ii+1)*n] = \
                SigVUgW[ii*n:(ii+1)*n, ii*n:(ii+1)*n] + SigV[num.lamVzGroup[ii]]

        if num.scalar_out:
            for ii in range(pu):
                # gpmsa: blkRange=(ii-1)*n+1:ii*n;
                SigVUgW[ii*n:(ii+1)*n, ii*n:(ii+1)*n] = \
                    SigVUgW[ii*n:(ii+1)*n, ii*n:(ii+1)*n] + SigUgW[ii]
        else:
            for ii in range(pu):
                # gpmsa: blkRange=n*pv+((ii-1)*n+1:ii*n);
                SigVUgW[n*pv+ii*n:n*pv+(ii+1)*n, n*pv+ii*n:n*pv+(ii+1)*n] = \
                    SigVUgW[n*pv+ii*n:n*pv+(ii+1)*n, n*pv+ii*n:n*pv+(ii+1)*n] + SigUgW[ii]

        # do this op: MuVUgW =W*model.w;
        MuVUgW = np.zeros((n*pu, 1))
        w = num.w
        for ii in range(pu):
            # gpmsa: blkRange1=(ii-1)*n+1:ii*n;
            # gpmsa: blkRange2=(ii-1)*m+1:ii*m;
            #if isempty(data.ztSep)
            # gpmsa: MuVUgW(blkRange1)=W(ii).mat*model.w(blkRange2);
            MuVUgW[ii*n:(ii+1)*n, 0] = W[ii] @ w[ii*m:(ii+1)*m, 0]
            #else
            #    % computation for a kron/separable design
            #    zp=sepQuadFormCalc(model.V(ii).mats,model.w(blkRange2));
            #    zp2=zp .* model.Dki2(ii).vec;
            #    MuVUgW(blkRange1)=W(ii).mat*zp2;

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

