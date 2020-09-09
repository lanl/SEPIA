
import numpy as np
import itertools
import scipy.stats

from sepia.SepiaDistCov import SepiaDistCov

# TODO this is not finished!

def sensitivity(model, samples_dict=False, sampleset=False, ngrid=21, varlist=[], jelist=[], rg=None, option='mean'):

    # Extract things from model
    p = model.num.p
    q = model.num.q
    nv = p + q
    pu = model.num.pu
    m = model.num.m

    if not samples_dict: samples_dict = {p.name: p.mcmc_to_array(sampleset=sampleset, flat=True) for p in model.params.mcmcList}
    betaU = samples_dict['betaU']
    lamUz = samples_dict['lamUz']
    lamWs = samples_dict['lamWs']

    # Get x variable indices and ranges
    ii0 = None
    if rg is None:
        rg = np.zeros((nv, 2))
        rg[:, 1] = 1
    else:
        ii0 = np.where(rg[:, 0] == rg[:, 1])
        if ii0.shape[0] > 0:
            ii0 = np.setxor1d(range(nv), ii0)
            nv = ii0.shape[0]
            rg = rg[ii0, :]
    if nv == 0:
        print('error:  no free variables\n')
        return
    if ii0 is None:
        ii0 = range(nv)

    # Set up grid of x values
    xe = np.zeros((ngrid, nv))
    for ii in range(nv):
        xe[:, ii] = np.linspace(rg[ii, 0], rg[ii, 1], ngrid)

    # Get posterior mean/median/user defined values for betaU/lamUz/lamWOs
    if option == 'samples':
        npvec = len(sampleset)
    elif option == 'mean':
        betaU = np.mean(betaU, 0, keepdims=True)
        lamUz = np.mean(lamUz, 0, keepdims=True)
        lamWs = np.mean(lamWs, 0, keepdims=True)
        npvec = betaU.shape[0]
    elif option == 'median':
        betaU = np.median(betaU, 0, keepdims=True)
        lamUz = np.median(lamUz, 0, keepdims=True)
        lamWs = np.median(lamWs, 0, keepdims=True)
        npvec = betaU.shape[0]
    elif isinstance(option, dict):
        betaU = option['betaU']
        lamUz = option['lamUz']
        lamWs = option['lamWs']
        npvec = betaU.shape[0]
    else:
        print('invalid option (choose mean, median, or samples, or pass dict of values for betaU, lamUz, lamWs)')
        return

    # Set up varlist
    if varlist == 'all':
        varlist = list(itertools.combinations(range(nv), 2))

    # component Sens
    sim_xt = model.data.zt
    w = model.num.w.reshape((m, pu), order='F') # TODO check
    sa = []
    for ii in range(pu):
        bind = [ind + ii*nv for ind in ii0]
        betaU_sub = betaU[:, bind]
        lamUz_sub = lamUz[:, ii]
        lamWs_sub = lamWs[:, ii]
        sa.append(component_sens(sim_xt[:, ii0], w[:, ii], betaU_sub, lamUz_sub, lamWs_sub, xe, ngrid, varlist, jelist, rg)) # TODO match sig

    ymean=model.data.sim_data.orig_y_mean; ysd=model.data.sim_data.orig_y_sd;
    if np.isscalar(ysd): ysd = np.tile(ysd,ymean.shape[0])
    
    ksmm=model.data.sim_data.K.T
    lam = np.diag(np.matmul(ksmm.T,ksmm))
    sme=np.zeros((npvec,nv))
    ste=np.zeros((npvec,nv))
    vt=np.zeros(npvec)
    for ii in range(npvec):
        vt0=0
        for jj in range(pu):
            sme[ii,:]=sme[ii,:]+lam[jj]*sa[jj]['sme'][ii,:]*sa[jj]['vt'][ii]
            ste[ii,:]=ste[ii,:]+lam[jj]*sa[jj]['ste'][ii,:]*sa[jj]['vt'][ii]
            vt0=vt0+lam[jj]*sa[jj]['vt'][ii]
        sme[ii,:]=sme[ii,:]/vt0
        ste[ii,:]=ste[ii,:]/vt0
        vt[ii]=vt0

    smePm=np.squeeze(np.mean(sme,0))
    stePm=np.squeeze(np.mean(ste,0))
    
    if varlist:
        sie=np.zeros((npvec,len(varlist)))
        for ii in range(npvec):
            for jj in range(pu):
                sie[ii,:]=sie[ii,:]+lam[jj]*sa[jj]['sie'][ii,:]*sa[jj]['vt'][ii]
            sie[ii,:]=sie[ii,:]/vt[ii]
        siePm=np.squeeze(np.mean(sie,0))
        
    if jelist:
        sje=np.zeros(npvec,len(jelist))
        for ii in range(npvec):
            for jj in range(pu):
                sje[ii,:]=sje[ii,:]+lam[jj]*sa[jj]['sje'][ii,:]*sa[jj]['vt'][ii]
            sje[ii,:]=sje[ii,:]/vt[ii]
        sjePm=np.squeeze(np.mean(sje,0))
        
    # unscales
    e0=np.zeros(ksmm.shape[0])
    mef_m=np.zeros((pu, nv, ksmm.shape[0], ngrid))
    mef_sd=np.zeros((pu, nv, ksmm.shape[0], ngrid))
    
    meanmat=np.tile(ymean,(ngrid,1)).T
    ysdmat=np.tile(ysd,(ngrid,1)).T
    
    for jj in range(pu):
        e0=e0+ksmm[:,jj]*np.mean(sa[jj]['e0'])
        for kk in range(nv):
            mef_m[jj,kk,:,:]=np.kron(ksmm[:,jj],np.mean(sa[jj]['mef_m'][:,kk,:],0).reshape((ngrid,-1))).T*\
                                                                                    ysdmat+meanmat # element multiplication no matmul
            mef_sd[jj,kk,:,:]=np.sqrt(np.kron(ksmm[:,jj]**2,\
                                           np.var(sa[jj]['mef_m'][:,kk,:],0).reshape((ngrid,-1)))*\
                                      np.kron(ksmm[:,jj]**2,np.mean(sa[jj]['mef_v'][:,kk,:],0).\
                                           reshape(ngrid,-1))).T*ysdmat
    e0=e0*ysd+ymean
    
    a=mef_m.shape
    #print('a',a)
    tmef_m = np.reshape(np.sum(mef_m,0),(a[1],a[2],a[3]))
    for kk in range(nv):
        tmef_m[kk,:,:].reshape((a[2],a[3]))-(pu-1)*meanmat
    tmef_m.squeeze()
    tmef_sd=np.zeros((a[1],a[2],a[3]))
    for jj in range(nv):
        for kk in range(pu):
            tmef_sd[jj,:,:]=tmef_sd[jj,:,:].reshape((-1,ngrid))+np.kron(ksmm[:,kk]**2,\
                                                      np.reshape(np.mean(sa[kk]['mef_v'][:,jj,:],0),(ngrid,-1))).T
    tmp=np.zeros((npvec,a[1],a[3]))
    for ii in range(ksmm.shape[0]):
        for jj in range(nv):
            for kk in range(pu):
                tmp[:,jj,:]=tmp[:,jj,:].reshape((-1,ngrid))+ksmm[ii,kk]*\
                sa[kk]['mef_m'][:,jj,:].reshape((-1,ngrid))
            tmef_sd[jj,ii,:]+=np.var(tmp[:,jj,:],axis=0)
    for kk in range(nv):
        #print('final shape',tmef_sd[kk,:,:].shape)
        #print(tmef_sd[kk,:,:].reshape((a[2],a[3])).shape)
        #print(ysdmat.shape)
        tmef_sd[kk,:,:]=np.sqrt(tmef_sd[kk,:,:].reshape((a[2],a[3])))*ysdmat
    tmef_sd.squeeze()
    
    if varlist:
        jef_m=np.zeros((pu,len(varlist),ngrid*ksmm.shape[0],ngrid))
        jef_sd=np.zeros(jef_m.shape)
        meanmat=np.tile(ymead,(ngrid,ngrid))
        ysdmat=np.tile(ysd,(ngrid,ngrid))
        for jj in range(pu):
            for kk in range(len(varlist)):
                jef_m[jj,kk,:,:]=np.kron(np.mean(sa[jj]['jef_m'][:,kk,:,:],0).reshape((1,ngrid)),ksmm[:,jj])*\
                ysdmat+meanmat
                jef_sd[jj,kk,:,:]=np.sqrt(kron(np.var(sa[jj]['jef_m'][:,kk,:,:],axis=0).reshape((1,ngrid)),\
                                              ksmm[:,jj]**2)+\
                                         kron(np.mean(sa[jj]['jef_v'][:,kk,:,:],axis=0).reshape((1,ngrid)),\
                                             ksmm[:,jj]**2))*ysdmat
        a=jef_m.shape
        tjef_m=np.sum(jef_m,0).reshape((a[1],a[2],a[3]))
        for kk in range(len(varlist)):
            tjef_m[kk,:,:]=tjef_m[kk,:,:].reshape((a[2],a[3]))-(pu-1)*meanmat
        tjef_m.sqeeze()
        tjef_sd=np.zeros((a[1],a[2],a[3]))
        for jj in range(len(varlist)):
            for kk in range(pu):
                tjef_sd[jj,:,:]=tjef_sd[jj,:,:].reshape((1,ngrid))+\
                                kron(np.mean(sa[kk]['jef_v'][:,jj,:,:],axis=0).reshape((1,ngrid)),\
                                    ksmm[:,kk]**2)
        tmp=np.zeros((npvec,a[1],a[2]))
        for hh in range(ngrid):
            for ii in range(ksmm.shape[0]):
                for jj in range(len(varlist)):
                    for kk in range(pu):
                        tmp[:,jj,:]=tmp[:,jj,:].reshape((1,ngrid))+ksmm[ii,kk]*\
                                    sa[kk]['jef_m'][:,jj,hh,:].reshape((1,ngrid))
                    tjef_sd[jj,(hh-1)*ksmm.shape[0]+ii,:]+=np.var(tmp[:,jj,:],axis=0)
        for kk in range(len(varlist)):
            tjef_sd[kk,:,:]=np.sqrt(tjef_sd[kk,:,:].reshape((a[2],a[3]))*ysdmat)
        tjef_sd.squeeze()
    
    sens = {'sa':sa,\
            'totalMean':e0,\
            'totalVar':vt,\
            'smePm':smePm,\
            'stePm':stePm,\
            'mef_m':mef_m,\
            'mef_sd':mef_sd,\
            'tmef_m':tmef_m,\
            'tmef_sd':tmef_sd,\
           }
    if varlist:
        sens['siePm']=siePm
        sens['jef']=jef
        sens['tjef']=tjef
    if jelist:
        sens['sjePm']=sjePm
    return sens
            
def component_sens(x, y, beta, lamUz, lamWs, xe, ngrid, varlist, jelist, rg):

    diff = rg[:, 1] - rg[:, 0]
    nmcmc, p = beta.shape
    m = x.shape[0]

    # TODO skipping loading catInd from model for now since not part of model
    cat_ind = np.zeros(p)

    # Calculate x distances
    xdist = SepiaDistCov(x)
    xexdist = []
    xedist = []
    for ii in range(p):
        xexdist.append(SepiaDistCov(xe[:, ii][:, None], x[:, ii][:, None]))
        xedist.append(SepiaDistCov(xe[:, ii][:, None]))

    if len(varlist) > 0:
        for ii in range(len(varlist)):
            xte = np.array([(vi, vj) for vi in xe[:, varlist[ii][0]] for vj in xe[:, varlist[ii][1]]])
            xexdist.append(SepiaDistCov(xte, x[:, varlist[ii]]))
            xedist.append(SepiaDistCov(xte))

    # Calculate GP stuff
    P = np.zeros((nmcmc, m, m))
    Q = np.zeros((nmcmc, m, m))
    My = np.zeros((m, nmcmc))
    for ii in range(nmcmc):
        betaei = beta[ii, :]
        lamUzi = lamUz[ii]
        lamWsi = lamWs[ii]

        # eta cov for the data & prediction locations
        S = xdist.compute_cov_mat(betaei, lamUzi, lamWsi)

        P[ii, :, :] = np.linalg.inv(S)
        My[:, ii] = np.linalg.solve(S, y)
        Q[ii, :, :] = P[ii, :, :] - np.outer(My[:, ii], My[:, ii])

    # Compute variance and functions
    e0 = np.zeros(nmcmc)
    e2 = np.zeros(nmcmc)
    vt = np.zeros(nmcmc)
    sme = np.zeros((nmcmc, p))
    ste = np.zeros((nmcmc, p))
    mef_m = np.zeros((nmcmc, p, ngrid))
    mef_v = np.zeros((nmcmc, p, ngrid))
    u1 = np.zeros(m)
    u2 = np.zeros(m)
    if len(varlist) > 0:
        sie = np.zeros((nmcmc, len(varlist)))
        jef_m = np.zeros((nmcmc, len(varlist), ngrid, ngrid))
        jef_v = np.zeros((nmcmc, len(varlist), ngrid, ngrid))
        #ue = np.zeros(m)
    else:
        sie = None
        jef_m = None
        jef_v = None
    if len(jelist) > 0:
        sje = np.zeros((nmcmc, len(jelist)))
    else:
        sje = None
    for ii in range(nmcmc):
        betaei = beta[ii, :]
        lamUzi = lamUz[ii]
        lamWsi = lamWs[ii]
        # initial calculations
        c1 = calc1(betaei, diff)
        C2 = calc2(x, xdist, m, rg, betaei, diff)
        c3 = np.zeros((m,diff.shape[0]))
        for jj in range(m):
            c3[jj,:]=calc3(x[jj,:],rg,betaei,diff)
        u2=np.prod(c3,1)
        e2[ii]=np.prod(c1)/lamUzi-np.trace(np.squeeze(Q[ii,:,:])*\
                                         varf(m,p,[],C2,u2))/lamUzi**2
        e0[ii]=np.matmul(u2.T,My[:,ii])/lamUzi
        # total variance
        vt[ii]=1/lamUzi-np.trace(np.squeeze(Q[ii,:,:])*\
                                varf(m,p,np.arange(p),C2,[]))/lamUzi**2-e2[ii]
        # 1:p might be an index so we need an arrange from 0 to p-1
        # main/total effect indices; main effect functions
        for jj in range(p):
            Js=[jj]; ll=np.setxor1d(np.arange(p),Js)
            u1=np.prod(c3[:,ll],1); u4=np.prod(c1[ll])
            sme[ii,jj]=u4/lamUzi-np.trace(np.squeeze(Q[ii,:,:])*\
                                         varf(m,p,Js,C2,u1))/lamUzi**2-e2[ii]
            sme[ii,jj]=sme[ii,jj]/vt[ii]
            ME=etae(Js,x,u1,u4,xexdist[jj],xedist[jj],betaei,lamUzi,\
                   lamWsi,My[:,ii],np.squeeze(P[ii,:,:]))
            mef_m[ii,jj,:]=ME.m; mef_v[ii,jj,:]=ME.v
            ll=[jj]; Js=np.setxor1d(np.arange(p),ll); u2=np.prod(c3[:,ll],1)
            ste[ii,jj]=c1[ll]/lamUzi-np.trace(np.squeeze(Q[ii,:,:])*\
                                              varf(m,p,Js,C2,u2))/lamUzi**2-e2[ii]
            ste[ii,jj]=1-ste[ii,jj]/vt[ii]
        # two-factor interaction indices, joint effects
        if not varlist:
            for jj in range(len(varlist)):
                Js=varlist[jj,:]; ll=np.setxor1d(np.arange(p),Js)
                u3=np.prod(c3[:,ll],1); u5=np.prod(c1[ll])
                sie[ii,jj]=u5/lamUzi-np.trace(np.squeeze(Q[ii,:,:])*\
                                              varf(m,p,Js,C2,u3))/lamUzi**2-e2[ii]
                sie[ii,jj]=sie[ii,jj]/vt[ii]-sme[ii,varlist[jj,1]]-\
                     sme[ii,varlist[jj,2]]
                JE=etae(Js,x,u3,u5,xexdist[p+jj],xedist[p+jj],betaei,lamUzi,\
                       lamWsi,My[:,ii],np.squeeze(P[ii,:,:]))
                jef_m[ii,jj,:,:]=np.reshape(JE.m,(ngrid,ngrid))
                jef_v[ii,jj,:,:]=np.reshape(JE.v,(ngrid,ngrid))
        # joint effect indices
        if not jelist:
            for jj in range(len(jelist)):
                Js=jelist[jj]; ll=np.setxor1d(np.arange(p),Js)
                u6=np.prod(c3[:,ll],1); u7=np.prod(c1[ll])
                sje[ii,jj]=u7/lamUzi-np.trace(np.squeeze(Q[ii,:,:])*\
                                             varf(m,p,Js,C2,u6))/lamUzi**2-e2[ii]
                sje[ii,jj]=sje[ii,jj]/vt[ii]
                
    sa = {'e0':e0,\
          'vt':vt,\
          'sme':sme,\
          'ste':ste,\
          'mef_m':mef_m,\
          'mef_v':mef_v}
    if varlist:
        sa['sie']=sie
        sa['jef_m']=jef_m
        sa['jef_v']=jef_v
    if jelist:
        sa['sje']=sje
    return sa
        
def calc1(beta, diff):
    ncdf = scipy.stats.norm.cdf(np.sqrt(2 * beta) * diff)
    c1 = (np.sqrt(np.pi/beta) * diff * (2 * ncdf - 1) - (1./beta) * (1 - np.sqrt(2*np.pi) * ncdf)) / np.square(diff)
    return c1

def calc2(x ,xdist, m, rg, beta, diff):
    kk = 0
    C2 = np.zeros((int(m*(m+1)/2) + m, beta.shape[0]))
    for ii in range(m):
        for jj in range(ii+1, m):
            mp = (x[ii,:] + x[jj,:])/2
            di = np.where(np.logical_and(xdist.ind[0]==ii, xdist.ind[1]==jj))
            C2[kk, :] = calc3(mp, rg, 2*beta, diff) * np.exp(-beta * xdist.sqdist[di, :]/2)
            kk += 1
    for ii in range(m):
        kk += 1
        C2[kk, :] = calc3(x[ii, :], rg, 2.*beta, diff)
    return C2

def calc3(x, rg, beta, diff):
    ncdf1 = scipy.stats.norm.cdf(np.sqrt(2 * beta) * (rg[:, 1] - x))
    ncdf0 = scipy.stats.norm.cdf(np.sqrt(2 * beta) * (rg[:, 0] - x))
    c3 = np.sqrt(np.pi / beta) * (ncdf1 - ncdf0) / diff
    return c3

def varf(m,p,Js,C2,ef):
    kk=0; ll=np.setxor1d(np.arange(p),Js); Vf=np.zeros((m,m));
    for ii in range(m-1):
        for jj in range(ii,m):
            kk=kk+1; Vf[ii,jj]=1
            if len(Js) != 0: Vf[ii,jj]=np.prod(C2[kk,Js])
            if len(ll) != 0: Vf[ii,jj]=Vf[ii,jj]*ef[ii]*ef[jj]
    Vf=Vf+Vf.T;
    for ii in range(m):
        kk=kk+1; Vf[ii,ii]=1;
        if len(Js) != 0: Vf[ii,ii]=np.prod(C2[kk,Js])
        if len(ll) != 0: Vf[ii,ii]=Vf[ii,ii]*(ef[ii]**2)
    return Vf

class ee_struct:
    def __init__(self,m_dim,v_dim):
        self.m = np.zeros(m_dim)
        self.v = np.zeros(v_dim)
def etae(Js,x,ef,vf,xexdist,xedist,beta,lamUz,lamWs,My,P):
    nxe=xedist.n
    ee = ee_struct(m_dim=nxe,v_dim=nxe)
    Ct = xexdist.compute_cov_mat(beta[Js].T,lamUz)
    Ct = Ct*np.tile(ef.T,(nxe,1))
    ee.m=np.matmul(Ct,My)
    C = xedist.compute_cov_mat(beta[Js].T,lamUz,lamWs)
    ee.v=np.diag(C*vf-np.matmul(np.matmul(Ct,P),Ct.T))
    return ee

if __name__ == '__main__':
    from dev_test.setup_test_cases import *

    seed = 42.
    m = 20
    nt = 10
    n_pc = 4
    nx = 5
    n_mcmc = 100

    model, matlab_output = setup_multi_sim_only(m=m, nt=nt, nx=nx, n_pc=n_pc, seed=seed)
    model.do_mcmc(n_mcmc)

    #sensitivity(model)

    sensitivity(model, sampleset=[1, 2, 3, 4, 5], ngrid=10, varlist='all', rg=None, option='mean')






