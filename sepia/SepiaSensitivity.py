
import numpy as np
import itertools
import scipy.stats

from SepiaDistCov import SepiaDistCov

# TODO this is not finished!

def sensitivity(model, sampleset=False, ngrid=21, varlist=[], jelist=[], rg=None, option='mean'):

    # Extract things from model
    p = model.num.p
    q = model.num.q
    nv = p + q
    pu = model.num.pu
    m = model.num.m

    samples_dict = {p.name: p.mcmc_to_array(sampleset=sampleset, flat=True) for p in model.params.mcmcList}
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

    print('fin')

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
        ue = np.zeros(m)
    if len(jelist) > 0:
        sje = np.zeros((nmcmc, len(jelist)))
    for ii in range(nmcmc):
        betaei = beta[ii, :]
        lamUzi = lamUz[ii]
        lamWsi = lamWs[ii]
        # initial calculations
        c1 = calc1(betaei, diff)
        C2 = calc2(x, xdist, m, rg, betaei, diff)
        # TODO stopping here, at matlab gSens line 322

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






