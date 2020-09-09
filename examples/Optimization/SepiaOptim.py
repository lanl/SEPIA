import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import pandas as pd
import sys

class SepiaOptim():
    """
    Optimization routines for optimizing the model posterior with respect to the model parameters.
    Implements two graident-free optimizers: Nelder-Mead simplex and particle swarm.

    :var sepia.SepiaModel model: instantiated SepiaModel
    :var list idx_to_transform: indices of parameters to transform for optimization

    """

    def __init__(self, model):
        """
        Initialize optimizer object.

        :param sepia.SepiaModel model: instantiated SepiaModel object

        """
        self.model = model
        self.idx_to_transform = []
    
    def set_model_params(self,p):
        """
        Set optimized parameters into model object.

        :param numpy.ndarray p: array of parameters as returned by nelder_mead or particle_swarm

        """
        i=0
        for prm in self.model.params.mcmcList:
            if prm.name == 'logPost': continue
            for ind in range(int(np.prod(prm.val_shape))):
                arr_ind = np.unravel_index(ind, prm.val_shape, order='F')
                prm.val[arr_ind] = p[i]
                i+=1
    
    def get_model_params(self):
        """
        Get model parameters into array form used for optimization.

        :return: numpy.ndarray of flattened/concatenated parameters in a form amenable to optimization
        """
        # Gets model parameters into array used for optimizer internally
        params = []
        i=0
        for prm in self.model.params.mcmcList:
            if prm.name == 'logPost': continue
            for ind in range(int(np.prod(prm.val_shape))):
                arr_ind = np.unravel_index(ind, prm.val_shape, order='F')
                params.append(prm.val[arr_ind])
                i+=1
        params = np.array(params)
        return np.array(params)
    
    def log_transform(self,x):
        # 1-1 log transformation, linear on [0,1]
        if x >= 1: return np.log(x)
        else: return x-1
    
    def inv_log_transform(self,x):
        # inverse of log_transform
        exp_idx = x>=0
        add_idx = x<0
        x[exp_idx]=np.exp(x[exp_idx])
        x[add_idx]= x[add_idx]+1
        return x
    
    def check_params_valid(self,x):
        # Check that proposed parameter values lie within required bounds
        valid=True
        i = 0
        for prm in self.model.params.mcmcList:
            if not prm.prior.is_in_bounds(x[i:(i+int(np.prod(prm.val_shape)))].reshape(prm.val_shape)):
                valid = False
                break
            i += int(np.prod(prm.val_shape))
            #for ind in range(int(np.prod(prm.val_shape))):
            #    if not prm.prior.is_in_bounds(x[i]): valid = False
            #    i+=1
        return valid
    
    def optim_logPost(self,x):
        #Wrapper for the optimization of logPost. Not called by users.
        #Checks that parameter values are in bounds, then updates model parameters
        #and returns the new logPost value.
        x_cpy = deepcopy(x)
        x_cpy[self.idx_to_transform] = self.inv_log_transform(x_cpy[self.idx_to_transform])
        if self.check_params_valid(x_cpy):
            i = 0
            for prm in self.model.params.mcmcList:
                if prm.name == 'logPost': continue
                for ind in range(int(np.prod(prm.val_shape))):
                    arr_ind = np.unravel_index(ind, prm.val_shape, order='F')
                    prm.val[arr_ind] = x_cpy[i]
                    i+=1
        else:
            return sys.maxsize - 1000 + np.random.uniform(0,1000)
            # this seems strange but we can't use np.inf as it causes problems with convergence criteria
            # we also dont want to return the same value twice in a row, as it may trigger convergence
            # so we want a value as large as possible, but also with a random fluctuation

        return -1*self.model.logPost()
    
    def particle_swarm(self,w_max=.9,w_min=.4,c1=.5,c2=.3,\
                             maxiter=1000,swarmsize=10,obj_tol=1e-8,step_tol=1e-8,
                            log_transform=None,verbose=True):
        """        
        Optimize model log posterior using particle swarm optimization. Intertial weight parameter w, decreases linearly from w_max to w_min by iteration number.

        :param float w_max: maximum inertial weight parameter
        :param float w_min: minimum intertial weight parameter
        :param float c1: personal parameter (how much particles are drawn to personal best)
        :param float c2: global parameter (how much particles are drawn to global best)
        :param int maxiter: maximum iterations
        :param int swarmsize: swarm size for PSO optimization
        :param float obj_tol: convergence tolerance - objective function value 
        :param float step_tol: convergence tolerance - particle position 
        :param list/NoneType log_transform: list of names of variables for which to apply log transform
        :param bool verbose: print optimization info
        :return: tuple containing: x_opt (optimal parameter values), f_opt (function value at optimum),
                                   f_hist (function value history), it (iterations to convergence), fnc_calls (objective function calls to convergence), p_native (optimal parameter values, undo log transform)
        """
        # don't want verbose model for optimizer but want to change back after
        was_verbose = False
        if self.model.verbose: 
            self.model.verbose=False
            was_verbose=True
        
        # get parameter indices for transform
        if log_transform:
            self.idx_to_transform.clear()
            i = 0
            for prm in self.model.params.mcmcList:
                for ind in range(int(np.prod(prm.val_shape))):
                    if prm.name in log_transform:
                        self.idx_to_transform.append(i)
                    i+=1

        lb = []
        ub = []
        i = 0
        names = []
        for prm in self.model.params.mcmcList:
            if prm.name == 'logPost': continue
            for ind in range(int(np.prod(prm.val_shape))):
                arr_ind = np.unravel_index(ind, prm.val_shape, order='F')
                if i in self.idx_to_transform:
                    lb.append(np.log(prm.prior.bounds[0][arr_ind]) if prm.prior.bounds[0][arr_ind] != 0 else -1)
                    if prm.name=='betaU' or prm.name=='betaV':
                        ub.append(3)
                    else:
                        ub.append(np.log(prm.prior.bounds[1][arr_ind]) if prm.prior.bounds[1][arr_ind] != np.inf else np.log(100000))
                else:
                    lb.append(prm.prior.bounds[0][arr_ind])
                    ub.append(prm.prior.bounds[1][arr_ind] if prm.prior.bounds[1][arr_ind] != np.inf else 100000)
                names.append(prm.name)
                i+=1

        if verbose: print('optimization bounds:',(lb,ub))
        x_opt, f_opt, f_hist, it, fnc_calls = pso(self.optim_logPost, lb, ub, maxiter=maxiter, \
                                                 minstep=step_tol, minfunc=obj_tol, swarmsize=swarmsize,\
                                                w_max=w_max, w_min=w_min,c1=c1, c2=c2)
        if verbose: print('max obj fnc:',f_opt)
        p_native=deepcopy(x_opt)
        p_native[self.idx_to_transform] = self.inv_log_transform(p_native[self.idx_to_transform])
        if verbose: print(pd.DataFrame(data={'param': names, 'opt value native': p_native}).to_string(index=False))
        self.verbose=was_verbose
        return x_opt, f_opt, f_hist, it, fnc_calls, p_native

    def nelder_mead(self,maxiter=1000,step_tol=.0001,obj_tol=.0001,\
                    log_transform=None,verbose=True,start='default'):
        """
        Optimize model log posterior using Nelder-Mead simplex algorithm.

        :param int maxiter: maximum iterations
        :param float obj_tol: convergence tolerance - objective function value
        :param float step_tol: convergence tolerance - simplex position 
        :param list/NoneType log_transform: list of names of variables for which to apply log transform
        :param bool verbose: print optimization info
        :param string start: how to choose start location (default: current model parameter values, random: random value in parameter space)
        :return: tuple containing: x_opt (optimal parameter values), lp_hist (log posterior history), p_native
        """
    
        # don't want verbose model for optimizer but want to change back after
        was_verbose = False
        if self.model.verbose: 
            self.model.verbose=False
            was_verbose=True
        
        # get parameter indices for transform
        if log_transform:
            self.idx_to_transform.clear()
            i = 0
            for prm in self.model.params.mcmcList:
                for ind in range(int(np.prod(prm.val_shape))):
                    if prm.name in log_transform:
                        self.idx_to_transform.append(i)
                    i+=1
        
        # get parameter bounds for random start
        lb = []
        ub = []
        if start=='random':
            i = 0
            for prm in self.model.params.mcmcList:
                if prm.name == 'logPost': continue
                for ind in range(int(np.prod(prm.val_shape))):
                    arr_ind = np.unravel_index(ind, prm.val_shape, order='F')
                    if i in self.idx_to_transform:
                        lb.append(np.log(prm.prior.bounds[0][arr_ind]) if prm.prior.bounds[0][arr_ind] != 0 else -1)
                        if prm.name=='betaU' or prm.name=='betaV':
                            ub.append(3)
                        else:
                            ub.append(np.log(prm.prior.bounds[1][arr_ind]) if prm.prior.bounds[1][arr_ind] != np.inf else np.log(100000))
                    elif prm.name=='lamUz':
                        lb.append(0)
                        ub.append(5)
                    else:
                        lb.append(prm.prior.bounds[0][arr_ind])
                        ub.append(prm.prior.bounds[1][arr_ind] if prm.prior.bounds[1][arr_ind] != np.inf else 100000)
                    i+=1

        # set initial values
        i = 0
        names = []
        x0 = []
        for prm in self.model.params.mcmcList:
            if prm.name == 'logPost': continue
            for ind in range(int(np.prod(prm.val_shape))):
                arr_ind = np.unravel_index(ind, prm.val_shape, order='F')
                if start=='default':
                    if i in self.idx_to_transform:
                        x0.append(self.log_transform(prm.val[arr_ind]))
                    else:
                        x0.append(prm.val[arr_ind])
                elif start=='random':
                    x0.append(np.random.uniform(lb[i],ub[i]))
                else:
                    raise ValueError('start must be either default, or random')
                names.append(prm.name)
                i+=1
        
        # callback function to store history at each iteration
        lp_hist = []
        param_hist = []
        def callback(x):
            fobj = self.optim_logPost(x)
            lp_hist.append(fobj)
            param_hist.append(x)

        # call optimizer
        x_opt = minimize(self.optim_logPost, x0, method='nelder-mead',callback=callback,
               options={'xatol': step_tol, 'fatol': obj_tol,'disp': True,'maxiter':maxiter, 'adaptive': True})
        
        # get opt params on original scale and return
        p_native = deepcopy(x_opt['x'])
        p_native[self.idx_to_transform] = self.inv_log_transform(p_native[self.idx_to_transform])
        if verbose: 
            print('logPost value:',x_opt['fun'])
            print(pd.DataFrame(data={'param': names, 'init value': x0, 'opt value native': p_native}).to_string(index=False))
        self.verbose=was_verbose
        return x_opt,lp_hist,p_native
    
###################### PARTICLE SWARM OPTIMIZATION ALGORITHM ##########################
# Based on pso.py from https://github.com/tisimst/pyswarm.

from functools import partial
import numpy as np
from tqdm import tqdm

def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)

def _is_feasible_wrapper(func, x):
    return np.all(func(x)>=0)

def _cons_none_wrapper(x):
    return np.array([0])

def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])

def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))
    
def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        swarmsize=10, w_max=0.9, w_min=.4, c1=0.5, c2=0.3, maxiter=100, 
        minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
        particle_output=False):
    """
    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    w_max : scalar
        Maximum particle velocity scaling factor (Default: 0.9)
    w_min : scalar
        Minimum particle velocity scaling factor (Default: 0.4)
    c1 : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    c2 : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.3)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and 
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    fg : scalar
        The objective value at ``g``
    fg_hist : array
        list of objective function value by iteration
    it : scalar
        number of iterations to convergence
    fnc_calls: scalar
        number of objective function calls to convergence
   
    """
   
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args, kwargs)
    
    # Check for constraint function(s) #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)
        
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value
    fg_hist = [] # store best objective value at every iteration
    fnc_calls = 0
    w = np.linspace(w_max,w_min,maxiter,endpoint=True) # linearly decreasing w
    # Initialize the particle's position
    x = lb + x*(ub - lb)

    # Calculate objective and constraints for each particle
    if processes > 1:
        fx = np.array(mp_pool.map(obj, x))
        fs = np.array(mp_pool.map(is_feasible, x))
    else:
        for i in range(S):
            fx[i] = obj(x[i, :]); fnc_calls += 1
            fs[i] = is_feasible(x[i, :])
            
    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()
       
    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)
       
    # Iterate until termination criterion met ##################################
    for it in tqdm(range(1,maxiter+1)):
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        v = w[it-1]*v + c1*rp*(p - x) + c2*rg*(g - x)
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku

        # Update objectives and constraints
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :]); fnc_calls+=1
                fs[i] = is_feasible(x[i, :])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}'\
                    .format(it, p[i_min, :], fp[i_min]))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min)**2))

            if np.abs(fg - fp[i_min]) <= minfunc:
                print('Stopping search: Swarm best objective change less than {:}'\
                    .format(minfunc))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min], fg_hist, it, fnc_calls
            elif stepsize <= minstep:
                print('Stopping search: Swarm best position change less than {:}'\
                    .format(minstep))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min], fg_hist, it, fnc_calls
            else:
                g = p_min.copy()
                fg = fp[i_min]
        
        fg_hist.append(fg)

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    
    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg, fg_hist, it, fnc_calls