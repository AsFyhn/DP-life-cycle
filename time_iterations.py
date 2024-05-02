from utils import (
    nonlinspace,
    util,
    marg_util,
    inv_marg_util,
                   )
from scipy import interpolate, optimize
import numpy as np


def solve_ti(par):
     # initialize solution class
    class sol: pass
    sol.C = np.zeros(par.dim)
    
    # Last period, consume everything
    sol.C[:,par.Tr_N] = par.gamma1 * par.grid_xhat
    
    # Loop over periods
    for i, t in enumerate(range(par.Tr_N-1,par.t0_N-1,-1)):  #from period T-2, until period 0 
            # Picking some arbitrary small starting value
            x0 = np.ones(par.num_xhat)*1.0e-7 
            
            # Define the objective function
            obj_fun = lambda x: euler_error_func(x,t,par,sol)

            # Find roots
            res = optimize.root(obj_fun, x0)
            x1 = res.x #Unpack roots
            if i==20:
                print(x0)

            # Handle corner solutions
            I = x1>par.grid_xhat # find indices where consumption is larger than assets
            x1[I] = par.grid_xhat[I] # set consumption to assets (consume everything)
            
            # final solution
            sol.C[:,t] = x1
    return sol

def euler_error_func(c,t,par,sol):
    
    #Find next period's assets
    if t == par.Tr_N-1:
        xhat_next_w = par.R * (par.grid_xhat - c)[:,np.newaxis] + np.zeros(12)[np.newaxis,:]
        xhat_next_wo = par.R * (par.grid_xhat - c)[:,np.newaxis] + np.zeros(12)[np.newaxis,:]
        eps_weight = par.eta_w
    else:
        xhat_next_w = par.R/par.G * (par.grid_xhat - c)[:,np.newaxis] * np.exp(par.eta)[np.newaxis,:] + np.exp(-par.mu)[np.newaxis,:]
        xhat_next_wo = par.R/par.G * (par.grid_xhat - c)[:,np.newaxis] * np.exp(par.eta)[np.newaxis,:]
        eps_weight = par.eta_w
    #Interpolate next period's consumption
    interp = interpolate.interp1d(par.grid_xhat,sol.C[:,t+1], bounds_error=False, fill_value = "extrapolate") 
    c_next_0 = interp(xhat_next_wo)
    c_next_1 = interp(xhat_next_w)
    if t==39:
         print(c_next_0, xhat_next_wo)
         print(c_next_1, xhat_next_w)
    # Calculate next period expected marginal utility
    EU_next0 = par.pi*np.sum(eps_weight[np.newaxis,:]*marg_util(c_next_0,par)*par.G*par.eta, axis=1)
    EU_next1 = (1-par.pi)*np.sum((eps_weight)[np.newaxis,:]*marg_util(c_next_1,par)*par.G*par.eta,axis=1)
    EU_next = EU_next0 + EU_next1 

    # Calculate current period marginal utility
    U_now = marg_util(c,par) 

    # Calculate Euler error
    euler_error = U_now-par.beta*par.R*EU_next

    return euler_error
