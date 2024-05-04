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
            # if t < par.Tr_N-1:
            #     break
            # Picking some arbitrary small starting value
            x0 = np.ones(par.num_xhat)*1.0e-7 
            
            # Define the objective function
            obj_fun = lambda x: euler_error_func(x,t,par,sol)

            # Find roots
            res = optimize.root(obj_fun, x0,method='lm',options={'maxiter':10000})
            if not res.success: raise Exception(res.message, t)
            x1 = res.x #Unpack roots

            # # Handle corner solutions
            # I = x1>par.grid_xhat # find indices where consumption is larger than assets
            # x1[I] = par.grid_xhat[I] # set consumption to assets (consume everything)
            
            # final solution
            sol.C[:,t] = x1
    return sol

def euler_error_func(c,t,par,sol):
    #Find next period's assets
    gh_node_pos = gh_node_neg = np.ones(len(par.mu))
    gh_weight = np.ones(len(par.mu))
    if t<par.Tr_N-1:
        gh_node_pos = gh_node_pos*np.exp(par.mu)
        gh_node_neg = gh_node_neg*np.exp(-par.mu)
        gh_weight = gh_weight*par.mu_w

    xhat_next1 = par.R/par.G * (par.grid_xhat - c)[:,np.newaxis]*(np.exp(par.eta))[np.newaxis,:] + (np.exp(-par.mu))[np.newaxis,:] 
    xhat_next2 = par.R/par.G * (par.grid_xhat - c)[:,np.newaxis]*(np.exp(par.eta))[np.newaxis,:] 
    #Interpolate next period's consumption
    interp = interpolate.interp1d(par.grid_xhat,sol.C[:,t+1], bounds_error=False, fill_value = "extrapolate") 
    c_next1 = interp(xhat_next1)*par.G*(np.exp(-par.eta))[np.newaxis,:]
    c_next2 = interp(xhat_next2)*par.G*(np.exp(-par.eta))[np.newaxis,:]
    
    # EU_next0 = par.pi*np.sum(eps_weight[np.newaxis,:]*marg_util(c_next_0,par)*par.G*par.eta, axis=1)
    # EU_next1 = (1-par.pi)*np.sum((eps_weight)[np.newaxis,:]*marg_util(c_next_1,par)*par.G*par.eta,axis=1)

    EU_next = (
         par.pi  * np.sum(gh_weight[np.newaxis,:]*(1/(np.pi))*marg_util(c_next1,par),axis=1) 
         + (1-par.pi) * np.sum(gh_weight[np.newaxis,:]* (1/(np.pi)) *marg_util(c_next2,par),axis=1))
    # print(EU_next)
    
    
    # Calculate current period marginal utility
    U_now = marg_util(c,par) 
    # print(f'Shape of EU_next: {EU_next.shape}, shape of U_now: {U_now.shape}')
    # Calculate Euler error
    euler_error = U_now-par.beta*par.R*EU_next
    
    return euler_error
