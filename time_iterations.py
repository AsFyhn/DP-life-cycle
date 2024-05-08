from old_gold.utils_old import (
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
    shape = par.dim # [par.num_xhat,par.Tr_N+1]
    sol.c = np.empty(shape)
    sol.m = np.empty(shape)
    
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
            sol.c[:,t] = x1
    return sol

def euler_error_func(c,t,par,sol):
    #Find next period's assets
    if t == par.Tr_N-1:
        EU_next = expected_utility_bf_retirement(c,t,par,sol)
    else:
        EU_next = expected_utility(c,t,par,sol)

    # Calculate current period marginal utility
    U_now = marg_util(c,par) 
    # print(f'Shape of EU_next: {EU_next.shape}, shape of U_now: {U_now.shape}')
    # Calculate Euler error
    euler_error = U_now-par.beta*par.R*EU_next
    
    return euler_error
    
def expected_utility_bf_retirement(c,t,par,sol):
    c_plus = (par.gamma1*par.R*(par.grid_xhat))/par.G # growth factor is 1, so no indexation
    dU = marg_util(par.G*c_plus,par)
    return dU

def expected_utility(c,t,par,sol):
    c_next = np.zeros(par.num_xhat+1)
    m_next = np.zeros(par.num_xhat+1) 
    c_next[1:par.num_xhat+1] = sol.c[:,t+1]
    m_next[1:par.num_xhat+1] = sol.m[:,t+1]

    fac = par.G*par.eta
    m_plus = (par.R/fac)*m_next[1:par.num_xhat+1] + par.mu

    interp = interpolate.interp1d(m_next,c_next, bounds_error=False, fill_value = "extrapolate")  # Interpolation function
    c_plus = interp(m_plus)
    c_plus = np.fmax(1.0e-10 , c_plus ) # consumption must be non-negative

    # expected marginal utility
    w = par.mu_w*par.eta_w
    Eu = w*marg_util(fac*c_plus,par) # In the original code they do not include all in fac as I do here
    return Eu
