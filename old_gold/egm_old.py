from old_gold.utils_old import (
    nonlinspace,
    util,
    marg_util,
    inv_marg_util,
                   )
from scipy import interpolate
import numpy as np


def EGM_loop (sol,t,par):
    interp = interpolate.interp1d(sol.xhat[:,t+1],sol.C[:,t+1], bounds_error=False, fill_value = "extrapolate")  # Interpolation function

    for i_xhat,xhat in enumerate(par.grid_xhat): # Loop over end-of-period assets
        # Future m and c
        if t == par.Tr_N:
            xhat_next_w = par.R * xhat 
            xhat_next_wo = par.R * xhat 
            eps_weight = 1
            print(xhat, xhat_next_w)
        else:
            xhat_next_w = par.R/par.G * xhat * np.exp(par.eta) + np.exp(-par.mu)
            xhat_next_wo = par.R/par.G * xhat * np.exp(par.eta)
            eps_weight = par.eta_w

        c_next_wo = par.pi*interp(xhat_next_wo) * par.G * np.exp(-par.eta) 
        c_next_w = (1-par.pi)*interp(xhat_next_w) * par.G * np.exp(-par.eta)
        
        # Future expected marginal utility
        EU_next = np.sum(eps_weight*marg_util(c_next_w,par))+np.sum(eps_weight**2*marg_util(c_next_wo,par)) # gauss hermite weights: look into whether this is correct

        # Current consumption
        c_now = inv_marg_util(par.R * par.beta * EU_next, par)
        sol.C[i_xhat,t]= c_now
        sol.xhat[i_xhat,t]= xhat

    return sol

def EGM_vectorized (sol,t,par):

    interp = interpolate.interp1d(sol.M[:,t+1],sol.C[:,t+1], bounds_error=False, fill_value = "extrapolate") # Interpolation function
    
    # Future m and c
    m_next = par.R*par.grid_a[:,np.newaxis] + par.eps[np.newaxis,:] # Next period assets  
    c_next = interp(m_next)
    
    # Future expected marginal utility
    EU_next = np.sum(par.eps_w[np.newaxis,:] * marg_util(c_next,par),axis=1)
    
    # Current consumption
    c_now = inv_marg_util(par.beta * par.R * EU_next,par)
    
    # Index 0 is used for the corner solution, so start at index 1
    sol.C[1:,t] = c_now
    sol.M[1:,t] = c_now + par.grid_a
    return sol


def solve_EGM(par, vector = False):
     # initialize solution class
    class sol: pass
    shape = par.dim
    sol.C = np.nan + np.zeros(shape)
    sol.xhat = np.nan + np.zeros(shape)
    
    # Last period, consume everything
    sol.xhat[:,len(sol.C[0,:])-1] = par.grid_xhat.copy()
    sol.C[:,len(sol.C[0,:])-1]= par.gamma1 * sol.xhat[:,len(sol.C[0,:])-1]

    # Loop over periods
    for t in range(par.Tr_N-1, par.t0_N-1, -1):  #from period T-2, until period 0, backwards
        if vector == True:
            sol = EGM_vectorized(sol, t, par)
        else:
            sol = EGM_loop(sol, t, par)
        # add zero consumption to account for borrowing constraint
        # sol.a[0,t] = 0
        # sol.C[0,t] = 0
    return sol
