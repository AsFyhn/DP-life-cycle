from utils import (
    nonlinspace,
    util,
    marg_util,
    inv_marg_util,
                   )
from scipy import interpolate
import numpy as np


def EGM_loop (sol,t,par):
    interp = interpolate.interp1d(sol.M[:,t+1],sol.C[:,t+1], bounds_error=False, fill_value = "extrapolate")  # Interpolation function

    for i_a,a in enumerate(par.grid_a): # Loop over end-of-period assets
        
        # Future m and c
        m_next = par.R * a + par.eps
        c_next = interp(m_next)
        
        # Future expected marginal utility
        EU_next = np.sum(par.eps_w*marg_util(c_next,par))

        # Current consumption
        c_now = inv_marg_util(par.R * par.beta * EU_next, par)
        
        # Index 0 is used for the corner solution, so start at index 1
        sol.C[i_a+1,t]= c_now
        sol.M[i_a+1,t]= c_now + a

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
    sol.a = np.nan + np.zeros(shape)
    
    # Last period, consume everything
    sol.a[:,par.Tr_N-1] = par.a.copy()
    sol.C[:,par.Tr_N-1]= sol.a[:,par.T-1].copy()

    # Loop over periods
    for t in range(par.Tr_N-2, par.t0_N, -1):  #from period T-2, until period 0, backwards
        if vector == True:
            sol = EGM_vectorized(sol, t, par)
        else:
            sol = EGM_loop(sol, t, par)
        # add zero consumption to account for borrowing constraint
        sol.a[0,t] = 0
        sol.C[0,t] = 0
    return sol