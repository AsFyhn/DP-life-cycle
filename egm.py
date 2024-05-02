from utils import (
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
            xhat_next = par.R * xhat 
            eps_weight = 1
        else:
            xhat_next = par.R/par.G * xhat * par.eps_eta + par.eps_mu
            eps_weight = par.eps_w
        c_next = interp(xhat_next)
        
        # Future expected marginal utility
        EU_next = np.sum(eps_weight*marg_util(c_next,par))

        # Current consumption
        c_now = inv_marg_util(par.R * par.beta * EU_next, par)
        
        # Index 0 is used for the corner solution, so start at index 1
        sol.C[i_xhat+1,t]= c_now
        sol.xhat[i_xhat+1,t]= xhat

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
    # shape = par.dim
    shape = [10,40+1]
    sol.C = np.nan + np.zeros(shape)
    sol.xhat = np.nan + np.zeros(shape)
    
    # Last period, consume everything
    sol.xhat[:,par.Tr_N+1] = par.grid_xhat.copy()
    sol.C[:,par.Tr_N+1]= par.gamma * sol.xhat[:,par.Tr_N+1]

    # Loop over periods
    for t in range(par.Tr_N-1, par.t0_N, -1):  #from period T-2, until period 0, backwards
        print(t)
        if vector == True:
            sol = EGM_vectorized(sol, t, par)
        else:
            sol = EGM_loop(sol, t, par)
        # add zero consumption to account for borrowing constraint
        sol.a[0,t] = 0
        sol.C[0,t] = 0
    return sol

C = np.nan + np.zeros([10,41])
C[:,41]
for i in range(40,0,-1):
    print(i)