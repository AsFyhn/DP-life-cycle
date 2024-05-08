from thj_utils import (
    nonlinspace,
    util,
    marg_util,
    inv_marg_util,
                   )
from scipy import interpolate
import numpy as np
import time

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


def solve(par): # solve the model
    # initialize solution class
    class sol: pass
    # update some parameters and grids
    shape = par.dim
    sol.c = np.empty(shape)
    sol.m = np.empty(shape)

    # setup grids
    # self.setup_grids()
    
    # b. backwards induction
    for t in reversed(range(par.Tr_N)):
        
        tic = time.time()
        
        # i. last working period
        if t == par.Tr_N-1:
            solve_bf_retirement(t,sol,par)

        # ii. all other periods
        else:
            solve_egm(t,sol,par) 
            
        # iii. print
        toc = time.time()
        # print(f' t = {t} solved in {toc-tic:.1f} secs')
    
    return sol.c, sol.m

def solve_bf_retirement(t,sol,par):
    c_plus = (
        #par.gamma0 + 
        par.gamma1*par.R*(par.grid_xhat))/par.G # growth factor is 1, so no indexation
    
    dU = marg_util(par.G*c_plus,par)
    sol.c[:,t] = inv_marg_util(par.beta*par.R*dU,par)
    sol.m[:,t] = par.grid_xhat + sol.c[:,t]    

def solve_egm(t,sol,par):

    c_next = np.zeros(par.num_xhat+1)
    m_next = np.zeros(par.num_xhat+1) 
    c_next[1:par.num_xhat+1] = sol.c[:,t+1]
    m_next[1:par.num_xhat+1] = sol.m[:,t+1]

    c_plus = np.empty(par.num_xhat)

    # loop over shocks
    Eu = np.zeros(par.num_xhat)
    for i in range(par.Nshocks):

        # next-period resources
        fac = par.G*par.eta[i]
        m_plus = (par.R/fac)*par.grid_xhat + par.eta[i]

        # interpolate next-period consumption
        interp = interpolate.interp1d(m_next,c_next, bounds_error=False, fill_value = "extrapolate")  # Interpolation function
        c_plus = interp(m_plus)
        c_plus = np.fmax(1.0e-6 , c_plus ) # consumption must be non-negative

        # expected marginal utility
        w = par.mu_w[i]*par.eta_w[i]
        Eu += w*marg_util(fac*c_plus,par) # In the original code they do not include all in fac as I do here

    # invert Euler equation
    sol.c[:,t] = inv_marg_util(par.beta*par.R*Eu,par) 
    sol.m[:,t] = par.grid_xhat + sol.c[:,t]
