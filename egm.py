from utils import (
    nonlinspace,
    util,
    marg_util,
    inv_marg_util,
                   )
from scipy import interpolate
import numpy as np
import time

def solve(par): # solve the model
    # initialize solution class
    class sol: pass
    # update some parameters and grids
    shape = par.dim # [par.num_xhat,par.Tr_N+1]
    sol.c = np.empty(shape)
    sol.m = np.empty(shape)

    # setup grids
    # self.setup_grids()
    
    # b. backwards induction
    for t in reversed(range(par.Tr_N)):
        
        tic = time.time()
        
        # i. last working period
        if t == par.Tr_N-1:
            c, m = solve_bf_retirement(t,sol,par)
            sol.c[:,t] = c[:,t]
            sol.m[:,t] = m[:,t]
            del c, m
        # ii. all other periods
        else:
            c, m = solve_egm(t,sol,par) 
            sol.c[:,t] = c[:,t]
            sol.m[:,t] = m[:,t]
            del c, m
        # iii. print
        toc = time.time()
        # print(f' t = {t} solved in {toc-tic:.1f} secs')
    
    return sol

def solve_bf_retirement(t,sol,par):
    c_plus = (
        #par.gamma0 + 
        par.gamma1*par.R*(par.grid_xhat))/par.G[t] # growth factor is 1, so no indexation
    
    dU = marg_util(par.G[t]*c_plus,par)
    sol.c[:,t] = inv_marg_util(par.beta*par.R*dU,par)
    sol.m[:,t] = par.grid_xhat + sol.c[:,t]    
    return sol.c, sol.m

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
        fac = par.G[t]*par.eta[i]
        m_plus = (par.R/fac)*par.grid_xhat + par.mu[i]

        # interpolate next-period consumption
        interp = interpolate.interp1d(m_next,c_next, bounds_error=False, fill_value = "extrapolate")  # Interpolation function
        c_plus = interp(m_plus)
        c_plus = np.fmax(1.0e-10 , c_plus ) # consumption must be non-negative

        # expected marginal utility
        w = par.mu_w[i]*par.eta_w[i]
        Eu += w*marg_util(fac*c_plus,par) # In the original code they do not include all in fac as I do here

    # invert Euler equation
    sol.c[:,t] = inv_marg_util(par.beta*par.R*Eu,par) 
    sol.m[:,t] = par.grid_xhat + sol.c[:,t]

    return sol.c, sol.m
