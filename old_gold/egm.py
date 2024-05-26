from utils import (
    nonlinspace,
    util,
    marg_util,
    inv_marg_util,
                   )
from scipy import interpolate
import numpy as np
import time

class solver:
    def __init__(self,par) -> None:
        self.par = par
        # initialize solution class
        class sol: pass
        self.sol = sol
        par = self.par
        # a. allocate memory
        shape = par.dim 
        self.sol.c = np.empty(shape)
        self.sol.m = np.empty(shape)

    def solve(self,do_print=False): 
        """
        Solve the model using the endogenous grid method
        """
        par = self.par
    
        # b. backwards induction
        for t in reversed(range(par.Tr_N)):
            
            tic = time.time()
            
            # i. last working period
            if t == par.Tr_N-1:
                self.solve_bf_retirement(t)
            # ii. all other periods
            else:
                self.solve_egm(t) 

            # iv. print
            toc = time.time()
            if do_print:
                print(f' t = {t} solved in {toc-tic:.1f} secs')

    def solve_bf_retirement(self,t):
        par = self.par

        c_plus = (par.gamma1*par.R*(par.grid_xhat-par.xmin))/par.G[t] # par.xmin is 0 in GP but not in LC
        
        dU = marg_util(par.G[t]*c_plus,par)
        self.sol.c[:,t] = inv_marg_util(par.betas*par.R*dU,par)
        self.sol.m[:,t] = par.grid_xhat + self.sol.c[:,t]    

        # check if consumption is equal in the two states
        if par.betas[0] == par.betas[1]:
            if not np.allclose(self.sol.c[:,t,0],self.sol.c[:,t,1]):
                raise ValueError('The consumption in the two states are not equal')
        else:
            if np.allclose(self.sol.c[:,t,0],self.sol.c[:,t,1]):
                raise ValueError('The consumption in the two states are equal') 



    def solve_egm(self,t):
        par = self.par
        # a. initialize
        c_next = np.zeros((par.num_xhat+1,2))
        m_next = np.zeros((par.num_xhat+1,2)) + par.xmin
        c_next[1:par.num_xhat+1,:] = self.sol.c[:,t+1,:]
        m_next[1:par.num_xhat+1,:] = self.sol.m[:,t+1,:]

        c_plus = np.empty((par.num_xhat,2))
        m_plus = np.empty((par.num_xhat,2))

        # loop over shocks
        Eu = np.zeros((par.num_xhat,2))
        for i in range(par.Nshocks):
            # next-period resources
            fac = par.G[t]*par.eta[i]
            m_plus = (par.R/fac)*par.grid_xhat + par.mu[i]

            # interpolate next-period consumption
            for consumer in range(m_next.shape[1]):
                interp = interpolate.interp1d(m_next[:,consumer],c_next[:,consumer], bounds_error=False, fill_value = "extrapolate")
                c_plus[:,consumer] = interp(m_plus[:,0]) # the slicing of m_plus is to make it a 1D array
                c_plus[:,consumer] = np.fmax(1.0e-10 , c_plus[:,consumer] ) # consumption must be non-negative
            # expected marginal utility
            # w = par.mu_w[i]*par.eta_w[i]
            w = par.w[i]
            Eu += w*marg_util(fac*c_plus,par) 
        # invert Euler equation
        self.sol.c[:,t] = inv_marg_util(par.betas*par.R*Eu,par) 
        self.sol.m[:,t] = par.grid_xhat + self.sol.c[:,t]


