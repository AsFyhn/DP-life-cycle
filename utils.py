import numpy as np

def util(c,par):
    """ Utility function"""
    return ((c**(1.0-par.rho))/(1.0-par.rho))

def marg_util(c,par):
    """ Marginal utility function"""
    return c**(-par.rho)

def inv_marg_util(u,par):
    """ Inverse of marginal utility function"""
    return u**(-1/par.rho)

def setup():
    # Setup specifications in class. 
    class par: pass
    par.beta = 0.90#0.98 # discount factor
    par.R = 1.034 # interest rate
    par.rho = 0.5 # risk aversion parameter in the utility function
    par.gamma1 = 0.07 # mpc for retiress. maybe 
    par.pi = 0.1 # probability of income shock 

    par.sigma_mu = 0.2
    par.sigma_eta = 0.2

    par.G = 1

    par.Tr = 65 # retirement age
    par.t0 = 25 # start working
    par.Tr_N = par.Tr - par.t0 # normalized
    par.t0_N = par.t0 - par.t0 # normalized

    # Gauss Hermite weights and points
    par.order = 12
    x,w = gauss_hermite(par.order,numpy=True)

    par.eta = np.sqrt(2)*par.sigma_eta*x
    par.eta_w = w/np.sqrt(np.pi)
    par.mu = np.sqrt(2)*par.sigma_mu*x
    par.mu_w = w/np.sqrt(np.pi)

    # Grid
    par.num_xhat = 50 #100 they use 100 in their paper

    #4. End of period assets
    par.xhat = 3
    par.grid_xhat = linspace_kink(x_min=1e-6,x_max=par.xhat,n=par.num_xhat, x_int=2)#nonlinspace(0 + 1e-6,par.xhat,par.num_xhat,phi=1.1) # for phi > 1 non-linear
    # Dimension of value function space
    par.dim = [par.num_xhat,par.Tr_N+1]
    
    return par

def gauss_hermite(n,numpy=True):
    if numpy:
        x,w = np.polynomial.hermite.hermgauss(n)
    else:
        # a. calculations
        i = np.arange(1,n)
        a = np.sqrt(i/2)
        CM = np.diag(a,1) + np.diag(a,-1)
        L,V = np.linalg.eig(CM)
        I = L.argsort()
        V = V[:,I].T

        # b. nodes and weights
        x = L[I]
        w = np.sqrt(np.pi)*V[:,0]**2
    return x,w

import numpy as np
def linspace_kink(x_min, x_max, n,x_int):
    """
    like np.linspace between with unequal spacing
    """
    # 1. recursion
    y = np.empty(n)
 
    y[0] = x_min
    for i in range(1, n//2):
        y[i] = y[i-1] + (x_int-y[i-1]) / (n-i)
    for i in range(n//2, n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)
    # 3. assert increaing
    assert np.all(np.diff(y) > 0)
 
    return y

def nonlinspace(x_min, x_max, n, phi):
    """ like np.linspace between with unequal spacing
    phi = 1 -> eqaul spacing
    phi up -> more points closer to minimum
    """
    assert x_max > x_min
    assert n >= 2
    assert phi >= 1
 
    # 1. recursion
    y = np.empty(n)
 
    y[0] = x_min
    for i in range(1, n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi
    
    # 3. assert increaing
    assert np.all(np.diff(y) > 0)
 
    return y