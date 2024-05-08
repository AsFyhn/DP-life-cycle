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
    # https://github.com/ThomasHJorgensen/Sensitivity/blob/master/GP2002/tools/misc.py
    # Setup specifications in class. 
    class par: pass
    par.beta = 0.95 # discount factor
    par.R = 1.05 # interest rate
    par.rho = 0.5 # risk aversion parameter in the utility function
    par.gamma1 = 0.07 # mpc for retiress. maybe 
    par.pi = 0.1 # probability of income shock 

    par.sigma_mu = 0.1
    par.sigma_eta = 0.1

    par.G = 1

    par.Tr = 65 # retirement age
    par.t0 = 25 # start working
    par.Tr_N = par.Tr - par.t0 # normalized
    par.t0_N = par.t0 - par.t0 # normalized

    # Gauss Hermite weights and points: # mu is transtoritory income shock and eta is permanent income shock
    par.order = 12
    
    par.eta, par.eta_w, par.mu, par.mu_w, par.Nshocks = create_shocks(
        sigma_psi=par.sigma_eta,
        Npsi=par.order,
        sigma_xi=par.sigma_mu,
        Nxi=par.order,
        pi=par.pi,
        mu=0,
        mu_psi=0,
        mu_xi=0)

    # Weights for each combination of shocks
    par.w = par.mu_w * par.eta_w
    assert (1-sum(par.w) < 1e-8), f'{par.w}'

    # Grid
    par.num_xhat = 50 #100 they use 100 in their paper

    #4. End of period assets
    par.xhat = 3
    par.grid_xhat = linspace_kink(x_min=1e-6,x_max=par.xhat,n=par.num_xhat, x_int=2)#nonlinspace(0 + 1e-6,par.xhat,par.num_xhat,phi=1.1) # for phi > 1 non-linear
    # Dimension of value function space
    par.dim = [par.num_xhat,par.Tr_N+1]
    
    return par

def create_shocks(sigma_psi,Npsi,sigma_xi,Nxi,pi,mu,mu_psi=None,mu_xi=None):
    
    # a. gauss hermite
    psi, psi_w = log_normal_gauss_hermite(sigma_psi, Npsi,mu_psi)
    xi, xi_w = log_normal_gauss_hermite(sigma_xi, Nxi,mu_xi)
 
    # b. add low inncome shock
    if pi > 0:
        # a. weights
        xi_w *= (1.0-pi)
        xi_w = np.insert(xi_w,0,pi)

        # b. values
        xi = (xi-mu*pi) #/ (1.0-pi)
        xi = np.insert(xi,0,mu)

    
    # c. tensor product
    psi,xi = np.meshgrid(psi,xi,indexing='ij')
    psi_w,xi_w = np.meshgrid(psi_w,xi_w,indexing='ij')

    return psi.ravel(), psi_w.ravel(), xi.ravel(), xi_w.ravel(), psi.size

def log_normal_gauss_hermite(sigma, N,mu=None):
    # a. gauss hermite
    x,w = gauss_hermite(N,numpy=True)
 
    # b. log normal
    psi = np.exp(mu + sigma*x)
    psi_w = w/np.sqrt(np.pi)
 
    return psi, psi_w

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