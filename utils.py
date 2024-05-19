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

def log_normal_gauss_hermite(sigma, N,mu=None):
    # a. gauss hermite
    x,w = gauss_hermite(N,numpy=True)
 
    # b. log normal
    eta = np.exp(mu + sigma*x)
    eta_w = w/np.sqrt(np.pi)
 
    return eta, eta_w

def create_shocks(sigma_eta,mean_eta,N_eta,sigma_mu,mean_mu,N_mu,pi):
    """
    This function creates the shocks and weights for the permanent and transitory income shocks.
        Args:
            sigma_eta (float): standard deviation of the permanent income shock
            mean_eta (float): mean of the permanent income shock
            Npsi (int): number of nodes in the permanent income shock
            sigma_mu (float): standard
            mean_mu (float): mean of the transitory income shock
            N_mu (int): number of nodes in the transitory income shock
            pi (float): probability of a low income shock
    """
    
    # a. gauss hermite
    eta, eta_w = log_normal_gauss_hermite(sigma_eta, N_eta,mean_eta)
    mu, mu_w = log_normal_gauss_hermite(sigma_mu, N_mu,mean_mu)

    # b. add low inncome shock
    #   in order to deal with mu = 0 with probability pi, we add zero to the mu array and pi to the mu_w array
    if pi > 0:
        
        # a. weights
        mu_w *= (1.0-pi)
        mu_w = np.insert(mu_w,0,pi)

        # b. values 
        no_income = 0
        mu = (mu-no_income*pi) # not used in practice since income is zero with probability pi
        mu = np.insert(mu,0,no_income)

    
    # c. tensor product
    eta,mu = np.meshgrid(eta,mu,indexing='ij')
    eta_w,mu_w = np.meshgrid(eta_w,mu_w,indexing='ij')

    return eta.ravel(), eta_w.ravel(), mu.ravel(), mu_w.ravel(), eta.size


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