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

def setup(g_constant:bool=True):
    # https://github.com/ThomasHJorgensen/Sensitivity/blob/master/GP2002/tools/misc.py
    # Setup specifications in class. 
    class par: pass
    par.beta = 0.96 # discount factor
    par.R = 1.034 # interest rate
    par.rho = 0.514 # risk aversion parameter in the utility function
    par.gamma1 = 0.071 # mpc for retiress. maybe 
    par.pi = 0.00302 # probability of income shock 
     

    par.Tr = 65 # retirement age
    par.t0 = 26 # start working
    par.Tr_N = par.Tr - par.t0 + 1  # normalized
    par.t0_N = par.t0 - par.t0 # normalized

    if g_constant:
        par.G = np.ones(par.Tr_N) # constant growth rate
    else:
        grid_age = [float(age) for age in range(par.t0,par.Tr+1+1)]
        par.grid_age = np.array(grid_age)
        agep = np.empty((6,len(par.grid_age)))
        for i in range(6):
            agep[i,:] = par.grid_age**i

        # family shifter
        polF = np.array([0.0, 0.13964975, -0.0047742190, 8.5155210e-5, -7.9110880e-7, 2.9789550e-9]) # constant first (irrelevant)
        v = polF @ agep 

        # permanent income growth
        polY = np.array([6.8013936, 0.3264338, -0.0148947, 0.000363424, -4.411685e-6, 2.056916e-8]) # constant first
        Ybar = np.exp(polY @ agep ) # matrix multiplication
        print(f'Shape of Ybar {Ybar[par.Tr_N].shape}')
        print(f'Shape of first element {Ybar[1:(par.Tr_N+1)].shape}, and second element {Ybar[0:(par.Tr_N)].shape}')
        par.G = Ybar[1:(par.Tr_N+1)]/Ybar[0:par.Tr_N] # growth rate is shiftet forward, so g[t+1] is G[t] in code

    # Gauss Hermite weights and points: # mu is transtoritory income shock and eta is permanent income shock
    par.sigma_mu = 0.0440
    par.sigma_eta = 0.0212
    
    par.order = 12
    
    par.eta, par.eta_w, par.mu, par.mu_w, par.Nshocks = create_shocks(
        sigma_eta=par.sigma_eta,
        Npsi=par.order,
        sigma_mu=par.sigma_mu,
        Nxi=par.order,
        pi=par.pi,
        mu=0,
        mu_psi=0,
        mu_xi=0)

    # Weights for each combination of shocks
    par.w = par.mu_w * par.eta_w
    assert (1-sum(par.w) < 1e-8), f'{par.w}'

    # Grid
    par.num_xhat = 300 #100 they use 100 in their paper

    #4. End of period assets
    par.xhat = 3
    par.grid_xhat = linspace_kink(x_min=1e-6,x_max=par.xhat,n=par.num_xhat, x_int=1)#nonlinspace(0 + 1e-6,par.xhat,par.num_xhat,phi=1.1) # for phi > 1 non-linear
    # Dimension of value function space
    par.dim = [par.num_xhat,par.Tr_N+1]

    # simulation parameters
    par.init_P = 10
    
    return par

def create_shocks(sigma_eta,Npsi,sigma_mu,Nxi,pi,mu,mu_psi=None,mu_xi=None):
    
    # a. gauss hermite
    psi, psi_w = log_normal_gauss_hermite(sigma_eta, Npsi,mu_psi)
    xi, xi_w = log_normal_gauss_hermite(sigma_mu, Nxi,mu_xi)
 
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