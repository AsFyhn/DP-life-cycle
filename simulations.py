import numpy as np
from scipy import interpolate

def sim_setup(par,simN=10000):
    """
    Simulate the data for the model
    """
    class sim: pass
    np.random.seed(2023) 
    sim.simN = simN
    shape = (par.Tr,sim.simN)


    sim.trans = np.empty(shape)
    sim.perm = np.empty(shape)
    sim.uni = np.empty(shape)
    sim.init_a = np.empty(sim.simN)

    return sim


def draw_random(par,sim):
    """
    
    """
    shape = sim.shape = (par.Tr_N,sim.simN)
    sim.trans = np.random.normal(size=shape) 
    sim.perm  = np.random.normal(size=shape)

    sim.uni = np.random.uniform(0,1,size=shape)

    # c. initial wealth
    sim.init_a = np.random.normal(size=sim.simN) 

def simulate(sim, par, sol): # simulate the model
    
    # a. allocate
    sim_shape = sim.shape 
    sim.c = np.empty(sim_shape)
    sim.C = np.empty(sim_shape)
    sim.C_avg = np.empty(par.Tr_N)
    sim.m = np.empty(sim_shape)
    sim.a = np.empty(sim_shape)
    sim.Y = np.empty(sim_shape)
    sim.Y_avg = np.empty(par.Tr_N)
    sim.P = np.empty(sim_shape)
    sim.S = np.empty(sim_shape)
    
    sim.age = np.empty(sim_shape)

    sim.init_P = par.init_P*np.ones(sim.simN) 

    # d. call
    for t in range(par.Tr_N):
        simulate_name_change(t,par,sol,sim,sim.trans[t,:],sim.perm[t,:],sim.uni[t,:])
        
        # avg consumption without zero-shocks
        I = sim.Y[t]>0
        sim.C_avg[t] = np.exp(np.mean(np.log(sim.C[t,I])))
        sim.Y_avg[t] = (np.mean(sim.Y[t]))
    return sim

    
def simulate_name_change(t,par,sol,sim,trans,perm,uni):
    
    c_sol = np.zeros(par.num_xhat+1)
    m_sol = np.zeros(par.num_xhat+1) 
    c_sol[1:par.num_xhat+1] = sol.c[:,t]
    m_sol[1:par.num_xhat+1] = sol.m[:,t]

    c = sim.c[t,:]
    # a. shocks variance 
    sigma_perm = par.sigma_eta
    sigma_trans = par.sigma_mu

    perm_shock = np.exp(sigma_perm*perm) # log-normal distribution with variance sigma_perm 
    trans_shock = np.exp(sigma_trans*trans)*(uni>par.pi) + 0*(uni<=par.pi)
    
    # par.init_P, par.mu_a_init, par.sigma_a_init*sim.init_a

    if t==0:
        sim.P[t,:] = sim.init_P*perm_shock
        initW = 0 #par.mu_a_init*np.exp(par.sigma_a_init*sim.init_a) 
        sim.m[t,:] = initW + trans_shock # a + y 
   
    else:
        sim.P[t] = par.G[t-1]*sim.P[t-1]*perm_shock
        fac = par.G[t-1]*perm_shock
        sim.m[t] = par.R*sim.a[t-1]/fac + trans_shock 

    # Income 
    sim.Y[t] = sim.P[t]*trans_shock

    # interpolate optimal consumption
    interp = interpolate.interp1d(m_sol,c_sol, bounds_error=False, fill_value = "extrapolate")
    c = interp(sim.m[t,:])
    sim.C[t,:] = c*sim.P[t,:]

    # end-of-period wealth and saving
    sim.a[t,:] = sim.m[t,:] - c
    
    if t>0:
        sim.S[t] = (sim.a[t]*sim.P[t] - sim.a[t-1]*sim.P[t-1]) # do not divide with R because I use A and not W

