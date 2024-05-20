import numpy as np
from scipy import interpolate

class Simulator:
    def __init__(self,par,sol,simN=10000):
        self.par = par
        self.simN = simN
        self.sol = sol
        self.main()
        # return self.sol
        
    def main(self):
        self.sim_setup()
        self.draw_random()
        self.simulate()
    def sim_setup(self,):
        """
        Simulate the data for the model
        """
        np.random.seed(2023) # set seed

        class sim: pass
        self.sim = sim

        self.sim.simN = self.simN
        self.shape = (self.par.Tr,self.sim.simN)


        self.sim.trans = np.empty(self.shape)
        self.sim.perm = np.empty(self.shape)
        self.sim.uni = np.empty(self.shape)
        self.sim.init_a = np.empty(self.sim.simN)


    def draw_random(self):
        """
        
        """
        
        self.sim.trans = np.random.normal(size=self.shape) 
        self.sim.perm  = np.random.normal(size=self.shape)

        self.sim.uni = np.random.uniform(0,1,size=self.shape)

        # c. initial wealth
        self.sim.init_a = np.random.normal(size=self.sim.simN) 

    def simulate(self,): # simulate the model
        
        # a. allocate
        # self.sim_shape = self.sim.shape 
        self.sim.c = np.empty(self.shape)
        self.sim.C = np.empty(self.shape)
        self.sim.C_avg = np.empty(self.par.Tr_N)
        self.sim.a = np.empty(self.shape)
        self.sim.m = np.empty(self.shape)
        self.sim.Y = np.empty(self.shape)
        self.sim.Y_avg = np.empty(self.par.Tr_N)
        self.sim.P = np.empty(self.shape)
        self.sim.S = np.empty(self.shape)
        
        self.sim.age = np.empty(self.shape)

        self.sim.init_P = self.par.init_P*np.ones(self.sim.simN) 

        # d. call
        for t in range(self.par.Tr_N):
            self.simulate_name_change(t,self.sim.trans[t,:],self.sim.perm[t,:],self.sim.uni[t,:])
            
            # avg consumption without zero-shocks
            I = self.sim.Y[t]>0
            self.sim.C_avg[t] = np.exp(np.mean(np.log(self.sim.C[t,I])))
            self.sim.Y_avg[t] = (np.mean(self.sim.Y[t]))


        
    def simulate_name_change(self,t,trans,perm,uni):
        
        c_sol = np.zeros(self.par.num_xhat+1)
        m_sol = np.zeros(self.par.num_xhat+1) 
        c_sol[1:self.par.num_xhat+1] = self.sol.c[:,t]
        m_sol[1:self.par.num_xhat+1] = self.sol.m[:,t]

        c = self.sim.c[t,:]
        # a. shocks variance 
        sigma_perm = self.par.sigma_eta
        sigma_trans = self.par.sigma_mu

        perm_shock = np.exp(sigma_perm*perm) # log-normal distribution with variance sigma_perm 
        trans_shock = np.exp(sigma_trans*trans)*(uni>self.par.pi) + 0*(uni<=self.par.pi)
        
        # par.init_P, par.mu_a_init, par.sigma_a_init*sim.init_a

        if t==0:
            self.sim.P[t,:] = self.sim.init_P*perm_shock
            initW = 0 #par.mu_a_init*np.exp(par.sigma_a_init*sim.init_a) 
            self.sim.m[t,:] = initW + trans_shock # a + y 
    
        else:
            self.sim.P[t] = self.par.G[t-1]*self.sim.P[t-1]*perm_shock
            fac = self.par.G[t-1]*perm_shock
            self.sim.m[t] = self.par.R*self.sim.a[t-1]/fac + trans_shock 

        # Income 
        self.sim.Y[t] =self. sim.P[t]*trans_shock

        # interpolate optimal consumption
        interp = interpolate.interp1d(m_sol,c_sol, bounds_error=False, fill_value = "extrapolate")
        c = interp(self.sim.m[t,:])
        self.sim.C[t,:] = c*self.sim.P[t,:]

        # end-of-period wealth and saving
        self.sim.a[t,:] = self.sim.m[t,:] - c
        
        if t>0:
            self.sim.S[t] = (self.sim.a[t]*self.sim.P[t] - self.sim.a[t-1]*self.sim.P[t-1]) # do not divide with R because I use A and not W

