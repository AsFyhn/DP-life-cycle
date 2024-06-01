import numpy as np
from scipy import interpolate

class Simulator:
    def __init__(self,par,sol,simN=10_000):
        self.par = par
        self.simN = simN
        self.sol = sol
        # self.main()
        # return self.sol
        
    def main(self):
        self.sim_setup()
        self.draw_random()
        sim_res = self.simulate()
        return sim_res

    def sim_setup(self,):
        """
        Simulate the data for the model
        """
        np.random.seed(2023) # set seed

        class sim: pass
        self.sim = sim

        self.sim.simN = self.simN
        self.shape = (self.par.Tr_N,self.sim.simN)


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
        single_shape = (self.shape[0],self.shape[1])
        shape = (self.shape[0],self.shape[1],2)
        avg_shape = (self.shape[0])
        # add extra dimension to the shape
        self.sim.c = np.empty(shape)
        self.sim.C = np.empty(shape)
        self.sim.C_var = np.empty(avg_shape)
        self.sim.C_avg_type1 = np.empty(avg_shape)
        self.sim.C_avg_type2 = np.empty(avg_shape)
        self.sim.C_avg = np.empty(avg_shape)
        
        self.sim.a = np.empty(shape)
        self.sim.A = np.empty(shape)
        self.sim.m = np.empty(shape)
        self.sim.M = np.empty(shape)
        self.sim.Y = np.empty(shape)
        self.sim.Y_avg = np.empty(avg_shape)
        self.sim.P = np.empty(shape)
        self.sim.S = np.empty(shape)
        self.sim.age = np.empty(shape)

        self.sim.init_P = self.par.init_P*np.ones((self.sim.simN,2)) 

        # d. call
        for t in range(self.par.Tr_N):
            self.simulate_name_change(t,self.sim.trans[t,:],self.sim.perm[t,:],self.sim.uni[t,:])
            
            # avg consumption without zero-shocks
            I = self.sim.Y[t,:,0]>-100 # both consumers have same income proces so we only need to check one of them
            
            avg_C_both = np.sum(self.sim.C[t,I] * np.array([self.par.share,1-self.par.share]),axis=1)
            
            self.sim.C_avg_type1[t] = np.mean(self.sim.C[t,I,0])
            self.sim.C_avg_type2[t] = np.mean(self.sim.C[t,I,1])
            self.sim.C_avg[t] = np.mean(avg_C_both)
            self.sim.C_var[t] = np.var(avg_C_both)
            self.sim.Y_avg[t] = (np.mean(self.sim.Y[t,I]))

            # # end-of-period wealth and saving
            # self.sim.a[t] = self.sim.m[t] - self.sim.C[t]
            # if t>0:
            #     self.sim.S[t] = (self.sim.a[t]*self.sim.P[t] - self.sim.a[t-1]*self.sim.P[t-1]) # do not divide with R because I use A and not W
        
        return self.sim

    def simulate_name_change(self,t,trans,perm,uni):
        """Change the name of the function later"""
        c_sol = np.zeros((self.par.num_xhat+1,2))
        m_sol = np.zeros((self.par.num_xhat+1,2)) + self.par.xmin
        
        c_sol[1:self.par.num_xhat+1] = self.sol.c[:,t]
        m_sol[1:self.par.num_xhat+1] = self.sol.m[:,t]

        c = self.sim.c[t,:]
        # a. shocks variables 
        sigma_perm = self.par.sigma_eta
        sigma_trans = self.par.sigma_mu

        perm_shock = np.exp(sigma_perm*perm) # log-normal distribution with variance sigma_perm 
        perm_shock = np.repeat(perm_shock.reshape(len(perm_shock),1),2,axis=1)
        trans_shock = np.exp(sigma_trans*trans)*(uni>self.par.pi) + 0*(uni<=self.par.pi) # log-normal distribution with variance -- notice that the very last 
        trans_shock = np.repeat(trans_shock.reshape(len(trans_shock),1),2,axis=1)
        if t==0:
            self.sim.P[t,:] = self.sim.init_P*perm_shock
            initW = 0.061 #par.mu_a_init*np.exp(par.sigma_a_init*sim.init_a) 
            self.sim.m[t,:] = initW + trans_shock
        else:
            self.sim.P[t] = self.sim.P[t-1]*self.par.G[t-1]*perm_shock
            fac = self.par.G[t-1]*perm_shock
            self.sim.m[t] = (1+self.par.r)*self.sim.a[t-1]/fac + trans_shock 

        # Income 
        self.sim.Y[t] = self.sim.P[t]*trans_shock

        # interpolate optimal consumption
        for consumer in range(m_sol.shape[1]):
            interp = interpolate.interp1d(m_sol[:,consumer],c_sol[:,consumer], bounds_error=False, fill_value = "extrapolate")
            c[:,consumer] = interp(self.sim.m[t,:,consumer])
            self.sim.C[t,:,consumer] = c[:,consumer] * self.sim.P[t,:,consumer]
            self.sim.M[t,:,consumer] = self.sim.m[t,:,consumer]*self.sim.P[t,:,consumer]
        
        # used for checking that the code works as intended
        if self.par.betas[0] == self.par.betas[1]:
            if not np.allclose(self.sim.C[t,:,0], self.sim.C[t,:,1]):
                # check if they are all nan 
                if np.all(np.isnan(self.sim.C[t,:,0])) and np.all(np.isnan(self.sim.C[t,:,1])):
                    pass
                # find where they are not equal
                else:
                    print(self.sim.C[t,:,0], self.sim.C[t,:,1])
                    raise ValueError('C0 and C1 are not equal')

        # end-of-period wealth and saving
        self.sim.a[t,:,:] = self.sim.m[t,:,:] - c[:,:]

        if t>0:
            self.sim.S[t] = (self.sim.a[t]*self.sim.P[t] - self.sim.a[t-1]*self.sim.P[t-1]) # do not divide with R because I use A and not W
