import numpy as np
from scipy import interpolate

class Simulator:
    def __init__(self,par,sol,simN=10_000):
        self.par = par
        self.simN = simN
        self.sol = sol
        
    def main(self):
        """
        Executes the main simulation process.

        This method sets up the simulation, draws random values, and performs the simulation.
        It returns the simulation results.

        Returns:
            sim_res (any): The simulation results.
        """
        # a) setup
        self.sim_setup()
        # b) draw random values
        self.draw_random()
        # c) simulate
        sim_res = self.simulate()

        return sim_res

    def sim_setup(self):
        """
        Setup the simulation.

        This method sets up the simulation by performing the following steps:
        a) Set the seed for random number generation.
        b) Initialize the simulation object.
        c) Set simulation parameters.
        d) Allocate memory for simulation arrays.
        """
        # a) set seed
        np.random.seed(2023) 

        # b) initialize simulation object
        class sim: pass
        self.sim = sim

        # c) set simulation parameters
        self.sim.simN = self.simN
        self.shape = (self.par.Tr_N,self.sim.simN)

        # d) allocate memory
        self.sim.trans = np.empty(self.shape)
        self.sim.perm = np.empty(self.shape)
        self.sim.uni = np.empty(self.shape)


    def draw_random(self):
        """
        Draw random values for the simulation.

        This method generates random values for the simulation by drawing from
        normal and uniform distributions. The generated values are stored in the
        `trans`, `perm` and `uni` attributes of the `sim` object.
        """
        # a) draw random values of transitory and permanent shocks        
        self.sim.trans = np.random.normal(size=self.shape)
        self.sim.perm  = np.random.normal(size=self.shape)
        # b) draw random values of the uniform distribution -- used for the zero income probability
        self.sim.uni = np.random.uniform(0,1,size=self.shape)

    def simulate(self,): # simulate the model
        
        # a. allocate
        shape = (self.shape[0],self.shape[1],2)
        avg_shape = (self.shape[0]) # across consumers thus only one dimension
        # add extra dimension to the shape
        self.sim.c = np.empty(shape)
        self.sim.C = np.empty(shape)
        self.sim.C_var = np.empty(avg_shape)
        self.sim.C_avg_type1 = np.empty(avg_shape)
        self.sim.C_avg_type2 = np.empty(avg_shape)
        self.sim.C_avg = np.empty(avg_shape)
        self.sim.a = np.empty(shape)
        self.sim.A = np.empty(shape)
        self.sim.x = np.empty(shape)
        self.sim.X = np.empty(shape)
        self.sim.Y = np.empty(shape)
        self.sim.Y_avg = np.empty(avg_shape)
        self.sim.P = np.empty(shape)
        self.sim.S = np.empty(shape)
        self.sim.age = np.empty(shape)
        self.sim.init_P = self.par.init_P*np.ones((self.sim.simN,2)) 

        # d. call
        for t in range(self.par.Tr_N):
            self.do_simulation(t,self.sim.trans[t,:],self.sim.perm[t,:],self.sim.uni[t,:])
            
            # avg consumption without zero-shocks
            I = self.sim.Y[t,:,0]>0 # both consumers have same income proces so we only need to check one of them
            
            avg_C_both = np.sum(self.sim.C[t,I] * np.array([self.par.share,1-self.par.share]),axis=1)
            
            self.sim.C_avg_type1[t] = np.mean(self.sim.C[t,I,0])
            self.sim.C_avg_type2[t] = np.mean(self.sim.C[t,I,1])
            self.sim.C_avg[t] = np.mean(avg_C_both)
            self.sim.C_var[t] = np.var(avg_C_both)
            self.sim.Y_avg[t] = (np.mean(self.sim.Y[t,I]))
        
        return self.sim

    def do_simulation(self, t, trans, perm, uni):
        """
        Perform a simulation for a given time period and store the results in the simulation object.

        Parameters:
        - t (int): The time period.
        - trans (numpy.ndarray): Array of shock variables for transitory income.
        - perm (numpy.ndarray): Array of shock variables for permanent income.
        - uni (numpy.ndarray): Array of uniform random variables.
        """

        # a. allocate
        c_sol = np.zeros((self.par.num_xhat+1,2))
        x_sol = np.zeros((self.par.num_xhat+1,2)) + self.par.xmin
        
        # next period consumption and cash-on-hand
        c_sol[1:self.par.num_xhat+1] = self.sol.c[:,t]
        x_sol[1:self.par.num_xhat+1] = self.sol.x[:,t]

        c = self.sim.c[t,:]
        
        # b. shocks variables 
        sigma_perm = self.par.sigma_eta
        sigma_trans = self.par.sigma_mu

        perm_shock = np.exp(sigma_perm*perm) # log-normal distribution with variance sigma_perm 
        perm_shock = np.repeat(perm_shock.reshape(len(perm_shock),1),2,axis=1)
        trans_shock = np.exp(sigma_trans*trans)*(uni>self.par.pi) + 0*(uni<=self.par.pi) # log-normal distribution with variance -- notice that the very last 
        trans_shock = np.repeat(trans_shock.reshape(len(trans_shock),1),2,axis=1)

        # c. calculate permanent income and cash-on-hand
        if t==0:
            self.sim.P[t,:] = self.sim.init_P*perm_shock
            inita = 0.061 # initial wealth
            self.sim.x[t,:] = inita + trans_shock 
        else:
            self.sim.P[t] = self.sim.P[t-1]*self.par.G[t-1]*perm_shock
            fac = self.par.G[t-1]*perm_shock
            self.sim.x[t] = (1+self.par.r)*self.sim.a[t-1]/fac + trans_shock 

        # Income 
        self.sim.Y[t] = self.sim.P[t]*trans_shock

        # interpolate optimal consumption for each consumer
        for consumer in range(x_sol.shape[1]):
            interp = interpolate.interp1d(x_sol[:,consumer],c_sol[:,consumer], bounds_error=False, fill_value = "extrapolate")
            c[:,consumer] = interp(self.sim.x[t,:,consumer])
            self.sim.C[t,:,consumer] = c[:,consumer] * self.sim.P[t,:,consumer]
            self.sim.X[t,:,consumer] = self.sim.x[t,:,consumer]*self.sim.P[t,:,consumer]
        
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
        self.sim.a[t,:,:] = self.sim.x[t,:,:] - c[:,:]

        if t>0:
            self.sim.S[t] = (self.sim.a[t]*self.sim.P[t] - self.sim.a[t-1]*self.sim.P[t-1]) # do not divide with R because we use a and not x
