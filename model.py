import numpy as np
import os 
import sys
sys.path.append(os.path.abspath(''))
from utils import create_shocks, linspace_kink

class gp_model:
    def __init__(self,**kwargs) -> None:
        """ 
        This class contains the parameters and other deterministic variables for the model.
        """
        class par: pass
        self.par = par
        # set parameters 
        
        self.set_parameters(**kwargs)
        # setup shocks, grid and income shifter
        self.main_setup()
    
    def main_setup(self,):
        self._setup_shocks()
        self._setup_grid()
        self._setup_income_shifter()

    def set_parameters(self,**kwargs):
        # a) utility and consumption parameters
        self.par.beta = 0.96 # discount factor
        self.par.beta2 = 0.96
        self.par.betas = np.array([self.par.beta,self.par.beta2]) # discount factor
        self.par.share = 1
        self.par.R = 1.034 # interest rate
        self.par.rho = 0.514 # risk aversion parameter in the utility function
        self.par.gamma1 = 0.071 # mpc for retiress. maybe 
        self.par.pi = 0.00302 # probability of income shock 
        
        # b) life cycle
        self.par.Tr = 66 # retirement age
        self.par.t0 = 26 # start working
        self.par.Tr_N = self.par.Tr - self.par.t0  # normalized
        self.par.t0_N = self.par.t0 - self.par.t0 # normalized

        # c) Gauss Hermite weights and points: # mu is transtoritory income shock and eta is permanent income shock
        self.par.sigma_mu = 0.0440
        self.par.sigma_eta = 0.0212
        
        self.par.order = 12

        #  d) grid parameters
        self.par.num_xhat = 300 # number of points in the grid
        self.par.xhat = 3 # End of period assets
        self.par.xmin = 0 # minimum value of the grid

        
        # f) kwargs and additional parameters
        for key, value in kwargs.items():
            setattr(self.par, key, value)
    
        self.par.dim = [self.par.num_xhat,self.par.Tr_N+1,2] # Dimension of value function space

    def _setup_shocks(self):
        self.par.eta, self.par.eta_w, self.par.mu, self.par.mu_w, self.par.Nshocks = create_shocks(
            sigma_eta=self.par.sigma_eta,   # standard deviation of the permanent income shock
            mean_eta=0,                     # mean of the permanent income shock
            N_eta=self.par.order,           # number of nodes in the permanent income shock
            sigma_mu=self.par.sigma_mu,     # standard deviation of the transitory income shock
            mean_mu=0,                      # mean of the transitory income shock
            N_mu=self.par.order,            # number of nodes in the transitory income shock
            pi=self.par.pi,                 # probability of the bad state
            )

        # Weights for each combination of shocks
        self.par.w = self.par.mu_w * self.par.eta_w
        # Check if the sum of weights is close to 1
        assert (1-sum(self.par.w) < 1e-8), f'{self.par.w}'    
    
    def _setup_grid(self):
        self.par.grid_xhat = linspace_kink(x_min=self.par.xmin+1e-6,x_max=self.par.xhat,n=self.par.num_xhat, x_int=1) # create a grid with more points below 1
        self.par.grid_xhat = self.par.grid_xhat.reshape((self.par.num_xhat,1))
    def _setup_income_shifter(self,):
        # a) define income data
        age_groups = [29.8, 39.5, 49.5, 59.6, 69.3]
        income = [80_911, 103_476, 110_254, 90_334, 63_187]# after tax income 
        #[89514, 118149, 128980, 105498, 68059] #income before tax
        consumption = [67883, 86049, 91074, 78079, 60844]
        # create dictionary        
        self.income_data = {age:income for age,income in zip(age_groups,income)}
        self.consumption_data = {age:consumption for age,consumption in zip(age_groups,consumption)}


        # b) permanent income growth
        polY = np.polyfit(age_groups, income, 4) # create a polynomial of 4th degree
        polC = np.polyfit(age_groups, consumption, 4) # create a polynomial of 4th degree

        self.grid_age = np.array([float(age) for age in range(self.par.t0,self.par.Tr+1)])  # create a grid of different ages

        Ybar=np.polyval(polY,self.grid_age) # interpolate
        self.Ybar = Ybar #np.log(Ybar)
        self.Cbar= np.log(np.polyval(polC,self.grid_age))

        # create growth rate in income
        self.par.G = Ybar[1:(self.par.Tr_N+1)]/Ybar[0:self.par.Tr_N] # growth rate is shiftet forward

        # c) set simulation parameters
        self.par.init_P = self.Ybar[0]