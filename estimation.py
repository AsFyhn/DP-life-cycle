from scipy.optimize import minimize
import numpy as np

class SMD:
    def __init__(self, model, solver, simulator,mom_data):
        self.model = model
        self.mom_data = mom_data
        self.solver = solver
        self.simulator = simulator
    def mom_fun(self,data):
        return (data.C_avg)
    
    def obj_function(self,theta,est_par,W):
        # 1. update parameters 
        for i in range(len(est_par)):
            setattr(self.model,est_par[i],theta[i]) # like par.key = val

        # 2. solve model with current parameters
        sol = self.solver(self.model) # consider adding the solve method to the class beforehand

        # 3. simulate data from the model and calculate moments [have this as a complete function, used for standard errors]
        sim = self.simulator.sim_setup(self.model)
        self.simulator.draw_random(self.model, sim)
        self.sim = self.simulator.simulate(sim, self.model, sol)
        self.mom_sim = np.empty(4) + np.nan
        for j, i in enumerate(range(0,len(self.sim.C_avg),10)):
            self.mom_sim[j] = self.sim.C_avg[i:i+10].mean()
        # self.mom_sim = self.mom_fun(self.sim)

        # self.model.simulate() # consider adding the solve method to the class beforehand
        # self.mom_sim = self.mom_fun(self.model.sim)

        # 4. calculate objective function and return it
        self.diff = self.mom_data - self.mom_sim
        self.obj  = (np.transpose(self.diff) @ W) @ self.diff

        return self.obj 

    def estimate(self,theta0,est_par,W):
        assert(len(W[0])==len(self.mom_data)) # check dimensions of W and mom_data

        self.est_out = minimize(self.obj_function, theta0, (est_par,W,), method='nelder-mead',options={'disp': False})        
        # return output
        self.est = self.est_out.x
        self.W = W

    def estimate_using_grid(self,beta_grid,rho_grid,est_par,W,):
        assert(len(W[0])==len(self.mom_data)) # check dimensions of W and mom_data

        self.beta_grid = beta_grid
        self.rho_grid = rho_grid
        self.grid = np.empty((len(self.beta_grid),len(self.rho_grid))) + np.nan
        for i, beta in enumerate(self.beta_grid):
            for j, rho in enumerate(self.rho_grid):
                self.grid[i,j] = self.obj_function([beta,rho],est_par,W)
        


