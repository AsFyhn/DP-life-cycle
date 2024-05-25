from scipy.optimize import minimize
import numpy as np

class SMD:
    def __init__(self, model, solver, simulator,mom_data):
        self.model = model
        self.mom_data = mom_data
        self.solver = solver(model)
        self.simulator = simulator
        self.age_groups = False # default is False

    def mom_fun(self,data):
        return np.log(data.C_avg)
    
    def obj_function(self,theta,est_par,W):
        # a. update parameters 
        for i in range(len(est_par)):
            setattr(self.solver.par,est_par[i],theta[i]) # like par.key = val

        # b. solve model with current parameters
        self.solver.solve() # consider adding the solve method to the class beforehand
        sol = self.solver.sol

        # c. simulate data from the model and calculate moments [have this as a complete function, used for standard errors]
        # sim = self.simulator.sim_setup(self.solver.par)
        # self.simulator.draw_random(self.solver.par, sim)
        # self.sim = self.simulator.simulate(sim, self.solver.par, sol)
        
        sim_Object = self.simulator(par=self.solver.par,sol=sol)
        sim = sim_Object.main() 

        if self.age_groups:
            self.mom_sim = np.empty(4) + np.nan
            for j, i in enumerate(range(0,len(self.sim.C_avg),10)):
                self.mom_sim[j] = self.sim.C_avg[i:i+10].mean()
        else:
            self.mom_sim = self.mom_fun(sim)

        # d. calculate objective function and return it
        self.diff = self.mom_data - self.mom_sim
        self.obj  = (np.transpose(self.diff) @ W) @ self.diff

        return self.obj 

    def estimate(self,theta0,est_par,bounds=None,W=None, grid=False):
        """
        Estimate the model parameters using the simulated method of moments
            Args:
                theta0: initial guess for the parameters
                est_par: list of the parameters to be estimated
                bounds: bounds for the parameters
                W: weight matrix
                grid: boolean indicating whether to estimate the parameters on a grid
            Returns:
                None
            
            Note: 
                - If W is not provided, the function uses the identity matrix
                - The function stores the estimated parameters in self.est
                - The function stores the weight matrix in self.W
            """
        if W is None: 
            W = np.eye(len(self.mom_data))

        # a. store initial parameters
        thetainit = [getattr(self.model,est_par[i]) for i in range(len(est_par))]

        # b. Check dimensions of weight matrix and data
        assert(len(W[0])==len(self.mom_data)) 

        # c. estimate parameters
        if not grid:
            # i. estimate parameters by minimizing the objective function
            self.est_out = minimize(self.obj_function, theta0, (est_par,W,), bounds=bounds,
                                    method='nelder-mead',options={'disp': False}, )
            x_opt = self.est_out.x
        else:
            # ii. estimate parameters by minimizing the objective function
            if not hasattr(self,'beta_grid') or not hasattr(self,'rho_grid'):
                raise ValueError('Please provide beta_grid and rho_grid')
            self.grid = np.empty((len(self.beta_grid),len(self.rho_grid))) + np.nan
            for i, beta in enumerate(self.beta_grid):
                for j, rho in enumerate(self.rho_grid):
                    self.grid[i,j] = self.obj_function([beta,rho],est_par,W)
            # Find the indices of the minimum value
            min_index = np.unravel_index(np.argmin(self.grid), self.grid.shape)
            x_opt = [self.beta_grid[min_index[0]],self.rho_grid[min_index[1]]]

        # -- store the estimated parameters
        self.est = x_opt
        self.W = W

        # d. reset the model parameters to the initial values
        for i in range(len(est_par)):
            setattr(self.model,est_par[i],thetainit[i])

    def estimate_variance_covariance_matrix(self, theta, est_par, W=None):
        """
        Estimate the variance-covariance matrix of the parameter estimates using the delta method
            Args:
                theta: estimated parameters
                est_par: list of the parameters to be estimated
                W: weight matrix
            Returns:
                var_cov_matrix: variance-covariance matrix of the parameter estimates
        """
        if W is None:
            W = np.eye(len(self.mom_data))

        # Compute the gradient of the moments function with respect to the parameters
        grad_moments = np.zeros((len(self.mom_data), len(est_par)))
        eps = 1e-6  # Small perturbation for numerical differentiation
        for i, par in enumerate(est_par):
            theta_plus = np.array(theta)
            theta_plus[i] += eps
            obj_plus = self.obj_function(theta_plus, est_par, W)

            theta_minus = np.array(theta)
            theta_minus[i] -= eps
            obj_minus = self.obj_function(theta_minus, est_par, W)

            grad_moments[:, i] = (obj_plus - obj_minus) / (2 * eps)

        # Compute the variance-covariance matrix using the delta method
        var_cov_matrix = np.linalg.inv(grad_moments.T @ W @ grad_moments)

        return var_cov_matrix
