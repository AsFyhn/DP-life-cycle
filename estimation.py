from scipy.optimize import minimize
import numpy as np
class SMD:
    def __init__(self, model, simulator, mom_data):
        """
        Initialize the Estimation class.

        Args:
            model: The model object used for estimation.
            simulator: The simulator object used for estimation.
            mom_data: The moment data used for estimation.

        Attributes:
            model: The model object used for estimation.
            simulator: The simulator object used for estimation.
            mom_data: The moment data used for estimation.
            age_groups: A boolean indicating whether age groups are used. Default is False.
        """
        self.model = model
        self.simulator = simulator
        self.mom_data = mom_data
        self.age_groups = False  # default is False

    def mom_fun(self,data):
        return np.log(data.C_avg)
    
    def obj_function(self, theta, est_par, W):
        """
        Calculate the objective function value based on the given parameters.

        Parameters:
        - theta (list): List of parameter values.
        - est_par (list): List of parameter names to be updated.
        - W (numpy.ndarray): Weight matrix.

        Returns:
        - obj (float): Objective function value.
        """

        # a. update parameters 
        orig_params = [getattr(self.model.par, est_par[i]) for i in range(len(est_par))]

        for i in range(len(est_par)):
            setattr(self.model.par, est_par[i], theta[i]) # like par.key = val

        # b. solve model with current parameters
        self.model.solve_model() # consider adding the solve method to the class beforehand

        # c. simulate data from the model and calculate moments [have this as a complete function, used for standard errors]
        sim_Object = self.simulator(par=self.model.par, sol=self.model.sol)
        sim = sim_Object.main() 

        if self.age_groups:
            self.mom_sim = np.empty(4) + np.nan
            for j, i in enumerate(range(0, len(self.sim.C_avg), 10)):
                self.mom_sim[j] = self.sim.C_avg[i:i+10].mean()
        else:
            self.mom_sim = self.mom_fun(sim)

        # d. calculate objective function and return it
        self.diff = self.mom_data - self.mom_sim
        self.obj = (np.transpose(self.diff) @ W) @ self.diff

        # e. reset parameters
        for i in range(len(est_par)):
            setattr(self.model.par, est_par[i], orig_params[i])
        return self.obj

    def estimate(self,theta0,est_par,bounds=None,W=None,constraint=None, grid=False):
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
        thetainit = [getattr(self.model.par,est_par[i]) for i in range(len(est_par))]

        # b. Check dimensions of weight matrix and data
        assert(len(W[0])==len(self.mom_data)) 

        # c. estimate parameters
        if not grid:
            # i. estimate parameters by minimizing the objective function
            if constraint is None:
                constraints = None
                optimizer_method = 'nelder-mead'
            else:
                constraints = ({'type': 'ineq', 'fun': constraint})
                optimizer_method = 'SLSQP' # nelder-mead does not support constraints
            print('used optimizer: ', optimizer_method)
            self.est_out = minimize(self.obj_function, theta0, (est_par,W,), bounds=bounds,
                                    method=optimizer_method,options={'disp': False}, 
                                    constraints=constraints
                                    )
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
            setattr(self.model.par,est_par[i],thetainit[i])

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

    def sensitivity(self, theta, est_par, W, phi_st, step=1.0e-7):
        """
        Calculate sensitivity measures for the moment function.

        Parameters:
        - theta (array-like): The parameter values for which to calculate sensitivity measures.
        - est_par (array-like): The estimated parameter values.
        - W (array-like): The weight matrix.
        - phi_st (array-like): The names of the fixed parameters.
        - step (float, optional): The step size for numerical differentiation. Default is 1.0e-7.

        Returns:
        - S_elasticity (ndarray): The sensitivity measures.

        """
        # 1. numerical gradient of moment function wrt theta. 
        grad = self.num_grad_moms(est_par,theta, W, step=step)

        # 2. calculate key components
        GW = np.transpose(grad) @ W
        GWG = GW @ grad
        Psi = - np.linalg.solve(GWG, GW)

        # 3. calculate sensitivity measures
        # construct vector of fixed values
        phi = np.array([getattr(self.model.par, name) for name in phi_st])

        # calculate gradient of the moment function with respect to gamma
        Lambda = self.num_grad_moms(phi, phi_st, W, step=step)
        
        self.sens = Psi @ Lambda

        S_elasticity = np.empty((len(theta), len(phi)))
        for t in range(len(theta)):
            for g in range(len(phi)):
                # print(f'{theta[t]}, {phi_st[g]}',self.sens[t,g], phi[g]/est_par[t])
                S_elasticity[t, g] = self.sens[t, g] * phi[g] / est_par[t]    

        return S_elasticity

    def num_grad_moms(self,params,names,W,step=1.0e-4):
        """ 
        Returns the numerical gradient of the moment vector
        Inputs:
            params (1d array): K-element vector of parameters
            W (2d array): J x J weighting matrix
            step (float): step size in finite difference
            *args: additional objective function arguments
        
        Output:
            grad (2d array): J x K gradient of the moment function wrt to the elements in params
        """
        num_par = len(params)
        num_mom = len(W[0])

        # a. numerical gradient. The objective function is (data - sim)'*W*(data - sim) so take the negative of mom_sim
        grad = np.empty((num_mom,num_par))
        for p in range(num_par):
            params_now = params.copy()

            step_now  = np.zeros(num_par)
            step_now[p] = step #np.fmax(step,np.abs(step*params_now[p]))

            self.obj_function(params_now + step_now,names,W,)
            mom_forward = self.diff.copy()

            self.obj_function(params_now - step_now,names,W,)
            mom_backward = self.diff.copy()

            grad[:,p] = (mom_forward - mom_backward)/(2.0*step_now[p])

        # b. reset the parameters in the model to params
        for i in range(len(names)):
            setattr(self.model.par,names[i],params[i]) 
        
        # c. return gradient
        return grad

