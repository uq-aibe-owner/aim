#======================================================================
#
#     This routine interfaces with IPOPT
#     It sets the optimization problem for every training point
#     during the VFI.
#
#     Simon Scheidegger, 11/16 ; 07/17; 01/19
#======================================================================

from parameters import *
from ipopt_wrapperA import EV_F_ITER, EV_GRAD_F_ITER, EV_G_ITER, EV_JAC_G_ITER
import numpy as np
#import pyipopt
import cyipopt

class HS071():

    def __init__(self, X, n_agents, k_init, NELE_JAC, NELE_HESS, gp_old):
        self.x = X
        self.n_agents = n_agents
        self.k_init = k_init
        self.NELE_JAC = NELE_JAC
        self.NELE_HESS = NELE_HESS
        #self.flag
        self.gp_old = gp_old


    #Create ev_f, eval_f, eval_grad_f, eval_g, eval_jac_g for given k_init and n_agent
    def eval_f(self,x):
        return EV_F_ITER(x, self.k_init, self.n_agents, self.gp_old)

    def eval_grad_f(self,x):
        return EV_GRAD_F_ITER(x, self.k_init, self.n_agents, self.gp_old)

    def eval_g(self,x):
        return EV_G_ITER(x, self.k_init, self.n_agents)
        
    def eval_jac_g(self,x, flag):
        return EV_JAC_G_ITER(x, flag, self.k_init, self.n_agents)
    
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
#       return x[0] * x[3] * np.sum(x[0:3]) + x[2]"""
        #print("afdsadas",x)
        return self.eval_f(x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        """return np.array([
            x[0]*x[3] + x[3]*np.sum(x[0:3]),
            x[0]*x[3],
            x[0]*x[3] + 1.0,
            x[0]*np.sum(x[0:3])
        ])"""
        return self.eval_grad_f(x)
    def constraints(self, x):
        """Returns the constraints."""
#       return np.array((np.prod(x), np.dot(x, x)))
        return self.eval_g(x)

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
#       return np.concatenate((np.prod(x)/x, 2*x))
        return self.eval_jac_g(x, False)

        
    #def hessianstructure(self):
        """Returns the row and column indices for non-zero vales of the
        Hessian."""
        
        # NOTE: The default hessian structure is of a lower triangular matrix,
        # therefore this function is redundant. It is included as an example
        # for structure callback.

    #    return np.nonzero(np.tril(np.ones((4, 4)))) # unsure

    #def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""
        
        """

        H = obj_factor*np.array((
            (2*x[3], 0, 0, 0),
            (x[3],   0, 0, 0),
            (x[3],   0, 0, 0),
            (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
            (0, 0, 0, 0),
            (x[2]*x[3], 0, 0, 0),
            (x[1]*x[3], x[0]*x[3], 0, 0),
            (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        row, col = self.hessianstructure()

        
        """
        

        #return H[row, col]
    #    print("tried to call H")
    #    return None
        #return self.NELE_HESS # unsure, very unsure

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        msg = "Objective value at iteration #{:d} is - {:g}"

        print(msg.format(iter_count, obj_value))


def iterate(k_init, n_agents, gp_old):
    #print("Calling iterate", gp_old)
    
    # IPOPT PARAMETERS below "
    N=3*n_agents    # number of vars
    M=3*n_agents+1  # number of constraints
    NELE_JAC=N*M
    NELE_HESS=(N**2-N)/2 + N    # number of non-zero entries of Hess matrix
    
    # Vector of variables -> solution of non-linear equation system 
    X=np.empty(N)

    LAM=np.empty(M) # multipliers
    G=np.empty(M)   # (in-)equality constraints

    # Vector of lower and upper bounds
    G_L=np.empty(M)
    G_U=np.empty(M)

    X_L=np.empty(N)
    X_U=np.empty(N)

    Z_L=np.empty(N)
    Z_U=np.empty(N)

    # get coords of an individual points 
    grid_pt_box=k_init
    
    X_L[:n_agents]=c_bar
    X_U[:n_agents]=c_up

    X_L[n_agents:2*n_agents]=l_bar
    X_U[n_agents:2*n_agents]=l_up

    X_L[2*n_agents:3*n_agents]=inv_bar
    X_U[2*n_agents:3*n_agents]=inv_up

    # Set bounds for the constraints 
    G_L[:n_agents]=c_bar
    G_U[:n_agents]=c_up

    G_L[n_agents:2*n_agents]=l_bar
    G_U[n_agents:2*n_agents]=l_up

    G_L[2*n_agents:3*n_agents]=inv_bar
    G_U[2*n_agents:3*n_agents]=inv_up

    G_L[3*n_agents]=0.0 # both values set to 0 for equality contraints
    G_U[3*n_agents]=0.0

    # initial guesses for first iteration
    cons_init=0.5*(X_U[:n_agents] - X_L[:n_agents]) + X_L[:n_agents]
    lab_init=0.5*(X_U[n_agents:2*n_agents] - X_L[n_agents:2*n_agents]) + X_L[n_agents:2*n_agents]
    inv_init=0.5*(X_U[2*n_agents:3*n_agents] - X_L[2*n_agents:3*n_agents]) + X_L[2*n_agents:3*n_agents]

    X[:n_agents]=cons_init
    X[n_agents:2*n_agents]=lab_init
    X[2*n_agents:3*n_agents]=inv_init
    
    # Create ev_f, eval_f, eval_grad_f, eval_g, eval_jac_g for given k_init and n_agent 
        
    #def eval_f(X):
    #    return EV_F_ITER(X, k_init, n_agents, gp_old)
        
    #def eval_grad_f(X):
    #    return EV_GRAD_F_ITER(X, k_init, n_agents, gp_old)
        
    #def eval_g(X):
    #    return EV_G_ITER(X, k_init, n_agents)
        
    #def eval_jac_g(X, flag):
    #    return EV_JAC_G_ITER(X, flag, k_init, n_agents)
       
    # First create a handle for the Ipopt problem 
    """
    nlp=ipyopt.create(N, X_L, X_U, M, G_L, G_U, NELE_JAC, NELE_HESS, eval_f, eval_grad_f, eval_g, eval_jac_g)
    nlp.num_option("obj_scaling_factor", -1.0)
    nlp.num_option("tol", 1e-6)
    nlp.num_option("acceptable_tol", 1e-5)
    nlp.str_option("derivative_test", "first-order")
    nlp.str_option("hessian_approximation", "limited-memory")
    nlp.int_option("print_level", 1)
    
    x, z_l, z_u, constraint_multipliers, obj, status=nlp.solve(X)
    """

    HS07 = HS071(X, n_agents, k_init, NELE_JAC, NELE_HESS, gp_old) # creates an instance of the class
        
    # First create a handle for the Ipopt problem 
    nlp=cyipopt.Problem(n=N, m = M, problem_obj=HS07, lb=X_L, ub=X_U, cl=G_L, cu=G_U,) 

    nlp.addOption("obj_scaling_factor", -1.00) # max function 
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-6)
    nlp.addOption("print_level", 0)
    nlp.addOption("hessian_approximation", "limited-memory")

    A, B = nlp.solve(X) 
    #print("A", A, "B", B) 
    x = B['x'] 
    print(x)

    g = B['g']
    obj = B['obj_val']
    nlp.close()
    # x: Solution of the primal variables
    # z_l, z_u: Solution of the bound multipliers
    # constraint_multipliers: Solution of the constraint multipliers
    # obj: Objective value
    # status: Exit Status

    # Unpack Consumption, Labor, and Investment
    c=x[:n_agents]
    l=x[n_agents:2*n_agents]
    inv=x[2*n_agents:3*n_agents]
    to_print=np.hstack((obj,x)) 

    print("iteration output")

    print("Consumption", c)
    print("Labour",l)
    print("Investment",inv)
    
    # === debug
    #f=open("results.txt", 'a')
    #np.savetxt(f, np.transpose(to_print) #, fmt=len(x)*'%10.10f ')
    #for num in to_print:
    #    f.write(str(num)+"\t")
    #f.write("\n")
    #f.close()
    
    return obj, c, l, inv
