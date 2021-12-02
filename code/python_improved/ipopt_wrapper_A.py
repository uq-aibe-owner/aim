#=======================================================================
#
#     ipopt_wrapper.py : an interface to IPOPT and PYIPOPT
#
#     Simon Scheidegger, 06/17
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
#
#=======================================================================

from parameters import *
import numpy as np

#=======================================================================
#   Objective Function to start VFI (in our case, the value function)
        
def EV_F(X, kap, n_agt):
    
    # Extract Variables
    con=X[(i_con-1)*n_agt:i_con*n_agt]
    lab=X[(i_lab-1)*n_agt:i_lab*n_agt]
    inv=X[(i_inv-1)*n_agt:i_inv*n_agt]

    kap_nxt= (1-delta)*kap + inv
    
    # Compute Value Function
    VT_sum=utility(con, lab) + beta*V_INFINITY(kap_nxt)
       
    return VT_sum

# initial guess of the value function
def V_INFINITY(kap=[]):
    e=np.ones(len(kap))
    c=output_f(kap,e)
    v_infinity=utility(c,e)/(1-beta)
    return v_infinity

#=======================================================================
#   Objective Function during VFI (note - we need to interpolate on an "old" GPR)
    
def EV_F_ITER(X, kap, n_agt, gp_old):
    
    # Extract Variables
    con=X[(i_con-1)*n_agt:i_con*n_agt]
    lab=X[(i_lab-1)*n_agt:i_lab*n_agt]
    inv=X[(i_inv-1)*n_agt:i_inv*n_agt]
    kap_nxt=X[(i_inv-1)*n_agt:i_inv*n_agt]


    
    # initialize correct data format for training point
    s = (1,n_agt)
    Xtest = np.zeros(s)
    Xtest[0,:] = kap_nxt
    
    # interpolate the function, and get the point-wise std.
    V_old, sigma_test = gp_old.predict(Xtest, return_std=True)
    
    VT_sum = utility(con, lab) + beta*V_old
    
    return VT_sum
    
    
#=======================================================================
#   Computation of gradient (first order finite difference) of initial objective function 

def EV_GRAD_F(X, kap, n_agt):
    
    N=len(X)
    GRAD=np.zeros(N, float) # Initial Gradient of Objective Function
    h=1e-4
    
    
    for ixN in range(N):
        xAdj=np.copy(X)
        
        if (xAdj[ixN] - h >= 0):
            xAdj[ixN]=X[ixN] + h            
            fx2=EV_F(xAdj, kap, n_agt)
            
            xAdj[ixN]=X[ixN] - h
            fx1=EV_F(xAdj, kap, n_agt)
            
            GRAD[ixN]=(fx2-fx1)/(2.0*h)
            
        else:
            xAdj[ixN]=X[ixN] + h
            fx2=EV_F(xAdj, kap, n_agt)
            
            xAdj[ixN]=X[ixN]
            fx1=EV_F(xAdj, kap, n_agt)
            GRAD[ixN]=(fx2-fx1)/h
            
    return GRAD
    
#=======================================================================
#   Computation of gradient (first order finite difference) of the objective function 
    
def EV_GRAD_F_ITER(X, kap, n_agt, gp_old):
    
    N=len(X)
    GRAD=np.zeros(N, float) # Initial Gradient of Objective Function
    h=1e-4
    
    
    for ixN in range(N):
        xAdj=np.copy(X)
        
        if (xAdj[ixN] - h >= 0):
            xAdj[ixN]=X[ixN] + h            
            fx2=EV_F_ITER(xAdj, kap, n_agt, gp_old)
            
            xAdj[ixN]=X[ixN] - h
            fx1=EV_F_ITER(xAdj, kap, n_agt, gp_old)
            
            GRAD[ixN]=(fx2-fx1)/(2.0*h)
            
        else:
            xAdj[ixN]=X[ixN] + h
            fx2=EV_F_ITER(xAdj, kap, n_agt, gp_old)
            
            xAdj[ixN]=X[ixN]
            fx1=EV_F_ITER(xAdj, kap, n_agt, gp_old)
            GRAD[ixN]=(fx2-fx1)/h
            
    return GRAD
       
#======================================================================
#   Equality constraints for the first time step of the model
      
def EV_G(X, kap, n_agt):
    M=n_ctt*n_agt
    G=np.empty(M, float)
    
    # Extract Variables
    con=X[(i_con-1)*n_agt:i_con*n_agt]
    lab=X[(i_lab-1)*n_agt:i_lab*n_agt]
    inv=X[(i_inv-1)*n_agt:i_inv*n_agt]
    kap_nxt=X[(i_kap_nxt-1)*n_agt:i_kap_nxt*n_agt]

    # pull in constraints
    dct_ctt = fcn_ctt(con, inv, lab, kap, kap_nxt)
    # apply constraints
    for iter in dct_ctt_ind_key:
        print(len(dct_ctt[iter]))
        G[(dct_ctt_ind[iter]-1)*n_agt:dct_ctt_ind[iter]*n_agt] = dct_ctt[iter]


    return G

#======================================================================
#   Equality constraints during the VFI of the model
def EV_G_ITER(X, kap, n_agt):
    
    return EV_G(X, kap, n_agt)

#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints 
#   for first time step
    
def EV_JAC_G(X, flag, kap, n_agt):
    N=len(X)
    M=n_ctt
    NZ=M*N
    A=np.empty(NZ, float)
    ACON=np.empty(NZ, int)
    AVAR=np.empty(NZ, int)    
    
    # Jacobian matrix structure
    
    if (flag):
        for ixM in range(M):
            for ixN in range(N):
                ACON[ixN + (ixM)*N]=ixM
                AVAR[ixN + (ixM)*N]=ixN
                
        return (ACON, AVAR)
        
    else:
        # Finite Differences
        h=1e-4
        gx1=EV_G(X, kap, n_agt)
        
        for ixM in range(M):
            for ixN in range(N):
                xAdj=np.copy(X)
                xAdj[ixN]=xAdj[ixN]+h
                gx2=EV_G(xAdj, kap, n_agt)
                A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
        return A
  
#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints 
#   during iteration  
  
def EV_JAC_G_ITER(X, flag, kap, n_agt):
    N=len(X)
    M=n_ctt
    NZ=M*N
    A=np.empty(NZ, float)
    ACON=np.empty(NZ, int)
    AVAR=np.empty(NZ, int)

    # Jacobian matrix structure

    if (flag):
        for ixM in range(M):
            for ixN in range(N):
                ACON[ixN + (ixM)*N]=ixM
                AVAR[ixN + (ixM)*N]=ixN

        return (ACON, AVAR)

    else:
        # Finite Differences
        h=1e-4
        gx1=EV_G_ITER(X, kap, n_agt)

        for ixM in range(M):
            for ixN in range(N):
                xAdj=np.copy(X)
                xAdj[ixN]=xAdj[ixN]+h
                gx2=EV_G_ITER(xAdj, kap, n_agt)
                A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
        return A
    
#======================================================================

class ipopt_obj(): 
    """
    Class for the optimization problem to be passed to cyipopt 
    Further optimisations may be possible here by including a hessian (optional param) 
    Uses the existing instance of the Gaussian Process (GP OLD) 
    """

    def __init__(self, X, n_agents, k_init, NELE_JAC, NELE_HESS, gp_old=None,initial=False, verbose=False): 
        self.x = X 
        self.n_agents = n_agents 
        self.k_init = k_init 
        self.NELE_JAC = NELE_JAC 
        self.NELE_HESS = NELE_HESS 
        self.gp_old = gp_old 
        self.initial = initial 
        self.verbose = verbose 

    # Create ev_f, eval_f, eval_grad_f, eval_g, eval_jac_g for given k_init and n_agent 
    def eval_f(self, x): 
        if self.initial: 
            return EV_F(x, self.k_init, self.n_agents) 
        else: 
            return EV_F_ITER(x, self.k_init, self.n_agents, self.gp_old) 

    def eval_grad_f(self, x): 
        if self.initial: 
            return EV_GRAD_F(x, self.k_init, self.n_agents)  
        else: 
            return EV_GRAD_F_ITER(x, self.k_init, self.n_agents, self.gp_old) 

    def eval_g(self, x): 
        if self.initial: 
            return EV_G(x, self.k_init, self.n_agents)  
        else: 
            return EV_G_ITER(x, self.k_init, self.n_agents) 

    def eval_jac_g(self, x, flag): 
        if self.initial: 
            return EV_JAC_G(x, flag, self.k_init, self.n_agents) 

        else: 
            return EV_JAC_G_ITER(x, flag, self.k_init, self.n_agents) 

    def objective(self, x): 
        # Returns the scalar value of the objective given x. 
        return self.eval_f(x) 

    def gradient(self, x): 
        # Returns the gradient fo the objective with respect to x.""" 
        return self.eval_grad_f(x) 

    def constraints(self, x): 
        # Returns the constraints 
        return self.eval_g(x) 

    def jacobian(self, x): 
        # Returns the Jacobian of the constraints with respect to x. 
        return self.eval_jac_g(x, False) 

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        if self.verbose: 
            msg = "Objective value at iteration #{:d} is - {:g}"
            print(msg.format(iter_count, obj_value))
