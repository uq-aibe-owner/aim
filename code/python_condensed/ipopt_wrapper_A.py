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
    
    """ # Extract Variables
    # this loop extracts the variables more expandably than doing them individualy as before
    for iter in i_pol_key:
        # forms the  2d intermediate variables into globals of the same name but in matrix form
        if d_pol[iter] == 2:
            globals()[iter] = np.zeros((n_agt,n_agt))
            for row in range(n_agt):
                for col in range(n_agt):
                    globals()[iter][row,col] = X[I[iter][0]+col+row*n_agt]
        else:
            # forms the 1d intermediate variables into globals of the same name in vector(list) form
            globals()[iter] = [X[ring] for ring in I[iter]] """
    """ val = X[I["val"]]
    V_old1 = V_INFINITY(X[I["knx"]])
    # Compute Value Function
    VT_sum=utility(X[I["con"]], X[I["lab"]]) + beta*V_old1 """
       
    return X[I["utl"]] + beta*X[I["val"]]

#=======================================================================
#   Objective Function during VFI (note - we need to interpolate on an "old" GPR)
    
def EV_F_ITER(X, kap, n_agt, gp_old):
    
    """ # initialize correct data format for training point
    s = (1,n_agt)
    kap2 = np.zeros(s)
    kap2[0,:] = X[I["knx"]]
    
    # interpolate the function, and get the point-wise std.
    val = X[I["val"]]
    
    VT_sum = X[I["utl"]] + beta*val """
    
    return X[I["utl"]] + beta*X[I["val"]]
    
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
      
def EV_G(X, kap, n_agt, gp_old):
    M=n_ctt
    G=np.empty(M, float)

    s = (1,n_agt)
    kap2 = np.zeros(s)
    kap2[0,:] = X[I["knx"]]

    """ print("should be the same")
    #print(type(X[I["knx"]]))
    print(np.shape(X[I["knx"]]))
    #print(type(kap2))
    print(np.shape(kap2)) """

    # pull in constraints
    e_ctt = f_ctt(X, gp_old, kap2, 1, kap)
    # apply all constraints with this one loop
    for iter in ctt_key:
        G[I_ctt[iter]] = e_ctt[iter]

    return G

#======================================================================
#   Equality constraints during the VFI of the model
def EV_G_ITER(X, kap, n_agt, gp_old):
    
    M=n_ctt
    G=np.empty(M, float)

    s = (1,n_agt)
    kap2 = np.zeros(s)
    kap2[0,:] = X[I["knx"]]

    """ print("should be the same")
    print(type(X[I["knx"]]))
    print(np.shape(X[I["knx"]]))
    print(type(kap2))
    print(np.shape(kap2)) """

    # pull in constraints
    e_ctt = f_ctt(X, gp_old, kap2, 0, kap)
    # apply all constraints with this one loop
    for iter in ctt_key:
        G[I_ctt[iter]] = e_ctt[iter]

    return G

#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints 
#   for first time step
    
def EV_JAC_G(X, flag, kap, n_agt, gp_old):
    N=n_pol
    M=n_ctt
    #print(N, "  ",M) #testing testing
    NZ=n_pol*n_ctt # J - could it be this?
    A=np.empty(NZ, float)
    ACON=np.empty(NZ, int) # its cause int is already a global variable cause i made it
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
        gx1=EV_G(X, kap, n_agt, gp_old)
        
        for ixM in range(M):
            for ixN in range(N):
                xAdj=np.copy(X)
                xAdj[ixN]=xAdj[ixN]+h
                gx2=EV_G(xAdj, kap, n_agt, gp_old)
                A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
        return A
  
#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints 
#   during iteration  
  
def EV_JAC_G_ITER(X, flag, kap, n_agt, gp_old):
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
        gx1=EV_G_ITER(X, kap, n_agt, gp_old)

        for ixM in range(M):
            for ixN in range(N):
                xAdj=np.copy(X)
                xAdj[ixN]=xAdj[ixN]+h
                gx2=EV_G_ITER(xAdj, kap, n_agt, gp_old)
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
            return EV_G(x, self.k_init, self.n_agents, self.gp_old)  
        else: 
            return EV_G_ITER(x, self.k_init, self.n_agents, self.gp_old) 

    def eval_jac_g(self, x, flag):
        if self.initial: 
            return EV_JAC_G(x, flag, self.k_init, self.n_agents, self.gp_old) 

        else: 
            return EV_JAC_G_ITER(x, flag, self.k_init, self.n_agents, self.gp_old) 

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
