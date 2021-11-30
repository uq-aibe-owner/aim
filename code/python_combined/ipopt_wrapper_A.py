#=======================================================================
#
#     ipopt_wrapper.py : an interface to IPOPT and PYIPOPT
#
#     Simon Scheidegger, 06/17
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
#
#=======================================================================

from parameters import *
from econ import *
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

# V infinity
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

    kap_nxt= (1-delta)*kap + inv

    #transform to comp. domain of the model
    kap_nxt_cube = kap_nxt #box_to_cube(kap_nxt)
    
    # initialize correct data format for training point
    s = (1,n_agt)
    Xtest = np.zeros(s)
    Xtest[0,:] = kap_nxt_cube
    
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
    N=len(X)
    M=n_ctt
    G=np.empty(M, float)
    
    # Extract Variables
    con=X[(i_con-1)*n_agt:i_con*n_agt]
    lab=X[(i_lab-1)*n_agt:i_lab*n_agt]
    inv=X[(i_inv-1)*n_agt:i_inv*n_agt]

    # variables for the market clearing constraints
    f_prod=output_f(kap, lab)
    gam_ad=Gamma_adjust(kap,inv)
    # sector sum (as defined in paper)
    if n_mcl == 1:
        #scs=con+inv-f_prod
        scs=con+inv-delta*kap-(f_prod-gam_ad)
    else:
    # canonical market clearing constraint
        mcl = con + inv - f_prod

    # constraints
    G[(i_con-1)*n_agt:i_con*n_agt] = con
    G[(i_lab-1)*n_agt:i_lab*n_agt] = lab
    G[(i_inv-1)*n_agt:i_inv*n_agt] = inv
    if n_mcl == 1:
        G[(i_mcl-1)*n_agt:i_mcl*n_agt] = sum(scs**2)
    else:
        G[(i_mcl-1)*n_agt:i_mcl*n_agt] = mcl

    return G

#======================================================================
#   Equality constraints during the VFI of the model
def EV_G_ITER(X, kap, n_agt):
#    N=len(X)
#    M=n_ctt
#    G=np.empty(M, float)
#
#    # Extract Variables
#    con=X[(i_con-1)*n_agt:i_con*n_agt]
#    lab=X[(i_lab-1)*n_agt:i_lab*n_agt]
#    inv=X[(i_inv-1)*n_agt:i_inv*n_agt]
#
#    # variables for the market clearing constraints
#    f_prod=output_f(kap, lab)
#    gam_ad=Gamma_adjust(kap,inv)
#    # sector sum (as defined in paper)
#    # scs=con+inv-delta*kap-(f_prod-gam_ad))
#    # canonical market clearing constraint
#    mcl = con + inv - f_prod
#    #
#    # constraints
#    G[(i_con-1)*n_agt:i_con*n_agt] = con
#    G[(i_lab-1)*n_agt:i_lab*n_agt] = lab
#    G[(i_inv-1)*n_agt:i_inv*n_agt] = inv
#
#    if n_ctt == 1:
#        G[(i_scs-1)*n_agt:i_scs*n_agt] = scs
#    else:
#        G[(i_mcl-1)*n_agt:i_mcl*n_agt] = mcl
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

    
    
    
    
    
    
    
    
    
            
            
            
    
    
    
    
    
    
