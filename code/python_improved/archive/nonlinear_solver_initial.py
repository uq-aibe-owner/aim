# ======================================================================
#
#     This routine interfaces with IPOPT
#     It sets the optimization problem for every training point
#     at the beginning of the VFI.
#
#     Simon Scheidegger, 11/16 ; 07/17; 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
#     Main difference is the shift from pyipopt to cyipopt
#     Involves a class to pass the optimisation problem to ipopt
# ======================================================================

from parameters import *
from ipopt_wrapper_A import EV_F, EV_GRAD_F, EV_G, EV_JAC_G
import numpy as np

# import pyipopt
import cyipopt
from HS071_initial import *

# ======================================================================


def initial(k_init, n_agt):
    # IPOPT PARAMETERS below

    N = n_pol * n_agt  # number of vars
    M = n_ctt  # number of constraints
    NELE_JAC = N * M
    NELE_HESS = (N+1)*N/2  # number of non-zero entries of Hess matrix

    # Vector of variables -> solution of non-linear equation system
    X = np.empty(N)

    LAM = np.empty(M)  # multipliers
    G = np.empty(M)  # (in-)equality constraints

    # Vector of lower and upper bounds
    G_L = np.empty(M)
    G_U = np.empty(M)

    X_L = np.empty(N)
    X_U = np.empty(N)

    Z_L = np.empty(N)
    Z_U = np.empty(N)

    # get coords of an individual grid points
    grid_pt_box = k_init
    X_L[(i_con-1)*n_agt:i_con*n_agt] = con_L
    X_U[(i_con-1)*n_agt:i_con*n_agt] = con_U

    X_L[(i_lab-1)*n_agt:i_lab*n_agt] = lab_L
    X_U[(i_lab-1)*n_agt:i_lab*n_agt] = lab_U

    X_L[(i_inv-1)*n_agt:i_inv*n_agt] = inv_L
    X_U[(i_inv-1)*n_agt:i_inv*n_agt] = inv_U

    # Set bounds for the constraints
    G_L[(i_con-1)*n_agt:i_con*n_agt] = con_L
    G_U[(i_con-1)*n_agt:i_con*n_agt] = con_U

    G_L[(i_lab-1)*n_agt:i_lab*n_agt] = lab_L
    G_U[(i_lab-1)*n_agt:i_lab*n_agt] = lab_U

    G_L[(i_inv-1)*n_agt:i_inv*n_agt] = inv_L
    G_U[(i_inv-1)*n_agt:i_inv*n_agt] = inv_U

    #for the market clearing constraints
    mcl_L = mcl_U = 0.0
    if n_mcl == 1:
        G_L[(i_mcl-1)*n_agt] = mcl_L
        G_U[(i_mcl-1)*n_agt] = mcl_U
    else:
        G_L[(i_mcl-1)*n_agt:i_mcl*n_agt] = mcl_L
        G_U[(i_mcl-1)*n_agt:i_mcl*n_agt] = mcl_U

    # initial guesses for first iteration (aka a warm start)
    mu = 0.5
    con_init = mu*X_U[(i_con-1)*n_agt:i_con*n_agt]+(1-mu)*X_L[(i_con-1)*n_agt:i_con*n_agt]
    lab_init = mu*X_U[(i_lab-1)*n_agt:i_lab*n_agt]+(1-mu)*X_L[(i_lab-1)*n_agt:i_lab*n_agt]
    inv_init = mu*X_U[(i_inv-1)*n_agt:i_inv*n_agt]+(1-mu)*X_L[(i_inv-1)*n_agt:i_inv*n_agt]

    X[(i_con-1)*n_agt:i_con*n_agt] = con_init
    X[(i_lab-1)*n_agt:i_lab*n_agt] = lab_init
    X[(i_inv-1)*n_agt:i_inv*n_agt] = inv_init
    # X=np.ones(N)

    """
    Superseded by cyipopt object 
    # Create ev_f, eval_f, eval_grad_f, eval_g, eval_jac_g for given k_init and n_agent 
    def eval_f(X):
        return EV_F(X, k_init, n_agt)
    
    def eval_grad_f(X):
        return EV_GRAD_F(X,k_init, n_agt)
    
    def eval_g(X):
        return EV_G(X, k_init, n_agt)
        
    def eval_jac_g(X, flag):
        return EV_JAC_G(X, flag, k_init, n_agt)
    """

    # create problem object
    HS07 = HS071(X, n_agt, k_init, NELE_JAC, NELE_HESS)

    # First create a handle for the Ipopt problem
    # nlp=pyipopt.create(N, X_L, X_U, M, G_L, G_U, NELE_JAC, NELE_HESS, eval_f, eval_grad_f, eval_g, eval_jac_g)

    nlp = cyipopt.Problem(
        n=N,
        m=M,
        problem_obj=HS07,
        lb=X_L,
        ub=X_U,
        cl=G_L,
        cu=G_U,
    )
    nlp.add_option("obj_scaling_factor", -1.00)  # max function
    nlp.add_option("mu_strategy", "adaptive")
    nlp.add_option("tol", 1e-4)
    nlp.add_option("print_level", 0)
    nlp.add_option("hessian_approximation", "limited-memory")

    # x, z_l, z_u, constraint_multipliers, obj, status=nlp.solve(X)
    optimal_soln, info = nlp.solve(X)

    x = info["x"]  # soln of the primal variables
    ctt = info["g"]  # constraint values
    obj = info["obj_val"]  # objective value

    nlp.close()

    # Unpack Consumption, Labor, and Investment
    con = x[(i_con-1)*n_agt:i_con*n_agt]
    lab = x[(i_lab-1)*n_agt:i_lab*n_agt]
    inv = x[(i_inv-1)*n_agt:i_inv*n_agt]

    to_print = np.hstack((obj, x))

    # == debug ==
    # f=open("results.txt", 'a')
    # np.savetxt(f, np.transpose(to_print) #, fmt=len(x)*'%10.10f ')
    # for num in to_print:
    #    f.write(str(num)+"\t")
    # f.write("\n")
    # f.close()
    res = dict();
    res['obj'] = obj
    res['con'] = con
    res['lab'] = lab
    res['inv'] = inv
    res['ctt'] = ctt
    return res
