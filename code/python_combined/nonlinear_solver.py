# ======================================================================
#
#     This routine interfaces with IPOPT
#     It sets the optimization problem for every training point
#     during the VFI.
#
#     Simon Scheidegger, 11/16 ; 07/17; 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021

#     Main difference is the shift from pyipopt to cyipopt
#     Involves a class to pass the optimisation problem to ipopt
# ======================================================================

from parameters import * 
from ipopt_wrapper_A import EV_F, EV_GRAD_F, EV_G, EV_JAC_G
from ipopt_wrapper_A import EV_F_ITER, EV_GRAD_F_ITER, EV_G_ITER, EV_JAC_G_ITER
import numpy as np

# import pyipopt
import cyipopt 

from HS071 import HS071 




def iterate(k_init, n_agt, gp_old=None, final=False, initial=False, verbose=False):

    # IPOPT PARAMETERS below "
    N = n_pol * n_agt * n_grid # number of vars
    M = n_ctt * n_grid # number of constraints
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
    X_L[(i_con-1)*n_agt* n_grid:i_con*n_agt* n_grid] = con_L
    X_U[(i_con-1)*n_agt* n_grid:i_con*n_agt* n_grid] = con_U

    X_L[(i_lab-1)*n_agt* n_grid:i_lab*n_agt* n_grid] = lab_L
    X_U[(i_lab-1)*n_agt* n_grid:i_lab*n_agt* n_grid] = lab_U

    X_L[(i_inv-1)*n_agt* n_grid:i_inv*n_agt* n_grid] = inv_L
    X_U[(i_inv-1)*n_agt* n_grid:i_inv*n_agt* n_grid] = inv_U

    # Set bounds for the constraints
    G_L[(i_con-1)*n_agt* n_grid:i_con*n_agt* n_grid] = con_L
    G_U[(i_con-1)*n_agt* n_grid:i_con*n_agt* n_grid] = con_U

    G_L[(i_lab-1)*n_agt* n_grid:i_lab*n_agt* n_grid] = lab_L
    G_U[(i_lab-1)*n_agt* n_grid:i_lab*n_agt* n_grid] = lab_U

    G_L[(i_inv-1)*n_agt* n_grid:i_inv*n_agt* n_grid] = inv_L
    G_U[(i_inv-1)*n_agt* n_grid:i_inv*n_agt* n_grid] = inv_U

    for iI in range(n_grid):

        #for the market clearing constraints
        mcl_L = mcl_U = 0.0
        if n_mcl == 1:
            G_L[(i_mcl-1)*n_agt* n_grid] = mcl_L
            G_U[(i_mcl-1)*n_agt* n_grid] = mcl_U
        else:
            G_L[(i_mcl-1)*n_agt* n_grid:i_mcl*n_agt* n_grid] = mcl_L
            G_U[(i_mcl-1)*n_agt* n_grid:i_mcl*n_agt* n_grid] = mcl_U

    # initial guesses for first iteration (aka a warm start)
    mu = 0.5
    con_init = mu*X_U[(i_con-1)*n_agt* n_grid:i_con*n_agt* n_grid]+(1-mu)*X_L[(i_con-1)*n_agt* n_grid:i_con*n_agt* n_grid]
    lab_init = mu*X_U[(i_lab-1)*n_agt* n_grid:i_lab*n_agt* n_grid]+(1-mu)*X_L[(i_lab-1)*n_agt* n_grid:i_lab*n_agt* n_grid]
    inv_init = mu*X_U[(i_inv-1)*n_agt* n_grid:i_inv*n_agt* n_grid]+(1-mu)*X_L[(i_inv-1)*n_agt* n_grid:i_inv*n_agt* n_grid]

    X[(i_con-1)*n_agt* n_grid:i_con*n_agt* n_grid] = con_init
    X[(i_lab-1)*n_agt* n_grid:i_lab*n_agt* n_grid] = lab_init
    """
    Superseded by cyipopt 
    # Create ev_f, eval_f, eval_grad_f, eval_g, eval_jac_g for given k_init and n_agent 
        
    def eval_f(X):
        return EV_F_ITER(X, k_init, n_agt, gp_old)
        
    def eval_grad_f(X):
        return EV_GRAD_F_ITER(X, k_init, n_agt, gp_old)
        
    def eval_g(X):
        return EV_G_ITER(X, k_init, n_agt)
        
    def eval_jac_g(X, flag):
        return EV_JAC_G_ITER(X, flag, k_init, n_agt)
    """ 

    HS07 = HS071(X, n_agents=n_agt, k_init=k_init, NELE_JAC=NELE_JAC, NELE_HESS=NELE_HESS, gp_old=gp_old, initial=initial, verbose=verbose) 

    """

    if initial: 
        from HS071_initial import HS071 as HS071_initial_run 
        HS07 = HS071_initial_run(
            X, n_agt, k_init, NELE_JAC, NELE_HESS
        )  # creates an instance of the class

    else: 
        from HS071_iter import HS071 as HS071_iterate

        HS07 = HS071_iterate(
            X, n_agt, k_init, NELE_JAC, NELE_HESS, gp_old
        )  # creates an instance of the class
    """ 

    """
    # First create a handle for the Ipopt problem 
    nlp=pyipopt.create(N, X_L, X_U, M, G_L, G_U, NELE_JAC, NELE_HESS, eval_f, eval_grad_f, eval_g, eval_jac_g)
    nlp.num_option("obj_scaling_factor", -1.0)
    nlp.num_option("tol", 1e-6)
    nlp.num_option("acceptable_tol", 1e-5)
    nlp.str_option("derivative_test", "first-order")
    nlp.str_option("hessian_approximation", "limited-memory")
    nlp.int_option("print_level", 1)
    
    x, z_l, z_u, constraint_multipliers, obj, status=nlp.solve(X)

    """
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

    optimal_soln, info = nlp.solve(X)

    x = info["x"]  # soln of the primal variables
    ctt = info["g"]  # constraint multipliers
    obj = info["obj_val"]  # objective value

    if final != True:
        nlp.close()

    # x: Solution of the primal variables
    # z_l, z_u: Solution of the bound multipliers
    # constraint_multipliers: Solution of the constraint multipliers
    # obj: Objective value
    # status: Exit Status

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
