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
from ipopt_wrapper_A import ipopt_obj

import numpy as np

# import pyipopt
import cyipopt 

def iterate(k_init, n_agt, gp_old=None, final=False, initial=False, verbose=False):

    # IPOPT PARAMETERS below "
    N = n_pol  # number of vars
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
    X_L[(i["con"]-1)*n_agt:i["con"]*n_agt] = con_L
    X_U[(i["con"]-1)*n_agt:i["con"]*n_agt] = con_U

    X_L[(i["lab"]-1)*n_agt:i["lab"]*n_agt] = lab_L
    X_U[(i["lab"]-1)*n_agt:i["lab"]*n_agt] = lab_U

    X_L[(i["inv"]-1)*n_agt:i["inv"]*n_agt] = inv_L
    X_U[(i["inv"]-1)*n_agt:i["inv"]*n_agt] = inv_U

    # Set bounds for the constraints
    """ G_L[(i["con"]-1)*n_agt:i["con"]*n_agt] = con_L
    G_U[(i["con"]-1)*n_agt:i["con"]*n_agt] = con_U

    G_L[(i["lab"]-1)*n_agt:i["lab"]*n_agt] = lab_L
    G_U[(i["lab"]-1)*n_agt:i["lab"]*n_agt] = lab_U

    G_L[(i["inv"]-1)*n_agt:i["inv"]*n_agt] = inv_L
    G_U[(i["inv"]-1)*n_agt:i["inv"]*n_agt] = inv_U """

    #for the market clearing constraints
    """ G_L[(i_mcl-1)*n_agt:i_mcl*n_agt] = mcl_L
    G_U[(i_mcl-1)*n_agt:i_mcl*n_agt] = mcl_U """
    G_L[:] = mcl_L
    G_U[:] = mcl_U

    # initial guesses for first iteration (aka a warm start)
    mu = 0.5
    con_init = mu*X_U[(i["con"]-1)*n_agt:i["con"]*n_agt]+(1-mu)*X_L[(i["con"]-1)*n_agt:i["con"]*n_agt]
    lab_init = mu*X_U[(i["lab"]-1)*n_agt:i["lab"]*n_agt]+(1-mu)*X_L[(i["lab"]-1)*n_agt:i["lab"]*n_agt]
    inv_init = mu*X_U[(i["inv"]-1)*n_agt:i["inv"]*n_agt]+(1-mu)*X_L[(i["inv"]-1)*n_agt:i["inv"]*n_agt]

    X[(i["con"]-1)*n_agt:i["con"]*n_agt] = con_init
    X[(i["lab"]-1)*n_agt:i["lab"]*n_agt] = lab_init
    #X[(i["inv"]-1)*n_agt:i["inv"]*n_agt] = inv_init

    HS07 = ipopt_obj(X, n_agents=n_agt, k_init=k_init, NELE_JAC=NELE_JAC, NELE_HESS=NELE_HESS, gp_old=gp_old, initial=initial, verbose=verbose) 

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
    nlp.add_option("tol", alphaSK)
    nlp.add_option("print_level", 0)
    nlp.add_option("hessian_approximation", "limited-memory")

    optimal_soln, info = nlp.solve(X)

    x = info["x"]  # soln of the primal variables
    ctt = info["g"]  # constraints
    obj = info["obj_val"]  # objective value

    if final != True:
        nlp.close()

    # x: Solution of the primal variables
    # z_l, z_u: Solution of the bound multipliers
    # constraint_multipliers: Solution of the constraint
    # obj: Objective value
    # status: Exit Status

    # Unpack Consumption, Labor, and Investment
    """ for iter in i_pol_key:
        globals()[iter] = x[(i[iter]-1)*n_agt:i[iter]*n_agt] """

    con = x[(i["con"]-1)*n_agt:i["con"]*n_agt]
    lab = x[(i["lab"]-1)*n_agt:i["lab"]*n_agt]
    inv = x[(i["inv"]-1)*n_agt:i["inv"]*n_agt]
    #knx = x[(i["knx"]-1)*n_agt:i["knx"]*n_agt]

    to_print = np.hstack((obj, x))

    # == debug ==
    # f=open("results.txt", 'a')
    # np.savetxt(f, np.transpose(to_print) #, fmt=len(x)*'%10.10f ')
    # for num in to_print:
    #    f.write(str(num)+"\t")
    # f.write("\n")
    # f.close()
    res = dict()
    res['obj'] = obj
    res['con'] = con
    res['lab'] = lab
    res['inv'] = inv
    #res['knx'] = knx
    res['ctt'] = ctt
    return res
