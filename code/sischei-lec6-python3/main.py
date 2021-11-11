# ======================================================================
#
#     This routine solves an infinite horizon growth model
#     with dynamic programming and sparse grids
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     external libraries needed:
#     - IPOPT (https://projects.coin-or.org/Ipopt)
#     - PYIPOPT (https://github.com/xuy/pyipopt)
#     - scikit-learn GPR (https://scikit-learn.org)
#
#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
# ======================================================================

import nonlinear_solver_initial as solver  # solves opt. problems for terminal VF
import nonlinear_solver_iterate as solviter  # solves opt. problems during VFI
from parameters import *  # parameters of model
import interpolation as interpol  # interface to sparse grid library/terminal VF
import interpolation_iter as interpol_iter  # interface to sparse grid library/iteration
import interpolation_combined as interpol_comb
import postprocessing as post  # computes the L2 and Linfinity error of the model
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
from econ import *

# ======================================================================
# Start with Value Function Iteration


# Set up a container to save the k/obj pairs as they are sampled & optimised

start = time.time()



for i in range(numstart, numits):
    # terminal value function

    """
    if i == 1:
        print("start with Value Function Iteration")
        interpol.GPR_init(i)

    elif i == numits - 1:
        print("Now, we are in Value Function Iteration step", i)
        ctnr = interpol_iter.GPR_iter(i)
    else:
        print("Now, we are in Value Function Iteration step", i)
        interpol_iter.GPR_iter(i)
    """ 
    interpol_comb.GPR_iter(i)

#for j in range(len(ctnr)):
#    sample_container["capital"].append(ctnr[j]['kap'])
#    sample_container["value"].append(ctnr[j]['obj'])
#    sample_container["iteration"].append(ctnr[j]['itr'])
#    sample_container["consumption"].append(ctnr[j]['con'])
#    sample_container["investment"].append(ctnr[j]['inv'])
#    sample_container["labor"].append(ctnr[j]['lab'])

# ======================================================================
print("===============================================================")
print(" ")
print(
    " Computation of a growth model of dimension ",
    n_agt,
    " finished after ",
    numits,
    " steps",
)
print(" ")
print("===============================================================")
# ======================================================================

# compute errors
avg_err = post.ls_error(n_agt, numstart, numits, No_samples_postprocess)

# ======================================================================
print("===============================================================")
print(" ")
# print " Errors are computed -- see error.txt"
print(" ")
print("===============================================================")
# ======================================================================
end = time.time()


#def plot_scatterplot():
#
#    # for all sampled points (not all will have converged, but will give an approximate view of the surface)
#    sample_container["kap"] = np.array(sample_container["kap"])
#    sample_container["value"] = np.array(sample_container["value"])
#
#    matplotlib.use("tkagg")
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection="3d")  # (projection='3d')
#    ax.set_xlabel("k (sector 1)")
#    ax.set_ylabel("k (sector 2)")
#    ax.set_zlabel("Value")
#
#    # colormap = matplotlib.cm(sample_container['iteration'])
#
#    img = ax.scatter(
#        sample_container["kap"][:, 0],
#        sample_container["kap"][:, 1],
#        sample_container["value"],
#        c=sample_container["iteration"],
#    )
#    plt.colorbar(img)
#
#    # plt.show()
#    plt.show()


def get_gaussian_process():
    with open("./restart/restart_file_step_" + str(numits - 1) + ".pcl", "rb") as fd:
        gp_old = pickle.load(fd)

    fd.close()
    return gp_old


def get_values(kap):
    Gaussian_Process = get_gaussian_process()
    values = Gaussian_Process.predict(kap, return_std=False)
    return values

def generate_random_k_vals(): 
    return np.random.uniform(kap_L+0.2, kap_U-0.2, (No_samples, n_agt)) 

def solve_for_kvals(kap, n_agt, gp_old): 

    result = np.empty((kap.shape[0]))
    for i in range(kap.shape[0]): 
        result[i] = solviter.iterate(k_init=kap[i], n_agt=n_agt, gp_old=gp_old)['obj']

    return result





def convergence_check():
    # tests for convergence by checking the predicted values at the sampled points of the final
    # iterate and then testing on the optimized value #v_old - val_tst

    # load the final instance of Gaussian Process

    np.random.seed(0)

    gp_old = get_gaussian_process() 

    random_k = generate_random_k_vals() 

    val_old = get_values(random_k) 

    val_new = solve_for_kvals(random_k, n_agt, gp_old)







    #for i in ctnr:
    #    kap_tst.append(i['kap'])
    #    val_tst.append(i['obj'])

    #kap_tst = np.array(kap_tst)
    #val_tst = np.array(val_tst)


    print("=================== Convergence Check ===================")
    print(" ")
    print("Should be close to zero for all values")

    np.set_printoptions(precision=2)

    print(val_old - val_new)

    print("maximum difference between value function iterates is",np.max(np.abs(val_old-val_new)))

    print("generated from k vals",random_k)

    return val_old - val_new


#def extract_variables(default=True, k_vals=None):
#    # extract the consumption, investment, labour variables (from the final iteration if default=True)
#    # if false, specify random points and calculate
#
#    kap_tst = []
#    val_tst = []
#    consumption = []
#    investment = []
#    labor = []
#
#    if default:
#        for i in ctnr:
#            kap_tst.append(i[0])
#            val_tst.append(i[1])
#            consumption.append(i[3])
#            investment.append(i[4])
#            labor.append(i[5])
#
#    kap_tst = np.array(kap_tst)
#    val_tst = np.array(val_tst)
#    consumption = np.array(consumption)
#    investment = np.array(investment)
#    labor = np.array(labor)
#
#    return kap_tst, val_tst, consumption, investment, labor


conv = convergence_check()
#kap_tst, val_tst, consumption, investment, labor = extract_variables()
# print(consumption)
# print(investment)
# print(labor)
def help():
    print(" ========== Finished ==========")
    print("Time elapsed: ", round(end - start, 2))

    print("Call variables kap_tst, val_tst, consumption, investment, labor")
    print("Use plot_scatterplot() to visualise")
    print("Use get_gaussian_process() to get the Gaussian Process")
    print(
        "Predict values for a given level of capital k with get_values(k), e.g. values = get_values(kap_tst)"
    )
    print("For options / prompts type help()")


help()

avg_err = post.ls_error(n_agt, numstart, numits, No_samples_postprocess)


#plot_scatterplot()


"""

"""

# test and debug
