# ======================================================================
#
#     This routine interfaces with Gaussian Process Regression
#     The crucial part is
#
#     y[iI] = solver.initial(Xtraining[iI], n_agt,gp_old)[0]
#     => at every training point, we solve an optimization problem
#
#     check kernels here: https://scikit-learn.org/stable/auto_examples/gaussian_process
#       /plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py
#
#
#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
# ======================================================================

import numpy as np
from parameters import *
import nonlinear_solver as solver

# import cPickle as pickle
import pickle
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from numpy.random import PCG64
from datetime import datetime
# ======================================================================


#def GPR_iter(iteration, rng, save_data=True):
def GPR_iter(iteration, save_data=True):
    
    if iteration == 0: 
        gp_old = None 

    elif iteration > 0: 
        # Load the model from the previous iteration step
        restart_data = filename + str(iteration - 1) + ".pcl"
        with open(restart_data, "rb") as fd_old:
            gp_old = pickle.load(fd_old)
            print("data from iteration step ", iteration - 1, "loaded from disk")
        fd_old.close()

    ##generate sample aPoints
    stt_nlp = time.time()
    dt = int(now.strftime("%d%H%M%S%f"))#"%M%S%f"))
    # fix seed
    rng = np.random.default_rng(12345)
    dim = n_agt
    Xtraining = rng.uniform(kap_L, kap_U, (No_samples, dim))
    y = np.zeros(No_samples, float)  # training targets

    ctnr = []
    # solve bellman equations at training points
    # Xtraining is our initial level of capital for this iteration

    for iI in range(len(Xtraining)):
        if iteration == 0: 
            res = solver.iterate(Xtraining[iI], n_agt,initial=True,verbose=verbose)
        else: 
            res = solver.iterate(Xtraining[iI], n_agt, gp_old,initial=False,verbose=verbose)
        SAV_add = np.zeros(n_agt, float)
        ITM_add = np.zeros(n_agt, float)
        for iter in range(n_agt):
            SAV_add[iter] = np.add(res["SAV"][iter*n_agt], res["SAV"][iter*n_agt+1])
            ITM_add[iter] = res["ITM"][iter*n_agt] + res["ITM"][iter*n_agt+1]
        res['kap'] = Xtraining[iI]
        res['itr'] = iteration
        y[iI] = res['obj']
        ctt = res['ctt']
        msg = "Constraint values: " + str(ctt) + os.linesep
        msg += "a quick check using output_f - con - SAV_add - ITM_add" + os.linesep
        msg += (
            str(output_f(Xtraining[iI], res['lab'], res["itm"]) - np.add(res['con'], SAV_add, ITM_add)) + os.linesep
        )
        msg += (
            "and consumption, labor, investment and intermediate inputs are, respectively,"
            + os.linesep
            + str(res['con'])
            + os.linesep
            + str(res['lab'])
            + os.linesep
            + str(res['SAV'])
            + str(res['sav'])
            + str(SAV_add)
            + os.linesep
            + str(res['ITM'])
            + str(res['itm'])
            + str(ITM_add)
        )
        if economic_verbose:
            print("{}".format(msg))
        if iteration == numits - 1:
            ctnr.append(res)
    end_nlp = time.time()
    # print data for debugging purposes
    # for iI in range(len(Xtraining)):
    # print Xtraining[iI], y[iI]

    # Instantiate a Gaussian Process model
    kernel = RBF(length_scale_bounds=length_scale_bounds) 

    # Instantiate a Gaussian Process model
    # kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

    # kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 2e2)) \
    # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e+0))

    # kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 2e2))
    # kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5)
    stt_gpr = time.time()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, alpha=alphaSK)
    end_gpr = time.time()
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(Xtraining, y)

    ##save the model to a file
    output_file = filename + str(iteration) + ".pcl"
    print(output_file)
    with open(output_file, "wb") as fd:
        pickle.dump(gp, fd, protocol=pickle.HIGHEST_PROTOCOL)
        print("data of step ", iteration, "  written to disk")
        print(" -------------------------------------------")
    fd.close()

    if iteration == numits - 1:
        return ctnr


# ======================================================================
