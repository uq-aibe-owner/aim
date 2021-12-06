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
    
    if iteration == 1: 
        gp_old = None 

    elif iteration > 1: 
        # Load the model from the previous iteration step
        restart_data = filename + str(iteration - 1) + ".pcl"
        with open(restart_data, "rb") as fd_old:
            gp_old = pickle.load(fd_old)
            print("data from iteration step ", iteration - 1, "loaded from disk")
        fd_old.close()

    ##generate sample aPoints
    now = datetime.now()
    dt = int(now.strftime("%H"))#%M%S%f"))
    # fix seed
    rng = np.random.default_rng(dt)
    dim = n_agt
    Xtraining = rng.uniform(kap_L, kap_U, (No_samples, dim))
    y = np.zeros(No_samples, float)  # training targets

    ctnr = []
    # solve bellman equations at training points
    # Xtraining is our initial level of capital for this iteration
    for iI in range(len(Xtraining)):
        if iteration == 1: 
            res = solver.iterate(Xtraining[iI], n_agt,initial=True,verbose=verbose)
        else: 
            res = solver.iterate(Xtraining[iI], n_agt, gp_old,initial=False,verbose=verbose)

        res['kap'] = Xtraining[iI]
        res['itr'] = iteration
        y[iI] = res['obj']
        ctt = res['ctt']
        msg = "Excess demand is " + str(ctt[len(ctt) - n_agt: len(ctt)]) + os.linesep
        msg += "a quick check using output_f - consumption - investment" + os.linesep
        msg += (
            str(output_f(Xtraining[iI], res['lab']) - res['con'] - res['inv']) + os.linesep
        )
        msg += (
            "and consumption, labor and investment are, respectively,"
            + os.linesep
            + str(res['con'])
            + str(res['lab'])
            + str(res['inv'])
        )
        if economic_verbose:
            print("{}".format(msg))
        if iteration == numits - 1:
            ctnr.append(res)

    # print data for debugging purposes
    # for iI in range(len(Xtraining)):
    # print Xtraining[iI], y[iI]

    # Instantiate a Gaussian Process model
    kernel = RBF()#length_scale_bounds=length_scale_bounds) 

    # Instantiate a Gaussian Process model
    # kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

    # kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 2e2)) \
    # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e+0))

    # kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 2e2))
    # kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=alphaSK)#n_restarts_optimizer=10, alpha=alphaSK)

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
