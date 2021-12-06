#======================================================================
#
#     This module contains routines to postprocess the VFI 
#     solutions.
#
#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
#======================================================================

import numpy as np
from parameters import *
#import cPickle as pickle
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

from datetime import datetime
from numpy.random import PCG64

#======================================================================    
# Routine compute the errors
def ls_error(n_agents, t1, t2, num_points):
    file=open('errors.txt', 'w')
    now = datetime.now()
    dt = int(now.strftime("%H%M%S%f"))
    print("Time seed = ", dt)
    rng = np.random.default_rng(dt)
    unif=rng.uniform(0, 1, (num_points, n_agents))
    #sample of states
    kap_smp = kap_L+(unif)*(kap_U-kap_L)
    to_print=np.empty((1,3))
        
    for i in range(t1, t2-1):
        sum_diffs=0
        diff = 0
        unif=rng.uniform(0, 1, (num_points, n_agents))
        #sample of states
        kap_smp = kap_L+(unif)*(kap_U-kap_L)
        # Load the model from the previous iteration step
        restart_data = filename + str(i) + ".pcl"
        with open(restart_data, 'rb') as fd_old:
            gp_old = pickle.load(fd_old)
            print("data from iteration step ", i , "loaded from disk")
        fd_old.close()      
      
        # Load the model from the previous iteration step
        restart_data = filename + str(i+1) + ".pcl"
        with open(restart_data, 'rb') as fd_new:
            gp_new = pickle.load(fd_new)
            print("data from iteration step ", i+1 , "loaded from disk")
        fd_new.close()        
      
        y_pred_old, sigma_old = gp_old.predict(kap_smp, return_std=True)
        y_pred_new, sigma_new = gp_new.predict(kap_smp, return_std=True)

        # plot predictive mean and 95% quantiles
        #for j in range(num_points):
            #print kap_smp[j], " ",y_pred_new[j], " ",y_pred_new[j] + 1.96*sigma_new[j]," ",y_pred_new[j] - 1.96*sigma_new[j]

        diff = y_pred_old-y_pred_new
        max_abs_diff=np.amax(np.fabs(diff))
        average = np.average(np.fabs(diff))
        
        to_print[0,0]= i+1
        to_print[0,1]= max_abs_diff
        to_print[0,2]= average
        
        np.savetxt(file, to_print, fmt='%2.16f')
        np.set_printoptions(suppress=True)
        msg=diff
        print(msg)
        print("===================================")

        
    file.close()
    
    return 
        
#======================================================================
