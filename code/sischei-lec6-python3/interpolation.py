#======================================================================
#
#     This routine interfaces with Gaussian Process Regression
#     The crucial part is 
#
#     y[iI] = solver.initial(Xtraining[iI], n_agt)[0]
#     => at every training point, we solve an optimization problem
#
#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
#======================================================================

import numpy as np
from parameters import *
import nonlinear_solver_initial as solver
#import cPickle as pickle
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

#======================================================================

def GPR_init(iteration, save_data=True):
    
    print("hello from step ", iteration)

    # create container to save samples 

    iter_container = [] 
    
  
    #fix seed
    np.random.seed(666)
    
    #generate sample aPoints
    dim = n_agt
    Xtraining = np.random.uniform(kap_L, kap_U, (No_samples, dim))
    y = np.zeros(No_samples, float) # training targets
    
    # solve bellman equations at training points
    for iI in range(len(Xtraining)):
        res = solver.initial(Xtraining[iI], n_agt)
        y[iI] = res['obj']
       # print(#Xtraining[iI],
              #y[iI],
              #iteration,
              #consumption,
              #investment,
              #labor,
       #       g)
       # iter_container.append([Xtraining[iI],
       #                        y[iI],
       #                        iteration,
       #                        consumption,
       #                        investment,
       #                        labor])

#    for iI in range(len(Xtraining)):
#        print(Xtraining[iI], y[iI])
        
    # Instantiate a Gaussian Process model
    kernel = RBF() 
      
    #kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    #kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5)
    
    #kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 2e2)) \
    #+ WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e+0))   
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(Xtraining, y)    
     
    #save the model to a file
    output_file = filename + str(iteration) + ".pcl"
    print(output_file)
    with open(output_file, 'wb') as fd:
        pickle.dump(gp, fd, protocol=pickle.HIGHEST_PROTOCOL)
        print("data of step ", iteration ,"  written to disk")
        print(" -------------------------------------------")
    fd.close()
    

    if save_data: 
        return iter_container 
#======================================================================

