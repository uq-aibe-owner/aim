
#======================================================================
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
#======================================================================

import nonlinear_solver_initial as solver     #solves opt. problems for terminal VF
import nonlinear_solver_iterate as solviter   #solves opt. problems during VFI
from parameters import *                      #parameters of model
import interpolation as interpol              #interface to sparse grid library/terminal VF
import interpolation_iter as interpol_iter    #interface to sparse grid library/iteration
import postprocessing as post                 #computes the L2 and Linfinity error of the model
import numpy as np


#import cPickle as pickle
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern


import matplotlib.pyplot as plt 
#======================================================================
# Start with Value Function Iteration


for i in range(numstart, numits):
# terminal value function
    if (i==1):
        print("start with Value Function Iteration")
        interpol.GPR_init(i)
    
    else:     
        print("Now, we are in Value Function Iteration step", i)
        interpol_iter.GPR_iter(i)
    
    
#======================================================================
print("===============================================================")
print(" ")
print(" Computation of a growth model of dimension ", n_agents ," finished after ", numits, " steps")
print(" ")
print("===============================================================")
#======================================================================

# compute errors   
avg_err=post.ls_error(n_agents, numstart, numits, No_samples_postprocess)

print("Average error is: ", avg_err)

#======================================================================
print("===============================================================")
print(" ")
print(" Errors are computed -- see error.txt")
print(" ")
print("===============================================================")
#======================================================================


print("interpreting data")
# Load the model from the previous iteration step
restart_data = filename + str(numits-1) + ".pcl"
with open(restart_data, 'rb') as fd_old:
    gp_old = pickle.load(fd_old)
    print("data from iteration step ", numits -1 , "loaded from disk")
fd_old.close()

#ans = []
x_ = [] 
c_ = [] 
l_ = [] 
inv_ = [] 

np.random.seed(100)   #fix seed
dim = n_agents
Xtraining = np.random.uniform(k_bar, k_up, (No_samples*2, dim))
y = np.zeros(No_samples*2, float) # training targets    
print("solve")
# solve bellman equations at training points
for iI in range(len(Xtraining)):
    print(iI)
    #y[iI] = solviter.iterate(Xtraining[iI], n_agents,gp_old)[0] 
    obj, c, l, inv = solviter.iterate(Xtraining[iI], n_agents,gp_old)
    x_.append(Xtraining[iI])
    c_.append(c)
    l_.append(l)
    inv_.append(inv)

print('x',x_)
print('c',c_)
print('l',l_)
print('inv',inv_) 

plt.plot(x_,c_)




#plt.plot(Xtraining,y)
#plt.show()
print("done")
