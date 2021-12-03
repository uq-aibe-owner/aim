# ======================================================================
#
#     sets the parameters and economic functions for the model
#     "Growth Model"
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
# ======================================================================

import numpy as np

# ======================================================================

# Move all to NL solver.py

verbose = False 
economic_verbose = False

# How many training points for GPR
n_agt = 2  # number of continuous dimensions of the model
No_samples = 10 * n_agt

n_pol=3
# number of market clearing constraints: (in S&B17 this is 1)
n_mcl = n_agt
#n_fkp = n_agt

# arbitrary indices for the policy variables 

""" dct_pol = {
    "con": 1,
    "lab": 2,
    "inv": 3
   # "kap_nxt":4
} """

# number of policy variables, eg: con, lab, inv. per agent
#n_pol = len(dct_pol)*n_agt


# creating variable names from policy variable dict
# creating list of the dict keys
#dct_pol_key = list(dct_pol.keys()) 
#for iter in range(len(dct_pol)):
    # retrieve current iterations policy variable string
#    var = dct_pol_key[iter]
    # turn into a variable name with "i_" in front
#    globals()["i_"+var] = dct_pol[var]

i_con = 1 
i_lab = 2
i_inv = 3

# setup variables for constraints
""" dct_ctt_ind = {
    "mcl": 1
    #"knx": 2
} """

# number of constraints per agent
i_mcl = n_pol+1
n_ctt = n_pol*n_agt + n_mcl

# creating variable names from constraint variable dict (similar to as with policy dict above)
""" dct_ctt_ind_key = list(dct_ctt_ind.keys()) 
for iter in range(len(dct_ctt_ind)):
    var = dct_ctt_ind_key[iter]
    globals()["i_"+var] = dct_ctt_ind[var] """



# the i for the market clearing constraints
#i_mcl = 1
#n_ctt = n_mcl + n_fkp #n_pol * n_agt + n_mcl  # number of constraints

# control of iterations
numstart = 1  # which is iteration to start (numstart = 1: start from scratch, number=/0: restart)
numits = 10  # which is the iteration to end

length_scale_bounds=(10e-5,10e5)


alphaSK = 10e-3


filename = "restart/restart_file_step_"  # folder with the restart/result files

# ======================================================================
# Move this all to to econ.py
# Model Paramters

beta = 0.5
rho = 0.95
zeta = 0.0
psi = 0.86
gamma = 2.0
delta = 0.4
eta = 1
big_A = (1.0 - beta) / (psi * beta)

# Ranges For States
kap_L = 0.2
kap_U = 3
range_cube = kap_U - kap_L  # range of [0..1]^d in 1D


# Ranges for Controls
con_L = 1e-2
con_U = 10.0

lab_L = 1e-2
lab_U = 10.0

inv_L = 1e-2
inv_U = 10.0

#mcl_L = mcl_U = 0.0

# ======================================================================

# Number of test points to compute the error in the postprocessing
No_samples_postprocess = 20

#====================================================================== 
#utility function u(c,l) 

def utility(con=[], lab=[]):
    sum_util=0.0
    n=len(con)
    for i in range(n):
        nom1=(con[i]/big_A)**(1.0-gamma) -1.0
        den1=1.0-gamma
        
        nom2=(1.0-psi)*((lab[i]**(1.0+eta)) -1.0)
        den2=1.0+eta
        
        sum_util+=(nom1/den1 - nom2/den2)
    
    util=sum_util
    
    return util 


#====================================================================== 
# output_f 

def output_f(kap=[], lab=[]):
    fun_val = big_A*(kap**psi)*(lab**(1.0 - psi))
    return fun_val

#======================================================================
# Constraints

""" def fcn_ctt(con, inv, lab, kap, kap_nxt):
    f_prod=output_f(kap, lab)
    dct_ctt = dict()
    # canonical market clearing constraint
    dct_ctt["mcl"] = con + inv - f_prod
    #dct_ctt["knx"] = (1-delta)*kap + inv - kap_nxt
#    dct_ctt["blah blah blah"] = constraint rearranged into form that can be equated to zero
    return dct_ctt """

#======================================================================
# adjustment cost
#
#def Gamma_adjust(kap=[], inv=[]):
#    fun_val = 0.5*zeta*kap*((inv/kap - delta)**2.0)
#    return fun_val

#======================================================================
# transformation to comp domain -- range of [k_bar, k_up]

def box_to_cube(knext=[]):
    # transformation onto cube [0,1]^d      
    knext_box = np.clip(knext, kap_L, kap_U)

    return knext_box

#======================================================================
