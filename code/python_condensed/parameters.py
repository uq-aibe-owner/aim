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
economic_verbose = True

# How many training points for GPR
n_agt = 2  # number of continuous dimensions of the model
No_samples = 10 * n_agt
# control of iterations
numstart = 1  # which is iteration to start (numstart = 1: start from scratch, number=/0: restart)
numits = 9  # which is the iteration to end

#length_scale_bounds=(10e-1,10e1)

alphaSK = 10e-1

filename = "restart/restart_file_step_"  # folder with the restart/result files


# arbitrary indices for the policy variables 

i_pol = {
    "con": 1,
    "lab": 2,
    "inv": 3,
    "knx": 4,
    #"INT": 5,
    #"INV": 6,
    #"int": 7
    #"uty": 8,
    #""
}

""" d_pol = {
    "con": 1,
    "lab": 1,
    "inv": 1,
    "knx": 1,
    "INT": 2,
    "INV": 2,
    "int": 1
}
 """


# number of policy variables, eg: con, lab, inv.
""" for iter in 
 """

n_pol = len(i_pol)*n_agt

# creating variable names from policy variable dict
# creating list of the dict keys
i_pol_key = list(i_pol.keys()) 
""" for iter in range(len(i_pol)):
    # retrieve current iterations policy variable string
    var = i_pol_key[iter]
    # turn into a variable name with "i_" in front
    globals()["i_"+var] = i_pol[var] """

# setup variables for constraints
i_ctt = {
    "mcl": 1,
    "knx": 2
}

# merge two index dicts
i = {**i_pol, **i_ctt}

# number of constraints
n_ctt = n_agt*len(i_ctt) #n_pol +

# creating variable names from constraint variable dict (similar to as with policy dict above)
i_ctt_key = list(i_ctt.keys()) 
"""for iter in range(len(i_ctt)):
    var = i_ctt_key[iter]
    # accounting for how policy variables are in the constraints vector
    globals()["i_"+var] = i_ctt[var] #len(i_pol)+ """


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

mcl_L = mcl_U = 0.0

# ======================================================================

# Number of test points to compute the error in the postprocessing
No_samples_postprocess = 20

#====================================================================== 
#utility function u(c,l) 

def utility(con=[], lab=[]):
    sum_util=0.0
    n=len(con)
    for iter in range(n):
        nom1=(con[iter]/big_A)**(1.0-gamma) -1.0
        den1=1.0-gamma
        
        nom2=(1.0-psi)*((lab[iter]**(1.0+eta)) -1.0)
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

def f_ctt(con, inv, lab, kap, knx):
    f_prod=output_f(kap, lab)
    e_ctt = dict()
    # canonical market clearing constraint
    e_ctt["mcl"] = con + inv - f_prod
    e_ctt["knx"] = (1-delta)*kap + inv - knx
#    e_ctt["blah blah blah"] = constraint rearranged into form that can be equated to zero
#    e_ctt["blah blah blah"] = constraint rearranged into form that can be equated to zero
    return e_ctt

#======================================================================
# adjustment cost
#
#def Gamma_adjust(kap=[], inv=[]):
#    fun_val = 0.5*zeta*kap*((inv/kap - delta)**2.0)
#    return fun_val

#======================================================================
# transformation to comp domain -- range of [k_bar, k_up]

""" def box_to_cube(knext=[]):
    # transformation onto cube [0,1]^d      
    knext_box = np.clip(knext, kap_L, kap_U)

    return knext_box
 """
#======================================================================
