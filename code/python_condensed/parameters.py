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
 # number of continuous dimensions of the model
n_agt = 2 
No_samples = 10 * n_agt
# control of iterations
numstart = 1 # which is iteration to start (numstart = 1: start from scratch, number=/0: restart)
numits = 10  # which is the iteration to end

length_scale_bounds=(10e-2,10e1)

alphaSK = 10e-3

filename = "restart/restart_file_step_"  # folder with the restart/result files

# arbitrary indices for the policy variables 
i_pol = {
    "con": 1,
    "lab": 2,
    "inv": 3,
    "knx": 4,
    "ITM": 5,
    "INV": 6,
    "itm": 7
    #"uty": 8
    #""
}

# dimensions of each policy variable
d_pol = {
    "con": 1,
    "lab": 1,
    "inv": 1,
    "knx": 1,
    "ITM": 2,
    "INV": 2,
    "itm": 1
}

# creating list of the dict keys
i_pol_key = list(i_pol.keys()) 

# number of policy variables in total, to be used for lengths of X/x vectors
n_pol = 0
for iter in i_pol_key:
    n_pol += d_pol[iter]   
    
n_pol *= n_agt

# setup variables for constraints
i_ctt = {
    "mcl": 1,
    "cnx": 2, # has to be a different key name to knx for combined dicts
    "cnv": 3, # same story as above
    "ctm": 4  # same
}

# merge two index dicts into one for referencing
i = {**i_pol, **i_ctt}

# number of constraints
n_ctt = n_agt*len(i_ctt) 

# for use in running through loops
i_ctt_key = list(i_ctt.keys()) 

# dict for indices of each policy variable in X/x
I_pol = dict()
# temporary variable to keep track of previous highest index
prv_ind = 0
# allocating lists of indices to each policy variable as a key
for iter in i_pol_key:
    I_pol[iter] = np.arange(prv_ind,prv_ind+n_agt**d_pol[iter])
    prv_ind += n_agt**d_pol[iter]

# ======================================================================
# Model Paramters

beta = 0.99
#rho = 0.95
#zeta = 0.0
psi = 0.75
gamma = 2.0
delta = 0.1
eta = 1
big_A = (1.0 - beta) / (psi * beta)

# Ranges For States
kap_L = 2
kap_U = 5
range_cube = kap_U - kap_L  # range of [0..1]^d in 1D

# Ranges for Controls
con_L = 1e-2
con_U = 10.0

lab_L = 1e-2
lab_U = 10.0

inv_L = 1e-2
inv_U = 10.0

knx_L = 1e-2
knx_U = 10.0

INV_L = 1e-2
INV_U = 10.0

ITM_L = 1e-2
ITM_U = 10.0

itm_L = 1e-2
itm_U = 10.0

mcl_L = -1e-1
mcl_U = 1e-1

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

def output_f(kap, lab):
    fun_val = big_A*(np.power(kap,psi))*(np.power(lab,(1.0 - psi)))
    return fun_val

#======================================================================
# Constraints

def f_ctt(con, inv, lab, kap, knx, INV, ITM, itm):
    f_prod=output_f(kap, lab)
    e_ctt = dict()
    # canonical market clearing constraint
    e_ctt["mcl"] = np.add(con,inv) - f_prod
    e_ctt["cnx"] = (1-delta)*kap + inv - knx
    # intermediate sum constraints, just switch the first letter of the policy variables they are linked to with a "c", could change
    e_ctt["cnv"] = np.sum(INV,axis=0) - inv
    e_ctt["ctm"] = np.sum(ITM,axis=0) - itm
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
