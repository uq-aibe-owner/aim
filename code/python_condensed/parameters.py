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

verbose = True
economic_verbose = True

# How many training points for GPR
 # number of continuous dimensions of the model
n_agt = 2 
No_samples = 10 * n_agt
# control of iterations
numstart = 3 # which is iteration to start (numstart = 1: start from scratch, number=/0: restart)
numits = 4  # which is the iteration to end

length_scale_bounds=(10e-2,10e1)

alphaSK = 1e-0

filename = "restart/restart_file_step_"  # folder with the restart/result files

# arbitrary indices for the policy variables 
i_pol = {
    "con": 0,
    "lab": 1,
    "sav": 2,
    "knx": 3,
    "ITM": 4,
    "SAV": 5,
    "itm": 6
}

# dimensions of each policy variable
d_pol = {
    "con": 1,
    "lab": 1,
    "sav": 1,
    "knx": 1,
    "ITM": 2,
    "SAV": 2,
    "itm": 1
}

# setup variables for constraints
i_ctt = {
    "mclt": 0,
    "knxt": 1, # has to be a different key name to knx for combined dicts
    "savt": 2, # same story as above
    "itmt": 3  # same
}

d_ctt = {
    "mclt": 1,
    "knxt": 1, # has to be a different key name to knx for combined dicts
    "savt": 1, # same story as above
    "itmt": 1  # same
}
# ======================================================================
# Model Paramters

beta = 0.99
#rho = 0.95
#zeta = 0.0
phik = 0.4
phil = 0.35
gamma = 2.0
delta = 0.1
eta = 1
big_A = (1.0 - beta) / (phil**phil * phik**phik * (1-phik-phil)**(1-phik-phil) * beta)
xi = np.ones(n_agt)*1/n_agt
mu = np.ones(n_agt)*1/n_agt

# Ranges For States
kap_L = 2
kap_U = 5
range_cube = kap_U - kap_L  # range of [0..1]^d in 1D

# Ranges for Controls

# In same order as i_pol
pol_L = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
pol_U = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

# In same order as i_ctt
ctt_L = [-1e-3, -1e-3, -1e-3, -1e-3]
ctt_U = [1e-3, 1e-3, 1e-3, 1e-3]

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
        
        nom2=(1.0-phil)*((lab[iter]**(1.0+eta)) -1.0)
        den2=1.0+eta
        
        sum_util+=(nom1/den1 - nom2/den2)
    
    util=sum_util
    
    return util 

#====================================================================== 
# output_f 

def output_f(kap, lab, itm):
    fun_val = big_A*(np.power(kap,phik))*(np.power(lab, phil))*(np.power(lab,(1.0 - phik - phil)))
    return fun_val

#======================================================================
# Constraints

def f_ctt(con, sav, lab, kap, knx, SAV, ITM, itm):
    f_prod=output_f(kap, lab, itm)

    # Summing the 2d policy variables 
    SAV_com = np.zeros(n_agt, float)
    ITM_com = np.zeros(n_agt, float)
    for iter in range(n_agt):
        for ring in range(n_agt):
            SAV_com[iter] *= SAV[iter+n_agt*ring]**xi[ring]
            ITM_com[iter] *= ITM[iter+n_agt*ring]**mu[ring]

    e_ctt = dict()
    # canonical market clearing constraint
    e_ctt["mclt"] = np.add(con,sav) - f_prod
    e_ctt["knxt"] = (1-delta)*kap + sav - knx
    # intermediate sum constraints, just switch the first letter of the policy variables they are linked to with a "c", could change
    e_ctt["savt"] = SAV_com - sav
    e_ctt["itmt"] = ITM_com - itm
#    e_ctt["blah blah blah"] = constraint rearranged into form that can be equated to zero
    return e_ctt

#======================================================================
# Automated stuff, for indexing etc.

# creating list of the dict keys
i_pol_key = list(i_pol.keys())
# number of policy variables in total, to be used for lengths of X/x vectors
n_pol = 0
for iter in i_pol_key:
    n_pol += n_agt**d_pol[iter]
# dict for indices of each policy variable in X/x
I_pol = dict()
# temporary variable to keep track of previous highest index
prv_ind = 0
# allocating lists of indices to each policy variable as a key
for iter in i_pol_key:
    I_pol[iter] = slice(prv_ind,prv_ind+n_agt**d_pol[iter])
    prv_ind += n_agt**d_pol[iter]

# for use in running through loops
i_ctt_key = list(i_ctt.keys()) 
# number of constraints
n_ctt = 0
for iter in i_ctt_key:
    n_ctt += n_agt**d_ctt[iter]
# dict for indices of each constraint variable in G/g
I_ctt = dict()
# temporary variable to keep track of previous highest index
prv_ind = 0
# allocating lists of indices to each constraint variable as a key
for iter in i_ctt_key:
    I_ctt[iter] = slice(prv_ind,prv_ind+n_agt**d_ctt[iter])
    prv_ind += n_agt**d_ctt[iter]

# merge two index dicts into one for referencing
i = {**i_pol, **i_ctt} 