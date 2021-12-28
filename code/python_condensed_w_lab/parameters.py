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
## Verbosity of print output
verbose = True
economic_verbose = True
# ======================================================================
# number of continuous dimensions of the model
n_agt = 2
# Number of training points to generate via NLP for GPR
No_samples = 10 * n_agt
# ======================================================================
## Control of Iterations
# To start from scratch, set numstart = 0.
# Otherwise set numstart equal to previous numits. (Equivalently,
#set numstart equal to the last restart file number plus one.)
numstart = 0
#how many iterations
fthrits = 4
numits = numstart + fthrits
# ======================================================================
# Number of test points to compute the error in postprocessing
No_samples_postprocess = 20

# ======================================================================
length_scale_bounds=(10e-9,10e10)

alphaSK = 1e-1
n_restarts_optimizer=10
filename = "restart/restart_file_step_"  # folder with the restart/result files

# dimensions of each policy variable
d_pol = {
    "con": 1,
    "lab": 1,
    "sav": 1,
    "knx": 1,
    "ITM": 2,
    "SAV": 2,
    "itm": 1,
    "val": 0,
    "utl": 0,
    "out": 1
}

# dimensions of each constraint variable
d_ctt = {
    "mclt": 1,
    "knxt": 1,
    "savt": 1, 
    "itmt": 1, 
    "valt": 0,
    "utlt": 0,
    "outt": 1
}
# ======================================================================
# Model Paramters

beta = 0.99
#rho = 0.95
#zeta = 0.0
""" phi = {
    "itm": 0.5,
    "kap": 0.5
} """

phik =0.35
phim =0.35

gamma = 2.0
delta = 0.1
eta = 1
big_A = 1/(phim**phim * phik**phik * (1-phik-phim)**(1-phik-phim))
xi = np.ones(n_agt)*1/n_agt
mu = np.ones(n_agt)*1/n_agt

# Ranges For States
kap_L = 2
kap_U = 5
range_cube = kap_U - kap_L  # range of [0..1]^d in 1D

# Ranges for Controls
pL = 1e-1
pU = 1e3
# Lower policy variables bounds
pol_L = {
    "con": 1.1,
    "lab": pL,
    "sav": pL,
    "knx": pL,
    "ITM": pL,
    "SAV": pL,
    "itm": pL,
    "val": -pU,
    "utl": pL,
    "out": pL
}
# Upper policy variables bounds
pol_U = {
    "con": pU,
    "lab": pU,
    "sav": pU,
    "knx": pU,
    "ITM": pU,
    "SAV": pU,
    "itm": pU,
    "val": pU,
    "utl": pU,
    "out": pU
}
# Warm start
pol_S = {
    "con": 10,
    "lab": 10,
    "sav": 10,
    "knx": 10,
    "ITM": 10,
    "SAV": 10,
    "itm": 10,
    "val": -300,
    "utl": 10,
    "out": 10
}

if not len(d_pol)==len(pol_U)==len(pol_L)==len(pol_S):
    raise ValueError("Policy-variable-related Dicts are not all the same length, check parameters.py")

# Constraint variables bounds
cL = 0*-1e-5
cU = 0*1e-5
ctt_L = {
    "mclt": cL,
    "knxt": cL,
    "savt": cL, 
    "itmt": cL, 
    "valt": cL,
    "utlt": cL,
    "outt": cL
}
ctt_U = {
    "mclt": cU,
    "knxt": cU,
    "savt": cU, 
    "itmt": cU, 
    "valt": cU,
    "utlt": cU,
    "outt": cU
}

# Check dicts are all same length
if not len(d_ctt)==len(ctt_U)==len(ctt_L):
    raise ValueError("Constraint-related Dicts are not all the same length, check parameters.py")

#====================================================================== 
#utility function u(c,l) 
def utility(con, lab):
    return sum(np.log(con)) + sum(lab) # -J could make cobb-douglas, may fix /0 issue

#====================================================================== 
# initial guess of the value function v(k)
def V_INFINITY(kap, lab):
    e=np.ones(len(kap))
    c=output_f(kap, kap/3, lab)
    v_infinity=utility(c, lab)/(1-beta)
    return v_infinity

#====================================================================== 
# output_f
def output_f(kap, itm, lab):
    fun_val = big_A*kap**phik*itm**phim*lab**(1- phik - phim)
    return fun_val

#====================================================================== 
# output_f
def value_f(init, gp_old, Kap2, lab):
    if init:
        return V_INFINITY(Kap2, lab)
    else:
        return gp_old.predict(Kap2, return_std=True)[1]
        
#======================================================================
# Constraints

def f_ctt(X, gp_old, Kap2, init, kap):
    #f_prod=output_f(kap, lab, itm)

    # Summing the 2d policy variables 
    SAV_com = np.ones(n_agt, float)
    SAV_add = np.zeros(n_agt, float)
    ITM_com = np.ones(n_agt, float)
    ITM_add = np.zeros(n_agt, float)
    for iter in range(n_agt):
        for ring in range(n_agt):
            SAV_com[iter] *= X[I["SAV"]][iter+n_agt*ring]**xi[ring]
            ITM_com[iter] *= X[I["ITM"]][iter+n_agt*ring]**mu[ring]
            SAV_add[iter] += X[I["SAV"]][iter*n_agt+ring]
            ITM_add[iter] += X[I["ITM"]][iter*n_agt+ring]
    e_ctt = dict()
    # canonical market clearing constraint
    e_ctt["mclt"] = X[I["con"]] + SAV_add + ITM_add - X[I["out"]]
    # capital next period constraint
    e_ctt["knxt"] = (1-delta)*kap + X[I["sav"]] - X[I["knx"]]
    # intermediate sum constraints
    e_ctt["savt"] = SAV_com - X[I["sav"]]
    e_ctt["itmt"] = ITM_com - X[I["itm"]]
    # value function constraint
    e_ctt["valt"] = X[I["val"]] - sum(value_f(init, gp_old, Kap2, X[I["lab"]]))
    # output constraint
    e_ctt["outt"] = X[I["out"]] - output_f(kap,X[I["itm"]],X[I["lab"]])
    #utility constraint
    e_ctt["utlt"] = X[I["utl"]] - utility(X[I["con"]], X[I["lab"]])
    #e_ctt["blah blah blah"] = constraint rearranged into form that can be equated to zero
    
    # Check dicts are all same length
    if not len(d_ctt)==len(ctt_U)==len(ctt_L)==len(e_ctt):
        raise ValueError("Constraint-related Dicts are not all the same length, check f_cct in parameters.py")

    return e_ctt

#======================================================================
# Automated stuff, for indexing etc, shouldnt need to be altered if we are just altering economics

# creating list of the dict keys
pol_key = list(d_pol.keys())
# number of policy variables in total, to be used for lengths of X/x vectors
n_pol = 0
# temporary variable to keep track of previous highest index
prv_ind = 0
# dict for indices of each policy variable in X/x
I = dict()
for iter in pol_key:
    n_pol += n_agt**d_pol[iter]
    # allocating slices of indices to each policy variable as a key
    I[iter] = slice(prv_ind,prv_ind+n_agt**d_pol[iter])
    prv_ind += n_agt**d_pol[iter]

# for use in running through loops
ctt_key = list(d_ctt.keys()) 
# number of constraints
n_ctt = 0
# dict for indices of each constraint variable in G/g
I_ctt = dict()
# temporary variable to keep track of previous highest index
prv_ind = 0
for iter in ctt_key:
    # add to number of total constraint values
    n_ctt += n_agt**d_ctt[iter]
    # allocating slicess of indices to each constraint variable as a key
    I_ctt[iter] = slice(prv_ind,prv_ind+n_agt**d_ctt[iter])
    prv_ind += n_agt**d_ctt[iter]