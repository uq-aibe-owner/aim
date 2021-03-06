#======================================================================
# 
#     sets the economic functions for the "Growth Model", i.e., 
#     the production function, the utility function
#     
#
#     Simon Scheidegger, 11/16 ; 07/17
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
#====================================================================== 

from parameters import *
import numpy as np


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
# adjustment cost
#
def Gamma_adjust(kap=[], inv=[]):
    fun_val = 0.5*zeta*kap*((inv/kap - delta)**2.0)
    return fun_val

#======================================================================
# transformation to comp domain -- range of [k_bar, k_up]

def box_to_cube(knext=[]):
    # transformation onto cube [0,1]^d      
    knext_box = np.clip(knext, kap_L, kap_U)

    return knext_box

#======================================================================
