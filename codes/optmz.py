#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 22:50:13 2021

@author: yigongqin
"""


import importlib
import os, sys
import numpy as np
# from macro_input3 import phys_parameter, simu_parameter 
from op import phys_parameter, simu_parameter 
from scipy import sparse as sp
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
import scipy.io as sio
from math import pi
import time
from scipy.optimize import fsolve, brentq

from scipy.interpolate import interp2d as itp2d
from scipy.interpolate import interp1d 
from scipy.interpolate import griddata

### Analytical model

Q = 150 
x_span = 50
t_span = 800e-3


phys = phys_parameter( Q, x_span, t_span )
simu = simu_parameter( phys )

h  = simu.h
nx = simu.nx
ny = simu.ny
dt = simu.dt

t_source = np.linspace()



def objective_function(u, u_targ):
    
    nx = u.shape[0]
    ny = u.shape[1]
    
    return np.sum( 0.5*(u-u_targ)**2 )/(nx*ny)
    

def Hunt_model_Structure(G, R):

    '''
    Input:
        G (K/um), R (um/s) when the interface move upwards, the part always lsolid is -1 
    Output:
        S: the output morphology field based on Hunt model
    '''
    S = np.zeros_like(G)-100

    DTc = 2* np.pow( ( 2*phys.c_infm *(1-phys.k)*phys.GT/ phys.Dl )* R, 0.5)
    
    G_thre_u = 0.617* np.pow(100, 1./3)* phys.line_den * (1 - (phys.DT_N/DTc)**3 ) *DTc  ## fully columnar
    
    G_thre_d = 0.617*                    phys.line_den * (1 - (phys.DT_N/DTc)**3 ) *DTc  ## fully equiaxed


    S[G >= G_thre_u] = 1
    S[G_thre_d < G < G_thre_u] = 0.5
    S[0< G <=G_thre_d] = 0
    S[G<=0] = -1

    assert np.all(S>-100)

    return S 


def target_Structure( solid_y, c_per, e_per, type_layer, yy ):
    
    '''
    Input:
        solid_y: the y coordinates of S-L interface, there should be only one corner above the 
                 can be predetemined as analytical sol or can vary accoding to laser parameters
    Output:
        target structure
    '''

    lowest_y = solid_y[-1]  
    highest_y = solid_y[0]
    
    assert highest_y == yy[0, -1]
    assert lowest_y > yy[0, 0]     
    
    
    S = np.zeros_like(G) - 100
    cet_per = 1- e_per - c_per
    
    ## solid line is the y_max coordinate for every x location
    nx = yy.shape[0]
    ny = yy.shape[1]
    
    resolid = ( yy - np.repeat(solid_y[:,np.newaxis], ny, axis = 1 ) )>0
    resolid_all = np.sum(resolid)
    columnar_all = resolid_all*c_per
    equax_all = resolid_all*e_per
    cet_all = resolid*cet_per
    
    
    S[1-resolid] = -1
    
    if type_layer == 'c_all':      
        
        S[resolid] = 1
        
    elif type_layer == 'e_all':
        S[resolid] = 0
        
    elif type_layer == 'band':
        
        summ_c = 0
        summ_et = 0
        
        for j in range(lowest_y, highest_y):
            
            summ_c += 0
            
            
        
    elif type_layer == 'ring':
    
            pass
    
    else: raise ValueError('structure type not specified') 
    
    
    assert np.all(S>-100)
    
    return S



class laser_parameter():
      
    def __init__(self, num_laser, num_time, x_span, t_span, x_source, t_source):
        
        self.num_pulse = num_laser*num_time
        self.num_laser = num_laser
        self.num_time = num_time
        self.x_span = x_span
        self.t_span = t_span
        self.x0 = x_source
        self.t0 = t_source
        self.coeff = np.random.rand(num_laser, num_time)
        

    def laser_flux(self, x, t, phys):
        
        '''
        x is the coordinates on the top surface, t is just one number
        '''
        
        x_s = phys.n1 * np.exp( -2*(x-self.x0)**2 / (self.x_span/phys.len_scale)**2 )
        
        t_s = np.sum( self.coeff[0,:]*  np.exp( -2*(t-self.t0)**2 / (self.t_span/phys.time_scale)**2 )  )
        
        
        return x_s*t_s
    
    


S_targ = target_Structure( solid_y, c_per, e_per, type_layer, yy )


G, R = forward_heat()

S = Hunt_model_Structure(G, R)

J_func = lambda S: objective_function(S, S_targ)



















