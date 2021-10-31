#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 22:50:13 2021

@author: yigongqin
"""

import cma

import importlib
import os, sys, copy
import numpy as np

from optmz_input import phys_parameter, simu_parameter 
from scipy import sparse as sp
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection

import scipy.io as sio
from math import pi
import time
from scipy.optimize import fsolve, brentq, minimize

from scipy.interpolate import interp2d as itp2d
from scipy.interpolate import interp1d 
from scipy.interpolate import griddata

from forward_heat_func import export_mat, dphase_trans, sparse_cg, sparse_laplacian, set_top_bc

def len_dim(x): return x *  phys.len_scale      # [m]
def len_um(x): return x *  phys.len_scale * 1e6    # [um]
def temp_dim(u): return u * phys.deltaT + phys.Ts  # [K]
def heat_dim(u): return u * phys.deltaT  # [K]

'''
u_funcs
'''


def Hunt_model(G, R, phys):
    
    '''
    G, R is a number, phys is class for parameters
    
    '''
    
    if G<0:
        
        return 0
    
    DTc = 2* np.sqrt( ( 2*phys.c_infm *(1-phys.k)*phys.GT/ phys.Dl )* R )
   # print(DTc)
    
    G_thre_u = 0.617* np.power(100, 1./3)* phys.line_den * (1 - (phys.DT_N/DTc)**3 ) *DTc  ## fully columnar
    
    G_thre_d = 0.617*                      phys.line_den * (1 - (phys.DT_N/DTc)**3 ) *DTc  ## fully equiaxed


    if G > G_thre_u: return 1
    elif G > G_thre_d: return 2
    elif G > 0: return 3
    else: raise ValueError('whats wrong here')


def resolid_condition(u):
    
    try:
        return list((u>0)*1).index(1)
    except ValueError:
        return -1

'''
u_funcs
'''

def input_constraints(m, up_limit):
    
    return (np.all(m>0)) and (np.all(m<up_limit))



def disctns_objective_function(m_targ, pl, solver, struct_class):
    
    if not input_constraints(m_targ, up_limit):
        print('failed, not in the input space')
        return np.NaN


    get_structure, S = solver.forward_solve(pl, m_targ)
    
    if not get_structure: 
        print('failed, not getting a melt pool')
        return np.NaN
    
    else:
    
    
        S_targ = target_Structure( S, struct_class )
        new_solid_area = np.sum( (S_targ>0)*1 )
        fval =  np.sum( (S-S_targ)**2 )/new_solid_area      
        print('value of function evaluation', fval)
        return fval
    




def target_Structure( S, p):
    
    '''
    Input:
        solid_y: the y coordinates of S-L interface, there should be only one corner above the 
                 can be predetemined as analytical sol or can vary accoding to laser parameters
    Output:
        target structure
    '''
    
    new_solid_area = np.sum( (S>0)*1 )
    resolid_y = np.apply_along_axis(resolid_condition, 1, S)      

    cet_per = 1- p.e_per - p.c_per
    
    ## solid line is the y_max coordinate for every x location
    nx = S.shape[0]
    ny = S.shape[1]
    
    up_bound = ny - 1
    x_idx = np.arange(nx)
    
    columnar_all = int(new_solid_area*p.c_per)
    cet_all = int(new_solid_area*cet_per)
    
    
    S_targ = copy.deepcopy(S)
    location = copy.deepcopy(resolid_y)
    
    where_resolid = np.where(S>0)
    S_targ[where_resolid] = 3
    
    if p.type_layer == 'columnar':
       
        S_targ[where_resolid] = 1

    elif p.type_layer == 'equiaxed':

        pass
        
    elif p.type_layer == 'band':
            
        pass
        
    elif p.type_layer == 'ring':
    
        summ_c = 0
        summ_cet = 0
        while summ_c<columnar_all:           
            for j in range(nx):               
                if location[j]>-1: 
                    S_targ[j,location[j]] = 1
                    location[j] += 1
                    summ_c += 1
                    if location[j] > up_bound: location[j]=-1
        
                    
        while summ_cet<cet_all:           
            for j in range(nx):               
                if location[j]>-1: 
                    S_targ[j,location[j]] = 2
                    location[j] += 1
                    summ_cet += 1
                    if location[j] > up_bound: location[j]=-1           
    
    else: raise ValueError('structure type not specified') 
    
    
    assert new_solid_area == np.sum( (S_targ>0)*1 )
    return S_targ


class structure():
    def __init__(self, c_per, e_per, type_layer):
        self.c_per = c_per
        self.e_per = e_per
        self.type_layer = type_layer
        

class laser_parameter():
    '''
    a non-dimensional laser 
    '''
    
    def __init__(self, num_laser, num_pulse, x_span, t_span, x_source, t_source, end_time, ini_guess):
        
        
        self.num_laser = num_laser
        self.num_pulse = num_pulse
        self.x_span = x_span    ## nondimensionalized y phys.len_scale
        self.t_span = t_span
        self.x0 = x_source
        self.t0 = t_source  # an array
        self.t_arr = np.linspace(0, end_time, 100)
        self.ini_guess = ini_guess

    def laser_flux(self, x, t, coeff):
        
        '''
        x is the coordinates on the top surface, t is just one number
        '''
        coeff_p = coeff[:self.num_pulse]
        coeff_r = coeff[self.num_pulse:]
        
        x_s = phys.n1 * np.exp( -2*(    (x-self.x0)/self.x_span   )**2  )
        
        t_s = np.sum( self.ini_guess*coeff_p * np.exp( -2*(   (t-self.t0)/self.t_span/coeff_r  )**2  )  )

        return x_s*t_s
    
    def laser_power(self, t, coeff, idx):

        coeff_p = coeff[:self.num_pulse]
        coeff_r = coeff[self.num_pulse:]
        return phys.Q*np.sum( self.ini_guess[idx]*coeff_p[idx] * np.exp( -2*(   (t-self.t0[idx])/self.t_span[idx]/coeff_r[idx]  )**2  )  ) 
    
    def plot_laser_profile_in_time(self, coeff, ax):
        
        
        if ax == None: fig, ax = plt.subplots()      
        
        for i in range(self.num_pulse):
    
            Q_x0 = lambda t: self.laser_power(t, coeff, i)
            nodim_Q = np.array(list(map(Q_x0, self.t_arr)))     
            ax.plot(self.t_arr*phys.time_scale*1e3, nodim_Q, '--',label='pulse '+str(i+1))
        
        Q_x0 = lambda t: self.laser_power(t, coeff, np.arange(self.num_pulse))
        nodim_Q = np.array(list(map(Q_x0, self.t_arr)))
        
        ax.plot(self.t_arr*phys.time_scale*1e3, nodim_Q, label='actual power')
        ax.set_xlabel('time [ms]')
        ax.set_ylabel('laser power [W]')
        ax.legend()
        ax.set_title('laser power control (1 laser at the center)')

    

class heat_solver():
    
    def __init__(self, simu, phys):
        
        
        self.simu = simu
        self.phys = phys
        
        self.x = np.linspace(-simu.lx/2, simu.lx/2, num=simu.nx)   ## computation domain is much larger
        self.y = np.linspace(-simu.ly, 0, num=simu.ny) 
        self.nv = simu.ny * simu.nx #number of DOFs
        self.nv_int = (simu.ny-2)*(simu.nx-2) # number of interior DOFs          

        self.CFL = simu.dt/ simu.h**2
        self.inv_Ste = 1./phys.Ste
        self.matrices = export_mat(self.CFL, simu, self.nv)  
       

        
        ## target analysis region is the DNS part
        self.x_dns = np.linspace( -simu.lx_dns, 0, num=simu.nx_dns)   
        self.y_dns = np.linspace( -simu.ly_dns, 0, num=simu.ny_dns)
        self.nv_dns = simu.nx_dns * simu.ny_dns
        self.yy_dns, self.xx_dns = np.meshgrid(self.y_dns, self.x_dns)
        
        self.iter_max = simu.sol_max
        self.wall_time = 0
        
        self.sol_temp = np.zeros((simu.nx_dns, simu.ny_dns))
        self.sol_G = np.zeros((simu.nx_dns, simu.ny_dns))
        self.sol_R = np.zeros((simu.nx_dns, simu.ny_dns))
        self.Hunt_struct = np.zeros((simu.nx_dns, simu.ny_dns))

    def DNS_window(self,u_all):
        ## recap later
         
        return u_all[ self.simu.nx//2 - self.simu.nx_dns:self.simu.nx//2 , -1-self.simu.ny_dns:-1]

    def shape_back_2d(self, u):
        
        return np.reshape(u, (self.simu.nx, self.simu.ny), order='F')

    def extract_new_GR(self, u, unew, G, R):
        
        u    = self.shape_back_2d(u)
        unew = self.shape_back_2d(unew)
        
        dTdt = np.abs((unew - u)/self.simu.dt)
        
        
        gradT_x =  ( unew[2:,  1:-1] - unew[:-2, 1:-1]) / (2*self.simu.h)
        gradT_y =  ( unew[1:-1,  2:] - unew[1:-1, :-2]) / (2*self.simu.h)
        G_t = np.sqrt( gradT_x**2 + gradT_y**2 )
        R_t = dTdt[1:-1,1:-1]/G_t
        
        
        #a[np.where((a>3)&(b>3))] = b[np.where((a>3)&(b>3))]
        
        where_new_liquid = np.where( (u < 1) & ( unew > 1) )
        #where_new_solid  = (u > 1) & ( unew < 1)  
        where_new_solid_inter  = np.where( (u[1:-1,1:-1] > 1) & ( unew[1:-1,1:-1] < 1) )
        
        G[where_new_liquid] = 0.0
        R[where_new_liquid] = 0.0
        
    
        G[1:-1,1:-1][where_new_solid_inter] = G_t[where_new_solid_inter]
        R[1:-1,1:-1][where_new_solid_inter] = R_t[where_new_solid_inter]

        #print(G[1:-1,1:-1][where_new_solid_inter])
        
        return G, R 

    def examine_GR(self,G):
        
        xgrid = G.shape[0]
        ygrid = G.shape[1]
        
        assert xgrid == self.simu.nx_dns
        assert ygrid == self.simu.ny_dns
        
        succeeded = True
        
        if np.all(G < 0):
            succeeded = False
            print('never has the chance to resolid')
        
        if 0 in G:
            
            succeeded = False
            print('still has liquid there, solver stops too early')
    

        if G[int(0.1*xgrid),-1] >0 or G[-1,int(0.1*ygrid)] >0:
            succeeded = False
            print('melt pool too large')  
            
        if G[int(0.5*xgrid),-1] <0 or G[-1,int(0.7*ygrid)] <0:
            succeeded = False
            print('melt pool too small')           
            
        return succeeded
    
    def Hunt_model_Structure(self):

        '''
        Input:
            G (K/um), R (um/s) when the interface move upwards, the part always solid is -1 
        Output:
            S: the output morphology field based on Hunt model
        '''
        assemble_hunt_model = np.vectorize(Hunt_model)
        
        return assemble_hunt_model(self.sol_G, self.sol_R, self.phys)
    
    def plot_temperature(self):
        
        xd = len_um(self.xx_dns)
        yd = len_um(self.yy_dns)
        ud = temp_dim(self.sol_temp)

        fig, ax = plt.subplots()
        h1 = ax.pcolormesh( xd,yd, ud, cmap = 'hot')
        ax.contour(xd,yd,ud, [self.phys.Tl])
        ax.set_xlabel('x [um]')
        ax.set_ylabel('y [um]')
        fig.colorbar(h1)
    
    def plot_GR(self):
        
        xd = len_um(self.xx_dns)
        yd = len_um(self.yy_dns)


        fig, ax = plt.subplots(1,2,figsize=(16,6))
        h1 = ax[0].pcolormesh( xd,yd, self.sol_G, cmap = 'plasma')
      #  ax[0].contour(xd,yd,Gd, [0])
        ax[0].set_xlabel('x [um]')
        ax[0].set_ylabel('y [um]')
        ax[0].set_title('G [K/um]')
        fig.colorbar(h1, ax = ax[0])      
        
        h2 = ax[1].pcolormesh( xd,yd, 1e-6*self.sol_R, cmap = 'inferno')
      #  ax[1].contour(xd,yd,Rd, [0])
        ax[1].set_xlabel('x [um]')
        ax[1].set_ylabel('y [um]')
        ax[1].set_title('R [m/s]')
        fig.colorbar(h2, ax = ax[1])         
        
    def plot_hunt_structure(self, pl, m_targ):
        
        
        
        
        fig, ax = plt.subplots(1,2,figsize=(12.8,4.8))
        
        pl.plot_laser_profile_in_time( m_targ, ax[0])
        
        #discrete color scheme
        cMap = ListedColormap(['purple', 'blue', 'green', 'yellow'])
        xd = len_um(self.xx_dns)
        yd = len_um(self.yy_dns)

        
        heatmap = ax[1].pcolormesh( xd,yd, self.Hunt_struct, vmin = 0, vmax = 3, cmap = cMap)
        ax[1].set_xlabel('x [um]')
        ax[1].set_ylabel('y [um]')
        
        cbar = plt.colorbar(heatmap)
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['$B$','$C$','$T$','$E$']):
            cbar.ax.text(1.5, (6 * j + 3) / 8.0, lab, ha='center', va='center',color='black')
           # cbar.ax.get_yaxis().labelpad = 15
        ax[1].text( -50, -50,'B:base, C:columnar, T:transition, E:equiaxed',fontsize=8)
        ax[1].set_title('Grain structure based on Hunt model')
    
    
    def forward_solve(self, pl, m_targ):   ## pl is the input to the heat solver that changes the output u
        
        #Lap, Q, I, A0, M = export_mat(self.CFL)
        
        # initial condition

        
        Q = self.matrices[1]
        A0 = self.matrices[3]
        M = self.matrices[4]
        
        iteration = 0
        t=0      
        stop_early=False
     
        '''
        
        G, R start with -1, means never touched
        if u exceeds 1 (T=Tl), set G,R to zero
        if u gets lower than 1, calculate G,R accordingly
        
        '''
        
        
        u = self.phys.u0 * np.ones(self.nv)        
        G = - np.ones((self.simu.nx, self.simu.ny))
        R = - np.ones((self.simu.nx, self.simu.ny))
        
        start = time.time()
        while  (iteration < self.iter_max) :
            
            A_l = dphase_trans(u)
            
            A = A0 + self.inv_Ste*Q@A_l
             
            b = u + self.inv_Ste*A_l*u 
                       
            u_top = u[-self.simu.nx:]
                        
            qs = pl.laser_flux(self.x, t, m_targ)
            
        
            set_top_bc(b, qs, u_top, t, self.CFL, self.phys, self.simu)
            
            
            unew,stat,num_iter = sparse_cg(A, Q@b, u, self.simu.cg_tol, M, self.simu.maxit)
            
 
            window = self.DNS_window(self.shape_back_2d(unew))
            checkpoint_ux = window[-1,0]
            checkpoint_uy = window[0,-1]
            if checkpoint_ux >1 or checkpoint_uy >1: 
                
                print('at iteration', iteration,'temperature already too high', \
                      checkpoint_ux, checkpoint_uy)
                '''
                temperaure is too high, BC is gonna affect
                '''
                get_structure = False
                stop_early=True
                break       
                
            
            
            G, R = self.extract_new_GR(u, unew, G, R)   
            #print(self.DNS_window(self.shape_back_2d(unew)))
    
            u = unew
            
            t += self.simu.dt
            iteration += 1
           # print(iteration)
            
         #   if iteration%10 == 0:
                
         #       self.sol_temp = self.DNS_window(self.shape_back_2d(u))
         #       self.plot_temperature()

        end = time.time()
        self.wall_time = end -start
        print('\n solver time %1.2f s'%self.wall_time)
        
        
        if stop_early==True: return get_structure, self.Hunt_struct
        
        '''
        post processing
        '''
        
        
        #self.plot_temperature()
        
        G = self.DNS_window(G)
        R = self.DNS_window(R)
        
        get_structure = self.examine_GR(G)
            
        self.sol_temp = self.DNS_window(self.shape_back_2d(u))
        self.sol_G = heat_dim(G)/ ( self.phys.len_scale*1e6 )
        self.sol_R = len_um(R)/self.phys.time_scale 
        
        self.Hunt_struct = self.Hunt_model_Structure()
        #self.plot_hunt_structure(pl, m_targ)
        
        return get_structure, self.Hunt_struct
        
    

#==============================================================================
#
#     Outside the optimization loop
#     
#==============================================================================    

## settig the estimated scale for power, length and time

Q = 150
x_span = 50e-6
t_span = 0.5e-3


phys = phys_parameter( Q, x_span, t_span )
simu = simu_parameter( phys )

print('dimensionless domain lx x ly: {:.2f} x {:.2f}'.format(simu.lx, simu.ly))
print('grid size nx x ny: {:d} x {:d}'.format(simu.nx, simu.ny))
print('melt pool grid size nx x ny: {:d} x {:d}'.format(simu.nx_dns, simu.ny_dns))
print('mesh width h: {:1.2e}'.format( simu.h ))
print('time step: {:3d}'.format( simu.sol_max))
print('number of laser: {:3d}'.format(simu.num_laser))
print('laser pulse time:', (simu.start_pulse)*phys.time_scale*1000)

solver = heat_solver(simu, phys)
print('initial temperature', temp_dim(solver.phys.u0))
#==============================================================================
#
#     Optimization 
#     
#==============================================================================    


'''

Case 1: fixed: laser location (1), laser pulse time (5)
        parameter space: [0, up_limit]^5 for 5 laser pulse amplitude
'''


up_limit = 1.5
maxP = 0.8
minP = 0.05
P0 = np.linspace(maxP, minP, simu.num_pulse) ## initial guess for coefficient
#P0[1] = 0.01
#P0[2] = 0.01

t0 = np.ones(simu.num_pulse)*t_span
'''
P0 = np.array([maxP, maxP, minP, minP])*0.9

P0 = np.array([maxP, maxP, maxP, minP])*0.6

P0 = np.array([maxP, minP, minP, minP])*1.8

P0 = np.array([minP, maxP, minP, minP])*1.2

P0 = np.array([minP, maxP, maxP, maxP])*0.6


t0[3]*=0.3
'''
m0= np.ones(simu.num_pulse*2)
print('initial guess of parameters', m0)


pl = laser_parameter(simu.num_laser, simu.num_pulse, x_span/phys.len_scale, t0/phys.time_scale,\
                     0, simu.start_pulse, simu.t_end/phys.time_scale, P0)

#fig, ax = plt.subplots()    
#pl.plot_laser_profile_in_time( m_targ, ax, None)

#struct_class =  structure( 0.3, 0.3, 'ring')
    
#struct_class =  structure( 1, 0, 'columnar')

struct_class =  structure( 0, 1, 'equiaxed')

out = disctns_objective_function(m0, pl, solver, struct_class)

print('initial function value', out)

J_func = lambda m_targ: disctns_objective_function(m_targ, pl, solver, struct_class)




options = {'tolfun':1e-1,'maxiter': 1}

res = cma.fmin(J_func, m0, 0.02, options)

#res = minimize(J_func, m_targ, method='Nelder-Mead', tol=1e-2)
#es = cma.CMAEvolutionStrategy(m_targ, 0.05, options=options).optimize(J_func).result

#solver.forward_solve(pl, res[0])
#solver.plot_hunt_structure(pl, res[0])


def plot_hunt_structure(solver, S_targ):
        #discrete color scheme
        cMap = ListedColormap(['purple', 'blue', 'green', 'yellow'])
        #norm = BoundaryNorm([-1, -0.5, 0.5, 1], cMap.N)
        #lc = LineCollection(segments, cmap=cMap, norm=norm)

        xd = len_um(solver.xx_dns)
        yd = len_um(solver.yy_dns)

        fig, ax = plt.subplots(figsize = (5.6,4.8))
        heatmap = ax.pcolormesh( xd,yd, S_targ, vmin = 0, vmax = 3, cmap = cMap)
        ax.set_xlabel('x [um]')
        ax.set_ylabel('y [um]')
        
        cbar = plt.colorbar(heatmap)
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['$B$','$C$','$T$','$E$']):
            cbar.ax.text(1.5, (6 * j + 3) / 8.0, lab, ha='center', va='center',color='black')
           # cbar.ax.get_yaxis().labelpad = 15
        fig.text(0.2,0.2,'B:base, C:columnar, T:transition, E:equiaxed',fontsize=8)
        ax.set_title('targed grain structure')


#plot_hunt_structure(solver, target_Structure( solver.Hunt_struct, struct_class ))






