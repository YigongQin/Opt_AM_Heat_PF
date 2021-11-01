#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:48:39 2021

@author: yuanxun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:02:28 2020

@author: yigongqin, yuanxunbao
"""

from math import pi
import numpy as np

class phys_parameter:
    
    
    def __init__(self,arg1,arg2,arg3):
        
        self.Q = arg1      # laser power [W]    
        self.rb = arg2     # radius of heat source [m] 
        self.t_spot_on = arg3 # the span of the gaussian time [s]
        
    
        # Properties of Al-Cu
        self.k = 0.14
        Tm = 933.3
        c_infty = 3
        liq_slope = 2.6
        self.c_infm = c_infty*liq_slope
        
        self.GT = 0.24                       # GT coefficient Kum
        self.Dl = 3000                       # liquid diffusion coefficient      um**2/s
        self.d0 = 5.0e-3                       # capillary length -- associated with GT coefficient   um
        
        self.Teu = 821
        self.Ts = Tm - self.c_infm/self.k
        self.Tl = Tm - self.c_infm
        self.deltaT = self.Tl-self.Ts
        
        self.T0 = 25+273       # ambient temperature   
        self.DT_N = 0.75
        self.line_den = 0.1   ## /um
        
        K = 210                 # thermal conductivity [W/(m*K)]
        rho = 2768              # density  [kg/m^3]
        Cp = 900                # specific heat [J/ (kg*K)]
        self.kappa = K/(rho*Cp)      # thermal diffusivity [m^2/s]
    
        Lf =  3.95e5             # latent heat [J/kg] 
        
        self.Ste = Cp*(self.Tl-self.Ts)/Lf     # Stefan number
                
  
        A = 0.09   # abosorption coefficient
   

        
        # environment background 
        
        hc = 10           # convective heat transfer coefficient [W m^(-2) K^(-1)]
        epsilon = 0.13    # thermal radiation coeffcient
        sigma = 5.67e-8   # stefan-boltzmann constant [W m^(-2) K^(-4)]
        
        
        
        
        self.len_scale = self.Q*self.t_spot_on*self.kappa / (self.rb**2*K*self.deltaT)
        self.time_scale = self.len_scale**2 / self.kappa   
        
        
        
        self.n1 = 2*self.Q*A*self.len_scale / (pi*self.rb**2*K*self.deltaT) # dimensionless heat intensity
        self.n2 = hc * self.len_scale / K                                   # dimensionless heat convection coeff
        self.n3 = (self.Ts-self.T0)/ self.deltaT
        self.n4 = epsilon*sigma*self.len_scale*self.deltaT**3 / K   # dimensionless radiation coeff
        self.n5 = self.Ts/ self.deltaT
        self.n6 = self.T0/ self.deltaT
        

        
        
        self.Ste = Cp* self.deltaT / Lf
        
        self.u0 = (self.T0 - self.Ts) / self.deltaT
        
        
        self.param1 = self.n1
        self.param2 = self.t_spot_on / self.time_scale
  
        


class simu_parameter:  
      
    
    def __init__(self,p):
    
        lxd = 480e-6     # dimensional length [m]
        asp_ratio = 1/2  # height is the half of the size
        
        
        lxd_dns =  0.125*lxd # 13e-4
        lyd_dns =  lxd_dns # 12e-4
        
        
        self.lx = lxd / p.len_scale    # non-dimensional length

        self.nx = 256*4+1
        self.ny = int((self.nx-1)*asp_ratio+1)
        self.h = self.lx / (self.nx-1)
        self.ly = (self.ny-1)*self.h 
        
        
        
        self.lx_dns = lxd_dns / p.len_scale
        self.ly_dns = lyd_dns / p.len_scale        
        self.nx_dns = int( self.lx_dns / self.h) + 1
        self.ny_dns = int( self.ly_dns / self.h) + 1
        
        # actual dns ly
        #self.ly_dns = (self.ny_dns-1) * self.h
        

        #self.cg_tol = 1e-8
        self.cg_tol = 1e-6
        
        self.maxit = 80
        self.source_x = [0,0] # -self.lx/4
        self.near_top = self.h * 1.1
        
        
        
        self.num_laser = 1
        self.num_pulse = 4
        
        self.t_end = self.num_pulse * p.t_spot_on   ## has dimension s

        
        self.Mt_spot_on = 25
        
        self.dt = (p.t_spot_on/p.time_scale) / self.Mt_spot_on
        self.sol_max = int( ( (self.t_end)/p.time_scale)  / self.dt) + 1 
        
        self.start_pulse = np.linspace(0, self.t_end/p.time_scale, self.num_pulse,endpoint=False)

        
    

        
        self.direc = './'
        # filename = 'head2d_temp_nonlinear'+'_nx'+str(nx)+'_asp'+str(asp_ratio)+'_dt'+str('%4.2e'%dt)+'_Mt'+str(Mt)+'.mat'
        # outname = 'macro_output_low'+'_dt'+str('%4.2e'%dt)+'_dx'+str('%4.2e'%dx)+'_Tt'+str('%4.2e'%(dt*Mt))+'.csv'
        
        # outname = 'macro_output_highQ'+'_dt'+str('%4.2e'%dt)+'_dx'+str('%4.2e'%dx)+'_Tt'+str('%4.2e'%(dt*Mt))+'.mat'
        
        # nxs = int((nx-1)/2+1)
    
    

        
        
        
