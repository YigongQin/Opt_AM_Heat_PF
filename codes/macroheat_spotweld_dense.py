#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:04:23 2020

@author: yuanxun
"""

import importlib
import os, sys
import numpy as np
# from macro_input3 import phys_parameter, simu_parameter 
from macro_input_welding import phys_parameter, simu_parameter 
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


def sparse_cg(A, b, u0, TOL, P, maxit):

      num_iters = 0

      def callback(xk):
         nonlocal num_iters
         num_iters+=1

      x,status = spla.cg(A, b, x0=u0, tol=TOL, M=P, maxiter = maxit, callback=callback)
      return x,status,num_iters
  

def sparse_laplacian(nx,ny):
    
    # Neumann BC Laplacian
    Ix = sp.eye(nx,format='csc'); Iy = sp.eye(ny,format='csc')
    Ix[0,0] = 0.5; Ix[-1,-1] = 0.5
    Iy[0,0] = 0.5; Iy[-1,-1] = 0.5
    


    Dxx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)).toarray()
    Dxx[0,1]=2; Dxx[-1,-2]=2

    Dyy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)).toarray()
    Dyy[0,1]=2; Dyy[-1,-2]=2


    # L = sp.kronsum(Dyy,Dxx,format='csc')
    L = sp.kronsum(Dxx,Dyy,format='csc')

    Q = sp.kron(Ix,Iy,format='csc')
    # Q = sp.kron(Ix,Iy,format='csc')
    
    return L,Q


def convert_sl_equidist(X,Y, n):
    
    
    # arc lengh of initial curve    
    # starting point is the 90 degree point
    s= np.cumsum( ( np.sqrt( (X[1:] - X[:-1])**2 + (Y[1:] - Y[:-1])**2 ) ) )
    
    print('ARC LEN SPACING IS:', len_um( s[-1]/n) )
    
    st = np.linspace( s[0], s[-1], n)
    f = ( np.array( [ X[1:].T, Y[1:].T ] ) )
            
    f_itp = interp1d( s, f)
            
    Z = f_itp(st)

    return Z[0,:], Z[1,:]

    # return np.append( Z[0,:], X[-1] ),  np.append( Z[1,:], Y[-1] )

def gradient(psi, h):
    
    gx = ( psi[2:,  1:-1] - psi[:-2, 1:-1]) / (2*h)
    gy = ( psi[1:-1,  2:] - psi[1:-1, :-2]) / (2*h)
    
    return gx**2 + gy**2
    
    

def gradients2(y, dTdt, h):
    # yn is new result, yp is previous time step
    Tn = np.reshape(y, (nx,ny), order='F')
    dTdt2d = np.reshape(dTdt, (nx,ny), order='F')
    
    #gradT_n = np.ones((ny,nx))11
    # gradT_x = np.ones((ny-1,nx-1)); gradT_y = np.ones((ny-1,nx-1)); 
    gradT_x =  ( Tn[2:,  1:-1] - Tn[:-2, 1:-1]) / (2*h)
    gradT_y =  ( Tn[1:-1,  2:] - Tn[1:-1, :-2]) / (2*h)
    gradT_n = np.sqrt( gradT_x**2 + gradT_y**2 )
    
    # set boundary condition for G
    
    #gradT_n[0,:] = 
    #gradT_n[-1,:] = gradT_n[-1,:]
    
    
    #grady_n = np.reshape(gradT_n,(nv,1),order='F') 
    
    #dTdt = ( y_new - y_old )/dt
    
    return gradT_n, -dTdt2d[1:-1,1:-1]/gradT_n, gradT_x, gradT_y


def convert_dim(x,y,u):
    
    
    xd = x * phys.len_scale * 1e6   # [um]
    yd = y * phys.len_scale * 1e6   # [um]
    Temp = u*(phys.Tl-phys.Ts)+phys.Ts   # [K]
    
    return xd,yd,Temp


def len_dim(x): return x *  phys.len_scale      # [m]

def len_um(x): return x *  phys.len_scale * 1e6    # [um]

def temp_dim(u): return u * phys.deltaT + phys.Ts  # [K]

def heat_dim(u): return u * phys.deltaT  # [K]

def vel_dim(v): return v * (phys.len_scale * 1e6) / phys.time_scale  # [um/s]

def time_dim(t): return t * phys.time_scale    #[s]

def gradT_dim(g): return heat_dim(g)/ ( phys.len_scale ) # [K/m]

def gradT_um(g): return heat_dim(g)/ ( phys.len_scale * 1e6 ) # [K/um]

def dT_dim(dT): return heat_dim(dT)/ phys.time_scale




def heat_source( x,t, x0 ):
    
    return phys.n1 * np.exp( -2*(x-x0)**2 / (phys.rb/phys.len_scale)**2 )

    # return p.q0 * np.exp( -2*(x-l0)**2 / p.rb**2 ) * np.exp( -2*(p.Vs*t)**2 / p.rb**2 )



def dphase_trans(u):
    
    mask = (u > 0 ) & (u < 1)
    
    T_lat = pi/2. * np.sin(pi * u) * mask;
    
    return sp.diags( T_lat, format = 'csc')


def set_top_bc(rhs, qs, u_top, t):
    
    # qs_t = heat_source_t(t
    
    # bU = - 2*dx/p.K*( -qs + p.hc*(T_top-p.Te) + p.epsilon*p.sigma*(T_top**4-p.Te**4) ) 
    
    bU = -2*simu.h * ( -qs + phys.n2*(u_top + phys.n3) + phys.n4*((u_top + phys.n5)**4 - phys.n6**4  ) )
    
    # rhs[0:-1:simu.ny] =  rhs[0:-1:simu.ny] + bU * CFL 
    rhs[-simu.nx:] = rhs[-simu.nx:] + bU * CFL 
    
    # return qs, bU
    
    
    
def get_initial_contour_T(u_interp, G_interp, R_interp, center , theta ):
    
    x0=center[0]
    y0=center[1]
    
    
    # print(x0,y0)
    # u2d = np.reshape(u, (ny,nx), order='F')
    
    # u_interp = itp2d(x,y, u2d)
    
    # R_interp = itp2d(x_i, y_i, R)
    # G_interp = itp2d(x_i, y_i, G)
    
    
    X = np.zeros(theta.size)
    Y = np.zeros(theta.size)
    Gt= np.zeros(theta.size)
    Rt= np.zeros(theta.size)
    Tt= np.zeros(theta.size)
    
    
    for j in range(theta.size):
        
        
        f = lambda s : u_interp( x0 + s*np.cos(theta[j]) , y0 + s*np.sin(theta[j]) ) - 1.0
        
        sj = brentq(f, y.min() ,0)
        
        # print(sj)
        
        X[j] = x0 + sj *np.cos(theta[j])
        Y[j] = y0 + sj *np.sin(theta[j])
        
        Gt[j] = G_interp(X[j], Y[j])
        Rt[j] = R_interp(X[j], Y[j])
        Tt[j] = u_interp(X[j], Y[j])
        
    
    return X, Y, Gt, Rt, Tt


def get_initial_contour_T2(u_interp, G_interp, R_interp, X,Y):
    
    Gt= np.zeros(X.size)
    Rt= np.zeros(X.size)
    Tt= np.zeros(X.size)
    
    for j in range(X.size):
        
        Gt[j] = G_interp(X[j], Y[j])
        Rt[j] = R_interp(X[j], Y[j])
        Tt[j] = u_interp(X[j], Y[j])
        
        
    return Gt,Rt,Tt


def get_sl_interface(T2d, x, y, x_top, Tl):
    
    # T2d is 2d temperature
    # x : 1d array, e.g. x_dns
    # y : 1d array, e.g. y_dns
    # x_top: x-coordinate on the top boundary
    # Tl the temperature of your target interface
    # The solver loops over x_top, shoots a ray from x_top[j], and find the intersection with T=Tl
    # if no intersection is found, fill with nan.
    
    
    T_interp = itp2d(x, y, T2d.T)
    
    
    X = np.zeros(x_top.size)
    Y = np.zeros(x_top.size)
    
    for j in range(x_top.size):
        
        f = lambda s : T_interp(x_top[j], s) - Tl
        
        try:
        
            X[j] = x_top[j]
            Y[j] = brentq(f, y.min() ,0)
        except ValueError:
            
            X[j] = np.nan
            Y[j] = np.nan
        
    return X,Y
        
        
        
        
    
    


def track_sl_interface(x0,y0,gx,gy,unew_interp, Gnew_interp, Rnew_interp):
    
    vx = gx(x0,y0)
    vy = gy(x0,y0)
    
    vn = np.sqrt(vx**2+vy**2)
    vx = vx/vn
    vy = vy/vn

    f = lambda s: u_interp( x0+s*vx, y0+s*vy ) - 1
    
    s0 = fsolve(f, [0.0])

    # print(s0)

    x1 = x0 + s0*vx
    y1 = y0 + s0*vy
    
    return x1,y1, Gnew_interp(x1,y1), Rnew_interp(x1,y1)
    
    
    
def comp_pool_depth(u_interp, center):
    
    f = lambda s : u_interp(  center[0] , center[1] + s*np.sin(np.pi/2) ) - 1
                            
    pd = brentq(f, y.min() ,0)
    
    # x_bottom[0] = center[0]
    # x_bottom[1] = center[1] + sj * np.sin(np.pi/2)    
    
    # print( u_interp(x_bottom[0], x_bottom[1]) )
    
    
    return np.abs(pd) 


def compute_sl_normal(X, Y, gx, gy):
    
            
    npts = X.size
    vx = np.zeros(npts)
    vy = np.zeros(npts)
    
    # print(npts)
    
    for j in range(npts):
        
        
        # print(j)
    
        vx[j] = gx(X[j],Y[j])
        vy[j] = gy(X[j],Y[j])
        
    
    vn = np.sqrt(vx**2 + vy**2)
    
    # print(vn)
    
    alpha = np.arctan2(vx,vy)
    
    return vx/vn, vy/vn, alpha
       

    
def interp_unstructed2grid(xg,yg, alpha, X,Y):
    
    pts= np.array( ( X.flatten(), Y.flatten() ) ).T
    
    val= alpha.flatten()
    
    alpha_z0 = griddata( pts, val, ( xg.flatten() , yg.flatten() ), method = 'linear')
    
    alpha_z1 = griddata( pts, val, ( xg.flatten() , yg.flatten() ), method = 'nearest')
    
    
    # fill nan with nearest
    alpha_z0[ np.isnan(alpha_z0) ] = alpha_z1[ np.isnan(alpha_z0) ]
    
    return alpha_z0
    
    
    
def psi_dns_initial( gradTx, gradTy, x,y, X,Y, max_len ):
    
    
    yy, xx = np.meshgrid(y,x)
    
    # compute normal of s-l interface: X,Y
    nhat_x = np.zeros(X.size)
    nhat_y = np.zeros(Y.size)
    
    gx_interp = itp2d(x, y, gradTx.T, kind='linear')
    gy_interp = itp2d(x, y, gradTy.T, kind='linear')

    for j in range(X.size):

        vx = gx_interp(X[j],Y[j])
        vy = gy_interp(X[j],Y[j])
        vn = np.sqrt(vx**2+vy**2)
        
        nhat_x[j] = vx / vn
        nhat_y[j] = vy / vn
        
        
    # LEFT EXTENSTION
    ds = np.sqrt( (X[0]-X[1])**2 + (Y[0]-Y[1])**2 )
    u = (X[0]-X[1]) / ds
    v = (Y[0]-Y[1]) / ds
    
    npts_ext = int(  (X[0] - x[0])/ ds )
    print('number of extension points = %d'%(npts_ext))
    
    x_ext = np.zeros(npts_ext)
    y_ext = np.zeros(npts_ext)
    for k in range(npts_ext):
        
        x_ext[k] = X[0] + u * ds * k
        y_ext[k] = Y[0] + v * ds * k    
    
    # compute normal of extended interface
    nhat_x_ext = y_ext[1:] - y_ext[:-1]
    nhat_y_ext = -x_ext[1:] + x_ext[:-1]
    n2_ext = np.sqrt( nhat_x_ext**2 + nhat_y_ext**2 )
    nhat_x_ext = nhat_x_ext / n2_ext
    nhat_y_ext = nhat_y_ext / n2_ext

                
    # append extended interface with original 
    X = np.append(x_ext[1:], X)
    Y = np.append(y_ext[1:], Y)
    nhat_x = np.append(nhat_x_ext, nhat_x )
    nhat_y = np.append(nhat_y_ext, nhat_y)
    
    
    ## RIGHT EXTENSION
    ds = np.sqrt( (X[-1]-X[-2])**2 + (Y[-1]-Y[-2])**2 )
    u = (X[-1]-X[-2]) / ds
    v = (Y[-1]-Y[-2]) / ds
    
    
    x_ext = np.zeros(npts_ext)
    y_ext = np.zeros(npts_ext)
    for k in range(npts_ext):
        
        x_ext[k] = X[-1] + u * ds * k
        y_ext[k] = Y[-1] + v * ds * k    
    
    # compute normal of extended interface
    nhat_x_ext = -y_ext[1:] + y_ext[:-1]
    nhat_y_ext =  x_ext[1:] - x_ext[:-1]
    n2_ext = np.sqrt( nhat_x_ext**2 + nhat_y_ext**2 )
    nhat_x_ext = nhat_x_ext / n2_ext
    nhat_y_ext = nhat_y_ext / n2_ext

                
    # append extended interface with original 
    X = np.append(X, x_ext[1:])
    Y = np.append(Y, y_ext[1:])
    nhat_x = np.append( nhat_x , nhat_x_ext)
    nhat_y = np.append( nhat_y , nhat_y_ext)
    
    
    
    # characteristics step size
    ds =  np.sqrt( ( x[1]-x[0] )**2 + ( y[1] - y[0] )**2)
    nstep = int(max_len / ds)
    
    
    X_char = np.zeros( (X.size, 2*nstep-1))
    Y_char = np.zeros( (X.size, 2*nstep-1))
    psi_char = np.zeros( (X.size, 2*nstep-1))
    
    for kk in range(nstep):
        
        
        if kk == 0:
            
            X_char[:,0] = X
            Y_char[:,0] = Y
            psi_char[:,0] = 0
            
        else:
        
            # follow characteristic in the dirction of the interior
            X_char[:,kk] = X_char[:,kk-1] + nhat_x * ds
            Y_char[:,kk] = Y_char[:,kk-1] + nhat_y * ds      
            psi_char[:,kk] = psi_char[:,kk-1] - ds
    
            
            # follow characteristic in the direction of the exterior
            X_char[:,-kk] = X_char[:,-kk+1] - nhat_x * ds
            Y_char[:,-kk] = Y_char[:,-kk+1] - nhat_y * ds      
            psi_char[:,-kk] = psi_char[:,-kk+1] + ds
    
    
    # pts= np.array( ( X_char.flatten(), Y_char.flatten() ) ).T
        
    # val= psi_char.flatten()
        
    
    # psi0 = griddata( pts, val, ( xx, yy ), method = 'cubic')
    # psi1 = griddata( pts, val, ( xx, yy ), method = 'nearest')
    
    # psi0[np.isnan(psi0)] = psi1[np.isnan(psi0)]
        
    # fig,ax=plt.subplots()
    # ax.pcolormesh( xx, yy , psi0)
    
    
    
    return X_char, Y_char, psi_char




def psi_dns_initial_no_ext( gradTx, gradTy, x,y, X,Y, max_len ):
    
    
    yy, xx = np.meshgrid(y,x)
    
    # compute normal of s-l interface: X,Y
    nhat_x = np.zeros(X.size)
    nhat_y = np.zeros(Y.size)
    
    gx_interp = itp2d(x, y, gradTx.T, kind='linear')
    gy_interp = itp2d(x, y, gradTy.T, kind='linear')

    for j in range(X.size):

        vx = gx_interp(X[j],Y[j])
        vy = gy_interp(X[j],Y[j])
        vn = np.sqrt(vx**2+vy**2)
        
        nhat_x[j] = vx / vn
        nhat_y[j] = vy / vn
        
        
    
    ds =  np.sqrt( ( x[1]-x[0] )**2 + ( y[1] - y[0] )**2)
    nstep = int(max_len / ds)
    
    
    X_char = np.zeros( (X.size, 2*nstep-1))
    Y_char = np.zeros( (X.size, 2*nstep-1))
    psi_char = np.zeros( (X.size, 2*nstep-1))
    
    for kk in range(nstep):
        
        
        if kk == 0:
            
            X_char[:,0] = X
            Y_char[:,0] = Y
            psi_char[:,0] = 0
            
        else:
        
            # follow characteristic in the dirction of the interior
            X_char[:,kk] = X_char[:,kk-1] + nhat_x * ds
            Y_char[:,kk] = Y_char[:,kk-1] + nhat_y * ds      
            psi_char[:,kk] = psi_char[:,kk-1] - ds
    
            
            # follow characteristic in the direction of the exterior
            X_char[:,-kk] = X_char[:,-kk+1] - nhat_x * ds
            Y_char[:,-kk] = Y_char[:,-kk+1] - nhat_y * ds      
            psi_char[:,-kk] = psi_char[:,-kk+1] + ds
    
    
    # pts= np.array( ( X_char.flatten(), Y_char.flatten() ) ).T
        
    # val= psi_char.flatten()
        
    
    # psi0 = griddata( pts, val, ( xx, yy ), method = 'cubic')
    # psi1 = griddata( pts, val, ( xx, yy ), method = 'nearest')
    
    # psi0[np.isnan(psi0)] = psi1[np.isnan(psi0)]
        
    # fig,ax=plt.subplots()
    # ax.pcolormesh( xx, yy , psi0)
    
    
    
    return X_char, Y_char, psi_char
    
    
    
def export_mat(CFL):
    
    Lap,Q = sparse_laplacian(simu.nx, simu.ny)
    I = sp.eye(simu.nx * simu.ny,format='csc')
    
    A0 = Q @ ( I - CFL*Lap)

    # preconditioner
    M2 = spla.splu(A0)
    M_x = lambda x: M2.solve(x)
    M = spla.LinearOperator((nv,nv), M_x)

    return Lap, Q, I, A0, M


def comp_dist2top_v(X,Y,gx,gy):

    d2t = np.zeros((X.size))
    
    for j in range(X.size):
        
        x0 = X[j]
        y0 = Y[j]
    
        s = - y0 / gy(x0,y0)
        x1 = x0 + s * gx(x0,y0)
    
        d2t[j] = np.sqrt(  (x1-x0)**2 + (0-y0)**2  )  
        
    return d2t



def comp_dist2top_p(x0,y0,gx,gy):

    s = - y0 / gy(x0,y0)
    x1 = x0 + s * gx(x0,y0)
    
    return np.sqrt(  (x1-x0)**2 + (0-y0)**2  )  
        


if len(sys.argv) == 4:
    
    phys = phys_parameter( float(sys.argv[1]),\
                           float(sys.argv[2]),\
                           float(sys.argv[3]))
 
        
    simu = simu_parameter(phys)
    
    h  = simu.h
    nx = simu.nx
    ny = simu.ny
    dt = simu.dt
    dt_off = simu.dt_off
    
else:  
    
    # Q, rb, t_spot
    
    
    Q = 2000 # 2000 # 200
    rb = 1000e-6# 50e-6
    ts = 800e-3
    tm = ts * 2
        
    
    '''
    Q  = 200
    rb = 75e-6
    ts = 0.5e-3
    tm = ts * 4
    '''
    
    phys = phys_parameter( Q, rb, ts, tm)
    simu = simu_parameter( phys )
    
    h  = simu.h
    nx = simu.nx
    ny = simu.ny
    dt = simu.dt
    dt_off = simu.dt_off
    
    




#=============================================================================
#
#   COMPUTATIONAL GRID    
#
#=============================================================================
    
print('dimensionless domain lx x ly: {:.2f} x {:.2f}'.format(simu.lx, simu.ly))
print('grid size nx x ny: {:d} x {:d}'.format(simu.nx, simu.ny))
print('mesh width h: {:.4f}'.format( simu.h ))

nv = simu.ny * simu.nx #number of DOFs
nv_int = (simu.ny-2)*(simu.nx-2) # number of interior DOFs

x = np.linspace(-simu.lx/2,simu.lx/2, simu.nx)
y = np.linspace(-simu.ly,0,simu.ny)

yy,xx = np.meshgrid(y,x)


# centered grid
x_c = x[1:] - simu.h/2
y_c = y[1:] - simu.h/2

# interior point
x_i = x[1:-1]
y_i = y[1:-1]



'''
# grid for dns
lw = 2e-3
n_dns = 200
x_dns = np.linspace(lx/2 - lw, lx/2 + lw, num=2*n_dns+1, endpoint =True )
y_dns = np.linspace(0,-lw, num=n_dns+1, endpoint = True)
nv_dns = x_dns.size * y_dns.size
xx_dns, yy_dns = np.meshgrid(x_dns,y_dns)
'''


x_dns = np.linspace( -simu.lx_dns, 0, num=simu.nx_dns, endpoint = True)
y_dns = np.linspace( -simu.ly_dns, 0, num=simu.ny_dns, endpoint = True)
nv_dns = x_dns.size * y_dns.size
yy_dns, xx_dns = np.meshgrid(y_dns,x_dns)



#=============================================================================
#
#   PRECOMPUTE MATRICES
#
#=============================================================================

CFL = simu.dt/ simu.h**2  #CFL number
inv_Ste = 1./phys.Ste
Lap, Q, I, A0, M = export_mat(CFL)



# initial condition
u = phys.u0 * np.ones(nv)


# This loop start from t=0 and stop when the deepest S-L interface is reached


iter_melt = 0
t=0
pool_depth = y.max()

nplot=1
dTdt = 0

start = time.time()


#==============================================================================
#
#     MELTING, SPOT ON
#     
#==============================================================================

# while (reach_bottom == False and iter_melt < Mt):
while iter_melt < simu.Mt_spot_on :
    
    A_l = dphase_trans(u)
    
    A = A0 + inv_Ste*Q@A_l
    
    # obtain right hand side
    # b = u + lat*A_l*u  
    b = u + inv_Ste*A_l*u 
    
    
    # u_top = u[0:-1:simu.ny]
    u_top = u[-simu.nx:]
    
        
    qs = heat_source(x,t*0, simu.source_x[0])
    

    set_top_bc(b, qs, u_top, t)
    
    
    unew,stat,num_iter = sparse_cg(A, Q@b, u, simu.cg_tol, M, simu.maxit)
    
    dTdt_new = np.abs((unew - u)/simu.dt) 
    
    u2d = np.reshape(unew, (simu.nx, simu.ny), order='F')
    u_interp = itp2d(x,y,u2d.T)
    
    
    
    try:
        
        pd = comp_pool_depth(u_interp, simu.source_x)

        print( 'cur depth', len_dim(pd)  )
            
    except ValueError:
        
        print('Temperature is below T_liquid')
    
    
    
    u = unew
    dTdt = dTdt_new
    t += simu.dt
    iter_melt += 1


    
cur_depth = pd


# sys.exit()


#==============================================================================
#
#     SPOT OFF
#     
#==============================================================================
    

n_sl = 128
theta_sl = np.linspace(0,90, n_sl, endpoint = True) * pi/180
# theta = np.arange(5,91,2.5) * pi/180
ntraj = 20


X_arr = np.zeros((ntraj, simu.sol_max))
Y_arr = np.zeros((ntraj, simu.sol_max))
G_arr = np.zeros((ntraj, simu.sol_max))
R_arr = np.zeros((ntraj,  simu.sol_max))
dist_to_sl = np.zeros((ntraj, simu.sol_max))  # distance from s-l interface (arc length)
d2t_arr = np.zeros((ntraj,  simu.sol_max))   # approxiamte distance to top boundary

time_traj=np.arange(simu.sol_max)*dt


X_sl = np.zeros((n_sl, simu.sol_max))
Y_sl = np.zeros((n_sl, simu.sol_max))

ndir_x = np.zeros((n_sl, simu.sol_max))
ndir_y = np.zeros((n_sl, simu.sol_max))


# Allocate space for DNS data
T_dns = np.zeros((x_dns.size, y_dns.size, simu.sol_max))
alpha_dns = np.zeros( ( theta_sl.size, simu.sol_max))



CFL = simu.dt_off/ simu.h**2  #CFL number
Lap, Q, I, A0, M = export_mat(CFL)

all_reach_top = False
reach_bottom = False
iter_sol = 0
t=0
iter_max_depth=0
reach_top = np.zeros((ntraj, simu.sol_max),dtype = bool)


G, R, grad_x, grad_y = gradients2(u, dTdt, h)
u_interp = itp2d(x,y,u2d.T)            
R_interp  = itp2d(x_i, y_i, R.T)
G_interp  = itp2d(x_i, y_i, G.T)
gx_interp = itp2d(x_i, y_i, grad_x.T)
gy_interp = itp2d(x_i, y_i, grad_y.T)



while all_reach_top == False & iter_sol < simu.sol_max : 
    
# while iter_sol < 80:
    
    
    A_l = dphase_trans(u)
    
    A = A0 + inv_Ste*Q@A_l
    
    # obtain right hand side
    # b = u + lat*A_l*u  
    b = u + inv_Ste*A_l*u 
    
    
    # u_top = u[0:-1:simu.ny]
    u_top = u[-simu.nx:]
    
        
    qs = heat_source(x,t*0, simu.source_x[0]) * (1 - t/(phys.t_taper/phys.time_scale) )  
    
    print('t/tm', t/(phys.t_taper/phys.time_scale) )
    

    set_top_bc(b, qs, u_top, t)
    
    
    unew,stat,num_iter = sparse_cg(A, Q@b, u, simu.cg_tol, M, simu.maxit)
    
    dTdt_new = (unew - u)/simu.dt_off 
    
    
    G, R, grad_x, grad_y = gradients2(unew, dTdt_new, h)
    
    u2d = np.reshape(unew, (simu.nx, simu.ny), order='F')
    
    
    unew_interp = itp2d(x,y,u2d.T)
    Rnew_interp  = itp2d(x_i, y_i, R.T)
    Gnew_interp  = itp2d(x_i, y_i, G.T)
    gxnew_interp = itp2d(x_i, y_i, grad_x.T)
    gynew_interp = itp2d(x_i, y_i, grad_y.T)


    try:
        
        
        
        pd = comp_pool_depth(unew_interp, simu.source_x)

        if pd > cur_depth :
            
            print( 'still melting, cur depth is %.6f, iter = %d\n'%(len_dim(pd), iter_sol)  )
        
        elif reach_bottom:
            
            
            print( 'solidificaition, cur depth is %.6f, iter= %d\n'%(len_dim(pd), iter_sol)  )
            
            
            kk = iter_sol-iter_max_depth
            
            print( 'kk = ', kk)
            
            
            X_sl[:,kk],Y_sl[:,kk], gg, rr,Tt =\
                get_initial_contour_T(unew_interp, Gnew_interp, Rnew_interp, simu.source_x, theta_sl ) 
                
                
            T_dns[:,:,kk] = unew_interp( x_dns, y_dns).T
            
            
            ndir_x[:,kk], ndir_y[:,kk], alpha_dns[:,kk] =  compute_sl_normal(X_sl[:,kk], Y_sl[:,kk], gxnew_interp, gynew_interp)
            # ndir_x[:,kk], ndir_y[:,kk] =  compute_sl_normal(X_sl[:,kk], Y_sl[:,kk], grad_x, grad_y, xx[1:-1,1:-1], yy[1:-1,1:-1])
            # alpha_dns[:,kk] = np.arctan2( ndir_x[:,kk], ndir_y[:,kk] )
            
            
            
            for jj in range(ntraj):
        
                if reach_top[jj,iter_sol] == False :  
            
                
                    X_arr[jj,kk], Y_arr[jj,kk], G_arr[jj,kk], R_arr[jj,kk]  = \
                            track_sl_interface(X_arr[jj,kk-1], Y_arr[jj,kk-1], gx_interp, gy_interp, unew_interp, Gnew_interp, Rnew_interp)
                        
                    # d2c_arr[jj,iter_sol+1] = np.sqrt( (X_arr[jj,iter_sol+1]-center[0])**2 + (Y_arr[jj,iter_sol+1]-center[1])**2 )
                    dist_to_sl[jj,kk] = dist_to_sl[jj,kk-1] + np.sqrt(  (X_arr[jj,kk]-X_arr[jj,kk-1])**2 + (Y_arr[jj,kk]-Y_arr[jj,kk-1])**2  )
                    
                    
                    
                    
                    
                    d2t_arr[jj, kk] = comp_dist2top_p(X_arr[jj,kk], Y_arr[jj,kk], gxnew_interp, gynew_interp)
                
                    if d2t_arr[jj,kk] > simu.near_top : 
                    # if Y_arr[jj,iter_sol+1]+dx  < y.max() : 
                         reach_top[jj,kk] = False
                    else :
                         reach_top[jj,kk:] = True
                        
                         X_arr[jj,kk+1:] = X_arr[jj,kk]
                         Y_arr[jj,kk+1:] = Y_arr[jj,kk]
                         d2t_arr[jj,kk+1:] = d2t_arr[jj,kk]
                         dist_to_sl[jj,kk+1:] = dist_to_sl[jj,kk]
                        
                        
                else:
                    
                    T_min = unew_interp(X_arr[jj,iter_sol+1], X_arr[jj,iter_sol+1])
                    #T_max = unew_interp(Xtop_arr[jj], Ytop_arr[jj])
                    
                    # print(T_min)
                    # print(T_max)
                    
                    
                    
            if len(np.flatnonzero( ~reach_top[:,kk])) == 0: 
                
                
                all_reach_top = True
                iter_reach_top = kk
            
            
            
            
            
            
        else:
            
            max_depth = cur_depth
            iter_max_depth = iter_sol
            
            reach_bottom = True
            print( 'bottom is reached, cur depth is %.6f, iter = %d\n'%(len_dim(pd), iter_sol) )
            
            
            
        
            # initial normal
 #            X_arr[:,0],Y_arr[:,0], G_arr[:,0], R_arr[:,0],Tt =\
 #               get_initial_contour_T(unew_interp, Gnew_interp, Rnew_interp, simu.source_x, theta ) 
                

            # initial interface    
            X_sl[:,0],Y_sl[:,0], gg, rr,  Tt =\
                get_initial_contour_T(unew_interp, Gnew_interp, Rnew_interp, simu.source_x, theta_sl ) 
                
                
            X_arr[:,0], Y_arr[:,0] = convert_sl_equidist( np.flip(X_sl[:,0]) , np.flip( Y_sl[:,0]) , ntraj )
            
            
            
            G_arr[:,0], R_arr[:,0], Tt = get_initial_contour_T2(u_interp, G_interp, R_interp, X_arr[:,0], Y_arr[:,0])
            
            X0_nx, Y0_ny, theta =  compute_sl_normal(X_arr[:,0], Y_arr[:,0], gxnew_interp, gynew_interp)
            theta = np.arctan2(Y0_ny, X0_nx)
            
            
            '''
            tau_x = -X_sl[1:,0] + X_sl[:-1,0]
            tau_y = Y_sl[1:,0] - Y_sl[:-1,0]
            tau_norm = np.sqrt( tau_x**2 + tau_y**2 )
            tau_x = tau_x / tau_norm
            tau_y = tau_y / tau_norm
            '''    
            
                
            ndir_x[:,0], ndir_y[:,0], alpha_dns[:,0] =  compute_sl_normal(X_sl[:,0], Y_sl[:,0], gxnew_interp, gynew_interp)
            
            
            
            
            # alpha_dns[:,0] = np.arctan2( ndir_x[:,0], ndir_y[:,0])
                
            
            T_dns[:,:,0] = unew_interp( x_dns, y_dns).T
            

            
            
            gradTx_ini = gxnew_interp(x_dns, y_dns).T
            gradTy_ini = gynew_interp(x_dns, y_dns).T
            
                
                
            # predicted distance to top 
            d2t_arr[:,0] = comp_dist2top_v(X_arr[:,0], Y_arr[:,0], gxnew_interp, gynew_interp)
                
            
                
            
    except ValueError:
        
        
        print('=============== jj = %d',jj)
        print('An exception is raised, iter %d, pd = %.6f'%(iter_sol, len_dim(pd)))
        
        
        
    
    
    
    cur_depth = pd
    dTdt = dTdt_new
    u = unew
    t += simu.dt_off
    iter_sol += 1
    
    
    u_interp = unew_interp
    G_interp = Gnew_interp
    R_interp = Rnew_interp
    gx_interp = gxnew_interp
    gy_interp = gynew_interp
    

        
        

        
# once finished, resize time
        
        

u2dnew = temp_dim(u2d)
Rnew = vel_dim(R)
Gnew = gradT_dim(G)
dTnew = dT_dim( np.reshape(dTdt, (nx,ny), order = 'F') )


Rnew_arr = vel_dim( R_arr )
Gnew_arr = gradT_dim(G_arr)


X_arr = ( X_arr[:, :iter_reach_top] )
Y_arr = ( Y_arr[:, :iter_reach_top] )
X_sl = ( X_sl[:, :iter_reach_top] )  
Y_sl = ( Y_sl[:, :iter_reach_top] )


G_arr = G_arr[:, :iter_reach_top] 
R_arr = R_arr[:, :iter_reach_top] 
dist_to_sl = dist_to_sl[:, :iter_reach_top] 
d2t_arr = d2t_arr[:, :iter_reach_top]    
reach_top = reach_top[:, :iter_reach_top]     


t_macro = time_traj[:iter_reach_top] 

#

X0 = len_um( X_sl[:,0] )
Y0 = len_um( Y_sl[:,0] )


X_char, Y_char, psi_char = psi_dns_initial( gradTx_ini , gradTy_ini, len_um(x_dns), len_um(y_dns),  X0,Y0 , np.abs( len_um( x_dns[0])  ) )

pts= np.array( ( X_char.flatten(), Y_char.flatten() ) ).T
val= psi_char.flatten()
psi0 = griddata( pts, val, ( len_um( xx_dns) , len_um(yy_dns) ), method = 'cubic')
psi1 = griddata( pts, val, ( len_um( xx_dns) , len_um(yy_dns) ), method = 'nearest')
psi0[np.isnan(psi0)] = psi1[np.isnan(psi0)]
fig,ax=plt.subplots()
# im= ax.pcolormesh( xx_dns, yy_dns , psi0)
# plt.colorbar(im, ax = ax)

gpsi = 1- gradient(psi0, len_um( x_dns[1]-x_dns[0] ) )
im = ax.pcolormesh( len_um(xx_dns[1:-1,1:-1]), len_um(yy_dns[1:-1,1:-1]), np.abs(gpsi) )
plt.colorbar(im,ax=ax)






#### POST PROCESSING TO GET DATA
T_dns = T_dns[:,:,:iter_reach_top]

alpha_dns = alpha_dns[:,:iter_reach_top]
alpha_dns = interp_unstructed2grid( xx_dns, yy_dns, alpha_dns, X_sl, Y_sl )
alpha_dns = np.reshape(alpha_dns, (simu.nx_dns, simu.ny_dns))




    
fig2,ax2 = plt.subplots(2,1)
nskip = 5
ax2[0].plot( len_um( X_sl[:,::nskip]), len_um( Y_sl[:,::nskip]), color = 'r')
ax2[0].plot( len_um( X_arr.T), len_um(Y_arr.T), color='b')
ax2[0].axis('equal')
ax2[0].set_xlabel(r'$x \,  [\mu m]$')
ax2[0].set_ylabel(r'$y \, [\mu m]$')
ax2[0].set_title('$Q=%dW, r_b=%.2f \mu m, t_s=%.2fms, t_m=%.2fms$'%(phys.Q, phys.rb*1e6, phys.t_spot_on * 1e3, phys.t_taper * 1e3))



ax2[1].plot(  X_sl[:,0:-1:nskip],  Y_sl[:,0:-1:nskip], color = 'r')
ax2[1].plot(  X_arr.T, Y_arr.T, color='b')
ax2[1].axis('equal')
ax2[1].set_xlabel(r'$x/l^*$')
ax2[1].set_ylabel(r'$y/l^*$')
ax2[1].set_title(r'$\tilde{Q}=%.1e, \tilde{t}_s=%.1e, \tilde{t}_m=%.1e$'%(phys.param1, phys.param2, phys.param3))


fig2.tight_layout()
plt.show()



fig3,ax3 = plt.subplots()
ax3.pcolormesh( xx_dns, yy_dns, alpha_dns)
ax3.plot(  X_arr.T, Y_arr.T, color='b')
ax3.axis('equal')

fig3.tight_layout()
plt.show()


figname = './meltpool/meltpool_Q%dW_rb%.2fum_ts%.2fms_tm%.2fms.png'%(phys.Q, phys.rb*1e6, phys.t_spot_on * 1e3,phys.t_taper * 1e3)
fig2.savefig(figname, dpi=300,bbox_inches='tight')


ll = np.abs( len_um( x_dns[0]) )
ww = np.abs( len_um( y_dns[0]) )
print('DNS domain dimension = %.1f x %.1f'%( ll, ww  ))
print('R_max = %.4f [um/s]'%( vel_dim( R_arr.max() ) ) )
print('length scale = %.4f [m], time scale = %.4f [s]'%(phys.len_scale, phys.time_scale) )



W0 = 0.15 * phys.Dl / vel_dim( R_arr.max() )
lamd = 5*np.sqrt(2)/8*W0/ phys.d0     # coupling constan
tau0 = 0.6267*lamd * W0**2/ phys.Dl     # time scale               s
Dl_tilde = phys.Dl*tau0/W0**2


dx_dns = 0.8 * ( W0 )              # [um]
# dx_dns = 0.8 * phys.W0
print ('estimated DNS mesh width %.6f [um]'%(dx_dns))
print('estimated computational domain %d x %d'%( int(ll/dx_dns), int(ww/dx_dns)) )


dt_dns = 0.8*(dx_dns/W0)**2/(4*Dl_tilde)       # time step size for forward euler
Mt_dns = np.ceil( time_dim(t_macro[-1])/tau0  /  dt_dns ) # total  number of time steps (even number)
print('estimated number of DNS time steps are %d  '%( Mt_dns ) )




Xt,Yt = get_sl_interface( temp_dim(T_dns[:,:,0]) , len_um(x_dns) , len_um(y_dns), len_um(x_dns), 914)


# macro QoI points
n_dns_qoi = 500
ds = np.sum( dist_to_sl[:,-1] ) / n_dns_qoi
XY_dns_qoi = []
dist_qoi = []
line_index_qoi = []

for jj in range(ntraj):
    
    XY_dns = np.array( [X_arr[jj,:].T, Y_arr[jj,:].T]  )
    s_dns = dist_to_sl[jj,:]

    f_dns =  interp1d(s_dns,XY_dns)

    kk=1
    while (kk * ds < dist_to_sl[jj,-1]) :
        
        
        print(jj)
        XY_dns_qoi.append( f_dns( kk*ds ) )
        dist_qoi.append( kk*ds )
        line_index_qoi.append( jj )
        
        kk+=1

XY_dns_qoi = np.array(  XY_dns_qoi  ) 
dist_qoi = np.array( dist_qoi )
line_index_qoi = np.array( line_index_qoi )

fig3,ax3 = plt.subplots()
ax3.pcolormesh( len_um( xx_dns) , len_um( yy_dns) , alpha_dns)
ax3.plot(  len_um( X_arr.T) , len_um( Y_arr.T), color='k')
# ax3.plot(  len_um( XY_dns_qoi[:,0]), len_um( XY_dns_qoi[:,1] ) , 'r.' )
# ax3.plot(  len_um( X_arr[2,:].T) , len_um( Y_arr[2,:].T), color='r')
# ax3.plot(  len_um( X_arr[7,:].T) , len_um( Y_arr[7,:].T), color='r')
# ax3.plot(  len_um( X_arr[-8,:].T) , len_um( Y_arr[-8,:].T), color='r')
ax3.plot(  len_um( X_sl[:,0].T),    len_um(Y_sl[:,0].T), color='b')
ax3.set_title('$Q=%dW, r_b=%.2f \mu m, t_s=%.2fms, t_m=%.2fms$'%(phys.Q, phys.rb*1e6, phys.t_spot_on * 1e3, phys.t_taper * 1e3))


ax3.axis('equal')

fig3.tight_layout()
plt.show()


figname = './meltpool/meltpool_Q%dW_rb%.2fum_ts%.2fms_tm%.2fms.png'%(phys.Q, phys.rb*1e6, phys.t_spot_on * 1e3,phys.t_taper * 1e3)
fig3.savefig(figname, dpi=300,bbox_inches='tight')


traj_filename2 = "macrodata_Q%dW_rb%.2fum_ts%.2fms_tm%.2fms.mat"%(phys.Q, phys.rb*1e6, phys.t_spot_on * 1e3, phys.t_taper*1e3)
sio.savemat(os.path.join(simu.direc, traj_filename2), \
            {'T_dns': temp_dim(T_dns), 'alpha_dns': alpha_dns,\
             'gradTx_ini': gradT_um(gradTx_ini), 'gradTy_ini': gradT_um(gradTy_ini),\
             'x_dns': len_um(x_dns), 'y_dns': len_um(y_dns),\
             'G_t': gradT_um(G_arr), 'R_t': vel_dim( R_arr), 't_macro': time_dim(t_macro),\
             'reach_top':reach_top,\
             'X_arr': len_um(X_arr), 'Y_arr': len_um(Y_arr),'dist_arr': len_um(dist_to_sl),\
             'X_sl': len_um(X_sl), 'Y_sl': len_um(Y_sl),\
             'X_qoi': len_um(XY_dns_qoi[:,0]), 'Y_qoi': len_um(XY_dns_qoi[:,1]), 'dist_qoi': len_um(dist_qoi),'line_index_qoi':line_index_qoi,\
             'X_char':X_char, 'Y_char':Y_char, 'psi_char':psi_char,\
             'theta':theta/pi*180,\
             'G0trans': np.mean(gradT_um(G_arr[:,0])), 'Rmax':vel_dim( R_arr.max()) })
    