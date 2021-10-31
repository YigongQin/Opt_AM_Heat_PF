#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 22:22:39 2021

@author: yigongqin
"""
import numpy as np
from math import pi
from scipy import sparse as sp
from scipy.sparse import linalg as spla
from scipy.interpolate import interp2d as itp2d
from scipy.optimize import fsolve, brentq
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


def dphase_trans(u):
    
    mask = (u > 0 ) & (u < 1)
    
    T_lat = pi/2. * np.sin(pi * u) * mask;
    
    return sp.diags( T_lat, format = 'csc')


def set_top_bc(rhs, qs, u_top, t, CFL, phys, simu):
    
    
    bU = -2*simu.h * ( -qs + phys.n2*(u_top + phys.n3) + phys.n4*((u_top + phys.n5)**4 - phys.n6**4  ) )
    
    rhs[-simu.nx:] = rhs[-simu.nx:] + bU * CFL 
    
    
def interp_unstructed2grid(xg,yg, alpha, X,Y):
    
    pts= np.array( ( X.flatten(), Y.flatten() ) ).T
    
    val= alpha.flatten()
    
    alpha_z0 = griddata( pts, val, ( xg.flatten() , yg.flatten() ), method = 'linear')
    
    alpha_z1 = griddata( pts, val, ( xg.flatten() , yg.flatten() ), method = 'nearest')
    
    
    # fill nan with nearest
    alpha_z0[ np.isnan(alpha_z0) ] = alpha_z1[ np.isnan(alpha_z0) ]
    
    return alpha_z0
    
    
    
    
    
def export_mat(CFL, simu, nv):
    
    Lap,Q = sparse_laplacian(simu.nx, simu.ny)
    I = sp.eye(simu.nx * simu.ny,format='csc')
    
    A0 = Q @ ( I - CFL*Lap)

    # preconditioner
    M2 = spla.splu(A0)
    M_x = lambda x: M2.solve(x)
    M = spla.LinearOperator((nv,nv), M_x)

    return Lap, Q, I, A0, M


def comp_pool_depth(u, center, x, y):
    
    u2d = np.reshape(u, (simu.nx, simu.ny), order='F')
    u_interp = itp2d(x, y, u2d.T)
    
    f = lambda s : u_interp(  center[0] , center[1] + s ) - 1
                            
    pd = brentq(f, y.min() ,0)
      
    
    assert pd<0
    return np.abs(pd) 




