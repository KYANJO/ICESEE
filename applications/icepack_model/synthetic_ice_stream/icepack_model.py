# ==============================================================================
# @des: This file contains run functions for icepack data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2024-11-4
# @author: Brian Kyanjo
# ==============================================================================

import sys
import os
import numpy as np
from scipy.stats import multivariate_normal,norm

import firedrake
import icepack
import tqdm

from firedrake import *

# --- icepack model ---
def Icepack(solver, h, u, a, b, dt, h0, **kwargs):
    """inputs: solver - icepack solver
                h - ice thickness
                u - ice velocity
                a - ice accumulation
                b - ice bed
                dt - time step
                h0 - ice thickness inflow
                *args - additional arguments for the model
        outputs: h - updated ice thickness
                 u - updated ice velocity
    """
    h = solver.prognostic_solve(
        dt = dt,
        thickness = h,
        velocity = u,
        accumulation = a,
        thickness_inflow = h0,
    )

    s = icepack.compute_surface(thickness = h, bed = b)

    u = solver.diagnostic_solve(
        velocity = u,
        thickness = h,
        surface = s,
        **kwargs
    )

    return h, u

# --- Run model for the icepack model ---
def run_model(ens, ensemble, nd, params, **kwargs):
    """des: icepack model function
        inputs: ensemble - current state of the model
                **kwargs - additional arguments for the model
        outputs: model run
    """

    # unpack the **kwargs
    # a = kwargs.get('a', None)
    b  = kwargs.get('b', None)
    dt = kwargs.get('dt', None)
    h0 = kwargs.get('h0', None)
    A  = kwargs.get('A', None)
    C  = kwargs.get('C', None)
    Q  = kwargs.get('Q', None)
    V  = kwargs.get('V', None)
    solver = kwargs.get('solver', None)
   
    ndim = nd // params["num_state_vars"]
    num_state_vars = params["num_state_vars"]
    
    # unpack h,u,v from the ensemble member
    h_vec = ensemble[:ndim,ens]
    u_vec = ensemble[ndim:2*ndim,ens]
    v_vec = ensemble[2*ndim:3*ndim,ens]

    # joint estimation
    if kwargs["joint_estimation"]:
        # Use analysis step to update the accumulation rate
        # - pack accumulation rate with the state variables to
        #   get ensemble = [h,u,v,a]
        a_vec = ensemble[3*ndim:,ens]
        a = Function(Q)
        a.dat.data[:] = a_vec.copy()
    else:
        # don't update the accumulation rate (updates smb)
        a = kwargs.get('a', None)

    # create firedrake functions from the ensemble members
    h = Function(Q)
    h.dat.data[:] = h_vec.copy()

    u = Function(V)
    u.dat.data[:,0] = u_vec.copy()
    u.dat.data[:,1] = v_vec.copy()

    # call the ice stream model to update the state variables
    h, u = Icepack(solver, h, u, a, b, dt, h0, fluidity = A, friction = C)

    # update the ensemble members with the new state variables and noise 
    ensemble[:ndim,ens]              = h.dat.data_ro       
    ensemble[ndim:2*ndim,ens]        = u.dat.data_ro[:,0]  
    ensemble[2*ndim:3*ndim,ens]      = u.dat.data_ro[:,1] 

    return ensemble[:,ens]

def copy_data2funcspace(ensemble, func, params, **kwargs):
    """des: copy data to a function space
        inputs: data - data to be copied
                func - function space to copy data to
        outputs: data copied to the function space
    """
    # unpack the **kwargs
    Q  = kwargs.get('Q', None)
    V  = kwargs.get('V', None)

    # unpack the params
    nd = params["nd"]

    ndim = nd // params["num_state_vars"]

    # unpack h,u,v from all the ensemble members
    h_vec = ensemble[:ndim,:]
    u_vec = ensemble[ndim:2*ndim,:]
    v_vec = ensemble[2*ndim:3*ndim,:]

    # create firedrake functions from the ensemble members
    h = Function(Q)
    h.dat.data[:] = h_vec

    u = Function(V)
    u.dat.data[:,0] = u_vec
    u.dat.data[:,1] = v_vec


