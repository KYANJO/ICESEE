# ==============================================================================
# @des: This file contains run functions for icepack data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2024-11-4
# @author: Brian Kyanjo
# ==============================================================================

import sys
import os
import numpy as np
import re
from scipy.stats import multivariate_normal,norm

# --- import run_simulation function from the available examples ---
from synthetic_ice_stream._icepack_model import *

# --- Forecast step ---
def forecast_step_single(ens=None, ensemble=None, nd=None, Q_err=None, params=None, **kwargs):
    """ensemble: packs the state variables:h,u,v of a single ensemble member
                 where h is thickness, u and v are the x and y components 
                 of the velocity field
    Returns: ensemble: updated ensemble member
    """
    # get the dimension of the state variables
    hdim = nd // (params["num_state_vars"] + params["num_param_vars"])
    state_block_size = hdim*params["num_state_vars"]


    #  call the run_model fun to push the state forward in time
    ensemble[:,ens] = run_model(ens, ensemble, nd, params, **kwargs)

    # add noise to the state variables
    noise = np.random.multivariate_normal(np.zeros(state_block_size), Q_err)

    # update the ensemble with the noise
    ensemble[:state_block_size,ens] = ensemble[:state_block_size,ens] + noise

    return ensemble[:,ens]

# --- Background step ---
def background_step(k=None,statevec_bg=None, hdim=None, **kwargs):
    """ computes the background state of the model
    Args:
        k: time step index
        statevec_bg: background state of the model
        hdim: dimension of the state variables
    Returns:
        statevec_bg: updated background state of the model
    """
    # unpack the **kwargs
    # a = kwargs.get('a', None)
    b = kwargs.get('b', None)
    dt = kwargs.get('dt', None)
    h0 = kwargs.get('h0', None)
    A = kwargs.get('A', None)
    C = kwargs.get('C', None)
    Q = kwargs.get('Q', None)
    V = kwargs.get('V', None)
    solver = kwargs.get('solver', None)

    hb = Function(Q)
    ub = Function(V)

    hb.dat.data[:]   = statevec_bg[:hdim,k]
    ub.dat.data[:,0] = statevec_bg[hdim:2*hdim,k]
    ub.dat.data[:,1] = statevec_bg[2*hdim:3*hdim,k]

    # call the ice stream model to update the state variables
    hb, ub = Icepack(solver, hb, ub, a, b, dt, h0, fluidity = A, friction = C)

    # update the background state at the next time step
    statevec_bg[:hdim,k+1] = hb.dat.data_ro
    statevec_bg[hdim:2*hdim,k+1] = ub.dat.data_ro[:,0]
    statevec_bg[2*hdim:3*hdim,k+1] = ub.dat.data_ro[:,1]

    if kwargs["joint_estimation"]:
        a = kwargs.get('a', None)
        statevec_bg[3*hdim:,0] = a.dat.data_ro
        statevec_bg[3*hdim:,k+1] = a.dat.data_ro

    return statevec_bg

# --- generate true state ---
def generate_true_state(statevec_true=None,params=None, **kwargs):
    """generate the true state of the model"""
    nd, nt = statevec_true.shape
    nt = nt - 1
    hdim = nd // (params["num_state_vars"] + params["num_param_vars"])

    # unpack the **kwargs
    a = kwargs.get('a', None)
    b = kwargs.get('b', None)
    dt = kwargs.get('dt', None)
    A = kwargs.get('A', None)
    C = kwargs.get('C', None)
    Q = kwargs.get('Q', None)
    V = kwargs.get('V', None)
    h0 = kwargs.get('h0', None)
    u0 = kwargs.get('u0', None)
    solver = kwargs.get('solver', None)

    statevec_true[:hdim,0]       = h0.dat.data_ro
    statevec_true[hdim:2*hdim,0] = u0.dat.data_ro[:,0]
    statevec_true[2*hdim:3*hdim,0]     = u0.dat.data_ro[:,1]

    # intialize the accumulation rate if joint estimation is enabled at the initial time step
    if kwargs["joint_estimation"]:
        statevec_true[3*hdim:,0] = a.dat.data_ro

    h = h0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)
    for k in tqdm.trange(nt):
        # call the ice stream model to update the state variables
        h, u = Icepack(solver, h, u, a, b, dt, h0, fluidity = A, friction = C)

        statevec_true[:hdim,k+1]        = h.dat.data_ro
        statevec_true[hdim:2*hdim,k+1]  = u.dat.data_ro[:,0]
        statevec_true[2*hdim:3*hdim,k+1]      = u.dat.data_ro[:,1]

        # update the accumulation rate if joint estimation is enabled
        if kwargs["joint_estimation"]:
            statevec_true[3*hdim:,k+1] = a.dat.data_ro

    return statevec_true

def generate_nurged_state(statevec_nurged=None,params=None,**kwargs):
    """generate the nurged state of the model"""
    nd, nt = statevec_nurged.shape
    nt = nt - 1
    hdim = nd // (params["num_state_vars"] + params["num_param_vars"])

    # unpack the **kwargs
    a = kwargs.get('a', None)
    t = kwargs.get('t', None)
    x = kwargs.get('x', None)
    Lx = kwargs.get('Lx', None)
    b = kwargs.get('b', None)
    dt = kwargs.get('dt', None)
    A = kwargs.get('A', None)
    C = kwargs.get('C', None)
    Q = kwargs.get('Q', None)
    V = kwargs.get('V', None)
    h0 = kwargs.get('h0', None)
    u0 = kwargs.get('u0', None)
    solver = kwargs.get('solver', None)
    a_in_p = kwargs.get('a_in_p', None)
    da_p = kwargs.get('da_p', None)
    da = kwargs.get('da', None)
    h_nurge_ic      = kwargs.get('h_nurge_ic', None)
    u_nurge_ic      = kwargs.get('u_nurge_ic', None)
    nurged_entries  = kwargs.get('nurged_entries', None)

    #  create a bump -100 to 0
    h_indx = int(np.ceil(nurged_entries+1))
    # u_indx = int(np.ceil(u_nurge_ic+1))
    u_indx = 1
    h_bump = np.linspace(-h_nurge_ic,0,h_indx)
    u_bump = np.linspace(-u_nurge_ic,0,h_indx)
    # h_bump = np.random.uniform(-h_nurge_ic,0,h_indx)
    # u_bump = np.random.uniform(-u_nurge_ic,0,h_indx)

    h_with_bump = h_bump + h0.dat.data_ro[:h_indx]
    u_with_bump = u_bump + u0.dat.data_ro[:h_indx,0]
    v_with_bump = u_bump + u0.dat.data_ro[:h_indx,1]

    h_perturbed = np.concatenate((h_with_bump, h0.dat.data_ro[h_indx:]))
    u_perturbed = np.concatenate((u_with_bump, u0.dat.data_ro[h_indx:,0]))
    v_perturbed = np.concatenate((v_with_bump, u0.dat.data_ro[h_indx:,1]))

    # if velocity is nurged, then run to get a solution to be used as am initial guess for velocity.
    if u_nurge_ic != 0.0:
        h = Function(Q)
        u = Function(V)
        h.dat.data[:]   = h_perturbed
        u.dat.data[:,0] = u_perturbed
        u.dat.data[:,1] = v_perturbed
        h0 = h.copy(deepcopy=True)
        # call the solver
        h, u = Icepack(solver, h, u, a, b, dt, h0, fluidity = A, friction = C)

        # update the nurged state with the solution
        h_perturbed = h.dat.data_ro
        u_perturbed = u.dat.data_ro[:,0]
        v_perturbed = u.dat.data_ro[:,1]


    statevec_nurged[:hdim,0]            = h_perturbed
    statevec_nurged[hdim:2*hdim,0]      = u_perturbed
    statevec_nurged[2*hdim:3*hdim,0]    = v_perturbed

    # h = h0.copy(deepcopy=True)
    # u = u0.copy(deepcopy=True)
    # update h0 and u0 with the nurged values

    h = Function(Q)
    u = Function(V)
    h.dat.data[:] = h_perturbed
    u.dat.data[:,0] = u_perturbed
    u.dat.data[:,1] = v_perturbed
    h0 = h.copy(deepcopy=True)

    tnur = np.linspace(.1, 2, nt)
    # intialize the accumulation rate if joint estimation is enabled at the initial time step
    if kwargs["joint_estimation"]:
        # aa   = a_in_p*(np.sin(tnur[0]) + 1)
        # daa  = da_p*(np.sin(tnur[0]) + 1)
        aa = a_in_p
        daa = da_p
        a_in = firedrake.Constant(aa)
        da_  = firedrake.Constant(daa)
        a    = firedrake.interpolate(a_in + da_ * x / Lx, Q)
        statevec_nurged[3*hdim:,0] = a.dat.data_ro

    for k in tqdm.trange(nt):
        # aa   = a_in_p*(np.sin(tnur[k]) + 1)
        # daa  = da_p*(np.sin(tnur[k]) + 1)
        aa = a_in_p
        daa = da_p
        a_in = firedrake.Constant(aa)
        da_  = firedrake.Constant(daa)
        a    = firedrake.interpolate(a_in + da_ * x / Lx, Q)
        # call the ice stream model to update the state variables
        h, u = Icepack(solver, h, u, a, b, dt, h0, fluidity = A, friction = C)

        statevec_nurged[:hdim,k+1]          = h.dat.data_ro
        statevec_nurged[hdim:2*hdim,k+1]    = u.dat.data_ro[:,0]
        statevec_nurged[2*hdim:3*hdim,k+1]  = u.dat.data_ro[:,1]

        if kwargs["joint_estimation"]:
            statevec_nurged[3*hdim:,k+1] = a.dat.data_ro

    return statevec_nurged


# --- initialize the ensemble members ---
def initialize_ensemble(statevec_bg=None, statevec_ens=None, \
                        statevec_ens_mean=None, statevec_ens_full=None, params=None,**kwargs):
    
    """initialize the ensemble members"""
    nd, N = statevec_ens.shape
    hdim = nd // (params["num_state_vars"] + params["num_param_vars"])

    # unpack the **kwargs
    h0 = kwargs.get('h0', None)
    u0 = kwargs.get('u0', None)
    # a  = kwargs.get('a', None)
    b  = kwargs.get('b', None)
    dt = kwargs.get('dt', None)
    A  = kwargs.get('A', None)
    C  = kwargs.get('C', None)
    Q  = kwargs.get('Q', None)
    V  = kwargs.get('V', None)
    solver = kwargs.get('solver', None)
    h_nurge_ic      = kwargs.get('h_nurge_ic', None)
    u_nurge_ic      = kwargs.get('u_nurge_ic', None)
    nurged_entries  = kwargs.get('nurged_entries', None)

    # call the nurged state to initialize the ensemble
    statevec_nurged = generate_nurged_state( np.zeros_like(statevec_bg), params, **kwargs)
                                           
    # fetch h u, and v from the nurged state
    h_perturbed = statevec_nurged[:hdim,0]
    u_perturbed = statevec_nurged[hdim:2*hdim,0]
    v_perturbed = statevec_nurged[2*hdim:3*hdim,0]

    # initialize the ensemble members
    h_indx = int(np.ceil(nurged_entries+1))
    for i in range(N):
        # intial thickness perturbed by bump
        # h_bump = np.random.uniform(-h_nurge_ic,0,h_indx)
        # h_bump = np.random.normal(-h_nurge_ic,0.1,h_indx)
        h_bump = np.linspace(-h_nurge_ic,0,h_indx)
        # h_with_bump = h_bump + h_perturbed[:h_indx]
        # h_perturbed = np.concatenate((h_with_bump, h_perturbed[h_indx:]))
        h_with_bump = h_bump + h0.dat.data_ro[:h_indx]
        h_perturbed = np.concatenate((h_with_bump, h0.dat.data_ro[h_indx:]))
        statevec_ens[:hdim,i] = h_perturbed 

        # intial velocity unperturbed
        # statevec_ens[hdim:2*hdim,i] = u_perturbed
        # statevec_ens[2*hdim:3*hdim,i]     = v_perturbed
        statevec_ens[hdim:2*hdim,i] = u0.dat.data_ro[:,0]
        statevec_ens[2*hdim:3*hdim,i]     = u0.dat.data_ro[:,1]

        # initilize the accumulation rate if joint estimation is enabled
        if kwargs["joint_estimation"]:
            # add some spread to the intial accumulation rate
            initial_smb = statevec_nurged[3*hdim:,0]
            # create a normal distribution spread for the accumulation rate
            spread = np.random.normal(0, 0.08, initial_smb.shape)
            # print(f"[Debug]: spread and initail_smb shapes: {spread.shape}, {initial_smb.shape}")
            statevec_ens[3*hdim:,i] = initial_smb + spread

            # statevec_ens[3*hdim:,i] = statevec_nurged[3*hdim:,0]

    statevec_ens_full[:,:,0] = statevec_ens

    # initialize the background state
    statevec_bg[:hdim,0]       = h_perturbed
    statevec_bg[hdim:2*hdim,0] = u_perturbed
    statevec_bg[2*hdim:3*hdim,0]     = v_perturbed

    # initialize the ensemble mean
    statevec_ens_mean[:hdim,0]       = h_perturbed
    statevec_ens_mean[hdim:2*hdim,0] = u_perturbed
    statevec_ens_mean[2*hdim:3*hdim,0]     = v_perturbed

    # intialize the joint estimation variables
    if kwargs["joint_estimation"]:
        # a = kwargs.get('a', None)
        statevec_bg[3*hdim:,0]       = statevec_nurged[3*hdim:,0]
        statevec_ens_mean[3*hdim:,0] = statevec_nurged[3*hdim:,0]

    return statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full