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
import h5py
from scipy.stats import multivariate_normal,norm

# --- import run_simulation function from the available examples ---
from synthetic_ice_stream._icepack_model import *
from scipy import linalg

# --- Utility imports ---
sys.path.insert(0, '../../config')
from _utility_imports import icesee_get_index

# --- globally define the state variables ---
global vec_inputs 
vec_inputs = ['h','u','v','smb']

# --- Forecast step ---
def forecast_step_single(ensemble=None, **kwargs):
    """ensemble: packs the state variables:h,u,v of a single ensemble member
                 where h is thickness, u and v are the x and y components 
                 of the velocity field
    Returns: ensemble: updated ensemble member
    """
    #  call the run_model fun to push the state forward in time
    return run_model(ensemble, **kwargs)

# --- Background step ---
def background_step(k=None, **kwargs):
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
    statevec_bg = kwargs["statevec_bg"]

    hb = Function(Q)
    ub = Function(V)

     # --- define the state variables list ---
    global vec_inputs 

    # call the icesee_get_index function to get the indices of the state variables
    vecs, indx_map, dim_per_proc = icesee_get_index(statevec_bg, vec_inputs, **kwargs)

    # fetch the state variables
    hb.dat.data[:]   = statevec_bg[indx_map["h"],k]
    ub.dat.data[:,0] = statevec_bg[indx_map["u"],k]
    ub.dat.data[:,1] = statevec_bg[indx_map["v"],k]

    # call the ice stream model to update the state variables
    hb, ub = Icepack(solver, hb, ub, a, b, dt, h0, fluidity = A, friction = C)

    # update the background state at the next time step
    statevec_bg[indx_map["h"],k+1] = hb.dat.data_ro
    statevec_bg[indx_map["u"],k+1] = ub.dat.data_ro[:,0]
    statevec_bg[indx_map["v"],k+1] = ub.dat.data_ro[:,1]

    if kwargs["joint_estimation"]:
        a = kwargs.get('a', None)
        statevec_bg[indx_map["smb"],0] = a.dat.data_ro
        statevec_bg[indx_map["smb"],k+1] = a.dat.data_ro

    return statevec_bg

# --- generate true state ---
def generate_true_state(**kwargs):
    """generate the true state of the model"""

    # unpack the **kwargs
    a  = kwargs.get('a', None)
    b  = kwargs.get('b', None)
    dt = kwargs.get('dt', None)
    A  = kwargs.get('A', None)
    C  = kwargs.get('C', None)
    Q  = kwargs.get('Q', None)
    V  = kwargs.get('V', None)
    h0 = kwargs.get('h0', None)
    u0 = kwargs.get('u0', None)
    solver = kwargs.get('solver', None)
    statevec_true = kwargs["statevec_true"]

    params = kwargs["params"]
    
    # --- define the state variables list ---
    global vec_inputs 

    # call the icesee_get_index function to get the indices of the state variables
    vecs, indx_map, dim_per_proc = icesee_get_index(statevec_true, vec_inputs, **kwargs)
    
    # --- fetch the state variables ---
    statevec_true[indx_map["h"],0] = h0.dat.data_ro
    statevec_true[indx_map["u"],0] = u0.dat.data_ro[:,0]
    statevec_true[indx_map["v"],0] = u0.dat.data_ro[:,1]

    # intialize the accumulation rate if joint estimation is enabled at the initial time step
    if kwargs["joint_estimation"]:
        statevec_true[indx_map["smb"],0] = a.dat.data_ro

    h = h0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)
    for k in range(params['nt']):
        # call the ice stream model to update the state variables
        h, u = Icepack(solver, h, u, a, b, dt, h0, fluidity = A, friction = C)

        statevec_true[indx_map["h"],k+1] = h.dat.data_ro
        statevec_true[indx_map["u"],k+1] = u.dat.data_ro[:,0]
        statevec_true[indx_map["v"],k+1] = u.dat.data_ro[:,1]

        # update the accumulation rate if joint estimation is enabled
        if kwargs["joint_estimation"]:
            statevec_true[indx_map["smb"],k+1] = a.dat.data_ro

    update_state = {'h': statevec_true[indx_map["h"],:], 
                    'u': statevec_true[indx_map["u"],:], 
                    'v': statevec_true[indx_map["v"],:]}
    # -- for joint estimation --
    if kwargs["joint_estimation"]:
        update_state['smb'] = statevec_true[indx_map["smb"],:]
    return update_state

def generate_nurged_state(**kwargs):
    """generate the nurged state of the model"""
    
    params = kwargs["params"]
    nt = params["nt"] - 1

    # unpack the **kwargs
    a = kwargs.get('a_p', None)
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
    nurged_entries_percentage  = kwargs.get('nurged_entries_percentage', None)

    statevec_nurged = kwargs["statevec_nurged"]

     # --- define the state variables list ---
    global vec_inputs 

    # call the icesee_get_index function to get the indices of the state variables
    vecs, indx_map, dim_per_proc = icesee_get_index(statevec_nurged, vec_inputs, **kwargs)

    #  create a bump -100 to 0
    # h_indx = int(np.ceil(nurged_entries+1))
    hdim = vecs['h'].shape[0]
    h_indx = int(np.ceil(nurged_entries_percentage*hdim+1))
    # if 0.5*hdim > int(np.ceil(nurged_entries+1)):
    #     nurged_entries = nurged_entries_percentage*hdim
    #     h_indx = int(np.ceil(nurged_entries+1))
    # else:
    #     # 5% of the hdim so that the bump is not too large
    #     h_indx = int(np.ceil(hdim*0.025))
    #     # h_indx = int(np.ceil(0.05*nurged_entries+1))
   
    # u_indx = int(np.ceil(u_nurge_ic+1))
    u_indx = 1
    h_bump = np.linspace(-h_nurge_ic,0,h_indx)
    u_bump = np.linspace(-u_nurge_ic,0,h_indx)
    # h_bump = np.random.uniform(-h_nurge_ic,0,h_indx)
    # u_bump = np.random.uniform(-u_nurge_ic,0,h_indx)
    # print(f"hdim: {hdim}, h_indx: {h_indx}")
    # print(f"[Debug]: h_bump shape: {h_bump.shape} h0_index: {h0.dat.data_ro[:h_indx].shape}")
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

    statevec_nurged[indx_map["h"],0]   = h_perturbed
    statevec_nurged[indx_map["u"],0]   = u_perturbed
    statevec_nurged[indx_map["v"],0]   = v_perturbed

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
        statevec_nurged[indx_map["smb"],0] = a.dat.data_ro

    for k in range(params['nt']):
        # aa   = a_in_p*(np.sin(tnur[k]) + 1)
        # daa  = da_p*(np.sin(tnur[k]) + 1)
        aa = a_in_p
        daa = da_p
        a_in = firedrake.Constant(aa)
        da_  = firedrake.Constant(daa)
        a    = firedrake.interpolate(a_in + da_ * x / Lx, Q)
        # call the ice stream model to update the state variables
        h, u = Icepack(solver, h, u, a, b, dt, h0, fluidity = A, friction = C)

        statevec_nurged[indx_map["h"],k+1] = h.dat.data_ro
        statevec_nurged[indx_map["u"],k+1] = u.dat.data_ro[:,0]
        statevec_nurged[indx_map["v"],k+1] = u.dat.data_ro[:,1]

        if kwargs["joint_estimation"]:
            statevec_nurged[indx_map["smb"],k+1] = a.dat.data_ro

    return statevec_nurged


# --- initialize the ensemble members ---
def initialize_ensemble(ens, **kwargs):
    
    """initialize the ensemble members"""

    # unpack the **kwargs
    h0 = kwargs.get('h0', None)
    u0 = kwargs.get('u0', None)
    params = kwargs["params"]
    # a  = kwargs.get('a', None)
    b  = kwargs.get('b', None)
    dt = kwargs.get('dt', None)
    A  = kwargs.get('A', None)
    C  = kwargs.get('C', None)
    Q  = kwargs.get('Q', None)
    V  = kwargs.get('V', None)
    a_in_p = kwargs.get('a_in_p', None)
    da_p = kwargs.get('da_p', None)
    da = kwargs.get('da', None)
    solver = kwargs.get('solver', None)
    h_nurge_ic      = kwargs.get('h_nurge_ic', None)
    u_nurge_ic      = kwargs.get('u_nurge_ic', None)
    nurged_entries_percentage  = kwargs.get('nurged_entries_percentage', None)
    statevec_ens    = kwargs["statevec_ens"]

    # extract the ensemble size
    N = params["Nens"]

     # --- define the state variables list ---
    global vec_inputs 

    # call the icesee_get_index function to get the indices of the state variables
    vecs, indx_map, dim_per_proc = icesee_get_index(statevec_ens, vec_inputs, **kwargs)

    # initialize the ensemble members
    # hdim = vecs['h'].shape[0]
    hdim = h0.dat.data_ro.size
    h_indx = int(np.ceil(nurged_entries_percentage*hdim+1))

    # create a bump -100 to 0
    h_bump = np.linspace(-h_nurge_ic,0,h_indx)
    h_with_bump = h_bump + h0.dat.data_ro[:h_indx]
    h_perturbed = np.concatenate((h_with_bump, h0.dat.data_ro[h_indx:]))
    statevec_ens[:hdim,ens] = h_perturbed 

    initialized_state = {'h': h_perturbed, 
                         'u': u0.dat.data_ro[:,0], 
                         'v': u0.dat.data_ro[:,1]}
    
    # -- for joint estimation --
    if kwargs["joint_estimation"]:
        a_in = firedrake.Constant(a_in_p)
        da_  = firedrake.Constant(da_p)
        a   = firedrake.interpolate(a_in + da_ * kwargs["x"] / kwargs["Lx"], Q)
        initialized_state['smb'] = a.dat.data_ro + np.random.normal(0, 0.01, a.dat.data_ro.size)
       
    return initialized_state

    
# debug function
def initialize_ensemble_debug(color,**kwargs):
    # unpack the **kwargs
    h0 = kwargs.get('h0', None)
    u0 = kwargs.get('u0', None)
    params = kwargs["params"]
    # a  = kwargs.get('a', None)
    b  = kwargs.get('b', None)
    dt = kwargs.get('dt', None)
    A  = kwargs.get('A', None)
    C  = kwargs.get('C', None)
    Q  = kwargs.get('Q', None)
    V  = kwargs.get('V', None)
    a_in_p = kwargs.get('a_in_p', None)
    da_p = kwargs.get('da_p', None)
    da = kwargs.get('da', None)
    solver = kwargs.get('solver', None)
    h_nurge_ic      = kwargs.get('h_nurge_ic', None)
    u_nurge_ic      = kwargs.get('u_nurge_ic', None)
    nurged_entries_percentage  = kwargs.get('nurged_entries_percentage', None)
    statevec_ens    = kwargs["statevec_ens"]
    from mpi4py import MPI

    comm = kwargs.get("comm_world", None)
    subcomm = kwargs.get("subcomm", None)
    subrank = subcomm.Get_rank()
    rank_world = comm.Get_rank()

    # extract the ensemble size
    N = params["Nens"]

     # --- define the state variables list ---
    global vec_inputs 

    # call the icesee_get_index function to get the indices of the state variables
    vecs, indx_map, dim_per_proc = icesee_get_index(statevec_ens, vec_inputs, **kwargs)

    # --------------------*
    # statevec_nurged = generate_nurged_state(**kwargs)
    # h_perturbed = statevec_nurged[indx_map["h"],0]
    h_perturbed = h0.dat.data_ro
    hdim = vecs['h'].shape[0]
    h_indx = int(np.ceil(nurged_entries_percentage*hdim+1))
    # if 0.5*hdim > int(np.ceil(nurged_entries+1)):
    #     h_indx = int(np.ceil(nurged_entries+1))
    # else:
    #     h_indx = int(np.ceil(hdim*0.025))
    h_bump = np.linspace(-h_nurge_ic,0,h_indx)
    h_with_bump = h_bump + h0.dat.data_ro[:h_indx]
    h_perturbed = np.concatenate((h_with_bump, h0.dat.data_ro[h_indx:]))
    statevec_ens[:hdim,0] = h_perturbed

    # ------------------------*
    # # ***create file with parallel acess
    # f = h5py.File(f"ensemble_data.h5", "w", driver='mpio', comm=comm)
    # color_group = f.create_group(f"color_{color}") #** create a group for each color

    # global_h0 = subcomm.gather(h0.dat.data_ro, root=0)
    global_h0 = subcomm.gather(h_perturbed, root=0) #**
    global_u = subcomm.gather(u0.dat.data_ro[:,0], root=0)
    global_v = subcomm.gather(u0.dat.data_ro[:,1], root=0)
    if kwargs["joint_estimation"]:
        if kwargs["parameter_estimation"]:
            a_in = firedrake.Constant(a_in_p)
            da_  = firedrake.Constant(da_p)
            a   = firedrake.interpolate(a_in + da_ * kwargs["x"] / kwargs["Lx"], Q)
            global_smb = subcomm.gather(a.dat.data_ro, root=0)


        # global_smb = subcomm.gather(kwargs["a"].dat.data_ro, root=0)
        # global_smb = subcomm.gather(statevec_nurged[indx_map["smb"],0], root=0)
        # global_smb = subcomm.gather(statevec_nurged[indx_map["smb"],0] + np.random.normal(0, 0.01, hdim), root=0) 

    # *** create a dataset for each state variable
    # dset_h = color_group.create_dataset("h", shape=global_h0.shape, dtype='f8')
    # dset_u = color_group.create_dataset("u", shape=global_u.shape, dtype='f8')
    # dset_v = color_group.create_dataset("v", shape=global_v.shape, dtype='f8')
    # dset_smb = color_group.create_dataset("smb", shape=global_smb.shape, dtype='f8')

    #stacked_state = np.hstack([global_h0,global_u,global_v])
    if subrank == 0:
        # dset_h[:] = global_h0
        # dset_u[:] = global_u
        # dset_v[:] = global_v
        # dset_smb[:] = global_smb

        # ------------------------
        global_h0 = np.hstack(global_h0) 
        global_u = np.hstack(global_u)
        global_v = np.hstack(global_v)
        if kwargs["joint_estimation"]:
            global_smb = np.hstack(global_smb)
            # add noise to smb   
            global_smb = global_smb + np.random.normal(0, 0.01, global_smb.shape) 
        

        # add some kind of perturbations  with mean 0 and variance 1
        noise = np.random.normal(0, 0.1, global_h0.shape)
        global_h0 = global_h0 + noise
        global_u = global_u + noise
        global_v = global_v + noise
        # stack all the state variables
        if kwargs["joint_estimation"]:
            stacked_state = np.hstack([global_h0,global_u,global_v,global_smb])
        else:
            stacked_state = np.hstack([global_h0,global_u,global_v])
        shape_ = stacked_state.shape
        # hdim = shape_[0]//params['total_state_param_vars']
        # state_size = hdim*params['num_state_vars']
        # noise = np.random.normal(0, 0.1, (state_size,))  # Shape (1275, 4)
        # stacked_state[:state_size] += noise
    else:
        shape_ = np.empty(2,dtype=int)

    shape_ = comm.bcast(shape_, root=0)

    # store subrank as attribute
    # color_group.attrs["subrank"] = subrank
    # f.close()
    
    if subrank != 0:
        stacked_state = np.empty(shape_,dtype=float)

    all_colors = comm.gather(stacked_state if subrank == 0 else None, root=0)
    if rank_world == 0:
        all_colors = [arr for arr in all_colors if arr is not None]
        statevec_ens = np.column_stack(all_colors)
        # print(f"[Debug]: all_colors shape: {all_colors.shape}")
        # add noise here to the ensemble members
        hdim = statevec_ens.shape[0]//params['total_state_param_vars']
        state_size = hdim*params['num_state_vars']
        # # noise = np.random.normal(0, 0.1, statevec_ens.shape)
        # noise = np.random.normal(0, 0.1, (state_size, statevec_ens.shape[1]))  # Shape (1275, 4)
        # statevec_ens[:state_size, :] += noise
        # add some noise to smb
        # if kwargs["joint_estimation"]:
        #     noise = np.random.normal(0, 0.01, (hdim,params["Nens"])) 
        #     statevec_ens[state_size:, :] += noise
        
    else: 
        # None
        statevec_ens = np.empty((shape_[0], params['Nens']),dtype=float)
    # comm.Bcast(statevec_ens, root=0)


    return statevec_ens,shape_

def generate_random_field(kernel='gaussian',**kwargs):
    """
    Generate a 2D pseudorandom field with mean 0 and variance 1.
    
    Parameters:
    - size: tuple of (height, width) for the field dimensions
    - length_scale: float, controls smoothness (larger = smoother)
    - num_points: int, number of grid points per dimension
    - kernel: str, type of covariance kernel ('gaussian' or 'exponential')
    
    Returns:
    - field: 2D numpy array with the random field
    """

    Lx, Ly = kwargs["Lx"], kwargs["Ly"]
    nx, ny = kwargs["nx"], kwargs["ny"]

    length_scale = 0.2*max(Lx,Ly)
    
    # Create grid
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    X, Y = np.meshgrid(x, y)
    
    # Compute distances between all points
    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    # Define covariance kernel
    if kernel == 'gaussian':
        cov = np.exp(-dist**2 / (2 * length_scale**2))
    elif kernel == 'exponential':
        cov = np.exp(-dist / length_scale)
    else:
        raise ValueError("Kernel must be 'gaussian' or 'exponential'")
    
    # Ensure positive definiteness and symmetry
    cov = (cov + cov.T) / 2  # Make perfectly symmetric
    cov += np.eye(cov.shape[0]) * 1e-6  # Add small jitter for stability
    
    # Generate random field using Cholesky decomposition
    L = linalg.cholesky(cov, lower=True)
    # z = np.random.normal(0, 1, size=num_points * num_points)
    z = np.random.normal(0, 1, size=(nx+1)*(ny+1))
    field_flat = L @ z
    
    # Reshape and normalize to mean 0, variance 1
    # field = field_flat.reshape(num_points, num_points)
    field = field_flat.reshape(nx+1, ny+1)
    field = (field - np.mean(field)) / np.std(field)
    
    return field
