# ==============================================================================
# @des: This file contains run functions for lorenz data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2025-01-13
# @author: Brian Kyanjo
# ==============================================================================

import sys
import os
import numpy as np
from scipy.stats import multivariate_normal,norm

# --- import run_simulation function from the lorenz96 model ---
from lorenz96_model import *

# --- Forecast step for the Lorenz96 model ---
def forecast_step_single(ens=None, ensemble=None, nd=None, Q_err=None, params=None, **kwargs):
    """inputs: run_simulation - function that runs the model
                ensemble - current state of the model
                dt - time step
                *args - additional arguments for the model
         outputs: uai - updated state of the model after one time step
    """

    # call the run_simulation fun to push the state forward in time
    ensemble[:,ens] = run_simulation(ensemble[:,ens], **kwargs)

    # add noise to the state variables
    # noise = multivariate_normal.rvs(mean=np.zeros(nd), cov=Q_err)
    noise = np.random.multivariate_normal(np.zeros(nd), Q_err)

    # update the ensemble with the noise
    ndim = nd//params['num_state_vars']
    ensemble[:,ens] = ensemble[:,ens] + noise
    
    return ensemble[:,ens]

# --- Background step for the Lorenz96 model ---
def background_step(k=None,statevec_bg=None, hdim=None, **kwargs):
    """inputs: k - current time step
                run_simulation - function that runs the model
                state - current state of the model
                dt - time step
                *args - additional arguments for the model
        outputs: state - updated state of the model after one time step
    """
    # Call the run_simulationfunction to push the state forward in time
    statevec_bg[:,k+1] = run_simulation(statevec_bg[:,k], **kwargs)

    return statevec_bg

# --- generate true state ---
def generate_true_state(statevec_true=None,params=None, **kwargs):
    """inputs: statevec_true - true state of the model
                params - parameters of the model
                *args - additional arguments for the model
        outputs: statevec_true - updated true state of the model after one time step
    """
    # Unpack the parameters
    nd = params['nd']
    nt = params['nt']
    dt = params['dt']
    num_state_vars = params['num_state_vars']
    u0True = kwargs.get('u0True', None)


    # Set the initial condition
    statevec_true[:, 0] = u0True

    # Run the model forward in time
    for k in range(nt):
        statevec_true[:, k + 1] = run_simulation(statevec_true[:, k], **kwargs)

    return statevec_true

# --- initialize the ensemble members ---
def initialize_ensemble(statevec_bg=None, statevec_ens=None, \
                        statevec_ens_mean=None, statevec_ens_full=None, params=None,**kwargs):
    """initialize the ensemble members"""
    nd, N = statevec_ens.shape
    hdim = nd // params["num_state_vars"]

    u0b = kwargs.get('u0b', None)
    B = params["sig_model"]**2 * np.eye(nd)
    for ens in range(N):
        statevec_ens[:, ens] = u0b + np.random.multivariate_normal(np.zeros(nd), B)  

    statevec_bg[:,0] = u0b
    statevec_ens_mean[:,0] = u0b
    statevec_ens_full[:,:,0] = statevec_ens
    return statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full