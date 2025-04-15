# ==============================================================================
# @des: This file contains run functions for ISSM data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2025-03-25
# @author: Brian Kyanjo
# ==============================================================================

import sys
import os
import numpy as np
import re
import h5py
from scipy import linalg
from scipy.stats import multivariate_normal,norm


# --- import utility functions ---
from _issm_model import *

# --- Forecast step ---
def forecast_step_single(ensemble=None, **kwargs):
    """ensemble: packs the state variables and parameters of a single ensemble member
    Returns: ensemble: updated ensemble member
    """
    #  call the run_model fun to push the state forward in time
    return run_model(ensemble, **kwargs)

# --- generate true state ---
def generate_true_state(**kwargs):
    """des: generate the true state of the model
    Returns: true_state: the true state of the model
    """
    

    # for k, t in enumerate(kwargs['t']):
    run_model(np.zeros(4), **kwargs)
        
#  --- initialize ensemble members ---
def initialize_ensemble(ens, **kwargs):
    """des: initialize the ensemble members
    Returns: ensemble: the ensemble members
    """
    import h5py

    #  -- call the ISSM_model to initialize the ensemble members
    k = 0
    time = kwargs.get('time')
    kwargs.update({'k':k})
    kwargs.update({'dt':time[1]-time[0]})
    kwargs.update({'tinitial': time[k]})
    kwargs.update({'tfinal': time[k+1]})
    ISSM_model(**kwargs)

    rank = 0
    output_filename = f'ensemble_output_{rank}.h5'
    with h5py.File(output_filename, 'r') as f:
        # Read the data from the file
        # for key in f.keys():
        #     output_dic[key] = f[key][:]
        return {
        'Vx': f['Vx'][0],
        'Vy': f['Vy'][0],
        'Vz': f['Vz'][0],
        'Pressure': f['Pressure'][0]
        }