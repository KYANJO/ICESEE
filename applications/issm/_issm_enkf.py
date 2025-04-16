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
# sys.path.insert(0, '../../config')
# from _utility_imports import icesee_get_index

# --- Forecast step ---
def forecast_step_single(ensemble=None, **kwargs):
    """ensemble: packs the state variables and parameters of a single ensemble member
    Returns: ensemble: updated ensemble member
    """
    #  -- control time stepping   
    # time = kwargs.get('time')
    # k = kwargs.get('k')
    # time = kwargs.get('t')
    # k = kwargs.get('t_indx')
    # kwargs.update({'tinitial': time[k], 'tfinal': time[k+1]})

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

    server = kwargs.get('server')

    try:
        # -- call the run_model fun to push the state forward in time
        ISSM_model(**kwargs)

        rank = 0
        icesee_path = kwargs.get('icesee_path')
        data_path = kwargs.get('data_path')
        # output_filename = f'ensemble_output_{rank}.h5'
        output_filename = f'{icesee_path}{data_path}/ensemble_output_{rank}.h5'
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
        
    except RuntimeError as e:
        print(f"[DEBUG] Error in run_model: {e}")
        server.shutdown()
        server.reset_terminal()
        sys.exit(1)