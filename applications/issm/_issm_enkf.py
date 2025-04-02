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
        