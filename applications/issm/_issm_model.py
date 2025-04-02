# ==============================================================================
# @des: This file contains run functions for ISSM model python wrapper.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2025-03-26
# @author: Brian Kyanjo
# ==============================================================================

# --- python imports ---
import sys
import os
import shutil
import numpy as np
from scipy.stats import multivariate_normal,norm


# utility imports
from matlab2python.mat2py_utils import subprocess_cmd_run, MatlabServer

# ISSM_DIR = os.environ.get("ISSM_DIR")
# server = MatlabServer(ISSM_DIR)
# server.start()

# --- model initialization ---
def initialize_model(physical_params, modeling_params, comm):
    """ des: intialize the issm model
        - calls the issm initalize_model.m matlab function to initialize the model
    """
    # --- copy intialize_model.m to the current directory
    shutil.copyfile(os.path.join(os.path.dirname(__file__), 'initialize_model.m'), 'initialize_model.m')

    # --- call the initalize_model.m function ---
    issm_cmd = (
    f'matlab -nodisplay -nosplash -nodesktop '
    f'-r "run(\'issm_env\'); initialize_model; exit"'
    )
    subprocess_cmd_run(issm_cmd, 0, 1)

    # try to run the matlab server
    # server.start()
    # server.send_command("run(\'issm_env\'); initialize_model")
    # server.send_command("initialize_model")

    # --- remove the copied file
    # os.remove('initialize_model.m')

    
# ---- ISSM model ----
def ISSM_model(**kwargs):
    """ des: run the issm model
        - calls the issm run_model.m matlab function to run the model
    """

    # --- get the number of processors ---
    nprocs = kwargs.get('nprocs')
    k = kwargs.get('k')
    dt = kwargs.get('dt')
    tinitial = kwargs.get('tinitial')
    tfinal = kwargs.get('tfinal')

    # --- copy run_model.m to the current directory
    shutil.copyfile(os.path.join(os.path.dirname(__file__), 'run_model.m'), 'run_model.m')

    # --- call the run_model.m function ---
    issm_cmd = (
    f'matlab -nodisplay -nosplash -nodesktop '
    f'-r "run(\'issm_env\'); run_model({nprocs},{k},{dt},{tinitial},{tfinal}); exit"'
    )
    subprocess_cmd_run(issm_cmd, nprocs, kwargs.get('verbose'))

    # try to run the matlab server
    # server.send_command(f"run_model({nprocs},{k},{dt},{tinitial},{tfinal})")

    # --- remove the copied file
    # os.remove('run_model.m')

# ---- Run model for ISSM ----
def run_model(ensemble, **kwargs):
    """ des: run the issm model with ensemble matrix from ICESEE
    """
    # --- get the number of processors ---
    nprocs = kwargs.get('nprocs')

    # --- call the issm model  to update the state and parameters variables ---
    ISSM_model(**kwargs)