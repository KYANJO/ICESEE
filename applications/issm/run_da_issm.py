# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-03-24
# @description: ISSM Model with Data Assimilation using a Python Wrapper.
#                
# =============================================================================

# --- Imports ---
import sys
import os
import subprocess   
import numpy as np

# --- Configuration ---
sys.path.insert(0, '../../config')
from _utility_imports import *
from _utility_imports import params, kwargs, modeling_params, enkf_params, physical_params
from run_models_da import icesee_model_data_assimilation

# --- Initialize MPI ---
from parallel_mpi.icesee_mpi_parallel_manager import ParallelManager
icesee_rank, icesee_size, icesee_comm = ParallelManager().icesee_mpi_init(params)

kwargs.update({
               'example_name': modeling_params.get('example_name')
})

# --- get current working directory ---
icesee_cwd = os.getcwd()

# --- change directory to issm model directory: make sure ISSM_DIR is set in the environment
issm_dir = os.environ.get('ISSM_DIR')
if issm_dir:
    if os.path.isdir(issm_dir):
        # Optionally add ISSM_DIR and all subdirectories to sys.path
        for root, dirs, _ in os.walk(issm_dir):
            sys.path.insert(0, root)

        print(f"Added ISSM directory and subdirectories from path: {issm_dir}")
    else:
        raise FileNotFoundError(f"The ISSM_DIR directory does not exist: {issm_dir}")
else:
    raise EnvironmentError("ISSM_DIR is not set. Please set the ISSM_DIR environment variable.")

# --- make the examples directory available ---
issm_examples_dir = os.path.join(issm_dir, 'examples',kwargs.get('example_name'))

# --- change directory to the examples directory ---
os.chdir(issm_examples_dir)
print(f"[DEBUG] current working directory: {os.getcwd()}")

# call the run me file to run the model: ISSM uses runme.m to run the model
nprocs = icesee_size
issm_run_cmd = f'matlab -nodisplay -nosplash -nodesktop -r "run(\'runme({nprocs})\'); exit"'
subprocess.run(issm_run_cmd, shell=True, check=True)


#   save the output to a file

# go back to the original directory
os.chdir(icesee_cwd)
print(f"[DEBUG] current working directory: {os.getcwd()}")

# --- Run the ISSM model with data assimilation ---
# icesee_model_data_assimilation(params, kwargs, modeling_params, enkf_params, physical_params)

#  repeat the process for the next time step

