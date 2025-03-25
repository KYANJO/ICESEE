# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-03-24
# @description: ISSM Model with Data Assimilation using a Python Wrapper.
#                
# =============================================================================

# --- Imports ---
import sys
import os
import shutil
import subprocess   
import numpy as np
import scipy.io as sio
import h5py

# --- Configuration ---
sys.path.insert(0, '../../config')
from _utility_imports import *
from _utility_imports import params, kwargs, modeling_params, enkf_params, physical_params
from run_models_da import icesee_model_data_assimilation
from matlab2python.mat2py_utils import subprocess_cmd_run

# --- Initialize MPI ---
from parallel_mpi.icesee_mpi_parallel_manager import ParallelManager
icesee_rank, icesee_size, icesee_comm = ParallelManager().icesee_mpi_init(params)

model_kwargs = {
               'example_name': modeling_params.get('example_name'),
               'Lx': int(float(physical_params.get('Lx'))), 'Ly': int(float(physical_params.get('Ly'))),
                'nx': int(float(physical_params.get('nx'))), 'ny': int(float(physical_params.get('ny'))),
                'ParamFile': modeling_params.get('ParamFile'),
}

# --- update Icesee kwargs with model kwargs ---
kwargs.update(model_kwargs)

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

# save model kwargs to a .mat file inside the examples directory
sio.savemat('model_kwargs.mat', model_kwargs)

# call the run me file to run the model: ISSM uses runme.m to run the model
nprocs = icesee_size

# copy the issm_env.m from icesee_cwd  file to the examples directory
shutil.copy(os.path.join(icesee_cwd,'matlab2python', 'issm_env.m'), issm_examples_dir)

issm_cmd = (
    f'matlab -nodisplay -nosplash -nodesktop '
    f'-r "run(\'issm_env\'); runme({nprocs}); exit"'
)

# subprocess.run(issm_cmd, shell=True, check=True)

# using subprocess to popen
# p = subprocess.Popen(
#     issm_cmd, 
#     shell=True, 
#     stdout=subprocess.PIPE, 
#     stderr=subprocess.PIPE,
#     universal_newlines=True)

# # capture the output
# stdout, stderr = p.communicate()

# # print the output
# verbose = 1
# if verbose:
#     print(f"STDOUT: {stdout}")
#     print(f"STDERR: {stderr}")

# # --- wait for the process to complete
# p.wait()

subprocess_cmd_run(issm_cmd, nprocs, kwargs.get('verbose'))

# -- mimic a forecast run
# Nens = 4
# for ens in range(Nens):
#     print(f"Ensemble member: {ens}")
#     subprocess.run(issm_cmd, shell=True, check=True)
    # --- Run the ISSM model with data assimilation ---


#   save the output to a file

# remove the issm_env.m file from the examples directory
os.remove(os.path.join(issm_examples_dir, 'issm_env.m'))

# go back to the original directory
os.chdir(icesee_cwd)
print(f"[DEBUG] current working directory: {os.getcwd()}")

# --- Run the ISSM model with data assimilation ---
# icesee_model_data_assimilation(params, kwargs, modeling_params, enkf_params, physical_params)

#  repeat the process for the next time step

