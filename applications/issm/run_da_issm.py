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
import socket
import numpy as np
import scipy.io as sio
import h5py

# --- Configuration ---
sys.path.insert(0, '../../config')
from _utility_imports import *
from _utility_imports import params, kwargs, modeling_params, enkf_params, physical_params
from run_models_da import icesee_model_data_assimilation
from matlab2python.mat2py_utils import subprocess_cmd_run, MatlabServer

# ISSM_DIR = os.environ.get("ISSM_DIR")
# server = MatlabServer()
# server.launch()

# --- Utility Functions ---
from _issm_model import initialize_model

# --- Initialize MPI ---
from parallel_mpi.icesee_mpi_parallel_manager import ParallelManager
icesee_rank, icesee_size, icesee_comm = ParallelManager().icesee_mpi_init(params)

model_kwargs = {
               'example_name': modeling_params.get('example_name'),
               'Lx': int(float(physical_params.get('Lx'))), 'Ly': int(float(physical_params.get('Ly'))),
                'nx': int(float(physical_params.get('nx'))), 'ny': int(float(physical_params.get('ny'))),
                'ParamFile': modeling_params.get('ParamFile'),
                'cluster_name': socket.gethostname().replace('-', ''),
                'extrusion_layers': int(float(modeling_params.get('extrusion_layers'))),
                'extrusion_exponent': int(float(modeling_params.get('extrusion_exponent'))),
                'steps': int(float(modeling_params.get('steps'))),
                'flow_model': modeling_params.get('flow_model'),
                'sliding_vx': float(modeling_params.get('sliding_vx')),
                'sliding_vy': float(modeling_params.get('sliding_vy')),
                'dt': float(modeling_params.get('timesteps_per_year')),
                'tinitial': float(modeling_params.get('tinitial')),
                'tfinal': float(modeling_params.get('num_years')),
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
# print(f"[DEBUG] current working directory: {os.getcwd()}")

# save model kwargs to a .mat file inside the examples directory
sio.savemat('model_kwargs.mat', model_kwargs)

# copy the issm_env.m from icesee_cwd  file to the examples directory
shutil.copy(os.path.join(icesee_cwd,'matlab2python', 'issm_env.m'), issm_examples_dir)
# shutil.copy(os.path.join(icesee_cwd,'initialize_model.m'), issm_examples_dir)
# shutil.copy(os.path.join(icesee_cwd,'run_model.m'), issm_examples_dir)

# --- initaialize the ISSM model ---
server = MatlabServer()
server.launch()
if icesee_rank == 0:
    initialize_model(physical_params, modeling_params, icesee_comm)
else:
    pass
# wait for rank 0 to write to file before proceeding
icesee_comm.Barrier()

from _issm_enkf import forecast_step_single

kwargs.update({'nprocs': icesee_size, 'verbose': 1})
# forecast_step_single(ensemble=None, **kwargs)
# nprocs = kwargs.get('nprocs')
# issm_cmd = (
#     'matlab -nodisplay -nosplash -nodesktop '
#     '-r "addpath(genpath(getenv(\'ISSM_DIR\'))); '
#     'run(\'issm_env\'); run_model({nprocs}); exit"'
#     .format(nprocs=nprocs)
#     )
# subprocess_cmd_run(issm_cmd, nprocs, kwargs.get('verbose'))

print(f"[DEBUG] Testing the forecast step function")
# time = np.linspace(0,80,20)
dt = 4
time = np.arange(0, 13, dt)
tinitial = 0
Nens = 2

kwargs.update({'time': time})

# -> server approach
# server = MatlabServer()
# server.launch()
try:
    # Send a test command
    # if not server.send_command("disp('Hello from Python')"):
    #     raise RuntimeError("Test command failed.")

    kwargs.update({'server': server})
    for k in range(len(time)-1):
        kwargs.update({'k':k})
        kwargs.update({'dt':dt})
        kwargs.update({'tinitial': time[k]})
        kwargs.update({'tfinal': time[k+1]})
        print(f"\n[DEBUG] Running the model from time: {time[k]} to {time[k+1]}\n")
        for ens in range(Nens):
            print(f"[DEBUG] Running ensemble member: {ens}")
            forecast_step_single(ensemble=None, **kwargs)

    # shutdown the matlab server
    server.shutdown()
    server.reset_terminal()

except RuntimeError as e:
    print(f"[Laucher] Error: {e}")
    server.shutdown()
    server.reset_terminal()
    sys.exit(1)




    
# for k in range(len(time)-1):
#     kwargs.update({'k':k})
#     kwargs.update({'dt':dt})
#     kwargs.update({'tinitial': time[k]})
#     kwargs.update({'tfinal': time[k+1]})
#     print(f"\n[DEBUG] Running the model from time: {time[k]} to {time[k+1]}\n")
#     for ens in range(Nens):
#         print(f"[DEBUG] Running ensemble member: {ens}")
#         forecast_step_single(ensemble=None, **kwargs)
    # forecast_step_single(ensemble=None, **kwargs)

# shut down the matlab server
# server.shutdown()

# remove the issm_env.m file from the examples directory
# os.remove(os.path.join(issm_examples_dir, 'issm_env.m'))

# go back to the original directory
os.chdir(icesee_cwd)
print(f"[DEBUG] current working directory: {os.getcwd()}")

# --- Run the ISSM model with data assimilation ---
# icesee_model_data_assimilation(params, kwargs, modeling_params, enkf_params, physical_params)

#  repeat the process for the next time step

