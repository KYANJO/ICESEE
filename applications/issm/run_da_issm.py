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
from matlab2python.mat2py_utils import  add_issm_dir_to_sys_path, MatlabServer

# --- Utility Functions ---
from _issm_model import initialize_model

# --- Initialize MPI ---
from parallel_mpi.icesee_mpi_parallel_manager import ParallelManager
icesee_rank, icesee_size, icesee_comm = ParallelManager().icesee_mpi_init(params)

# --- get current working directory ---
icesee_cwd = os.getcwd()

# --- change directory to issm model directory: make sure ISSM_DIR is set in the environment
issm_dir = os.environ.get('ISSM_DIR')  # make sure ISSM_DIR is set in the environment
add_issm_dir_to_sys_path(issm_dir)     # add the issm directory to the system path 

# --- make the examples directory available ---
issm_examples_dir = os.path.join(issm_dir, 'examples',kwargs.get('example_name'))

# --- fetch the modeling parameters ---
model_kwargs = {
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
                'icesee_path': icesee_cwd,
                'data_path': kwargs.get('data_path'),
                'issm_dir': issm_dir,
                'issm_examples_dir': issm_examples_dir,
                'rank': icesee_rank,
                'nprocs': icesee_size,
}

# --- save model kwargs to file and update Icesee kwargs ---
sio.savemat('model_kwargs.mat', model_kwargs)
kwargs.update(model_kwargs)

# copy the issm_env.m from icesee_cwd  file to the examples directory
shutil.copy(os.path.join(icesee_cwd,'matlab2python', 'issm_env.m'), issm_examples_dir)
shutil.copy(os.path.join(icesee_cwd,'matlab2python', 'matlab_server.m'), issm_examples_dir)
shutil.copy(os.path.join(icesee_cwd, 'model_kwargs.mat'), issm_examples_dir)

# --- change directory to the examples directory ---
os.chdir(issm_examples_dir)

# --- intialize the matlab server ---
server = MatlabServer(verbose=1)
server.launch() # start the server
modeling_params.update({'server': server})

if icesee_rank == 0:
    initialize_model(physical_params, modeling_params, icesee_comm)
else:
    pass
# wait for rank 0 to write to file before proceeding
icesee_comm.Barrier()

# --- change directory back to the original directory ---
os.chdir(icesee_cwd)

from _issm_enkf import forecast_step_single, initialize_ensemble


print(f"[DEBUG] current working directory: {os.getcwd()}")
# time = np.linspace(0,80,20)
dt = 4
time = np.arange(0, 13, dt)
tinitial = 0
Nens = 2

kwargs.update({'t': time})

try:
    kwargs.update({'server': server})

    # -- initialize the ensemble members --
    rank =0
    # output_filename = f'ensemble_output_{rank}.h5'
    ndim = 4500
    ensemble = np.zeros((ndim*4, Nens))
    for ens in range(Nens):
        ensemble_dic = initialize_ensemble(ens, **kwargs)
        ensemble[:, ens] = np.concatenate((ensemble_dic['Vx'], ensemble_dic['Vy'], ensemble_dic['Vz'], ensemble_dic['Pressure']))
        noise = np.random.normal(0, 0.1, ensemble[:, ens].shape)
        ensemble[:, ens] += noise
            

    for k in range(len(time)-1):
        kwargs.update({'k':k})
        # kwargs.update({'dt':dt})
        # kwargs.update({'tinitial': time[k]})
        # kwargs.update({'tfinal': time[k+1]})
        print(f"\n[DEBUG] Running the model from time: {time[k]} to {time[k+1]}\n")
        for ens in range(Nens):
            print(f"[DEBUG] Running ensemble member: {ens}")
            forecast_step_single(ensemble[:,ens], **kwargs)

    # shutdown the matlab server
    server.shutdown()
    server.reset_terminal()
    print(f"[DEBUG] current working directory: {os.getcwd()}")
    sys.exit(1)

except RuntimeError as e:
    print(f"[Laucher] Error: {e}")
    server.shutdown()
    server.reset_terminal()
    sys.exit(1)


# go back to the original directory
# os.chdir(icesee_cwd)
print(f"[DEBUG] current working directory: {os.getcwd()}")

# --- Run the ISSM model with data assimilation ---
# icesee_model_data_assimilation(params, kwargs, modeling_params, enkf_params, physical_params)

#  repeat the process for the next time step

