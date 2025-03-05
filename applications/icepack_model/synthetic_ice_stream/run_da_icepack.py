# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-11-06
# @description: Synthetic ice stream with data assimilation
# =============================================================================

# --- Imports ---
import sys
import os
import numpy as np

# --- Configuration ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PETSC_CONFIGURE_OPTIONS"] = "--download-mpich-device=ch3:sock"

# from mpi4py import MPI
from firedrake.petsc import PETSc

# --- Utility imports ---
sys.path.insert(0, '../../../config')
from _utility_imports import * # contains params, kwargs, modeling_params, enkf_params, physical_params 
applications_dir = os.path.join(project_root, 'applications','icepack_model')
sys.path.insert(0, applications_dir)

# --- Utility Functions ---
from _icepack_model import initialize_model
from run_models_da import icesee_model_data_assimilation

# --- Initialize MPI ---
from parallel_mpi.icesee_mpi_parallel_manager import ParallelManager
rank, size, comm = ParallelManager().icesee_mpi_init(params)

PETSc.Sys.Print("Fetching the model parameters ...")

# --- Ensemble Parameters ---
params.update({
"nt": int(float(modeling_params["num_years"])) * int(float(modeling_params["timesteps_per_year"])),
"dt": 1.0 / float(modeling_params["timesteps_per_year"])
})

# --- Model intialization --- 
PETSc.Sys.Print("Initializing icepack model ...")
nx,ny,Lx,Ly,x,y,h,u,a,a_p,b,b_in,b_out,h0,u0,solver_weertman,A,C,Q,V = initialize_model(
    physical_params, modeling_params,comm
)

# update the parameters
params["nd"]=  h0.dat.data.size * params["total_state_param_vars"] # get the size of the entire vector
kwargs.update({"a":a, "h0":h0, "u0":u0, "C":C, "A":A,"Q":Q,"V":V, "da":float(modeling_params["da"]),
        "b":b, "dt":params["dt"], "seed":float(enkf_params["seed"]), "x":x, "y":y,
        "Lx":Lx, "Ly":Ly, "nx":nx, "ny":ny, "h_nurge_ic":float(enkf_params["h_nurge_ic"]), 
        "u_nurge_ic":float(enkf_params["u_nurge_ic"]),"nurged_entries":float(enkf_params["nurged_entries"]),
        "a_in_p":float(modeling_params["a_in_p"]), "da_p":float(modeling_params["da_p"]),
        "solver":solver_weertman,
        "a_p":a_p, "b_in":b_in, "b_out":b_out,
})

# --- observations parameters ---
sig_obs = np.zeros(params["nt"]+1)
for i in range(len(kwargs["obs_index"])):
    sig_obs[kwargs["obs_index"][i]] = params["sig_obs"]
params["sig_obs"] = sig_obs


# --- Run Data Assimilation ---
ndim = params["nd"] // (params["num_state_vars"] + params["num_param_vars"])
Q_err = np.eye(ndim*params["num_state_vars"]) * params["sig_Q"] ** 2 #process noise covariance matrix

# Additional arguments for the EnKF
kwargs.update({"Q_err": Q_err,
              "params": params,
              })

PETSc.Sys.Print("Data assimilation with ICESEE ...")
icesee_model_data_assimilation(
    enkf_params["model_name"],
    enkf_params["filter_type"],
    **kwargs  
)



