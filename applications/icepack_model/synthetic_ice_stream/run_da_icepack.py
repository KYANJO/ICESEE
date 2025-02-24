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
from _utility_imports import *
applications_dir = os.path.join(project_root, 'applications','icepack_model')
sys.path.insert(0, applications_dir)

# --- Utility Functions ---
from _icepack_model import initialize_model
from run_models_da import run_model_with_filter
from _icepack_enkf import generate_true_state, generate_nurged_state, initialize_ensemble

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
})

# --- observations parameters ---
sig_obs = np.zeros(params["nt"]+1)
for i in range(len(kwargs["obs_index"])):
    sig_obs[kwargs["obs_index"][i]] = params["sig_obs"]
params["sig_obs"] = sig_obs

# --- Generate True and Nurged States ---
PETSc.Sys.Print("Generating true state ...")
statevec_true = generate_true_state(
    np.zeros([params["nd"], params["nt"] + 1]), 
    params,  
    **kwargs  
)


PETSc.Sys.Print("Generating nurged state ...")
kwargs["a"] = a_p # Update accumulation with nurged accumulation
statevec_nurged = generate_nurged_state(
    np.zeros([params["nd"], params["nt"] + 1]), 
    params, 
    **kwargs  
)

# --- Synthetic Observations ---
PETSc.Sys.Print("Generating synthetic observations ...")
utils_funs = UtilsFunctions(params, statevec_true)
hu_obs = utils_funs._create_synthetic_observations(statevec_true,**kwargs)


PETSc.Sys.Print("Initializing the ensemble ...")
statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full = initialize_ensemble(
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"]]),
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"], params["nt"] + 1]),
    params, **kwargs
)

# --- Run Data Assimilation ---
PETSc.Sys.Print("Running the model with Data Assimilation ...")
ndim = params["nd"] // (params["num_state_vars"] + params["num_param_vars"])
Q_err = np.eye(ndim*params["num_state_vars"]) * params["sig_Q"] ** 2
# Q_err = np.eye(params["nd"]) * params["sig_Q"] ** 2

# Additional arguments for the EnKF
da_args = [
    enkf_params["parallel_flag"],
    params,
    Q_err,
    hu_obs,
    statevec_ens,
    statevec_bg,
    statevec_ens_mean,
    statevec_ens_full,
    enkf_params["commandlinerun"],
]

if True:
    statevec_ens_full, statevec_ens_mean, statevec_bg = run_model_with_filter(
        enkf_params["model_name"],
        enkf_params["filter_type"],
        *da_args,
        **kwargs  
    )

    # load data to be written to file
    comm.Barrier()
    if rank == 0:
        PETSc.Sys.Print("Saving data ...")
        save_all_data(
            enkf_params=enkf_params,
            nofilter=True,
            t=kwargs["t"], b_io=np.array([b_in,b_out]),
            Lxy=np.array([Lx,Ly]),nxy=np.array([nx,ny]),
            statevec_true=statevec_true,
            statevec_nurged=statevec_nurged, 
            obs_max_time=np.array([params["obs_max_time"]]),
            obs_index=kwargs["obs_index"],
            w=hu_obs
        )

        # --- Save final data ---
        save_all_data(
            enkf_params=enkf_params,
            statevec_ens_full=statevec_ens_full,
            statevec_ens_mean=statevec_ens_mean,
            statevec_bg=statevec_bg
        )
    else:
        None
