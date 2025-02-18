# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-11-06
# @description: Synthetic ice stream with data assimilation
# =============================================================================

# --- Imports ---
import sys
import os
import numpy as np
from mpi4py import MPI

from firedrake.petsc import PETSc

# --- Configuration ---
os.environ["PETSC_CONFIGURE_OPTIONS"] = "--download-mpich-device=ch3:sock"
os.environ["OMP_NUM_THREADS"] = "1"

# --- Utility imports ---
sys.path.insert(0, '../../../config')
from _utility_imports import *
applications_dir = os.path.join(project_root, 'applications','icepack_model')
sys.path.insert(0, applications_dir)

# --- Utility Functions ---
from _icepack_model import initialize_model
from run_models_da import run_model_with_filter
from _icepack_enkf import generate_true_state, generate_nurged_state, initialize_ensemble

# --- initialize parallelization ---
from parallel_mpi.icesee_mpi_parallel_manager import icesee_mpi_parallelization, ParallelManager
global_init = ParallelManager().init_parallel_non_mpi_model()
comm_init  =  global_init.COMM_model
size_init = comm_init.Get_size()
rank_init = comm_init.Get_rank()

if rank_init == 0:
    PETSc.Sys.Print("Fetching the model parameters ...")
        # --- Load Parameters ---
    # Load parameters from a YAML file
    parameters_file = "params.yaml"
    parameters = load_yaml_to_dict(parameters_file)

    physical_params = get_section(parameters, "physical-parameters")
    modeling_params = get_section(parameters, "modeling-parameters")
    enkf_params = get_section(parameters, "enkf-parameters")

    # --- Ensemble Parameters ---
    params = {
        "nt": int(float(modeling_params["num_years"])) * int(float(modeling_params["timesteps_per_year"])),
        "dt": 1.0 / float(modeling_params["timesteps_per_year"]),
        "num_state_vars": int(float(enkf_params["num_state_vars"])),
        "num_param_vars": int(float(enkf_params["num_param_vars"])),
        "Nens": int(float(enkf_params["Nens"])),
        "number_obs_instants": int(int(float(enkf_params["obs_max_time"]))/float(enkf_params["freq_obs"])),
        "inflation_factor": float(enkf_params["inflation_factor"]),
        "sig_model": float(enkf_params["sig_model"]),
        "sig_obs": float(enkf_params["sig_obs"]),
        "sig_Q": float(enkf_params["sig_Q"]),
        "freq_obs": float(enkf_params["freq_obs"]),
        "obs_max_time": int(float(enkf_params["obs_max_time"])),
        "obs_start_time": float(enkf_params["obs_start_time"]),
        "localization_flag": bool(enkf_params["localization_flag"]),
        "parallel_flag": enkf_params["parallel_flag"],
        "n_modeltasks": int(enkf_params["n_modeltasks"]),
        }
else :
    params, physical_params, modeling_params, enkf_params = None, None, None, None

#--- Broadcast the parameters ---
params, physical_params, modeling_params, enkf_params = comm_init.bcast([params, 
                                                                         physical_params, 
                                                                         modeling_params, 
                                                                         enkf_params], 
                                                                         root=0)


# === initialize ICESEE parallelization ===
Nens = params["Nens"]
n_modeltasks = params["n_modeltasks"]
parallel_manager = icesee_mpi_parallelization(Nens=Nens, n_modeltasks=n_modeltasks, screen_output=0)
comm = parallel_manager.COMM_model
rank = parallel_manager.rank_model
size = parallel_manager.size_model

    # --- Model intialization ---
PETSc.Sys.Print("Initializing icepack model ...")
model_comm = parallel_manager.COMM_model # Get the model communicator
nx,ny,Lx,Ly,x,y,h,u,a,a_p,b,b_in,b_out,h0,u0,solver_weertman,A,C,Q,V = initialize_model(
    physical_params, modeling_params,model_comm
)

# update the parameters
params["nd"]=  h0.dat.data.size * int(float(enkf_params["num_state_vars"])+ int(float(enkf_params["num_param_vars"])))

kwargs = {"a":a, "h0":h0, "u0":u0, "C":C, "A":A,"Q":Q,"V":V, "da":float(modeling_params["da"]),
        "b":b, "dt":params["dt"], "seed":float(enkf_params["seed"]), "x":x, "y":y,
        "t":np.linspace(0, int(float(modeling_params["num_years"])), params["nt"] + 1),
        "Lx":Lx, "Ly":Ly, "nx":nx, "ny":ny, "h_nurge_ic":float(enkf_params["h_nurge_ic"]), 
        "u_nurge_ic":float(enkf_params["u_nurge_ic"]),"nurged_entries":float(enkf_params["nurged_entries"]),
        "a_in_p":float(modeling_params["a_in_p"]), "da_p":float(modeling_params["da_p"]),
        "solver":solver_weertman,
        "obs_index": (np.linspace(int(params["freq_obs"]/params["dt"]), \
                        int(params["obs_max_time"]/params["dt"]), int(params["number_obs_instants"]))).astype(int),
        "joint_estimation": bool(enkf_params["joint_estimation"]),
        "parameter_estimation": bool(enkf_params["parameter_estimation"]),
        "state_estimation": bool(enkf_params["state_estimation"]),
}


obs_t, obs_idx, num_observations = UtilsFunctions(params).generate_observation_schedule(**kwargs)
print(obs_t)
kwargs["obs_index"] = obs_idx
params["number_obs_instants"] = num_observations

# --- observations parameters ---
sig_obs = np.zeros(params["nt"]+1)
for i in range(len(kwargs["obs_index"])):
    sig_obs[kwargs["obs_index"][i]] = float(enkf_params["sig_obs"])
params["sig_obs"] = sig_obs

# --- Generate True and Nurged States ---
# comm.barrier()
if rank == 0:
    PETSc.Sys.Print("Generating true state ...")
    statevec_true = generate_true_state(
        np.zeros([params["nd"], params["nt"] + 1]), 
        params,  
        **kwargs  
    )
else:
    # initialize data on all worker processes
    statevec_true = np.empty((params["nd"], params["nt"] + 1)) 
    
# comm.barrier() # wait for all processes to reach this point
comm.bcast(statevec_true, root=0)
# comm.barrier() # wait for all processes to reach this point

if rank == 0:
    PETSc.Sys.Print("Generating nurged state ...")
    kwargs["a"] = a_p # Update accumulation with nurged accumulation
    statevec_nurged = generate_nurged_state(
        np.zeros([params["nd"], params["nt"] + 1]), 
        params, 
        **kwargs  
    )
else:
    # initialize data on all worker processes
    statevec_nurged = np.empty((params["nd"], params["nt"] + 1))

comm.bcast(statevec_nurged, root=0)

# --- Synthetic Observations ---
if rank == 0:
    PETSc.Sys.Print("Generating synthetic observations ...")
    utils_funs = UtilsFunctions(params, statevec_true)
    hu_obs = utils_funs._create_synthetic_observations(statevec_true,**kwargs)
    shape = np.array(hu_obs.shape, dtype=np.int32) # send shape info
else:
    shape = np.empty(2, dtype=np.int32)

# broadcast the shape of the observations
comm.Bcast(shape, root=0)

if rank != 0:
    hu_obs = np.empty(shape, dtype=np.float64)

# broadcast the observations
comm.Bcast([hu_obs, MPI.DOUBLE], root=0)

# hdim = params["nd"] // (params["num_state_vars"] + params["num_param_vars"])
# state_block_size = hdim*params["num_state_vars"]
# #  don't observe the accumulation rate
# hu_obs[state_block_size:, :] = 0

    # load data to be written to file

# --- Initialize Ensemble ---
if rank == 0:
    PETSc.Sys.Print("Initializing the ensemble ...")
    statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full = initialize_ensemble(
        np.zeros([params["nd"], params["nt"] + 1]),
        np.zeros([params["nd"], params["Nens"]]),
        np.zeros([params["nd"], params["nt"] + 1]),
        np.zeros([params["nd"], params["Nens"], params["nt"] + 1]),
        params, **kwargs
    )
else:
    # initialize data on all worker processes
    statevec_bg = np.empty((params["nd"], params["nt"] + 1))
    statevec_ens = np.empty((params["nd"], params["Nens"]))
    statevec_ens_mean = np.empty((params["nd"], params["nt"] + 1))
    statevec_ens_full = np.empty((params["nd"], params["Nens"], params["nt"] + 1))

comm.Bcast([statevec_bg, MPI.DOUBLE], root=0)
comm.Bcast([statevec_ens, MPI.DOUBLE], root=0)
comm.Bcast([statevec_ens_mean, MPI.DOUBLE], root=0)
comm.Bcast([statevec_ens_full, MPI.DOUBLE], root=0)
    
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

statevec_ens_full, statevec_ens_mean, statevec_bg = run_model_with_filter(
    enkf_params["model_name"],
    enkf_params["filter_type"],
    *da_args,
    **kwargs  
)


# load data to be written to file
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
    pass