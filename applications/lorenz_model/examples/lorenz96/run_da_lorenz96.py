# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-01-13
# @description: Lorenz96 model with data assimilation
# =============================================================================

# --- Imports ---
import sys
import os
import numpy as np

# --- Configuration ---
sys.path.insert(0, '../../config')
from _utility_imports import *

# --- Utility Functions ---
from lorenz96_enkf import generate_true_state,  initialize_ensemble
from run_models_da import run_model_with_filter

# --- Load Parameters ---
# Load parameters from a YAML file
parameters_file = "params.yaml"
parameters = load_yaml_to_dict(parameters_file)

physical_params = get_section(parameters, "physical-parameters")
modeling_params = get_section(parameters, "modeling-parameters")
enkf_params     = get_section(parameters, "enkf-parameters")

# --- Prepare observations and ensemble parameters ---
# --- Ensemble Parameters ---
params = {
    "nt": int(float(modeling_params["num_years"])/float(modeling_params["dt"])),
    "dt": float(modeling_params["dt"]),
    "nd": int(float(enkf_params["num_state_vars"])),
    "num_state_vars": int(float(enkf_params["num_state_vars"])),
    "Nens": int(float(enkf_params["Nens"])),
    "number_obs_instants": int(float(enkf_params["number_obs_instants"])),
    "inflation_factor": float(enkf_params["inflation_factor"]),
    "sig_model": float(enkf_params["sig_model"]),
    "sig_Q": float(enkf_params["sig_Q"]),
    "freq_obs": float(enkf_params["freq_obs"]),
    "obs_max_time": int(float(enkf_params["obs_max_time"])),
    "obs_start_time": float(enkf_params["obs_start_time"]),
}

# --- model parameters ---
kwargs = { "dt":params["dt"], "seed":float(enkf_params["seed"]),
          "t":np.linspace(0, int(float(modeling_params["num_years"])), params["nt"] + 1), 
          "u0True": np.array([1,1,1]), "u0b": np.array([2.0,3.0,4.0]), 
          "sigma":float(physical_params["sigma"]), "beta":eval(physical_params["beta"]),
          "rho":float(physical_params["rho"]),
           "obs_index": (np.linspace(int(params["freq_obs"]/params["dt"]), \
                             int(params["obs_max_time"]/params["dt"]), int(params["number_obs_instants"]))).astype(int)
}

# --- observations parameters ---
sig_obs = np.zeros(params["nt"]+1)
for i in range(len(kwargs["obs_index"])):
    sig_obs[kwargs["obs_index"][i]] = float(enkf_params["sig_obs"])
params["sig_obs"] = sig_obs

# --- Generate True and Nurged States ---
print("Generating true state ...")
statevec_true = generate_true_state(
    np.zeros([params["nd"], params["nt"] + 1]), 
    params,  
    **kwargs  
)

# print("Generating nurged state ...")
# statevec_nurged = generate_nurged_state(
#     np.zeros([params["nd"], params["nt"] + 1]), 
#     params, 
#     **kwargs  
# )

# --- Synthetic Observations ---
print("Generating synthetic observations ...")
utils_funs = UtilsFunctions(params, statevec_true)
w = utils_funs._create_synthetic_observations(statevec_true,**kwargs)

# load data to be written to file
save_all_data(
    enkf_params=enkf_params,
    nofilter=True,
    t=kwargs["t"], 
    obs_max_time=np.array([params["obs_max_time"]]),
    obs_index=kwargs["obs_index"],
    statevec_true=statevec_true,
    w=w
)

# --- Initialize Ensemble ---
print("Initializing the ensemble ...")
statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full = initialize_ensemble(
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"]]),
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"], params["nt"] + 1]),
    params, **kwargs
)

# --- Run Data Assimilation ---
print("Running the model with Data Assimilation ...")

# Additional arguments for the EnKF
da_args = [
    enkf_params["parallel_flag"],
    params,
    np.eye(params["nd"]) * params["sig_Q"] ** 2,
    w,
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
save_all_data(
    enkf_params=enkf_params,
    statevec_ens_full=statevec_ens_full,
    statevec_ens_mean=statevec_ens_mean,
    statevec_bg=statevec_bg
)
