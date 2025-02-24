# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-01-13
# @description: Lorenz96 model with data assimilation
# =============================================================================

# --- Imports ---
import os
import sys
import h5py
import numpy as np
import warnings
from scipy.stats import norm, multivariate_normal
from tqdm import tqdm
import yaml
from argparse import ArgumentParser

# Suppress warnings
warnings.filterwarnings("ignore")

def get_project_root():
    """Automatically determines the root of the project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of the current script
    
    # Traverse upwards until we reach the root of the project (assuming 'src' folder exists at root)
    while not os.path.exists(os.path.join(current_dir, 'src')):
        current_dir = os.path.dirname(current_dir)  # Move one level up
    
    return current_dir

# Get the root of the project
project_root = get_project_root()

# Construct the path to 'src/models' from the project root
models_dir = os.path.join(project_root, 'src', 'models')
utils_dir = os.path.join(project_root, 'src', 'utils')
run_model_da_dir = os.path.join(project_root, 'src', 'run_model_da')
config_loader_dir = os.path.join(project_root, 'config')
applications_dir = os.path.join(project_root, 'applications')
parallelization_dir = os.path.join(project_root, 'src', 'parallelization')

# Insert the models directory at the beginning of sys.path
sys.path.insert(0, models_dir)
sys.path.insert(0, utils_dir)
sys.path.insert(0, run_model_da_dir)
sys.path.insert(0, config_loader_dir)
sys.path.insert(0, parallelization_dir)

# import the necessary modules
from tools import save_arrays_to_h5, extract_datasets_from_h5, save_all_data
from utils import UtilsFunctions
from config_loader import load_yaml_to_dict, get_section

# --- CL args.
# Mapping for execution mode
execution_modes_str = {
    "default_run": 0,
    "sequential_run": 1,
    "even_distribution": 2
}
execution_modes_int = {v: k for k, v in execution_modes_str.items()}  # Reverse mapping

# CL args.
parser = ArgumentParser(description='ICESEE: Ice Sheet Parameter and State Estimation model')
parser.add_argument('--Nens', type=int, required=True, help='ensemble members')
parser.add_argument('--verbose', action='store_true', help='verbose output')
parser.add_argument('--default_run', action='store_true', help='default run')
parser.add_argument('--sequential_run', action='store_true', help='sequential run')
parser.add_argument('--even_distribution', action='store_true', help='even distribution')
parser.add_argument('execution_mode', type=int, choices=[0, 1, 2], nargs='?', help='Execution mode: 0=default_run, 1=sequential_run, 2=even_distribution')

args = parser.parse_args()

# Determine execution mode
selected_mode = "default_run"  # Default mode

if args.execution_mode is not None:
    selected_mode = execution_modes_int[args.execution_mode]  # Convert int to string
else:
    for mode in execution_modes_str.keys():
        if getattr(args, mode):
            selected_mode = mode
            break

# Set flags explicitly
args.default_run = (selected_mode == "default_run")
args.sequential_run = (selected_mode == "sequential_run")
args.even_distribution = (selected_mode == "even_distribution")

# Create params dictionary
params = {
    "Nens": int(args.Nens),
    "default_run": args.default_run,
    "sequential_run": args.sequential_run,
    "even_distribution": args.even_distribution,
}

# print(f"Execution mode selected: {selected_mode}")
# print(f"Params: {params}")

# Load parameters from a YAML file
parameters_file = "params.yaml"
parameters = load_yaml_to_dict(parameters_file)

physical_params = get_section(parameters, "physical-parameters")
modeling_params = get_section(parameters, "modeling-parameters")
enkf_params = get_section(parameters, "enkf-parameters")
