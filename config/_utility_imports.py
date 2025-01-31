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
import matplotlib.pyplot as plt

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

# Insert the models directory at the beginning of sys.path
sys.path.insert(0, models_dir)
sys.path.insert(0, utils_dir)
sys.path.insert(0, run_model_da_dir)
sys.path.insert(0, config_loader_dir)


# import the necessary modules
from tools import save_arrays_to_h5, extract_datasets_from_h5, save_all_data
from utils import UtilsFunctions
from config_loader import load_yaml_to_dict, get_section


