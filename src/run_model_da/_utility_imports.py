# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-01-13
# @description: Lorenz96 model with data assimilation
# =============================================================================

# --- Imports ---
import sys
import os
import ast
import tqdm
import h5py
import json
import argparse
import numpy as np
from scipy.stats import multivariate_normal,norm

def get_project_root():
    """Automatically determines the root of the project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of the current script
    
    # Traverse upwards until we reach the root of the project (assuming 'src' folder exists at root)
    while not os.path.exists(os.path.join(current_dir, 'src')):
        current_dir = os.path.dirname(current_dir)  # Move one level up
    
    return current_dir

# Get the root of the project
project_root = get_project_root()
