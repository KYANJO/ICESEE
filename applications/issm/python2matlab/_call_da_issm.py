# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-01-13
# @description: This a helper function that calls the ISSM model with Data Assimilation using a Python Wrapper.
# =============================================================================

# --- Imports ---
import sys
import os
import numpy as np

# --- Configuration ---
sys.path.insert(0, '../../../config')
# from _utility_imports import *

# --- utility functions ---


# --- Load Parameters ---
# Load parameters from a YAML file
# parameters_file = "params.yaml"
# parameters = load_yaml_to_dict(parameters_file)

# physical_params = get_section(parameters, "physical-parameters")
# modeling_params = get_section(parameters, "modeling-parameters")
# enkf_params = get_section(parameters, "enkf-parameters")