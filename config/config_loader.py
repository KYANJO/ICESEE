# =============================================================================
# src/project_name/config_loader.py
# @author: Brian Kyanjo
# @date: 2025-01-10
# @description: This file is used to load the parameters from the YAML file
# =============================================================================

# Import the required modules
import yaml
import os

def load_yaml_to_dict(file_path):
    """
    Load a YAML file and store its contents in a dictionary.

    Parameters:
        file_path (str): Path to the YAML file.

    Returns:
        dict: A dictionary containing the parsed YAML entries.
    """
    # Check if the file exists before attempting to read it
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    try:
        with open(file_path, "r") as yaml_file:
            # Use safe_load for security and robust parsing
            return yaml.safe_load(yaml_file)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{file_path}': {e}")


def get_section(data, section_name):
    """
    Safely retrieve a section from the loaded YAML dictionary.

    Parameters:
        data (dict): Dictionary containing YAML data.
        section_name (str): Section name to retrieve.

    Returns:
        dict: The requested section as a dictionary or an empty dictionary if not found.
    """
    return data.get(section_name, {})

