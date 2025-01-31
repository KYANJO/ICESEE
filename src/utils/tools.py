# ==============================================================================
# @des: This file contains helper functions that are used in the main script.
# @date: 2024-10-4
# @author: Brian Kyanjo
# ==============================================================================

import os
import sys
import re
import subprocess
import h5py
import numpy as np

# Function to safely change directory
def safe_chdir(main_directory,target_directory):
    # Get the absolute path of the target directory
    target_path = os.path.abspath(target_directory)

    # Check if the target path starts with the main directory path
    if target_path.startswith(main_directory):
        os.chdir(target_directory)
    # else:
    #     print(f"Error: Attempted to leave the main directory '{main_directory}'.")


def install_requirements(force_install=False, verbose=False):
    """
    Install dependencies listed in the requirements.txt file if not already installed,
    or if `force_install` is set to True.
    """
    # Check if the `.installed` file exists to determine if installation is needed
    if os.path.exists(".installed") and not force_install:
        print("Dependencies are already installed. Skipping installation.")
        return
    
    try:
        # Run the command to install the requirements from requirements.txt
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "../requirements.txt"])
        
        # Create a `.installed` marker file to indicate successful installation
        with open(".installed", "w") as f:
            f.write("Dependencies installed successfully.\n")

        print("All dependencies are installed and verified.")
    except subprocess.CalledProcessError as e:
        # Print the error and raise a more meaningful exception
        print(f"Error occurred while installing dependencies: {e}")
        raise RuntimeError("Failed to install dependencies from requirements.txt. Please check the file and try again.")

# ==== saves arrays to h5 file
def save_arrays_to_h5(filter_type=None, model=None, parallel_flag=None, commandlinerun=None, **datasets):
    """
    Save multiple arrays to an HDF5 file, optionally in a parallel environment (MPI).

    Parameters:
        filter_type (str): Type of filter used (e.g., 'ENEnKF', 'DEnKF').
        model (str): Name of the model (e.g., 'icepack').
        parallel_flag (str): Flag to indicate if MPI parallelism is enabled. Default is 'MPI'.
        commandlinerun (bool): Indicates if the function is triggered by a command-line run. Default is False.
        **datasets (dict): Keyword arguments where keys are dataset names and values are arrays to save.

    Returns:
        dict: The datasets if not running in parallel, else None.
    """
    output_dir = "results"
    output_file = f"{output_dir}/{filter_type}-{model}.h5"

    if parallel_flag == "MPI" or commandlinerun:
        # Create the results folder if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Creating results folder")

        # Remove the existing file, if any
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Existing file {output_file} removed.")

        print(f"Writing data to {output_file}")
        with h5py.File(output_file, "w") as f:
            for name, data in datasets.items():
                f.create_dataset(name, data=data, compression="gzip")
                print(f"Dataset '{name}' written to file")
        print(f"Data successfully written to {output_file}")
    else:
        print("Non-MPI or non-commandline run. Returning datasets.")
        return datasets

# Routine extracts datasets from a .h5 file
def extract_datasets_from_h5(file_path):
    """
    Extracts all datasets from an HDF5 file and returns them as a dictionary.

    Parameters:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary where keys are dataset names and values are numpy arrays.

    Raises:
        FileNotFoundError: If the specified HDF5 file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    datasets = {}
    print(f"Reading data from {file_path}...")

    with h5py.File(file_path, "r") as f:
        def extract_group(group, datasets):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    datasets[key] = np.array(item)
                    print(f"Dataset '{key}' extracted with shape {item.shape}")
                elif isinstance(item, h5py.Group):
                    extract_group(item, datasets)

        extract_group(f, datasets)

    print("Data extraction complete.")
    return datasets

# --- best for saving all data to h5 file in parallel environment
def save_all_data(enkf_params=None, nofilter=None, **kwargs):
    """
    General function to save datasets based on the provided parameters.
    """
    # Update filter_type only if nofilter is provided
    filter_type = "true-wrong" if nofilter else enkf_params["filter_type"]

    # --- Local MPI implementation ---
    if re.match(r"\AMPI\Z", enkf_params["parallel_flag"], re.IGNORECASE):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD  # Initialize MPI
        rank = comm.Get_rank()  # Get rank of current MPI process
        size = comm.Get_size()  # Get total number of MPI processes

        comm.Barrier()
        if rank == 0:
            save_arrays_to_h5(
                filter_type=filter_type,  # Use updated or original filter_type
                model=enkf_params["model_name"],
                parallel_flag=enkf_params["parallel_flag"],
                commandlinerun=enkf_params["commandlinerun"],
                **kwargs
            )
    else:
        save_arrays_to_h5(
            filter_type=filter_type,  # Use updated or original filter_type
            model=enkf_params["model_name"],
            parallel_flag=enkf_params["parallel_flag"],
            commandlinerun=enkf_params["commandlinerun"],
            **kwargs
        )

# --- using checkpointing to save icepack output ---
# def save_icepack_output(h_save, u_save, smb, params,kwargs):
            
