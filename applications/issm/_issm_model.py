# ==============================================================================
# @des: This file contains run functions for ISSM model python wrapper.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2025-03-26
# @author: Brian Kyanjo
# ==============================================================================

# --- python imports ---
import sys
import os
import shutil
import numpy as np
from scipy.stats import multivariate_normal,norm

# --- Utility imports ---
sys.path.insert(0, '../../config')
from _utility_imports import icesee_get_index
from matlab2python.server_utils import run_icesee_with_server

# --- model initialization ---
def initialize_model(physical_params, modeling_params, comm):
    """ des: intialize the issm model
        - calls the issm initalize_model.m matlab function to initialize the model
    """
    import h5py
    import scipy.io as sio

    # --- copy intialize_model.m to the current directory
    shutil.copyfile(os.path.join(os.path.dirname(__file__), 'initialize_model.m'), 'initialize_model.m')

    icesee_rank = comm.Get_rank()
    icesee_size = comm.Get_size()

    # read the model kwargs from the file
    server      = modeling_params.get('server')
    icesee_path = modeling_params.get('icesee_path')
    data_path   = modeling_params.get('data_path')
    issm_cmd = f"run(\'issm_env\'); initialize_model({icesee_rank}, {icesee_size})"
    result = run_icesee_with_server(lambda: server.send_command(issm_cmd),server)
    if not result:
        sys.exit(1)
    
    # fetch model size from output file
    try: 
        output_filename = f'{icesee_path}{data_path}/ensemble_output_{icesee_rank}.h5'
        # print(f"[DEBUG] Attempting to open file: {output_filename}")
        if not os.path.exists(output_filename):
            print(f"[ERROR] File does not exist: {output_filename}")
            return None
        with h5py.File(output_filename, 'r') as f:
            Vx = f['Vx'][0]
            # get the size of the Vx variable
            nd = Vx.shape[0]
            # print(f"[DEBUG] Vx shape: {Vx.shape}")
            return nd
    except Exception as e:
        print(f"[DEBUG] Error reading the file: {e}")

        
    
# ---- ISSM model ----
def ISSM_model(**kwargs):
    """ des: run the issm model
        - calls the issm run_model.m matlab function to run the model
    """

    # --- get the number of processors ---
    nprocs = kwargs.get('nprocs')
    k = kwargs.get('k')
    dt = kwargs.get('dt')
    tinitial = kwargs.get('tinitial')
    tfinal = kwargs.get('tfinal')

    # --- copy run_model.m to the current directory
    shutil.copyfile(os.path.join(os.path.dirname(__file__), 'run_model.m'), 'run_model.m')

    # --- call the run_model.m function ---
    server = kwargs.get('server')
    cmd = f'run(\'issm_env\'); run_model({nprocs},{k},{dt},{tinitial},{tfinal})'
    result = run_icesee_with_server(lambda: server.send_command(cmd),server)
    if not result:
        sys.exit(1)

# ---- Run model for ISSM ----
def run_model(ensemble, **kwargs):
    """ des: run the issm model with ensemble matrix from ICESEE
        returns: dictionary of the output from the issm model
    """
    import h5py
    import numpy as np
    import os

    # --- get the number of processors ---
    nprocs = kwargs.get('nprocs')
    server              = kwargs.get('server')
    rank                = kwargs.get('rank')
    issm_examples_dir   = kwargs.get('issm_examples_dir')
    icesee_path         = kwargs.get('icesee_path')

    #  --- change directory to the issm directory ---
    os.chdir(issm_examples_dir)

    try: 
        # Generate output filename based on rank
        # input_filename = f'ensemble_output_{rank}.h5'
        icesee_path = kwargs.get('icesee_path')
        data_path = kwargs.get('data_path')
        input_filename = f'{icesee_path}{data_path}/ensemble_output_{rank}.h5'

        #  write the ensemble to the h5 file
        k = kwargs.get('k')

        # -- call teh icess_get_index function to get the index of the ensemble
        print(f"[DEBUG] Ensemble shape: {ensemble.shape}")
        print(f"[DEBUG] Ensemble: {ensemble[:5]}")
        vecs, indx_map, _ = icesee_get_index(ensemble, **kwargs)

        if k > 0:
            # -- create our ensemble for test purposes
            #  read from input file for now
            # with h5py.File(input_filename, 'r') as f:
            #     Vx = f['Vx'][:]
            #     Vy = f['Vy'][:]
            #     Vz = f['Vz'][:]
            #     Pressure = f['Pressure'][:]
            # ndim = ensemble.shape[0] // 4
            # Vx = ensemble[0:ndim]
            # Vy = ensemble[ndim:2*ndim]
            # Vz = ensemble[2*ndim:3*ndim]
            # Pressure = ensemble[3*ndim:4*ndim]
            Vx = ensemble[indx_map["Vx"]]
            Vy = ensemble[indx_map["Vy"]]
            Vz = ensemble[indx_map["Vz"]]
            Pressure = ensemble[indx_map["Pressure"]]
            # -> fetch the vx, vy, vz, and pressure from the ensemble
            
            # -----
            with h5py.File(input_filename, 'w') as f:
                # f.create_dataset('ensemble', data=ensemble)
                f.create_dataset('Vx', data=Vx)
                f.create_dataset('Vy', data=Vy)
                f.create_dataset('Vz', data=Vz)
                f.create_dataset('Pressure', data=Pressure)
            print(f"[HDF5] Saved: {input_filename}")

        # --- call the issm model  to update the state and parameters variables ---
        ISSM_model(**kwargs)

        # -- change directory back to the original directory
        os.chdir(icesee_path)
        
        #  --- read the output from the h5 file ISSM model ---
        try:
            # output_filename = f'ensemble_output_{rank}.h5'
            icesee_path = kwargs.get('icesee_path')
            data_path = kwargs.get('data_path')
            # output_filename = f'ensemble_output_{rank}.h5'
            output_filename = f'{icesee_path}{data_path}/ensemble_output_{rank}.h5'
            with h5py.File(output_filename, 'r') as f:
                # Read the data from the file
                # for key in f.keys():
                #     output_dic[key] = f[key][:]
                return {
                'Vx': f['Vx'][0],
                'Vy': f['Vy'][0],
                'Vz': f['Vz'][0],
                'Pressure': f['Pressure'][0]
                }

        except Exception as e:
            print(f"[DEBUG] Error reading the file: {e}")
            return None
        
    except Exception as e:
        print(f"[DEBUG] Error sending command: {e}")
    finally:
        try:
            server.shutdown()
            server.reset_terminal()
        except Exception as e:
            print(f"[DEBUG] Error shutting down server: {e}")
        sys.exit(1)
    
    # -- change directory back to the original directory
    os.chdir(icesee_path)
