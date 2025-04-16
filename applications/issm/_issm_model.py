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


# utility imports
from matlab2python.mat2py_utils import subprocess_cmd_run, MatlabServer

# ISSM_DIR = os.environ.get("ISSM_DIR")
# server = MatlabServer(ISSM_DIR)
# server.start()

# --- model initialization ---
def initialize_model(physical_params, modeling_params, comm):
    """ des: intialize the issm model
        - calls the issm initalize_model.m matlab function to initialize the model
    """
    import h5py
    import scipy.io as sio

    # --- copy intialize_model.m to the current directory
    shutil.copyfile(os.path.join(os.path.dirname(__file__), 'initialize_model.m'), 'initialize_model.m')
    # shutil.copyfile(os.path.join(os.path.dirname(__file__), 'matlab2python', 'matlab_server.m'), 'matlab_server.m')

    # --- call the initalize_model.m function ---
    # issm_cmd = (
    # f'matlab -nodisplay -nosplash -nodesktop '
    # f'-r "run(\'issm_env\'); initialize_model; exit"'
    # )
    # subprocess_cmd_run(issm_cmd, 0, 1)

    # read the model kwargs from the file
    server = modeling_params.get('server')
    try:
        issm_cmd = f"run(\'issm_env\'); initialize_model"
        if not server.send_command(issm_cmd):
            raise RuntimeError("Command failed.")
    except Exception as e:
        print(f"[DEBUG] Error sending command: {e}")
        server.shutdown()
        server.reset_terminal()
        sys.exit(1)

    # --- remove the copied file
    # os.remove('initialize_model.m')

    
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
    shutil.copyfile(os.path.join(os.path.dirname(__file__), 'matlab2python', 'matlab_server.m'), 'matlab_server.m')

    # --- call the run_model.m function ---
    # issm_cmd = (
    # f'matlab -nodisplay -nosplash -nodesktop '
    # f'-r "run(\'issm_env\'); run_model({nprocs},{k},{dt},{tinitial},{tfinal}); exit"'
    # )
    # subprocess_cmd_run(issm_cmd, nprocs, kwargs.get('verbose'))

    # -> server approach
    # issm_cmd = (
    #     f'run(\'issm_env\'); run_model({nprocs},{k},{dt},{tinitial},{tfinal})'
    # )
    server = kwargs.get('server')
    cmd = f'run(\'issm_env\'); run_model({nprocs},{k},{dt},{tinitial},{tfinal})'
    try:
        if not server.send_command(cmd):
            raise RuntimeError(f"Command at step {k} failed.")
    except Exception as e:
        print(f"[DEBUG] Error sending command: {e}")
        server.shutdown()
        server.reset_terminal()
        sys.exit(1)

    # try to run the matlab server
    # server.send_command(f"run_model({nprocs},{k},{dt},{tinitial},{tfinal})")

    # --- remove the copied file
    # os.remove('run_model.m')

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
    server = kwargs.get('server')
    # rank = kwargs.get('rank')
    rank = 0

    try: 
        # Generate output filename based on rank
        # input_filename = f'ensemble_output_{rank}.h5'
        icesee_path = kwargs.get('icesee_path')
        data_path = kwargs.get('data_path')
        input_filename = f'{icesee_path}{data_path}/ensemble_output_{rank}.h5'

        #  write the ensemble to the h5 file
        k = kwargs.get('k')

        if k > 0:
            # -- create our ensemble for test purposes
            #  read from input file for now
            # with h5py.File(input_filename, 'r') as f:
            #     Vx = f['Vx'][:]
            #     Vy = f['Vy'][:]
            #     Vz = f['Vz'][:]
            #     Pressure = f['Pressure'][:]
            ndim = ensemble.shape[0] // 4
            Vx = ensemble[:ndim]
            Vy = ensemble[ndim:2*ndim]
            Vz = ensemble[2*ndim:3*ndim]
            Pressure = ensemble[3*ndim:4*ndim]
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

        #  --- read the output from the h5 file ISSM model ---
        # output_dic = {}
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
        print(f"[DEBUG] Error in run_model: {e}")
        server.shutdown()
        server.reset_terminal()
        sys.exit(1)
    
    # return output_dic
