# ==============================================================================
# @des: This file contains run functions for any model with data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2024-11-4
# @author: Brian Kyanjo
# ==============================================================================
    
# --- Imports ---
from _utility_imports import *
from tqdm import tqdm 
from scipy.sparse import csr_matrix
from scipy.sparse import block_diag
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
import bigmpi4py as BM # BigMPI for large data transfer and communication
import gc # garbage collector to free up memory
import copy
import numexpr as ne # for fast numerical computations


# --- Add required paths ---
src_dir = os.path.join(project_root, 'src')
applications_dir = os.path.join(project_root, 'applications')
parallelization_dir = os.path.join(project_root, 'parallelization')
sys.path.insert(0, src_dir)
sys.path.insert(0, applications_dir)
sys.path.insert(0, parallelization_dir)

# class instance of the observation operator and its Jacobian
from utils import *
import re
from EnKF.python_enkf.EnKF import EnsembleKalmanFilter as EnKF
from supported_models import SupportedModels
from localization_func import localization

# ---- Run model with EnKF ----
def gaspari_cohn(r):
    """
    Gaspari-Cohn taper function for localization in EnKF.
    Defined for 0 <= r <= 2.
    """
    r = np.abs(r)
    taper = np.zeros_like(r)
    
    mask1 = (r >= 0) & (r <= 1)
    mask2 = (r > 1) & (r <= 2)

    taper[mask1] = (((-0.25 * r[mask1] + 0.5) * r[mask1] + 0.625) * r[mask1] - 5/3) * r[mask1]**2 + 1
    taper[mask2] = ((((1/12 * r[mask2] - 0.5) * r[mask2] + 0.625) * r[mask2] + 5/3) * r[mask2] - 5) * r[mask2]**2 + 4 - 2/(3 * r[mask2])

    return np.maximum(taper, 0)  # Ensure non-negative values

def run_model_with_filter(model=None, filter_type=None, *da_args, **model_kwargs): 
    """ General function to run any kind of model with the Ensemble Kalman Filter """

    # unpack the data assimilation arguments
    parallel_flag     = da_args[0]   # parallel flag
    params            = da_args[1]   # parameters
    Q_err             = da_args[2]   # process noise
    hu_obs            = da_args[3]   # observation vector
    ensemble_vec      = da_args[4]   # ensemble of model state
    statevec_bg       = da_args[5]   # background state
    ensemble_vec_mean = da_args[6]   # ensemble mean
    ensemble_vec_full = da_args[7]   # full ensemble
    commandlinerun    = da_args[8]   # run through the terminal

    nd, Nens = ensemble_vec.shape
    # print(f"Ensemble shape: {ensemble_vec.shape}")
    # take it False if KeyError: 'joint_estimation' is raised
    # if "joint_estimation" in model_kwargs and "localization_flag" in params:
    #     pass
    # else:
    #     model_kwargs["joint_estimation"] = False
    #     params["localization_flag"] = False

    if model_kwargs["joint_estimation"] or params["localization_flag"]:
        hdim = nd // (params["num_state_vars"] + params["num_param_vars"])
    else:
        hdim = nd // params["num_state_vars"]

    # call curently supported model Class
    model_module = SupportedModels(model=model).call_model()

    # Define filter flags
    EnKF_flag   = re.match(r"\AEnKF\Z", filter_type, re.IGNORECASE)
    DEnKF_flag  = re.match(r"\ADEnKF\Z", filter_type, re.IGNORECASE)
    EnRSKF_flag = re.match(r"\AEnRSKF\Z", filter_type, re.IGNORECASE)
    EnTKF_flag  = re.match(r"\AEnTKF\Z", filter_type, re.IGNORECASE)

    # get the grid points
    if params.get("localization_flag", False):
        #  for both localization and joint estimation
        # - apply Gaspari-Cohn localization to only state variables [h,u,v] in [h,u,v,smb]
        # - for parameters eg. smb and others, don't apply localization
        # if model_kwargs["joint_estimation"]:
        # get state variables indices
        num_state_vars = params["num_state_vars"]
        num_params = params["num_param_vars"]
        # get the the inital smb
        # smb_init = ensemble_vec[num_state_vars*hdim:,:]
        inflation_factor = params["inflation_factor"] #TODO: store this, for localization debuging
        
        if True:
            Lx, Ly = model_kwargs["Lx"], model_kwargs["Ly"]
            nx, ny = model_kwargs["nx"], model_kwargs["ny"]

            # --- call the localization function (with adaptive localization) ---
            state_size = params["total_state_param_vars"]*hdim
            adaptive_localization = False   
            if not adaptive_localization:
                x_points = np.linspace(0, model_kwargs["Lx"], model_kwargs["nx"]+1)
                y_points = np.linspace(0, model_kwargs["Ly"], model_kwargs["ny"]+1)
                grid_x, grid_y = np.meshgrid(x_points, y_points)

                grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

                # Adjust grid if n_points != nx * ny (interpolating for 425 points)
                n_points = hdim
                missing_rows = n_points - grid_points.shape[0]
                if missing_rows > 0:
                    last_row = grid_points[-1]  # Get the last available row
                    extrapolated_rows = np.tile(last_row, (missing_rows, 1))  # Repeat last row
                    grid_points = np.vstack([grid_points, extrapolated_rows])  # Append extrapolated rows

                dist_matrix = distance_matrix(grid_points, grid_points) 

                # Normalize distance matrix
                L = 2654
                r_matrix = dist_matrix / L
            else:
                loc_matrix = localization(Lx,Ly,nx, ny, hdim, params["total_state_param_vars"], Nens, state_size)

        # dx, dy = model_kwargs["Lx"] / model_kwargs["nx"], model_kwargs["Ly"] / model_kwargs["ny"]  # Grid spacing

        # # Create 2D grid
        # x_grid = np.arange(model_kwargs["nx"]) * dx
        # y_grid = np.arange(model_kwargs["nx"]) * dy
        # X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        

    # --- call the ICESEE mpi parallel manager ---
    if re.match(r"\AMPI_model\Z", parallel_flag, re.IGNORECASE):
        from mpi4py import MPI
        # from parallel_mpi.icesee_mpi_parallel_manager import icesee_mpi_parallelization
        from parallel_mpi.icesee_mpi_parallel_manager import ParallelManager
        # Nens = params["Nens"]
        # n_modeltasks = params["n_modeltasks"]
        # parallel_manager = icesee_mpi_parallelization(Nens=Nens, n_modeltasks=n_modeltasks, screen_output=0)
        # # --- model communicators ---
        # comm_model = parallel_manager.COMM_model
        # rank_model = parallel_manager.rank_model
        # size_model = parallel_manager.size_model
        # # print(f"Rank: {rank}")
        # # load balancing
        # ensemble_local,start_model,stop_model, subcomm = parallel_manager.ensembles_load_distribution(ensemble = ensemble_vec,comm = comm_model)

        # --- filter communicator ---
        # comm_filter = parallel_manager.COMM_filter
        # rank_filter = parallel_manager.rank_filter
        # size_filter = parallel_manager.size_filter
        # ensemble_filter_local,start_filter,stop_filter,comm_ = parallel_manager.ensembles_load_distribution(ensemble=ensemble_vec,comm=comm_filter)
        # ensemble_vec_local, start_vec, stop_vec = parallel_manager.state_vector_load_distribution(ensemble_vec, comm_filter)

        # --- icesee mpi parallel manager ---------------------------------------------------
        # --- ensemble load distribution ---
        rounds, color, sub_rank, sub_size, subcomm, subcomm_size, rank_world, size_world, comm_world, start, stop = ParallelManager().icesee_mpi_ens_distribution(params)
        if params["even_distribution"]:
            ensemble_local = copy.deepcopy(ensemble_vec[:,start:stop])
            
        # --- row vector load distribution ---   
        # local_rows, start_row, end_row = ParallelManager().icesee_mpi_row_distribution(ensemble_vec, params)

        parallel_manager = None # debugging flag for now
        
    else:
        parallel_manager = None

    # --- Initialize the EnKF class ---
    EnKFclass = EnKF(parameters=params, parallel_manager=parallel_manager, parallel_flag = parallel_flag)

    # tqdm progress bar
    if rank_world == 0:
        pbar = tqdm(total=params["nt"], desc=f"[ICESEE] Progress on {size_world} processors", position=0)

    # ==== Time loop =======================================================================================
    km = 0
    # radius = 2
    for k in range(params["nt"]):
        
        # background step
        # statevec_bg = model_module.background_step(k,statevec_bg, hdim, **model_kwargs)

        # save a copy of initial ensemble
        # ensemble_init = ensemble_vec.copy()

        if re.match(r"\AMPI_model\Z", parallel_flag, re.IGNORECASE):                                   
            
            # === Four approaches of forecast step mpi parallelization ===
            # --- case 1: Each forecast runs squentially using all available processors
            if params.get("sequential_run", False):
                ensemble_col_stack = []
                for ens in range(Nens):
                    comm_world.Barrier() # make sure all processors are in sync
                    ensemble_vec[:,ens] = model_module.forecast_step_single(ens=ens, ensemble=ensemble_vec, nd=nd, Q_err=Q_err, params=params, **model_kwargs)
                    comm_world.Barrier() # make sure all processors reach this point before moving on
                   
                    gathered_ensemble = comm_world.allgather(ensemble_vec[:,ens]) #TODO: check if Allgather works
            
                    if rank_world == 0:
                        # print(f"[Rank {rank_world}] Gathered shapes: {[arr.shape for arr in ens_all]}")
                        ensemble_stack = np.hstack(gathered_ensemble)
                        ensemble_col_stack.append(ensemble_stack)

                # transpose the ensemble column
                if rank_world == 0:
                    ens_T = np.array(ensemble_col_stack).T
                    # print(f"Ensemble column shape: {ens_T.shape}")
                    shape_ens = np.array(ens_T.shape, dtype=np.int32) # send shape info
                else:
                    shape_ens = np.empty(2, dtype=np.int32)

                # broadcast the shape to all processors
                comm_world.Bcast([shape_ens, MPI.INT], root=0)

                if rank_world != 0:
                    # if k == 0:
                    ens_T = np.empty(shape_ens, dtype=np.float64)

                # broadcast the ensemble to all processors
                comm_world.Bcast([ens_T, MPI.DOUBLE], root=0)
                # print(f"Rank: {rank_world}, Ensemble shape: {ens_T.shape}")

                # compute the ensemble mean
                # if k == 0: # only do this at the first time step
                #     # gather from all processors ensemble_vec_mean[:,k+1]
                #     gathered_ensemble_vec_mean = comm_world.allgather(ensemble_vec_mean[:,k])
                #     if rank_world == 0:
                #         # print(f"Ensemble mean shape: {[arr.shape for arr in gathered_ensemble_vec_mean]}")
                #         stack_ensemble_vec_mean = np.hstack(gathered_ensemble_vec_mean)
                #         ensemble_vec_mean = np.empty((shape_ens[0],params["nt"]+1), dtype=np.float64)
                #         ensemble_vec_mean[:,k] = np.mean(stack_ensemble_vec_mean, axis=1)
                #     else: 
                #         ensemble_vec_mean = np.empty((shape_ens[0],params["nt"]), dtype=np.float64)
                    
                #     # broadcast the ensemble mean to all processors
                #     comm_world.Bcast([ensemble_vec_mean, MPI.DOUBLE], root=0)
                #     print(f"Rank: {rank_world}, Ensemble mean shape: {ensemble_vec_mean.shape}") 

                ensemble_vec_mean[:,k+1] = np.mean(ens_T[:nd,:], axis=1)
                # ensemble_vec_mean[:,k+1] = ParallelManager().compute_mean(ens_T[:nd,:], comm_world)

                # Analysis step
                obs_index = model_kwargs["obs_index"]
                if (km < params["number_obs_instants"]) and (k+1 == obs_index[km]):
                #     local_ensemble_centered = ensemble_local -  np.mean(ensemble_local, axis=1).reshape(-1,1)  # Center data
                    if EnKF_flag or DEnKF_flag:
                        diff = ens_T[:nd,:] - ensemble_vec_mean[:,k+1].reshape(-1,1)
                        Cov_model = diff @ diff.T / (Nens - 1)
                    elif EnRSKF_flag or EnTKF_flag:
                        diff = ens_T[:nd,:] - ensemble_vec_mean[:,k+1].reshape(-1,1)
                        Cov_model = diff / (Nens - 1)
                
                    # localization
                    if params.get("localization_flag", False):
                        # try with the localization matrix
                        cutoff_distance = 6000

                        # rho = np.zeros_like(Cov_model)
                        rho = np.ones_like(Cov_model)
                        # for j in range(Cov_model.shape[0]):
                        #     for i in range(Cov_model.shape[1]):
                        #         rad_x = np.abs(X[j] - X[i])
                        #         rad_y = np.abs(Y[j] - Y[i])
                        #         rad = np.sqrt(rad_x**2 + rad_y**2)
                        #         rad = rad/cutoff_distance
                        #         rho[j,i] = gaspari_cohn(rad)

                        Cov_model = rho * Cov_model

                        # if EnKF_flag or DEnKF_flag:
                    analysis  = EnKF(Observation_vec=  UtilsFunctions(params, ensemble_vec).Obs_fun(hu_obs[:nd,km]), 
                                            Cov_obs=params["sig_obs"][k+1]**2 * np.eye(2*params["number_obs_instants"]+1), \
                                            Cov_model= Cov_model, \
                                            Observation_function=UtilsFunctions(params, ensemble_vec).Obs_fun, \
                                            Obs_Jacobian=UtilsFunctions(params, ensemble_vec).JObs_fun, \
                                            parameters=  params,\
                                            parallel_flag=   parallel_flag)
                    # compute the analysis ensemble
                    if EnKF_flag:
                        ens_T[:nd,:], Cov_model = analysis.EnKF_Analysis(ens_T[:nd,:])
                    elif DEnKF_flag:
                        ens_T[:nd,:], Cov_model = analysis.DEnKF_Analysis(ens_T[:nd,:])
                    elif EnRSKF_flag:
                        ens_T[:nd,:], Cov_model = analysis.EnRSKF_Analysis(ens_T[:nd,:])
                    elif EnTKF_flag:
                        ens_T[:nd,:], Cov_model = analysis.EnTKF_Analysis(ens_T[:nd,:])
                    else:
                        raise ValueError("Filter type not supported")
                    
                    # update the ensemble mean
                    ensemble_vec_mean[:,k+1] = np.mean(ens_T[:nd,:], axis=1)

                    # update observation index
                    km += 1

                    # inflate the ensemble
                    ens_T = UtilsFunctions(params, ens_T[:nd,:]).inflate_ensemble(in_place=True)
                            
                    # ensemble_vec = copy.deepcopy(ens_T[:nd,:])
                    ensemble_vec = ens_T[:,:]
            
                # save the ensemble
                ensemble_vec_full[:,:,k+1] = ensemble_vec[:nd,:]
                
                # before exiting the time loop, we have to gather data from all processors
                if k == params["nt"] - 1:
                    # we are interested in ensemble_vec_full, ensemble_vec_mean, statevec_bg
                    gathered_ens_vec_mean = comm_world.allgather(ensemble_vec_mean)
                    gathered_ens_vec_full = comm_world.allgather(ensemble_vec_full)
                    if rank_world == 0:
                        # print(f"Ensemble mean shape: {[arr.shape for arr in gathered_ens_vec_mean]}")
                        ensemble_vec_mean = np.vstack(gathered_ens_vec_mean)
                        ensemble_vec_full = np.vstack(gathered_ens_vec_full)
                        print(f"Ensemble mean shape: {ensemble_vec_mean.shape}")
                    else:
                        ensemble_vec_mean = np.empty((shape_ens[0],params["nt"]+1), dtype=np.float64)
                        ensemble_vec_full = np.empty((shape_ens[0],Nens,params["nt"]+1), dtype=np.float64)

            # ------------------------------------------------- end of case 1 -------------------------------------------------

            # --- cases 2 & 3 ---
            # case 2: Form batches of sub-communicators and distribute resources among them
            #          - only works for Nens >= size_world
            # case 3: form Nens sub-communicators and distribute resources among them
            #          - only works for size_world > Nens
            #          - even distribution and load balancing leading to performance improvement
            #          - best for size_world/Nens is a whole number
            if params["default_run"]:
                # --- case 2: Form batches of sub-communicators and distribute resources among them ---
                if Nens >= size_world:
                    # store results for each round
                    ens_list = []
                    for round_id in range(rounds):
                        ensemble_id = color + round_id * subcomm_size  # Global ensemble index

                        if ensemble_id < Nens:  # Only process valid ensembles
                            print(f"Rank {rank_world} processing ensemble {ensemble_id} in round {round_id + 1}/{rounds}")

                            # Ensure all ranks in the subcommunicator are synchronized before running
                            subcomm.Barrier()

                            # Call the forecast step function
                            ens = ensemble_id
                            ensemble_vec[:,ens] = model_module.forecast_step_single(ens=ens, ensemble=ensemble_vec, nd=nd, Q_err=Q_err, params=params, **model_kwargs)

                            # Ensure all ranks in the subcommunicator are synchronized before moving on
                            subcomm.Barrier()

                            # Gather results within each subcommunicator
                            gathered_ensemble = subcomm.allgather(ensemble_vec[:,ens])

                            # Ensure only rank = 0 in each subcommunicator gathers the results
                            if sub_rank == 0:
                                 gathered_ensemble = np.hstack(gathered_ensemble)

                            ens_list.append(gathered_ensemble if sub_rank == 0 else None)

                        # Gather results from all subcommunicators
                        gathered_ensemble_global = comm_world.allgather(ens_list)
                # --- case 3: Form Nens sub-communicators and distribute resources among them ---
                elif Nens < size_world:
                    # Ensure all ranks in subcomm are in sync 
                    subcomm.Barrier()

                    # Call the forecast step fucntion- Each subcomm runs the function indepadently
                    ens = color # each subcomm has a unique color
                    ensemble_vec[:,ens] = model_module.forecast_step_single(ens=ens, ensemble=ensemble_vec, nd=nd, Q_err=Q_err, params=params, **model_kwargs)

                    # Ensure all ranks synchronize before moving on
                    # subcomm.Barrier()

                    # Gather results within each subcomm
                    gathered_ensemble = subcomm.allgather(ensemble_vec[:,ens])

                    # Ensure only rank = 0 in each subcomm gathers the results
                    subcomm.Barrier()
                    if sub_rank == 0:
                        gathered_ensemble = np.hstack(gathered_ensemble)

                    # Gather results from all subcomms
                    gathered_ensemble_global = comm_world.allgather(gathered_ensemble)

                    # gather all observations
                    # hu_obs = comm_world.allgather(hu_obs)

                if rank_world == 0:
                    if size_world > Nens:
                        # print(f"Ensemble shape: {[arr.shape for arr in gathered_ensemble_global]}")
                        ensemble_vec = [arr for arr in gathered_ensemble_global if isinstance(arr, np.ndarray)]
                        hu_obs = [arr for arr in hu_obs if isinstance(arr, np.ndarray)]
                    else:
                        ensemble_vec = [arr for sublist in gathered_ensemble_global for arr in sublist if arr is not None]
                        hu_obs = [arr for sublist in hu_obs for arr in sublist if arr is not None]
                    ensemble_vec = np.column_stack(ensemble_vec)
                    # print(f"hu_obs type: {[ arr.shape for arr in hu_obs]}")
                    # hu_obs = np.hstack(hu_obs)
                    
                    shape_ens = np.array(ensemble_vec.shape, dtype=np.int32) # send shape info
                    if True:
                        # calculate the ensemble mean
                        # ensemble_vec_mean[:,k+1] = np.mean(ensemble_vec, axis=1)
                        obs_index = model_kwargs["obs_index"]
                        if (km < params["number_obs_instants"]) and (k+1 == obs_index[km]):
                            if EnKF_flag or DEnKF_flag:
                                diff = ensemble_vec - np.mean(ensemble_vec, axis=1).reshape(-1,1)
                                Cov_model = diff @ diff.T / (Nens - 1)
                            
                            # print(f"Rank: {rank_world} hu_obs shape: {hu_obs.shape}")
                            # print(f"Rank: {rank_world} hu_obs type: {type(hu_obs)}")
                            hu_obs = np.array(hu_obs)
                            print(f"Rank: {rank_world} hu_obs shape: {hu_obs.shape}")
                            analysis  = EnKF(Observation_vec=  UtilsFunctions(params, ensemble_vec).Obs_fun(hu_obs[:,km]), 
                                            Cov_obs=params["sig_obs"][k+1]**2 * np.eye(2*params["number_obs_instants"]+1), \
                                            Cov_model= Cov_model, \
                                            Observation_function=UtilsFunctions(params, ensemble_vec).Obs_fun, \
                                            Obs_Jacobian=UtilsFunctions(params, ensemble_vec).JObs_fun, \
                                            parameters=  params,\
                                            parallel_flag=   parallel_flag)
                            # compute the analysis ensemble
                            if EnKF_flag:
                                ensemble_vec, Cov_model = analysis.EnKF_Analysis(ensemble_vec)
                            
                            # inflate the ensemble
                            ensemble_vec = UtilsFunctions(params, ensemble_vec).inflate_ensemble(in_place=True)

                else:
                    shape_ens = np.empty(2, dtype=np.int32)

                # Step 1: Broadcast the shape to all processors
                comm_world.Bcast([shape_ens, MPI.INT], root=0)

                if rank_world != 0:
                    ensemble_vec = np.empty(shape_ens, dtype=np.float64)

                # broadcast the ensemble to all processors
                comm_world.Bcast([ensemble_vec, MPI.DOUBLE], root=0)

                # ensemble_vec= ensemble_vec[:nd,:] #TODO: this is a temporary fix

                # Step 2: Compute row-wise distribution (only needed on Rank 0)
                if False:
                    num_rows, num_cols = shape_ens
                    rows_per_rank = num_rows // size_world
                    extra = num_rows % size_world

                    if rank_world == 0:
                        send_counts = [(rows_per_rank + 1) * num_cols if rank < extra else rows_per_rank * num_cols for rank in range(size_world)]
                        displacements = [sum(send_counts[:rank]) for rank in range(size_world)]
                    else:
                        send_counts = None
                        displacements = None

                    # Step 3: Allocate space for local chunk
                    local_rows = rows_per_rank + 1 if rank_world < extra else rows_per_rank
                    local_ensemble = np.zeros((local_rows, num_cols), dtype=np.float64)

                    # Step 4: Scatter data row-wise
                    comm_world.Scatterv([ensemble_vec, send_counts, displacements, MPI.DOUBLE], local_ensemble, root=0)
                    # print(f"Rank {rank_world}: Received shape {local_ensemble.shape}")

                    # Step 5: Compute the local ensemble mean
                    ensemble_vec_mean_local = np.mean(local_ensemble, axis=1)

                    # Analysis step
                    obs_index = model_kwargs["obs_index"]
                    if (km < params["number_obs_instants"]) and (k+1 == obs_index[km]):
                        # --- compute covariance matrix based on the EnKF type ---
                        diff = local_ensemble -  ensemble_vec_mean_local.reshape(-1,1)
                        if EnKF_flag or DEnKF_flag:
                            Cov_model = diff @ diff.T / (Nens - 1)
                        elif EnRSKF_flag or EnTKF_flag:
                            Cov_model = diff / (Nens - 1)

                        #  --- localization
                        if params.get("localization_flag", False):
                            # try with the localization matrix
                            cutoff_distance = 6000

                            # rho = np.zeros_like(Cov_model)
                            rho = np.ones_like(Cov_model)
                            # for j in range(Cov_model.shape[0]):
                            #     for i in range(Cov_model.shape[1]):
                            #         rad_x = np.abs(X[j] - X[i])
                            #         rad_y = np.abs(Y[j] - Y[i])
                            #         rad = np.sqrt(rad_x**2 + rad_y**2)
                            #         rad = rad/cutoff_distance
                            #         rho[j,i] = gaspari_cohn(rad)

                            Cov_model = rho * Cov_model
                        # analysis step
                        shape_a = local_ensemble.shape[0]
                        analysis = EnKF(Observation_vec=  UtilsFunctions(params, ensemble_vec).Obs_fun(hu_obs[:shape_a,km]), 
                                        Cov_obs=params["sig_obs"][k+1]**2 * np.eye(2*params["number_obs_instants"]+1), \
                                        Cov_model= Cov_model, \
                                        Observation_function=UtilsFunctions(params, ensemble_vec).Obs_fun, \
                                        Obs_Jacobian=UtilsFunctions(params, ensemble_vec).JObs_fun, \
                                        parameters=  params,\
                                        parallel_flag=   parallel_flag)
                        
                        # compute the analysis ensemble
                        if EnKF_flag:
                            local_ensemble, Cov_model = analysis.EnKF_Analysis(local_ensemble)
                        elif DEnKF_flag:
                            local_ensemble, Cov_model = analysis.DEnKF_Analysis(local_ensemble)
                        elif EnRSKF_flag:
                            local_ensemble, Cov_model = analysis.EnRSKF_Analysis(local_ensemble)
                        elif EnTKF_flag:
                            local_ensemble, Cov_model = analysis.EnTKF_Analysis(local_ensemble)
                        else:
                            raise ValueError("Filter type not supported")
                        
                        # update the ensemble mean
                        ensemble_vec_mean_local = np.mean(local_ensemble, axis=1)

                        # inflate the ensemble
                        local_ensemble = UtilsFunctions(params, local_ensemble).inflate_ensemble(in_place=True)

                        # update the observation index
                        km += 1
                    
                    # Step 6: Gather data back to Rank 0
                    if rank_world == 0:
                            recv_counts = send_counts  # Receiving the same amount we originally sent
                            recv_displacements = displacements
                            gathered_vec = np.zeros((num_rows, num_cols), dtype=np.float64)  # Allocate space for all data
                    else:
                        recv_counts = None
                        recv_displacements = None
                        gathered_vec = None  # Only root stores the final result

                    # Step 7: Gather Data Back to Rank 0
                    comm_world.Gatherv(local_ensemble, [gathered_vec, recv_counts, recv_displacements, MPI.DOUBLE], root=0)

                    # Step 8: Rescatter Data to all processors of dimensions (nd, Nens)
                    ensemble_vec_ = np.empty((nd, Nens), dtype=np.float64)
                    comm_world.Scatterv(gathered_vec, ensemble_vec_, root=0)

                    # Step 9: Compute the ensemble mean per proc
                    ensemble_vec_mean[:,k+1] = np.mean(ensemble_vec_, axis=1)
                    # save the ensemble
                    ensemble_vec_full[:,:,k+1] = ensemble_vec_

                    # before exiting the time loop, we have to gather data from all processors
                    if k == params["nt"] - 1:
                        # we are interested in ensemble_vec_full, ensemble_vec_mean, statevec_bg
                        gathered_ens_vec_mean = comm_world.allgather(ensemble_vec_mean)
                        gathered_ens_vec_full = comm_world.allgather(ensemble_vec_full)
                        if rank_world == 0:
                            # print(f"Ensemble mean shape: {[arr.shape for arr in gathered_ens_vec_mean]}")
                            ensemble_vec_mean = np.vstack(gathered_ens_vec_mean)
                            ensemble_vec_full = np.vstack(gathered_ens_vec_full)
                            print(f"Ensemble mean shape: {ensemble_vec_mean.shape}")
                        else:
                            ensemble_vec_mean = np.empty((shape_ens[0],params["nt"]+1), dtype=np.float64)
                            ensemble_vec_full = np.empty((shape_ens[0],Nens,params["nt"]+1), dtype=np.float64)
                # exit()


                # Now we will gather all local ensembles from all processors
                # if k == params["nt"] - 1:
                #     if rank_world == 0:
                #         recv_counts = send_counts  # Receiving the same amount we originally sent
                #         recv_displacements = displacements
                #         gathered_ensemble = np.zeros((num_rows, num_cols), dtype=np.float64)  # Allocate space for all data
                #     else:
                #         recv_counts = None
                #         recv_displacements = None
                #         gathered_ensemble = None  # Only root stores the final result

                #     # Step 7: Gather Data Back to Rank 0
                #     comm_world.Gatherv(local_ensemble, [gathered_ensemble, recv_counts, recv_displacements, MPI.DOUBLE], root=0)

                #     # Step 8: Root Rank Outputs the Result
                #     if rank_world == 0:
                #         print(f"Rank 0: Gathered ensemble shape: {gathered_ensemble.shape}")


            # -------------------------------------------------- end of cases 2 & 3 --------------------------------------------

            # --- case 4: Evenly distribute ensemble members among processors 
            #         - each processor runs a subset of ensemble members
            #         - best for size_world/Nens is a whole number and Nens >= size_world
            #         - size_world = 2^n where n is an integer
            if params["even_distribution"]:
                # check if Nens is divisible by size_world and greater or equal to size_world
                if Nens >= size_world and Nens % size_world == 0:
                    for ens in range(ensemble_local.shape[1]):
                        ensemble_local[:, ens] = model_module.forecast_step_single(ens=ens, ensemble=ensemble_local, nd=nd, Q_err=Q_err, params=params, **model_kwargs)

                    # --- compute the ensemble mean ---
                    ensemble_vec_mean[:,k+1] = ParallelManager().compute_mean(ensemble_local, comm_world)

                    # --- gather all local ensembles from all processors to root---
                    gathered_ensemble = ParallelManager().gather_data(comm_world, ensemble_local, root=0)
                    if rank_world == 0:
                        ensemble_vec = np.hstack(gathered_ensemble)
                    else:
                        ensemble_vec = np.empty((nd, Nens), dtype=np.float64)

                    # Analysis step
                    obs_index = model_kwargs["obs_index"]
                    if (km < params["number_obs_instants"]) and (k+1 == obs_index[km]):
                
                        if rank_world == 0:
    
                            H = UtilsFunctions(params, ensemble_vec).JObs_fun(ensemble_vec.shape[0]) #TODO: maybe it should be Obs_fun instead of JObs_fun???
                            h = UtilsFunctions(params, ensemble_vec).Obs_fun # observation operator

                            # compute the observation covariance matrix
                            Cov_obs = params["sig_obs"][k+1]**2 * np.eye(2*params["number_obs_instants"]+1)

                            # --- vector of measurements
                            d = UtilsFunctions(params, ensemble_vec).Obs_fun(hu_obs[:,km])

                            if EnKF_flag:
                                Eta = np.zeros((d.shape[0], Nens)) # mxNens, ensemble pertubations
                                D   = np.zeros_like(Eta) # mxNens #virtual observations
                                HA  = np.zeros_like(D)
                                for ens in range(Nens):
                                    Eta[:,ens] = multivariate_normal.rvs(mean=np.zeros(d.shape[0]), cov=Cov_obs) 
                                    D[:,ens] = d + Eta[:,ens]
                                    HA[:,ens] = h(ensemble_vec[:,ens])
                                
                                # --- compute the innovations D` = D-HA
                                Dprime = D - HA # mxNens

                                # --- compute HAbar
                                HAbar = np.mean(HA, axis=1) # mx1
                                # --- compute HAprime
                                HAprime = HA - HAbar.reshape(-1,1) # mxNens (requires H to be linear)
                                
                                # Aprime = ensemble_vec@(np.eye(Nens) - one_N) # mxNens
                                # one_N = np.ones((Nens,Nens))/Nens
                                # HAprime= HA@(np.eye(Nens) - one_N) # mxNens

                                # get the min(m,Nens)
                                m_obs = d.shape[0]
                                nrmin = min(m_obs, Nens)

                                # --- compute HA' + eta
                                HAprime_eta = HAprime + Eta

                                # --- compute the SVD of HA' + eta
                                U, sig, _ = np.linalg.svd(HAprime_eta, full_matrices=False)

                                # --- convert s to eigenvalues
                                sig = sig**2
                                # for i in range(nrmin):
                                #     sig[i] = sig[i]**2
                                
                                # ---compute the number of significant eigenvalues
                                sigsum = np.sum(sig[:nrmin])  # Compute total sum of the first `nrmin` eigenvalues
                                sigsum1 = 0.0
                                nrsigma = 0

                                for i in range(nrmin):
                                    if sigsum1 / sigsum < 0.999:
                                        nrsigma += 1
                                        sigsum1 += sig[i]
                                        sig[i] = 1.0 / sig[i]  # Inverse of eigenvalue
                                    else:
                                        sig[i:nrmin] = 0.0  # Set remaining eigenvalues to 0
                                        break  # Exit the loop
                                
                                # compute X1 = sig*UT #Nens x m_obs
                                X1 = np.empty((nrmin, m_obs))
                                for j in range(m_obs):
                                    for i in range(nrmin):
                                        X1[i,j] =sig[i]*U[j,i]
                                
                                # compute X2 = X1*Dprime # Nens x Nens
                                X2 = np.dot(X1, Dprime)
                                del Cov_obs, sig, X1, Dprime; gc.collect()
                                
                                # print(f"Rank: {rank_world} X2 shape: {X2.shape}")
                                #  compute X3 = U*X2 # m_obs x Nens
                                X3 = np.dot(U, X2)

                                # print(f"Rank: {rank_world} X3 shape: {X3.shape}")
                                # compute X4 = (HAprime.T)*X3 # Nens x Nens
                                X4 = np.dot(HAprime.T, X3)
                                del X2, X3, U, HAprime; gc.collect()
                                
                                # print(f"Rank: {rank_world} X4 shape: {X4.shape}")
                                # compute X5 = X4 + I
                                X5 = X4 + np.eye(Nens)
                                del X4; gc.collect()

                                y_i = np.sum(X5, axis=1)
                                ensemble_vec_mean[:,k+1] = (1/Nens)*(ensemble_vec @ y_i.reshape(-1,1)).ravel()
                                
                        else:
                            X5 = np.empty((Nens, Nens))

                        # clean the memory
                        del ensemble_local, gathered_ensemble; gc.collect()

                        # Broadcast X5 to all processors
                        X5 = BM.bcast(X5, comm_world)
                        # the sum of each column of X5 should be equal to 1
                        # print(f"Rank: {rank_world} columns of X5: {bd_arr}")
                        # print(f"Rank: {rank_world} column sum of X5: {np.sum(X5, axis=0)}")

                        # --- scatter ensemble_vec to all processors ---
                        scatter_ensemble = BM.scatter(ensemble_vec, comm_world)

                        # do the ensemble analysis update: A_j = Fj*X5 
                        analysis_vec = np.dot(scatter_ensemble, X5)

                        # gather from all processors
                        ensemble_vec = BM.allgather(analysis_vec, comm_world)
    
                        # clean the memory
                        del scatter_ensemble, analysis_vec; gc.collect()

                        # update the ensemble with observations instants
                        km += 1

                        # inflate the ensemble
                        # params["inflation_factor"] = inflation_factor
                        ensemble_vec = UtilsFunctions(params, ensemble_vec).inflate_ensemble(in_place=True)
                        # ensemble_vec = UtilsFunctions(params, ensemble_vec)._inflate_ensemble()
                    
                        # update the local ensemble
                        ensemble_local = copy.deepcopy(ensemble_vec[:,start:stop])

                    # Save the ensemble
                    if rank_world == 0:
                        ensemble_vec_full[:,:,k+1] = ensemble_vec
                    else:
                        ensemble_vec_full = np.empty((nd, Nens, params["nt"]+1), dtype=np.float64)

                    # free up memory
                    del ensemble_vec; gc.collect()
                else:
                    raise ValueError("Nens must be divisible by size_world and greater or equal to size_world. size_world must be a power of 2")

            # -------------------------------------------------- end of case 4 -------------------------------------------------    
        #  ====== Serial run ======
        else:
            ensemble_vec = EnKFclass.forecast_step(ensemble_vec, \
                                               model_module.forecast_step_single, \
                                                Q_err, **model_kwargs)


            #  compute the ensemble mean
            ensemble_vec_mean[:,k+1] = np.mean(ensemble_vec, axis=1)

            # Analysis step
            obs_index = model_kwargs["obs_index"]
            if (km < params["number_obs_instants"]) and (k+1 == obs_index[km]):

                # Compute the model covariance
                diff = ensemble_vec - np.tile(ensemble_vec_mean[:,k+1].reshape(-1,1),Nens)
                if EnKF_flag or DEnKF_flag:
                    Cov_model = 1/(Nens-1) * diff @ diff.T
                elif EnRSKF_flag or EnTKF_flag:
                    Cov_model = 1/(Nens-1) * diff 

                # --- localization ---
                if params["localization_flag"]:
                    if not adaptive_localization:
                        # call the gahpari-cohn localization function
                        loc_matrix_spatial = gaspari_cohn(r_matrix)

                        # expand to full state space
                        loc_matrix = np.empty_like(Cov_model)
                        for var_i in range(params["total_state_param_vars"]):
                            for var_j in range(params["total_state_param_vars"]):
                                start_i, start_j = var_i * hdim, var_j * hdim
                                loc_matrix[start_i:start_i+hdim, start_j:start_j+hdim] = loc_matrix_spatial
                        
                        # apply the localization matrix
                        # Cov_model = loc_matrix * Cov_model
                        
                    Cov_model = loc_matrix * Cov_model

                    # inflate the top-left (smb h) and bottom-right (h smb) blocks of the covariance matrix 
                    state_block_size = num_state_vars*hdim
                    h_smb_block = Cov_model[:hdim,state_block_size:]
                    smb_h_block = Cov_model[state_block_size:,:hdim]

                    # apply the inflation factor
                    params["inflation_factor"] = 1.2
                    smb_h_block = UtilsFunctions(params, smb_h_block).inflate_ensemble(in_place=True)
                    h_smb_block = UtilsFunctions(params, h_smb_block).inflate_ensemble(in_place=True)

                    # update the covariance matrix
                    Cov_model[:hdim,state_block_size:] = h_smb_block
                    Cov_model[state_block_size:,:hdim] = smb_h_block

                # check if params["sig_obs"] is a scalar
                if isinstance(params["sig_obs"], (int, float)):
                    params["sig_obs"] = np.ones(params["nt"]+1) * params["sig_obs"]


                # Call the EnKF class for the analysis step
                analysis  = EnKF(Observation_vec=  UtilsFunctions(params, ensemble_vec).Obs_fun(hu_obs[:,km]), 
                                Cov_obs=params["sig_obs"][k+1]**2 * np.eye(2*params["number_obs_instants"]+1), \
                                Cov_model= Cov_model, \
                                Observation_function=UtilsFunctions(params, ensemble_vec).Obs_fun, \
                                Obs_Jacobian=UtilsFunctions(params, ensemble_vec).JObs_fun, \
                                parameters=  params,\
                                parallel_flag=   parallel_flag)
                
                # Compute the analysis ensemble
                if EnKF_flag:
                    ensemble_vec, Cov_model = analysis.EnKF_Analysis(ensemble_vec)
                elif DEnKF_flag:
                    ensemble_vec, Cov_model = analysis.DEnKF_Analysis(ensemble_vec)
                elif EnRSKF_flag:
                    ensemble_vec, Cov_model = analysis.EnRSKF_Analysis(ensemble_vec)
                elif EnTKF_flag:
                    ensemble_vec, Cov_model = analysis.EnTKF_Analysis(ensemble_vec)
                else:
                    raise ValueError("Filter type not supported")

                ensemble_vec_mean[:,k+1] = np.mean(ensemble_vec, axis=1)
                
                # update the ensemble with observations instants
                km += 1

                # inflate the ensemble
                ensemble_vec = UtilsFunctions(params, ensemble_vec).inflate_ensemble(in_place=True)
                # ensemble_vec = UtilsFunctions(params, ensemble_vec)._inflate_ensemble()
            
                # ensemble_vec_mean[:,k+1] = np.mean(ensemble_vec, axis=1)

            # Save the ensemble
            ensemble_vec_full[:,:,k+1] = ensemble_vec

        # update the progress bar
        if rank_world == 0:
            pbar.update(1)

    # close the progress bar
    if rank_world == 0:
        pbar.close()
    comm_world.Barrier()
    return ensemble_vec_full, ensemble_vec_mean, statevec_bg


