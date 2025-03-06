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
import re
import numexpr as ne # for fast numerical computations


# --- Add required paths ---
src_dir             = os.path.join(project_root, 'src')               # source files directory
applications_dir    = os.path.join(project_root, 'applications')      # applications directory
parallelization_dir = os.path.join(project_root, 'parallelization')   # parallelization directory
sys.path.insert(0, src_dir)                  # add the source directory to the path
sys.path.insert(0, applications_dir)         # add the applications directory to the path
sys.path.insert(0, parallelization_dir)      # add the parallelization directory to the path

# class instance of the observation operator and its Jacobian
from utils import *                                                # utility functions for the model
from EnKF.python_enkf.EnKF import EnsembleKalmanFilter as EnKF     # Ensemble Kalman Filter
from supported_models import SupportedModels                       # supported models for data assimilation routine
from localization_func import localization                         # localization function for EnKF

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

# ======================== Run model with EnKF ========================
def icesee_model_data_assimilation(model=None, filter_type=None, **model_kwargs): 
    """ General function to run any kind of model with the Ensemble Kalman Filter """

    # --- unpack the data assimilation arguments
    parallel_flag     = model_kwargs.get("parallel_flag",False)      # parallel flag
    params            = model_kwargs.get("params",None)              # parameters
    Q_err             = model_kwargs.get("Q_err",None)               # process noise
    commandlinerun    = model_kwargs.get("commandlinerun",None)      # run through the terminal
    Lx, Ly            = model_kwargs["Lx"], model_kwargs["Ly"]
    nx, ny            = model_kwargs["nx"], model_kwargs["ny"]
    b_in, b_out       = model_kwargs["b_in"], model_kwargs["b_out"]

    # --- call curently supported model Class
    model_module = SupportedModels(model=model).call_model()

    # --- call the ICESEE mpi parallel manager ---
    if re.match(r"\AMPI_model\Z", parallel_flag, re.IGNORECASE):
        from mpi4py import MPI
        from parallel_mpi.icesee_mpi_parallel_manager import ParallelManager
        from mpi_analysis_functions import analysis_enkf_update, EnKF_X5

        # --- icesee mpi parallel manager ---------------------------------------------------
        # --- ensemble load distribution --
        rounds, color, sub_rank, sub_size, subcomm, subcomm_size, rank_world, size_world, comm_world, start, stop = ParallelManager().icesee_mpi_ens_distribution(params)

        # --- initialize seed for reproducibility ---
        ParallelManager().initialize_seed(comm_world, base_seed=0)

        # --- Generate True and Nurged States ---
        if params["even_distribution"] or (params["default_run"] and size_world < params["Nens"]):
            if rank_world == 0:
                print("Generating true state ...")
                ensemble_true_state = model_module.generate_true_state(
                   statevec_true = np.zeros([params["nd"], params["nt"] + 1]),  
                    **model_kwargs  
                )

                print("Generating nurged state ...")
                ensemble_nurged_state = model_module.generate_nurged_state(
                    statevec_nurged = np.zeros([params["nd"], params["nt"] + 1]), 
                    **model_kwargs  
                )
            else:
                ensemble_true_state = np.empty((params["nd"],params["nt"]+1),dtype=np.float64)
                ensemble_nurged_state = np.empty((params["nd"],params["nt"]+1),dtype=np.float64)
            # Bcast the true and nurged states
            comm_world.Bcast(ensemble_true_state, root=0)
            comm_world.Bcast(ensemble_nurged_state, root=0)
        else:
            # --- Generate True and Nurged States ---
            print("Generating true state ...")
            ensemble_true_state = model_module.generate_true_state(
                statevec_ture = np.zeros([params["nd"], params["nt"] + 1]), 
                **model_kwargs  
            )

            print("Generating nurged state ...")
            ensemble_nurged_state = model_module.generate_nurged_state(
                statevec_nurged = np.zeros([params["nd"], params["nt"] + 1]), 
                **model_kwargs  
            )

        # --- Synthetic Observations ---
        if params["even_distribution"] or (params["default_run"] and size_world < params["Nens"]):
            if rank_world == 0:
                # --- Synthetic Observations ---
                print("Generating synthetic observations ...")
                utils_funs = UtilsFunctions(params, ensemble_true_state)
                hu_obs = utils_funs._create_synthetic_observations(ensemble_true_state,**model_kwargs)
            else:
                hu_obs = np.empty((params["nd"],params["number_obs_instants"]),dtype=np.float64)

            hu_obs = comm_world.bcast(hu_obs, root=0)
        else:
            # gather the true and nurged states from all the processors
            ensemble_true_state = comm_world.gather(ensemble_true_state)
            ensemble_nurged_state = comm_world.gather(ensemble_nurged_state)
            if rank_world == 0:
                ensemble_true_state = [arr for arr in ensemble_true_state if isinstance(arr,np.ndarray)]
                ensemble_nurged_state = [arr for arr in ensemble_nurged_state if isinstance(arr,np.ndarray)]
                # print(f"{[arr.shape for arr in ensemble_true_state]}")
                ensemble_true_state = np.vstack(ensemble_true_state)
                ensemble_nurged_state = np.vstack(ensemble_nurged_state)
                shape_ = np.array(ensemble_true_state.shape,dtype=np.int32)
            else:
                shape_ = np.empty(2,dtype=np.int32)

            shape_ = comm_world.bcast(shape_, root=0)

            # --- Synthetic Observations ---
            if rank_world == 0:
                print("Generating synthetic observations ...")
                utils_funs = UtilsFunctions(params, ensemble_true_state)
                hu_obs = utils_funs._create_synthetic_observations(ensemble_true_state,**model_kwargs)
                # print(f"Rank [{rank}] Shape of the observations: {hu_obs.shape}")
            else:
                hu_obs = np.empty((shape_[0],params["number_obs_instants"]),dtype=np.float64)

            hu_obs = comm_world.bcast(hu_obs, root=0)
            print(f"Shape of the observations: {hu_obs.shape}")

            if rank_world != 0:
                ensemble_true_state = np.empty(shape_,dtype=np.float64)
                ensemble_nurged_state = np.empty(shape_,dtype=np.float64)

            # Bcast the true and nurged states
            comm_world.Bcast(ensemble_true_state, root=0)
            comm_world.Bcast(ensemble_nurged_state, root=0)

        # --- Initialize the ensemble ---
        if params["even_distribution"] or (params["default_run"] and size_world < params["Nens"]):
            if rank_world == 0:
                print("Initializing the ensemble ...")
                ensemble_bg, ensemble_vec, ensemble_vec_mean, ensemble_vec_full = model_module.initialize_ensemble(
                    np.zeros([params["nd"], params["nt"] + 1]),
                    np.zeros([params["nd"], params["Nens"]]),
                    np.zeros([params["nd"], params["nt"] + 1]),
                    np.zeros([params["nd"], params["Nens"], params["nt"] + 1]),
                    **model_kwargs
                )
            else:
                ensemble_bg = np.empty((params["nd"],params["nt"]+1),dtype=np.float64)
                ensemble_vec = np.empty((params["nd"],params["Nens"]),dtype=np.float64)
                ensemble_vec_mean = np.empty((params["nd"],params["nt"]+1),dtype=np.float64)
                ensemble_vec_full = np.empty((params["nd"],params["Nens"],params["nt"]+1),dtype=np.float64)

            # Bcast the ensemble
            comm_world.Bcast(ensemble_bg, root=0)
            comm_world.Bcast(ensemble_vec, root=0)
            comm_world.Bcast(ensemble_vec_mean, root=0)
            comm_world.Bcast(ensemble_vec_full, root=0)
            # print("using this ***************")
        else:
            print("Initializing the ensemble ...")
            ensemble_bg, ensemble_vec, ensemble_vec_mean, ensemble_vec_full = model_module.initialize_ensemble(
                np.zeros([params["nd"], params["nt"] + 1]),
                np.zeros([params["nd"], params["Nens"]]),
                np.zeros([params["nd"], params["nt"] + 1]),
                np.zeros([params["nd"], params["Nens"], params["nt"] + 1]),
                **model_kwargs
            )

        # print(f"Rank [{rank_world}] Shape of the ensemble: {ensemble_vec[:5,:]}")
        # --- get the ensemble size
        nd, Nens = ensemble_vec.shape

        if params["even_distribution"]:
            ensemble_local = copy.deepcopy(ensemble_vec[:,start:stop])
                
        # --- row vector load distribution ---   
        # local_rows, start_row, end_row = ParallelManager().icesee_mpi_row_distribution(ensemble_vec, params)

        parallel_manager = None # debugging flag for now
        
    else:
        parallel_manager = None

        # --- get the ensemble size
        nd, Nens = ensemble_vec.shape

    # --- hdim based on nd ---
    if model_kwargs["joint_estimation"] or params["localization_flag"]:
        hdim = nd // (params["num_state_vars"] + params["num_param_vars"])
    else:
        hdim = nd // params["num_state_vars"]

    # --- Define filter flags
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
        # ensemble_bg = model_module.background_step(k,ensemble_bg, hdim, **model_kwargs)

        # save a copy of initial ensemble
        # ensemble_init = ensemble_vec.copy()

        if re.match(r"\AMPI_model\Z", parallel_flag, re.IGNORECASE):                                   
            
            # === Four approaches of forecast step mpi parallelization ===
            # --- case 1: Each forecast runs squentially using all available processors
            if params.get("sequential_run", False):
                ensemble_col_stack = []
                for ens in range(Nens):
                    comm_world.Barrier() # make sure all processors are in sync
                    ensemble_vec[:,ens] = model_module.forecast_step_single(ens=ens, ensemble=ensemble_vec, nd=nd,  **model_kwargs)
                    comm_world.Barrier() # make sure all processors reach this point before moving on
                   
                    # gather the ensemble from all processors to rank 0
                    gathered_ensemble = ParallelManager().gather_data(comm_world, ensemble_vec, root=0)
                    if rank_world == 0:
                        # print(f"[Rank {rank_world}] Gathered shapes: {[arr.shape for arr in ens_all]}")
                        ensemble_stack = np.hstack(gathered_ensemble)
                        # print(f"Ensemble stack shape: {ensemble_stack.shape}")
                        ensemble_col_stack.append(ensemble_stack)
                
                # transpose the ensemble column
                if rank_world == 0:
                    ens_T = np.array(ensemble_col_stack).T
                    print(f"Ensemble column shape: {ens_T.shape}")
                    shape_ens = np.array(ens_T.shape, dtype=np.int32) # send shape info
                else:
                    shape_ens = np.empty(2, dtype=np.int32)
                exit()
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
                    # we are interested in ensemble_vec_full, ensemble_vec_mean, ensemble_bg
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
                            # print(f"Rank {rank_world} processing ensemble {ensemble_id} in round {round_id + 1}/{rounds}")

                            # Ensure all ranks in the subcommunicator are synchronized before running
                            subcomm.Barrier()

                            # Call the forecast step function
                            ens = ensemble_id
                            ensemble_vec[:,ens] = model_module.forecast_step_single(ens=ens, ensemble=ensemble_vec, nd=nd, **model_kwargs)

                            # Ensure all ranks in the subcommunicator are synchronized before moving on
                            # subcomm.Barrier()

                            # Gather results within each subcommunicator
                            gathered_ensemble = ParallelManager().gather_data(subcomm, copy.deepcopy(ensemble_vec[:,ens]), root=0)
                            # gathered_ensemble = subcomm.gather(ensemble_vec[:,ens], root=0)

                            # Ensure only rank = 0 in each subcommunicator gathers the results
                            if sub_rank == 0:
                                 gathered_ensemble = np.hstack(gathered_ensemble)

                            ens_list.append(gathered_ensemble if sub_rank == 0 else None)

                        # Gather results from all subcommunicators
                        gathered_ensemble_global = ParallelManager().gather_data(comm_world, ens_list, root=0)
                        # gathered_ensemble_global = comm_world.gather(ens_list, root=0)
                        #  free up memory
                        del gathered_ensemble; gc.collect()

                # --- case 3: Form Nens sub-communicators and distribute resources among them ---
                elif Nens < size_world:
                    # Ensure all ranks in subcomm are in sync 
                    subcomm.Barrier()

                    # Call the forecast step fucntion- Each subcomm runs the function indepadently
                    ens = color # each subcomm has a unique color
                    ensemble_vec[:,ens] = model_module.forecast_step_single(ens=ens, ensemble=ensemble_vec, nd=nd, **model_kwargs)

                    # Ensure all ranks synchronize before moving on
                    # subcomm.Barrier()

                    # Gather results within each subcomm
                    gathered_ensemble = subcomm.gather(ensemble_vec[:,ens], root=0)
                    # gathered_ensemble = ParallelManager().gather_data(subcomm, ensemble_vec, root=0)

                    # Ensure only rank = 0 in each subcomm gathers the results
                    subcomm.Barrier()
                    if sub_rank == 0:
                        # print(f"subcomm shapes: {[array.shape for array in gathered_ensemble]} for each color {color}")
                        gathered_ensemble = np.hstack(gathered_ensemble)

                    # Gather results from all subcomms
                    # gathered_ensemble_global = ParallelManager().gather_data(comm_world, gathered_ensemble, root=0)
                    gathered_ensemble_global = comm_world.gather(gathered_ensemble, root=0)

                    # gather all observations
                    # hu_obs = comm_world.allgather(hu_obs)

                    # free up memory
                    del gathered_ensemble; gc.collect()

                if rank_world == 0:
                    if size_world > Nens:
                        # print(f"Ensemble shape: {[arr.shape for arr in gathered_ensemble_global if isinstance(arr, np.ndarray)]}")
                        ensemble_vec = [arr for arr in gathered_ensemble_global if isinstance(arr, np.ndarray)]
                        # hu_obs = [arr for arr in hu_obs if isinstance(arr, np.ndarray)]
                    else:
                        # print(f"Ensemble shape: {[arr.shape for arr in gathered_ensemble_global]}")
                        ensemble_vec = [arr for sublist in gathered_ensemble_global for arr in sublist if arr is not None]
                        # hu_obs = [arr for sublist in hu_obs for arr in sublist if isinstance(arr, np.ndarray)]
                    ensemble_vec = np.column_stack(ensemble_vec)

                    # get the shape of the ensemble
                    shape_ens = np.array(ensemble_vec.shape, dtype=np.int32)
                else:
                    shape_ens = np.empty(2, dtype=np.int32)

                # broadcast the shape to all processors
                comm_world.Bcast([shape_ens, MPI.INT], root=0)

                # --- compute the mean
                ens_mean = ParallelManager().compute_mean_matrix_from_root(ensemble_vec, shape_ens[0], Nens, comm_world, root=0)

                # Analysis step
                obs_index = model_kwargs["obs_index"]
                if (km < params["number_obs_instants"]) and (k+1 == obs_index[km]):
            
                    if rank_world == 0:

                        H = UtilsFunctions(params, ensemble_vec).JObs_fun(ensemble_vec.shape[0]) #TODO: maybe it should be Obs_fun instead of JObs_fun???
                        h = UtilsFunctions(params, ensemble_vec).Obs_fun # observation operator

                        # compute the observation covariance matrix
                        Cov_obs = params["sig_obs"][k+1]**2 * np.eye(2*params["number_obs_instants"]+1)

                        # --- vector of measurements
                        # print(f"hu_obs shape: {hu_obs.shape}")
                        d = UtilsFunctions(params, ensemble_vec).Obs_fun(hu_obs[:,km])

                        if EnKF_flag:
                            # compute the X5 matrix
                            X5 = EnKF_X5(ensemble_vec, Cov_obs, Nens, h, d)
                            y_i = np.sum(X5, axis=1)
                            ensemble_vec_mean[:,k+1] = (1/Nens)*(ensemble_vec @ y_i.reshape(-1,1)).ravel()
                            
                    else:
                        X5 = np.empty((Nens, Nens))

                    # call the analysis update function
                    ensemble_vec = analysis_enkf_update(ensemble_vec, shape_ens, X5, comm_world)
                
                    # update the observation index
                    km += 1

                    # inflate the ensemble
                    ensemble_vec = UtilsFunctions(params, ensemble_vec).inflate_ensemble(in_place=True)

                    ensemble_vec = copy.deepcopy(ensemble_vec[:,:])

                # save the ensemble    
                if rank_world == 0:
                    ensemble_vec_full[:,:,k+1] = ensemble_vec[:,:]     
                    ensemble_vec_mean[:,k+1]   = ens_mean      
                else:
                    ensemble_vec_full = np.empty((shape_ens[0],Nens,params["nt"]+1), dtype=np.float64)
                    ensemble_vec_mean = np.empty((shape_ens[0],params["nt"]+1), dtype=np.float64)

                # free up memory
                del gathered_ensemble_global; gc.collect()

            # -------------------------------------------------- end of cases 2 & 3 --------------------------------------------

            # --- case 4: Evenly distribute ensemble members among processors 
            #         - each processor runs a subset of ensemble members
            #         - best for size_world/Nens is a whole number and Nens >= size_world
            #         - size_world = 2^n where n is an integer
            if params["even_distribution"]:
                # check if Nens is divisible by size_world and greater or equal to size_world
                if Nens >= size_world and Nens % size_world == 0:
                    for ens in range(ensemble_local.shape[1]):
                        ensemble_local[:, ens] = model_module.forecast_step_single(ens=ens, ensemble=ensemble_local, nd=nd, **model_kwargs)

                    # --- compute the ensemble mean ---
                    ensemble_vec_mean[:,k+1] = ParallelManager().compute_mean_from_local_matrix(ensemble_local, comm_world)

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
                                # compute the X5 matrix
                                X5 = EnKF_X5(ensemble_vec, Cov_obs, Nens, h, d)
                                y_i = np.sum(X5, axis=1)
                                ensemble_vec_mean[:,k+1] = (1/Nens)*(ensemble_vec @ y_i.reshape(-1,1)).ravel()
                                
                        else:
                            X5 = np.empty((Nens, Nens))

                        # clean the memory
                        del ensemble_local, gathered_ensemble; gc.collect()

                        # call the analysis update function
                        shape_ens = ensemble_vec.shape # get the shape of the ensemble
                        ensemble_vec = analysis_enkf_update(ensemble_vec, shape_ens, X5, comm_world)

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
    # comm_world.Barrier()

    # ====== load data to be written to file ======
    # print("Saving data ...")
    save_all_data(
        enkf_params=enkf_params,
        nofilter=True,
        t=kwargs["t"], b_io=np.array([b_in,b_out]),
        Lxy=np.array([Lx,Ly]),nxy=np.array([nx,ny]),
        ensemble_true_state=ensemble_true_state,
        ensemble_nurged_state=ensemble_nurged_state, 
        obs_max_time=np.array([params["obs_max_time"]]),
        obs_index=kwargs["obs_index"],
        w=hu_obs
    )

    # --- Save final data ---
    save_all_data(
        enkf_params=enkf_params,
        ensemble_vec_full=ensemble_vec_full,
        ensemble_vec_mean=ensemble_vec_mean,
        ensemble_bg=ensemble_bg
    )
    # return ensemble_vec_full, ensemble_vec_mean, ensemble_bg
    return None


