# ==============================================================================
# @des: This file contains run functions for any model with data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2024-11-4
# @author: Brian Kyanjo
# ==============================================================================
    
# --- Imports ---
from _utility_imports import *
import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse import block_diag
from scipy.ndimage import zoom
import copy

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

# ---- Run model with EnKF ----
# @njit
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
    EnKF_flag = re.match(r"\AEnKF\Z", filter_type, re.IGNORECASE)
    DEnKF_flag = re.match(r"\ADEnKF\Z", filter_type, re.IGNORECASE)
    EnRSKF_flag = re.match(r"\AEnRSKF\Z", filter_type, re.IGNORECASE)
    EnTKF_flag = re.match(r"\AEnTKF\Z", filter_type, re.IGNORECASE)

    # get the grid points
    if params["localization_flag"]:
        #  for both localization and joint estimation
        # - apply Gaspari-Cohn localization to only state variables [h,u,v] in [h,u,v,smb]
        # - for parameters eg. smb and others, don't apply localization
        # if model_kwargs["joint_estimation"]:
        # get state variables indices
        num_state_vars = params["num_state_vars"]
        num_params = params["num_param_vars"]
        # get the the inital smb
        smb_init = ensemble_vec[num_state_vars*hdim:,:]
        
        x_points = np.linspace(0, model_kwargs["Lx"], model_kwargs["nx"])
        y_points = np.linspace(0, model_kwargs["Ly"], model_kwargs["ny"])
        # use nd instead of nx and ny
        # x_points = np.linspace(0, model_kwargs["Lx"], nd)
        # y_points = np.linspace(0, model_kwargs["Ly"], nd)
        grid_x, grid_y = np.meshgrid(x_points, y_points)

    # --- call the ICESEE mpi parallel manager ---
    if re.match(r"\AMPI_model\Z", parallel_flag, re.IGNORECASE):
        from mpi4py import MPI
        from parallel_mpi.icesee_mpi_parallel_manager import icesee_mpi_parallelization
        Nens = params["Nens"]
        n_modeltasks = params["n_modeltasks"]
        parallel_manager = icesee_mpi_parallelization(Nens=Nens, n_modeltasks=n_modeltasks, screen_output=0)
        comm = parallel_manager.COMM_model
        rank = parallel_manager.rank_model
        size = parallel_manager.size_model
        # print(f"Rank: {rank}")
        # load balancing
        ensemble_local,start,stop = parallel_manager.load_balancing(ensemble_vec,comm)
        
    else:
        parallel_manager = None

    # --- Initialize the EnKF class ---
    EnKFclass = EnKF(parameters=params, parallel_manager=parallel_manager, parallel_flag = parallel_flag)

    km = 0
    # radius = 2
    for k in tqdm.trange(params["nt"]):
        # background step
        # statevec_bg = model_module.background_step(k,statevec_bg, hdim, **model_kwargs)

        # save a copy of initial ensemble
        # ensemble_init = ensemble_vec.copy()

        if re.match(r"\AMPI_model\Z", parallel_flag, re.IGNORECASE):                                   
            # print(f"\nranks: {rank}, size: {size}\n")
            for ens in range(ensemble_local.shape[1]):
                    ensemble_local[:, ens] = model_module.forecast_step_single(ens=ens, ensemble=ensemble_local, nd=nd, Q_err=Q_err, params=params, **model_kwargs)

            
            # comm.Barrier()

            # if parallel_manager.memory_usage(nd,Nens,8) > 8:
            if True:
                
                # --- gather local ensembles from all processors ---
                gathered_ensemble = parallel_manager.all_gather_data(comm, ensemble_local)
                if rank == 0:
                    ensemble_vec = np.hstack(gathered_ensemble)
                else:
                    ensemble_vec = np.empty((nd, Nens), dtype=np.float64)

                # --- broadcast the ensemble to all processors ---
                ensemble_vec = parallel_manager.broadcast_data(comm, ensemble_vec, root=0)

                # compute the global ensemble mean
                ensemble_vec_mean[:,k+1] = parallel_manager.compute_mean(ensemble_local, comm)

                # Analysis step
                obs_index = model_kwargs["obs_index"]
                if (km < params["number_obs_instants"]) and (k+1 == obs_index[km]):
                    # --- compute covariance matrix based on the EnKF type ---
                    # local_ensemble_centered = ensemble_local -  ensemble_vec_mean[:,k+1]  # Center data
                    local_ensemble_centered = ensemble_local -  np.mean(ensemble_local, axis=1).reshape(-1,1)  # Center data
                    if EnKF_flag or DEnKF_flag:
                        local_cov = local_ensemble_centered @ local_ensemble_centered.T / (Nens - 1)
                        Cov_model = np.zeros_like(local_cov)
                        comm.Allreduce([local_cov, MPI.DOUBLE], [Cov_model, MPI.DOUBLE], op=MPI.SUM)
                    elif EnRSKF_flag or EnTKF_flag:
                        Cov_model = np.zeros_like(local_ensemble_centered)
                        comm.Allreduce([local_ensemble_centered, MPI.DOUBLE], [Cov_model, MPI.DOUBLE], op=MPI.SUM)

                    # method 3
                    if params["localization_flag"]:
                        
                        # sate block size
                        # ---------------------------------------------
                        state_block_size = num_state_vars*hdim
                        # radius = 1.5
                        radius = UtilsFunctions(params, ensemble_vec[:state_block_size,:]).compute_adaptive_localization_radius(grid_x, grid_y, base_radius=1.5, method='variance')
                        localization_weights = UtilsFunctions(params, ensemble_vec[:state_block_size,:]).create_tapering_matrix(grid_x, grid_y, radius)
                        # ---------------------------------------------
                        # radius = UtilsFunctions(params, ensemble_vec[:,:]).compute_adaptive_localization_radius(grid_x, grid_y, base_radius=2.0, method='correlation')
                        # localization_weights = UtilsFunctions(params, ensemble_vec[:,:]).create_tapering_matrix(grid_x, grid_y, radius)
                        if EnKF_flag or DEnKF_flag:
                            # ------------------------------
                            localization_weights_resized = np.eye(Cov_model[:state_block_size,:state_block_size].shape[0])
                            localization_weights_resized[:localization_weights.shape[0], :localization_weights.shape[1]] = localization_weights
                            Cov_model[:state_block_size, :state_block_size] *= localization_weights_resized 

                            # check if maximum value of smb is greater than 1.25*smb_obs 
                            smb = ensemble_vec[state_block_size:,:]
                            smb_crit = 1.05*np.max(np.abs(hu_obs[state_block_size:,km]))
                            smb_crit2 = np.max(Cov_model[567:,567:])
                            smb_cov = np.cov(smb_init)
                            smb_flag1 = smb_crit < np.max(np.abs(smb))
                            smb_flag2 = smb_crit2 > 1.02*np.max(smb_cov)
                            if smb_flag2:
                                # force the smb to be 5% 0f the smb_obs
                                # t = model_kwargs["t"]
                                # ensemble_vec[state_block_size:,:] = np.min(smb_init, smb_init + (smb-smb_init)*t[k]/(t[params["nt"]-1] - t[0]))
                                ensemble_vec[state_block_size:,:] = smb_init
                        elif EnRSKF_flag or EnTKF_flag:
                            localization_weights_resized = np.eye(Cov_model[:state_block_size, :].shape[0])
                            print("localization_weights:", localization_weights)
                            localization_weights_resized[:localization_weights.shape[0], :Nens] = localization_weights
                            Cov_model[:state_block_size, :] *= localization_weights_resized

                    # Call the EnKF class for the analysis step
                    mpi_start = MPI.Wtime()

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

                    mpi_stop = MPI.Wtime()

                    # print(f"Rank: {rank}, Time taken for analysis step: {mpi_stop - mpi_start}")
                    # get total time taken for the analysis step
                    total_time = comm.reduce(mpi_stop - mpi_start, op=MPI.SUM, root=0)
                    if rank == 0:
                        print(f"Total time taken for analysis step: {total_time}")


                    # update the ensemble with observations instants
                    km += 1

                    # inflate the ensemble
                    ensemble_vec = UtilsFunctions(params, ensemble_vec).inflate_ensemble(in_place=True)
                    # ensemble_vec = UtilsFunctions(params, ensemble_vec)._inflate_ensemble()
                
                    # update the local ensemble
                    ensemble_local = copy.deepcopy(ensemble_vec[:,start:stop])
                    
                    # update ensemble

                # Save the ensemble
                ensemble_vec_full[:,:,k+1] = ensemble_vec

            # else:
            #     gathered_ensemble = parallel_manager.all_gather_data(comm, ensemble_local)
                
            #     if rank == 0:
            #         ensemble_vec = np.hstack(gathered_ensemble)
            #         ensemble_vec_mean[:,k+1] = np.mean(ensemble_vec, axis=1)
            #     else:
            #         ensemble_vec = np.empty((nd, Nens), dtype=np.float64)
            #         ensemble_vec_mean = np.empty((nd, params["nt"]+1), dtype=np.float64)

            #     ensemble_vec = parallel_manager.broadcast_data(comm, ensemble_vec, root=0)
            #     ensemble_vec_mean = parallel_manager.broadcast_data(comm, ensemble_vec_mean, root=0)

            
                
         
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

                print(f"[DEBUG] diff shape: {diff.shape}") # Debugging
                print(f"[Debug] ensemble_vec_mean shape: {ensemble_vec_mean[:,k+1].shape}") # Debugging
                print(f"[DEBUG] Cov_model max: {np.max(Cov_model[567:,567:])}") # Debugging

                # check if params["sig_obs"] is a scalar
                if isinstance(params["sig_obs"], (int, float)):
                    params["sig_obs"] = np.ones(params["nt"]+1) * params["sig_obs"]

                # --- Addaptive localization
                # compute the distance between observation and the ensemble members
                # dist = np.linalg.norm(ensemble_vec - np.tile(hu_obs[:,km].reshape(-1,1),N), axis=1)
                    
                # get the localization weights
                # localization_weights = UtilsFunctions(params, ensemble_vec)._adaptive_localization(dist, \
                                                    # ensemble_init=ensemble_init, loc_type="Gaspari-Cohn")

                # method 2
                # get the cut off distance between grid points
                # cutoff_distance = np.linspace(0, 5, Cov_model.shape[0])
                # localization_weights = UtilsFunctions(params, ensemble_vec)._adaptive_localization_v2(cutoff_distance)
                # print(localization_weights)
                # get the shur product of the covariance matrix and the localization matrix
                # Cov_model = np.multiply(Cov_model, localization_weights)

                # method 3
                if params["localization_flag"]:
                    
                    # sate block size
                    # ---------------------------------------------
                    state_block_size = num_state_vars*hdim
                    # radius = 1.5
                    radius = UtilsFunctions(params, ensemble_vec[:state_block_size,:]).compute_adaptive_localization_radius(grid_x, grid_y, base_radius=1.5, method='variance')
                    localization_weights = UtilsFunctions(params, ensemble_vec[:state_block_size,:]).create_tapering_matrix(grid_x, grid_y, radius)
                    # ---------------------------------------------
                    # radius = UtilsFunctions(params, ensemble_vec[:,:]).compute_adaptive_localization_radius(grid_x, grid_y, base_radius=2.0, method='correlation')
                    # localization_weights = UtilsFunctions(params, ensemble_vec[:,:]).create_tapering_matrix(grid_x, grid_y, radius)
                    if EnKF_flag or DEnKF_flag:
                        # ------------------------------
                        localization_weights_resized = np.eye(Cov_model[:state_block_size,:state_block_size].shape[0])
                        localization_weights_resized[:localization_weights.shape[0], :localization_weights.shape[1]] = localization_weights
                        Cov_model[:state_block_size, :state_block_size] *= localization_weights_resized 

                        # check if maximum value of smb is greater than 1.25*smb_obs 
                        smb = ensemble_vec[state_block_size:,:]
                        smb_crit = 1.05*np.max(np.abs(hu_obs[state_block_size:,km]))
                        smb_crit2 = np.max(Cov_model[567:,567:])
                        smb_cov = np.cov(smb_init)
                        smb_flag1 = smb_crit < np.max(np.abs(smb))
                        smb_flag2 = smb_crit2 > 1.02*np.max(smb_cov)
                        if smb_flag2:
                            # force the smb to be 5% 0f the smb_obs
                            # t = model_kwargs["t"]
                            # ensemble_vec[state_block_size:,:] = np.min(smb_init, smb_init + (smb-smb_init)*t[k]/(t[params["nt"]-1] - t[0]))
                            ensemble_vec[state_block_size:,:] = smb_init

                        # ensemble_vec = UtilsFunctions(params, ensemble_vec).compute_smb_mask(k=k, km=km, state_block_size= state_block_size, hu_obs=hu_obs, smb_init=smb_init, model_kwargs=model_kwargs)

                        # smb localization
                        # compute the cross correlation between the state variables and the smb
                        # corr =np.corrcoef(ensemble_vec[:,:])
                        # #  set a threshold for the correlation coefficient
                        # corr_threshold = 0.2
                        # localized_mask = np.abs(corr) > corr_threshold
                        # Cov_model *= localized_mask


                        # ------------------------------
                        # radius = UtilsFunctions(params, ensemble_vec).compute_adaptive_localization_radius(grid_x, grid_y, base_radius=2.0, method='correlation')
                        # # print(f"Adaptive localization radius: {radius}")
                        # localization_weights = UtilsFunctions(params, ensemble_vec).create_tapering_matrix(grid_x, grid_y, radius)
                        # localization_weights_resized = np.eye(Cov_model.shape[0])
                        # localization_weights_resized[:localization_weights.shape[0], :localization_weights.shape[1]] = localization_weights

                        # # Convert to sparse representation
                        # localization_weights = csr_matrix(localization_weights_resized)

                        # # Apply sparse multiplication
                        # Cov_model = csr_matrix(Cov_model).multiply(localization_weights)
                    elif EnRSKF_flag or EnTKF_flag:
                        localization_weights_resized = np.eye(Cov_model[:state_block_size, :].shape[0])
                        print("localization_weights:", localization_weights)
                        localization_weights_resized[:localization_weights.shape[0], :Nens] = localization_weights
                        Cov_model[:state_block_size, :] *= localization_weights_resized

                    # Convert to sparse representation
                    # localization_weights = csr_matrix(localization_weights_resized)
                    # localization_weights = localization_weights_resized

                    # print("Cov_model shape:", Cov_model.shape)
                    # print("state_block_size:", state_block_size)
                    # print("localization_weights_resized shape:", localization_weights_resized.shape)


                    # Apply localization to state covariance only
                    # Cov_model[:3*hdim, :3*hdim] = csr_matrix(Cov_model[:3*hdim, :3*hdim]).multiply(localization_weights)
                    # Cov_model[:state_block_size, :state_block_size] *= localization_weights_resized 

                # Lets no observe the smb (forece entries to [state_block_size:] to zero)
                # hu_obs[state_block_size:, km] = 0

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

                # Adaptive localization
                # radius = 2
                # calculate the correlation coefficient with the background ensembles
                # R = (np.corrcoef(ensemble_vec))**2
                # #  compute the euclidean distance between the grid points
                # cutoff_distance = np.linspace(0, 5e3, Cov_model.shape[0])
                # #  the distance at which the correlation coefficient is less than 1/sqrt(N-1) is the radius
                # # radius = 
                # method = "Gaspari-Cohn"
                # localization_weights = localization_matrix(radius, cutoff_distance, method)
                # # perform a schur product to localize the covariance matrix
                # Cov_model = np.multiply(Cov_model, localization_weights)
                
                # update the ensemble with observations instants
                km += 1

                # inflate the ensemble
                ensemble_vec = UtilsFunctions(params, ensemble_vec).inflate_ensemble(in_place=True)
                # ensemble_vec = UtilsFunctions(params, ensemble_vec)._inflate_ensemble()
            
                # ensemble_vec_mean[:,k+1] = np.mean(ensemble_vec, axis=1)

            # Save the ensemble
            ensemble_vec_full[:,:,k+1] = ensemble_vec

    # comm.Barrier()
    return ensemble_vec_full, ensemble_vec_mean, statevec_bg


