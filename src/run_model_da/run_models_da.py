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

src_dir = os.path.join(project_root, 'src')
applications_dir = os.path.join(project_root, 'applications')
sys.path.insert(0, src_dir)
sys.path.insert(0, applications_dir)

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
    statevec_ens      = da_args[4]   # ensemble of model state
    statevec_bg       = da_args[5]   # background state
    statevec_ens_mean = da_args[6]   # ensemble mean
    statevec_ens_full = da_args[7]   # full ensemble
    commandlinerun    = da_args[8]   # run through the terminal

    nd, N = statevec_ens.shape
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
        x_points = np.linspace(0, model_kwargs["Lx"], model_kwargs["nx"])
        y_points = np.linspace(0, model_kwargs["Ly"], model_kwargs["ny"])
        # use nd instead of nx and ny
        # x_points = np.linspace(0, model_kwargs["Lx"], nd)
        # y_points = np.linspace(0, model_kwargs["Ly"], nd)
        grid_x, grid_y = np.meshgrid(x_points, y_points)


    km = 0
    # radius = 2
    for k in tqdm.trange(params["nt"]):
        # background step
        # statevec_bg = model_module.background_step(k,statevec_bg, hdim, **model_kwargs)

        EnKFclass = EnKF(parameters=params, parallel_flag = parallel_flag)

        # save a copy of initial ensemble
        ensemble_init = statevec_ens.copy()
        
        statevec_ens = EnKFclass.forecast_step(statevec_ens, \
                                               model_module.forecast_step_single, \
                                                Q_err, **model_kwargs)

        # Compute the ensemble mean
        statevec_ens_mean[:,k+1] = np.mean(statevec_ens, axis=1)

        # Analysis step
        obs_index = model_kwargs["obs_index"]
        if (km < params["number_obs_instants"]) and (k+1 == obs_index[km]):

            # Compute the model covariance
            diff = statevec_ens - np.tile(statevec_ens_mean[:,k+1].reshape(-1,1),N)
            if EnKF_flag or DEnKF_flag:
                Cov_model = 1/(N-1) * diff @ diff.T
            elif EnRSKF_flag or EnTKF_flag:
                Cov_model = 1/(N-1) * diff 

            # check if params["sig_obs"] is a scalar
            if isinstance(params["sig_obs"], (int, float)):
                params["sig_obs"] = np.ones(params["nt"]+1) * params["sig_obs"]

            # --- Addaptive localization
            # compute the distance between observation and the ensemble members
            # dist = np.linalg.norm(statevec_ens - np.tile(hu_obs[:,km].reshape(-1,1),N), axis=1)
                
            # get the localization weights
            # localization_weights = UtilsFunctions(params, statevec_ens)._adaptive_localization(dist, \
                                                # ensemble_init=ensemble_init, loc_type="Gaspari-Cohn")

            # method 2
            # get the cut off distance between grid points
            # cutoff_distance = np.linspace(0, 5, Cov_model.shape[0])
            # localization_weights = UtilsFunctions(params, statevec_ens)._adaptive_localization_v2(cutoff_distance)
            # print(localization_weights)
            # get the shur product of the covariance matrix and the localization matrix
            # Cov_model = np.multiply(Cov_model, localization_weights)

            # method 3
            if params["localization_flag"]:
                # radius = 1.5
                radius = UtilsFunctions(params, statevec_ens).compute_adaptive_localization_radius(grid_x, grid_y, base_radius=2.0, method='correlation')
                # print(f"Adaptive localization radius: {radius}")
                localization_weights = UtilsFunctions(params, statevec_ens).create_tapering_matrix(grid_x, grid_y, radius)
                localization_weights_resized = np.eye(Cov_model.shape[0])
                localization_weights_resized[:localization_weights.shape[0], :localization_weights.shape[1]] = localization_weights

                # Convert to sparse representation
                localization_weights = csr_matrix(localization_weights_resized)

                # Apply sparse multiplication
                Cov_model = csr_matrix(Cov_model).multiply(localization_weights)

            # Call the EnKF class for the analysis step
            analysis  = EnKF(Observation_vec=  UtilsFunctions(params, statevec_ens).Obs_fun(hu_obs[:,km]), 
                            Cov_obs=params["sig_obs"][k+1]**2 * np.eye(2*params["number_obs_instants"]+1), \
                            Cov_model= Cov_model, \
                            Observation_function=UtilsFunctions(params, statevec_ens).Obs_fun, \
                            Obs_Jacobian=UtilsFunctions(params, statevec_ens).JObs_fun, \
                            parameters=  params,\
                            parallel_flag=   parallel_flag)
            
            # Compute the analysis ensemble
            if EnKF_flag:
                statevec_ens, Cov_model = analysis.EnKF_Analysis(statevec_ens)
            elif DEnKF_flag:
                statevec_ens, Cov_model = analysis.DEnKF_Analysis(statevec_ens)
            elif EnRSKF_flag:
                statevec_ens, Cov_model = analysis.EnRSKF_Analysis(statevec_ens)
            elif EnTKF_flag:
                statevec_ens, Cov_model = analysis.EnTKF_Analysis(statevec_ens)
            else:
                raise ValueError("Filter type not supported")

            statevec_ens_mean[:,k+1] = np.mean(statevec_ens, axis=1)

            # Adaptive localization
            # radius = 2
            # calculate the correlation coefficient with the background ensembles
            # R = (np.corrcoef(statevec_ens))**2
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
            statevec_ens = UtilsFunctions(params, statevec_ens).inflate_ensemble(in_place=True)
            
        # Save the ensemble
        statevec_ens_full[:,:,k+1] = statevec_ens

    return statevec_ens_full, statevec_ens_mean, statevec_bg


