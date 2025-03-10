# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-03-06
# @description: computes the X5 matrix for the EnKF
#               - the new formulation is based on the paper by Geir Evensen: The Ensemble Kalman Filter: Theoretical Formulation And Practical Implementation
#               - this formulation supports our need for mpi parallelization and no need for localizations
# =============================================================================

import gc
import copy
import numpy as np
import bigmpi4py as BM
from scipy.stats import multivariate_normal

# seed the random number generator
np.random.seed(0)

# ============================ EnKF functions ============================  
def EnKF_X5(ensemble_vec, Cov_obs, Nens, h, d):
    """
    Function to compute the X5 matrix for the EnKF
        - ensemble_vec: ensemble matrix of size (ndxNens)
        - Cov_obs: observation covariance matrix
        - Nens: ensemble size
        - d: observation vector
    """
    Eta = np.zeros((d.shape[0], Nens)) # mxNens, ensemble pertubations
    D   = np.zeros_like(Eta) # mxNens #virtual observations
    HA  = np.zeros_like(D)
    for ens in range(Nens):
        Eta[:,ens] = np.random.multivariate_normal(mean=np.zeros(d.shape[0]), cov=Cov_obs) 
        D[:,ens] = d + Eta[:,ens]
        HA[:,ens] = h(ensemble_vec[:,ens])
    
    # --- compute the innovations D` = D-HA
    Dprime = D - HA # mxNens

    # --- compute HAbar
    HAbar = np.mean(HA, axis=1) # mx1
    # --- compute HAprime
    # HAprime = HA - HAbar.reshape(-1,1) # mxNens (requires H to be linear)
    
    # Aprime = ensemble_vec@(np.eye(Nens) - one_N) # mxNens
    one_N = np.ones((Nens,Nens))/Nens
    HAprime= HA@(np.eye(Nens) - one_N) # mxNens

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

    return X5

def analysis_enkf_update(ensemble_vec, shape_ens, X5, comm_world):
    """
    Function to perform the analysis update using the EnKF
        - broadcast X5 to all processors
        - initialize an empty ensemble vector for the rest of the processors
        - scatter ensemble_vec to all processors
        - do the ensemble analysis update: A_j = Fj*X5
        - gather from all processors
    """
    # get the rank and size of the world communicator
    rank_world = comm_world.Get_rank()
    # broadcast X5 to all processors
    X5 = BM.bcast(X5, comm=comm_world)

    # initialize the an empty ensemble vector for the rest of the processors
    if rank_world != 0:
        ensemble_vec = np.empty(shape_ens, dtype=np.float64)

    # --- scatter ensemble_vec to all processors ---
    scatter_ensemble = BM.scatter(ensemble_vec, comm_world)

    # do the ensemble analysis update: A_j = Fj*X5 
    analysis_vec = np.dot(scatter_ensemble, X5)

    # gather from all processors
    ensemble_vec = BM.allgather(analysis_vec, comm_world)

    # clean the memory
    del scatter_ensemble, analysis_vec; gc.collect()

    return ensemble_vec
# ============================ EnKF functions ============================

# ============================ DEnKF functions ============================


# ============================ EnSRF functions ============================


# ============================ EnTKF functions ============================


# ============================ Other functions ============================

def gather_and_broadcast_data_default_run(updated_state, subcomm, sub_rank, comm_world, rank_world, params):
    """
    Gathers, processes, and broadcasts ensemble data across MPI processes.

    Parameters:
    - updated_state: dict, contains state variables to be gathered
    - subcomm: MPI communicator for subgroups
    - sub_rank: int, rank within the subcommunicator
    - comm_world: MPI communicator for all processes
    - rank_world: int, rank within the world communicator
    - params: dict, contains necessary parameters like "total_state_param_vars"
    - BM: object with a `bcast` method for broadcasting data

    Returns:
    - ensemble_vec: The processed and broadcasted ensemble data
    """

    # Step 1: Gather data from all sub-ranks
    global_data = {key: subcomm.gather(data, root=0) for key, data in updated_state.items()}

    # Step 2: Process on sub_rank 0
    if sub_rank == 0:
        for key in global_data:
            global_data[key] = np.hstack(global_data[key])

        # Stack all variables into a single array
        stacked = np.hstack([global_data[key] for key in updated_state.keys()])
        shape_ = np.array(stacked.shape, dtype=np.int32)
    else:
        shape_ = np.empty(2, dtype=np.int32)

    # Step 3: Broadcast the shape to all processors
    shape_ = comm_world.bcast(shape_, root=0)

    # Step 4: Prepare the stacked array for non-root sub-ranks
    if sub_rank != 0:
        stacked = np.empty(shape_, dtype=np.float64)

    # Step 5: Gather the stacked arrays from all sub-ranks
    all_ens = comm_world.gather(stacked if sub_rank == 0 else None, root=0)

    # Step 6: Final processing on world rank 0
    if rank_world == 0:
        all_ens = [arr for arr in all_ens if isinstance(arr, np.ndarray)]
        ensemble_vec = np.column_stack(all_ens)
        hdim = ensemble_vec.shape[0] // params["total_state_param_vars"]
    else:
        ensemble_vec = np.empty((shape_[0], params["Nens"]), dtype=np.float64)

    # Step 7: Broadcast the final ensemble vector
    ensemble_vec = BM.bcast(ensemble_vec, comm_world)

    return ensemble_vec