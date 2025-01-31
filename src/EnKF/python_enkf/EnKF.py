# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-11-06
# @description: Conatins the class for analysis steps all Ensemble Kalman Filter (EnKF)
#              EnKF include both Stochastic and Deterministic EnKF options: EnKF,
#              EnSRF, DEnKF, and EnTKF.
#             - The class further includes parallelized forecast and analysis steps with
#               MPI, Dask, Ray, and Multiprocessing.
# =============================================================================

import os, sys
import re
import numpy as np
import scipy
from scipy.stats import multivariate_normal

class EnsembleKalmanFilter:
    def __init__(self, Observation_vec=None, Cov_obs=None, Cov_model=None, \
                 Observation_function=None, Obs_Jacobian=None, parameters=None, taper_matrix=None, parallel_flag="serial"):
        """
        Initializes the Analysis class for the Ensemble Kalman Filter (EnKF).
        
        Parameters:
        Observation_function: Callable - Observation function mapping state space to observation space.
        observation_vec: ndarray - Observation vector (m x 1).
        Obs_Jacobian: Callable - Jacobian of the observation function.
        Cov_obs: ndarray - Observation covariance matrix (m x m).
        Cov_model: ndarray - Model covariance matrix (n x n).
        taper: ndarray - Covariance taper matrix (n x n).
        params: dict - Dictionary containing parameters like "m_obs" and others.\
        parallel_flag: str - Flag for parallelization (serial,MPI, Dask, Ray, Multiprocessing).
        """
        self.Observation_vec        = Observation_vec
        self.Cov_obs                = Cov_obs
        self.Cov_model              = Cov_model
        self.Obs_Jacobian           = Obs_Jacobian
        self.parameters             = parameters
        self.taper_matrix           = taper_matrix
        self.Observation_function   = Observation_function
        self.parallel_flag          = parallel_flag

    def _compute_kalman_gain(self):
        """Compute the Kalman gain based on the Jacobian of the observation function."""
        Jobs = self.Obs_Jacobian(self.Cov_model.shape[0])
        inv_matrix = np.linalg.inv(Jobs @ self.Cov_model @ Jobs.T + self.Cov_obs)
        KalGain = self.Cov_model @ Jobs.T @ inv_matrix
        return KalGain
    
    # Forecast step
    def forecast_step(self, ensemble=None, forecast_step_single=None, Q_err=None, **model_kwags):
        """
        Forecast step for the Ensemble Kalman Filter (EnKF).
        
        Parameters:
            forecast_step_single: Callable - Function for the forecast step of each ensemble member.
            Q_err: ndarray - Process noise matrix.
            parallel_flag: str - Flag for parallelization (serial,MPI, Dask, Ray, Multiprocessing).
            **model_kwags: dict - Keyword arguments for the model.
        
        Returns:
            ensemble: ndarray - Updated ensemble matrix.
        """
        
        if re.match(r"\Aserial\Z", self.parallel_flag, re.IGNORECASE):
            # Serial forecast step
            nd, Nens = ensemble.shape # Get the number of ensemble members

            # Loop over the ensemble members
            for ens in range(Nens):
                ensemble[:,ens] = forecast_step_single(ens, ensemble, nd, \
                                             Q_err, self.parameters, **model_kwags)
            return ensemble

        # Parallel forecast step using MPI
        elif re.match(r"\AMPI\Z", self.parallel_flag, re.IGNORECASE):
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            # Get the number of ensemble members
            nd, Nens = ensemble.shape

            size = min(comm.Get_size(), Nens)  # Limit processes to ensemble size

            # Balanced workload
            chunk_sizes = [(Nens // size) + (1 if i < (Nens % size) else 0) for i in range(size)]
            displacements = [sum(chunk_sizes[:i]) for i in range(size)]
            start_idx = displacements[rank]
            end_idx = start_idx + chunk_sizes[rank]

            # Local chunk
            local_ensemble = ensemble[:, start_idx:end_idx]

            # Perform forecast step
            for ens in range(local_ensemble.shape[1]):
                local_ensemble[:, ens] = forecast_step_single( ens, local_ensemble, nd, Q_err, self.parameters, **model_kwags)

            # Avoid gather; update ensemble in place
            gathered_ensemble = comm.allgather(local_ensemble)

            if rank == 0:
                # print(f"Gathered ensemble shape: {np.hstack(gathered_ensemble).shape}")
                ensemble = np.hstack(gathered_ensemble)

            return ensemble


        # Parallel forecast step using Dask
        elif re.match(r"\ADask\Z", self.parallel_flag, re.IGNORECASE):
            import dask
            import dask.array as da
            from dask import compute, delayed
            import copy

            nd, Nens = ensemble.shape

            # Create delayed tasks for each ensemble member
            tasks = [
                    delayed(forecast_step_single)(ens, ensemble, nd, Q_err, self.parameters, **model_kwags)
                    for ens in range(Nens)
                ]

            # Compute all tasks in parallel and collect the results
            results = compute(*tasks)

            # Update the ensemble matrix
            ensemble = np.array(results).T

            return ensemble

        # Parallel forecast step using Ray
        elif re.match(r"\ARay\Z", self.parallel_flag, re.IGNORECASE):
            import ray

            nd, Nens = ensemble.shape

            # Initialize Ray
            ray.init(ignore_reinit_error=True)

            @ray.remote
            def ray_worker( ensemble_member, Q_err, parameters, model_kwargs):
                """
                Remote function to perform forecast step for a single ensemble member.
                This function will be executed in parallel by Ray workers.
                """
                return forecast_step_single(ens, ensemble_member, nd, Q_err, parameters, **model_kwargs)
            
            _, Nens = ensemble.shape

            # Launch tasks in parallel using Ray
            futures = [
                ray_worker.remote(ens, ensemble[:, ens], Q_err, self.parameters, **model_kwargs)
                for ens in range(Nens)
            ]

            # Collect results from all Ray tasks
            results = ray.get(futures)

            # Convert the list of results back to a numpy array and transpose
            ensemble = np.array(results).T
            return ensemble
        
        # Parallel forecast step using Multiprocessing
        elif re.match(r"\AMultiprocessing\Z", self.parallel_flag, re.IGNORECASE):
            import multiprocessing as mp

            nd, Nens = ensemble.shape

            # Define a helper function to handle arguments for each worker
            def worker(ens_idx):
                return forecast_step_single(ens, ensemble[:, ens_idx], nd, Q_err, self.parameters, **model_kwargs)

            # Create a pool of workers
            with mp.Pool(mp.cpu_count()) as pool:
                # Use pool.map to parallelize the forecast step
                results = pool.map(worker, range(Nens))

            # Convert the list of results back to a numpy array and transpose
            ensemble = np.array(results).T
            return ensemble
        
        # python openmp parallelization
        elif re.match(r"\APyomp\Z", self.parallel_flag, re.IGNORECASE):
            # check if the Pyomp module is installed
            try:
                # check python version: it should be [3.9 - 3.10]
                if sys.version_info[0] == 3 and (sys.version_info[1] > 9 or sys.version_info[1] <= 10):
                    from numba import njit
                    from numba.openmp import openmp_context as openmp
                    from numba.openmp import omp_get_thread_num, omp_get_num_threads
                    from collections import namedtuple

                    nd, Nens = ensemble.shape

                    #  get the maximum number of cores
                    MaxTHREADS = os.cpu_count()

                    ModelKwargs = namedtuple("ModelKwargs", model_kwags.keys())
                    processed_args = ModelKwargs(**model_kwags)

                    # create a wrapper for the forecast step function to use @njit
                    @njit
                    def forecast_wrapper():
                        partialEnsembles = np.zeros((nd,MaxTHREADS))
                        with openmp("parallel shared(partialEnsembles, numThrds) private(threadID,ens,localEnsemble)"):
                            threadID = omp_get_thread_num() #thread rank = 0...(numThreads-1)
                            with openmp("single"): #one thread does the work, others wait
                                numThrds = omp_get_num_threads() #get number of threads
                            for ens in range(threadID,Nens,numThrds):
                                partialEnsembles[:,threadID] = forecast_step_single(ens, ensemble, nd, Q_err, self.parameters, processed_args)
                                
                        return partialEnsembles

                    return forecast_wrapper()
                
            except ImportError:
                raise ImportError("Pyomp module not found. Please install it using 'conda install -c python-for-hpc -c conda-forge pyomp' of see https://github.com/Python-for-HPC/PyOMP for more details.")
            
        else:
            raise ValueError("Invalid parallel flag. Choose from 'serial', 'MPI', 'Dask', 'Ray', 'Multiprocessing'.")

    # Analysis steps
    def EnKF_Analysis(self, ensemble):
        """
        Stochastic Ensemble Kalman Filter (EnKF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        # Compute the Kalman gain
        KalGain = self._compute_kalman_gain()

        n,N = ensemble.shape
        m   =self.Observation_vec.shape[0] 

        # compute virtual observations and ensemble analysis
        virtual_observations  = np.zeros((m,N))
        ensemble_analysis     = np.zeros_like(ensemble)

        for i in range(N):
            # Generate virtual observations
            virtual_observations[:,i] =  self.Observation_vec+ multivariate_normal.rvs(mean=np.zeros(m), cov=self.Cov_obs)
            # Compute the analysis step
            ensemble_analysis[:,i] = ensemble[:,i] + KalGain @ (virtual_observations[:,i] - self.Observation_function(ensemble[:,i]))

        # compute the analysis error covariance
        difference = ensemble_analysis - np.mean(ensemble_analysis, axis=1,keepdims=True)
        analysis_error_cov =(1/(N-1)) * difference @ difference.T

        return ensemble_analysis, analysis_error_cov
       
    def DEnKF_Analysis(self, ensemble):
        """
        Deterministic Ensemble Kalman Filter (DEnKF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        # Compute the Kalman gain
        KalGain = self._compute_kalman_gain()

        n,N = ensemble.shape
        m   = self.Observation_vec.shape[0] 

        # compute ensemble mean
        ensemble_forecast_mean = np.mean(ensemble, axis=1)

        # compute the anlysis mean
        analysis_mean = ensemble_forecast_mean + KalGain @ (self.Observation_vec - self.Observation_function(ensemble_forecast_mean))

        # compute the forecast and analysis anomalies
        forecast_anomalies = np.zeros_like(ensemble)
        analysis_anomalies = np.zeros_like(ensemble)

        for i in range(N):
            forecast_anomalies[:,i] = ensemble[:,i] - ensemble_forecast_mean
            analysis_anomalies[:,i] = forecast_anomalies[:,i] - 0.5 * KalGain @ self.Observation_function(forecast_anomalies[:,i])

        # compute the analysis ensemble
        ensemble_analysis = analysis_anomalies + analysis_mean.reshape(-1,1)

        # compute the analysis error covariance
        analysis_error_cov =(1/(N-1)) * analysis_anomalies @ analysis_anomalies.T

        return ensemble_analysis, analysis_error_cov
    
    # def DEnKF_Analysis(self, ensemble):
    #     """
    #     Deterministic Ensemble Kalman Filter (DEnKF) analysis step using MPI.
        
    #     Parameters:
    #         ensemble: ndarray - Ensemble matrix (n x N).

    #     Returns:
    #         ensemble_analysis: updated ensemble matrix (n x N).
    #         analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
    #     """

    #     if re.match(r"\Aserial\Z", self.parallel_flag, re.IGNORECASE):

    #         print("DEnKF Analysis: Starting analysis step")
            
    #         # Compute the Kalman gain
    #         print("DEnKF Analysis: Computing Kalman gain")
    #         KalGain = self._compute_kalman_gain()
            
    #         # Get dimensions
    #         n, N = ensemble.shape
    #         m = self.Observation_vec.shape[0]
    #         print(f"DEnKF Analysis: Ensemble dimensions (n={n}, N={N}), Observation size (m={m})")

    #         # Compute ensemble mean
    #         print("DEnKF Analysis: Computing ensemble forecast mean")
    #         ensemble_forecast_mean = np.mean(ensemble, axis=1)
    #         print(f"DEnKF Analysis: Ensemble forecast mean calculated: {ensemble_forecast_mean}")

    #         # Compute analysis mean
    #         print("DEnKF Analysis: Computing analysis mean")
    #         analysis_mean = ensemble_forecast_mean + KalGain @ (self.Observation_vec - self.Observation_function(ensemble_forecast_mean))
    #         print(f"DEnKF Analysis: Analysis mean calculated: {analysis_mean}")

    #         # Compute forecast and analysis anomalies
    #         print("DEnKF Analysis: Computing forecast and analysis anomalies")
    #         forecast_anomalies = np.zeros_like(ensemble)
    #         analysis_anomalies = np.zeros_like(ensemble)

    #         for i in range(N):
    #             forecast_anomalies[:, i] = ensemble[:, i] - ensemble_forecast_mean
    #             analysis_anomalies[:, i] = forecast_anomalies[:, i] - 0.5 * KalGain @ self.Observation_function(forecast_anomalies[:, i])
    #             print(f"DEnKF Analysis: Anomalies for member {i} calculated")

    #         # Compute the analysis ensemble
    #         print("DEnKF Analysis: Computing analysis ensemble")
    #         ensemble_analysis = analysis_anomalies + analysis_mean.reshape(-1, 1)
    #         print(f"DEnKF Analysis: Analysis ensemble computed: {ensemble_analysis}")

    #         # Compute the analysis error covariance
    #         print("DEnKF Analysis: Computing analysis error covariance")
    #         analysis_error_cov = (1 / (N - 1)) * analysis_anomalies @ analysis_anomalies.T
    #         print(f"DEnKF Analysis: Analysis error covariance computed: {analysis_error_cov}")

    #         print("DEnKF Analysis: Analysis step completed")
    #         return ensemble_analysis, analysis_error_cov

    #     elif re.match(r"\AMPI\Z", self.parallel_flag, re.IGNORECASE):

    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         rank = comm.Get_rank()
    #         size = comm.Get_size()

    #         n, N = ensemble.shape

    #         # Distribute the ensemble among processes
    #         chunk_sizes = [(N // size) + (1 if i < (N % size) else 0) for i in range(size)]
    #         displacements = [sum(chunk_sizes[:i]) for i in range(size)]
    #         start_idx = displacements[rank]
    #         end_idx = start_idx + chunk_sizes[rank]
    #         local_ensemble = ensemble[:, start_idx:end_idx]

    #         # 1. Compute local ensemble mean
    #         print(f"Rank {rank}: Starting local mean calculation")
    #         local_mean = np.mean(local_ensemble, axis=1)

    #         # 2. Compute the global mean
    #         global_mean = np.zeros(n)
    #         comm.Allreduce(local_mean, global_mean, op=MPI.SUM)
    #         global_mean /= N
    #         print(f"Rank {rank}: Global mean calculated: {global_mean}")

    #         # 3. Compute Kalman gain
    #         KalGain = self._compute_kalman_gain()

    #         # 4. Compute the analysis mean
    #         if rank == 0:
    #             print(f"Rank {rank}: Computing analysis mean")
    #         analysis_mean = global_mean + KalGain @ (self.Observation_vec - self.Observation_function(global_mean))

    #         # 5. Compute local anomalies
    #         local_anomalies = local_ensemble - global_mean.reshape(-1, 1)

    #         # 6. Apply correction to local anomalies
    #         local_analysis_anomalies = np.zeros_like(local_anomalies)
    #         for i in range(local_anomalies.shape[1]):
    #             local_analysis_anomalies[:, i] = (
    #                 local_anomalies[:, i] - 0.5 * KalGain @ self.Observation_function(local_anomalies[:, i])
    #             )
    #         print(f"Rank {rank}: Local analysis anomalies calculated")

    #         # 7. Gather analysis anomalies on rank 0
    #         gathered_anomalies = comm.gather(local_analysis_anomalies, root=0)

    #         # 8. Assemble analysis ensemble and compute covariance on rank 0
    #         if rank == 0:
    #             print(f"Rank {rank}: Assembling analysis anomalies")
    #             analysis_anomalies = np.hstack(gathered_anomalies)
    #             ensemble_analysis = analysis_anomalies + analysis_mean.reshape(-1, 1)
    #             analysis_error_cov = (1 / (N - 1)) * analysis_anomalies @ analysis_anomalies.T
    #         else:
    #             ensemble_analysis = None
    #             analysis_error_cov = None

    #         # 9. Broadcast results to all processes
    #         ensemble_analysis = comm.bcast(ensemble_analysis, root=0)
    #         analysis_error_cov = comm.bcast(analysis_error_cov, root=0)
    #         print(f"Rank {rank}: Final analysis broadcast completed")

    #         return ensemble_analysis, analysis_error_cov

    def EnRSKF_Analysis(self, ensemble):
        """
        Deterministic Ensemble Square Root Filter (EnSRF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        _,N = ensemble.shape
        I = np.eye(N)

        # compute the mean of the forecast ensemble
        ensemble_forecast_mean = np.mean(ensemble, axis=1)

        obs_anomaly =self.Observation_vec.reshape(-1,1) - self.Observation_function(ensemble)

        V =  self.Observation_function(self.Cov_model)
        IN = self.Cov_obs + V@V.T

        # compute the analysis ensemble
        ensemble_analysis  = ensemble_forecast_mean.reshape(-1,1) + self.Cov_model @ V.T @ np.linalg.solve(IN, obs_anomaly)

        #  singular value decomposition
        U,S,Vt = np.linalg.svd(I - V.T @ np.linalg.solve(IN, V))

        # compute the analysis error covariance
        analysis_error_cov = ensemble_analysis + self.Cov_model@(U@np.diag(np.sqrt(S))@U.T)

        return ensemble_analysis, analysis_error_cov
    
    def EnTKF_Analysis(self, ensemble):
        """
        Ensemble Transform Kalman Filter (EnTKF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        
        # compute the mean of the forecast ensemble
        ensemble_forecast_mean = np.mean(ensemble, axis=1)

        obs_anomaly =self.Observation_vec.reshape(-1,1) - self.Observation_function(ensemble)

        _obs_operator = self.Observation_function(self.Cov_model)

        # perform singular value decomposition
        U,S, _ = np.linalg.svd(_obs_operator.T @ np.linalg.solve(self.Cov_obs, _obs_operator))

        # compute the inverse of the analysis error covariance
        inv_analysis_error_cov = U @ np.diag(1/(S+1)) @ U.T

        # Right hand side of the analysis equation
        rhs = _obs_operator.T @ np.linalg.solve(self.Cov_obs, obs_anomaly)

        # compute the analysis ensemble
        ensemble_analysis = ensemble_forecast_mean.reshape(-1,1) + self.Cov_model@inv_analysis_error_cov@rhs

        # compute the analysis ensemble increment
        ensemble_increment = self.Cov_model @ U @ np.diag(np.sqrt(1/(S+1))) @ U.T

        # compute the analysis error covariance
        analysis_error_cov = ensemble_analysis + ensemble_increment

        return ensemble_analysis, analysis_error_cov