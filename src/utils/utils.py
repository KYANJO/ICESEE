# =============================================================================
# @Author: Brian Kyanjo
# @Date: 2024-09-24
# @Description: This script includes the observation operator and its Jacobian 
#               for the EnKF data assimilation scheme. It also includes the bed
#               topography function used in the model.
# =============================================================================

# import libraries
import numpy as np
import re
import jax.numpy as jnp
from collections.abc import Iterable
from scipy.stats import norm

# --- helper functions ---
def isiterable(obj):
    return isinstance(obj, Iterable)

class UtilsFunctions:
    def __init__(self, params,ensemble=None):
        """
        Initialize the utility functions with model parameters.
        
        Parameters:
        params (dict): Model parameters, including those used for bed topography.
        """
        self.params   = params
        self.ensemble = ensemble

    def Obs_fun(self, virtual_obs):
        """
        Observation operator that reduces the full observation vector into a smaller subset.
        
        Parameters:
        huxg_virtual_obs (numpy array): The virtual observation vector (n-dimensional).
        m_obs (int): The number of observations.
        
        Returns:
        numpy array: The reduced observation vector.
        """
        n = virtual_obs.shape[0]

        # Initialize the H matrix
        H = np.zeros((self.params["m_obs"] * 2 + 1, n))

        # Calculate distance between measurements
        di = int((n - 2) / (2 * self.params["m_obs"]))

        # Fill the H matrix
        for i in range(self.params["m_obs"]):
            H[i, i * di] = 1
            H[self.params["m_obs"] + i, int((n - 2) / 2) + i * di] = 1

        H[self.params["m_obs"] * 2, n - 2] = 1  # Final element

        # Perform matrix multiplication
        z = H @ virtual_obs
        return z

    def JObs_fun(self, n_model):
        """
        Jacobian of the observation operator.
        
        Parameters:
        n_model (int): The size of the model state vector.
        m_obs (int): The number of observations.
        
        Returns:
        numpy array: The Jacobian matrix of the observation operator.
        """
        n = n_model

        # Initialize the H matrix
        H = np.zeros((self.params["m_obs"] * 2 + 1, n))

        # Calculate distance between measurements
        di = int((n - 2) / (2 * self.params["m_obs"]))

        # Fill the H matrix
        for i in range(self.params["m_obs"]):
            H[i, i * di] = 1
            H[self.params["m_obs"] + i, int((n - 2) / 2) + i * di] = 1

        H[self.params["m_obs"] * 2, n - 2] = 1  # Final element

        return H
    
    def bed(self, x):
        """
        Bed topography function, which computes the bed shape based on input x and model parameters.
        
        Parameters:
        x (jax.numpy array): Input spatial grid points.
        
        Returns:
        jax.numpy array: The bed topography values at each x location.
        """
        # Ensure parameters are floats
        params     = self.params
        sillamp    = float(params['sillamp'])
        sillsmooth = float(params['sillsmooth'])
        xsill      = float(params['xsill'])

        # Compute the bed topography
        b = sillamp * (-2 * jnp.arccos((1 - sillsmooth) * jnp.sin(jnp.pi * x / (2 * xsill))) / jnp.pi - 1)
        return b
    
    def inflate_ensemble(self,in_place=True):
        """
        Inflate ensemble members by a given factor.
        
        Args: 
            ensemble: ndarray (n x N) - The ensemble matrix of model states (n is state size, N is ensemble size).
            inflation_factor: float - scalar or iterable length equal to model states
            in_place: bool - whether to update the ensemble in place
        Returns:
            inflated_ensemble: ndarray (n x N) - The inflated ensemble.
        """
        # check if the inflation factor is scalar
        if np.isscalar(self.params['inflation_factor']):
            _scalar = True
            _inflation_factor = float(self.params['inflation_factor'])
        elif isiterable(self.params['inflation_factor']):
            if len(self.params['inflation_factor']) == self.ensemble.shape[0]:
                _inflation_factor[:] = self.params['inflation_factor'][:]
                _scalar = False
            else:
                raise ValueError("Inflation factor length must be equal to the state size")
        
        # check if we need inflation
        if _scalar:
            if _inflation_factor == 1.0:
                return self.ensemble
            elif _inflation_factor < 0.0:
                raise ValueError("Inflation factor must be positive scalar")
        else:
            _inf = False
            for i in _inflation_factor:
                if i>1.0:
                    _inf = True
                    break
            if not _inf:
                return self.ensemble
        
        ens_size = self.ensemble.shape[1]
        mean_vec = np.mean(self.ensemble, axis=1)
        if in_place:
            inflated_ensemble = self.ensemble
            for ens_idx in range(ens_size):
                state = inflated_ensemble[:, ens_idx]
                if _scalar:
                    # state = (state.axpy(-1.0, mean_vec)).scale(_inflation_factor)
                    state = (state - mean_vec) * _inflation_factor
                else:
                    # state = (state.axpy(-1.0, mean_vec)).multiply(_inflation_factor)
                    state = (state - mean_vec) * _inflation_factor

                # inflated_ensemble[:, ens_idx] = state.add(mean_vec)
                inflated_ensemble[:, ens_idx] = state + mean_vec
        else:
            inflated_ensemble = np.zeros(self.ensemble.shape)
            for ens_idx in range(ens_size):
                state = self.ensemble[:, ens_idx].copy()
                if _scalar:
                    # state = (state.axpy(-1.0, mean_vec)).scale(_inflation_factor)
                    state = (state - mean_vec) * _inflation_factor
                else:
                    # state = (state.axpy(-1.0, mean_vec)).multiply(_inflation_factor)
                    state = (state - mean_vec) * _inflation_factor

                # inflated_ensemble[:, ens_idx] = state.add(mean_vec)
                inflated_ensemble[:, ens_idx] = state + mean_vec
        
        return inflated_ensemble

    def rmse(self,truth, estimate):
        """
        Calculate the Root Mean Squared Error (RMSE) between the true and estimated values.
        
        Parameters:
        truth (numpy array): The true values.
        estimate (numpy array): The estimated values.
        
        Returns:
        float: The RMSE value.
        """
        return np.sqrt(np.mean((truth - estimate) ** 2))

    # --- Create synthetic observations ---
    def _create_synthetic_observations(self,statevec_true):
        """create synthetic observations"""
        nd, nt = statevec_true.shape
        hdim = nd // self.params["num_state_vars"]

        # create synthetic observations
        hu_obs = np.zeros((nd,self.params["nt_m"]))
        # ind_m = self.params["ind_m"]
        ind_m = (np.linspace(int(self.params["dt_m"]/self.params["dt"]), \
                             int(self.params["nt"]), int(self.params["m_obs"]))).astype(int)
        km = 0
        for step in range(nt):
            if (km<self.params["nt_m"]) and (step+1 == ind_m[km]):
                hu_obs[:,km] = statevec_true[:,step+1]
                km += 1

        obs_dist = norm(loc=0,scale=self.params["sig_obs"])
        hu_obs = hu_obs + obs_dist.rvs(size=hu_obs.shape)

        return hu_obs
    

def localization_matrix(euclidean_distance, cutoff_radius, method):

    import re

    if re.match(r'\Agaspari(_|-)*cohn\Z', method, re.IGNORECASE):
        print('Using Gaspari-Cohn function')
        f = np.zeros(len(cutoff_radius))
        for i in range(len(cutoff_radius)):
            c = euclidean_distance
            r = cutoff_radius[i]
            if 0 <= abs(r) <= c:
                f[i] = -1/4*(abs(r)/c)**5 + 1/2*(abs(r)/c)**4 + 5/8*(abs(r)/c)**3 - 5/3*(abs(r)/c)**2 + 1
            elif c <= abs(r) <= 2*c:
                f[i] = 1/12*(abs(r)/c)**5 - 1/2*(abs(r)/c)**4 + 5/8*(abs(r)/c)**3 + 5/3*(abs(r)/c)**2 - 5*(abs(r)/c) + 4 - 2/3*(abs(r)/c)**-1
            elif abs(r) > 2*c:
                f[i] = 0
        # the localization matrix is a diagonal matrix
        return np.diag(f)

# --- Addaptive localization module ---
def adaptive_localization(ensemble, forecast_cov ,euclidean_distance, cutoff_radius, method):
    """
    @description: This function performs adaptive localization on the forecast covariance matrix based
                  on an adaptive radius and localization method.
    Args:
        ensemble: ndarray (n x N) - Ensemble matrix of model states (n: state size, N: ensemble size).
        forecast_cov: ndarray (n x n) - Forecast covariance matrix.
        euclidean_distance: ndarray (n x n) - Euclidean distance matrix.
        cutoff_radius: float - Decorrelation radius.
        method: str - Localization method ('Gauss', 'Cosine', 'Gaspari_Cohn', etc.).

    Returns:
        localized_cov: ndarray (n x n) - Localized covariance matrix.
    """

    localization_matrix = localization_matrix(euclidean_distance, cutoff_radius, method)

    #  weighted covariance matrix
    localized_cov = np.multiply(localization_matrix, forecast_cov)
    return localized_cov
    
# --- calculate localization coefficients ---
def calculate_localization_coefficients(radius, distances, method='Gauss', verbose=False):
    """
    Calculate spatial decorrelation coefficients using a specified method.

    Args:
        radius (float): Decorrelation radius.
        distances (array-like): Distances for coefficient calculation.
        method (str): Localization method ('Gauss', 'Cosine', 'Gaspari_Cohn', etc.).
        verbose (bool): If True, print warnings for zero radius.

    Returns:
        coefficients: Decorrelation coefficients, scalar if input is scalar.
    """
    # Define supported methods and check inputs
    supported_methods = ['gauss', 'cosine', 'cosine_squared', 'gaspari_cohn', 'exp3', 'cubic', 'quadro', 'step']
    if method.lower() not in supported_methods:
        raise ValueError(f"Unsupported method '{method}'. Supported methods: {supported_methods}")

    # Process distance inputs
    is_scalar = np.isscalar(distances)
    distances = np.atleast_1d(distances).astype(float)

    # Return zeros for zero radius unless distances are also zero
    if radius == 0:
        if verbose: print("Radius is zero; assuming delta function at zero distance.")
        coefficients = (distances == 0).astype(float)
        return coefficients[0] if is_scalar else coefficients

    # Calculate thresholds for non-zero radius
    thresh_map = {
        'gauss': radius, 'exp3': radius, 'cosine': radius * 2.3167,
        'cosine_squared': radius * 3.2080, 'gaspari_cohn': radius * 1.7386,
        'cubic': radius * 1.8676, 'quadro': radius * 1.7080
    }
    thresh = thresh_map.get(method.lower(), radius)

    # Initialize coefficients based on methods
    scaled_dist = distances / thresh
    if method.lower() == 'gauss':
        coefficients = np.exp(-0.5 * scaled_dist ** 2)
    elif method.lower() == 'exp3':
        coefficients = np.exp(-0.5 * scaled_dist ** 3)
    elif method.lower() == 'cosine':
        coefficients = np.where(distances <= thresh, (1 + np.cos(scaled_dist * np.pi)) / 2, 0)
    elif method.lower() == 'cosine_squared':
        coefficients = np.where(distances <= thresh, ((1 + np.cos(scaled_dist * np.pi)) / 2) ** 2, 0)
    elif method.lower() == 'gaspari_cohn':
        coefficients = np.where(distances <= thresh, 1 - scaled_dist ** 2 * (5 / 3 - scaled_dist ** 3 / 4), 0)
    elif method.lower() == 'cubic':
        coefficients = np.where(distances <= thresh, (1 - scaled_dist ** 3) ** 3, 0)
    elif method.lower() == 'quadro':
        coefficients = np.where(distances <= thresh, (1 - scaled_dist ** 4) ** 4, 0)
    elif method.lower() == 'step':
        coefficients = (distances < radius).astype(float)

    return coefficients[0] if is_scalar else coefficients

        
