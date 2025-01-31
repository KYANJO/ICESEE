# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-11-06
# @description: Synthetic ice stream with data assimilation
# =============================================================================

# --- Imports ---
import sys
import os
import numpy as np
from mpi4py import MPI

# --- Configuration ---
os.environ["PETSC_CONFIGURE_OPTIONS"] = "--download-mpich-device=ch3:sock"
os.environ["OMP_NUM_THREADS"] = "1"

# --- Utility imports ---
sys.path.insert(0, '../../../config')
from _utility_imports import *
applications_dir = os.path.join(project_root, 'applications','icepack_model')
sys.path.insert(0, applications_dir)

from firedrake import *
from firedrake.petsc import PETSc
from icepack.constants import (
    ice_density as rho_I,
    water_density as rho_W,
    gravity as g,
)


# --- Utility Functions ---
from run_models_da import run_model_with_filter
from icepack_enkf import generate_true_state, generate_nurged_state, initialize_ensemble

import icepack
import icepack.models.friction

# --- Load Parameters ---
# Load parameters from a YAML file
parameters_file = "params.yaml"
parameters = load_yaml_to_dict(parameters_file)

physical_params = get_section(parameters, "physical-parameters")
modeling_params = get_section(parameters, "modeling-parameters")
enkf_params = get_section(parameters, "enkf-parameters")

# --- Geometry and Mesh ---
PETSc.Sys.Print('Setting up mesh across %d processes' % COMM_WORLD.size)
Lx, Ly = int(float(physical_params["Lx"])), int(float(physical_params["Ly"]))
nx, ny = int(float(physical_params["nx"])), int(float(physical_params["ny"]))
print(f"Mesh dimensions: {Lx} x {Ly} with {nx} x {ny} elements")

comm = COMM_WORLD.Split(COMM_WORLD.rank % 2)
if COMM_WORLD.rank % 2 == 0:
   # Even ranks create a quad mesh
   mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True, comm=comm)
else:
   # Odd ranks create a triangular mesh
  mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, comm=comm)

# mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, comm=comm)
# mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)
PETSc.Sys.Print('  rank %d owns %d elements and can access %d vertices' \
                % (mesh.comm.rank, mesh.num_cells(), mesh.num_vertices()),
                comm=COMM_SELF)

# --- Local MPI implementation ---
comm = MPI.COMM_WORLD   # Initialize MPI
rank = comm.Get_rank()  # Get rank of current MPI process
size = comm.Get_size()  # Get total number of MPI processes

# #  Dimension of the mesh
# Ndim = Lx * Ly
# num_per_rank = Ndim // size
# lower_bound = rank * num_per_rank
# upper_bound = (rank + 1) * num_per_rank

Q = firedrake.FunctionSpace(mesh, "CG", int(float(physical_params["degree"])))
V = firedrake.VectorFunctionSpace(mesh, "CG", int(float(physical_params["degree"])))

x, y = firedrake.SpatialCoordinate(mesh)

# --- Bedrock and Surface Elevations ---
b_in, b_out = (float(physical_params["b_in"])), (float(physical_params["b_out"]))
s_in, s_out = (float(physical_params["s_in"])), (float(physical_params["s_out"]))

b = firedrake.interpolate(b_in - (b_in - b_out) * x / Lx, Q)
s0 = firedrake.interpolate(s_in - (s_in - s_out) * x / Lx, Q)
h0 = firedrake.interpolate(s0 - b, Q)

# --- Driving Stress ---
h_in = s_in - b_in
ds_dx = (s_out - s_in) / Lx
tau_D = -rho_I * g * h_in * ds_dx
PETSc.Sys.Print(f"Driving stress = {1000*tau_D} kPa")

# --- Initial Velocity ---
u_in, u_out = float(physical_params["u_in"]), float(physical_params["u_out"])
velocity_x = u_in + (u_out - u_in) * (x / Lx) ** 2
u0 = firedrake.interpolate(firedrake.as_vector((velocity_x, 0)), V)

# --- Friction Coefficient ---
PETSc.Sys.Print("Importing icepack ...")
T = firedrake.Constant(float(modeling_params["T"]))
A = icepack.rate_factor(T)

from icepack.constants import weertman_sliding_law as m
expr = (0.95 - 0.05 * x / Lx) * tau_D / u_in**(1 / m)
C = firedrake.interpolate(expr, Q)

p_W = rho_W * g * firedrake.max_value(0, h0 - s0)
p_I = rho_I * g * h0
phi = 1 - p_W / p_I

# --- Friction Law ---
def weertman_friction_with_ramp(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    s = kwargs["surface"]
    C = kwargs["friction"]

    p_W = rho_W * g * firedrake.max_value(0, h - s)
    p_I = rho_I * g * h
    phi = 1 - p_W / p_I
    return icepack.models.friction.bed_friction(
        velocity=u,
        friction=C * phi,
    )

# --- Ice Stream Model ---
model_weertman = icepack.models.IceStream(friction=weertman_friction_with_ramp)

opts = {"dirichlet_ids": [1], "side_wall_ids": [3, 4]}
solver_weertman = icepack.solvers.FlowSolver(model_weertman, **opts)

u0 = solver_weertman.diagnostic_solve(
    velocity=u0,
    thickness=h0,
    surface=s0,
    fluidity=A,
    friction=C,
)

expr = -1e3 * C * phi * sqrt(inner(u0, u0)) ** (1 / m - 1) * u0
tau_b = firedrake.interpolate(expr, V)

# --- Accumulation ---
a_in = firedrake.Constant(float(modeling_params["a_in"]))
da   = firedrake.Constant(float(modeling_params["da"]))
a    = firedrake.interpolate(a_in + da * x / Lx, Q)

# nurged accumulation
a_in_p  = firedrake.Constant(float(modeling_params["a_in_p"]))
da_p    = firedrake.Constant(float(modeling_params["da_p"]))
a_p     = firedrake.interpolate(a_in_p + da_p * x / Lx, Q)

# --- Update h and u ---
h = h0.copy(deepcopy=True)
u = u0.copy(deepcopy=True)

# --- Ensemble Parameters ---
params = {
    "nt": int(float(modeling_params["num_years"])) * int(float(modeling_params["timesteps_per_year"])),
    "dt": 1.0 / float(modeling_params["timesteps_per_year"]),
    "num_state_vars": int(float(enkf_params["num_state_vars"])),
    "nd": h0.dat.data.size * int(float(enkf_params["num_state_vars"])),
    "Nens": int(float(enkf_params["Nens"])),
    "number_obs_instants": int(int(float(enkf_params["obs_max_time"]))/float(enkf_params["freq_obs"])),
    "inflation_factor": float(enkf_params["inflation_factor"]),
    "sig_model": float(enkf_params["sig_model"]),
    "sig_obs": float(enkf_params["sig_obs"]),
    "sig_Q": float(enkf_params["sig_Q"]),
    "freq_obs": float(enkf_params["freq_obs"]),
    "obs_max_time": int(float(enkf_params["obs_max_time"])),
    "obs_start_time": float(enkf_params["obs_start_time"]),
}

kwargs = {"a":a, "h0":h0, "u0":u0, "C":C, "A":A,"Q":Q,"V":V, "da":float(modeling_params["da"]),
          "b":b, "dt":params["dt"], "seed":float(enkf_params["seed"]), "x":x, "y":y,
          "t":np.linspace(0, int(float(modeling_params["num_years"])), params["nt"] + 1),
          "Lx":Lx, "Ly":Ly, "nx":nx, "ny":ny, "h_nurge_ic":float(enkf_params["h_nurge_ic"]), 
          "u_nurge_ic":float(enkf_params["u_nurge_ic"]),"nurged_entries":float(enkf_params["nurged_entries"]),
         "a_in_p":float(modeling_params["a_in_p"]), "da_p":float(modeling_params["da_p"]),
         "solver":solver_weertman,
          "obs_index": (np.linspace(int(params["freq_obs"]/params["dt"]), \
                        int(params["obs_max_time"]/params["dt"]), int(params["number_obs_instants"]))).astype(int),
        "joint_estimation": bool(enkf_params["joint_estimation"]),
        "parameter_estimation": bool(enkf_params["parameter_estimation"]),
        "state_estimation": bool(enkf_params["state_estimation"]),
}


obs_t, obs_idx, num_observations = UtilsFunctions(params).generate_observation_schedule(**kwargs)
print(obs_t)
kwargs["obs_index"] = obs_idx
params["number_obs_instants"] = num_observations

# --- observations parameters ---
sig_obs = np.zeros(params["nt"]+1)
for i in range(len(kwargs["obs_index"])):
    sig_obs[kwargs["obs_index"][i]] = float(enkf_params["sig_obs"])
params["sig_obs"] = sig_obs

# --- Generate True and Nurged States ---
PETSc.Sys.Print("Generating true state ...")
statevec_true = generate_true_state(
    np.zeros([params["nd"], params["nt"] + 1]), 
    params,  
    **kwargs  
)

PETSc.Sys.Print("Generating nurged state ...")
kwargs["a"] = a_p # Update accumulation with nurged accumulation
statevec_nurged = generate_nurged_state(
    np.zeros([params["nd"], params["nt"] + 1]), 
    params, 
    **kwargs  
)

# --- Synthetic Observations ---
PETSc.Sys.Print("Generating synthetic observations ...")
utils_funs = UtilsFunctions(params, statevec_true)
hu_obs = utils_funs._create_synthetic_observations(statevec_true,**kwargs)

# load data to be written to file
save_all_data(
    enkf_params=enkf_params,
    nofilter=True,
    t=kwargs["t"], b_io=np.array([b_in,b_out]),
    Lxy=np.array([Lx,Ly]),nxy=np.array([nx,ny]),
    statevec_true=statevec_true,
    statevec_nurged=statevec_nurged, 
    obs_max_time=np.array([params["obs_max_time"]]),
    obs_index=kwargs["obs_index"],
    w=hu_obs
)

# --- Initialize Ensemble ---
PETSc.Sys.Print("Initializing the ensemble ...")
statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full = initialize_ensemble(
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"]]),
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"], params["nt"] + 1]),
    params, **kwargs
)

# --- Run Data Assimilation ---
PETSc.Sys.Print("Running the model with Data Assimilation ...")

# Additional arguments for the EnKF
da_args = [
    enkf_params["parallel_flag"],
    params,
    np.eye(params["nd"]) * params["sig_Q"] ** 2,
    hu_obs,
    statevec_ens,
    statevec_bg,
    statevec_ens_mean,
    statevec_ens_full,
    enkf_params["commandlinerun"],
]

statevec_ens_full, statevec_ens_mean, statevec_bg = run_model_with_filter(
    enkf_params["model_name"],
    enkf_params["filter_type"],
    *da_args,
    **kwargs  
)


# load data to be written to file
save_all_data(
    enkf_params=enkf_params,
    statevec_ens_full=statevec_ens_full,
    statevec_ens_mean=statevec_ens_mean,
    statevec_bg=statevec_bg
)