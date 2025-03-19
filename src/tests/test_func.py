
import sys, os
import warnings
import numpy as np
warnings.filterwarnings('ignore') 

os.environ["OMP_NUM_THREADS"] = "1"

# firedrake imports
import firedrake
from firedrake import *


from mpi4py import MPI

Lx, Ly = 50e2, 12e2
nx, ny = 12,8

b_in, b_out = 20, -40
s_in, s_out = 850, 50


# mpi communicator
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# split communicator
# === each rank gets a color and a key (best for cases when Nens > size_world)
# color = rank % size
# key = rank
# comm = comm.Split(color, key)
# ========================================

# --- method 1: run model squentially with all ranks for Nens times
def func(unsplitted_comm,h):
    # mesh
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True,comm=unsplitted_comm)


    Q = firedrake.FunctionSpace(mesh, "CG", 2)
    V = firedrake.VectorFunctionSpace(mesh, "CG", 2)
    x,y = firedrake.SpatialCoordinate(mesh)

    b = firedrake.interpolate(b_in - (b_in - b_out) * x / Lx, Q)
    s0 = firedrake.interpolate(s_in - (s_in - s_out) * x / Lx, Q)
    h0 = firedrake.interpolate(s0 - b, Q)

    # get the size of the function space
    h = Function(Q)
    # print(f"Size of the function space: {h.dat.data.size} on rank {rank}")
    return h0.dat.data_ro

def forcast_squential(comm,Nens, ensemble):
    # ensemble = np.empty((425, Nens))
    for ens in range(Nens):
        comm.barrier() # make sure all ranks are synchronized before running the model
        h = func(comm)
        # comm.barrier() # make sure all ranks have run the model before gathering the results
        h_all = comm.gather(h, root=0)
        if rank == 0:
            h_all = np.hstack(h_all)
            # determine the shape of ensemble (this will be done one at t==0 not here, since the model willbe updating the ensemble at each time step)
            # if ens == 0:
            #     ensemble = np.empty((len(h_all), Nens))
            ensemble[:,ens] = h_all
            print(f"shape of the gathered array: {ensemble.shape}")
    return ensemble if rank == 0 else None

# --- method 2: case1:  run model in parallel: form Nens subcommunicators and each subcommunicator runs the 
#               model with all available ranks in the subcommunicator. All the subcommunicators run
#               simultaneously. The remaining ranks_rem= size_world - Nens are split evenly among the
#               subcommunicators. this is possible if ranks_rem splits evenly among the subcommunicators.
#               case2: if ranks_rem does not split evenly among the subcommunicators, then we use a different
#               approach. Form Nens subcommunicators, dynamically evenly assign ranks to the subcommunicators
#               e.g if Nens=3 and size_world=10, then 3 subcommunicators are formed with 3,3,3 ranks respectively. 
#               if ranks_rem <= 1, idle else the subcommunicators should wait for the other to finish before burrowing
#               the idle ranks to execute its run. we do this to ensure all rows are equally distributed among 
#               the subcommunicators to avoid issues with stacking back the gathered arrays for the next run.
#
# def forcast_parallel(comm,Nens):
#     color = rank % Nens # group the ranks into Nens subcommunicators
#     key = rank // Nens  # assign a key to each rank
#     subcomm = comm.Split(color, key)

#     # Ensure all ranks in subcomm are synchronized before running  
#     subcomm.Barrier()  # Sync all ranks in the subcommunicator
#     h = func(subcomm)
#     # Ensure all ranks finish before proceeding
#     subcomm.Barrier()

#     # print ranks, size and groups
#     print(f"rank: {rank}, size: {size}, subcomm: {subcomm.Get_rank()} of {subcomm.Get_size()}")
#     # each group of ranks in a subcomm, has its own dimension of the function space
#     h_all = subcomm.gather(h, root=0)
#     if rank == 0:
#         h_all = np.hstack(h_all)
#         # print(f"shape of the gathered array: {h_all.shape}")
#     return h_all if rank == 0 else None

def forecast_parallel_(comm, Nens, h):
    """
    Runs multiple ensemble members in parallel using `Nens` subcommunicators.
    
    Each subcommunicator runs in parallel, and its ranks work together.
    The results are gathered within each subcommunicator and returned.

    Parameters:
        comm: MPI communicator (MPI.COMM_WORLD typically)
        Nens: Number of ensemble members (subcommunicators)
        func: Function to execute within each subcommunicator

    Returns:
        h_all: Gathered result from all subcommunicators (only on rank 0 of `comm`)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if Nens > size:
        # raise ValueError("Nens must be less than or equal to the number of available ranks")
        # Divide ranks into `size` subcommunicators
        subcomm_size = min(size, Nens)  # Use at most `Nens` groups
        color = rank % subcomm_size  # Group ranks into `subcomm_size` subcommunicators
        key = rank // subcomm_size  # Ordering within each subcommunicator
        subcomm = comm.Split(color, key)

        sub_rank = subcomm.Get_rank()  # Rank within subcommunicator
        sub_size = subcomm.Get_size()  # Size of subcommunicator

        # Determine how many rounds of processing are needed
        rounds = (Nens + subcomm_size - 1) // subcomm_size  # Ceiling division

        # Store results for each round
        h_list = []
        for round_id in range(rounds):
            ensemble_id = color + round_id * subcomm_size  # Global ensemble index

            if ensemble_id < Nens:  # Only process valid ensembles
                print(f"Rank {rank} processing ensemble {ensemble_id} in round {round_id + 1}/{rounds}")

                # Ensure all ranks in the subcommunicator are synchronized before running
                subcomm.Barrier()

                # Run the function in parallel within each subcommunicator
                h = func(subcomm,h)  # Each subcommunicator runs the function independently

                # Ensure all ranks in subcomm finish execution
                subcomm.Barrier()

                # Gather results within each subcommunicator
                h_all_sub = subcomm.gather(h, root=0)

                # Only sub_rank == 0 of each subcommunicator processes the gathered results
                if sub_rank == 0:
                    h_all_sub = np.column_stack(h_all_sub)  # Stack gathered data
                    # print(f"Subcomm {color} (Rank {sub_rank}) reduced array shape: {h_all_sub.shape}")

                h_list.append(h_all_sub if sub_rank == 0 else None)  # Collect only from sub_rank 0

        # Gather reduced results from subcommunicators to the global rank 0
        h_all_global = comm.gather(h_list, root=0)  
    else:
        # Standard case where each rank maps 1-to-1 with an ensemble member
        # Create subcommunicators (each ensemble runs in parallel)
        color = rank % Nens  # Assign each rank to a subcommunicator group
        key = rank // Nens   # Unique ordering within each subcommunicator
        subcomm = comm.Split(color, key)

        sub_rank = subcomm.Get_rank()  # Rank within subcommunicator
        sub_size = subcomm.Get_size()  # Size of subcommunicator

        # Ensure all ranks in subcomm are synchronized before running
        subcomm.Barrier()

        # Run the function in parallel within each subcommunicator
        h = func(subcomm,h)  # Each subcommunicator runs the function independently

        # Ensure all ranks in subcomm finish execution
        subcomm.Barrier()

        # Gather results within each subcommunicator
        h_all_sub = subcomm.gather(h, root=0)
        # check the shape of the gathered arrays
        # if sub_rank == 0:
            #  print(f"[Rank {rank}] Gathered shapes: {[arr.shape for arr in h_all_sub]}")

        # Ensure only the rank 0 of each subcommunicator processes gathered results
        if sub_rank == 0:
            h_all_sub = np.hstack(h_all_sub)  # Stack gathered data
            print(f"Subcomm {color} (Rank {sub_rank}) gathered array shape: {h_all_sub.shape}")

        # Gather results from subcommunicators to the global rank 0 (optional)
        h_all_global = comm.gather(h_all_sub, root=0)

    if rank == 0:
        # h_all_global = np.hstack([arr for arr in h_all_global if arr is not None])  # Stack results from all subcommunicators
        # print(f"Final gathered array shape: {h_all_global.shape}")
        # Stack correctly along the second dimension (DOFs, Nens)

        # Remove None values (non-subcomm ranks) before stacking
        if Nens < size:
            h_all_global = [arr for arr in h_all_global if arr is not None]
        else:
            h_all_global = [arr for sublist in h_all_global for arr in sublist if arr is not None]
        h = np.column_stack(h_all_global)

        print(f"Final ensemble shape: {h.shape}")

        return h_all_global if rank == 0 else None
    

def forecast_parallel_doc(comm, Nens, h):
    """
    Runs multiple ensemble members in parallel using `Nens` subcommunicators.
    
    If `Nens > size`, ensembles are handled in batches using the same subcommunicators.

    Parameters:
        comm: MPI communicator (MPI.COMM_WORLD typically)
        Nens: Number of ensemble members
        h: Initial state for processing

    Returns:
        h_all_global: Final gathered results (only on rank 0)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if Nens > size:
        # Divide ranks into `size` subcommunicators
        subcomm_size = min(size, Nens)  # Use at most `Nens` groups
        color = rank % subcomm_size  # Group ranks into `subcomm_size` subcommunicators
        key = rank // subcomm_size  # Ordering within each subcommunicator
        subcomm = comm.Split(color, key)

        sub_rank = subcomm.Get_rank()  # Rank within subcommunicator
        sub_size = subcomm.Get_size()  # Size of subcommunicator

        # Determine the number of processing rounds
        rounds = (Nens + subcomm_size - 1) // subcomm_size  # Ceiling division

        h_list = []
        for round_id in range(rounds):
            ensemble_id = color + round_id * subcomm_size  # Global ensemble index

            if ensemble_id < Nens:  # Only process valid ensembles
                print(f"Rank {rank} processing ensemble {ensemble_id} in round {round_id + 1}/{rounds}")

                subcomm.Barrier()  # Synchronize before execution

                # Run the function in parallel within each subcommunicator
                h = func(subcomm, h)  # Each subcommunicator runs the function independently

                subcomm.Barrier()  # Ensure all ranks finish

                # Gather results within each subcommunicator
                h_all_sub = subcomm.gather(h, root=0)

                if sub_rank == 0:
                    h_all_sub = np.column_stack(h_all_sub)  # Stack gathered data

                h_list.append(h_all_sub if sub_rank == 0 else None)  # Collect only from sub_rank 0

        # Gather results from subcommunicators to the global rank 0
        h_all_global = comm.gather(h_list, root=0)  

    else:
        # Standard case where each rank maps 1-to-1 with an ensemble
        color = rank % Nens  
        key = rank // Nens   
        subcomm = comm.Split(color, key)

        sub_rank = subcomm.Get_rank()
        sub_size = subcomm.Get_size()

        subcomm.Barrier()  # Synchronize

        h = func(subcomm, h)  

        subcomm.Barrier()

        h_all_sub = subcomm.gather(h, root=0)

        if sub_rank == 0:
            h_all_sub = np.hstack(h_all_sub)  
            print(f"Subcomm {color} (Rank {sub_rank}) gathered array shape: {h_all_sub.shape}")

        h_all_global = comm.gather(h_all_sub, root=0)

    if rank == 0:
        # h_all_global = [arr for sublist in h_all_global for arr in sublist if arr is not None]
        if Nens < size:
            h_all_global = [arr for arr in h_all_global if arr is not None]
        else:
            h_all_global = [arr for sublist in h_all_global for arr in sublist if arr is not None]
        # h = np.column_stack(h_all_global)
        h = np.column_stack(h_all_global)

        print(f"Final ensemble shape: {h.shape}")

        return h_all_global if rank == 0 else None


def forecast_parallel_dynamic(comm, Nens, h):
    """
    Runs multiple ensemble members in parallel with dynamic load balancing.
    
    Instead of statically assigning ensembles, workers request work dynamically.

    Parameters:
        comm: MPI communicator (MPI.COMM_WORLD)
        Nens: Total number of ensemble members
        h: Initial state for processing

    Returns:
        h_all_global: Final gathered results (only on rank 0)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 manages the workload queue
    if rank == 0:
        work_queue = list(range(Nens))  # List of remaining ensembles
        results = {}  # Dictionary to store ensemble results

    # While there are still ensemble members to process
    while True:
        # Rank 0 assigns tasks dynamically
        if rank == 0:
            # Check if there's work left to assign
            if work_queue:
                task = work_queue.pop(0)  # Assign the first available ensemble
            else:
                task = -1  # No work left

        else:
            task = None

        # Broadcast task to all ranks
        task = comm.bcast(task, root=0)

        # If no tasks remain, all processes exit
        if task == -1:
            break

        # All ranks execute their assigned ensemble member
        print(f"Rank {rank} processing ensemble {task}")
        # h_result = func(comm, h, task)  # Function executes task
        h_result = func(comm,h)

        # Rank 0 collects results dynamically
        h_all_sub = comm.gather(h_result, root=0)

        if rank == 0:
            results[task] = np.column_stack(h_all_sub)  # Store results

    # Rank 0 finalizes results
    if rank == 0:
        # Stack results in order of ensemble index
        h_all_global = np.column_stack([results[i] for i in sorted(results.keys())])
        print(f"Final ensemble shape: {h_all_global.shape}")
        return h_all_global

    return None  # Non-root ranks return None

# run the main function
if __name__ == "__main__":
    import time
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    Nens = 10
    ensemble= np.empty((425, Nens))
    start = time.time()
    # forcast_squential(comm,Nens, ensemble)
    # forcast_parallel(comm,Nens)
    # forecast_parallel_doc(comm, Nens, ensemble)
    forecast_parallel_dynamic(comm, Nens, ensemble)

    stop = time.time()
    # get total time from all ranks
    total_time = comm.reduce(stop-start, op=MPI.MAX, root=0)
    if rank == 0:
        print(f" Total time taken using {size} porcs: {total_time/60} minutes")

