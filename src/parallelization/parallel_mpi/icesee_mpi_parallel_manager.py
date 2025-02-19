# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-02-10
# @description: Initializes model communicators: size, rank, and variables for 
#               MPI parallelization to be shared by all modules and the ICESS 
#               package.
# =============================================================================

from mpi4py import MPI
import numpy as np
import math
import copy

class ParallelManager:
    """
    This class provides variables for MPI parallelization to be shared 
    between model-related routines.

    Implements Singleton pattern to ensure a single instance is used.
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Singleton pattern to ensure only one instance of ParallelManager exists."""
        if cls._instance is None:
            cls._instance = super(ParallelManager, cls).__new__(cls)
            cls._instance._initialized = False  # Track initialization state
        return cls._instance

    def __init__(self):
        """Initializes MPI communicator, size, rank, and variables."""
        if self._initialized:
            return  # Avoid re-initialization

        self._initialized = True  # Mark as initialized
        self._initialize_variables()

    def _initialize_variables(self):
        """Initializes default variables for MPI communication."""
        # Global communicator
        self.COMM_WORLD = None  # MPI communicator for all PEs
        self.rank_world = None  # Rank in MPI_COMM_WORLD
        self.size_world = None  # Size of MPI_COMM_WORLD

        # Model communicator (forecast step)
        self.COMM_model = None  # MPI communicator for model tasks
        self.rank_model = None  # Rank in the COMM_model communicator
        self.size_model = None  # Size of the COMM_model communicator

        # ICESS communicator for the analysis step
        self.COMM_filter = None  # MPI communicator for filter PEs
        self.rank_filter = None  # Rank in COMM_filter
        self.size_filter = None  # Number of PEs in COMM_filter
        self.n_filterpes = 1  # Number of parallel filter analysis tasks

        # ICESS communicator for coupling filter and model and also for initializations
        self.COMM_couple = None  # MPI communicator for coupling filter and model
        self.rank_couple = None  # Rank in COMM_couple
        self.size_couple = None  # Number of PEs in COMM_couple

        # ICESS variables
        self.n_modeltasks = None  # Number of parallel model tasks
        self.modelpe = False  # Whether PE is in COMM_model
        self.filterpe = False  # Whether PE is in COMM_filter
        self.task_id = None  # Index of model task (1,...,n_modeltasks)
        self.MPIerr = 0  # Error flag for MPI
        self.local_size_model = None  # Number of PEs per ensemble

        # Shared memory variables
        self. node_comm = None  # MPI communicator for shared memory
        self.node_rank = None  # Rank in the node communicator
        self.node_size = None  # Size of the node communicator
        self.shared_array = None  # Shared memory array
        self.win = None  # MPI window for shared memory
        self.mem_buf = None  # Memory buffer for shared memory
        self.mem_size = None  # Size of the shared memory buffer (Nens*state_block_size)
        self.local_start = None  # Start index of local ensemble
        self.local_stop = None  # Stop index of local ensemble


    def init_parallel_non_mpi_model(self):
        """
        Initializes MPI in a non-MPI model.

        Determines the number of PEs (`size_world`) and the rank of a PE (`rank_world`).
        The model is executed within the scope of `COMM_model`.
        """
        if not MPI.Is_initialized():
            MPI.Init()

        self.COMM_model = MPI.COMM_WORLD
        self.size_world = self.COMM_model.Get_size()
        self.rank_world = self.COMM_model.Get_rank()

        # Use MPI_COMM_WORLD as the model communicator
        self.size_model = self.size_world
        self.rank_model = self.rank_world

        #  return the initialized parallel manager
        return self

    # --- Parallel load distribution ---
    def ensembles_load_distribution(self, ensemble,comm):
        """
        Distributes ensemble members among MPI processes based on rank and size."""

        global_shape,Nens = ensemble.shape
        rank = comm.Get_rank()
        size = comm.Get_size()

        # --- Properly Distribute Tasks for All Cases ---
        if Nens > self.size_world:
            # Case 1: More ensembles than processes → Distribute as evenly as possible
            mem_per_task = Nens // size  # Base number of tasks per process
            remainder = Nens % size       # Extra tasks to distribute

            if rank < remainder:
                # the first remainder gets mem_per_task+1 tasks each
                start = rank * (mem_per_task + 1)
                stop = start + (mem_per_task + 1)
                # stop = start + mem_per_task
            else:
                #  the remaining (size - remainder) get mem_per_task tasks each
                start = rank * mem_per_task + remainder
                stop = start + mem_per_task
                # stop = start + mem_per_task-1
        else:
            # Case 2: More processes than ensembles → Assign at most one task per rank
            if rank < Nens:
                start, stop = rank, rank + 1
            else:
                # Extra ranks do nothing
                start, stop = 0, 0

        # --- Ensure All Ranks Participate (No Deadlocks) ---
        if start == stop:
            print(f"[Rank {rank}] No work assigned. Waiting at barrier.")
        else:
            print(f"[Rank {rank}] Processing ensembles {start} to {stop}")
        # return start, stop

        # form  local ensembles
        # ensemble_local = np.zeros((global_shape, stop-start))
        ensemble_local = ensemble[:global_shape,start:stop]
        # for memory issues return a deepcopy of the ensemble_local
        return copy.deepcopy(ensemble_local), start, stop
    
    # --- state vector load distribution ---
    def state_vector_load_distribution(self, state_vector,comm):
        """
        Distributes state vector among MPI processes based on rank and size."""

        global_shape, Nens = state_vector.shape
        rank = comm.Get_rank()
        size = comm.Get_size()

        # --- Properly Distribute Tasks for All Cases ---
        if global_shape > self.size_world:
           workloads = [global_shape // size for i in range(size)]
           for i in range(global_shape % size):
               workloads[i] += 1
           start = 0
           for i in range(rank):
               start += workloads[i]
           stop = start + workloads[rank]
        else:
            # Case 2: More processes than state variables → Assign at most one task per rank
            if rank < global_shape:
                start, stop = rank, rank + 1
            else:
                # Extra ranks do nothing
                start, stop = 0, 0
                
        if False:
            chuck_size = global_shape // size
            remainder = global_shape % size
            if rank < remainder:
                start = rank * (chuck_size + 1)
                stop = start + (chuck_size + 1)
            else:
                start = rank * chuck_size + remainder
                stop = start + chuck_size
        
        state_vector_local = copy.deepcopy(state_vector[start:stop,:])
        
        return state_vector_local, start, stop
    
    # --- memory formulation ---
    def memory_usage(self, global_shape, Nens, bytes_per_element=8):
        """
        Computes the memory usage of an ensemble in bytes."""
        return global_shape * Nens * bytes_per_element/1e9  # Convert to GB

    # ---- Collective Communication Operations ----
    # -- method to gather data from all ranks
    def all_gather_data(self, comm, data):
        """
        Gathers data from all ranks using collective communication."""

        size = comm.Get_size()  # Number of MPI processes
        data = np.asarray(data)  # Ensure data is a NumPy array

        # Get the shape of the incoming data
        local_shape = data.shape  # Should be (18915, 16) per rank
        global_shape = (size,) + local_shape  # Expected (size, 18915, 16)

        # Allocate the buffer for gathering
        gathered_data = np.zeros(global_shape, dtype=np.float64)

        # Use Allgather to collect data from all ranks
        comm.Allgather([data, MPI.DOUBLE], [gathered_data, MPI.DOUBLE])

        return gathered_data
    
    # -- method to scatter data to all ranks
    def scatter_data(self,comm, data):
        """
        Scatters data from one rank to all other ranks using collective communication."""

        rank = comm.Get_rank()
        size = comm.Get_size()

        # Ensure data is correctly divided
        local_rows = data.shape[0] // size
        recv_data = np.zeros((local_rows, data.shape[1]), dtype=np.float64)

        # Scatter from Rank 0
        comm.Scatter([data, MPI.DOUBLE], [recv_data, MPI.DOUBLE], root=0)

        return recv_data

    
    # -- method to Bcast data to all ranks
    def broadcast_data(self, comm, data, root=0):
        """
        Broadcasts data from one rank to all other ranks using collective communication."""
        data = np.asarray(data)  # Ensure it's an array
        comm.Bcast([data, MPI.DOUBLE], root=root)
        return data
    
    # -- method to exchange data between all ranks
    def alltoall_exchange(self, comm, data):
        """
        Exchanges data between all ranks using collective communication."""

        size = comm.Get_size()
        local_rows = data.shape[0] // size

        # Each rank prepares send buffer with `size` chunks
        sendbuf = np.split(data, size, axis=0)
        sendbuf = np.concatenate(sendbuf, axis=0)  # Flatten for Alltoall

        # Allocate receive buffer
        recvbuf = np.empty_like(sendbuf)

        # Perform Alltoall communication
        comm.Alltoall([sendbuf, MPI.DOUBLE], [recvbuf, MPI.DOUBLE])

        return recvbuf.reshape(size, local_rows, -1)  # Reshape into proper format

    # --- Point-to-Point Communication Operations ---
    def send_receive_data(self, comm, local_data, source=0, dest=1):
        """
        Sends data from one rank to another using point-to-point communication."""
        rank = comm.Get_rank()

        if rank == source:
            comm.Send([local_data, MPI.DOUBLE], dest=dest)
            print(f"[Rank {rank}] Sent data to Rank {dest}")

        elif rank == dest:
            recv_data = np.empty_like(local_data)
            comm.Recv([recv_data, MPI.DOUBLE], source=source)
            print(f"[Rank {rank}] Received data from Rank {source}")
            return recv_data
        
    # ==== Analysis parallel computations ====
    # -- method to compute the ensemble mean
    def compute_covariance(self,ensemble, mean, node_comm):
        """
        Computes the ensemble covariance matrix in parallel using shared memory.
        """
        N, state_dim = ensemble.shape
        local_ensemble_centered = ensemble - mean  # Center the ensemble
        local_cov = local_ensemble_centered.T @ local_ensemble_centered / (N - 1)

        # Sum covariance across ranks
        global_cov = np.zeros_like(local_cov)
        node_comm.Allreduce([local_cov, MPI.DOUBLE], [global_cov, MPI.DOUBLE], op=MPI.SUM)
        return global_cov
    
    # -- method to compute the ensemble mean
    def compute_mean(self,ensemble, node_comm):
        """
        Computes the ensemble mean in parallel using shared memory.
        """
        local_mean = np.mean(ensemble, axis=1)  # Each rank computes mean over its part
        global_mean = np.zeros_like(local_mean)
        node_comm.Allreduce([local_mean, MPI.DOUBLE], [global_mean, MPI.DOUBLE], op=MPI.SUM)
        global_mean /= node_comm.size  # Compute the final mean
        return global_mean
    
    # --- method for matrix-vector multiplication
    def matvec_product(self, matrix, vector, node_comm):
        """
        Computes the matrix-vector product in parallel using shared memory.
        """
        local_product = matrix @ vector
        global_product = np.zeros_like(local_product)
        node_comm.Allreduce([local_product, MPI.DOUBLE], [global_product, MPI.DOUBLE], op=MPI.SUM)
        return global_product
    
    # --- method for matrix-matrix multiplication
    def matmat_product(self, matrix1, matrix2, node_comm):
        """
        Computes the matrix-matrix product in parallel using shared memory.
        """
        local_product = matrix1 @ matrix2
        global_product = np.zeros_like(local_product)
        node_comm.Allreduce([local_product, MPI.DOUBLE], [global_product, MPI.DOUBLE], op=MPI.SUM)
        return global_product
    
    # --- method for Kalamn gain computation
    def compute_kalman_gain(self, ensemble, obs_cov, obs_op, state_dim, node_comm):
        """
        Computes the Kalman gain in parallel using shared memory.
        """
        N, _ = ensemble.shape
        local_cov = ensemble.T @ ensemble / (N - 1)  # Compute ensemble covariance
        local_gain = local_cov @ obs_op.T @ np.linalg.inv(obs_op @ local_cov @ obs_op.T + obs_cov)
        global_gain = np.zeros((state_dim, obs_cov.shape[0]))
        node_comm.Allreduce([local_gain, MPI.DOUBLE], [global_gain, MPI.DOUBLE], op=MPI.SUM)
        return global_gain



def icesee_mpi_parallelization(Nens, global_shape=1024, n_modeltasks=None, screen_output=True):
    """
    Initializes MPI communicators for parallel model tasks and determines `n_modeltasks` dynamically if not provided.

    Parameters:
        - Nens (int): Number of ensemble members.
        - n_modeltasks (int, optional): Number of parallel model tasks. If `None`, it is auto-determined.
        - screen_output (bool): Whether to print MPI configuration.

    Returns:
        - parallel_manager (ParallelManager): Initialized parallel manager instance.
    """

    parallel_manager = ParallelManager()
    COMM_WORLD = MPI.COMM_WORLD

    # Initialize MPI processing element (PE) information
    parallel_manager.size_world = COMM_WORLD.Get_size()
    parallel_manager.rank_world = COMM_WORLD.Get_rank()

    # --- check if size_world is divisible by Nens for Nens > size_world ---    
    # if Nens > parallel_manager.size_world:
    #     if parallel_manager.size_world < 6:
    #         n_modeltasks = 1

        # size_world should be divisible by Nens
        # if Nens % parallel_manager.size_world != 0:
            # effective_nprocs = find_largest_divisor(Nens, parallel_manager.size_world)
            # if parallel_manager.rank_world == 0:
            #     print(f"\n [ICESEE] Adjusting number of MPI processes from {parallel_manager.size_world} to {effective_nprocs} for even distribution of Ensemble ({Nens})\n")

            # # split MPI processes: only the first effective_nprocs will be used
            # color = 0 if parallel_manager.rank_world < effective_nprocs else 1
            # COMM_WORLD = COMM_WORLD.Split(color, key=parallel_manager.rank_world)
            # parallel_manager.size_world = COMM_WORLD.Get_size()
            # parallel_manager.rank_world = COMM_WORLD.Get_rank()
            # # redefine the n_modeltasks
            # n_modeltasks = max(2, n_modeltasks) if n_modeltasks is not None else None
            # # check if comm_world is redefined
            # if parallel_manager.rank_world == 0:
            #     print(f"\n [ICESEE] Reinitialized communicators with { parallel_manager.size_world} MPI processes\n")
            # raise ValueError(f"Number of MPI processes ({parallel_manager.size_world}) must be divisible by the number of ensemble members ({Nens})")

    # elif parallel_manager.size_world > Nens:
        # size_world should be divisible by Nens
        # if parallel_manager.size_world % Nens != 0:
            # raise ValueError(f"Number of MPI processes ({parallel_manager.size_world}) must be divisible by the number of ensemble members ({Nens})")

    # Display initialization message
    # if parallel_manager.rank_world == 0:
    #     print("\n [ICESEE] Initializing communicators...\n")

    # --- Determine `n_modeltasks` dynamically if not provided ---
    if n_modeltasks is None:
        # if Nens > parallel_manager.size_world:
        #     # Case: More ensembles than processes
        #     n_modeltasks = min(Nens // (parallel_manager.size_world // 2), parallel_manager.size_world // 2)
        # elif Nens <= parallel_manager.size_world:
        #     # Case: Fewer ensembles than processes
        #     n_modeltasks = min(Nens, parallel_manager.size_world // 4)
        # else:
        #     # Case: Roughly equal processes and ensembles
        #     n_modeltasks = min(Nens, parallel_manager.size_world // 2)

        # n_modeltasks = parallel_manager.size_world/(np.log2(parallel_manager.size_world+1))
        # n_modeltasks = int(parallel_manager.size_world / max(1, int(np.ceil(Nens / parallel_manager.size_world))))
        n_modeltasks = math.gcd(Nens, parallel_manager.size_world)
        
        # Ensure `n_modeltasks` is at least 1
        n_modeltasks = max(1, n_modeltasks)

    
    # else:
        # Ensure `n_modeltasks` does not exceed available resources
        # parallel_manager.n_modeltasks = min(n_modeltasks, parallel_manager.size_world, Nens)

    # update the parallel_manager with the number of model tasks
    parallel_manager.n_modeltasks = n_modeltasks

    # --- Check number of parallel ensemble tasks ---
    if parallel_manager.n_modeltasks > parallel_manager.size_world:
        parallel_manager.n_modeltasks = parallel_manager.size_world
    
    # Adjust `n_modeltasks` to ensemble size
    if Nens > 0 and parallel_manager.n_modeltasks > Nens:
        parallel_manager.n_modeltasks = Nens

    # --- Print Optimization Choice ---
    if parallel_manager.rank_world == 0:
        print(f"[ICESEE] Optimized Model Tasks: {parallel_manager.n_modeltasks} "
              f"(for {parallel_manager.size_world} MPI ranks, {Nens} ensembles)")

    # Generate communicator for ensemble tasks
    COMM_ensemble = COMM_WORLD
    size_ens = parallel_manager.size_world
    rank_ens = parallel_manager.rank_world

    # Allocate and distribute PEs per model task
    parallel_manager.local_size_model = np.full(
        parallel_manager.n_modeltasks,
        parallel_manager.size_world // parallel_manager.n_modeltasks,
        dtype=int
    )
    remainder = parallel_manager.size_world % parallel_manager.n_modeltasks
    parallel_manager.local_size_model[:remainder] += 1

    # Assign each PE to a model task
    pe_index = 0
    for i in range(parallel_manager.n_modeltasks):
        for j in range(parallel_manager.local_size_model[i]):
            if rank_ens == pe_index:
                parallel_manager.task_id = i + 1  # Convert to 1-based index
                break
            pe_index += 1
        if parallel_manager.task_id is not None:
            break  # Exit outer loop

    # Create COMM_MODEL communicator
    parallel_manager.COMM_model = COMM_ensemble.Split(color=parallel_manager.task_id, key=rank_ens)
    parallel_manager.size_model = parallel_manager.COMM_model.Get_size()
    parallel_manager.rank_model = parallel_manager.COMM_model.Get_rank()

    # Assign filter PEs
    parallel_manager.filterpe = (parallel_manager.task_id == 1)

    # Create COMM_FILTER communicator
    my_color = parallel_manager.task_id if parallel_manager.filterpe else MPI.UNDEFINED
    parallel_manager.COMM_filter = COMM_WORLD.Split(color=my_color, key=parallel_manager.rank_world)
    
    if parallel_manager.filterpe:
        parallel_manager.size_filter = parallel_manager.COMM_filter.Get_size()
        parallel_manager.rank_filter = parallel_manager.COMM_filter.Get_rank()

    # Create COMM_COUPLE communicator
    color_couple = parallel_manager.rank_model + 1
    parallel_manager.COMM_couple = COMM_ensemble.Split(color=color_couple, key=parallel_manager.rank_world)
    parallel_manager.rank_couple = parallel_manager.COMM_couple.Get_rank()
    parallel_manager.size_couple = parallel_manager.COMM_couple.Get_size()

    # --- split COMM_WORLD into sub-communicators able to use shared memory. 
    # - [node_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)] 
    # - and then use MPI.Win.Allocate_shared() within each node-local sub-communicator to easily access shared memory. (ranks can efficiently communicate with each othercwithout excessive communication overhead)
    
    # define the size of the shared memory (number of elements)
    # get the ensemble size
    parallel_manager.mem_size = Nens * global_shape

    # get the size(in bytes) of a double precision floating point number
    disp_unit = MPI.DOUBLE.Get_size()

    # split COMM_WORLD into node-local sub-communicators
    parallel_manager.node_comm = COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

    # allocate shared memory
    if False:
        parallel_manager.win = MPI.Win.Allocate_shared(
            parallel_manager.mem_size * disp_unit if parallel_manager.node_comm.rank == 0 else 0,
            disp_unit, 
            comm = parallel_manager.node_comm)
        
        # querying shared memory buffer
        parallel_manager.mem_buf, itemsize = parallel_manager.win.Shared_query(0)
        assert itemsize == MPI.DOUBLE.Get_size()

        # convert the memory buffer to a numpy array
        parallel_manager.mem_buf = np.array(parallel_manager.mem_buf, dtype='B', copy=False)

        # create a shared array
        # parallel_manager.shared_array = np.ndarray(buffer=parallel_manager.mem_buf, dtype='d', shape=( parallel_manager.mem_size,))
        parallel_manager.shared_array = np.ndarray(buffer=parallel_manager.mem_buf, dtype='d', shape=(global_shape, Nens))

        parallel_manager.node_rank = parallel_manager.node_comm.Get_rank()
        parallel_manager.node_size = parallel_manager.node_comm.Get_size()

        # perform ensemble analysis step using shared memory
        parallel_manager.local_start = parallel_manager.node_comm.rank * (Nens // parallel_manager.node_comm.size)
        parallel_manager.local_stop = (parallel_manager.node_comm.rank + 1) * (Nens // parallel_manager.node_comm.size)

    # Display MPI ICESEE Configuration
    if screen_output:
        display_pe_configuration(parallel_manager)

    return parallel_manager


def display_pe_configuration(parallel_manager):
    """
    Displays MPI parallel configuration in a structured format.
    Uses tabulate for better readability.
    """
    from tabulate import tabulate

    COMM_WORLD = MPI.COMM_WORLD
    rank = parallel_manager.rank_world

    rank_info = [
        rank,
        parallel_manager.rank_filter if parallel_manager.filterpe else "-",
        parallel_manager.task_id,
        parallel_manager.rank_model,
        parallel_manager.COMM_couple.Get_rank(),
        parallel_manager.node_rank,
        "✔" if parallel_manager.filterpe else "✘"
    ]

    # Gather all rank data
    pe_data = COMM_WORLD.gather(rank_info, root=0)

    COMM_WORLD.Barrier()

    if rank == 0:
        print("\n[ICESEE] Parallel Execution Configuration:\n")
        headers = ["World Rank", "Filter Rank", "Model Task", "Model Rank", "Couple Rank", "Shared Rank", "Filter PE"]
        print(tabulate(pe_data, headers=headers, tablefmt="double_grid"))

    COMM_WORLD.Barrier()


def find_largest_divisor(Nens, size_world):
    """
    Finds the largest divisor of `Nens` that is less than or equal to `size_world`.
    
    Parameters:
        - Nens (int): Number of ensemble members.
        - size_world (int): Total number of MPI processes.

    Returns:
        - best_size (int): Largest divisor of `Nens` ≤ `size_world`.
    """
    best_size = 1  # Start with the minimum valid size
    for i in range(1, size_world + 1):
        if Nens % i == 0:
            best_size = i  # Update with the largest divisor found
    return best_size


# --- Main Execution (test parallel manager) ---
if __name__ == "__main__":
    Nens = 8  # Number of ensemble members
    n_modeltasks = 2  # Set to `None` for automatic determination
    global_shape = 1024  # Number of state variables

    parallel_manager = icesee_mpi_parallelization(Nens=Nens, global_shape=global_shape, n_modeltasks=n_modeltasks, screen_output=True)

    # Finalize MPI
    if MPI.Is_initialized():
        MPI.Finalize()
