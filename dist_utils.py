import os
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

def get_dist_info():
    GLOBAL_WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    GLOBAL_RANK = int(os.environ["RANK"])
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    return GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK


def ddp_setup(logger, use_cuda):
    """
    Setups torch.distributed and creates p_groups for training.
    In every node, 1 process (process with local_rank == 0) is assigned to the agents_group, the remaining processes are
    assigned to the env_workers_group. To get a better understanding check out the example below.
    agents_group processes have an instance of RNDAgent and perform optimizations.
    env_workers_group processes have an instance of the environment and perform interactions with it.


    Example:
        Available from torchrun:
            nnodes: number of nodes = 3
            nproc_per_node: number of processes per node = 4
        ---

        ************** NODE 0:
        LOCAL_RANK 0: GPUs --> agents_group
        LOCAL_RANK != 0: CPUs --> env_workers_group
        **************
        ...

        ************** NODE: 1:
        LOCAL_RANK 0: GPUs --> agents_group
        LOCAL_RANK != 0: CPUs --> env_workers_group
        **************
        ...

        ************** NODE: 2:
        LOCAL_RANK 0: GPUs --> agents_group
        LOCAL_RANK != 0: CPUs --> env_workers_group
        **************

        -node0-  -node1-   -node2-
        0,1,2,3  4,5,6,7  8,9,10,11    ||    agents_group_ranks=[0,4,8], env_workers_group_rank=[remaining ranks]
        *        *        *

    """

    GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK = get_dist_info()

    if use_cuda:
        assert torch.cuda.is_available() == True, "use_cuda:True is passed but cuda is not available !"

    if torch.cuda.is_available() and use_cuda:
        gpu_id = "cuda:" + str(LOCAL_RANK % torch.cuda.device_count())
        backend = "nccl"
    else:
        gpu_id = "cpu"
        backend = "gloo"

    if torch.cuda.is_available() and use_cuda and is_agents_group_member():
        torch.cuda.set_device(gpu_id) # GPU should only be used by the agents_group processes and each process should have a unique cuda device (otherwise see the error at: https://github.com/pytorch/torchrec/issues/328)
    init_process_group(backend="gloo")

    agents_group, env_workers_group, agents_group_global_ranks, env_workers_group_global_ranks, env_workers_group_per_node_global_ranks = create_process_groups_for_training(agents_group_backend=backend, env_workers_group_backend="gloo")

    logger.log_msg_to_both_console_and_file(f'Initializing process with global_rank: {GLOBAL_RANK}, local_rank: {LOCAL_RANK}, local_world_size: {LOCAL_WORLD_SIZE}, global_world_size: {GLOBAL_WORLD_SIZE}, [{gpu_id}], group: {"agents_group" if GLOBAL_RANK in agents_group_global_ranks else "env_workers_group"}, group_backend: {backend if GLOBAL_RANK in agents_group_global_ranks else "gloo"}, default_backend: {backend}')
    
    return GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK, gpu_id, agents_group, env_workers_group, agents_group_global_ranks, env_workers_group_global_ranks, env_workers_group_per_node_global_ranks

def is_agents_group_member():
    """
    returns True if the process belongs to the agents_group, False otherwise.
    """
    GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK = get_dist_info()
    return GLOBAL_RANK % LOCAL_WORLD_SIZE == 0


def create_process_groups_for_training(agents_group_backend="nccl", env_workers_group_backend="gloo"):
    GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK = get_dist_info()

    # Calculate ranks for the groups:
    agents_group_global_ranks = list(filter(lambda x: x != None, [rank if rank % LOCAL_WORLD_SIZE == 0 else None for rank in range(GLOBAL_WORLD_SIZE)]))
    env_workers_group_global_ranks = list(filter(lambda x: x != None, [rank if rank % LOCAL_WORLD_SIZE != 0 else None for rank in range(GLOBAL_WORLD_SIZE)]))
    env_workers_group_per_node_global_ranks = [rank for rank in range((GLOBAL_RANK//LOCAL_WORLD_SIZE) * LOCAL_WORLD_SIZE, ((GLOBAL_RANK//LOCAL_WORLD_SIZE) + 1) * LOCAL_WORLD_SIZE)][1:]

    # Create groups from the calculated ranks:
    agents_group = dist.new_group(
        ranks=agents_group_global_ranks, 
        backend=agents_group_backend
        ) # group for agent processes across nodes

    env_workers_group = dist.new_group(
        ranks=env_workers_group_global_ranks, 
        backend=env_workers_group_backend
        ) # group for env_worker processes across nodes

    env_workers_group_per_node = dist.new_group(
        ranks=env_workers_group_per_node_global_ranks, 
        backend=env_workers_group_backend
        ) # group for env_worker processes per node

    assert len(env_workers_group_per_node_global_ranks) == (LOCAL_WORLD_SIZE - 1)

    return agents_group, env_workers_group, agents_group_global_ranks, env_workers_group_global_ranks, env_workers_group_per_node_global_ranks


def distributed_cleanup():
    dist.destroy_process_group()