import os
import torch
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(logger, use_cuda):

    GLOBAL_WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    GLOBAL_RANK = int(os.environ["RANK"])
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available() and use_cuda:
        gpu_id = "cuda:" + LOCAL_RANK % torch.cuda.device_count()
        backend = "nccl"
    else:
        gpu_id = "cpu"
        backend = "gloo"

    if GLOBAL_RANK == 0:
        logger.log_msg_to_both_console_and_file(f'Using distributed communication backend: {backend}, with cuda available: {torch.cuda.is_available()}')
    logger.log_msg_to_both_console_and_file(f'Initializing process with global_rank: {GLOBAL_RANK}, local_rank: {LOCAL_RANK}, local_world_size: {LOCAL_WORLD_SIZE}, global_world_size: {GLOBAL_WORLD_SIZE}, [{gpu_id}]')

    init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    
    return GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK, gpu_id