import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed() -> dict:
    """Initialize distributed training environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return {
        "local_rank": local_rank,
        "world_size": world_size,
        "rank": rank,
        "is_main": rank == 0,
        "distributed": world_size > 1,
    }


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()