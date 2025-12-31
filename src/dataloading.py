import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets import get_dataset_class


def create_dataloader(
    dataset_name: str,
    dataset_kwargs: Dict[str, Any],
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    sampler_seed: int = 0,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> DataLoader:
    dataset_cls = get_dataset_class(dataset_name)
    dataset = dataset_cls(**dataset_kwargs)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=_resolve_world_size(world_size),
            rank=_resolve_rank(rank),
            shuffle=shuffle,
            drop_last=drop_last,
            seed=sampler_seed,
        )
        shuffle = False

    effective_persistent = persistent_workers
    if effective_persistent is None:
        effective_persistent = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=effective_persistent,
    )
    return loader


def _resolve_world_size(world_size: Optional[int]) -> int:
    if world_size is not None:
        return world_size
    env_world = os.getenv("WORLD_SIZE")
    if env_world:
        return int(env_world)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _resolve_rank(rank: Optional[int]) -> int:
    if rank is not None:
        return rank
    env_rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if env_rank:
        return int(env_rank)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0
