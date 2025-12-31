from typing import Callable, Dict, Type

from .base import BasePatchDataset

DATASET_REGISTRY: Dict[str, Type[BasePatchDataset]] = {}


def register_dataset(name: str) -> Callable[[Type[BasePatchDataset]], Type[BasePatchDataset]]:
    def decorator(cls: Type[BasePatchDataset]):
        DATASET_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_dataset_class(name: str) -> Type[BasePatchDataset]:
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' is not registered")
    return DATASET_REGISTRY[key]


from .clam import CLAMSlideDataset  # noqa: E402

__all__ = [
    "BasePatchDataset",
    "register_dataset",
    "get_dataset_class",
    "DATASET_REGISTRY",
    "CLAMSlideDataset",
]
