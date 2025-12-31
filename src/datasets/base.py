from abc import ABC
from torch.utils.data import Dataset


class BasePatchDataset(Dataset, ABC):
    dataset_name = "base"

    def __init__(self, transform=None):
        self.transform = transform

    def _apply_transform(self, image):
        return self.transform(image) if self.transform else image

    def close(self):
        """Hook for datasets that keep file handles open."""
        return None
