import os
from typing import Dict, List, Sequence, Tuple, Union

import h5py
import openslide
import torch

from . import register_dataset
from .base import BasePatchDataset


@register_dataset("clam")
class CLAMSlideDataset(BasePatchDataset):
    dataset_name = "clam"

    def __init__(
        self,
        wsi_dir: str,
        seg_dirs: Union[str, Sequence[str]],
        patch_size: int = 256,
        transform=None,
    ):
        super().__init__(transform=transform)
        self.wsi_dir = os.path.abspath(wsi_dir)
        self.seg_dirs = self._normalize_dirs(seg_dirs)
        self.patch_size = patch_size
        self._patch_index = self._build_index()
        self._open_slides: Dict[str, openslide.OpenSlide] = {}

    def _normalize_dirs(self, seg_dirs: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(seg_dirs, str):
            return [seg_dirs]
        return [os.path.abspath(path) for path in seg_dirs]

    def _build_index(self) -> List[Tuple[str, int, int]]:
        index: List[Tuple[str, int, int]] = []
        for seg_dir in self.seg_dirs:
            patch_dir = os.path.join(seg_dir, "patches")
            if not os.path.isdir(patch_dir):
                continue
            for file_name in sorted(os.listdir(patch_dir)):
                if not file_name.endswith(".h5"):
                    continue
                slide_id = os.path.splitext(file_name)[0]
                h5_path = os.path.join(patch_dir, file_name)
                with h5py.File(h5_path, "r") as handle:
                    if "coords" not in handle:
                        continue
                    coords = handle["coords"][:]
                for x, y in coords:
                    index.append((slide_id, int(x), int(y)))
        return index

    def __len__(self) -> int:
        return len(self._patch_index)

    def __getitem__(self, idx: int):
        slide_id, x, y = self._patch_index[idx]
        slide = self._get_slide(slide_id)
        region = slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB")
        image = self._apply_transform(region)
        coords = torch.tensor([x, y], dtype=torch.long)
        return image, coords, slide_id

    def _get_slide(self, slide_id: str):
        if slide_id not in self._open_slides:
            slide_path = os.path.join(self.wsi_dir, f"{slide_id}.tif")
            if not os.path.exists(slide_path):
                raise FileNotFoundError(f"Slide {slide_id} not found under {self.wsi_dir}")
            self._open_slides[slide_id] = openslide.OpenSlide(slide_path)
        return self._open_slides[slide_id]

    def close(self):
        for slide in self._open_slides.values():
            slide.close()
        self._open_slides.clear()

    def __del__(self):
        self.close()
