# Pathology Foundation Model

Modernized pipeline for downloading Whole Slide Images (WSI), generating CLAM patches, and consuming them with PyTorch `Dataset`/`DataLoader` primitives that are ready for distributed training and multiple dataset backends.

## Quickstart

```bash

git clone https://github.com/mahmoodlab/CLAM.git

pip install -r requirements.txt
python main.py \
  --file-path CAMELYON16/images/ \
  --download-num 1 \
  --batch-size 64
```

Negative values for `--download-num` fetch all available `.tif` slides for the specified prefix.

## Project Layout

- `src/`
  - `pipeline.py`: S3 download + CLAM patch orchestration with configurable cache roots and patch scripts.
  - `datasets/`: Dataset registry plus the default `CLAMSlideDataset` implementation.
  - `dataloading.py`: Centralized dataloader factory that understands distributed sampling.
- `CLAM/`: Upstream CLAM repository (unchanged, still used for patch extraction).
- `main.py`: Reference CLI that ties the pipeline and dataloader pieces together.

Add new datasets by dropping a module under `pathology_foundation_model/datasets/`, decorating the class with `@register_dataset("my_dataset")`, and passing `--dataset my_dataset` to `main.py`.

## Distributed & Multi-Dataset DataLoader

The helper `create_dataloader` can wrap any registered dataset class and will automatically insert a `DistributedSampler` when `distributed=True`. You can override `rank`/`world_size` or rely on `torch.distributed` / environment variables (`RANK`, `WORLD_SIZE`) that are typically set by launchers such as `torchrun` or `accelerate`.

Example (two GPUs):

```bash
torchrun --nproc_per_node=2 main.py \
  --distributed \
  --batch-size 128 \
  --num-workers 8
```

Remember to call `dataloader.sampler.set_epoch(epoch)` inside your training loop when shuffling in DDP mode.

## Pipeline Configuration

`PipelineConfig` exposes:

- `bucket_name`: defaults to `camelyon-dataset`.
- `cache_root`: overrides `HF_DATASETS_CACHE`/`./cache`.
- `patch_script`: path to `create_patches_fp.py` (keep pointing at `CLAM/` unless you customize CLAM).
- `patch_size`, `seg_subdir`, `patches_subdir`, and `extra_patch_args`.

Pass overrides through the CLI flags in `main.py` or instantiate `PipelineConfig` directly if you embed the package.

```python
from pathology_foundation_model import PipelineConfig, setup_pipeline

cfg = PipelineConfig(cache_root="/data/cache", patch_size=512)
result = setup_pipeline("CAMELYON16/images/", download_num=5, config=cfg)
```

`PipelineResult` exposes `wsi_dir`, `seg_dirs`, `downloaded_slides`, and `has_data` so you can wire it into any downstream consumer.

## Requirements

Install Python dependencies (PyTorch + boto libraries + OpenSlide bindings):

```bash
pip install -r requirements.txt
```

OpenSlide also needs native libraries. On Ubuntu:

```bash
sudo apt-get install -y openslide-tools
```

## Testing the Dataloader

After running the pipeline, import and instantiate the dataloader inside your training script:

```python
from pathology_foundation_model import create_dataloader

dataset_kwargs = {
    "wsi_dir": "/abs/path/to/cache/health/camelyon-dataset/CAMELYON16/images",
    "seg_dirs": result.seg_dirs,
    "patch_size": 256,
    "transform": my_transforms,
}

dataloader = create_dataloader(
    "clam",
    dataset_kwargs,
    batch_size=128,
    num_workers=8,
    distributed=True,
)
```

Use `len(dataloader.dataset)` to inspect the total patch count, or iterate normally in a training epoch.
