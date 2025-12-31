import argparse
import logging

from torchvision import transforms

from src import PipelineConfig, create_dataloader, setup_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Pathology foundation model data pipeline")
    parser.add_argument("--file-path", default="CAMELYON16/images/", help="S3 prefix to crawl")
    parser.add_argument("--download-num", type=int, default=1, help="Number of WSI files to download (-1 for all)")
    parser.add_argument("--dataset", default="clam", help="Registered dataset to instantiate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedSampler for DDP jobs")
    parser.add_argument("--rank", type=int, default=None, help="Override distributed rank")
    parser.add_argument("--world-size", type=int, default=None, help="Override world size")
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--bucket-name", default="camelyon-dataset")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory for downloads")
    parser.add_argument("--patch-script", default="CLAM/create_patches_fp.py")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("pathology.foundation")

    pipeline_config = PipelineConfig(
        bucket_name=args.bucket_name,
        cache_root=args.cache_dir,
        patch_script=args.patch_script,
        patch_size=args.patch_size,
    )
    download_limit = None if args.download_num is not None and args.download_num < 0 else args.download_num
    pipeline_result = setup_pipeline(args.file_path, download_num=download_limit, config=pipeline_config)
    if not pipeline_result.has_data:
        logger.error("No segmented patches were found. Check the pipeline logs for failures.")
        return

    dataset_kwargs = {
        "wsi_dir": pipeline_result.wsi_dir,
        "seg_dirs": pipeline_result.seg_dirs,
        "patch_size": args.patch_size,
        "transform": build_transforms(),
    }

    dataloader = create_dataloader(
        args.dataset,
        dataset_kwargs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=not args.no_shuffle,
        distributed=args.distributed,
        drop_last=args.drop_last,
        sampler_seed=args.sampler_seed,
        rank=args.rank,
        world_size=args.world_size,
    )

    logger.info("Loaded %d patches across %d segmentation folders", len(dataloader.dataset), len(pipeline_result.seg_dirs))

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        logger.warning("Dataloader is empty after initialization")
        return

    images, coords, slide_ids = batch
    logger.info("Sample batch shape: %s | coords tensor shape: %s", tuple(images.shape), tuple(coords.shape))
    logger.info("Slides in first batch: %s", ", ".join(sorted(set(slide_ids))))


if __name__ == "__main__":
    main()
