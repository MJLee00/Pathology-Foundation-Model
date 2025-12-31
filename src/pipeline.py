import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    bucket_name: str = "camelyon-dataset"
    cache_root: Optional[str] = None
    patch_script: str = "CLAM/create_patches_fp.py"
    patch_size: int = 256
    seg_subdir: str = "seg"
    patches_subdir: str = "patches"
    stitch: bool = True
    extra_patch_args: Sequence[str] = field(default_factory=tuple)
    download_client_kwargs: Dict[str, object] = field(default_factory=dict)

    def resolved_cache_root(self) -> str:
        base = self.cache_root or os.environ.get("HF_DATASETS_CACHE", "./cache")
        return os.path.abspath(base)


@dataclass
class PipelineResult:
    wsi_dir: str
    seg_dirs: List[str]
    downloaded_slides: List[str]

    def __iter__(self):
        yield self.wsi_dir
        yield self.seg_dirs

    @property
    def has_data(self) -> bool:
        return bool(self.seg_dirs)


def _list_slides(client, bucket: str, prefix: str) -> List[str]:
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])
    slide_keys = [obj["Key"] for obj in contents if obj["Key"].endswith(".tif")]
    return sorted(slide_keys)


def setup_pipeline(
    file_path: str,
    download_num: Optional[int] = 1,
    *,
    config: Optional[PipelineConfig] = None,
    boto3_client=None,
) -> PipelineResult:
    cfg = config or PipelineConfig()
    wsi_dir = os.path.abspath(os.path.join(cfg.resolved_cache_root(), "health", cfg.bucket_name, file_path))
    os.makedirs(wsi_dir, exist_ok=True)

    client = boto3_client or boto3.client("s3", config=Config(signature_version=UNSIGNED), **cfg.download_client_kwargs)
    slide_keys = _list_slides(client, cfg.bucket_name, file_path)
    if download_num is not None:
        slide_keys = slide_keys[: download_num]

    seg_dirs: List[str] = []
    downloaded: List[str] = []

    for key in slide_keys:
        file_name = os.path.basename(key)
        slide_id, _ = os.path.splitext(file_name)
        target_path = os.path.join(wsi_dir, file_name)
        seg_dir = os.path.join(wsi_dir, cfg.seg_subdir, slide_id)
        patch_dir = os.path.join(seg_dir, cfg.patches_subdir)
        h5_path = os.path.join(patch_dir, f"{slide_id}.h5")

        if not os.path.exists(target_path):
            client.download_file(cfg.bucket_name, key, target_path)
            downloaded.append(file_name)

        if not os.path.exists(h5_path):
            os.makedirs(seg_dir, exist_ok=True)
            success = _run_clam(
                cfg.patch_script,
                wsi_dir,
                seg_dir,
                cfg.patch_size,
                cfg.stitch,
                cfg.extra_patch_args,
            )
            if not success:
                continue

        if os.path.exists(h5_path):
            seg_dirs.append(seg_dir)

    return PipelineResult(wsi_dir=wsi_dir, seg_dirs=seg_dirs, downloaded_slides=downloaded)


def _run_clam(
    script_path: str,
    source_dir: str,
    save_dir: str,
    patch_size: int,
    stitch: bool,
    extra_args: Iterable[str],
) -> None:
    abs_script = os.path.abspath(script_path)
    script_dir = os.path.dirname(abs_script)
    script_name = os.path.basename(abs_script)
    cmd = [
        "python",
        script_name if script_dir else abs_script,
        "--source",
        source_dir,
        "--save_dir",
        save_dir,
        "--patch_size",
        str(patch_size),
        "--seg",
        "--patch",
    ]
    if stitch:
        cmd.append("--stitch")
    cmd.extend(extra_args)

    try:
        subprocess.run(cmd, check=True, cwd=script_dir or None)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("CLAM patch extraction failed for %s: %s", save_dir, exc)
        return False
