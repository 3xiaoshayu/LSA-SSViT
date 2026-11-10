#!/usr/bin/env python
"""Download the pretrained weights of the comparators before training (no GPU needed).

    python comparators/prepare_weights.py --weights-dir outputs/comparator_weights

timm and Hugging Face weights go to the Hugging Face cache (HF_HOME; set HF_ENDPOINT to use
a mirror). The image encoder of Mammo-CLIP is extracted from its released checkpoint into
--weights-dir, together with the record of the data it was pre-trained on and the input
statistics stored in its configuration. The record is printed and compared with the
normalization of the spec in comparators/zoo.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparators.zoo import (  # noqa: E402
    COMPARATORS,
    MAMMO_CLIP_VARIANTS,
    Comparator,
    extract_mammo_clip_encoder,
)

WEIGHT_PATTERNS = ["*.json", "*.safetensors"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the pretrained weights of the comparators.")
    parser.add_argument("--models", nargs="+", default=list(COMPARATORS), choices=list(COMPARATORS))
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "outputs" / "comparator_weights")
    return parser.parse_args()


def snapshot(repository: str, extra_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(repository, allow_patterns=WEIGHT_PATTERNS + (extra_patterns or [])))
    if not any(folder.rglob("*.safetensors")):  # older repositories ship PyTorch pickles only
        folder = Path(snapshot_download(repository, allow_patterns=["*.bin", "*.pth", "*.pth.tar"]))
    return folder


def _numbers(entries: dict | None) -> list[float]:
    """Numeric values of the configuration entries recorded at extraction (scalars or per-channel lists)."""
    values: list[float] = []
    for value in (entries or {}).values():
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                values.append(float(item))
    return values


def report_mammo_clip(spec: Comparator, record: dict) -> bool:
    """Print the extraction record; True if the normalization of the spec differs from the checkpoint's."""
    for field in ("image_encoder", "data_train", "mean", "std", "image_size"):
        print(f"  {spec.key} {field}: {record.get(field, 'not recorded (extracted by an earlier version)')}")
    if spec.normalization is None:
        print(f"  {spec.key}: normalization is not set in comparators/zoo.py; enter the mean and std above")
        return False
    stored = {"mean": _numbers(record.get("mean")), "std": _numbers(record.get("std"))}
    if not stored["mean"] or not stored["std"]:
        print(f"  {spec.key}: no mean/std in the record; normalization {spec.normalization} is unchecked")
        return False
    differs = any(abs(value - expected) > 1e-6
                  for name, expected in zip(("mean", "std"), spec.normalization) for value in stored[name])
    print(f"  {spec.key}: normalization {spec.normalization} "
          + ("DIFFERS from the checkpoint configuration" if differs else "matches the checkpoint configuration"))
    return differs


def main() -> None:
    args = parse_args()
    mismatched = []
    for name in args.models:
        spec = COMPARATORS[name]
        if spec.loader == "timm":
            from timm.models import get_pretrained_cfg

            config = get_pretrained_cfg(spec.source)
            if config is None or not config.hf_hub_id:
                raise RuntimeError(f"{spec.source}: no Hugging Face weights in this timm version")
            folder = snapshot(config.hf_hub_id)
        elif spec.loader == "hf_dino":
            folder = snapshot(spec.source)
        elif spec.loader == "hf_mambavision":
            folder = snapshot(spec.source, ["*.py"])
        elif spec.loader == "mammo_clip":
            from huggingface_hub import hf_hub_download

            variant = MAMMO_CLIP_VARIANTS[name]
            output = args.weights_dir / variant.weights
            if output.is_file():
                record = torch.load(output, map_location="cpu", weights_only=True)
                record.pop("state_dict")
            else:
                tar_path = Path(hf_hub_download(variant.repository, variant.file))
                record = extract_mammo_clip_encoder(tar_path, output, name)
            if report_mammo_clip(spec, record):
                mismatched.append(name)
            folder = output.parent
        else:
            raise ValueError(f"Unknown loader {spec.loader!r}")
        size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) / 2**20
        print(f"{name:18s} {spec.source:48s} -> {folder} ({size:,.0f} MB)")
    if mismatched:
        raise SystemExit(f"Normalization in comparators/zoo.py differs from the checkpoint for {mismatched}")


if __name__ == "__main__":
    main()
