#!/usr/bin/env python
"""Build the preprocessing cache for every image listed in the split files (or in --csv).

    python scripts/build_cache.py --image-root VINDR_DICOM_ROOT --cache-root outputs/cache/vindr_1024
    python scripts/build_cache.py --image-root dataset/inbreast/AllDICOMs --csv outputs/inbreast/manifest.csv \
        --cache-root outputs/cache/inbreast_1024

Each image goes through the fixed, annotation-free pipeline of core.prepare_image once
(VOI LUT windowing, removal of padding bands and disconnected fragments, cropping to the
breast, resizing to 1024 px) and is stored as a compressed uint16 array in
<cache-root>/images/<xx>/<sha256>.npz. Interrupted builds resume where they stopped.
Finally, a random sample of images is preprocessed again and compared pixel by pixel
with the cache; training refuses a cache whose manifest is not marked "ready".
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (  # noqa: E402
    UINT16_SCALE,
    cache_image_path,
    load_cached_image,
    prepare_image,
    preprocessing_config,
    save_cached_image,
    write_json,
)

SPLITS = ("train", "val", "eval", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the preprocessing cache.")
    parser.add_argument("--image-root", type=Path, required=True, help="root folder of the DICOM files")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "splits" / "vindr")
    parser.add_argument("--csv", type=Path, nargs="+", default=None,
                        help="image lists to cache instead of the split files (e.g. the INbreast manifest)")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "outputs" / "cache" / "vindr_1024")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--verify-count", type=int, default=64, help="images re-processed for verification (0 = none)")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_rows(split_root: Path, csv_paths: list[Path] | None = None) -> pd.DataFrame:
    paths = csv_paths or [split_root / f"{name}.csv" for name in SPLITS if (split_root / f"{name}.csv").is_file()]
    frames = [pd.read_csv(path, dtype={"study_id": str, "series_id": str, "image_id": str}) for path in paths]
    if not frames:
        raise FileNotFoundError(f"No split files found in {split_root}")
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["study_id", "series_id", "image_id"]).reset_index(drop=True)


def _limit_threads() -> None:
    import torch

    torch.set_num_threads(1)


def build_one(payload: dict[str, Any]) -> str:
    _limit_threads()
    row = pd.Series(payload["row"])
    cache_root = Path(payload["cache_root"])
    if cache_image_path(cache_root, row).is_file():
        return "skipped"
    image = prepare_image(row, Path(payload["image_root"]))
    save_cached_image(cache_root, row, image)
    return "generated"


def verify_one(payload: dict[str, Any]) -> float:
    """Maximum absolute difference between a fresh preprocessing and the cached image."""
    _limit_threads()
    row = pd.Series(payload["row"])
    online = prepare_image(row, Path(payload["image_root"]))
    cached = load_cached_image(Path(payload["cache_root"]), row)
    return float(np.max(np.abs(online - cached)))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.split_root, args.csv)
    cache_root = args.cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    config = preprocessing_config()
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("config") or {}
        if any(previous.get(key) != value for key, value in config.items()) or previous.get("box_protection"):
            raise ValueError(f"{cache_root} holds images from a different preprocessing; use another --cache-root.")
        config = {**previous, **config}  # keep any additional entries of an existing manifest
    manifest: dict[str, Any] = {"status": "building", "config": config,
                                "updated_at": datetime.now().astimezone().isoformat()}
    write_json(manifest_path, manifest)

    print(f"Caching {len(rows)} images from {args.image_root} into {cache_root.resolve()}")
    payloads = [{"row": row, "image_root": str(args.image_root), "cache_root": str(cache_root)}
                for row in rows.to_dict("records")]
    started = time.perf_counter()
    counts = {"generated": 0, "skipped": 0}
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(build_one, payload): payload for payload in payloads}
        with tqdm(total=len(payloads), unit="image") as bar:
            for future in as_completed(futures):
                try:
                    counts[future.result()] += 1
                except Exception as error:
                    failures.append({"image_id": str(futures[future]["row"]["image_id"]),
                                     "error": f"{type(error).__name__}: {error}"})
                bar.update(1)
    if failures:
        write_json(cache_root / "failures.json", {"failures": failures})
        write_json(manifest_path, {**manifest, "status": "failed"})
        raise RuntimeError(f"{len(failures)} images failed (see failures.json); first: {failures[0]}")

    tolerance = 1.5 * 0.5 / UINT16_SCALE  # half a quantisation step, with margin
    selected = list(range(len(payloads)))
    random.Random(args.seed).shuffle(selected)
    selected = selected[: min(args.verify_count, len(payloads))]
    worst = 0.0
    if selected:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for difference in tqdm(executor.map(verify_one, [payloads[i] for i in selected]),
                                   total=len(selected), desc="verify", unit="image"):
                worst = max(worst, difference)
        if worst > tolerance:
            write_json(manifest_path, {**manifest, "status": "verification_failed"})
            raise RuntimeError(f"Cached images differ from a fresh preprocessing (max difference {worst:.2e}).")

    write_json(manifest_path, {
        "status": "ready",
        "config": config,
        "updated_at": datetime.now().astimezone().isoformat(),
        "images": len(payloads),
        "last_build": {**counts, "minutes": (time.perf_counter() - started) / 60,
                       "verified": len(selected), "max_verification_difference": worst},
    })
    print(f"Done: {counts['generated']} generated, {counts['skipped']} already cached, "
          f"{len(selected)} verified (max difference {worst:.2e}).")


if __name__ == "__main__":
    main()
