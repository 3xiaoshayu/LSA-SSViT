"""Data pipeline, model and training objective of LSA-SSViT.

Contents
--------
- Reproducibility helpers.
- Fixed, annotation-free preprocessing: DICOM decoding with the VOI LUT, removal of
  constant padding bands, breast segmentation, removal of disconnected fragments,
  cropping to the breast, resizing the long side to 1024 px and zero padding.
- A uint16 preprocessing cache and the training/evaluation dataset.
- The model: a DINOv3 ViT-B/16 backbone whose last four blocks are fine-tuned,
  optional GRAIN operators after each of these blocks, and a linear three-head
  classifier (mass, suspicious calcification, screening = BI-RADS 3-5).
- The CUSP training objective.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from metrics import CLASS_NAMES, parse_birads_ge3, parse_birads_level

# Fixed preprocessing (identical for training, evaluation and external testing).
IMAGE_SIZE = 1024
CLIP_LOW = 0.5
CLIP_HIGH = 99.95
BREAST_CROP = True
UINT16_SCALE = 65535.0
PREPROCESSING_VERSION = 2
VOI_INPUT_VERSION = 2
DICOM_EXTENSIONS = ["", ".dicom", ".dcm"]

# Model.
DEFAULT_BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"
UNFREEZE_LAST_BLOCKS = 4
CHECKPOINT_FORMAT = 1


# -----------------------------------------------------------------------------
# Reproducibility and file helpers
# -----------------------------------------------------------------------------


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, and request deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:  # older PyTorch without warn_only
        torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # One CPU thread per data-loading worker avoids oversubscription.
    torch.set_num_threads(1)


def capture_rng_state(train_generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if train_generator is not None:
        state["train_generator"] = train_generator.get_state()
    return state


def _cpu_rng_tensor(value: Any) -> Any:
    """torch.load(map_location="cuda") also moves RNG byte tensors; they must be on the CPU."""
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    return value


def restore_rng_state(state: dict[str, Any] | None, train_generator: torch.Generator | None = None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_rng_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([_cpu_rng_tensor(item) for item in state["torch_cuda"]])
    if train_generator is not None and state.get("train_generator") is not None:
        train_generator.set_state(_cpu_rng_tensor(state["train_generator"]))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# DICOM decoding and annotation-free preprocessing
# -----------------------------------------------------------------------------


def resolve_image_path(root: Path, row: pd.Series) -> Path:
    """Locate a DICOM file: explicit image_path column, then <study>/<image>, <series>/<image>, <image>."""
    if "image_path" in row.index and pd.notna(row.get("image_path")):
        explicit = Path(str(row["image_path"]))
        candidate = explicit if explicit.is_absolute() else root / explicit
        if candidate.is_file():
            return candidate

    study_id = str(row["study_id"])
    series_id = str(row.get("series_id", ""))
    image_id = str(row["image_id"])
    bases = [root / study_id / image_id, root / series_id / image_id, root / image_id]
    for base in bases:
        for extension in DICOM_EXTENSIONS:
            candidate = Path(str(base) + extension)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Image not found: image_id={image_id}. Searched the study, series and flat layouts under {root}"
    )


def _valid_normalization_values(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return finite
    min_value = float(finite.min())
    non_min = finite[finite > min_value]
    if non_min.size >= max(100, int(finite.size * 0.05)):
        return non_min
    return finite


def normalize_to_float(
    array: np.ndarray,
    mode: str = "percentile",
    clip_low: float = 0.5,
    clip_high: float = 99.95,
) -> np.ndarray:
    """Map grey values to [0, 1] (float32) by min-max or percentile scaling."""
    array = np.asarray(array, dtype=np.float32)
    values = _valid_normalization_values(array)
    if values.size == 0:
        return np.zeros(array.shape, dtype=np.float32)

    if mode == "minmax":
        low, high = float(values.min()), float(values.max())
    elif mode == "percentile":
        if not (0.0 <= clip_low < clip_high <= 100.0):
            raise ValueError("clip_low and clip_high must satisfy 0 <= low < high <= 100")
        low, high = [float(v) for v in np.percentile(values, [clip_low, clip_high])]
    else:
        raise ValueError("mode must be 'percentile' or 'minmax'")

    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    output = (array - low) / (high - low)
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def _connected_components(binary: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Connected components (8-connectivity) with OpenCV, or a pure-Python fallback."""
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        return int(count), labels, stats
    except Exception:
        height, width = binary.shape
        labels = np.zeros((height, width), dtype=np.int32)
        stats_rows = [[0, 0, width, height, int((~binary).sum())]]
        label = 0
        for start_y, start_x in zip(*np.where(binary & (labels == 0))):
            if labels[start_y, start_x] != 0:
                continue
            label += 1
            stack = [(int(start_y), int(start_x))]
            labels[start_y, start_x] = label
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                y, x = stack.pop()
                xs.append(x)
                ys.append(y)
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            stack.append((ny, nx))
            stats_rows.append([min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, len(xs)])
        return label + 1, labels, np.asarray(stats_rows, dtype=np.int32)


def _edge_constant_padding_mask(
    raw: np.ndarray,
    declared_mask: np.ndarray,
    min_component_ratio: float = 0.002,
) -> np.ndarray:
    """Find constant bands along the image border that the DICOM header does not declare as padding.

    Only whole rows or columns of one constant value that start at an image edge and
    are large enough are removed (for example the white bands left after inverting
    MONOCHROME1 images); breast tissue is never scanned.
    """
    height, width = raw.shape
    minimum_pixels = max(256, int(round(raw.size * min_component_ratio)))
    inferred = np.zeros(raw.shape, dtype=bool)

    column_min = raw.min(axis=0)
    column_max = raw.max(axis=0)
    column_uniform = column_min == column_max
    row_min = raw.min(axis=1)
    row_max = raw.max(axis=1)
    row_uniform = row_min == row_max

    def consume_columns(side: str) -> None:
        indices = range(width) if side == "left" else range(width - 1, -1, -1)
        ordered = list(indices)
        cursor = 0
        while cursor < len(ordered) and column_uniform[ordered[cursor]]:
            value = raw[0, ordered[cursor]]
            run: list[int] = []
            while (
                cursor < len(ordered)
                and column_uniform[ordered[cursor]]
                and raw[0, ordered[cursor]] == value
            ):
                run.append(ordered[cursor])
                cursor += 1
            if len(run) * height >= minimum_pixels:
                x1, x2 = min(run), max(run) + 1
                new_pixels = ~declared_mask[:, x1:x2] & ~inferred[:, x1:x2]
                if new_pixels.any():
                    inferred[:, x1:x2] |= new_pixels
            # Stop as soon as the next column is not constant.
            if cursor < len(ordered) and not column_uniform[ordered[cursor]]:
                break

    def consume_rows(side: str) -> None:
        indices = range(height) if side == "top" else range(height - 1, -1, -1)
        ordered = list(indices)
        cursor = 0
        while cursor < len(ordered) and row_uniform[ordered[cursor]]:
            value = raw[ordered[cursor], 0]
            run: list[int] = []
            while (
                cursor < len(ordered)
                and row_uniform[ordered[cursor]]
                and raw[ordered[cursor], 0] == value
            ):
                run.append(ordered[cursor])
                cursor += 1
            if len(run) * width >= minimum_pixels:
                y1, y2 = min(run), max(run) + 1
                new_pixels = ~declared_mask[y1:y2, :] & ~inferred[y1:y2, :]
                if new_pixels.any():
                    inferred[y1:y2, :] |= new_pixels
            if cursor < len(ordered) and not row_uniform[ordered[cursor]]:
                break

    consume_columns("left")
    consume_columns("right")
    consume_rows("top")
    consume_rows("bottom")
    return inferred


def _apply_voi_pipeline(
    raw: np.ndarray,
    dataset: Any,
    invalid_mask: np.ndarray,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    """Rescale, apply the VOI LUT / window, fix the photometric direction and scale to [0, 1].

    If the VOI transform cannot be applied, the image falls back to percentile scaling.
    Padding pixels are set to 0.
    """
    array = raw.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    array = array * slope + intercept
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()

    if photometric == "MONOCHROME1":
        valid = array[~invalid_mask]
        pivot = float(valid.max() + valid.min()) if valid.size else float(array.max() + array.min())
        array = pivot - array

    if invalid_mask.any():
        array = array.copy()
        array[invalid_mask] = np.nan

    try:
        try:
            from pydicom.pixels import apply_voi_lut
        except ImportError:
            from pydicom.pixel_data_handlers.util import apply_voi_lut

        # pydicom expects the modality (rescale) transform before the VOI transform, on the
        # un-inverted data. Integer data are kept integer when a VOI LUT sequence is present
        # (pydicom indexes the LUT with them); non-integer rescaling stays in floating point.
        has_voi_lut = bool(dataset.get("VOILUTSequence"))
        slope_is_integer = math.isfinite(slope) and float(slope).is_integer()
        intercept_is_integer = math.isfinite(intercept) and float(intercept).is_integer()
        if has_voi_lut and slope_is_integer and intercept_is_integer:
            slope_int = int(round(slope))
            intercept_int = int(round(intercept))
            if slope_int == 1 and intercept_int == 0:
                modality_for_voi = raw
            else:
                modality_for_voi = raw.astype(np.int64) * slope_int + intercept_int
        else:
            modality_for_voi = raw.astype(np.float64) * slope + intercept
        voi = np.asarray(apply_voi_lut(modality_for_voi, dataset), dtype=np.float32)
        if photometric == "MONOCHROME1":
            valid_voi = voi[~invalid_mask]
            pivot = float(valid_voi.max() + valid_voi.min()) if valid_voi.size else float(voi.max() + voi.min())
            voi = pivot - voi
        if invalid_mask.any():
            voi = voi.copy()
            voi[invalid_mask] = np.nan
        image = normalize_to_float(voi, "minmax", clip_low, clip_high)
    except Exception:
        image = normalize_to_float(array, "percentile", clip_low, clip_high)

    if invalid_mask.any():
        image = image.copy()
        image[invalid_mask] = 0.0
    return image


def read_dicom_image(path: Path, clip_low: float = CLIP_LOW, clip_high: float = CLIP_HIGH) -> np.ndarray:
    """Decode a mammogram and return a [0, 1] float32 image (padding and border bands set to 0)."""
    try:
        import pydicom

        dataset = pydicom.dcmread(str(path))
        raw = np.squeeze(np.asarray(dataset.pixel_array))
    except Exception as error:
        raise RuntimeError(
            f"Could not decode DICOM {path}. For compressed DICOM install the pylibjpeg decoders "
            f"(see requirements.txt). Original error: {error}"
        ) from error
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2-D mammogram, got pixel data of shape {raw.shape}")

    declared_padding = np.zeros(raw.shape, dtype=bool)
    if hasattr(dataset, "PixelPaddingValue"):
        padding_value = float(dataset.PixelPaddingValue)
        range_limit = getattr(dataset, "PixelPaddingRangeLimit", None)
        if range_limit is not None:
            try:
                limit = float(range_limit)
                low, high = sorted((padding_value, limit))
                declared_padding = (raw >= low) & (raw <= high)
            except (TypeError, ValueError):
                declared_padding = raw == padding_value
        else:
            declared_padding = raw == padding_value

    inferred_padding = _edge_constant_padding_mask(raw, declared_padding)
    return _apply_voi_pipeline(raw, dataset, declared_padding | inferred_padding, clip_low, clip_high)


def _largest_component_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    count, _, stats = _connected_components(binary.astype(bool))
    if count <= 1:
        return None
    component = 1 + int(np.argmax(stats[1:, 4]))
    x, y, width, height = [int(v) for v in stats[component, :4]]
    return x, y, x + width, y + height


def estimate_breast_mask(image: np.ndarray, max_side: int = 768) -> np.ndarray:
    """Coarse breast mask: threshold, close, keep the largest component and dilate slightly.

    Computed at reduced resolution and mapped back to the full image.
    """
    height, width = image.shape
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        small_h = max(1, int(round(height * scale)))
        small_w = max(1, int(round(width * scale)))
        small_image = F.interpolate(
            torch.from_numpy(image).float()[None, None],
            size=(small_h, small_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, 0].numpy()
    else:
        small_image = image
        small_h, small_w = height, width

    positive = small_image[small_image > 0]
    if positive.size < 100:
        return np.zeros((height, width), dtype=np.uint8)
    threshold = max(0.004, float(np.percentile(positive, 2.0)) * 0.20)
    binary = small_image > threshold

    try:
        import cv2

        kernel_size = max(3, int(round(min(small_h, small_w) * 0.008)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 15)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        cleaned = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if count <= 1:
            small_mask = binary.astype(np.uint8)
        else:
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            small_mask = (labels == component).astype(np.uint8)
        dilation = max(1, int(round(min(small_h, small_w) * 0.004)))
        dilation = min(dilation, 7)
        small_mask = cv2.dilate(
            small_mask, np.ones((dilation, dilation), dtype=np.uint8), iterations=1
        )
    except Exception:
        bbox = _largest_component_bbox(binary)
        if bbox is None:
            small_mask = binary.astype(np.uint8)
        else:
            x1, y1, x2, y2 = bbox
            small_mask = np.zeros_like(binary, dtype=np.uint8)
            small_mask[y1:y2, x1:x2] = binary[y1:y2, x1:x2]

    if (small_h, small_w) == (height, width):
        return (small_mask > 0).astype(np.uint8)
    full = F.interpolate(
        torch.from_numpy(small_mask.astype(np.float32))[None, None],
        size=(height, width),
        mode="nearest",
    )[0, 0].numpy()
    return (full > 0.5).astype(np.uint8)


def clean_non_primary_breast_regions(
    image: np.ndarray,
    breast_mask: np.ndarray,
    max_side: int = 768,
    safety_margin_ratio: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the main breast and remove disconnected tissue fragments (no annotation is used).

    An Otsu-based tissue seed separates tissue from near-black background residue, the
    largest seed component is kept as the breast, low-intensity skin near it is restored,
    a safety margin is added, and interior holes are filled.
    """
    height, width = image.shape
    if image.size == 0 or breast_mask.size == 0:
        return image, breast_mask

    scale = min(1.0, max_side / max(height, width))
    small_h = max(1, int(round(height * scale)))
    small_w = max(1, int(round(width * scale)))

    image_small = F.interpolate(
        torch.from_numpy(image.astype(np.float32))[None, None],
        size=(small_h, small_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0, 0].numpy()
    mask_small = F.interpolate(
        torch.from_numpy((breast_mask > 0).astype(np.float32))[None, None],
        size=(small_h, small_w),
        mode="nearest",
    )[0, 0].numpy().astype(np.uint8)

    positive = image_small[np.isfinite(image_small) & (image_small > 0)]
    if positive.size < 100:
        return image, breast_mask

    # Otsu separates near-black residue from tissue; x0.65 keeps low-density breast edges.
    try:
        import cv2

        image_u8 = (np.clip(image_small, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        otsu_value, _ = cv2.threshold(
            image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        seed_threshold = max(0.02, float(otsu_value) / 255.0 * 0.65)
    except Exception:
        seed_threshold = max(0.02, float(np.percentile(positive, 12.0)))

    robust_seed = (image_small > seed_threshold).astype(np.uint8)

    try:
        import cv2

        close_size = max(3, int(round(min(small_h, small_w) * 0.006)))
        if close_size % 2 == 0:
            close_size += 1
        close_size = min(close_size, 9)
        open_size = max(3, int(round(min(small_h, small_w) * 0.004)))
        if open_size % 2 == 0:
            open_size += 1
        open_size = min(open_size, 7)

        robust_seed = cv2.morphologyEx(
            robust_seed,
            cv2.MORPH_CLOSE,
            np.ones((close_size, close_size), dtype=np.uint8),
        )
        robust_seed = cv2.morphologyEx(
            robust_seed,
            cv2.MORPH_OPEN,
            np.ones((open_size, open_size), dtype=np.uint8),
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            robust_seed, connectivity=8
        )
        minimum_area = max(32, int(round(small_h * small_w * 0.0002)))
        valid_components = [
            index
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
        ]

        if not valid_components:
            retained_seed = mask_small.copy()
        else:
            main_component = max(
                valid_components, key=lambda index: int(stats[index, cv2.CC_STAT_AREA])
            )
            retained_seed = (labels == main_component).astype(np.uint8)

        # Restore low-intensity skin only close to the main breast.
        recovery_margin = max(4, int(round(min(small_h, small_w) * 0.020)))
        recovery_margin = min(recovery_margin, 20)
        nearby = cv2.dilate(
            retained_seed,
            np.ones((2 * recovery_margin + 1, 2 * recovery_margin + 1), dtype=np.uint8),
            iterations=1,
        )
        retained_small = retained_seed | ((mask_small > 0) & (nearby > 0))

        # Safety margin around the breast.
        retained_small = retained_small.astype(np.uint8)
        safety_margin = max(3, int(round(min(small_h, small_w) * safety_margin_ratio)))
        safety_margin = min(safety_margin, 16)
        retained_small = cv2.dilate(
            retained_small,
            np.ones((2 * safety_margin + 1, 2 * safety_margin + 1), dtype=np.uint8),
            iterations=1,
        )

        # Fill holes inside the retained region.
        found = cv2.findContours(retained_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        filled = np.zeros_like(retained_small)
        if contours:
            cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
            retained_small = filled
    except Exception:
        count, labels, stats = _connected_components(robust_seed > 0)
        if count <= 1:
            retained_small = mask_small.copy()
        else:
            valid_components = [
                index
                for index in range(1, count)
                if int(stats[index, 4]) >= max(32, int(round(small_h * small_w * 0.0002)))
            ]
            if not valid_components:
                retained_small = mask_small.copy()
            else:
                main_component = max(valid_components, key=lambda index: int(stats[index, 4]))
                retained_small = (labels == main_component).astype(np.uint8)
            margin = max(4, int(round(min(small_h, small_w) * 0.02)))
            tensor = torch.from_numpy(retained_small.astype(np.float32))[None, None]
            retained_small = (
                F.max_pool2d(tensor, kernel_size=2 * margin + 1, stride=1, padding=margin)[0, 0]
                .numpy()
                > 0
            ).astype(np.uint8)

    retained_full = F.interpolate(
        torch.from_numpy(retained_small.astype(np.float32))[None, None],
        size=(height, width),
        mode="nearest",
    )[0, 0].numpy()
    retained_full = (retained_full > 0.5).astype(np.uint8)

    cleaned = image.copy()
    cleaned[retained_full == 0] = 0.0

    refined_breast_mask = ((retained_full > 0) & (cleaned > 0)).astype(np.uint8)
    if refined_breast_mask.sum() == 0:
        refined_breast_mask = retained_full.astype(np.uint8)
    return cleaned, refined_breast_mask


def crop_breast_region(
    image: np.ndarray,
    breast_mask: np.ndarray,
    enabled: bool = True,
    margin_ratio: float = 0.02,
) -> np.ndarray:
    """Crop to the bounding box of the breast, enlarged by 2% of the image size on each side."""
    height, width = image.shape
    if not enabled:
        return image

    bbox = _largest_component_bbox(breast_mask > 0)
    if bbox is None:
        return image
    x1, y1, x2, y2 = bbox
    area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(height * width, 1)
    if area_ratio < 0.08:
        return image

    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return image
    return image[y1:y2, x1:x2]


def resize_and_pad(image: np.ndarray, output_size: int) -> np.ndarray:
    """Resize the long side to output_size (bilinear, antialiased) and zero-pad to a square."""
    height, width = image.shape
    scale = output_size / max(height, width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    top = (output_size - new_h) // 2
    left = (output_size - new_w) // 2

    image_tensor = torch.from_numpy(image).float()[None, None]
    image_resized = F.interpolate(
        image_tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0, 0]
    image_canvas = torch.zeros((output_size, output_size), dtype=torch.float32)
    image_canvas[top : top + new_h, left : left + new_w] = image_resized
    return image_canvas.numpy()


def prepare_image(
    row: pd.Series,
    image_root: Path,
    image_size: int = IMAGE_SIZE,
    clip_low: float = CLIP_LOW,
    clip_high: float = CLIP_HIGH,
    breast_crop: bool = BREAST_CROP,
) -> np.ndarray:
    """Full fixed preprocessing of one image; returns an image_size x image_size float32 array in [0, 1]."""
    path = resolve_image_path(image_root, row)
    image = read_dicom_image(path, clip_low, clip_high)
    breast_mask = estimate_breast_mask(image)
    image, breast_mask = clean_non_primary_breast_regions(image, breast_mask)
    image = crop_breast_region(image, breast_mask, enabled=breast_crop)
    return resize_and_pad(image, image_size)


# -----------------------------------------------------------------------------
# Preprocessing cache
# -----------------------------------------------------------------------------


def preprocessing_config(
    image_size: int = IMAGE_SIZE,
    clip_low: float = CLIP_LOW,
    clip_high: float = CLIP_HIGH,
    breast_crop: bool = BREAST_CROP,
) -> dict[str, Any]:
    """Everything that determines the cached images; stored in the cache manifest."""
    return {
        "preprocessing_cache_version": PREPROCESSING_VERSION,
        "voi_input_version": VOI_INPUT_VERSION,
        "image_size": int(image_size),
        "normalization": "voi",
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "breast_crop": bool(breast_crop),
        "image_dtype": "uint16",
    }


def cache_image_path(cache_root: Path, row: pd.Series | dict[str, Any]) -> Path:
    values = [str(row.get("study_id", "")), str(row.get("series_id", "")), str(row.get("image_id", ""))]
    key = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()
    return Path(cache_root) / "images" / key[:2] / f"{key}.npz"


def validate_cache_manifest(cache_root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Refuse a cache that is incomplete or was built with a different preprocessing."""
    path = Path(cache_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cache manifest not found: {path}. Run scripts/build_cache.py first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        raise ValueError(f"Cache at {cache_root} is not ready (status={manifest.get('status')}).")
    actual = manifest.get("config") or {}
    mismatched = {key: (value, actual.get(key)) for key, value in expected.items() if actual.get(key) != value}
    if mismatched:
        raise ValueError(f"Cache at {cache_root} was built with a different preprocessing: {mismatched}")
    if actual.get("box_protection"):
        raise ValueError(f"Cache at {cache_root} used annotations during preprocessing and cannot be used.")
    return manifest


def quantize_image_uint16(image: np.ndarray) -> np.ndarray:
    """[0, 1] float image -> uint16 (step 1/65535, i.e. at most 7.6e-6 rounding error per pixel)."""
    clipped = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.round(clipped * UINT16_SCALE).astype(np.uint16)


def dequantize_image_uint16(image_u16: np.ndarray) -> np.ndarray:
    return np.asarray(image_u16, dtype=np.float32) / np.float32(UINT16_SCALE)


def save_cached_image(cache_root: Path, row: pd.Series | dict[str, Any], image: np.ndarray) -> Path:
    path = cache_image_path(cache_root, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 2:
        raise ValueError(f"Cached images must be 2-D, got {image_array.shape}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(file, image_u16=quantize_image_uint16(image_array))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return path


def load_cached_image(cache_root: Path, row: pd.Series | dict[str, Any]) -> np.ndarray:
    path = cache_image_path(cache_root, row)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached image: {path} (image_id={row.get('image_id', '')})")
    with np.load(path, allow_pickle=False) as data:
        image = dequantize_image_uint16(data["image_u16"])
    return np.asarray(image, dtype=np.float32)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class MammographyDataset(Dataset):
    """One mammogram per item, with the three labels in CLASS_NAMES order.

    Images come from the preprocessing cache (cache_root) or are preprocessed on the
    fly from DICOM (image_root). Training augmentation: horizontal flip, random affine
    transform (scale 0.9-1.1, rotation +-10 deg, translation +-5%), contrast (0.9-1.1)
    and brightness (+-0.04) jitter. Images are returned as 1 x H x W tensors,
    standardised as (x - 0.5) / 0.25.
    """

    def __init__(
        self,
        csv_path: Path,
        train: bool,
        cache_root: Path | None = None,
        image_root: Path | None = None,
        image_size: int = IMAGE_SIZE,
        clip_low: float = CLIP_LOW,
        clip_high: float = CLIP_HIGH,
        breast_crop: bool = BREAST_CROP,
        class_names: tuple[str, ...] | list[str] = CLASS_NAMES,
    ) -> None:
        self.frame = pd.read_csv(csv_path, dtype={"study_id": str, "series_id": str, "image_id": str})
        self.train = bool(train)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.image_root = Path(image_root) if image_root is not None else None
        self.image_size = int(image_size)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.breast_crop = bool(breast_crop)
        self.class_names = list(class_names)

        required = {"study_id", "series_id", "image_id", "breast_birads", "mass", "calcification"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        self.frame["birads"] = self.frame["breast_birads"].map(parse_birads_ge3).astype(int)

        if self.cache_root is not None:
            expected = preprocessing_config(self.image_size, self.clip_low, self.clip_high, self.breast_crop)
            validate_cache_manifest(self.cache_root, expected)
            missing_items: list[str] = []
            for _, row in self.frame.iterrows():
                if not cache_image_path(self.cache_root, row).is_file():
                    missing_items.append(str(row["image_id"]))
                    if len(missing_items) >= 10:
                        break
            if missing_items:
                raise FileNotFoundError(f"Images missing from the cache (first ids): {missing_items}")
        elif self.image_root is None:
            raise ValueError("Pass cache_root or image_root.")

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _random_affine(image: torch.Tensor) -> torch.Tensor:
        """Random scaling, rotation and translation of a 1 x H x W image (bilinear, zero fill)."""
        scale = random.uniform(0.9, 1.1)
        angle = math.radians(random.uniform(-10.0, 10.0))
        tx = random.uniform(-0.05, 0.05) * 2.0
        ty = random.uniform(-0.05, 0.05) * 2.0

        # affine_grid maps output to input coordinates, so theta is the inverse transform.
        cos_a = math.cos(-angle) / scale
        sin_a = math.sin(-angle) / scale
        theta = torch.tensor(
            [[cos_a, -sin_a, -tx], [sin_a, cos_a, -ty]],
            dtype=torch.float32,
        ).unsqueeze(0)

        height, width = image.shape[-2:]
        grid = F.affine_grid(theta, size=(1, 1, height, width), align_corners=False)
        image_out = F.grid_sample(
            image.unsqueeze(0).float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[0]
        return image_out.clamp_(0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        if self.cache_root is not None:
            image = load_cached_image(self.cache_root, row)
        else:
            image = prepare_image(
                row, self.image_root, self.image_size, self.clip_low, self.clip_high, self.breast_crop
            )

        laterality = str(row.get("laterality", ""))
        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            side = laterality.strip().upper()
            if side in {"L", "LEFT"}:
                laterality = "R"
            elif side in {"R", "RIGHT"}:
                laterality = "L"

        tensor = torch.from_numpy(image).unsqueeze(0)
        if self.train:
            tensor = self._random_affine(tensor)
            contrast = random.uniform(0.9, 1.1)
            brightness = random.uniform(-0.04, 0.04)
            tensor = torch.clamp(tensor * contrast + brightness, 0.0, 1.0)

        tensor = (tensor - 0.5) / 0.25
        labels = torch.tensor([float(row[column]) for column in self.class_names], dtype=torch.float32)
        return {
            "image": tensor,
            "labels": labels,
            "image_id": str(row["image_id"]),
            "study_id": str(row["study_id"]),
            "laterality": laterality,
            "birads_level": parse_birads_level(row.get("breast_birads", "")),
        }


# -----------------------------------------------------------------------------
# Backbone
# -----------------------------------------------------------------------------


def load_backbone(source: str = DEFAULT_BACKBONE, pretrained: bool = True,
                  config: dict[str, Any] | None = None) -> nn.Module:
    """DINOv3 from Hugging Face transformers (model id or local directory).

    With pretrained=False the architecture is built without downloading weights, from
    `config` (the backbone configuration stored in our checkpoints) when given.
    """
    from transformers import AutoConfig, AutoModel

    if pretrained:
        return AutoModel.from_pretrained(source)
    if config is not None:
        from transformers import CONFIG_MAPPING

        return AutoModel.from_config(CONFIG_MAPPING[config["model_type"]].from_dict(dict(config)))
    return AutoModel.from_config(AutoConfig.from_pretrained(source))


def backbone_layers(backbone: nn.Module) -> nn.ModuleList:
    """Transformer blocks of a Hugging Face DINOv3 model (`model.layer` or `layer`, depending on the version)."""
    for owner in (getattr(backbone, "model", None), backbone):
        layers = getattr(owner, "layer", None) if owner is not None else None
        if isinstance(layers, nn.ModuleList) and len(layers) >= 1:
            return layers
    raise RuntimeError("Could not find the transformer blocks of the DINOv3 backbone.")


def apply_unfreeze_last_blocks(features: nn.Module, last_n: int) -> int:
    """Freeze the backbone except its last `last_n` blocks and its final LayerNorm."""
    last_n = int(last_n)
    if last_n < 1:
        raise ValueError("unfreeze_last_blocks must be >= 1")
    backbone = features.backbone if hasattr(features, "backbone") else features
    blocks = backbone_layers(backbone)
    if last_n > len(blocks):
        raise ValueError(f"unfreeze_last_blocks={last_n} exceeds the number of blocks ({len(blocks)})")
    for parameter in features.parameters():
        parameter.requires_grad_(False)
    for block in blocks[-last_n:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for module in (backbone, getattr(backbone, "encoder", None)):
        if module is None:
            continue
        for name, child in module.named_children():
            if "norm" not in name.lower():
                continue
            for parameter in child.parameters():
                parameter.requires_grad_(True)
    return int(sum(parameter.numel() for parameter in features.parameters() if parameter.requires_grad))


def _as_float32(value: Any) -> Any:
    """Cast a tensor, or a tuple/list of tensors (e.g. rotary embeddings), to float32."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, (tuple, list)):
        converted = [_as_float32(item) for item in value]
        return type(value)(converted)
    return value


class DINOv3Features(nn.Module):
    """Wraps the Hugging Face DINOv3 model; forward() returns the mean of the final patch tokens."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone, "config", None)
        registers = int(getattr(config, "num_register_tokens", 0) or 0) if config is not None else 0
        self.num_prefix_tokens = 1 + max(registers, 0)  # class token + register tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(pixel_values=x).last_hidden_state  # after the final LayerNorm
        return hidden[:, self.num_prefix_tokens :, :].mean(dim=1)

    def stack(self) -> tuple[nn.Module, nn.Module | None, nn.ModuleList, nn.Module]:
        backbone = self.backbone
        embeddings = getattr(backbone, "embeddings", None)
        if embeddings is None:
            raise RuntimeError("The DINOv3 backbone has no `embeddings` module.")
        rope = getattr(backbone, "rope_embeddings", None)
        norm = getattr(backbone, "norm", None)
        if norm is None:
            raise RuntimeError("The DINOv3 backbone has no final `norm` module.")
        return embeddings, rope, backbone_layers(backbone), norm

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
        embeddings, rope, _, _ = self.stack()
        pixel = x.to(embeddings.patch_embeddings.weight.dtype)
        hidden = embeddings(pixel_values=pixel)
        position = rope(pixel) if rope is not None else None
        return hidden, position

    @staticmethod
    def apply_block(layer: nn.Module, hidden: torch.Tensor, position: Any) -> torch.Tensor:
        # hidden is passed positionally, as in the Hugging Face encoder loop.
        out = layer(hidden) if position is None else layer(hidden, position_embeddings=position)
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
            return out[0]
        hidden_out = getattr(out, "last_hidden_state", None)
        if torch.is_tensor(hidden_out):
            return hidden_out
        raise RuntimeError(f"Unexpected output type from a DINOv3 block: {type(out)}")


# -----------------------------------------------------------------------------
# GRAIN
# -----------------------------------------------------------------------------


class GRAIN(nn.Module):
    """GRAIN: Gated Residual Aggregation of Inter-token Neighborhoods.

    The patch tokens (B x N x C) of a transformer block are arranged on their spatial
    grid (C x H x W). Each token's 7 x 7 neighbourhood is aggregated with a depthwise
    convolution, channels are mixed with a 1 x 1 convolution, and the result is added
    back through a bounded gate:

        X' = X + alpha * tanh(g) * tanh(PW(DW(X))),   alpha = 0.1,  g initialised to 0.

    With g = 0 the operator is the identity, so training starts from the unaltered
    pretrained backbone. The correction is computed and added in float32.
    Parameters: 49 C + C^2 + 1 (627,457 for C = 768).
    """

    KERNEL = 7
    WEIGHT_INIT = 0.02
    RESIDUAL_SCALE = 0.1

    def __init__(self, dim: int, kernel: int | None = None) -> None:
        super().__init__()
        k = int(self.KERNEL if kernel is None else kernel)
        if k < 1 or k % 2 == 0:
            raise ValueError("GRAIN kernel size must be a positive odd number")
        self.dw = nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        # Small non-zero weights so that the gate receives a gradient from the first step.
        nn.init.normal_(self.dw.weight, mean=0.0, std=self.WEIGHT_INIT)
        nn.init.normal_(self.pw.weight, mean=0.0, std=self.WEIGHT_INIT)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, n_patches, dim = patches.shape
        height = int(math.sqrt(n_patches))
        if height * height != n_patches:
            raise ValueError(f"GRAIN expects a square patch grid, got {n_patches} tokens")
        spatial = patches.reshape(batch, height, height, dim).permute(0, 3, 1, 2)
        delta = torch.tanh(self.pw(self.dw(spatial.float())))
        delta = self.RESIDUAL_SCALE * torch.tanh(self.gate) * delta
        delta = delta.permute(0, 2, 3, 1).reshape(batch, n_patches, dim)
        return (patches.float() + delta).to(dtype=patches.dtype)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class BreastClassifier(nn.Module):
    """DINOv3 ViT-B/16 + optional GRAIN + linear three-head classifier.

    The input is the 1-channel tensor returned by MammographyDataset; it is repeated to
    three channels and standardised with the ImageNet statistics. The patch tokens of
    the last block (after the final LayerNorm) are averaged and passed through dropout
    and a linear layer that outputs one logit per head (CLASS_NAMES order).

    With GRAIN, one operator follows each of the last `unfreeze_last_blocks` blocks and
    these blocks, their GRAIN operators and the final LayerNorm are computed in float32
    even under mixed precision (for numerical stability).

    The backbone keeps the default training-mode behaviour of the Hugging Face DINOv3
    implementation, including its random rescaling of the RoPE patch coordinates when the
    backbone configuration enables it (pos_embed_rescale).
    """

    def __init__(
        self,
        grain: bool = False,
        backbone: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        backbone_config: dict[str, Any] | None = None,
        unfreeze_last_blocks: int = UNFREEZE_LAST_BLOCKS,
        dropout: float = 0.25,
        num_classes: int = len(CLASS_NAMES),
    ) -> None:
        super().__init__()
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        network = load_backbone(backbone, pretrained=pretrained, config=backbone_config)
        embed_dim = int(network.config.hidden_size)
        self.features = DINOv3Features(network)
        apply_unfreeze_last_blocks(self.features, unfreeze_last_blocks)
        # Construction order (backbone, GRAIN, classifier) fixes the random initialisation.
        self.grain: nn.ModuleList | None = None
        if grain:
            self.grain = nn.ModuleList([GRAIN(embed_dim) for _ in range(int(unfreeze_last_blocks))])
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def prepare_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected a B x 1 x H x W tensor, got {tuple(x.shape)}")
        raw = (x.float() * 0.25 + 0.5).clamp(0.0, 1.0)
        rgb = raw.repeat(1, 3, 1, 1)
        return (rgb - self.imagenet_mean) / self.imagenet_std

    def pooled_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Mean of the final patch tokens (B x C)."""
        if self.grain is None:
            return self.features(pixel_values)

        hidden, position = self.features.embed(pixel_values)
        _, _, layers, norm = self.features.stack()
        if position is None:
            raise RuntimeError("The DINOv3 backbone returned no rotary position embeddings.")
        cut = len(layers) - len(self.grain)
        prefix = int(self.features.num_prefix_tokens)
        device = pixel_values.device
        fp32 = (
            torch.autocast(device_type=device.type, enabled=False)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        for index, layer in enumerate(layers):
            if index == cut:
                hidden = hidden.float()
                position = _as_float32(position)
            if index >= cut:
                with fp32:
                    hidden = self.features.apply_block(layer, hidden, position)
                    refined = self.grain[index - cut](hidden[:, prefix:, :])
                    hidden = torch.cat([hidden[:, :prefix], refined], dim=1)
            else:
                hidden = self.features.apply_block(layer, hidden, position)
        with fp32:
            hidden = norm(hidden.float())
        return hidden[:, prefix:, :].mean(dim=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        pooled = self.pooled_features(self.prepare_backbone_input(x))
        logits = self.classifier(pooled)
        return (logits, pooled) if return_features else logits


def model_config(model: BreastClassifier, **extra: Any) -> dict[str, Any]:
    """Configuration stored in checkpoints; enough to rebuild the model without downloading weights."""
    backbone_config = json.loads(json.dumps(model.features.backbone.config.to_dict(), default=str))
    # Private entries such as _name_or_path (a local directory) are not needed to rebuild the model.
    backbone_config = {key: value for key, value in backbone_config.items() if not key.startswith("_")}
    return {
        "grain": model.grain is not None,
        "unfreeze_last_blocks": UNFREEZE_LAST_BLOCKS if model.grain is None else len(model.grain),
        "dropout": float(model.classifier[0].p),
        "class_names": list(CLASS_NAMES),
        "backbone_config": backbone_config,
        "preprocessing": preprocessing_config(),
        **extra,
    }


def align_backbone_keys(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Match backbone key names across transformers versions (`backbone.model.layer` vs `backbone.layer`)."""
    expected = set(model.state_dict())
    if set(state) == expected:
        return state
    for old, new in (("features.backbone.model.", "features.backbone."),
                     ("features.backbone.layer.", "features.backbone.model.layer.")):
        renamed = {key.replace(old, new, 1) if key.startswith(old) else key: value for key, value in state.items()}
        if set(renamed) == expected:
            return renamed
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    raise RuntimeError(
        f"Checkpoint keys do not match the model: {len(missing)} missing (e.g. {missing[:3]}), "
        f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})"
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[BreastClassifier, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint written by train.py (no download needed)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(f"{checkpoint_path} is not a checkpoint written by this code.")
    config = checkpoint["config"]
    if list(config["class_names"]) != list(CLASS_NAMES):
        raise ValueError(f"Unexpected heads in {checkpoint_path}: {config['class_names']}")
    model = BreastClassifier(
        grain=bool(config["grain"]),
        pretrained=False,
        backbone_config=config["backbone_config"],
        unfreeze_last_blocks=int(config["unfreeze_last_blocks"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(align_backbone_keys(model, checkpoint["model_state"]), strict=True)
    return model.to(device).eval(), checkpoint


# -----------------------------------------------------------------------------
# CUSP
# -----------------------------------------------------------------------------

# Clinical prior pi(b) over the BI-RADS category b (index 0 = unknown). It peaks at
# BI-RADS 3, the category on the screening boundary (1-2 negative, 3-5 positive).
BIRADS_PRIOR = (1.0, 1.0, 1.2, 2.0, 1.4, 1.2)


def cusp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor | None,
    birads_level: Any,
    pos_weight: torch.Tensor | None = None,
    margin: float = 0.2,
) -> dict[str, torch.Tensor]:
    """CUSP: Clinically-guided, Uncertainty-weighted Supervision with Profile consistency.

    L_CU (clinically guided, uncertainty-weighted supervision):
        mean over images i of  pi(b_i) * mean_k a_ik * mean_k BCE_ik,
        a_ik = 1 - |2 p_ik - 1| (predictive ambiguity, no gradient), one weight per image
        shared by the three heads; BCE_ik uses the same positive weights as the main loss.
    L_PC (profile consistency), on the L2-normalised pooled features f_i:
        mean over usable anchors i of  softplus(d(f_i, f_p) - d(f_i, f_n) + margin),
        p = nearest image in the batch with the same label profile (all three labels equal),
        n = nearest image with a different profile.
    The total objective is L_BCE + lambda1 * L_CU + lambda2 * L_PC; nothing changes at inference.
    """
    logits = logits.float()
    labels = labels.float()
    batch = int(logits.shape[0])
    zero = logits.new_tensor(0.0)
    if batch < 1:
        return {"cu": zero, "pc": zero, "pc_anchors": zero}

    element = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    probability = torch.sigmoid(logits).detach()
    ambiguity = 1.0 - (2.0 * probability - 1.0).abs()
    prior = logits.new_ones(batch)
    if birads_level is not None:
        level = torch.as_tensor(birads_level, device=logits.device).to(dtype=torch.long).reshape(-1)[:batch]
        if int(level.numel()) != batch:
            raise ValueError("birads_level must have one entry per image")
        table = logits.new_tensor(BIRADS_PRIOR)
        prior = table[level.clamp(0, 5)]
    cu = (element.mean(dim=1) * ambiguity.mean(dim=1) * prior).mean()

    if features is None or batch < 2:
        return {"cu": cu, "pc": zero, "pc_anchors": zero}
    features = features.float()
    feat = F.normalize(features, dim=1, eps=1e-6)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).all(dim=-1)
    eye = torch.eye(batch, dtype=torch.bool, device=logits.device)
    pos_mask = same & ~eye
    neg_mask = ~same
    dist = torch.cdist(feat, feat, p=2)
    huge = dist.new_tensor(1.0e4)
    d_pos = torch.where(pos_mask, dist, huge).min(dim=1).values
    d_neg = torch.where(neg_mask, dist, huge).min(dim=1).values
    usable = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if int(usable.sum().item()) <= 0:
        return {"cu": cu, "pc": zero, "pc_anchors": zero}
    pc = F.softplus(d_pos[usable] - d_neg[usable] + float(margin)).mean()
    return {"cu": cu, "pc": pc, "pc_anchors": usable.sum().float()}


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def compute_pos_weight(
    frame: pd.DataFrame,
    cap: float = 20.0,
    columns: tuple[str, ...] | list[str] = CLASS_NAMES,
) -> torch.Tensor:
    """Positive weight per head: negatives / positives in the training set, capped."""
    weights = []
    for column in columns:
        positives = float(frame[column].sum())
        negatives = float(len(frame) - positives)
        weights.append(1.0 if positives <= 0 else min(negatives / positives, cap))
    return torch.tensor(weights, dtype=torch.float32)


def validate_split_disjoint(frames: dict[str, pd.DataFrame]) -> None:
    names = list(frames)
    for i, first in enumerate(names):
        first_studies = set(frames[first]["study_id"].astype(str))
        for second in names[i + 1 :]:
            overlap = first_studies.intersection(set(frames[second]["study_id"].astype(str)))
            if overlap:
                raise RuntimeError(f"Data leakage: {len(overlap)} exams appear in both {first} and {second}")


def dataframe_summary(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "images": int(len(frame)),
        "exams": int(frame["study_id"].nunique()),
        **{f"positive_{name}": int(frame[name].sum()) for name in CLASS_NAMES},
    }


def model_parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
