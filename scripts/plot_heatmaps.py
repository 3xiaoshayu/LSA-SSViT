#!/usr/bin/env python
"""Draw the heat-map figures of the paper.

    python scripts/plot_heatmaps.py --cache-root outputs/cache/vindr_1024 \\
        --inbreast-manifest outputs/inbreast/manifest.csv --inbreast-root /path/to/AllDICOMs

Figures (written to figures/ as vector PDF and 600-dpi PNG; style in scripts/figure_style.py):
    heatmap_evidence_maps       maps of the comparison networks, the Baseline and LSA-SSViT on VinDr-Mammo
                                test images, in two stacked blocks (BLOCKS); heatmap_evidence_maps_a and _b hold each
                                block on its own
    heatmap_localization        weakly supervised localization (results/heatmaps/localization_summary.json)
    heatmap_faithfulness_supp   deletion test and cascading randomization (supplementary figure)
    heatmap_inbreast_supp       the maps of heatmap_evidence_maps on INbreast images (supplementary figure); _a and _b
                                as for heatmap_evidence_maps

The localization and faithfulness figures need only results/heatmaps/. The two figures of maps also need the maps
written by scripts/heatmaps.py cam (outputs/heatmaps/; those of the comparison networks may be in --compare-cam-dir)
and the images. A comparison network whose maps are missing is left out (its column is not drawn); a figure is skipped
when no image source is given (--cache-root or --image-root; --inbreast-manifest and --inbreast-root) or the maps of the
Baseline or LSA-SSViT are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from figure_style import (  # noqa: E402
    BODY,
    COLOR,
    COMPARATOR_LABEL,
    DOUBLE_COLUMN,
    GREY_FRAME,
    GREY_TEXT,
    GRID_TITLE,
    HEAD_TITLE,
    INK,
    LABEL,
    LEGEND_LABEL,
    LETTER,
    MM,
    MODELS,
    REF_LINE,
    TITLE,
    decimal_formatter,
    line_width,
    panel_label,
    save,
    setup_style,
)
from heatmaps import (  # noqa: E402
    CLASS_NAMES,
    DELETION_FRACTIONS,
    IMAGE_SIZE,
    MASK_CHANNEL,
    RANDOMIZATION_STAGES,
    breast_mask,
    inbreast_lesion_masks,
    load_lesion_masks,
    preprocessing_geometry,
    read_split,
    unpack_masks,
    upsample,
)

STAGE_LABEL = {"original": "Trained", "head": "Head", "block12": "+Block 12", "block11": "+Block 11",
               "block10": "+Block 10", "block9": "+Block 9"}
CMAP = plt.get_cmap("turbo")
HEAT_FLOOR, HEAT_RAMP, ALPHA_MAX = 0.16, 0.55, 0.78  # overlay: transparent below the floor, opaque after the ramp
ASPECT = 1.20       # panel height / width
ZOOM_ASPECT = 0.80  # magnified rows: height / width
ROI_COLOR = "#FFD200"
IMAGE_DPI = 450     # resolution of the images embedded in the vector PDFs of the map figures

# Cases shown in the figures: (image_id, head, row label, add a magnified row). A row label "<head>, <detail>" is drawn
# on two lines, the detail in grey.
VINDR_CASES = [  # one example per head
    ("0f714f0b91dcd0ff05a60290440d1004", "birads", "Screening, BI-RADS 3", False),
    ("ad50a0920f06e7bf2ff5ea7e4eda123a", "mass", "Mass", False),
    ("5173f7390a4426b1d2ff13dc1db4111a", "calcification", "Calcification", True),
]
INBREAST_CASES = [  # the screening head, the only one scored on INbreast
    ("20587664", "birads", "Screening, BI-RADS 5", False),
    ("22580654", "birads", "Screening, BI-RADS 5", False),
    ("20587928", "birads", "Screening, BI-RADS 3", False),
    ("20588680", "birads", "Screening, BI-RADS 3", False),
]
# The two map figures stack two blocks with the same rows. The comparison networks are grouped by the spatial extent of
# token mixing in the stages that produce the mapped features: global, by self-attention over all tokens (block a, with
# the Baseline), or local, by convolution or within windows (block b; MambaVision-T applies its Mamba and self-attention
# blocks within 14 x 14- and 7 x 7-token windows at this input size). LSA-SSViT, the transformer with GRAIN's
# local mixing, closes both blocks. Within a block the comparison networks follow the comparison table; spans are
# (header, first and last network under it). Comparison networks without maps are left out.
BLOCKS = [
    {"columns": ["rope_vit_base", "dinov2_base", "rad_dino", "baseline", "LSA-SSViT"],
     "spans": [("Global self-attention", "rope_vit_base", "baseline")]},
    {"columns": ["resnet50", "efficientnet_b3", "convnext_tiny", "mobilenetv4_conv_medium", "mambavision_tiny",
                 "mammo_clip_b2", "LSA-SSViT"],
     "spans": [("Local token mixing", "resnet50", "mammo_clip_b2")]},
]
REQUIRED_COLUMNS = ["baseline", "LSA-SSViT"]
MAP_COLUMNS = list(dict.fromkeys(name for block in BLOCKS for name in block["columns"]))
# Column titles: the Baseline under the name of its backbone, as the comparison networks are named (comparison table:
# "Baseline (DINOv3-B/16)"), in INK like them; LSA-SSViT in its colour of the other figures and in bold.
NAME = {**LABEL, **COMPARATOR_LABEL, "baseline": "DINOv3-B/16"}
TITLE_COLOR = {name: COLOR["LSA-SSViT"] if name == "LSA-SSViT" else INK for name in MAP_COLUMNS}
OURS_TINT = "#E8F5F0"  # band behind the LSA-SSViT column of each block
BAND_PAD = 1.2  # mm of band around the LSA-SSViT panels
PT = 25.4 / 72  # mm per point
# Thin translucent dark edge under the white lesion boxes and the yellow region boxes: they stay visible on bright
# tissue and on yellow parts of a map, with no visible change on black or red.
EDGE = [patheffects.Stroke(linewidth=1.15, foreground=(0, 0, 0, 0.4)), patheffects.Normal()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the heat-map figures.")
    parser.add_argument("--cache-root", type=Path, default=None, help="VinDr-Mammo preprocessing cache")
    parser.add_argument("--image-root", type=Path, default=None, help="VinDr-Mammo DICOM root, used without a cache")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "splits" / "vindr" / "test.csv")
    parser.add_argument("--inbreast-manifest", type=Path, default=None, help="INbreast manifest (scripts/prepare_inbreast.py)")
    parser.add_argument("--inbreast-root", type=Path, default=None, help="INbreast DICOM folder")
    parser.add_argument("--inbreast-xml", type=Path, default=None,
                        help="INbreast AllXML folder (lesion contours); draws the annotated lesions as dashed boxes")
    parser.add_argument("--cam-dir", type=Path, default=PROJECT_ROOT / "outputs" / "heatmaps")
    parser.add_argument("--compare-cam-dir", type=Path, default=None,
                        help="maps of the comparison networks (scripts/heatmaps.py cam --models ...; default: --cam-dir)")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "heatmaps")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Cases
# -----------------------------------------------------------------------------


def map_file(args, dataset: str, name: str) -> Path:
    """Maps of one network (the comparison networks' in --compare-cam-dir when given)."""
    directory = args.compare_cam_dir if name in COMPARATOR_LABEL and args.compare_cam_dir else args.cam_dir
    return directory / f"cam_{dataset}_{name}.npz"


def missing_maps(args, dataset: str, columns: list[str] = MAP_COLUMNS) -> list[str]:
    return [name for name in columns if not map_file(args, dataset, name).is_file()]


def available_blocks(args, dataset: str) -> list[dict]:
    """BLOCKS without the comparison networks whose maps are missing (reported)."""
    absent = missing_maps(args, dataset, [name for name in MAP_COLUMNS if name not in REQUIRED_COLUMNS])
    if absent:
        print(f"  {dataset}: no maps of {', '.join(absent)} (scripts/heatmaps.py cam --models ...; --compare-cam-dir); "
              "their columns are left out")
    blocks = []
    for block in BLOCKS:
        spans = []
        for text, first, last in block["spans"]:  # each header over the networks of its range that remain
            names = block["columns"][block["columns"].index(first):block["columns"].index(last) + 1]
            kept = [name for name in names if name not in absent]
            if kept:
                spans.append((text, kept[0], kept[-1]))
        blocks.append({"columns": [name for name in block["columns"] if name not in absent], "spans": spans})
    return blocks


def block_columns(blocks: list[dict]) -> list[str]:
    return list(dict.fromkeys(name for block in blocks for name in block["columns"]))


class CaseLoader:
    """Image, lesion masks and maps of the networks in columns for the images shown in a figure."""

    def __init__(self, args, dataset: str, image_ids: list[str], columns: list[str] = MAP_COLUMNS) -> None:
        from core import load_cached_image, prepare_image

        self.columns = columns
        self.cams = {}
        for name in columns:
            with np.load(map_file(args, dataset, name)) as stored:
                index = {str(i): k for k, i in enumerate(stored["image_id"])}
                cam = stored["cam"]
                self.cams[name] = {i: cam[index[i]].astype(np.float32) for i in image_ids}
        if dataset == "vindr":
            frame = read_split(args.test_csv).set_index("image_id", drop=False)
            lesions = load_lesion_masks(args.results_dir)
            if args.cache_root is not None:
                self.images = {i: load_cached_image(args.cache_root, frame.loc[i]) for i in image_ids}
            else:
                self.images = {i: prepare_image(frame.loc[i], args.image_root) for i in image_ids}
            self.masks = {i: unpack_masks(lesions[i]) if i in lesions.files else None for i in image_ids}
        else:
            frame = pd.read_csv(args.inbreast_manifest, dtype={"study_id": str, "series_id": str, "image_id": str})
            frame = frame.set_index("image_id", drop=False)
            self.images = {i: prepare_image(frame.loc[i], args.inbreast_root) for i in image_ids}
            if getattr(args, "inbreast_xml", None) is not None:
                self.masks = {i: inbreast_lesion_masks(args.inbreast_xml / f"{i}.xml",
                                                       preprocessing_geometry(frame.loc[i], args.inbreast_root))
                              for i in image_ids}
            else:
                self.masks = {i: None for i in image_ids}

    def __call__(self, image_id: str) -> dict:
        masks = self.masks[image_id]
        return {"image": self.images[image_id],
                "masks": np.zeros((2, IMAGE_SIZE, IMAGE_SIZE), bool) if masks is None else masks,
                "cam": [self.cams[m][image_id] for m in self.columns]}  # 3 x H x W each (32, 64 or 73)


# -----------------------------------------------------------------------------
# Map panels
# -----------------------------------------------------------------------------


def chest_wall_on_right(breast: np.ndarray) -> bool:
    """The chest-wall side of the breast is the long, straight vertical edge."""
    columns = breast.sum(axis=0)
    xs = np.flatnonzero(columns > 0)
    if len(xs) < 20:
        return False
    return columns[xs[-1] - 12: xs[-1] - 2].mean() > columns[xs[0] + 3: xs[0] + 13].mean()


def display_map(cam: np.ndarray, breast: np.ndarray) -> np.ndarray:
    """Map for display: above-median values, scaled to the map's 99.5th percentile."""
    full = upsample(cam)
    full = full - np.median(full[breast])
    top = np.percentile(full[breast], 99.5)
    full = np.clip(full / top if top > 0 else full * 0, 0, 1) * breast
    return ndimage.gaussian_filter(full, 2.0)


def frame_for(breast: np.ndarray, aspect: float = ASPECT, pad: float = 0.025) -> tuple[int, int, int, int]:
    """Crop window (y0, y1, x0, x1) around the breast with a fixed aspect ratio."""
    ys, xs = np.where(breast)
    p = int(pad * IMAGE_SIZE)
    y0, y1 = max(0, ys.min() - p), min(IMAGE_SIZE, ys.max() + p)
    x0, x1 = max(0, xs.min() - p), min(IMAGE_SIZE, xs.max() + p)
    h, w = y1 - y0, x1 - x0
    if h / w > aspect:
        need = int(h / aspect) - w
        x0, x1 = x0 - need // 2, x1 + need - need // 2
    else:
        need = int(w * aspect) - h
        y0, y1 = y0 - need // 2, y1 + need - need // 2
    return y0, y1, x0, x1


def crop(array: np.ndarray, window, fill=0.0) -> np.ndarray:
    """Crop, padding outside the 1024 x 1024 canvas."""
    y0, y1, x0, x1 = window
    out = np.full((y1 - y0, x1 - x0) + array.shape[2:], fill, dtype=array.dtype)
    sy0, sy1, sx0, sx1 = max(0, y0), min(IMAGE_SIZE, y1), max(0, x0), min(IMAGE_SIZE, x1)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = array[sy0:sy1, sx0:sx1]
    return out


def compose_rgb(image: np.ndarray, heat: np.ndarray | None = None) -> np.ndarray:
    gray = np.clip(image, 0, 1) ** 0.9
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if heat is None:
        return rgb
    alpha = (np.clip((heat - HEAT_FLOOR) / HEAT_RAMP, 0, 1) * ALPHA_MAX)[..., None]
    return rgb * (1 - alpha) + CMAP(heat)[..., :3] * alpha


def lesion_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    labels, _ = ndimage.label(mask)
    return [(s[0].start, s[0].stop, s[1].start, s[1].stop) for s in ndimage.find_objects(labels) if s is not None]


def prepare(case: dict, head: str, window_image: bool = False) -> dict:
    """Image with the chest wall on the left, cropped, with the maps of the networks and the lesion boxes; every map is
    displayed with the same rule (display_map), whatever its grid. With window_image the image is displayed between
    the 0.5th and 99.9th percentiles of its breast pixels (stretched at most twofold) with a gamma of 1.5."""
    image = case["image"]
    breast = breast_mask(image)
    if window_image:
        low, high = np.percentile(image[breast], [0.5, 99.9])
        high = max(high, low + 0.5)
        image = np.clip((image - low) / (high - low), 0, 1) ** 1.5
    h = CLASS_NAMES.index(head)
    lesion = case["masks"][0] | case["masks"][1] if head == "birads" else case["masks"][MASK_CHANNEL[head]]
    heats = [display_map(cam[h], breast) for cam in case["cam"]]
    if chest_wall_on_right(breast):
        image, breast, lesion = image[:, ::-1], breast[:, ::-1], lesion[:, ::-1]
        heats = [m[:, ::-1] for m in heats]
    window = frame_for(breast)
    return {"image": crop(image, window), "heats": [crop(m, window) for m in heats], "lesion": crop(lesion, window, False),
            "valid": crop(np.ones((IMAGE_SIZE, IMAGE_SIZE), bool), window, False),  # inside the 1024 x 1024 canvas
            "breast": crop(breast, window, False)}


def magnified_window(lesion: np.ndarray, valid: np.ndarray, breast: np.ndarray | None = None,
                     scale: float = 2.4, minimum: int = 170, aspect: float = ZOOM_ASPECT):
    """Window (y0, y1, x0, x1) of the magnified row around the lesion, inside the image (valid), not its padding, and
    not left of the chest wall (breast)."""
    ys, xs = np.where(lesion)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    w = max(int(max(ys.max() - ys.min(), xs.max() - xs.min()) * scale), minimum)
    h = int(w * aspect)
    vy, vx = np.where(valid)
    y_lo, y_hi, x_lo, x_hi = vy.min(), vy.max() + 1, vx.min(), vx.max() + 1
    if breast is not None and breast.any():
        x_lo = max(x_lo, int(np.flatnonzero(breast.any(axis=0)).min()))
    y0 = int(np.clip(cy - h // 2, y_lo, max(y_hi - h, y_lo)))
    x0 = int(np.clip(cx - w // 2, x_lo, max(x_hi - w, x_lo)))
    return y0, min(y_hi, y0 + h), x0, min(x_hi, x0 + w)


def dashed_box(axis: plt.Axes, x0: float, y0: float, x1: float, y1: float, pt_per_px: float, lw: float = 0.75,
               dash: tuple[float, float] = (2.2, 1.4)) -> None:
    """Dashed rectangle drawn side by side, each side's dash pattern stretched slightly so that it starts and ends on
    a dash: all four corners are closed."""
    for xs, ys in (((x0, x1), (y0, y0)), ((x1, x1), (y0, y1)), ((x1, x0), (y1, y1)), ((x0, x0), (y1, y0))):
        units = max(abs(xs[1] - xs[0]), abs(ys[1] - ys[0])) * pt_per_px / lw  # side length in line widths
        n = max(1, round((units - dash[0]) / (dash[0] + dash[1])))
        k = units / (n * (dash[0] + dash[1]) + dash[0])
        axis.add_line(Line2D(xs, ys, color="white", lw=lw, dashes=(dash[0] * k, dash[1] * k), dash_capstyle="butt",
                             path_effects=EDGE))


def draw_panel(axis: plt.Axes, image, heat, lesion, roi=None) -> None:
    axis.imshow(compose_rgb(image, heat), interpolation="lanczos")
    box = axis.get_position()
    size = axis.figure.get_size_inches() * 72
    pt_per_px = min(box.width * size[0] / image.shape[1], box.height * size[1] / image.shape[0])
    for y0, y1, x0, x1 in lesion_boxes(lesion):
        dashed_box(axis, x0 - 0.5, y0 - 0.5, x1 - 0.5, y1 - 0.5, pt_per_px)
    if roi is not None:
        y0, y1, x0, x1 = roi
        axis.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=ROI_COLOR, lw=0.75,
                                         path_effects=EDGE))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def evidence_figure(loader: CaseLoader, cases: list, blocks: list[dict], window_image: bool = False,
                    grid: int | None = None, letters: bool = True) -> plt.Figure:
    """The blocks stacked, each with the same rows: one row per case (mammogram and the maps of the block's networks),
    plus a magnified row where requested. Lengths in mm. All blocks share one panel size and spacing, set by the widest
    block (180 mm wide); a narrower block is centred."""
    rows = []
    for image_id, head, label, magnify in cases:
        rows.append({"id": image_id, "head": head, "label": label, "part": "full", "magnify": magnify})
        if magnify:
            rows.append({"id": image_id, "head": head, "label": label, "part": "zoom", "magnify": True})
    width = DOUBLE_COLUMN / MM
    left, right, gap_anchor, gap_map = 7.6, 2.2, 2.5, 1.2
    widest = grid or max(len(block["columns"]) - 1 for block in blocks)  # map columns before the last one
    pw = (width - left - right - 2 * gap_anchor - (widest - 1) * gap_map) / (widest + 2)  # 19.9 mm
    header, row_gap, zoom_gap, block_gap, footer = 8.0, 1.5, 0.6, 5.0, 6.2
    heights = [pw * (ZOOM_ASPECT if row["part"] == "zoom" else ASPECT) for row in rows]
    gaps = [zoom_gap if rows[r + 1]["part"] == "zoom" else row_gap for r in range(len(rows) - 1)]
    block_height = header + sum(heights) + sum(gaps)
    height = len(blocks) * block_height + (len(blocks) - 1) * block_gap + footer
    figure = plt.figure(figsize=(width * MM, height * MM))

    def fy(y: float) -> float:  # mm from the top -> figure fraction
        return 1 - y / height

    prepared = {}
    for row in rows:
        if row["id"] not in prepared:
            prepared[row["id"]] = prepare(loader(row["id"]), row["head"], window_image)
    rois = {row["id"]: magnified_window(prepared[row["id"]]["lesion"], prepared[row["id"]]["valid"],
                                        prepared[row["id"]]["breast"])
            for row in rows if row["magnify"]}
    for b, block in enumerate(blocks):
        top = b * (block_height + block_gap)
        names = block["columns"]
        last = len(names)
        span = 2 * pw + 2 * gap_anchor + (last - 1) * pw + (last - 2) * gap_map  # mammogram to last column
        x_left = left + (width - left - right - span) / 2

        def x_of(c: int) -> float:  # left edge of column c (0: mammogram, last: LSA-SSViT)
            if c == 0:
                return x_left
            if c == last:
                return x_left + span - pw
            return x_left + pw + gap_anchor + (c - 1) * (pw + gap_map)

        # header: group names over thin rules (baseline 3.0 mm and rule 4.3 mm below the block top), then the column
        # titles (baseline 7.0 mm); the block letter at the left edge, on the group-name baseline
        for text, first_name, last_name in block["spans"]:
            if first_name not in names or last_name not in names:
                continue
            x0 = x_of(names.index(first_name) + 1) + 0.4
            x1 = x_of(names.index(last_name) + 1) + pw - 0.4
            figure.text((x0 + x1) / 2 / width, fy(top + 3.0), text, ha="center", va="baseline", fontsize=TITLE,
                        fontweight="bold", color=INK)
            figure.add_artist(Line2D([x0 / width, x1 / width], [fy(top + 4.3)] * 2, lw=0.5, color=INK))
        if letters:
            figure.text(1.9 / width, fy(top + 3.0), chr(ord("a") + b), ha="left", va="baseline", fontsize=LETTER,
                        fontweight="bold")
        for c in range(last + 1):
            name = names[c - 1] if c > 0 else None
            title, color = ("Mammogram", INK) if name is None else (NAME[name], TITLE_COLOR[name])
            figure.text((x_of(c) + pw / 2) / width, fy(top + 7.0), title, ha="center", va="baseline",
                        fontsize=GRID_TITLE, color=color, fontweight="bold" if name == "LSA-SSViT" else "normal")
        key_right = x_of(last) + pw  # the colour key ends under the last column of the last block
        # band behind the LSA-SSViT column, from the group-rule line to BAND_PAD below the last row
        band_top, band_bottom = top + 4.3, top + block_height + BAND_PAD
        figure.add_artist(patches.Rectangle(((x_of(last) - BAND_PAD) / width, fy(band_bottom)),
                                            (pw + 2 * BAND_PAD) / width, (band_bottom - band_top) / height,
                                            transform=figure.transFigure, fc=OURS_TINT, ec="none", zorder=-1))
        y = top + header
        for r, row in enumerate(rows):
            d = prepared[row["id"]]
            roi = rois.get(row["id"])
            for c in range(last + 1):
                axis = figure.add_axes([x_of(c) / width, fy(y + heights[r]), pw / width, heights[r] / height])
                heat = None if c == 0 else d["heats"][loader.columns.index(names[c - 1])]
                if row["part"] == "zoom":
                    y0, y1, x0, x1 = roi
                    draw_panel(axis, d["image"][y0:y1, x0:x1], None if heat is None else heat[y0:y1, x0:x1],
                               d["lesion"][y0:y1, x0:x1])
                    dx, dy = 0.375 * PT / pw, 0.375 * PT / heights[r]  # frame inside the panel: half its stroke
                    axis.add_patch(patches.Rectangle((dx, dy), 1 - 2 * dx, 1 - 2 * dy, transform=axis.transAxes,
                                                     fill=False, ec=ROI_COLOR, lw=0.75, clip_on=False))
                else:
                    draw_panel(axis, d["image"], heat, d["lesion"], roi=roi)
            # row label left of the mammogram: "<head>, <detail>" on two lines, the detail in grey; the inner line of
            # every label on one baseline 1.2 mm from the panels
            mid = fy(y + heights[r] / 2)
            label = {"rotation": 90, "rotation_mode": "anchor", "ha": "center", "va": "baseline"}
            inner, outer = x_left - 1.2, x_left - 3.9
            if row["part"] == "zoom":
                figure.text(inner / width, mid, "Magnified", fontsize=GRID_TITLE, color=GREY_TEXT, **label)
            else:
                head_text, _, detail = row["label"].partition(", ")
                if detail:
                    figure.text(outer / width, mid, head_text, fontsize=GRID_TITLE, color=INK, **label)
                    figure.text(inner / width, mid, detail, fontsize=GRID_TITLE, color=GREY_TEXT, **label)
                else:
                    figure.text(inner / width, mid, head_text, fontsize=GRID_TITLE, color=INK, **label)
            y += heights[r] + (gaps[r] if r < len(gaps) else 0)
    # colour key: one line in the footer, right-aligned to the last column of the last block
    renderer = figure.canvas.get_renderer()
    baseline = fy(height - 2.0)

    def text_width(text) -> float:  # mm
        return text.get_window_extent(renderer).width / figure.dpi / MM

    high = figure.text(key_right / width, baseline, "High", ha="right", va="baseline", fontsize=BODY, color=INK)
    bar_w, bar_h = 15.0, 1.3
    bar_x1 = key_right - text_width(high) - 1.0
    cax = figure.add_axes([(bar_x1 - bar_w) / width, fy(height - 2.0 - 0.11), bar_w / width, bar_h / height])
    gradient = np.tile(np.linspace(0, 1, 256), (8, 1))
    cax.imshow(compose_rgb(np.full(gradient.shape, 0.45), gradient), aspect="auto", extent=(0, 1, 0, 1),
               interpolation="bilinear")
    cax.set_xticks([])
    cax.set_yticks([])
    for spine in cax.spines.values():  # all four sides (the house style hides the top and right spines)
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color(GREY_FRAME)
    low = figure.text((bar_x1 - bar_w - 1.0) / width, baseline, "Low", ha="right", va="baseline", fontsize=BODY,
                      color=INK)
    figure.text((bar_x1 - bar_w - 1.0 - text_width(low) - 2.5) / width, baseline, "Relative map value", ha="right",
                va="baseline", fontsize=BODY, color=INK)
    return figure


# -----------------------------------------------------------------------------
# Localization and faithfulness
# -----------------------------------------------------------------------------


def shared_legend(figure: plt.Figure, axes: list, handles: list, labels: list, top_mm: float = 1.6) -> None:
    """One-row legend centred over the union of the axes."""
    lo = min(a.get_position().x0 for a in axes)
    hi = max(a.get_position().x1 for a in axes)
    height = figure.get_size_inches()[1]
    figure.legend(handles, labels, loc="upper center", ncol=len(labels),
                  bbox_to_anchor=((lo + hi) / 2, 1 - top_mm * MM / height))


def localization_figure(summary: dict) -> plt.Figure:
    heads = ["birads", "mass", "calcification"]
    metrics = [("hit", "Pointing game", "Hit rate"), ("top1", "Top-1% precision", "Precision")]
    width, height = DOUBLE_COLUMN, 54.1 * MM
    figure = plt.figure(figsize=(width, height))
    left, gap, right, bottom, top = 9.6 * MM, 19.0 * MM, 2.0 * MM, 7.6 * MM, 11.6 * MM
    panel_w = (width - left - gap - right) / 2
    axes = [figure.add_axes([(left + i * (panel_w + gap)) / width, bottom / height, panel_w / width,
                             (height - bottom - top) / height]) for i in range(2)]
    bar_w = 0.19
    for i, (metric, title, y_label) in enumerate(metrics):
        axis = axes[i]
        for g, head in enumerate(heads):
            cell = summary[head]
            chance = cell["baseline"][f"chance_{metric}"]
            for k, (name, _, color) in enumerate(MODELS):
                mean, low, high = cell[name][metric]
                x = g + (k - 1.5) * bar_w
                axis.bar(x, mean, bar_w * 0.86, color=color, lw=0, zorder=2,
                         label=LEGEND_LABEL[name] if (g == 0 and i == 0) else None)
                if name == "baseline":  # the black bar would hide the lower half of the interval: draw it in white
                    axis.vlines(x, low, mean, color="white", lw=0.6, zorder=2.2)
                    axis.plot([x], [low], marker="_", ms=3.2, mew=0.6, color="white", zorder=2.2)
                    axis.vlines(x, mean, high, color=INK, lw=0.6, zorder=3)
                    axis.plot([x], [high], marker="_", ms=3.2, mew=0.6, color=INK, zorder=3)
                else:
                    bars = axis.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="none", ecolor=INK,
                                         elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=3)
                    for cap in bars[1]:
                        cap.set_markerfacecolor(INK)
                delta = cell[name].get(f"delta_vs_baseline_{metric}")
                if delta is not None and delta[1] > 0 and mean > chance:
                    axis.annotate("*", xy=(x, high), xytext=(0, -1.0), textcoords="offset points", ha="center",
                                  va="baseline", fontsize=TITLE, annotation_clip=False)
            axis.plot([g - 2.2 * bar_w, g + 2.2 * bar_w], [chance, chance], zorder=2.5, dash_capstyle="butt", **REF_LINE)
        axis.set_xlim(-0.5, 2.5)
        axis.set_xticks(range(3))
        axis.set_xticklabels([f"{HEAD_TITLE[h]}\n($n$ = {summary[h]['n_images']})" for h in heads])
        axis.tick_params(axis="x", length=0, pad=3)
        axis.set_ylim(0, 0.7)
        axis.yaxis.set_major_locator(MultipleLocator(0.1))
        axis.yaxis.set_major_formatter(decimal_formatter(1))
        axis.set_ylabel(y_label)
        panel_label(axis, chr(ord("a") + i), title)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], **REF_LINE))
    labels.append("Chance level")
    shared_legend(figure, axes, handles, labels)
    return figure


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    return np.percentile([rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)], [2.5, 97.5])


RANDOM_DASH = (0, (2.4, 1.6))
# Line-style keys of the deletion AUC table, centred under their column heads: six whole dashes of the random-order
# style at 0.7 pt (dash lengths scale with the line width), so that the dashed key starts and ends on a dash.
TABLE_KEY_LEN_MM = (6 * 2.4 + 5 * 1.6) * 0.7 / 72 * 25.4


def hidden_curves(axis: plt.Axes, artists: list, deleted: pd.DataFrame, margin_pt: float = 1.0) -> list:
    """Mean curves that pass through the box of the table artists (display coordinates, plus margin_pt points)."""
    renderer = axis.figure.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer) for a in artists]
    pad = margin_pt * axis.figure.dpi / 72
    x0, x1 = min(b.x0 for b in boxes) - pad, max(b.x1 for b in boxes) + pad
    y0, y1 = min(b.y0 for b in boxes) - pad, max(b.y1 for b in boxes) + pad
    x = np.linspace(0, 1, 1001)
    hidden = []
    for name, _, _ in MODELS:
        for order in ("cam", "random"):
            rows = deleted[(deleted["model"] == name) & (deleted["order"] == order)]
            curve = np.array([json.loads(c) for c in rows["curve"]]).mean(axis=0)
            points = axis.transData.transform(np.column_stack([x * 100, np.interp(x, DELETION_FRACTIONS, curve)]))
            if ((points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)).any():
                hidden.append((name, order))
    return hidden


def deletion_panel(axis: plt.Axes, deleted: pd.DataFrame) -> None:
    for name, _, color in MODELS:
        for order in ("random", "cam"):
            rows = deleted[(deleted["model"] == name) & (deleted["order"] == order)]
            curve = np.array([json.loads(c) for c in rows["curve"]]).mean(axis=0)
            if order == "cam":
                axis.plot(DELETION_FRACTIONS * 100, curve, color=color, lw=line_width(name), zorder=3, clip_on=False,
                          solid_capstyle="butt")
            else:
                axis.plot(DELETION_FRACTIONS * 100, curve, color=color, lw=0.7, ls=RANDOM_DASH, zorder=2, clip_on=False)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 1.04)
    axis.spines["left"].set_bounds(0, 1.0)
    for side in ("left", "bottom"):
        axis.spines[side].set_position(("outward", 3))
    axis.xaxis.set_major_locator(MultipleLocator(20))
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.yaxis.set_major_formatter(decimal_formatter(1))
    axis.set_xlabel("Breast tokens removed (%)")
    axis.set_ylabel("Predicted probability")
    # Deletion AUC table in the empty lower-left corner; its column heads double as the line-style key. No background
    # box: the table must not hide any curve, which hidden_curves() checks on the rendered text.
    auc = deleted.groupby(["model", "order"])["auc"].mean()
    box = (0.025, 0.05, 0.585, 0.37)
    pitch, top = 0.058, box[1] + box[3] - 0.03
    text = {"transform": axis.transAxes, "fontsize": BODY, "zorder": 5}
    renderer = axis.figure.canvas.get_renderer()
    to_axes = axis.transAxes.inverted()
    artists = [axis.text(box[0] + 0.02, top, "Deletion AUC", ha="left", va="top", color=GREY_TEXT, **text)]
    centres = []
    for row, (name, label, color) in enumerate(MODELS):
        y = top - 0.12 - row * pitch
        artists.append(axis.add_line(Line2D([box[0] + 0.02, box[0] + 0.065], [y, y], transform=axis.transAxes,
                                            color=color, lw=line_width(name), solid_capstyle="butt", zorder=5)))
        artists.append(axis.text(box[0] + 0.08, y, label, ha="left", va="center_baseline", **text))
        for order, right in (("cam", 0.45), ("random", 0.595)):
            number = axis.text(right, y, f"{auc[(name, order)]:.2f}", ha="right", va="center_baseline", **text)
            artists.append(number)
            if row == 0:  # all numbers of a column have the same width (Arial digits): one centre per column
                extent = number.get_window_extent(renderer)
                centres.append(to_axes.transform([((extent.x0 + extent.x1) / 2, 0)])[0, 0])
    # column heads and line-style keys centred over their number columns
    axis_width = axis.get_window_extent().transformed(axis.figure.dpi_scale_trans.inverted()).width  # inches
    half = TABLE_KEY_LEN_MM / 2 * MM / axis_width
    for (head, style, lw), x in zip([("CAM", "-", 1.0), ("Random", RANDOM_DASH, 0.7)], centres):
        artists.append(axis.text(x, top, head, ha="center", va="top", color=GREY_TEXT, **text))
        artists.append(axis.add_line(Line2D([x - half, x + half], [top - 0.06, top - 0.06], transform=axis.transAxes,
                                            color=GREY_TEXT, lw=lw, ls=style, zorder=5)))
    hidden = hidden_curves(axis, artists, deleted)
    if hidden:
        print(f"  WARNING: the deletion AUC table hides part of {', '.join(f'{n}/{o}' for n, o in hidden)}")
    panel_label(axis, "a", f"Deletion, calcification head ($n$ = {deleted['image_id'].nunique()} images)")


def randomization_panel(axis: plt.Axes, randomized: pd.DataFrame) -> None:
    x = np.arange(len(RANDOMIZATION_STAGES))
    bands = {}
    for name in ("baseline", "LSA-SSViT"):
        means, lows, highs = [1.0], [1.0], [1.0]
        for stage in RANDOMIZATION_STAGES[1:]:
            values = randomized[(randomized["model"] == name) & (randomized["stage"] == stage)]["spearman_abs"].to_numpy()
            low, high = bootstrap_ci(values)
            means.append(values.mean())
            lows.append(low)
            highs.append(high)
        bands[name] = (np.array(lows), np.array(highs))
        axis.fill_between(x, lows, highs, color=COLOR[name], alpha=0.15, lw=0)
        axis.plot(x[1:], means[1:], color=COLOR[name], lw=line_width(name), marker="o", ms=2.5, mec="white", mew=0.3,
                  zorder=3)
        axis.plot(x[:2], means[:2], color=COLOR[name], lw=line_width(name), zorder=3)
    axis.plot([0], [1.0], marker="o", ms=2.5, color="#666666", mec="white", mew=0.3, zorder=4)
    axis.axhline(0, zorder=1, **REF_LINE)
    axis.set_xticks(x)
    axis.set_xticklabels([STAGE_LABEL[s] for s in RANDOMIZATION_STAGES])
    axis.set_xlim(-0.3, len(x) - 0.7)
    axis.set_xlabel("Layers re-initialized (cumulative)")
    axis.set_ylim(-0.2, 1.05)
    axis.set_yticks(np.arange(-0.2, 1.01, 0.2))
    axis.yaxis.set_major_formatter(decimal_formatter(1))
    axis.spines["left"].set_bounds(-0.2, 1.0)
    for side in ("left", "bottom"):
        axis.spines[side].set_position(("outward", 3))
    axis.set_ylabel(r"Spearman $\rho$ with the trained map")
    # Direct labels between Block 12 and Block 11, where the two bands are well separated.
    upper_base = (bands["baseline"][1][2] + bands["baseline"][1][3]) / 2
    upper_gc = (bands["LSA-SSViT"][1][2] + bands["LSA-SSViT"][1][3]) / 2
    lower_base = (bands["baseline"][0][2] + bands["baseline"][0][3]) / 2
    axis.text(2.5, upper_base + 0.04, "Baseline", color=COLOR["baseline"], fontsize=BODY, ha="center", va="bottom")
    axis.text(2.5, (upper_gc + lower_base) / 2, LABEL["LSA-SSViT"], color=COLOR["LSA-SSViT"], fontsize=BODY, ha="center",
              va="center")
    panel_label(axis, "b", f"Cascading randomization ($n$ = {randomized['image_id'].nunique()} images)")


def faithfulness_figure(deleted: pd.DataFrame, randomized: pd.DataFrame) -> plt.Figure:
    width, height = DOUBLE_COLUMN, 66.4 * MM
    figure = plt.figure(figsize=(width, height))
    # left as in the ROC figures, which have the same 3-pt outward spines (panel widths kept: the gap gives the 1.1 mm)
    left, gap, right, bottom, top = 10.7 * MM, 17.9 * MM, 2.0 * MM, 9.9 * MM, 11.6 * MM
    panel_w = (width - left - gap - right) / 2
    axes = [figure.add_axes([(left + i * (panel_w + gap)) / width, bottom / height, panel_w / width,
                             (height - bottom - top) / height]) for i in range(2)]
    deletion_panel(axes[0], deleted)
    randomization_panel(axes[1], randomized)
    handles = [Line2D([], [], color=color, lw=line_width(name)) for name, _, color in MODELS]
    shared_legend(figure, axes, handles, [LEGEND_LABEL[name] for name, _, _ in MODELS])
    return figure


def save_blocks(loader: CaseLoader, cases: list, blocks: list[dict], output_dir: Path, name: str,
                window_image: bool = False) -> None:
    """Each block of a map figure as a figure of its own (<name>_a, <name>_b): the same panels and spacing as in the
    combined figure, its own colour key, no block letter."""
    grid = max(len(block["columns"]) - 1 for block in blocks)
    for b, block in enumerate(blocks):
        save(evidence_figure(loader, cases, [block], window_image, grid=grid, letters=False), output_dir,
             f"{name}_{chr(ord('a') + b)}", IMAGE_DPI)


def main() -> None:
    args = parse_args()
    setup_style()
    print(f"Writing figures to {args.output_dir}")
    if args.cache_root is None and args.image_root is None:
        print("  heatmap_evidence_maps skipped (pass --cache-root or --image-root)")
    elif missing_maps(args, "vindr", REQUIRED_COLUMNS):
        print(f"  heatmap_evidence_maps skipped (maps of {', '.join(missing_maps(args, 'vindr', REQUIRED_COLUMNS))} "
              f"not found; run scripts/heatmaps.py cam)")
    else:
        blocks = available_blocks(args, "vindr")
        loader = CaseLoader(args, "vindr", [c[0] for c in VINDR_CASES], block_columns(blocks))
        save(evidence_figure(loader, VINDR_CASES, blocks), args.output_dir, "heatmap_evidence_maps", IMAGE_DPI)
        save_blocks(loader, VINDR_CASES, blocks, args.output_dir, "heatmap_evidence_maps")
    summary = json.loads((args.results_dir / "localization_summary.json").read_text(encoding="utf-8"))
    save(localization_figure(summary), args.output_dir, "heatmap_localization")
    deleted = pd.read_csv(args.results_dir / "faithfulness_deletion.csv", dtype={"image_id": str})
    randomized = pd.read_csv(args.results_dir / "faithfulness_randomization.csv", dtype={"image_id": str})
    save(faithfulness_figure(deleted, randomized), args.output_dir, "heatmap_faithfulness_supp")
    if args.inbreast_manifest is None or args.inbreast_root is None:
        print("  heatmap_inbreast_supp skipped (pass --inbreast-manifest and --inbreast-root)")
    elif missing_maps(args, "inbreast", REQUIRED_COLUMNS):
        print(f"  heatmap_inbreast_supp skipped (maps of {', '.join(missing_maps(args, 'inbreast', REQUIRED_COLUMNS))} "
              f"not found; run scripts/heatmaps.py cam --inbreast-manifest ...)")
    else:
        blocks = available_blocks(args, "inbreast")
        loader = CaseLoader(args, "inbreast", [c[0] for c in INBREAST_CASES], block_columns(blocks))
        save(evidence_figure(loader, INBREAST_CASES, blocks, window_image=True), args.output_dir, "heatmap_inbreast_supp",
             IMAGE_DPI)
        save_blocks(loader, INBREAST_CASES, blocks, args.output_dir, "heatmap_inbreast_supp", window_image=True)


if __name__ == "__main__":
    main()
