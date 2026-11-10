#!/usr/bin/env python
"""Draw the result figures of the paper from the released predictions.

Run scripts/paper_tables.py first: the confidence intervals and paired
differences shown in the figures are read from results/summary.json, so every
number in a figure is identical to the corresponding table.

Figures (written to figures/ as vector PDF and 600-dpi PNG; style in scripts/figure_style.py):
    roc_image_level        image-level ROC curves, three tasks, with zoomed insets
    auc_difference_forest  paired differences in image-level AUC (forest plot)
    confusion_matrices     confusion matrices at validation-selected thresholds
    roc_exam_level         exam-level ROC curves (supplementary figure)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from figure_style import (  # noqa: E402
    BODY,
    COLOR,
    DOUBLE_COLUMN,
    GREY_COMPARE,
    GREY_FRAME,
    GREY_TEXT,
    HEADS,
    INK,
    LABEL,
    LEGEND_LABEL,
    LETTER,
    MM,
    MODELS,
    ONE_HALF_COLUMN,
    REF_LINE,
    SMALL,
    TITLE,
    decimal_formatter,
    line_width,
    panel_label,
    save,
    setup_style,
)
from metrics import aggregate_views, safe_auc  # noqa: E402

# Zoomed windows (FPR min, FPR max, TPR min, TPR max): the range in which the curves separate.
ZOOM = {
    "image": {
        "birads": (0.05, 0.40, 0.45, 0.90),
        "mass": (0.05, 0.40, 0.45, 0.90),
        "calcification": (0.00, 0.15, 0.55, 0.95),
    },
    "exam": {
        "birads": (0.05, 0.40, 0.45, 0.90),
        "mass": (0.05, 0.40, 0.45, 0.90),
        "calcification": (0.00, 0.15, 0.55, 0.95),
    },
}
INSET = (0.53, 0.10, 0.44, 0.40)  # inset position in axes fraction
CARD = (0.41, 0.0, 0.59, 0.53)  # inset and its tick labels: no curve may pass here, the chance line pauses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the result figures of the paper.")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def load_predictions(results_root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for name, _, _ in MODELS:
        frames[name] = pd.read_csv(results_root / "vindr" / name / "predictions_test.csv",
                                   dtype={"study_id": str, "image_id": str})
    return frames


def load_summary(results_root: Path) -> dict:
    path = results_root / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run scripts/paper_tables.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# ROC curves
# -----------------------------------------------------------------------------


def points_offset(axis: plt.Axes, dx: float, dy: float) -> transforms.Transform:
    """Axes-fraction coordinates shifted by (dx, dy) points."""
    return axis.transAxes + transforms.ScaledTranslation(dx / 72, dy / 72, axis.figure.dpi_scale_trans)


def auc_table(axis: plt.Axes, rows: list[tuple[str, str, str, str]], panel_pt: float) -> None:
    """AUC (95% CI) key under a panel: line sample, model name and value in aligned columns."""
    x_name, x_value, y_head, pitch = 9.0, 86.0, -34.0, 7.2
    text = {"xy": (0, 0), "xycoords": "axes fraction", "textcoords": "offset points", "ha": "left", "va": "baseline",
            "fontsize": BODY, "annotation_clip": False}
    axis.annotate("AUC (95% CI)", xytext=(x_value, y_head), color=GREY_TEXT, **text)
    for row, (name, label, color, value) in enumerate(rows):
        y = y_head - (row + 1) * pitch
        axis.add_line(Line2D([0, 9 / panel_pt], [0, 0], transform=points_offset(axis, -3, y + 2.1), color=color,
                             lw=line_width(name), solid_capstyle="butt", clip_on=False))
        axis.annotate(label, xytext=(x_name, y), **text)
        axis.annotate(value, xytext=(x_value, y), **text)


def roc_figure(frames: dict[str, pd.DataFrame], summary: dict, level: str) -> plt.Figure:
    """Three ROC panels (screening, mass, calcification) with a zoomed inset each."""
    table = summary["vindr"]["image_level" if level == "image" else "exam_level"]
    data = {name: frame if level == "image" else aggregate_views(frame, "study") for name, frame in frames.items()}

    width, left, right, panel = DOUBLE_COLUMN, 10.7 * MM, 3.4 * MM, 48.5 * MM
    gap = (width - left - right - 3 * panel) / 2
    top, below = 7.1 * MM, 25.0 * MM
    height = top + panel + below
    figure = plt.figure(figsize=(width, height))
    panel_pt = panel * 72

    for index, (head, title) in enumerate(HEADS):
        axis = figure.add_axes([(left + index * (panel + gap)) / width, below / height, panel / width, panel / height])
        inset = axis.inset_axes(list(INSET))
        rows = []
        for order, (name, _, color) in enumerate(MODELS):
            target = data[name][f"target_{head}"].to_numpy()
            score = data[name][f"prob_{head}"].to_numpy()
            fpr, tpr, _ = roc_curve(target, score)
            cell = table[name][head]
            if abs(safe_auc(target, score) - cell["auc"]) > 1e-9:
                raise RuntimeError(f"{level}/{name}/{head}: AUC differs from results/summary.json")
            if np.any((fpr > CARD[0]) & (tpr < CARD[1] + CARD[3])):
                raise RuntimeError(f"{level}/{name}/{head}: the ROC curve passes behind the inset card")
            style = {"color": color, "lw": line_width(name), "solid_joinstyle": "miter", "zorder": 3 + order}
            axis.plot(fpr, tpr, clip_on=False, solid_capstyle="butt", **style)
            inset.plot(fpr, tpr, **style)
            value = f"{cell['auc']:.3f} ({cell['ci95'][0]:.3f}–{cell['ci95'][1]:.3f})"  # as in Table 2
            rows.append((name, LEGEND_LABEL[name], color, value))

        pause_start, pause_end = CARD[0] - 0.02, CARD[1] + CARD[3] + 0.02
        for start, end in ((0.0, pause_start), (pause_end, 1.0)):
            axis.plot([start, end], [start, end], zorder=1, dash_capstyle="butt", **REF_LINE)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal", adjustable="box")
        for side in ("left", "bottom"):
            axis.spines[side].set_position(("outward", 3))
        axis.xaxis.set_major_locator(MultipleLocator(0.2))
        axis.yaxis.set_major_locator(MultipleLocator(0.2))
        axis.xaxis.set_major_formatter(decimal_formatter(1))
        axis.yaxis.set_major_formatter(decimal_formatter(1))
        axis.set_xlabel("1 − Specificity")
        if index == 0:
            axis.set_ylabel("Sensitivity")

        x_min, x_max, y_min, y_max = ZOOM[level][head]
        inset.set_xlim(x_min, x_max)
        inset.set_ylim(y_min, y_max)
        inset.set_facecolor("white")
        inset.set_zorder(4)
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color(GREY_FRAME)
        step_x = 0.05 if x_max - x_min <= 0.2 else 0.1
        inset.xaxis.set_major_locator(MultipleLocator(step_x))
        inset.yaxis.set_major_locator(MultipleLocator(0.1))
        inset.xaxis.set_major_formatter(decimal_formatter(2 if step_x < 0.1 else 1))
        inset.yaxis.set_major_formatter(decimal_formatter(1))
        inset.tick_params(labelsize=SMALL, length=1.5, width=0.5, pad=1.2, color=GREY_FRAME)
        # Outline the zoomed window on the main axes, below the curves; connector lines are omitted.
        zoom = axis.indicate_inset_zoom(inset, edgecolor=GREY_FRAME, linewidth=0.5, alpha=1.0)
        rectangle = getattr(zoom, "rectangle", None) or (zoom[0] if isinstance(zoom, tuple) else None)
        if rectangle is not None:
            rectangle.set_zorder(2.2)
        for connector in getattr(zoom, "connectors", None) or (zoom[1] if isinstance(zoom, tuple) else []):
            connector.set_visible(False)

        panel_label(axis, "abc"[index], title)
        auc_table(axis, rows, panel_pt)
    return figure


# -----------------------------------------------------------------------------
# Forest plot of paired AUC differences
# -----------------------------------------------------------------------------


def _signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")  # 3 decimals, as in Table S1


CI_OFFSET_PT = 22.5  # width of "+0.000* " at 6 pt Arial: start of the interval column after the estimate


def forest_figure(summary: dict) -> plt.Figure:
    """Paired differences in image-level AUC with 95% CIs, grouped by task."""
    pairs = summary["vindr"]["paired"]
    entries: list[tuple[float, dict, dict]] = []
    headers: list[tuple[float, str]] = []
    y = 0.0
    for head, title in HEADS:
        headers.append((y, title))
        y += 1.0
        for row in pairs:
            entries.append((y, row, row[head]))
            y += 1.0
        y += 0.6
    last_row = y - 1.6

    width, margin, label_w, value_w = ONE_HALF_COLUMN, 2.0 * MM, 36.0 * MM, 28.5 * MM
    row_h, top, bottom = 4.0 * MM, 1.0 * MM, 13.5 * MM
    height = top + (last_row + 1.3) * row_h + bottom
    figure = plt.figure(figsize=(width, height))
    axis = figure.add_axes([label_w / width, bottom / height, (width - label_w - value_w) / width,
                            (height - top - bottom) / height])

    lows = [cell["ci95"][0] for _, _, cell in entries]
    highs = [cell["ci95"][1] for _, _, cell in entries]
    step = 0.005
    x_min = float(np.floor((min(lows) - 0.002) / step) * step)
    x_max = float(np.ceil((max(highs) + 0.002) / step) * step)
    axis.plot([0, 0], [0.45, last_row + 0.8], zorder=1, clip_on=False, **REF_LINE)
    row_text = {"textcoords": "offset points", "va": "center_baseline", "fontsize": BODY}
    for y_pos, row, cell in entries:
        candidate, reference = row["candidate"], row["reference"]
        color = COLOR[candidate]
        low, high = cell["ci95"]
        significant = cell["significant"]
        axis.plot([low, high], [y_pos, y_pos], color=color, lw=1.0, solid_capstyle="butt", zorder=2)
        for end_point in (low, high):
            axis.plot([end_point, end_point], [y_pos - 0.17, y_pos + 0.17], color=color, lw=1.0,
                      solid_capstyle="butt", zorder=2)
        axis.plot(cell["delta_auc"], y_pos, marker="s", ms=4.2, mew=0.9, mec=color,
                  mfc=color if significant else "white", zorder=3)
        axis.annotate(f"{LABEL[candidate]} vs {LABEL[reference]}", xy=((margin + 6 / 72) / width, y_pos),
                      xycoords=("figure fraction", "data"), xytext=(0, 0), ha="left", **row_text)
        # Estimate with its asterisk, as in Table S1; the intervals start in one column whether starred or not.
        mark = "*" if significant else ""
        axis.annotate(f"{_signed(cell['delta_auc'])}{mark}", xy=(1, y_pos), xycoords=("axes fraction", "data"),
                      xytext=(8, 0), ha="left", **row_text)
        axis.annotate(f"({_signed(low)}, {_signed(high)})", xy=(1, y_pos), xycoords=("axes fraction", "data"),
                      xytext=(8 + CI_OFFSET_PT, 0), ha="left", **row_text)
    for y_pos, title in headers:
        axis.annotate(title, xy=(margin / width, y_pos), xycoords=("figure fraction", "data"), xytext=(0, -2.4),
                      textcoords="offset points", ha="left", va="baseline", fontsize=TITLE, fontweight="bold")
    axis.annotate("ΔAUC (95% CI)", xy=(1, headers[0][0]), xycoords=("axes fraction", "data"), xytext=(8, -2.4),
                  textcoords="offset points", ha="left", va="baseline", fontsize=BODY, color=GREY_TEXT)

    axis.set_ylim(last_row + 0.8, -0.5)
    axis.set_xlim(x_min, x_max)
    axis.set_yticks([])
    axis.spines["left"].set_visible(False)
    axis.xaxis.set_major_locator(MultipleLocator(0.01))
    axis.xaxis.set_minor_locator(MultipleLocator(0.005))
    axis.tick_params(axis="x", which="minor", length=1.2, width=0.5)
    axis.xaxis.set_major_formatter(decimal_formatter(2))
    axis.set_xlabel("Difference in image-level AUC (model − Baseline)")
    key = [
        Line2D([], [], ls="none", marker="s", ms=4.2, mfc=GREY_COMPARE, mec=GREY_COMPARE, mew=0.9,
               label="95% CI excludes 0 (*)"),
        Line2D([], [], ls="none", marker="s", ms=4.2, mfc="white", mec=GREY_COMPARE, mew=0.9,
               label="95% CI includes 0"),
    ]
    figure.legend(handles=key, loc="lower left", bbox_to_anchor=((margin + 6 / 72) / width, 2.2 * MM / height),
                  ncol=2, borderpad=0, handlelength=0.8, handletextpad=0.6, columnspacing=1.6, fontsize=BODY)
    return figure


# -----------------------------------------------------------------------------
# Confusion matrices
# -----------------------------------------------------------------------------


def _relative_luminance(rgb) -> float:
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def confusion_figure(summary: dict) -> plt.Figure:
    """Confusion matrices at the validation-selected operating points (rows: tasks, columns: models)."""
    points = summary["vindr"]["operating_points"]
    cmap = LinearSegmentedColormap.from_list("blues", ["#F4F8FC", "#9CC3E0", "#2E75B6", "#0B3C6E"])
    width = ONE_HALF_COLUMN
    cell_size, gap_x, gap_y = 0.92, 0.18, 0.26
    left, top, bottom = 0.52, 0.22, 0.30
    right = width - left - 4 * cell_size - 3 * gap_x
    height = top + 3 * cell_size + 2 * gap_y + bottom
    figure = plt.figure(figsize=(width, height))
    mesh = None
    for row_index, (head, title) in enumerate(HEADS):
        for col_index, (name, _, color) in enumerate(MODELS):
            x0 = left + col_index * (cell_size + gap_x)
            y0 = bottom + (2 - row_index) * (cell_size + gap_y)
            axis = figure.add_axes([x0 / width, y0 / height, cell_size / width, cell_size / height])
            counts = points[name][head]["confusion"]
            matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=float)
            proportion = matrix / matrix.sum(axis=1, keepdims=True)
            mesh = axis.pcolormesh(proportion, cmap=cmap, vmin=0.0, vmax=1.0, edgecolors="face", linewidth=0)
            axis.set_xlim(0, 2)
            axis.set_ylim(2, 0)
            axis.set_aspect("equal")
            for i in range(2):
                for j in range(2):
                    # cut-off in the gap between 56.8% and 60.3%: similar tones never get different text colours
                    dark = _relative_luminance(cmap(proportion[i, j])[:3]) < 0.233
                    ink = "white" if dark else INK
                    axis.annotate(f"{int(matrix[i, j]):,}", xy=(j + 0.5, i + 0.5), xytext=(0, 4.4),
                                  textcoords="offset points", ha="center", va="center", fontsize=TITLE,
                                  fontweight="bold", color=ink)
                    axis.annotate(f"{proportion[i, j]:.1%}", xy=(j + 0.5, i + 0.5), xytext=(0, -5.4),
                                  textcoords="offset points", ha="center", va="center", fontsize=BODY, color=ink)
            axis.axhline(1, color="white", lw=1.0)
            axis.axvline(1, color="white", lw=1.0)
            axis.set_xticks([0.5, 1.5])
            axis.set_yticks([0.5, 1.5])
            axis.set_xticklabels(["Negative", "Positive"] if row_index == 2 else [])
            axis.set_yticklabels(["Negative", "Positive"] if col_index == 0 else [], rotation=90, va="center")
            axis.tick_params(length=0, pad=2, labelsize=BODY)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color(GREY_FRAME)
            if row_index == 2:
                axis.set_xlabel("Predicted", labelpad=2)
            if col_index == 0:
                axis.set_ylabel("Actual", labelpad=2)
                axis.annotate(title, xy=(0, 0.5), xycoords="axes fraction", xytext=(-24, 0),
                              textcoords="offset points", ha="right", va="center", rotation=90, fontsize=TITLE,
                              fontweight="bold")
                # Letter top-aligned with its row of matrices, as in the map grids (heatmap_evidence_maps).
                figure.text(1.9 * MM / width, (y0 + cell_size) / height, "abc"[row_index], ha="left", va="top",
                            fontsize=LETTER, fontweight="bold")
            if row_index == 0:
                # Name only: "(ours)" would not fit on one line above the matrix.
                axis.set_title(LABEL[name], fontsize=TITLE, color=color, pad=4,
                               fontweight="bold" if name == "LSA-SSViT" else "normal")
    cax = figure.add_axes([(width - right + 0.12) / width, bottom / height, 0.08 / width,
                           (3 * cell_size + 2 * gap_y) / height])
    bar = figure.colorbar(mesh, cax=cax)
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}"))
    bar.set_label("Percentage of the actual class", fontsize=TITLE, labelpad=3)
    bar.ax.tick_params(labelsize=BODY, length=2.0, width=0.5, color=GREY_FRAME, pad=2)
    bar.outline.set_linewidth(0.5)
    bar.outline.set_edgecolor(GREY_FRAME)
    bar.solids.set_edgecolor("face")
    bar.solids.set_rasterized(False)
    return figure


def main() -> None:
    args = parse_args()
    setup_style()
    summary = load_summary(args.results_root)
    frames = load_predictions(args.results_root)
    print(f"Writing figures to {args.output_dir}")
    save(roc_figure(frames, summary, "image"), args.output_dir, "roc_image_level")
    save(forest_figure(summary), args.output_dir, "auc_difference_forest")
    save(confusion_figure(summary), args.output_dir, "confusion_matrices")
    save(roc_figure(frames, summary, "exam"), args.output_dir, "roc_exam_level")


if __name__ == "__main__":
    main()
