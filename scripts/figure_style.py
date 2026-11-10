"""Shared figure style of the paper: one type scale, one set of colours and line weights, panel letters and export.

Imported by scripts/plot_figures.py and scripts/plot_heatmaps.py so that every figure follows the same specification:
Arial; panel letters 8 pt bold; titles and axis labels 7 pt; column titles and row labels of the map grids 6.5 pt;
ticks, legends and annotations 6 pt; inset ticks and in-image labels 5.5 pt; figures 180 mm (double column) or 136 mm
(one-and-a-half column) wide. Explanatory text (what a panel shows, error bars, abbreviations) belongs in the figure
legends, not in the images.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

MM = 1 / 25.4  # inches per millimetre
DOUBLE_COLUMN = 180 * MM
ONE_HALF_COLUMN = 136 * MM

# Okabe-Ito colours (colour-blind safe); LSA-SSViT, the proposed model (the Baseline with GRAIN and CUSP), is
# drawn slightly heavier and on top.
MODELS = [
    ("baseline", "Baseline", "#000000"),
    ("grain", "+GRAIN", "#0072B2"),
    ("cusp", "+CUSP", "#D55E00"),
    ("LSA-SSViT", "LSA-SSViT", "#009E73"),
]
LABEL = {name: label for name, label, _ in MODELS}
# Model legends and keys mark the proposed model with "(ours)"; column titles above image or matrix grids, "A vs B"
# comparisons and in-panel labels use the bold or coloured name only.
LEGEND_LABEL = {name: f"{label} (ours)" if name == "LSA-SSViT" else label for name, label, _ in MODELS}
COLOR = {name: color for name, _, color in MODELS}
# Names of the comparison networks of comparators/zoo.py, in the order of the comparison table (the heat-map figures
# show those of BLOCKS in scripts/plot_heatmaps.py). Their column titles are black: colours identify the four models of
# MODELS, here as in every other figure.
COMPARATOR_MODELS = [
    ("resnet50", "ResNet-50"),
    ("efficientnet_b3", "EfficientNet-B3"),
    ("convnext_tiny", "ConvNeXt-T"),
    ("mobilenetv4_conv_medium", "MobileNetV4-Conv-M"),
    ("rope_vit_base", "RoPE-ViT-B/16"),
    ("mambavision_tiny", "MambaVision-T"),
    ("dinov2_base", "DINOv2-B/14"),
    ("rad_dino", "RAD-DINO"),
    ("mammo_clip_b2", "Mammo-CLIP"),
]
COMPARATOR_LABEL = dict(COMPARATOR_MODELS)
HEADS = [("birads", "Screening"), ("mass", "Mass"), ("calcification", "Calcification")]
HEAD_TITLE = dict(HEADS)

LETTER, TITLE, BODY, SMALL = 8.0, 7.0, 6.0, 5.5  # font sizes (pt)
GRID_TITLE = 6.5  # column titles and row labels of the map grids (pt)
INK = "#222222"  # text, axes, error bars
GREY_TEXT = "#555555"  # legend titles, column headers of tables, sub-row labels
GREY_FRAME = "#6E6E6E"  # inset frames, zoom outlines, matrix and colour-bar outlines
GREY_COMPARE = "#4D4D4D"  # neutral markers of the forest-plot key
REF_LINE = {"color": "#8C8C8C", "lw": 0.6, "ls": (0, (3, 2))}  # chance, zero and diagonal reference lines
MODEL_LW, PROPOSED_LW = 0.8, 1.0


def line_width(name: str) -> float:
    return PROPOSED_LW if name == "LSA-SSViT" else MODEL_LW


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": BODY,
        "axes.titlesize": TITLE,
        "axes.labelsize": TITLE,
        "xtick.labelsize": BODY,
        "ytick.labelsize": BODY,
        "legend.fontsize": BODY,
        "legend.title_fontsize": BODY,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.4,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.0,
        "axes.linewidth": 0.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.labelpad": 3.0,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "lines.markeredgewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    _exact_mathtext_boxes()


def _exact_mathtext_boxes() -> None:
    """Keep mathtext ($n$, $\\rho$) on the baseline of the surrounding text in PDF output.

    Matplotlib rounds the height and depth of mathtext boxes up to whole points for vector output, which lifts
    lines containing mathtext by up to 1 pt in the PDF (the PNG is unaffected). Skipped silently if the private API
    differs in the installed matplotlib version.
    """
    try:
        from matplotlib import _mathtext

        def to_vector(self):
            w, h, d = self.box.width, self.box.height, self.box.depth
            glyphs = [(info.font, info.fontsize, info.num, ox, h - oy + info.offset) for ox, oy, info in self.glyphs]
            rects = [(x1, h - y2, x2 - x1, y2 - y1) for x1, y1, x2, y2 in self.rects]
            return _mathtext.VectorParse(w, h + d, d, glyphs, rects)

        if hasattr(_mathtext, "Output") and hasattr(_mathtext, "VectorParse"):
            _mathtext.Output.to_vector = to_vector
    except Exception:  # noqa: BLE001 - purely cosmetic
        pass


def decimal_formatter(digits: int) -> FuncFormatter:
    """Tick labels with a fixed number of decimals and a typographic minus sign."""
    return FuncFormatter(lambda value, _: f"{value:.{digits}f}".replace("-", "−"))


def panel_label(axis: plt.Axes, letter: str, title: str = "") -> None:
    """Bold panel letter and regular-weight title above the axes.

    The letter is placed at export time at the left edge of the panel's content (tick and axis labels included), so
    that letters line up with the panel and sit at the same distance from the figure edge in every figure.
    """
    figure = axis.figure
    if not hasattr(figure, "_panel_labels"):
        figure._panel_labels = []
    figure._panel_labels.append((axis, letter, title))


def _place_panel_labels(figure: plt.Figure) -> None:
    requests = getattr(figure, "_panel_labels", [])
    if not requests:
        return
    renderer = figure.canvas.get_renderer()
    width_px, height_px = figure.bbox.width, figure.bbox.height
    for axis, letter, title in requests:
        content = axis.get_tightbbox(renderer)
        top = axis.get_window_extent(renderer).y1
        x = content.x0 / width_px
        y = (top + 2.6 * MM * figure.dpi) / height_px
        mark = figure.text(x, y, letter, ha="left", va="baseline", fontsize=LETTER, fontweight="bold")
        if title:
            letter_width = mark.get_window_extent(renderer).width
            figure.text(x + (letter_width + 1.2 * MM * figure.dpi) / width_px, y, title, ha="left", va="baseline",
                        fontsize=TITLE)
    figure._panel_labels = []


def save(figure: plt.Figure, output_dir: Path, stem: str, image_dpi: int = 600) -> None:
    """Vector PDF (raster images inside it at image_dpi) and a 600-dpi RGB PNG, at the exact figure size."""
    from PIL import Image

    _place_panel_labels(figure)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.pdf", dpi=image_dpi)
    png = output_dir / f"{stem}.png"
    figure.savefig(png, dpi=600)
    plt.close(figure)
    with Image.open(png) as image:
        rgb = image.convert("RGB")
    rgb.save(png, dpi=(600, 600), optimize=True)
    print(f"  {output_dir / stem}.pdf / .png")
