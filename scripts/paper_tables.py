#!/usr/bin/env python
"""Reproduce every table of the paper from the released predictions.

Inputs (all shipped with the repository):
    splits/vindr/{train,val,eval,test}.csv
    results/vindr/<model>/predictions_test.csv   (written by evaluate.py)
    results/vindr/<model>/metrics_test.json      (validation-selected thresholds)
    results/inbreast/<model>/predictions.csv     (optional, written by scripts/evaluate_inbreast.py)

Outputs:
    results/summary.json     every number, consumed by scripts/plot_figures.py
    results/paper_tables.md  the tables as Markdown

Confidence intervals come from a non-parametric bootstrap that resamples exams
(patients on INbreast) with replacement, 2,000 replicates, seed 2026. Model
differences use a paired bootstrap over the same resamples. No GPU is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import (  # noqa: E402
    HEAD_TITLES,
    aggregate_views,
    bootstrap_auc_ci,
    paired_bootstrap_auc_difference,
    parse_birads_ge3,
    safe_ap,
    safe_auc,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)

MODELS = [
    ("baseline", "Baseline"),
    ("grain", "+GRAIN"),
    ("cusp", "+CUSP"),
    ("LSA-SSViT", "LSA-SSViT"),
]
PAIRS = [  # (candidate, reference)
    ("grain", "baseline"),
    ("cusp", "baseline"),
    ("LSA-SSViT", "baseline"),
]
HEADS = ["birads", "mass", "calcification"]  # column order of the paper tables
LABEL = dict(MODELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the tables of the paper from released predictions.")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "splits" / "vindr")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_vindr(results_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    predictions, thresholds = {}, {}
    for name, _ in MODELS:
        folder = results_root / "vindr" / name
        frame = pd.read_csv(folder / "predictions_test.csv", dtype={"study_id": str, "image_id": str})
        metrics = json.loads((folder / "metrics_test.json").read_text(encoding="utf-8"))
        predictions[name] = frame
        thresholds[name] = metrics["thresholds"]
    reference = predictions["baseline"]
    for name, frame in predictions.items():
        if not frame["image_id"].equals(reference["image_id"]):
            raise RuntimeError(f"{name}: images are not in the same order as the baseline predictions")
        for head in HEADS:
            if not frame[f"target_{head}"].equals(reference[f"target_{head}"]):
                raise RuntimeError(f"{name}: labels differ from the baseline predictions ({head})")
    return predictions, thresholds


def load_inbreast(results_root: Path) -> dict[str, pd.DataFrame] | None:
    frames = {}
    for name, _ in MODELS:
        path = results_root / "inbreast" / name / "predictions.csv"
        if not path.is_file():
            return None
        frames[name] = pd.read_csv(path, dtype={"study_id": str, "image_id": str})
    return frames


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------


def dataset_table(split_root: Path) -> dict:
    table = {}
    for split in ("train", "val", "eval", "test"):
        frame = pd.read_csv(split_root / f"{split}.csv", dtype={"study_id": str})
        screening = frame["breast_birads"].map(parse_birads_ge3)
        category = frame["breast_birads"].astype(str).str.extract(r"(\d)")[0].astype(int)
        table[split] = {
            "exams": int(frame["study_id"].nunique()),
            "images": int(len(frame)),
            "positive_birads": int(screening.sum()),
            "positive_mass": int(frame["mass"].sum()),
            "positive_calcification": int(frame["calcification"].sum()),
            "birads_3_4_5": [int((category == level).sum()) for level in (3, 4, 5)],
        }
    return table


def auc_table(frames: dict[str, pd.DataFrame], n_boot: int, seed: int, level: str) -> dict:
    """AUC with 95% CI (and AP) per model and head, at image or exam level."""
    table: dict[str, dict] = {}
    for name, frame in frames.items():
        data = frame if level == "image" else aggregate_views(frame, "study")
        groups = data["study_id"].to_numpy()
        table[name] = {}
        for head in HEADS:
            target = data[f"target_{head}"].to_numpy()
            score = data[f"prob_{head}"].to_numpy()
            low, high = bootstrap_auc_ci(target, score, groups, n_boot, seed)
            table[name][head] = {
                "auc": safe_auc(target, score),
                "ci95": [low, high],
                "ap": safe_ap(target, score),
                "n_positive": int(target.sum()),
                "n": int(len(target)),
            }
    return table


def pair_table(frames: dict[str, pd.DataFrame], pairs: list[tuple[str, str]], heads: list[str],
               n_boot: int, seed: int) -> list[dict]:
    rows = []
    for candidate, reference in pairs:
        merged = frames[reference].merge(frames[candidate], on="image_id", suffixes=("_ref", "_new"))
        groups = merged["study_id_ref"].to_numpy()
        row = {"candidate": candidate, "reference": reference}
        for head in heads:
            point, low, high = paired_bootstrap_auc_difference(
                merged[f"target_{head}_ref"].to_numpy(),
                merged[f"prob_{head}_ref"].to_numpy(),
                merged[f"prob_{head}_new"].to_numpy(),
                groups,
                n_boot,
                seed,
            )
            row[head] = {"delta_auc": point, "ci95": [low, high], "significant": bool(low > 0 or high < 0)}
        rows.append(row)
    return rows


def operating_point_table(frames: dict[str, pd.DataFrame], thresholds: dict[str, dict]) -> dict:
    """Sensitivity/specificity at the validation-selected threshold and at 90% operating points."""
    table: dict[str, dict] = {}
    for name, frame in frames.items():
        table[name] = {}
        for head in HEADS:
            target = frame[f"target_{head}"].to_numpy().astype(int)
            score = frame[f"prob_{head}"].to_numpy()
            threshold = float(thresholds[name][head])
            predicted = (score >= threshold).astype(int)
            if not np.array_equal(predicted, frame[f"pred_{head}"].to_numpy()):
                raise RuntimeError(f"{name}/{head}: pred column does not match the stored threshold")
            tp = int(((predicted == 1) & (target == 1)).sum())
            fn = int(((predicted == 0) & (target == 1)).sum())
            tn = int(((predicted == 0) & (target == 0)).sum())
            fp = int(((predicted == 1) & (target == 0)).sum())
            table[name][head] = {
                "threshold": threshold,
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "sensitivity_at_spec90": sensitivity_at_specificity(target, score, 0.90),
                "specificity_at_sens90": specificity_at_sensitivity(target, score, 0.90),
                "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            }
    return table


# -----------------------------------------------------------------------------
# Markdown
# -----------------------------------------------------------------------------


def _auc(cell: dict) -> str:
    return f"{cell['auc']:.4f} ({cell['ci95'][0]:.4f}–{cell['ci95'][1]:.4f})"


def _signed(value: float) -> str:
    return f"{value:+.4f}".replace("-", "−")


def markdown(summary: dict) -> str:
    lines: list[str] = []
    data = summary["dataset"]
    lines += [
        "## Table 1. Datasets (VinDr-Mammo)",
        "",
        "| Split | Exams | Images | Screening + | Mass + | Calcification + |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    split_names = {"train": "Training", "val": "Validation", "eval": "Held-out", "test": "Official test"}
    for split, row in data.items():
        lines.append(
            f"| {split_names.get(split, split)} | {row['exams']:,} | {row['images']:,} | {row['positive_birads']:,} | "
            f"{row['positive_mass']:,} | {row['positive_calcification']:,} |"
        )
    test_levels = data["test"]["birads_3_4_5"]
    lines += ["", f"Screening-positive test images by BI-RADS 3/4/5: {test_levels[0]}/{test_levels[1]}/{test_levels[2]}.", ""]

    image = summary["vindr"]["image_level"]
    lines += [
        "## Table 2. Image-level results on the official VinDr-Mammo test set (AUC with 95% CI)",
        "",
        "| Model | Screening AUC | Screening AP | Mass AUC | Mass AP | Calcification AUC | Calcification AP |",
        "| --- | --- | ---: | --- | ---: | --- | ---: |",
    ]
    for name, label in MODELS:
        cells = [f"{_auc(image[name][head])} | {image[name][head]['ap']:.3f}" for head in HEADS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Table 3. Paired differences in image-level AUC (* 95% CI excludes 0)",
        "",
        "| Comparison | Screening ΔAUC | Mass ΔAUC | Calcification ΔAUC |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary["vindr"]["paired"]:
        cells = []
        for head in HEADS:
            cell = row[head]
            text = f"{_signed(cell['delta_auc'])} ({_signed(cell['ci95'][0])}, {_signed(cell['ci95'][1])})"
            cells.append(text + ("\\*" if cell["significant"] else ""))
        lines.append(f"| {LABEL[row['candidate']]} vs {LABEL[row['reference']]} | " + " | ".join(cells) + " |")

    points = summary["vindr"]["operating_points"]
    lines += [
        "",
        "## Table 4. Operating points (thresholds selected on the validation set with Youden's index)",
        "",
        "| Task | Model | Sensitivity / specificity | Sensitivity at 90% specificity | Specificity at 90% sensitivity |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for head in HEADS:
        for name, label in MODELS:
            cell = points[name][head]
            lines.append(
                f"| {HEAD_TITLES[head]} | {label} | {cell['sensitivity']:.1%} / {cell['specificity']:.1%} | "
                f"{cell['sensitivity_at_spec90']:.1%} | {cell['specificity_at_sens90']:.1%} |"
            )

    exam = summary["vindr"]["exam_level"]
    counts = [exam["baseline"][head]["n_positive"] for head in HEADS]
    lines += [
        "",
        "## Table 5. Exam-level AUC (maximum score over the views of an exam; 95% CI)",
        "",
        f"Positive exams: screening {counts[0]}, mass {counts[1]}, calcification {counts[2]}.",
        "",
        "| Model | Screening AUC | Mass AUC | Calcification AUC |",
        "| --- | --- | --- | --- |",
    ]
    for name, label in MODELS:
        lines.append(f"| {label} | " + " | ".join(_auc(exam[name][head]) for head in HEADS) + " |")

    inbreast = summary.get("inbreast")
    lines += ["", "## Table 6. External test on INbreast without fine-tuning", ""]
    if inbreast is None:
        lines.append("Not available: run scripts/evaluate_inbreast.py first (INbreast images are not redistributed).")
    else:
        lines += [
            "| Model | Screening AUC (95% CI) | Screening AP | Mass AUC | Mass AP | Calcification AUC | Calcification AP |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, label in MODELS:
            cell = inbreast["image_level"][name]
            lines.append(
                f"| {label} | {_auc(cell['birads'])} | {cell['birads']['ap']:.3f} | "
                f"{cell['mass']['auc']:.4f} | {cell['mass']['ap']:.3f} | "
                f"{cell['calcification']['auc']:.4f} | {cell['calcification']['ap']:.3f} |"
            )
        lines += ["", "Paired difference in screening AUC versus the baseline (patients resampled):", ""]
        for row in inbreast["paired"]:
            cell = row["birads"]
            lines.append(
                f"- {LABEL[row['candidate']]}: {_signed(cell['delta_auc'])} "
                f"({_signed(cell['ci95'][0])}, {_signed(cell['ci95'][1])})" + (" *" if cell["significant"] else "")
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    predictions, thresholds = load_vindr(args.results_root)
    print(f"Bootstrap: {args.n_boot} replicates over exams, seed {args.seed}. This takes a few minutes.")
    summary: dict = {
        "protocol": {
            "bootstrap": "percentile 95% CI; exams (INbreast: patients) resampled with replacement",
            "n_boot": args.n_boot,
            "seed": args.seed,
            "thresholds": "selected on the validation set with Youden's index",
            "exam_level": "maximum score over the views of an exam",
        },
        "models": dict(MODELS),
        "dataset": dataset_table(args.split_root),
        "vindr": {},
    }
    summary["vindr"]["image_level"] = auc_table(predictions, args.n_boot, args.seed, "image")
    print(f"  image-level CIs done ({time.perf_counter() - started:.0f}s)")
    summary["vindr"]["paired"] = pair_table(predictions, PAIRS, HEADS, args.n_boot, args.seed)
    print(f"  paired differences done ({time.perf_counter() - started:.0f}s)")
    summary["vindr"]["exam_level"] = auc_table(predictions, args.n_boot, args.seed, "exam")
    summary["vindr"]["operating_points"] = operating_point_table(predictions, thresholds)

    inbreast = load_inbreast(args.results_root)
    if inbreast is not None:
        summary["inbreast"] = {
            "image_level": auc_table(inbreast, args.n_boot, args.seed, "image"),
            "paired": pair_table(inbreast, [(name, "baseline") for name, _ in MODELS[1:]], ["birads"],
                                 args.n_boot, args.seed),
        }

    json_path = args.results_root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = markdown(summary)
    (args.results_root / "paper_tables.md").write_text(text, encoding="utf-8", newline="\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the tables contain non-ASCII signs (−, Δ)
    print(text)
    print(f"Wrote {json_path} and {args.results_root / 'paper_tables.md'} ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
