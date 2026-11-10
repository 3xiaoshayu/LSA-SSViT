#!/usr/bin/env python
"""Table of the whole-model comparison from the comparator predictions.

    python comparators/summarize.py --runs-root results/comparators

Reads <runs-root>/<model>/predictions_test.csv for every comparator of comparators/zoo.py
that is present, and the released test predictions of the Baseline and of LSA-SSViT
(results/vindr). Writes <runs-root>/summary.json and <runs-root>/comparison_tables.md with
the image-level AUC (95% CI) and AP of every head on the official VinDr-Mammo test split,
as in Table 2, and the paired AUC difference LSA-SSViT minus each comparator.
Bootstrap: 2,000 replicates over exams, seed 2026. No GPU is needed.

Each comparator's run.json must show a completed run with the protocol of train.py (training
ended by early stopping or the epoch limit, the default settings, the VinDr-Mammo split);
other runs are refused unless --allow-incomplete is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "scripts"))

from comparators.zoo import COMPARATORS, HEAD_LR  # noqa: E402
from metrics import HEAD_TITLES  # noqa: E402
from paper_tables import HEADS, auc_table, pair_table  # noqa: E402

PROPOSED = "LSA-SSViT"
REFERENCE_ROWS = {"baseline": ("Baseline (DINOv3-B/16)", "Ours, without GRAIN and CUSP"),
                  PROPOSED: ("LSA-SSViT", "Ours")}
DATASETS = {
    "vindr": ("VinDr-Mammo official test set", "predictions_test.csv", "vindr/{name}/predictions_test.csv", "exams"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tables of the whole-model comparison.")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "results" / "comparators",
                        help="one folder per comparator with its prediction files")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results",
                        help="released predictions of the Baseline and of LSA-SSViT")
    parser.add_argument("--models", nargs="+", default=None, help="comparators to include (default: all present)")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="also accept runs that did not follow the full protocol (tests only)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


PROTOCOL = {"epochs": 20, "patience": 6, "batch_size": 8, "lr": HEAD_LR, "weight_decay": 1e-4, "dropout": 0.25,
            "pos_weight_cap": 20.0, "amp_init_scale": 2.0, "seed": 2026}


def check_run(path: Path, name: str) -> dict:
    """Training record of a comparator; lists every departure from the protocol."""
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config = record.get("config", {})
    problems = []
    if record.get("status") != "completed" or not record.get("training_finished"):
        problems.append(f"status {record.get('status')!r}, stop reason {record.get('stop_reason')!r}")
    expected = {**PROTOCOL, "backbone_lr": COMPARATORS[name].backbone_lr}
    problems += [f"{key}={config.get(key)!r} (protocol {value!r})" for key, value in expected.items()
                 if config.get(key) != value]
    if not str(config.get("split_root", "")).replace("\\", "/").rstrip("/").endswith("splits/vindr"):
        problems.append(f"split_root={config.get('split_root')!r}")
    return {"stop_reason": record.get("stop_reason"), "best_epoch": record.get("best_epoch"), "problems": problems}


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"study_id": str, "image_id": str})


def load_dataset(args: argparse.Namespace, dataset: str, models: list[str]) -> dict[str, pd.DataFrame]:
    _, comparator_file, release_pattern, _ = DATASETS[dataset]
    frames = {name: read_predictions(args.results_root / release_pattern.format(name=name)) for name in REFERENCE_ROWS}
    for name in models:
        path = args.runs_root / name / comparator_file
        if path.is_file():
            frames[name] = read_predictions(path)
    reference = frames[PROPOSED].set_index("image_id").sort_index()
    for name, frame in frames.items():
        indexed = frame.set_index("image_id").sort_index()
        if not indexed.index.equals(reference.index):
            raise RuntimeError(f"{dataset}/{name}: the images differ from the LSA-SSViT predictions")
        for head in HEADS:
            if not indexed[f"target_{head}"].equals(reference[f"target_{head}"]):
                raise RuntimeError(f"{dataset}/{name}: the {head} labels differ from the LSA-SSViT predictions")
    return frames


def _auc(cell: dict) -> str:
    return f"{cell['auc']:.4f} ({cell['ci95'][0]:.4f}–{cell['ci95'][1]:.4f})"


def _signed(value: float) -> str:
    return f"{value:+.4f}".replace("-", "−")


def markdown(summary: dict) -> str:
    lines: list[str] = []
    titles = summary["models"]
    for index, (dataset, block) in enumerate(summary["datasets"].items(), start=1):
        lines += [
            f"## Table A{index}. Whole-model comparison: {block['title']} (image-level AUC with 95% CI)",
            "",
            "| Model | Family | Screening AUC | Screening AP | Mass AUC | Mass AP | Calcification AUC | Calcification AP |",
            "| --- | --- | --- | ---: | --- | ---: | --- | ---: |",
        ]
        for name, cells in block["image_level"].items():
            row = " | ".join(f"{_auc(cells[head])} | {cells[head]['ap']:.3f}" for head in HEADS)
            lines.append(f"| {titles[name]['title']} | {titles[name]['family']} | {row} |")
        lines += [
            "",
            f"Paired AUC difference, LSA-SSViT minus the comparator ({block['unit']} resampled; * 95% CI excludes 0):",
            "",
            "| Comparator | " + " | ".join(f"{HEAD_TITLES[head]} ΔAUC" for head in HEADS) + " |",
            "| --- | --- | --- | --- |",
        ]
        for row in block["paired_vs_LSA-SSViT"]:
            cells = []
            for head in HEADS:
                cell = row[head]
                text = f"{_signed(cell['delta_auc'])} ({_signed(cell['ci95'][0])}, {_signed(cell['ci95'][1])})"
                cells.append(text + ("\\*" if cell["significant"] else ""))
            lines.append(f"| {titles[row['reference']]['title']} | " + " | ".join(cells) + " |")
        lines.append("")
    if summary["training"]:
        lines += ["Training of the comparators (checkpoint with the best validation screening AUC):", ""]
        for name, record in summary["training"].items():
            note = f"; departures: {'; '.join(record['problems'])}" if record["problems"] else ""
            lines.append(f"- {titles[name]['title']}: best epoch {record['best_epoch']}, {record['stop_reason']}{note}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    models = args.models or [name for name in COMPARATORS if (args.runs_root / name).is_dir()]
    unknown = sorted(set(models) - set(COMPARATORS))
    if unknown:
        raise SystemExit(f"Unknown comparators: {unknown}")
    training = {name: check_run(args.runs_root / name / "run.json", name) for name in models}
    for name, record in training.items():
        if record["problems"]:
            message = f"{name}: not a full-protocol run: " + "; ".join(record["problems"])
            if not args.allow_incomplete:
                raise SystemExit(message + " (use --allow-incomplete for tests)")
            print("WARNING " + message)
    titles = {name: {"title": title, "family": family} for name, (title, family) in REFERENCE_ROWS.items()}
    titles.update({name: {"title": COMPARATORS[name].title, "family": COMPARATORS[name].family} for name in models})
    summary: dict = {
        "protocol": {
            "bootstrap": "percentile 95% CI; exams resampled with replacement",
            "n_boot": args.n_boot,
            "seed": args.seed,
            "paired": "AUC(LSA-SSViT) - AUC(comparator) on the same resamples",
        },
        "models": titles,
        "training": training,
        "datasets": {},
    }
    for dataset, (title, _, _, unit) in DATASETS.items():
        frames = load_dataset(args, dataset, models)
        order = [name for name in ["baseline", *models, PROPOSED] if name in frames]
        frames = {name: frames[name] for name in order}
        missing = [name for name in models if name not in frames]
        if missing:
            message = f"{dataset}: no predictions for {missing}"
            if not args.allow_incomplete:
                raise SystemExit(message)
            print("WARNING " + message)
        pairs = [(PROPOSED, name) for name in order if name != PROPOSED]
        summary["datasets"][dataset] = {
            "title": title,
            "unit": unit,
            "image_level": auc_table(frames, args.n_boot, args.seed, "image"),
            "paired_vs_LSA-SSViT": pair_table(frames, pairs, HEADS, args.n_boot, args.seed),
        }
        print(f"{dataset}: {len(frames)} models summarised")

    args.runs_root.mkdir(parents=True, exist_ok=True)
    json_path = args.runs_root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = markdown(summary)
    (args.runs_root / "comparison_tables.md").write_text(text, encoding="utf-8", newline="\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    print(f"Wrote {json_path} and {args.runs_root / 'comparison_tables.md'}")


if __name__ == "__main__":
    main()
