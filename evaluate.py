#!/usr/bin/env python
"""Evaluate a trained checkpoint on VinDr-Mammo.

Operating thresholds are selected on the validation split with Youden's index and then
applied unchanged to the evaluated split (by default the official test split):

    python evaluate.py --checkpoint checkpoints/LSA-SSViT/best.pt --cache-root CACHE \
        --output-dir outputs/eval/LSA-SSViT

Writes metrics_<name>.json (image-, breast- and exam-level metrics) and
predictions_<name>.csv (per-image labels, logits, probabilities and decisions).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core import MammographyDataset, load_model, seed_everything, write_json
from metrics import CLASS_NAMES, choose_thresholds
from train import add_threshold_columns, evaluation_metrics, make_loader, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on VinDr-Mammo.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None, help="preprocessing cache (scripts/build_cache.py)")
    parser.add_argument("--image-root", type=Path, default=None, help="DICOM root, used when no cache is given")
    parser.add_argument("--val-csv", type=Path, default=Path("splits/vindr/val.csv"), help="selects the thresholds")
    parser.add_argument("--eval-csv", type=Path, default=Path("splits/vindr/test.csv"), help="split to evaluate")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: the checkpoint folder")
    parser.add_argument("--name", default="test", help="suffix of the output files")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    if args.cache_root is None and args.image_root is None:
        parser.error("pass --cache-root or --image-root")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(2026, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)

    output_dir = args.output_dir or args.checkpoint.parent
    json_path = output_dir / f"metrics_{args.name}.json"
    csv_path = output_dir / f"predictions_{args.name}.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(f"{json_path} or {csv_path} already exists; choose another --name or --output-dir")

    data_kwargs = {"cache_root": args.cache_root, "image_root": args.image_root}
    generator = torch.Generator().manual_seed(2026)
    pin = device.type == "cuda"
    val_loader = make_loader(MammographyDataset(args.val_csv, train=False, **data_kwargs),
                             args.batch_size, args.num_workers, False, pin, generator, 2)
    eval_loader = make_loader(MammographyDataset(args.eval_csv, train=False, **data_kwargs),
                              args.batch_size, args.num_workers, False, pin, generator, 2)
    # The loss is only reported; it does not influence predictions or thresholds.
    criterion = nn.BCEWithLogitsLoss()

    _, val_predictions = run_epoch(model, val_loader, criterion, device, "validation")
    thresholds = choose_thresholds(
        val_predictions[[f"target_{name}" for name in CLASS_NAMES]].to_numpy(dtype=np.float64),
        val_predictions[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=np.float64),
        class_names=CLASS_NAMES,
    )
    _, predictions = run_epoch(model, eval_loader, criterion, device, "evaluation")
    metrics = evaluation_metrics(predictions, thresholds)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(json_path, {
        "checkpoint": str(args.checkpoint),
        "val_csv": str(args.val_csv),
        "eval_csv": str(args.eval_csv),
        "thresholds": thresholds,
        "metrics": metrics,
    })
    add_threshold_columns(predictions, thresholds).to_csv(csv_path, index=False)

    grain, cusp = checkpoint["config"]["grain"], checkpoint["config"].get("cusp", False)
    print(f"{args.checkpoint} (GRAIN {'on' if grain else 'off'}, CUSP {'on' if cusp else 'off'}) on {args.eval_csv}")
    for name, title in (("birads", "screening"), ("mass", "mass"), ("calcification", "calcification")):
        print(f"  {title:13s} AUC {metrics[f'auc_{name}']:.4f}  AP {metrics[f'ap_{name}']:.3f}  "
              f"exam-level AUC {metrics[f'study_auc_{name}']:.4f}  "
              f"sensitivity/specificity {metrics[f'sensitivity_{name}']:.3f}/{metrics[f'specificity_{name}']:.3f}")
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
