#!/usr/bin/env python
"""External test on INbreast without fine-tuning.

The four VinDr-trained checkpoints are applied without fine-tuning or recalibration.
Images are preprocessed on the fly with the same fixed pipeline as VinDr-Mammo (no cache).

    python scripts/prepare_inbreast.py --xls ... --dicom-dir ...
    python scripts/evaluate_inbreast.py --image-root dataset/inbreast/AllDICOMs

Writes results/inbreast/<model>/predictions.csv and metrics.json, and
results/inbreast/summary.json with the screening AUC CI (patients resampled) and the
paired differences versus the baseline. scripts/paper_tables.py then builds the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import MammographyDataset, load_model, seed_everything, sha256_file, write_json  # noqa: E402
from metrics import (  # noqa: E402
    bootstrap_auc_ci,
    paired_bootstrap_auc_difference,
    parse_birads_ge3,
    safe_ap,
    safe_auc,
)
from train import make_loader, run_epoch  # noqa: E402

MODELS = ["baseline", "grain", "cusp", "LSA-SSViT"]
EXPECTED_POSITIVE = {"birads": 123, "mass": 108, "calcification": 308}
LABEL_RULES = {
    "birads": "BI-RADS first digit >= 3 (4a/4b/4c and 6 included), as for VinDr-Mammo",
    "mass": "INbreast 'Mass' annotation (benign findings included)",
    "calcification": "INbreast 'Micros' annotation (benign findings included)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation on INbreast without fine-tuning.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs" / "inbreast" / "manifest.csv")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "dataset" / "inbreast" / "AllDICOMs")
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "inbreast")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def check_manifest(path: Path) -> None:
    frame = pd.read_csv(path)
    counts = {
        "birads": int(frame["breast_birads"].map(parse_birads_ge3).sum()),
        "mass": int(frame["mass"].sum()),
        "calcification": int(frame["calcification"].sum()),
    }
    if len(frame) != 410 or counts != EXPECTED_POSITIVE:
        raise RuntimeError(f"Unexpected manifest: {len(frame)} rows, positives {counts}")


def image_metrics(frame: pd.DataFrame, n_boot: int, seed: int) -> dict:
    groups = frame["study_id"].to_numpy()
    record = {"n_images": int(len(frame)), "n_patients": int(frame["study_id"].nunique())}
    for name in ("birads", "mass", "calcification"):
        target = frame[f"target_{name}"].to_numpy()
        score = frame[f"prob_{name}"].to_numpy()
        record[f"auc_{name}"] = safe_auc(target, score)
        record[f"ap_{name}"] = safe_ap(target, score)
        record[f"n_positive_{name}"] = int(target.sum())
    record["auc_birads_ci95"] = list(bootstrap_auc_ci(frame["target_birads"].to_numpy(),
                                                      frame["prob_birads"].to_numpy(), groups, n_boot, seed))
    return record


def main() -> None:
    args = parse_args()
    check_manifest(args.manifest)
    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    frames: dict[str, pd.DataFrame] = {}
    for name in MODELS:
        checkpoint_path = args.checkpoint_root / name / "best.pt"
        output_dir = args.output_root / name
        prediction_path = output_dir / "predictions.csv"
        metrics_path = output_dir / "metrics.json"
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if prediction_path.is_file() and metrics_path.is_file():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            if previous.get("checkpoint_sha256") == checkpoint_sha256:
                print(f"{name}: predictions of this checkpoint already exist, reusing {prediction_path}")
                frames[name] = pd.read_csv(prediction_path, dtype={"study_id": str, "image_id": str})
                continue
        model, _ = load_model(checkpoint_path, device)
        dataset = MammographyDataset(args.manifest, train=False, image_root=args.image_root)
        loader = make_loader(dataset, args.batch_size, args.num_workers, False, device.type == "cuda", generator, 2)
        _, predictions = run_epoch(model, loader, nn.BCEWithLogitsLoss(), device, f"INbreast {name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(prediction_path, index=False)
        record = image_metrics(predictions, args.n_boot, args.seed)
        write_json(metrics_path, {"checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256, **record})
        frames[name] = pd.read_csv(prediction_path, dtype={"study_id": str, "image_id": str})
        print(f"{name}: screening AUC {record['auc_birads']:.4f}, mass AUC {record['auc_mass']:.4f}, "
              f"calcification AUC {record['auc_calcification']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cells = {name: image_metrics(frame, args.n_boot, args.seed) for name, frame in frames.items()}
    paired = []
    for name in MODELS[1:]:
        merged = frames["baseline"].merge(frames[name], on="image_id", suffixes=("_ref", "_new"))
        point, low, high = paired_bootstrap_auc_difference(
            merged["target_birads_ref"].to_numpy(),
            merged["prob_birads_ref"].to_numpy(),
            merged["prob_birads_new"].to_numpy(),
            merged["study_id_ref"].to_numpy(),
            args.n_boot,
            args.seed,
        )
        paired.append({"comparison": f"{name} - baseline", "delta_auc_birads": point, "ci95": [low, high],
                       "significant": bool(low > 0 or high < 0)})
    write_json(args.output_root / "summary.json", {
        "dataset": "INbreast (external test, no fine-tuning or recalibration)",
        "label_rules": LABEL_RULES,
        "bootstrap": {"unit": "patient", "n_boot": args.n_boot, "seed": args.seed},
        "models": cells,
        "paired_vs_baseline": paired,
    })
    print(f"Wrote {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
