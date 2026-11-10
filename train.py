#!/usr/bin/env python
"""Train the baseline, +GRAIN, +CUSP or LSA-SSViT model on VinDr-Mammo.

    python train.py --cache-root CACHE --experiment-name baseline
    python train.py --cache-root CACHE --experiment-name grain       --grain
    python train.py --cache-root CACHE --experiment-name cusp        --cusp
    python train.py --cache-root CACHE --experiment-name LSA-SSViT  --grain --cusp

All other hyperparameters default to the values used in the paper. The checkpoint with
the best validation screening AUC is kept as best.pt; at the end, operating thresholds
are selected on the validation split (Youden's index) and the held-out split is
evaluated. The official test split is only used by evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core import (
    DEFAULT_BACKBONE,
    CHECKPOINT_FORMAT,
    BreastClassifier,
    MammographyDataset,
    capture_rng_state,
    compute_pos_weight,
    cusp_loss,
    dataframe_summary,
    model_config,
    model_parameter_counts,
    restore_rng_state,
    seed_everything,
    seed_worker,
    sha256_file,
    validate_split_disjoint,
    write_json,
)
from metrics import (
    CLASS_NAMES,
    choose_thresholds,
    detailed_classification_metrics,
    grouped_probability_metrics,
    parse_birads_ge3,
    summarize_probabilities,
)

SELECTION_METRIC = "auc_birads"  # validation screening AUC
WARMUP_EPOCHS = 1
AMP_WARNING_SKIP_RATE = 0.10
AMP_ABORT_SKIP_RATE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRAIN / CUSP model on VinDr-Mammo.")
    parser.add_argument("--experiment-name", required=True, help="output folder: <output-root>/runs/<name>")
    parser.add_argument("--grain", action="store_true", help="add GRAIN after each of the last four blocks")
    parser.add_argument("--cusp", action="store_true", help="add the CUSP terms to the training loss")
    parser.add_argument("--cache-root", type=Path, default=None, help="preprocessing cache (scripts/build_cache.py)")
    parser.add_argument("--image-root", type=Path, default=None, help="DICOM root, used when no cache is given")
    parser.add_argument("--split-root", type=Path, default=Path("splits/vindr"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help="Hugging Face model id or local directory")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in the experiment folder")

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6, help="early stopping on the validation screening AUC")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate of the head and GRAIN")
    parser.add_argument("--backbone-lr-ratio", type=float, default=0.5, help="backbone lr = lr * ratio")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--amp-init-scale", type=float, default=2.0, help="initial GradScaler scale")
    parser.add_argument("--cusp-cu-weight", type=float, default=0.3, help="lambda1 (L_CU)")
    parser.add_argument("--cusp-pc-weight", type=float, default=0.1, help="lambda2 (L_PC)")
    parser.add_argument("--cusp-margin", type=float, default=0.2, help="margin of L_PC")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args()
    if args.cache_root is None and args.image_root is None:
        parser.error("pass --cache-root (recommended) or --image-root")
    if args.batch_size < 2 and args.cusp:
        parser.error("CUSP needs at least two images per batch")
    if any(character in args.experiment_name for character in '<>:"/\\|?*'):
        parser.error("the experiment name contains characters that are not allowed in folder names")
    return args


class Logger:
    """Print to the console and append to a log file."""

    def __init__(self, path: Path, append: bool = False) -> None:
        self.file = path.open("a" if append else "w", encoding="utf-8")

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def make_loader(
    dataset: MammographyDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    pin_memory: bool,
    generator: torch.Generator,
    prefetch_factor: int,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-epoch learning-rate factor.

    With warmup_epochs = 1 the factor is 1.0 in the first two epochs and then follows a
    cosine decay to the last epoch, with a floor of 1% of the base learning rate.
    """
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))

    def factor(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return max(0.1, (epoch_index + 1) / warmup_epochs)
        remaining = max(total_epochs - warmup_epochs - 1, 1)
        progress = min(max((epoch_index - warmup_epochs) / remaining, 0.0), 1.0)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def run_epoch(
    model: BreastClassifier,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    phase: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    cusp: bool = False,
    cusp_cu_weight: float = 0.3,
    cusp_pc_weight: float = 0.1,
    cusp_margin: float = 0.2,
) -> tuple[dict[str, float], pd.DataFrame]:
    """One pass over a loader. Training uses mixed precision on CUDA; evaluation runs in float32."""
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    totals = {"total": 0.0, "bce": 0.0, "cusp_cu": 0.0, "cusp_pc": 0.0}
    count = 0
    steps = skipped = 0
    amp_enabled = bool(training and scaler is not None and scaler.is_enabled())
    all_targets, all_logits, all_probabilities = [], [], []
    study_ids: list[str] = []
    image_ids: list[str] = []
    lateralities: list[str] = []
    start = time.perf_counter()
    progress = tqdm(loader, desc=phase, dynamic_ncols=True, leave=False, mininterval=0.5)

    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        terms = None
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=(training and device.type == "cuda")):
                logits, features = model(images, return_features=True)
                if training:
                    bce = criterion(logits, labels)
                    loss = bce
                    if cusp:
                        terms = cusp_loss(
                            logits,
                            labels,
                            features,
                            birads_level=batch["birads_level"],
                            pos_weight=criterion.pos_weight,
                            margin=cusp_margin,
                        )
                        loss = loss + cusp_cu_weight * terms["cu"] + cusp_pc_weight * terms["pc"]
            if not training:
                bce = criterion(logits.float(), labels.float())
                loss = bce

            if not bool(torch.isfinite(logits.detach()).all()):
                raise FloatingPointError(f"{phase}: non-finite logits in batch {step + 1}")
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(f"{phase}: non-finite loss in batch {step + 1}")

            if training:
                scaler.scale(loss).backward()
                scale_before = float(scaler.get_scale()) if amp_enabled else 1.0
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                steps += 1
                # GradScaler skips the step (and lowers the scale) when gradients overflow.
                if amp_enabled and float(scaler.get_scale()) < scale_before:
                    skipped += 1
                optimizer.zero_grad(set_to_none=True)

        batch_size = int(images.shape[0])
        count += batch_size
        totals["total"] += float(loss.detach().float().item()) * batch_size
        totals["bce"] += float(bce.detach().float().item()) * batch_size
        if terms is not None:
            totals["cusp_cu"] += float(terms["cu"].detach().float().item()) * batch_size
            totals["cusp_pc"] += float(terms["pc"].detach().float().item()) * batch_size
        all_targets.append(labels.detach().float().cpu().numpy())
        logits_fp32 = logits.detach().float()
        all_logits.append(logits_fp32.cpu().numpy().astype(np.float64, copy=False))
        all_probabilities.append(torch.sigmoid(logits_fp32).cpu().numpy().astype(np.float64, copy=False))
        study_ids.extend(str(value) for value in batch["study_id"])
        image_ids.extend(str(value) for value in batch["image_id"])
        lateralities.extend(str(value) for value in batch["laterality"])
        progress.set_postfix_str(f"loss {totals['total'] / count:.4f}")

    if training:
        finite = [torch.isfinite(parameter.detach()).all() for parameter in model.parameters()]
        if finite and not bool(torch.stack(finite).all()):
            raise FloatingPointError(f"{phase}: parameters became non-finite")

    targets = np.concatenate(all_targets, axis=0).astype(np.float64, copy=False)
    logits_np = np.concatenate(all_logits, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    metrics = summarize_probabilities(targets, probabilities, class_names=CLASS_NAMES)
    metrics.update({f"loss_{key}": value / max(count, 1) for key, value in totals.items()})
    metrics.update({
        "optimizer_steps": float(steps),
        "amp_skipped_steps": float(skipped),
        "amp_skip_rate": skipped / steps if steps else 0.0,
        "seconds": time.perf_counter() - start,
    })

    predictions = pd.DataFrame({"study_id": study_ids, "image_id": image_ids, "laterality": lateralities})
    for index, name in enumerate(CLASS_NAMES):
        predictions[f"target_{name}"] = targets[:, index].astype(int)
        predictions[f"logit_{name}"] = logits_np[:, index]
        predictions[f"prob_{name}"] = probabilities[:, index]
    return metrics, predictions


def add_threshold_columns(predictions: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    result = predictions.copy()
    for name in thresholds:
        threshold = float(thresholds[name])
        result[f"threshold_{name}"] = threshold
        result[f"pred_{name}"] = (result[f"prob_{name}"] >= threshold).astype(int)
    return result


def evaluation_metrics(predictions: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, float]:
    targets = predictions[[f"target_{name}" for name in CLASS_NAMES]].to_numpy(dtype=np.float64)
    probabilities = predictions[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=np.float64)
    metrics = detailed_classification_metrics(targets, probabilities, thresholds, class_names=CLASS_NAMES)
    metrics.update(grouped_probability_metrics(predictions, "breast"))
    metrics.update(grouped_probability_metrics(predictions, "study"))
    return metrics


def critical_config(args: argparse.Namespace) -> dict[str, Any]:
    """Settings that must not change when a run is resumed."""
    keys = ["grain", "cusp", "backbone", "seed", "epochs", "batch_size", "lr", "backbone_lr_ratio", "weight_decay",
            "dropout", "pos_weight_cap", "amp_init_scale", "cusp_cu_weight", "cusp_pc_weight", "cusp_margin",
            "num_workers", "split_root"]
    return {key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
            for key in keys}


def environment_info(device: torch.device) -> dict[str, Any]:
    import transformers

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / "runs" / args.experiment_name
    if args.resume:
        if not (output_dir / "last.pt").is_file():
            raise FileNotFoundError(f"Nothing to resume: {output_dir / 'last.pt'} does not exist")
    elif output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose another --experiment-name or pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(output_dir / "train.log", append=args.resume)

    try:
        seed_everything(args.seed, deterministic=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        split_paths = {name: args.split_root / f"{name}.csv" for name in ("train", "val", "eval")}
        frames = {name: pd.read_csv(path, dtype={"study_id": str}) for name, path in split_paths.items()}
        validate_split_disjoint(frames)
        for frame in frames.values():
            frame["birads"] = frame["breast_birads"].map(parse_birads_ge3).astype(int)

        train_generator = torch.Generator().manual_seed(args.seed)
        eval_generator = torch.Generator().manual_seed(args.seed + 999)
        data_kwargs = {"cache_root": args.cache_root, "image_root": args.image_root}
        datasets = {
            "train": MammographyDataset(split_paths["train"], train=True, **data_kwargs),
            "val": MammographyDataset(split_paths["val"], train=False, **data_kwargs),
            "eval": MammographyDataset(split_paths["eval"], train=False, **data_kwargs),
        }
        pin = device.type == "cuda"
        loaders = {
            "train": make_loader(datasets["train"], args.batch_size, args.num_workers, True, pin,
                                 train_generator, args.prefetch_factor),
            "val": make_loader(datasets["val"], args.batch_size, args.num_workers, False, pin,
                               eval_generator, args.prefetch_factor),
            "eval": make_loader(datasets["eval"], args.batch_size, args.num_workers, False, pin,
                                eval_generator, args.prefetch_factor),
        }

        model = BreastClassifier(grain=args.grain, backbone=args.backbone, pretrained=True,
                                 dropout=args.dropout).to(device)
        pos_weight = compute_pos_weight(frames["train"], args.pos_weight_cap, columns=CLASS_NAMES).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        backbone_ids = {id(parameter) for parameter in model.features.parameters()}
        backbone_parameters = [p for p in model.features.parameters() if p.requires_grad]
        head_parameters = [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": args.lr * args.backbone_lr_ratio, "group_name": "backbone"},
                {"params": head_parameters, "lr": args.lr, "group_name": "head"},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = build_scheduler(optimizer, args.epochs, WARMUP_EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=float(args.amp_init_scale))
        cusp_kwargs = {"cusp": args.cusp, "cusp_cu_weight": args.cusp_cu_weight,
                       "cusp_pc_weight": args.cusp_pc_weight, "cusp_margin": args.cusp_margin}

        counts = model_parameter_counts(model)
        grain_parameters = sum(p.numel() for p in model.grain.parameters()) if model.grain is not None else 0
        log("=" * 88)
        log(f"Experiment {args.experiment_name}: GRAIN {'on' if args.grain else 'off'}, "
            f"CUSP {'on' if args.cusp else 'off'} | device {device} | seed {args.seed}")
        for name, frame in frames.items():
            summary = dataframe_summary(frame)
            log(f"  {name:5s}: {summary['images']:5d} images, {summary['exams']:4d} exams, positives "
                f"mass {summary['positive_mass']}, calcification {summary['positive_calcification']}, "
                f"screening {summary['positive_birads']}")
        log(f"  parameters: {counts['total']:,} total, {counts['trainable']:,} trainable, GRAIN {grain_parameters:,}")
        log(f"  positive weights (mass, calcification, screening): "
            + ", ".join(f"{value:.3f}" for value in pos_weight.tolist()))
        log("=" * 88)

        run_path = output_dir / "run.json"
        run_record: dict[str, Any] = {
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "command": " ".join(sys.argv),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "environment": environment_info(device),
            "datasets": {name: dataframe_summary(frame) for name, frame in frames.items()},
            "split_sha256": {name: sha256_file(path) for name, path in split_paths.items()},
            "parameters": {**counts, "grain": int(grain_parameters)},
            "pos_weight": pos_weight.detach().cpu().tolist(),
        }
        if args.resume and run_path.is_file():
            previous = json.loads(run_path.read_text(encoding="utf-8"))
            run_record["started_at"] = previous.get("started_at", run_record["started_at"])
            run_record["resumed_at"] = previous.get("resumed_at", []) + [datetime.now().astimezone().isoformat()]
        write_json(run_path, run_record)

        history_path = output_dir / "history.csv"
        history = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.is_file() else []
        start_epoch, best_metric, best_epoch, bad_epochs = 1, -float("inf"), 0, 0
        if args.resume:
            state = torch.load(output_dir / "last.pt", map_location=device, weights_only=False)
            if state.get("critical_config") != critical_config(args):
                raise ValueError("The settings differ from the run being resumed.")
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            scaler.load_state_dict(state["scaler_state"])
            restore_rng_state(state["rng_state"], train_generator)
            start_epoch = int(state["epoch"]) + 1
            best_metric, best_epoch = float(state["best_metric"]), int(state["best_epoch"])
            bad_epochs = int(state["bad_epochs"])
            log(f"Resumed from epoch {state['epoch']}.")

        stop_reason = "maximum number of epochs reached"
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics, _ = run_epoch(model, loaders["train"], criterion, device, f"train {epoch:02d}/{args.epochs}",
                                         optimizer, scaler, **cusp_kwargs)
            steps, skipped = int(train_metrics["optimizer_steps"]), int(train_metrics["amp_skipped_steps"])
            if steps > 0 and (skipped == steps or skipped / steps >= AMP_ABORT_SKIP_RATE):
                raise RuntimeError(f"Mixed-precision instability: {skipped}/{steps} optimizer steps skipped "
                                   f"in epoch {epoch}.")
            val_metrics, _ = run_epoch(model, loaders["val"], criterion, device, f"val {epoch:02d}/{args.epochs}")

            current = float(val_metrics[SELECTION_METRIC])
            improved = not math.isnan(current) and current > best_metric
            if improved:
                best_metric, best_epoch, bad_epochs = current, epoch, 0
                torch.save(
                    {
                        "format_version": CHECKPOINT_FORMAT,
                        "model_state": model.state_dict(),
                        "config": model_config(model, cusp=bool(args.cusp), cusp_cu_weight=args.cusp_cu_weight,
                                               cusp_pc_weight=args.cusp_pc_weight, cusp_margin=args.cusp_margin,
                                               seed=args.seed),
                        "epoch": epoch,
                        "val_metrics": {k: float(v) for k, v in val_metrics.items() if k.startswith(("auc_", "ap_"))},
                    },
                    output_dir / "best.pt",
                )
            else:
                bad_epochs += 1

            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[1]["lr"]
            scheduler.step()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "rng_state": capture_rng_state(train_generator),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                    "critical_config": critical_config(args),
                },
                output_dir / "last.pt",
            )

            row = {"epoch": epoch, "lr_backbone": lr_backbone, "lr_head": lr_head}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)

            note = "new best" if improved else f"no improvement ({bad_epochs}/{args.patience})"
            if skipped and skipped / steps >= AMP_WARNING_SKIP_RATE:
                note += f"; warning: {skipped}/{steps} mixed-precision steps skipped"
            log(f"epoch {epoch:02d} | lr {lr_head:.2e}/{lr_backbone:.2e} | train loss {train_metrics['loss_total']:.4f} | "
                f"val AUC screening {val_metrics['auc_birads']:.4f} mass {val_metrics['auc_mass']:.4f} "
                f"calcification {val_metrics['auc_calcification']:.4f} | {note}")
            if bad_epochs >= args.patience:
                stop_reason = f"early stopping: validation screening AUC did not improve for {args.patience} epochs"
                break

        log(stop_reason)
        best = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best["model_state"])
        _, val_predictions = run_epoch(model, loaders["val"], criterion, device, "validation (best.pt)")
        thresholds = choose_thresholds(
            val_predictions[[f"target_{name}" for name in CLASS_NAMES]].to_numpy(dtype=np.float64),
            val_predictions[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=np.float64),
            class_names=CLASS_NAMES,
        )
        _, eval_predictions = run_epoch(model, loaders["eval"], criterion, device, "held-out split (best.pt)")
        eval_metrics = evaluation_metrics(eval_predictions, thresholds)
        add_threshold_columns(val_predictions, thresholds).to_csv(output_dir / "predictions_val.csv", index=False)
        add_threshold_columns(eval_predictions, thresholds).to_csv(output_dir / "predictions_eval.csv", index=False)
        write_json(output_dir / "metrics_eval.json", {"thresholds": thresholds, "metrics": eval_metrics})

        log(f"best epoch {best_epoch} (validation screening AUC {best_metric:.4f})")
        log("held-out split: " + " | ".join(
            f"{name} AUC {eval_metrics[f'auc_{name}']:.4f} AP {eval_metrics[f'ap_{name}']:.3f}" for name in CLASS_NAMES))
        run_record.update({
            "status": "completed",
            "ended_at": datetime.now().astimezone().isoformat(),
            "stop_reason": stop_reason,
            "best_epoch": best_epoch,
            "best_validation_screening_auc": best_metric,
            "thresholds_from_validation": thresholds,
            "held_out_metrics": eval_metrics,
        })
        write_json(run_path, run_record)
        log(f"Results saved to {output_dir.resolve()}")
    except Exception as error:
        log(f"Failed: {type(error).__name__}: {error}")
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
