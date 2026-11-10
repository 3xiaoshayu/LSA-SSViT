#!/usr/bin/env python
"""Re-train a comparator network (comparators/zoo.py) with the protocol of train.py, then evaluate it.

    python comparators/train_comparator.py --model resnet50 --cache-root CACHE

Training is that of train.py without GRAIN and CUSP: the checkpoint with the best validation
screening AUC is kept, thresholds are chosen on the validation split with Youden's index and
the held-out split is evaluated. The best checkpoint is then evaluated on the official
VinDr-Mammo test split (as evaluate.py). Everything goes to <output-root>/<model>/: train.log,
run.json, history.csv, best.pt, last.pt, predictions_{val,eval,test}.csv and
metrics_{eval,test}.json. --evaluate-only repeats the evaluation from best.pt. run.json records
whether training ended by the protocol (early stopping or the epoch limit: "training_finished"); results of a
run that was cut short are marked as such and refused by comparators/summarize.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "scripts"))

from comparators.zoo import (  # noqa: E402
    COMPARATOR_CHECKPOINT_FORMAT,
    COMPARATORS,
    HEAD_LR,
    ComparatorClassifier,
    comparator_config,
)
from core import (  # noqa: E402
    MammographyDataset,
    capture_rng_state,
    compute_pos_weight,
    dataframe_summary,
    model_parameter_counts,
    restore_rng_state,
    seed_everything,
    sha256_file,
    validate_split_disjoint,
    write_json,
)
from metrics import CLASS_NAMES, choose_thresholds, parse_birads_ge3  # noqa: E402
from train import (  # noqa: E402
    AMP_ABORT_SKIP_RATE,
    AMP_WARNING_SKIP_RATE,
    SELECTION_METRIC,
    WARMUP_EPOCHS,
    Logger,
    add_threshold_columns,
    build_scheduler,
    environment_info,
    evaluation_metrics,
    make_loader,
    run_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-train and evaluate a comparator network.")
    parser.add_argument("--model", required=True, choices=list(COMPARATORS), help="comparator (comparators/zoo.py)")
    parser.add_argument("--cache-root", type=Path, default=None, help="preprocessing cache (scripts/build_cache.py)")
    parser.add_argument("--image-root", type=Path, default=None, help="DICOM root, used when no cache is given")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "splits" / "vindr")
    parser.add_argument("--test-csv", type=Path, default=None, help="default: <split-root>/test.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "comparators")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "outputs" / "comparator_weights",
                        help="converted weights that are not on a model hub (Mammo-CLIP)")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt")
    parser.add_argument("--evaluate-only", action="store_true", help="only evaluate best.pt")

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6, help="early stopping on the validation screening AUC")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=HEAD_LR, help="learning rate of the classifier head")
    parser.add_argument("--backbone-lr", type=float, default=None, help="default: the value in comparators/zoo.py")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--amp-init-scale", type=float, default=2.0, help="initial GradScaler scale")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop training after the epoch that exceeds this wall-clock budget")
    args = parser.parse_args()
    if args.cache_root is None and args.image_root is None:
        parser.error("pass --cache-root (recommended) or --image-root")
    if args.backbone_lr is None:
        args.backbone_lr = COMPARATORS[args.model].backbone_lr
    if args.test_csv is None:
        args.test_csv = args.split_root / "test.csv"
    return args


def critical_config(args: argparse.Namespace) -> dict[str, Any]:
    """Settings that must not change when a run is resumed."""
    keys = ["model", "seed", "epochs", "batch_size", "lr", "backbone_lr", "weight_decay", "dropout",
            "pos_weight_cap", "amp_init_scale", "num_workers", "split_root"]
    return {key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
            for key in keys}


def save_atomic(payload: dict[str, Any], path: Path) -> None:
    """Write under a temporary name first, so that an interruption never leaves a truncated checkpoint."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate_best(args, model, loaders, criterion, device, output_dir: Path, log: Logger) -> dict[str, Any]:
    """Thresholds on validation, then the held-out split and the official test split."""
    best_path = output_dir / "best.pt"
    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    checkpoint_sha256 = sha256_file(best_path)
    plain = nn.BCEWithLogitsLoss()  # evaluation loss is only reported
    pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(args.seed)

    _, val_predictions = run_epoch(model, loaders["val"], criterion, device, "validation (best.pt)")
    thresholds = choose_thresholds(
        val_predictions[[f"target_{name}" for name in CLASS_NAMES]].to_numpy(dtype=np.float64),
        val_predictions[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=np.float64),
        class_names=CLASS_NAMES,
    )
    add_threshold_columns(val_predictions, thresholds).to_csv(output_dir / "predictions_val.csv", index=False)
    results: dict[str, Any] = {"thresholds_from_validation": thresholds, "best_epoch": int(best["epoch"])}

    _, eval_predictions = run_epoch(model, loaders["eval"], criterion, device, "held-out split (best.pt)")
    eval_metrics = evaluation_metrics(eval_predictions, thresholds)
    add_threshold_columns(eval_predictions, thresholds).to_csv(output_dir / "predictions_eval.csv", index=False)
    write_json(output_dir / "metrics_eval.json", {"thresholds": thresholds, "metrics": eval_metrics})
    results["held_out_metrics"] = eval_metrics

    data_kwargs = {"cache_root": args.cache_root, "image_root": args.image_root}
    test_loader = make_loader(MammographyDataset(args.test_csv, train=False, **data_kwargs), args.batch_size,
                              args.num_workers, False, pin, generator, args.prefetch_factor)
    _, test_predictions = run_epoch(model, test_loader, plain, device, "VinDr-Mammo test (best.pt)")
    test_metrics = evaluation_metrics(test_predictions, thresholds)
    add_threshold_columns(test_predictions, thresholds).to_csv(output_dir / "predictions_test.csv", index=False)
    write_json(output_dir / "metrics_test.json", {
        "checkpoint": str(best_path),
        "checkpoint_sha256": checkpoint_sha256,
        "val_csv": str(args.split_root / "val.csv"),
        "eval_csv": str(args.test_csv),
        "thresholds": thresholds,
        "metrics": test_metrics,
    })
    results["test_metrics"] = {key: value for key, value in test_metrics.items() if key.startswith(("auc_", "ap_"))}
    log("VinDr-Mammo test: " + " | ".join(
        f"{name} AUC {test_metrics[f'auc_{name}']:.4f} AP {test_metrics[f'ap_{name}']:.3f}" for name in CLASS_NAMES))

    return results


def main() -> None:
    args = parse_args()
    spec = COMPARATORS[args.model]
    output_dir = args.output_root / args.model
    if args.evaluate_only:
        if not (output_dir / "best.pt").is_file():
            raise FileNotFoundError(f"Nothing to evaluate: {output_dir / 'best.pt'} does not exist")
    elif args.resume:
        if not (output_dir / "last.pt").is_file():
            raise FileNotFoundError(f"Nothing to resume: {output_dir / 'last.pt'} does not exist")
    elif output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; pass --resume or --evaluate-only")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(output_dir / "train.log", append=args.resume or args.evaluate_only)
    started = time.perf_counter()

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

        model = ComparatorClassifier(args.model, pretrained=not args.evaluate_only, dropout=args.dropout,
                                     weights_dir=args.weights_dir).to(device)
        if spec.amp_dtype != "float16":  # used by run_epoch's torch.autocast (training only)
            dtype = getattr(torch, spec.amp_dtype)
            if hasattr(torch, "set_autocast_dtype"):  # PyTorch >= 2.4
                torch.set_autocast_dtype(device.type, dtype)
            else:
                torch.set_autocast_gpu_dtype(dtype)
        pos_weight = compute_pos_weight(frames["train"], args.pos_weight_cap, columns=CLASS_NAMES).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        backbone_parameters, head_parameters = model.parameter_groups()
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": args.backbone_lr, "group_name": "backbone"},
                {"params": head_parameters, "lr": args.lr, "group_name": "head"},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = build_scheduler(optimizer, args.epochs, WARMUP_EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=float(args.amp_init_scale))

        counts = model_parameter_counts(model)
        log("=" * 88)
        log(f"Comparator {spec.title} ({spec.source}): regime {spec.regime}, input {spec.input_size} px, "
            f"lr backbone {args.backbone_lr:.1e} / head {args.lr:.1e} | autocast {spec.amp_dtype} | device {device} | "
            f"seed {args.seed}")
        for name, frame in frames.items():
            summary = dataframe_summary(frame)
            log(f"  {name:5s}: {summary['images']:5d} images, {summary['exams']:4d} exams, positives "
                f"mass {summary['positive_mass']}, calcification {summary['positive_calcification']}, "
                f"screening {summary['positive_birads']}")
        log(f"  parameters: {counts['total']:,} total, {counts['trainable']:,} trainable")
        log("=" * 88)

        run_path = output_dir / "run.json"
        run_record: dict[str, Any] = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
        if not args.evaluate_only:
            import timm

            environment = environment_info(device)
            environment["timm"] = timm.__version__
            run_record.update({
                "status": "running",
                "started_at": run_record.get("started_at", datetime.now().astimezone().isoformat()),
                "command": " ".join(sys.argv),
                "comparator": comparator_config(model)["spec"],
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "environment": environment,
                "datasets": {name: dataframe_summary(frame) for name, frame in frames.items()},
                "split_sha256": {name: sha256_file(path) for name, path in split_paths.items()},
                "parameters": counts,
                "pos_weight": pos_weight.detach().cpu().tolist(),
            })
            if args.resume:
                run_record["resumed_at"] = run_record.get("resumed_at", []) + [datetime.now().astimezone().isoformat()]
            for key in ("stop_reason", "training_finished", "best_epoch", "best_validation_screening_auc"):
                run_record.pop(key, None)
            write_json(run_path, run_record)

            history_path = output_dir / "history.csv"
            history = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.is_file() else []
            start_epoch, best_metric, best_epoch, bad_epochs = 1, -float("inf"), 0, 0
            stop_reason = "maximum number of epochs reached"
            if args.resume:
                state = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
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
                del state
                if bad_epochs >= args.patience:  # the run had already stopped early: no further epoch
                    stop_reason = f"early stopping: validation screening AUC did not improve for {args.patience} epochs"
                    start_epoch = args.epochs + 1

            for epoch in range(start_epoch, args.epochs + 1):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                train_metrics, _ = run_epoch(model, loaders["train"], criterion, device,
                                             f"train {epoch:02d}/{args.epochs}", optimizer, scaler)
                steps, skipped = int(train_metrics["optimizer_steps"]), int(train_metrics["amp_skipped_steps"])
                if steps > 0 and (skipped == steps or skipped / steps >= AMP_ABORT_SKIP_RATE):
                    raise RuntimeError(f"Mixed-precision instability: {skipped}/{steps} optimizer steps skipped "
                                       f"in epoch {epoch}.")
                val_metrics, _ = run_epoch(model, loaders["val"], criterion, device, f"val {epoch:02d}/{args.epochs}")
                peak_gb = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0

                current = float(val_metrics[SELECTION_METRIC])
                improved = not math.isnan(current) and current > best_metric
                if improved:
                    best_metric, best_epoch, bad_epochs = current, epoch, 0
                    save_atomic(
                        {
                            "format_version": COMPARATOR_CHECKPOINT_FORMAT,
                            "model_state": model.state_dict(),
                            "config": comparator_config(model, seed=args.seed),
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
                save_atomic(
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

                row = {"epoch": epoch, "lr_backbone": lr_backbone, "lr_head": lr_head, "peak_memory_gb": peak_gb}
                row.update({f"train_{key}": value for key, value in train_metrics.items()})
                row.update({f"val_{key}": value for key, value in val_metrics.items()})
                history.append(row)
                pd.DataFrame(history).to_csv(history_path, index=False)

                note = "new best" if improved else f"no improvement ({bad_epochs}/{args.patience})"
                if skipped and skipped / steps >= AMP_WARNING_SKIP_RATE:
                    note += f"; warning: {skipped}/{steps} mixed-precision steps skipped"
                log(f"epoch {epoch:02d} | lr {lr_head:.2e}/{lr_backbone:.2e} | train loss "
                    f"{train_metrics['loss_total']:.4f} | val AUC screening {val_metrics['auc_birads']:.4f} "
                    f"mass {val_metrics['auc_mass']:.4f} calcification {val_metrics['auc_calcification']:.4f} | "
                    f"{train_metrics['seconds'] / 60:.1f} min, {peak_gb:.1f} GB | {note}")
                if bad_epochs >= args.patience:
                    stop_reason = f"early stopping: validation screening AUC did not improve for {args.patience} epochs"
                    break
                if args.max_hours is not None and time.perf_counter() - started > args.max_hours * 3600:
                    stop_reason = f"time budget of {args.max_hours:g} h reached"
                    break
            log(stop_reason)
            log(f"best epoch {best_epoch} (validation screening AUC {best_metric:.4f})")
            run_record.update({"stop_reason": stop_reason, "best_epoch": best_epoch,
                               "best_validation_screening_auc": best_metric,
                               "training_finished": not stop_reason.startswith("time budget")})
            write_json(run_path, run_record)

        results = evaluate_best(args, model, loaders, criterion, device, output_dir, log)
        status = "completed" if run_record.get("training_finished") else "evaluated_incomplete_training"
        run_record.update({"status": status, "ended_at": datetime.now().astimezone().isoformat(),
                           "hours_this_invocation": (time.perf_counter() - started) / 3600, **results})
        write_json(run_path, run_record)
        log(f"Results saved to {output_dir.resolve()}")
    except Exception as error:
        log(f"Failed: {type(error).__name__}: {error}")
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
