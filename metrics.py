"""Label definitions and evaluation metrics.

This module only depends on NumPy, pandas and scikit-learn, so the tables and
figures of the paper can be reproduced from the released predictions without
PyTorch or a GPU.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# Output heads, in the order of the model logits.
CLASS_NAMES = ("mass", "calcification", "birads")
HEAD_TITLES = {"birads": "Screening", "mass": "Mass", "calcification": "Calcification"}


# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------


def parse_birads_ge3(value: Any) -> int:
    """Screening label: BI-RADS 3/4/5 (and 6 on INbreast) -> 1, BI-RADS 1/2 -> 0.

    Only the first digit is read, so INbreast sub-categories such as "4a" follow
    their parent category.
    """
    match = re.search(r"(\d)", str(value))
    if match is None:
        raise ValueError(f"Cannot parse BI-RADS category: {value!r}")
    return int(int(match.group(1)) >= 3)


def parse_birads_level(value: Any) -> int:
    """BI-RADS category 1-5; 0 when the category is missing or outside 1-5."""
    match = re.search(r"(\d)", str(value))
    if match is None:
        return 0
    level = int(match.group(1))
    return level if 1 <= level <= 5 else 0


# -----------------------------------------------------------------------------
# Image-level metrics
# -----------------------------------------------------------------------------


def sensitivity_at_specificity(target: np.ndarray, score: np.ndarray, specificity: float = 0.90) -> float:
    target = np.asarray(target)
    score = np.asarray(score, dtype=np.float64)
    if np.unique(target).size < 2:
        return float("nan")
    false_positive, true_positive, _ = roc_curve(target, score)
    spec = 1.0 - false_positive
    keep = spec >= float(specificity)
    if not np.any(keep):
        return float(true_positive[int(np.argmax(spec))])
    return float(np.max(true_positive[keep]))


def specificity_at_sensitivity(target: np.ndarray, score: np.ndarray, sensitivity: float = 0.90) -> float:
    target = np.asarray(target)
    score = np.asarray(score, dtype=np.float64)
    if np.unique(target).size < 2:
        return float("nan")
    false_positive, true_positive, _ = roc_curve(target, score)
    spec = 1.0 - false_positive
    keep = true_positive >= float(sensitivity)
    if not np.any(keep):
        return float(spec[int(np.argmax(true_positive))])
    return float(np.max(spec[keep]))


def safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    return float("nan") if np.unique(target).size < 2 else float(roc_auc_score(target, score))


def safe_ap(target: np.ndarray, score: np.ndarray) -> float:
    return float("nan") if np.unique(target).size < 2 else float(average_precision_score(target, score))


def summarize_probabilities(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str] | tuple[str, ...] = CLASS_NAMES,
) -> dict[str, float]:
    """AUC, AP and simple score statistics per head.

    Scores are cast to float64 so that ties are not introduced by half precision.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    names = list(class_names)
    if targets.ndim != 2 or probabilities.shape[1] != len(names):
        raise ValueError("targets/probabilities must have one column per class name")
    metrics: dict[str, float] = {}
    aucs: list[float] = []
    aps: list[float] = []
    for index, name in enumerate(names):
        target = targets[:, index]
        score = probabilities[:, index]
        auc = safe_auc(target, score)
        ap = safe_ap(target, score)
        metrics[f"auc_{name}"] = auc
        metrics[f"ap_{name}"] = ap
        metrics[f"prob_mean_{name}"] = float(np.mean(score)) if score.size else float("nan")
        metrics[f"prob_std_{name}"] = float(np.std(score)) if score.size else float("nan")
        metrics[f"prob_min_{name}"] = float(np.min(score)) if score.size else float("nan")
        metrics[f"prob_max_{name}"] = float(np.max(score)) if score.size else float("nan")
        positive = score[target > 0.5]
        negative = score[target <= 0.5]
        metrics[f"prob_pos_mean_{name}"] = float(np.mean(positive)) if positive.size else float("nan")
        metrics[f"prob_neg_mean_{name}"] = float(np.mean(negative)) if negative.size else float("nan")
        metrics[f"prob_pos_std_{name}"] = float(np.std(positive)) if positive.size else float("nan")
        metrics[f"prob_neg_std_{name}"] = float(np.std(negative)) if negative.size else float("nan")
        metrics[f"prob_separation_{name}"] = (
            metrics[f"prob_pos_mean_{name}"] - metrics[f"prob_neg_mean_{name}"]
            if positive.size and negative.size else float("nan")
        )
        if not math.isnan(auc):
            aucs.append(auc)
        if not math.isnan(ap):
            aps.append(ap)
    metrics["auc_macro"] = float(np.mean(aucs)) if aucs else float("nan")
    metrics["ap_macro"] = float(np.mean(aps)) if aps else float("nan")
    return metrics


def choose_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str] | tuple[str, ...] = CLASS_NAMES,
) -> dict[str, float]:
    """Operating point of each head, chosen by Youden's index (use validation data)."""
    thresholds: dict[str, float] = {}
    for index, name in enumerate(class_names):
        target = targets[:, index]
        score = probabilities[:, index]
        if np.unique(target).size < 2:
            thresholds[name] = 0.5
            continue
        false_positive, true_positive, values = roc_curve(target, score)
        finite = np.isfinite(values)
        if not finite.any():
            thresholds[name] = 0.5
            continue
        youden = true_positive - false_positive
        youden[~finite] = -np.inf
        threshold = float(values[int(np.argmax(youden))])
        thresholds[name] = float(np.clip(threshold, 0.0, 1.0))
    return thresholds


def detailed_classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    class_names: list[str] | tuple[str, ...] = CLASS_NAMES,
) -> dict[str, float]:
    """AUC/AP plus sensitivity, specificity, F1 and accuracy at fixed thresholds.

    Also reports sensitivity at 90% specificity and specificity at 90% sensitivity.
    """
    names = list(class_names)
    metrics = summarize_probabilities(targets, probabilities, class_names=names)
    sensitivities: list[float] = []
    specificities: list[float] = []
    f1_values: list[float] = []
    accuracy_values: list[float] = []

    for index, name in enumerate(names):
        threshold = float(thresholds[name])
        target = targets[:, index].astype(int)
        prediction = (probabilities[:, index] >= threshold).astype(int)
        matrix = confusion_matrix(target, prediction, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        sensitivity = float(tp / max(tp + fn, 1))
        specificity = float(tn / max(tn + fp, 1))
        f1 = float(f1_score(target, prediction, zero_division=0))
        accuracy = float(accuracy_score(target, prediction))
        metrics[f"threshold_{name}"] = threshold
        metrics[f"sensitivity_{name}"] = sensitivity
        metrics[f"specificity_{name}"] = specificity
        metrics[f"f1_{name}"] = f1
        metrics[f"accuracy_{name}"] = accuracy
        metrics[f"sensitivity_at_spec90_{name}"] = sensitivity_at_specificity(target, probabilities[:, index], 0.90)
        metrics[f"specificity_at_sens90_{name}"] = specificity_at_sensitivity(target, probabilities[:, index], 0.90)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1_values.append(f1)
        accuracy_values.append(accuracy)

    metrics["sensitivity_macro"] = float(np.mean(sensitivities))
    metrics["specificity_macro"] = float(np.mean(specificities))
    metrics["f1_macro"] = float(np.mean(f1_values))
    metrics["accuracy_macro"] = float(np.mean(accuracy_values))
    return metrics


# -----------------------------------------------------------------------------
# Breast- and exam-level aggregation
# -----------------------------------------------------------------------------


def aggregate_views(predictions: pd.DataFrame, level: str) -> pd.DataFrame:
    """Maximum score (and label) over the views of a breast or of an exam.

    Scores are stored as float32, as in the metrics reported by evaluate.py.
    """
    if level == "breast":
        group_columns = ["study_id", "laterality"]
    elif level == "study":
        group_columns = ["study_id"]
    else:
        raise ValueError("level must be 'breast' or 'study'")
    names = [name for name in CLASS_NAMES if f"prob_{name}" in predictions.columns]
    aggregations = {}
    for name in names:
        aggregations[f"target_{name}"] = (f"target_{name}", "max")
        aggregations[f"prob_{name}"] = (f"prob_{name}", "max")
    grouped = predictions.groupby(group_columns, as_index=False).agg(**aggregations)
    for name in names:
        grouped[f"target_{name}"] = grouped[f"target_{name}"].astype(np.float32)
        grouped[f"prob_{name}"] = grouped[f"prob_{name}"].astype(np.float32)
    return grouped


def grouped_probability_metrics(predictions: pd.DataFrame, level: str) -> dict[str, float]:
    """AUC/AP after aggregating the views of each breast ("breast") or exam ("study")."""
    grouped = aggregate_views(predictions, level)
    names = [name for name in CLASS_NAMES if f"prob_{name}" in grouped.columns]
    targets = grouped[[f"target_{name}" for name in names]].to_numpy(dtype=np.float32)
    probabilities = grouped[[f"prob_{name}" for name in names]].to_numpy(dtype=np.float32)
    raw = summarize_probabilities(targets, probabilities, class_names=names)
    return {f"{level}_{key}": value for key, value in raw.items()}


# -----------------------------------------------------------------------------
# Bootstrap over exams (or patients)
# -----------------------------------------------------------------------------


def _resample_rows(groups: np.ndarray, n_boot: int, seed: int):
    """Yield row indices of bootstrap samples that resample whole groups."""
    unique_groups = pd.unique(groups)
    rng = np.random.default_rng(seed)
    rows_by_group = {group: np.where(groups == group)[0] for group in unique_groups}
    for _ in range(n_boot):
        pick = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        yield np.concatenate([rows_by_group[group] for group in pick])


def bootstrap_auc_ci(
    target: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Percentile 95% CI of the AUC; exams (or patients) are resampled with replacement.

    Resamples that contain a single class are skipped.
    """
    target = np.asarray(target)
    score = np.asarray(score)
    values = []
    for rows in _resample_rows(np.asarray(groups), n_boot, seed):
        y, p = target[rows], score[rows]
        if y.min() == y.max():
            continue
        values.append(roc_auc_score(y, p))
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def paired_bootstrap_auc_difference(
    target: np.ndarray,
    score_reference: np.ndarray,
    score_candidate: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    """AUC(candidate) - AUC(reference) with a paired percentile 95% CI.

    Both models are evaluated on the same resampled exams in every replicate.
    """
    target = np.asarray(target)
    reference = np.asarray(score_reference)
    candidate = np.asarray(score_candidate)
    point = float(roc_auc_score(target, candidate) - roc_auc_score(target, reference))
    deltas = []
    for rows in _resample_rows(np.asarray(groups), n_boot, seed):
        y = target[rows]
        if y.min() == y.max():
            continue
        deltas.append(roc_auc_score(y, candidate[rows]) - roc_auc_score(y, reference[rows]))
    array = np.asarray(deltas, dtype=np.float64)
    return point, float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))
