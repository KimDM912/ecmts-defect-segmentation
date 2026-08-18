#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for the thresholding experiment.

This module centralizes:
- repository-relative paths
- validation/test cohort loading
- checkpoint loading
- the exact training U-Net architecture
- letterbox preprocessing and valid-region masking
- threshold selection and metric calculations

Keeping these functions in one module prevents dataset/backbone wrappers and
secondary analyses from silently using different implementations.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from training.train_segmentation import (
        EncoderUNet,
        letterbox_image_and_mask,
    )
except Exception as exc:  # pragma: no cover - environment-specific import
    raise ImportError(
        "Could not import training/train_segmentation.py. "
        "Keep evaluation/ as a direct child of the repository root."
    ) from exc


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "magnetic": {
        "display_name": "Magnetic Tile Defect Dataset",
        "data_root": PROJECT_ROOT / "data" / "magnetic-tile-defect-datasets.-master",
        "cohort_manifest": (
            PROJECT_ROOT
            / "data"
            / "evaluation_manifests"
            / "magnetic_tile_defect_cohorts.json"
        ),
        "fallback_input_height": 512,
        "fallback_input_width": 512,
    },
    "severstal": {
        "display_name": "Severstal Steel Defect Detection",
        "data_root": PROJECT_ROOT / "data" / "severstal-steel-defect-detection",
        "cohort_manifest": (
            PROJECT_ROOT
            / "data"
            / "evaluation_manifests"
            / "severstal_defect_cohorts.json"
        ),
        "fallback_input_height": 256,
        "fallback_input_width": 1600,
    },
    "kolektor": {
        "display_name": "KolektorSDD2",
        "data_root": PROJECT_ROOT / "data" / "kolektor",
        "cohort_manifest": (
            PROJECT_ROOT
            / "data"
            / "evaluation_manifests"
            / "kolektor_defect_cohorts.json"
        ),
        "fallback_input_height": 640,
        "fallback_input_width": 384,
    },
}


BACKBONES = ("resnet18", "resnet34", "efficientnet_b0")
METHODS = ("Fixed", "Validation F2", "Otsu", "Kapur", "ECMTS")
BASELINES = ("Fixed", "Validation F2", "Otsu", "Kapur")

@dataclass(frozen=True)
class ModelRuntime:
    model: torch.nn.Module
    device: torch.device
    backbone: str
    input_height: int
    input_width: int
    checkpoint_config: dict[str, Any]


@dataclass(frozen=True)
class ProbabilityRecord:
    probability: np.ndarray
    ground_truth: np.ndarray
    valid_mask: np.ndarray
    image_path: Path
    mask_path: Path


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dataset_spec(dataset: str) -> dict[str, Any]:
    if dataset not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset: {dataset}")
    return DATASET_SPECS[dataset]


def default_checkpoint(dataset: str, backbone: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "checkpoints"
        / dataset
        / backbone
        / f"seed_{seed}"
        / "best.pth"
    )


def default_result_dir(dataset: str, backbone: str, seed: int) -> Path:
    """Return the default output directory for thresholding results."""
    return (
        PROJECT_ROOT
        / "evaluation"
        / "results_thresholding"
        / dataset
        / backbone
        / f"seed_{seed}"
    )


def item_value(item: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def resolve_existing_path(
    raw_path: str | Path,
    *,
    data_root: Path,
) -> Path:
    candidate = Path(raw_path).expanduser()
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend([PROJECT_ROOT / candidate, data_root / candidate])

    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved

    attempted = "\n  - ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve file: {raw_path}\nAttempted:\n  - {attempted}"
    )


def resolve_mask_path(item: dict[str, Any], data_root: Path) -> Path:
    raw = item_value(item, ("mask", "gt_path", "mask_path"))
    if raw is None:
        raise KeyError("Item has no mask/gt_path field.")
    return resolve_existing_path(raw, data_root=data_root)


def resolve_image_path(item: dict[str, Any], data_root: Path) -> Path:
    raw = item_value(item, ("image", "img_path"))
    if raw is None:
        raise KeyError("Item has no image/img_path field.")
    return resolve_existing_path(raw, data_root=data_root)


def item_identifier(item: dict[str, Any]) -> str:
    return str(
        item_value(
            item,
            (
                "image_id",
                "image",
                "img_path",
                "mask",
                "gt_path",
            ),
        )
        or "unknown"
    )


def deterministic_subset(
    items: Sequence[dict[str, Any]],
    max_images: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_images is None or max_images <= 0 or max_images >= len(items):
        return list(items)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(items), size=max_images, replace=False))
    return [items[int(index)] for index in indices]


# ---------------------------------------------------------------------------
# Cohort manifest parsing
# ---------------------------------------------------------------------------

def cohort_items(
    payload: dict[str, Any],
    subset: str,
) -> list[dict[str, Any]]:
    """Return a non-empty validation or test cohort."""
    if subset not in {"val", "test"}:
        raise ValueError(f"Unsupported cohort subset: {subset!r}")
    value = payload.get(subset)
    if not isinstance(value, list):
        raise KeyError(f"Cohort manifest requires a {subset!r} list.")
    if not value:
        raise RuntimeError(f"Cohort subset {subset!r} is empty.")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError(f"Every item in cohort subset {subset!r} must be an object.")
    return list(value)


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalized[new_key] = value
    return normalized


def load_model_runtime(
    *,
    checkpoint_path: Path,
    dataset: str,
    requested_backbone: str,
    input_height: int | None,
    input_width: int | None,
    device_name: str,
) -> ModelRuntime:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(
        device_name
        if device_name != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = _torch_load(checkpoint_path, device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(config, dict):
        config = {}

    stored_backbone = config.get("backbone")
    backbone = str(stored_backbone or requested_backbone)
    if stored_backbone and requested_backbone != stored_backbone:
        raise ValueError(
            f"Backbone mismatch: requested={requested_backbone}, "
            f"checkpoint={stored_backbone}."
        )

    spec = dataset_spec(dataset)
    height = int(
        input_height
        if input_height is not None
        else config.get("input_height", spec["fallback_input_height"])
    )
    width = int(
        input_width
        if input_width is not None
        else config.get("input_width", spec["fallback_input_width"])
    )
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            f"Input dimensions must be divisible by 32: {(height, width)}"
        )

    model = EncoderUNet(backbone=backbone, pretrained_encoder=False).to(device)
    state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return ModelRuntime(
        model=model,
        device=device,
        backbone=backbone,
        input_height=height,
        input_width=width,
        checkpoint_config=config,
    )


def load_binary_mask(path: Path, threshold: int = 127) -> np.ndarray:
    with Image.open(path) as mask:
        array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return (array > int(threshold)).astype(np.float32)


def normalize_image(image_rgb: np.ndarray) -> torch.Tensor:
    image_chw = image_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    normalized = (image_chw - mean) / std
    return torch.from_numpy(np.ascontiguousarray(normalized)).float()


def prepare_image(
    *,
    image_path: Path,
    original_mask: np.ndarray,
    input_height: int,
    input_width: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if image_rgb.shape[:2] != original_mask.shape:
        raise ValueError(
            f"Image/mask size mismatch: image={image_rgb.shape[:2]}, "
            f"mask={original_mask.shape}, path={image_path}"
        )
    canvas, mask_canvas, valid_canvas = letterbox_image_and_mask(
        image_rgb=image_rgb,
        mask_binary=original_mask,
        output_height=input_height,
        output_width=input_width,
    )
    return normalize_image(canvas), mask_canvas, valid_canvas


@torch.inference_mode()
def infer_prepared_tensor(
    *,
    runtime: ModelRuntime,
    tensor: torch.Tensor,
) -> np.ndarray:
    """Run model inference entirely in FP32.

    Evaluation deliberately avoids mixed precision so that all probability maps
    are generated with one numerically stable and consistent precision policy.
    """
    batch = tensor.unsqueeze(0).to(
        device=runtime.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    if not torch.isfinite(batch).all():
        bad_count = int((~torch.isfinite(batch)).sum().item())
        raise FloatingPointError(
            f"Input tensor contains {bad_count} non-finite values."
        )

    logits = runtime.model(batch)
    if not torch.isfinite(logits).all():
        bad_count = int((~torch.isfinite(logits)).sum().item())
        raise FloatingPointError(
            f"FP32 model logits contain {bad_count} non-finite values."
        )

    probability = torch.sigmoid(logits.float())
    if not torch.isfinite(probability).all():
        bad_count = int((~torch.isfinite(probability)).sum().item())
        raise FloatingPointError(
            f"FP32 probability map contains {bad_count} non-finite values."
        )

    return (
        probability[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


@torch.inference_mode()
def infer_probability_record(
    *,
    runtime: ModelRuntime,
    image_path: Path,
    mask_path: Path,
    mask_threshold: int,
) -> ProbabilityRecord:
    original_mask = load_binary_mask(mask_path, mask_threshold)
    tensor, mask_canvas, valid_canvas = prepare_image(
        image_path=image_path,
        original_mask=original_mask,
        input_height=runtime.input_height,
        input_width=runtime.input_width,
    )
    probability = infer_prepared_tensor(
        runtime=runtime,
        tensor=tensor,
    )
    return ProbabilityRecord(
        probability=probability,
        ground_truth=mask_canvas.astype(np.float32),
        valid_mask=valid_canvas.astype(np.float32),
        image_path=image_path,
        mask_path=mask_path,
    )


def valid_probability_and_truth(
    probability: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = valid_mask > 0.5
    if not np.any(valid):
        raise RuntimeError("The valid-region mask contains no valid pixels.")

    p = probability[valid].astype(np.float64)
    if not np.isfinite(p).all():
        bad_count = int((~np.isfinite(p)).sum())
        raise FloatingPointError(
            f"Valid probability values contain {bad_count} NaN/Inf entries."
        )

    y_raw = ground_truth[valid].astype(np.float64)
    if not np.isfinite(y_raw).all():
        bad_count = int((~np.isfinite(y_raw)).sum())
        raise FloatingPointError(
            f"Valid ground-truth values contain {bad_count} NaN/Inf entries."
        )

    p = np.clip(p, 0.0, 1.0)
    y = (y_raw > 0.5).astype(np.uint8)
    return p, y


# ---------------------------------------------------------------------------
# Thresholding and metrics
# ---------------------------------------------------------------------------

def fbeta_from_counts(
    tp: float,
    fp: float,
    fn: float,
    beta: float = 2.0,
) -> float:
    beta_squared = float(beta) ** 2
    denominator = (
        (1.0 + beta_squared) * tp
        + beta_squared * fn
        + fp
    )
    if denominator <= 0:
        return 0.0
    return float((1.0 + beta_squared) * tp / denominator)


def metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    beta: float = 2.0,
) -> dict[str, Any]:
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    f2 = fbeta_from_counts(tp, fp, fn, beta=beta)
    iou = tp / (tp + fp + fn + eps)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1_dice": float(f1),
        "f2": float(f2),
        "iou": float(iou),
    }


def confusion_at_threshold(
    probabilities: np.ndarray,
    truth: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    prediction = probabilities >= float(threshold)
    positive = truth > 0
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & ~positive))
    fn = int(np.sum(~prediction & positive))
    tn = int(np.sum(~prediction & ~positive))
    return tp, fp, fn, tn


def otsu_threshold(probabilities: np.ndarray, bins: int = 256) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.5
    values = np.clip(values, 0.0, 1.0)
    if float(values.max() - values.min()) < 1e-12:
        return float(np.median(values))

    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    if histogram.sum() <= 0:
        return float(np.median(values))

    pmf = histogram / histogram.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(pmf)
    mu = np.cumsum(pmf * centers)
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    between = np.full_like(denominator, np.nan)
    valid = denominator > 1e-12
    between[valid] = (
        (mu_total * omega[valid] - mu[valid]) ** 2
        / denominator[valid]
    )
    if not np.isfinite(between).any():
        return float(np.median(values))
    index = int(np.nanargmax(between))
    return float(np.clip(centers[index], 0.0, 1.0))


def kapur_threshold(probabilities: np.ndarray, bins: int = 256) -> float:
    """Select a threshold using Kapur's maximum-entropy criterion.

    The valid probability values are quantized into ``bins`` histogram bins.
    For every admissible split, the foreground and background entropies are
    computed from their normalized class histograms, and the split maximizing
    the sum of the two entropies is selected.

    Degenerate inputs for which no two-class split exists fall back to the
    median probability.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.5
    values = np.clip(values, 0.0, 1.0)
    if float(values.max() - values.min()) < 1e-12:
        return float(np.median(values))

    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return float(np.median(values))

    pmf = histogram / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega_background = np.cumsum(pmf)
    omega_foreground = 1.0 - omega_background

    p_log_p = np.zeros_like(pmf)
    positive = pmf > 0
    p_log_p[positive] = pmf[positive] * np.log(pmf[positive])
    cumulative_p_log_p = np.cumsum(p_log_p)
    total_p_log_p = cumulative_p_log_p[-1]

    criterion = np.full_like(pmf, np.nan, dtype=np.float64)
    valid = (omega_background > 1e-12) & (omega_foreground > 1e-12)
    if np.any(valid):
        background_entropy = (
            np.log(omega_background[valid])
            - cumulative_p_log_p[valid] / omega_background[valid]
        )
        foreground_entropy = (
            np.log(omega_foreground[valid])
            - (total_p_log_p - cumulative_p_log_p[valid])
            / omega_foreground[valid]
        )
        criterion[valid] = background_entropy + foreground_entropy

    if not np.isfinite(criterion).any():
        return float(np.median(values))
    index = int(np.nanargmax(criterion))
    return float(np.clip(centers[index], 0.0, 1.0))


def _sorted_prefix(
    probabilities: np.ndarray,
    truth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    order = np.argsort(probabilities, kind="mergesort")
    sorted_p = probabilities[order].astype(np.float64, copy=False)
    prefix_p = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(sorted_p, dtype=np.float64)]
    )
    prefix_y: np.ndarray | None = None
    if truth is not None:
        sorted_y = truth[order].astype(np.float64, copy=False)
        prefix_y = np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(sorted_y, dtype=np.float64)]
        )
    return sorted_p, prefix_p, prefix_y


def expected_f2_curve(
    probabilities: np.ndarray,
    threshold_grid: np.ndarray,
    beta: float = 2.0,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    sorted_p, prefix_p, _ = _sorted_prefix(probabilities)
    indices = np.searchsorted(sorted_p, threshold_grid, side="left")
    n = sorted_p.size
    total_p = prefix_p[-1]
    selected_count = n - indices
    tp = total_p - prefix_p[indices]
    fn = prefix_p[indices]
    fp = selected_count.astype(np.float64) - tp
    beta_squared = float(beta) ** 2
    denominator = (
        (1.0 + beta_squared) * tp
        + beta_squared * fn
        + fp
    )
    scores = np.zeros_like(threshold_grid, dtype=np.float64)
    valid = denominator > 0
    scores[valid] = (1.0 + beta_squared) * tp[valid] / denominator[valid]
    return scores


def true_confusion_curves(
    probabilities: np.ndarray,
    truth: np.ndarray,
    threshold_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.uint8)
    sorted_p, _, prefix_y = _sorted_prefix(probabilities, truth)
    assert prefix_y is not None
    indices = np.searchsorted(sorted_p, threshold_grid, side="left")
    n = sorted_p.size
    positives = prefix_y[-1]
    tp = positives - prefix_y[indices]
    fn = prefix_y[indices]
    predicted_positive = n - indices
    fp = predicted_positive.astype(np.float64) - tp
    tn = (n - positives) - fp
    return tp, fp, fn, tn


def true_f2_curve(
    probabilities: np.ndarray,
    truth: np.ndarray,
    threshold_grid: np.ndarray,
    beta: float = 2.0,
) -> np.ndarray:
    tp, fp, fn, _ = true_confusion_curves(probabilities, truth, threshold_grid)
    beta_squared = float(beta) ** 2
    denominator = (
        (1.0 + beta_squared) * tp
        + beta_squared * fn
        + fp
    )
    scores = np.zeros_like(threshold_grid, dtype=np.float64)
    valid = denominator > 0
    scores[valid] = (1.0 + beta_squared) * tp[valid] / denominator[valid]
    return scores



def expected_and_true_f2_curves(
    probabilities: np.ndarray,
    truth: np.ndarray,
    threshold_grid: np.ndarray,
    beta: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute expected and true F2 curves with one probability sort."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.uint8)
    order = np.argsort(probabilities, kind="mergesort")
    sorted_p = probabilities[order]
    sorted_y = truth[order].astype(np.float64, copy=False)

    prefix_p = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(sorted_p, dtype=np.float64)]
    )
    prefix_y = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(sorted_y, dtype=np.float64)]
    )
    indices = np.searchsorted(sorted_p, threshold_grid, side="left")
    n = sorted_p.size
    selected_count = n - indices

    total_probability = prefix_p[-1]
    expected_tp = total_probability - prefix_p[indices]
    expected_fn = prefix_p[indices]
    expected_fp = selected_count.astype(np.float64) - expected_tp

    total_positive = prefix_y[-1]
    true_tp = total_positive - prefix_y[indices]
    true_fn = prefix_y[indices]
    true_fp = selected_count.astype(np.float64) - true_tp

    beta_squared = float(beta) ** 2
    expected_denominator = (
        (1.0 + beta_squared) * expected_tp
        + beta_squared * expected_fn
        + expected_fp
    )
    true_denominator = (
        (1.0 + beta_squared) * true_tp
        + beta_squared * true_fn
        + true_fp
    )

    expected_scores = np.zeros_like(threshold_grid, dtype=np.float64)
    true_scores = np.zeros_like(threshold_grid, dtype=np.float64)
    expected_valid = expected_denominator > 0
    true_valid = true_denominator > 0
    expected_scores[expected_valid] = (
        (1.0 + beta_squared) * expected_tp[expected_valid]
        / expected_denominator[expected_valid]
    )
    true_scores[true_valid] = (
        (1.0 + beta_squared) * true_tp[true_valid]
        / true_denominator[true_valid]
    )
    return expected_scores, true_scores

def first_argmax(values: np.ndarray) -> int:
    finite = np.isfinite(values)
    if not finite.any():
        return 0
    maximum = np.nanmax(values)
    return int(np.flatnonzero(values == maximum)[0])


def proposed_threshold(
    probabilities: np.ndarray,
    threshold_grid: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, np.ndarray, int]:
    curve = expected_f2_curve(probabilities, threshold_grid, beta=beta)
    index = first_argmax(curve)
    return float(threshold_grid[index]), curve, index


def oracle_threshold(
    probabilities: np.ndarray,
    truth: np.ndarray,
    threshold_grid: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, np.ndarray, int]:
    curve = true_f2_curve(probabilities, truth, threshold_grid, beta=beta)
    index = first_argmax(curve)
    return float(threshold_grid[index]), curve, index


def brier_score(probabilities: np.ndarray, truth: np.ndarray) -> float:
    if probabilities.size == 0:
        return float("nan")
    return float(np.mean((probabilities - truth.astype(np.float64)) ** 2))


def class_separation(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    positive = probabilities[truth > 0]
    negative = probabilities[truth == 0]
    positive_mean = float(np.mean(positive)) if positive.size else float("nan")
    negative_mean = float(np.mean(negative)) if negative.size else float("nan")
    separation = (
        positive_mean - negative_mean
        if np.isfinite(positive_mean) and np.isfinite(negative_mean)
        else float("nan")
    )
    return {
        "defect_mean_probability": positive_mean,
        "normal_mean_probability": negative_mean,
        "class_separation": float(separation),
        "n_defect_pixels": int(positive.size),
        "n_normal_pixels": int(negative.size),
    }


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    return pearson_correlation(average_ranks(x), average_ranks(y))


def aggregate_confusions(
    rows: Iterable[dict[str, Any]],
    beta: float = 2.0,
) -> dict[str, Any]:
    """Aggregate image-wise metrics without pooling pixels across images.

    The primary reported values are macro averages over images. Summed confusion
    counts are retained only for auditing and are not used to compute the macro
    precision, recall, or F-scores.
    """
    rows = list(rows)
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)

    metric_names = ("precision", "recall", "f1_dice", "f2", "iou")
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in metric_names
    }

    result: dict[str, Any] = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_images": int(len(rows)),
        "aggregation": "image-wise macro",
    }

    for name, array in values.items():
        mean_value = float(np.mean(array)) if array.size else float("nan")
        std_value = (
            float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        )
        median_value = float(np.median(array)) if array.size else float("nan")
        result[name] = mean_value
        result[f"{name}_std"] = std_value
        result[f"{name}_median"] = median_value

    # Backward-compatible aliases used by existing analysis scripts.
    result["macro_precision_mean"] = result["precision"]
    result["macro_precision_std"] = result["precision_std"]
    result["macro_recall_mean"] = result["recall"]
    result["macro_recall_std"] = result["recall_std"]
    result["macro_f2_mean"] = result["f2"]
    result["macro_f2_std"] = result["f2_std"]
    result["macro_f2_median"] = result["f2_median"]
    return result

