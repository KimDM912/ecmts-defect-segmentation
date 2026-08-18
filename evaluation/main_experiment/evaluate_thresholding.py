#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the primary thresholding experiment.

Methods
-------
- Fixed: threshold 0.5
- Validation F2: one global threshold selected on the defect-only validation
  cohort by maximizing mean image-wise F2
- Otsu: image-wise between-class variance thresholding
- Kapur: image-wise maximum-entropy thresholding
- ECMTS: image-wise threshold maximizing plug-in expected F2

The cohort manifest must contain two disjoint lists:
- ``val``: defect-containing images used only to select the global Validation-F2
  threshold
- ``test``: defect-containing images used only for final method comparison

All metrics are calculated inside the same valid region used by the segmentation
preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation_common import (
    BACKBONES,
    METHODS,
    aggregate_confusions,
    brier_score,
    class_separation,
    cohort_items,
    confusion_at_threshold,
    dataset_spec,
    default_checkpoint,
    default_result_dir,
    deterministic_subset,
    expected_and_true_f2_curves,
    first_argmax,
    infer_probability_record,
    item_identifier,
    kapur_threshold,
    load_json,
    load_model_runtime,
    metrics_from_counts,
    otsu_threshold,
    pearson_correlation,
    resolve_image_path,
    resolve_mask_path,
    save_json,
    set_seed,
    spearman_correlation,
    true_f2_curve,
    valid_probability_and_truth,
    write_csv,
)


F_BETA = 2.0
FIXED_THRESHOLD = 0.5
THRESHOLD_GRID_SIZE = 2001
HISTOGRAM_BINS = 256
MASK_THRESHOLD = 127


def parse_args(
    default_dataset: str = "severstal",
    default_backbone: str = "resnet18",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Fixed 0.5, Validation-F2, Otsu, Kapur, and the ECMTS "
            "expected-confusion threshold on defect cohorts."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("magnetic", "severstal", "kolektor"),
        default=default_dataset,
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=default_backbone,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Defect-cohort JSON containing 'val' and 'test'. "
            "The dataset-specific path is used by default."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--max-validation-images",
        type=int,
        default=None,
        help="Deterministic validation subset for a smoke test only.",
    )
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=None,
        help="Deterministic test subset for a smoke test only.",
    )
    return parser.parse_args()


def validate_manifest_preprocessing(
    payload: dict[str, Any],
    *,
    runtime_height: int,
    runtime_width: int,
) -> None:
    """Validate preprocessing compatibility recorded in the cohort manifest."""
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return

    preprocessing = metadata.get("evaluation_preprocessing", {})
    if not isinstance(preprocessing, dict):
        return

    manifest_height = preprocessing.get("input_height")
    manifest_width = preprocessing.get("input_width")
    if manifest_height is not None and int(manifest_height) != runtime_height:
        raise ValueError(
            "Input-height mismatch between cohort generation and evaluation: "
            f"manifest={manifest_height}, runtime={runtime_height}"
        )
    if manifest_width is not None and int(manifest_width) != runtime_width:
        raise ValueError(
            "Input-width mismatch between cohort generation and evaluation: "
            f"manifest={manifest_width}, runtime={runtime_width}"
        )


def validate_disjoint_cohorts(
    val_items: list[dict[str, Any]],
    test_items: list[dict[str, Any]],
) -> None:
    val_ids = {item_identifier(item) for item in val_items}
    test_ids = {item_identifier(item) for item in test_items}
    overlap = sorted(val_ids & test_ids)
    if overlap:
        raise RuntimeError(
            "Validation/test cohort overlap detected. Examples: "
            + ", ".join(overlap[:5])
        )


def ensure_defect_truth(
    truth: np.ndarray,
    *,
    subset: str,
    image_id: str,
) -> None:
    if int(np.count_nonzero(truth > 0)) <= 0:
        raise RuntimeError(
            f"{subset} cohort contains an image with no positive GT pixel "
            f"inside the evaluation valid region: {image_id}"
        )


def select_validation_f2_threshold(
    *,
    runtime: Any,
    val_items: list[dict[str, Any]],
    data_root: Path,
    threshold_grid: np.ndarray,
) -> tuple[float, int, np.ndarray, float]:
    """Select one global threshold by mean image-wise validation F2."""
    curve_sum = np.zeros_like(threshold_grid, dtype=np.float64)
    start = time.time()

    for image_index, item in enumerate(val_items, start=1):
        image_id = item_identifier(item)
        image_path = resolve_image_path(item, data_root)
        mask_path = resolve_mask_path(item, data_root)

        record = infer_probability_record(
            runtime=runtime,
            image_path=image_path,
            mask_path=mask_path,
            mask_threshold=MASK_THRESHOLD,
        )
        probabilities, truth = valid_probability_and_truth(
            record.probability,
            record.ground_truth,
            record.valid_mask,
        )
        ensure_defect_truth(truth, subset="Validation", image_id=image_id)

        curve_sum += true_f2_curve(
            probabilities,
            truth,
            threshold_grid,
            beta=F_BETA,
        )

        if (
            image_index == 1
            or image_index % 10 == 0
            or image_index == len(val_items)
        ):
            print(
                f"[VAL  {image_index:>5}/{len(val_items)}] "
                f"elapsed={(time.time() - start) / 60.0:.1f} min "
                f"last={image_id}",
                flush=True,
            )

    mean_curve = curve_sum / float(len(val_items))
    selected_index = first_argmax(mean_curve)
    selected_threshold = float(threshold_grid[selected_index])
    elapsed = time.time() - start
    return selected_threshold, selected_index, mean_curve, elapsed


def build_validation_curve_rows(
    threshold_grid: np.ndarray,
    mean_curve: np.ndarray,
    selected_index: int,
) -> list[dict[str, Any]]:
    return [
        {
            "threshold_index": int(index),
            "threshold": float(threshold),
            "mean_image_wise_f2": float(mean_curve[index]),
            "selected": int(index == selected_index),
        }
        for index, threshold in enumerate(threshold_grid)
    ]


def build_aggregate_rows(
    per_image_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        groups[str(row["method"])].append(row)

    result: list[dict[str, Any]] = []
    method_order = {method: index for index, method in enumerate(METHODS)}
    for method in sorted(groups, key=lambda value: method_order.get(value, 10_000)):
        rows = groups[method]
        aggregate = aggregate_confusions(rows, beta=F_BETA)
        first = rows[0]
        result.append(
            {
                "dataset": first["dataset"],
                "backbone": first["backbone"],
                "method": method,
                **aggregate,
            }
        )
    return result


def main(
    default_dataset: str = "severstal",
    default_backbone: str = "resnet18",
) -> None:
    print("[START] Thresholding experiment", flush=True)
    args = parse_args(default_dataset, default_backbone)
    print(
        f"[REQUEST] dataset={args.dataset}, backbone={args.backbone}",
        flush=True,
    )
    set_seed(args.seed)

    spec = dataset_spec(args.dataset)
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else Path(spec["data_root"]).resolve()
    )
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else Path(spec["cohort_manifest"]).resolve()
    )
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else default_checkpoint(args.dataset, args.backbone, args.seed).resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_result_dir(args.dataset, args.backbone, args.seed).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_payload = load_json(manifest_path)
    val_all = cohort_items(cohort_payload, "val")
    test_all = cohort_items(cohort_payload, "test")
    validate_disjoint_cohorts(val_all, test_all)

    val_items = deterministic_subset(
        val_all,
        max_images=args.max_validation_images,
        seed=args.seed + 101,
    )
    test_items = deterministic_subset(
        test_all,
        max_images=args.max_test_images,
        seed=args.seed + 202,
    )

    runtime = load_model_runtime(
        checkpoint_path=checkpoint_path,
        dataset=args.dataset,
        requested_backbone=args.backbone,
        input_height=args.input_height,
        input_width=args.input_width,
        device_name=args.device,
    )
    validate_manifest_preprocessing(
        cohort_payload,
        runtime_height=runtime.input_height,
        runtime_width=runtime.input_width,
    )

    threshold_grid = np.linspace(
        0.0,
        1.0,
        THRESHOLD_GRID_SIZE,
        dtype=np.float64,
    )

    print("=" * 100)
    print(f"[DATASET] {spec['display_name']}")
    print(f"[BACKBONE] {runtime.backbone}")
    print(f"[DATA ROOT] {data_root}")
    print(f"[COHORT MANIFEST] {manifest_path}")
    print(f"[CHECKPOINT] {checkpoint_path}")
    print(f"[INPUT] {runtime.input_height}x{runtime.input_width}")
    print(f"[DEVICE] {runtime.device}")
    print("[INFERENCE PRECISION] FP32")
    print(f"[VALIDATION] {len(val_items)} / {len(val_all)} defect images")
    print(f"[TEST] {len(test_items)} / {len(test_all)} defect images")
    print(f"[METHODS] {', '.join(METHODS)}")
    print(f"[THRESHOLD GRID] {THRESHOLD_GRID_SIZE} points, step=0.0005")
    print(f"[OUTPUT] {output_dir}")
    print("=" * 100)

    print(
        "[1/3] Selecting one global Validation-F2 threshold on validation images ...",
        flush=True,
    )
    (
        validation_threshold,
        validation_index,
        validation_mean_curve,
        validation_elapsed,
    ) = select_validation_f2_threshold(
        runtime=runtime,
        val_items=val_items,
        data_root=data_root,
        threshold_grid=threshold_grid,
    )
    print(
        "[VALIDATION SELECTED] "
        f"threshold={validation_threshold:.4f}, "
        f"mean_image_wise_f2={validation_mean_curve[validation_index]:.6f}",
        flush=True,
    )

    validation_curve_rows = build_validation_curve_rows(
        threshold_grid,
        validation_mean_curve,
        validation_index,
    )
    write_csv(output_dir / "validation_f2_curve.csv", validation_curve_rows)
    validation_selection = {
        "dataset": args.dataset,
        "dataset_display_name": spec["display_name"],
        "backbone": runtime.backbone,
        "seed": args.seed,
        "selection_subset": "val",
        "selection_cohort": "defect-containing images retaining positive GT in valid region",
        "selection_metric": "mean image-wise F2",
        "beta": F_BETA,
        "n_validation_images": len(val_items),
        "n_validation_images_available": len(val_all),
        "threshold_grid_size": THRESHOLD_GRID_SIZE,
        "threshold_grid_step": 1.0 / (THRESHOLD_GRID_SIZE - 1),
        "tie_breaking": "lowest threshold among tied maxima",
        "selected_threshold_index": validation_index,
        "selected_threshold": validation_threshold,
        "selected_mean_image_wise_f2": float(
            validation_mean_curve[validation_index]
        ),
        "elapsed_seconds": validation_elapsed,
        "test_ground_truth_used": False,
    }
    save_json(
        output_dir / "validation_threshold_selection.json",
        validation_selection,
    )

    print("[2/3] Evaluating the five methods on test images ...", flush=True)
    per_image_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    test_start = time.time()

    for image_index, item in enumerate(test_items, start=1):
        image_id = item_identifier(item)
        image_path = resolve_image_path(item, data_root)
        mask_path = resolve_mask_path(item, data_root)

        record = infer_probability_record(
            runtime=runtime,
            image_path=image_path,
            mask_path=mask_path,
            mask_threshold=MASK_THRESHOLD,
        )
        probabilities, truth = valid_probability_and_truth(
            record.probability,
            record.ground_truth,
            record.valid_mask,
        )
        ensure_defect_truth(truth, subset="Test", image_id=image_id)

        expected_curve, true_curve = expected_and_true_f2_curves(
            probabilities,
            truth,
            threshold_grid,
            beta=F_BETA,
        )
        proposed_index = first_argmax(expected_curve)
        oracle_index = first_argmax(true_curve)

        tau_proposed = float(threshold_grid[proposed_index])
        tau_oracle = float(threshold_grid[oracle_index])
        tau_otsu = otsu_threshold(probabilities, bins=HISTOGRAM_BINS)
        tau_kapur = kapur_threshold(probabilities, bins=HISTOGRAM_BINS)

        threshold_map = {
            "Fixed": FIXED_THRESHOLD,
            "Validation F2": validation_threshold,
            "Otsu": tau_otsu,
            "Kapur": tau_kapur,
            "ECMTS": tau_proposed,
        }

        for method in METHODS:
            threshold = float(threshold_map[method])
            tp, fp, fn, tn = confusion_at_threshold(
                probabilities,
                truth,
                threshold,
            )
            metrics = metrics_from_counts(tp, fp, fn, tn, beta=F_BETA)
            row = {
                "dataset": args.dataset,
                "dataset_display_name": spec["display_name"],
                "backbone": runtime.backbone,
                "seed": args.seed,
                "image_index": image_index - 1,
                "image_id": image_id,
                "method": method,
                "threshold": threshold,
                "threshold_scope": (
                    "global"
                    if method in {"Fixed", "Validation F2"}
                    else "image-wise"
                ),
                "threshold_uses_ground_truth": int(method == "Validation F2"),
                "ground_truth_usage": (
                    "validation only"
                    if method == "Validation F2"
                    else "none for threshold selection"
                ),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "n_valid_pixels": int(probabilities.size),
                **metrics,
            }
            per_image_rows.append(row)
            threshold_rows.append(
                {
                    key: row[key]
                    for key in (
                        "dataset",
                        "backbone",
                        "seed",
                        "image_index",
                        "image_id",
                        "method",
                        "threshold",
                        "threshold_scope",
                        "threshold_uses_ground_truth",
                        "ground_truth_usage",
                    )
                }
            )

        separation = class_separation(probabilities, truth)
        true_at_proposed = float(true_curve[proposed_index])
        oracle_true = float(true_curve[oracle_index])
        expected_at_proposed = float(expected_curve[proposed_index])
        expected_at_oracle = float(expected_curve[oracle_index])
        defect_ratio = float(np.mean(truth.astype(np.float64)))
        mean_probability = float(np.mean(probabilities))

        diagnostic_rows.append(
            {
                "dataset": args.dataset,
                "dataset_display_name": spec["display_name"],
                "backbone": runtime.backbone,
                "seed": args.seed,
                "image_index": image_index - 1,
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "n_valid_pixels": int(probabilities.size),
                "defect_ratio": defect_ratio,
                "mean_probability_mass": mean_probability,
                "probability_mass_bias": mean_probability - defect_ratio,
                **separation,
                "brier_score": brier_score(probabilities, truth),
                "expected_true_pearson": pearson_correlation(
                    expected_curve,
                    true_curve,
                ),
                "expected_true_spearman": spearman_correlation(
                    expected_curve,
                    true_curve,
                ),
                "fixed_threshold": FIXED_THRESHOLD,
                "validation_f2_threshold": validation_threshold,
                "otsu_threshold": tau_otsu,
                "kapur_threshold": tau_kapur,
                "ECMTS_threshold": tau_proposed,
                "oracle_threshold": tau_oracle,
                "absolute_threshold_gap": abs(tau_proposed - tau_oracle),
                "expected_f2_at_ECMTS": expected_at_proposed,
                "expected_f2_at_oracle": expected_at_oracle,
                "true_f2_at_ECMTS": true_at_proposed,
                "oracle_true_f2": oracle_true,
                "oracle_regret": max(0.0, oracle_true - true_at_proposed),
                "threshold_grid_size": THRESHOLD_GRID_SIZE,
            }
        )

        if (
            image_index == 1
            or image_index % 10 == 0
            or image_index == len(test_items)
        ):
            print(
                f"[TEST {image_index:>5}/{len(test_items)}] "
                f"elapsed={(time.time() - test_start) / 60.0:.1f} min "
                f"last={image_id}",
                flush=True,
            )

    print("[3/3] Writing canonical outputs ...", flush=True)
    aggregate_rows = build_aggregate_rows(per_image_rows)

    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "selected_thresholds.csv", threshold_rows)
    write_csv(output_dir / "diagnostics.csv", diagnostic_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)

    test_elapsed = time.time() - test_start
    metadata = {
        "analysis": "main_thresholding_experiment",
        "dataset": args.dataset,
        "dataset_display_name": spec["display_name"],
        "backbone": runtime.backbone,
        "seed": args.seed,
        "data_root": str(data_root),
        "cohort_manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "input_height": runtime.input_height,
        "input_width": runtime.input_width,
        "device": str(runtime.device),
        "inference_precision": "float32",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "valid_region_only": True,
        "validation_cohort_defect_only": True,
        "test_cohort_defect_only": True,
        "n_validation_images": len(val_items),
        "n_validation_images_available": len(val_all),
        "n_test_images": len(test_items),
        "n_test_images_available": len(test_all),
        "n_per_image_metric_rows": len(per_image_rows),
        "n_diagnostic_rows": len(diagnostic_rows),
        "methods": list(METHODS),
        "method_configuration": {
            "Fixed": {
                "scope": "global fixed",
                "threshold": FIXED_THRESHOLD,
                "ground_truth_for_selection": "none",
            },
            "Validation F2": {
                "scope": "global validation-tuned",
                "threshold": validation_threshold,
                "ground_truth_for_selection": "validation cohort only",
                "objective": "mean image-wise F2",
            },
            "Otsu": {
                "scope": "image-wise",
                "ground_truth_for_selection": "none",
                "objective": "between-class histogram variance",
            },
            "Kapur": {
                "scope": "image-wise",
                "ground_truth_for_selection": "none",
                "objective": "sum of class histogram entropies",
            },
            "ECMTS": {
                "scope": "image-wise",
                "ground_truth_for_selection": "none",
                "objective": "plug-in expected F2",
            },
        },
        "test_ground_truth_used_for_threshold_selection": False,
        "oracle_used_for_diagnostics_only": True,
        "fixed_threshold": FIXED_THRESHOLD,
        "validation_f2_threshold": validation_threshold,
        "histogram_bins": HISTOGRAM_BINS,
        "threshold_grid_size": THRESHOLD_GRID_SIZE,
        "threshold_grid_step": 1.0 / (THRESHOLD_GRID_SIZE - 1),
        "f_beta": F_BETA,
        "primary_aggregation": "image-wise macro",
        "elapsed_seconds_validation_selection": validation_elapsed,
        "elapsed_seconds_test_evaluation": test_elapsed,
    }
    save_json(output_dir / "run_metadata.json", metadata)

    print("=" * 100)
    print("[OK] Thresholding experiment completed.")
    print(
        f"[VALIDATION THRESHOLD] {validation_threshold:.4f} "
        f"(mean F2={validation_mean_curve[validation_index]:.6f})"
    )
    for row in aggregate_rows:
        print(
            f"[RESULT] {row['method']:<14} "
            f"Precision={row['precision']:.6f} "
            f"Recall={row['recall']:.6f} "
            f"F2={row['f2']:.6f}"
        )
    print(f"[VAL CURVE] {output_dir / 'validation_f2_curve.csv'}")
    print(
        "[VAL SELECTION] "
        f"{output_dir / 'validation_threshold_selection.json'}"
    )
    print(f"[PER IMAGE] {output_dir / 'per_image_metrics.csv'}")
    print(f"[THRESHOLDS] {output_dir / 'selected_thresholds.csv'}")
    print(f"[DIAGNOSTICS] {output_dir / 'diagnostics.csv'}")
    print(f"[AGGREGATE] {output_dir / 'aggregate_metrics.csv'}")
    print(f"[METADATA] {output_dir / 'run_metadata.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
