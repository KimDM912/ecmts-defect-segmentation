#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark runtime for every dataset--backbone combination.

Default experiment
------------------
- 3 datasets x 3 backbones = 9 combinations
- exactly 50 deterministically sampled test images per combination
- repeated timing within each image; the median repeat is retained
- mean +/- sample standard deviation is then calculated across the 50 images

Image decoding, mask loading, letterbox resizing, and normalization are completed
before each timed inference call. Inference timing includes tensor transfer to the
selected device, forward propagation, sigmoid conversion, and probability-map
transfer to CPU.

Threshold timing reports only the time required to select a threshold.
Binary-mask generation is excluded. Fixed and Validation F2 use already selected
global thresholds, so their online threshold-selection time is defined as zero.
"""

from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common.result_analysis_common import (  # noqa: E402
    BACKBONES,
    DATASETS,
    METHODS,
    analysis_result_dir,
    evaluation_result_dir,
    read_json,
    save_json,
    write_csv,
)
from evaluation.main_experiment.evaluation_common import (  # noqa: E402
    cohort_items,
    dataset_spec,
    default_checkpoint,
    deterministic_subset,
    infer_prepared_tensor,
    item_identifier,
    kapur_threshold,
    load_binary_mask,
    load_json,
    load_model_runtime,
    otsu_threshold,
    prepare_image,
    proposed_threshold,
    resolve_image_path,
    resolve_mask_path,
    valid_probability_and_truth,
)


F_BETA = 2.0
FIXED_THRESHOLD = 0.5
THRESHOLD_GRID_SIZE = 2001
HISTOGRAM_BINS = 256
MASK_THRESHOLD = 127

RUNTIME_METRICS = (
    "model_inference_ms",
    "threshold_selection_ms",
    "inference_plus_threshold_selection_ms",
)


@dataclass(frozen=True)
class BenchmarkJob:
    dataset: str
    backbone: str
    data_root: Path
    manifest_path: Path
    checkpoint_path: Path
    evaluation_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to benchmark. Default: all datasets.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=BACKBONES,
        default=list(BACKBONES),
        help="Backbones to benchmark. Default: all backbones.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--images",
        type=int,
        default=50,
        help=(
            "Exact number of test images benchmarked for every combination. "
            "The script stops if a dataset has fewer images. Default: 50."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=2026,
        help=(
            "Seed used only to select the benchmark image subset. The same "
            "50 images are used across backbones within each dataset."
        ),
    )
    parser.add_argument("--model-warmup", type=int, default=10)
    parser.add_argument("--postprocess-warmup", type=int, default=2)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help=(
            "Timing repetitions per image and operation. The per-image median "
            "is retained before the 50-image mean and standard deviation."
        ),
    )
    return parser.parse_args()


def default_output_root(seed: int) -> Path:
    """Place the combined result beside the existing per-combination runtime tree."""
    example = analysis_result_dir(
        "runtime",
        str(DATASETS[0]),
        str(BACKBONES[0]),
        seed,
    ).resolve()
    try:
        runtime_root = example.parents[2]
    except IndexError:
        runtime_root = PROJECT_ROOT / "analysis" / "results" / "runtime"
    return runtime_root / f"seed_{seed}"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def milliseconds(
    function: Callable[[], Any],
    repeats: int,
    *,
    device: torch.device | None = None,
) -> tuple[float, Any]:
    """Return the median elapsed milliseconds and the final function result."""
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    elapsed: list[float] = []
    result: Any = None
    for _ in range(repeats):
        if device is not None:
            synchronize(device)
        start = time.perf_counter_ns()
        result = function()
        if device is not None:
            synchronize(device)
        end = time.perf_counter_ns()
        elapsed.append((end - start) / 1_000_000.0)
    return float(np.median(elapsed)), result


def summary_statistics(values: Iterable[float]) -> dict[str, float | int | str]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty runtime sequence.")
    if not np.isfinite(array).all():
        raise FloatingPointError("Runtime values contain NaN or infinity.")

    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    median = float(np.median(array))
    return {
        "n_images": int(array.size),
        "mean": mean,
        "std": std,
        "median": median,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean_plus_minus_std": f"{mean:.4f} ± {std:.4f}",
    }


def load_validation_threshold(evaluation_dir: Path) -> tuple[float, str]:
    path = evaluation_dir / "validation_threshold_selection.json"
    payload = read_json(path)
    if "selected_threshold" not in payload:
        raise KeyError(f"selected_threshold is missing from {path}")

    threshold = float(payload["selected_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Validation F2 threshold must be in [0, 1]: {threshold}")
    return threshold, str(path)


def build_jobs(args: argparse.Namespace) -> list[BenchmarkJob]:
    jobs: list[BenchmarkJob] = []
    for dataset in args.datasets:
        spec = dataset_spec(dataset)
        data_root = Path(spec["data_root"]).resolve()
        manifest_path = Path(spec["cohort_manifest"]).resolve()
        for backbone in args.backbones:
            jobs.append(
                BenchmarkJob(
                    dataset=str(dataset),
                    backbone=str(backbone),
                    data_root=data_root,
                    manifest_path=manifest_path,
                    checkpoint_path=default_checkpoint(
                        str(dataset),
                        str(backbone),
                        args.seed,
                    ).resolve(),
                    evaluation_dir=evaluation_result_dir(
                        str(dataset),
                        str(backbone),
                        args.seed,
                    ).resolve(),
                )
            )
    return jobs


def preflight(jobs: list[BenchmarkJob], args: argparse.Namespace) -> None:
    if args.images <= 0:
        raise ValueError("--images must be positive.")
    if args.model_warmup < 0 or args.postprocess_warmup < 0:
        raise ValueError("Warm-up counts cannot be negative.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    errors: list[str] = []
    checked_datasets: set[str] = set()

    for job in jobs:
        if not job.checkpoint_path.is_file():
            errors.append(f"Missing checkpoint: {job.checkpoint_path}")
        validation_path = job.evaluation_dir / "validation_threshold_selection.json"
        if not validation_path.is_file():
            errors.append(f"Missing validation threshold: {validation_path}")

        if job.dataset not in checked_datasets:
            checked_datasets.add(job.dataset)
            if not job.manifest_path.is_file():
                errors.append(f"Missing cohort manifest: {job.manifest_path}")
                continue
            try:
                manifest = load_json(job.manifest_path)
                test_items = cohort_items(manifest, "test")
                if len(test_items) < args.images:
                    errors.append(
                        f"Dataset {job.dataset!r} has only {len(test_items)} test "
                        f"images, but exactly {args.images} were requested."
                    )
            except Exception as exc:  # noqa: BLE001 - collect all preflight failures
                errors.append(f"Could not validate {job.manifest_path}: {exc}")

    if errors:
        formatted = "\n  - ".join(errors)
        raise RuntimeError(f"Preflight validation failed:\n  - {formatted}")


def build_long_summary(
    per_image_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        key = (
            str(row["dataset"]),
            str(row["backbone"]),
            str(row["method"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for method in METHODS:
                rows = grouped.get((str(dataset), str(backbone), str(method)), [])
                if not rows:
                    continue
                for metric in RUNTIME_METRICS:
                    summary_rows.append(
                        {
                            "dataset": dataset,
                            "backbone": backbone,
                            "method": method,
                            "metric": metric,
                            **summary_statistics(float(row[metric]) for row in rows),
                        }
                    )
    return summary_rows


def build_wide_summary(
    long_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in long_summary_rows:
        grouped[
            (
                str(row["dataset"]),
                str(row["backbone"]),
                str(row["method"]),
            )
        ].append(row)

    result: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for method in METHODS:
                rows = grouped.get((str(dataset), str(backbone), str(method)), [])
                if not rows:
                    continue
                output: dict[str, Any] = {
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    "n_images": int(rows[0]["n_images"]),
                }
                for row in rows:
                    metric = str(row["metric"])
                    output[f"{metric}_mean"] = float(row["mean"])
                    output[f"{metric}_std"] = float(row["std"])
                    output[f"{metric}_median"] = float(row["median"])
                    output[f"{metric}_mean_plus_minus_std"] = str(
                        row["mean_plus_minus_std"]
                    )
                result.append(output)
    return result


def build_threshold_paper_table(
    long_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, str], str] = {}
    for row in long_summary_rows:
        if row["metric"] != "threshold_selection_ms":
            continue
        lookup[
            (
                str(row["dataset"]),
                str(row["backbone"]),
                str(row["method"]),
            )
        ] = str(row["mean_plus_minus_std"])

    result: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            if not any(
                (str(dataset), str(backbone), str(method)) in lookup
                for method in METHODS
            ):
                continue
            row: dict[str, Any] = {
                "dataset": dataset,
                "backbone": backbone,
                "unit": "ms/image, mean ± SD over 50 images",
            }
            for method in METHODS:
                row[str(method)] = lookup.get(
                    (str(dataset), str(backbone), str(method)),
                    "",
                )
            result.append(row)
    return result


def build_inference_paper_table(
    per_image_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Inference time is identical across method rows for the same image. Retain
    # only one method to avoid counting each image repeatedly.
    one_method = str(METHODS[0])
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in per_image_rows:
        if str(row["method"]) != one_method:
            continue
        grouped[(str(row["dataset"]), str(row["backbone"]))].append(
            float(row["model_inference_ms"])
        )

    result: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            values = grouped.get((str(dataset), str(backbone)), [])
            if not values:
                continue
            stats = summary_statistics(values)
            result.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "n_images": stats["n_images"],
                    "model_inference_ms_mean": stats["mean"],
                    "model_inference_ms_std": stats["std"],
                    "model_inference_ms_mean_plus_minus_std": stats[
                        "mean_plus_minus_std"
                    ],
                }
            )
    return result


def benchmark_combination(
    job: BenchmarkJob,
    args: argparse.Namespace,
    threshold_grid: np.ndarray,
    combination_index: int,
    combination_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation_threshold, validation_threshold_source = load_validation_threshold(
        job.evaluation_dir
    )

    manifest = load_json(job.manifest_path)
    all_items = cohort_items(manifest, "test")
    if len(all_items) < args.images:
        raise RuntimeError(
            f"{job.dataset} has {len(all_items)} test images; {args.images} required."
        )
    items = deterministic_subset(all_items, args.images, args.sample_seed)
    if len(items) != args.images:
        raise RuntimeError(
            f"Expected exactly {args.images} sampled images for {job.dataset}, "
            f"but obtained {len(items)}."
        )

    runtime = load_model_runtime(
        checkpoint_path=job.checkpoint_path,
        dataset=job.dataset,
        requested_backbone=job.backbone,
        input_height=args.input_height,
        input_width=args.input_width,
        device_name=args.device,
    )

    print("=" * 100)
    print(
        f"[COMBINATION {combination_index}/{combination_count}] "
        f"dataset={job.dataset}, backbone={runtime.backbone}"
    )
    print(f"[DEVICE] {runtime.device}")
    print("[INFERENCE PRECISION] FP32")
    print(f"[TEST IMAGES] exactly {len(items)} / {len(all_items)}")
    print(f"[SAMPLE SEED] {args.sample_seed}")
    print(f"[VALIDATION F2 THRESHOLD] {validation_threshold:.4f}")

    per_image_rows: list[dict[str, Any]] = []
    model_warmed = False
    combination_start = time.time()

    for image_index, item in enumerate(items, start=1):
        mask_path = resolve_mask_path(item, job.data_root)
        image_path = resolve_image_path(item, job.data_root)
        original_mask = load_binary_mask(mask_path, threshold=MASK_THRESHOLD)
        image_id = item_identifier(item)

        tensor, mask_canvas, valid_canvas = prepare_image(
            image_path=image_path,
            original_mask=original_mask,
            input_height=runtime.input_height,
            input_width=runtime.input_width,
        )

        if not model_warmed:
            for _ in range(args.model_warmup):
                _ = infer_prepared_tensor(runtime=runtime, tensor=tensor)
            synchronize(runtime.device)
            model_warmed = True

        inference_ms, probability = milliseconds(
            lambda: infer_prepared_tensor(runtime=runtime, tensor=tensor),
            args.repeats,
            device=runtime.device,
        )
        probabilities, _ = valid_probability_and_truth(
            probability,
            mask_canvas,
            valid_canvas,
        )

        for _ in range(args.postprocess_warmup):
            _ = otsu_threshold(probabilities, bins=HISTOGRAM_BINS)
            _ = kapur_threshold(probabilities, bins=HISTOGRAM_BINS)
            _ = proposed_threshold(
                probabilities,
                threshold_grid,
                beta=F_BETA,
            )[0]

        selectors: dict[str, Callable[[], float]] = {
            "Fixed": lambda: FIXED_THRESHOLD,
            "Validation F2": lambda: validation_threshold,
            "Otsu": lambda: otsu_threshold(
                probabilities,
                bins=HISTOGRAM_BINS,
            ),
            "Kapur": lambda: kapur_threshold(
                probabilities,
                bins=HISTOGRAM_BINS,
            ),
            "ECMTS": lambda: proposed_threshold(
                probabilities,
                threshold_grid,
                beta=F_BETA,
            )[0],
        }

        for method in METHODS:
            selector = selectors[str(method)]

            if method in {"Fixed", "Validation F2"}:
                # These methods use predetermined global thresholds at deployment.
                # Their online threshold-selection time is therefore zero.
                threshold = float(selector())
                selection_ms = 0.0
            else:
                # Measure threshold selection only. Binary-mask generation is excluded.
                selection_ms, threshold = milliseconds(
                    selector,
                    args.repeats,
                )
                threshold = float(threshold)

            per_image_rows.append(
                {
                    "dataset": job.dataset,
                    "backbone": runtime.backbone,
                    "seed": args.seed,
                    "sample_seed": args.sample_seed,
                    "image_index": image_index - 1,
                    "image_id": image_id,
                    "method": method,
                    "n_valid_pixels": int(probabilities.size),
                    "selected_threshold": threshold,
                    "model_inference_ms": inference_ms,
                    "threshold_selection_ms": selection_ms,
                    # Derived from independently measured per-image medians.
                    "inference_plus_threshold_selection_ms": (
                        inference_ms + selection_ms
                    ),
                    "within_image_repeats": args.repeats,
                    "within_image_aggregation": "median",
                }
            )

        print(
            f"[RUNTIME {combination_index}/{combination_count} | "
            f"{image_index:>2}/{len(items)}] {image_id}",
            flush=True,
        )

    elapsed = time.time() - combination_start
    metadata = {
        "dataset": job.dataset,
        "backbone": runtime.backbone,
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "checkpoint": str(job.checkpoint_path),
        "cohort_manifest": str(job.manifest_path),
        "evaluation_dir": str(job.evaluation_dir),
        "input_height": runtime.input_height,
        "input_width": runtime.input_width,
        "device": str(runtime.device),
        "inference_precision": "float32",
        "test_images_available": len(all_items),
        "test_images_benchmarked": len(items),
        "validation_f2_threshold": validation_threshold,
        "validation_f2_threshold_source": validation_threshold_source,
        "elapsed_seconds": elapsed,
    }

    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return per_image_rows, metadata


def print_combination_summary(
    dataset: str,
    backbone: str,
    long_summary_rows: list[dict[str, Any]],
) -> None:
    print("[THRESHOLD-SELECTION TIME: MEAN ± SD ACROSS 50 IMAGES]")
    for method in METHODS:
        matching = [
            row
            for row in long_summary_rows
            if row["dataset"] == dataset
            and row["backbone"] == backbone
            and row["method"] == method
            and row["metric"] == "threshold_selection_ms"
        ]
        if matching:
            print(
                f"  {str(method):<14} "
                f"{matching[0]['mean_plus_minus_std']} ms/image"
            )


def main() -> None:
    print("[START] All-combination runtime benchmark", flush=True)
    args = parse_args()
    jobs = build_jobs(args)
    preflight(jobs, args)

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_root(args.seed).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    threshold_grid = np.linspace(
        0.0,
        1.0,
        THRESHOLD_GRID_SIZE,
        dtype=np.float64,
    )

    print(f"[JOBS] {len(jobs)} dataset--backbone combinations")
    print(f"[IMAGES PER JOB] exactly {args.images}")
    print(f"[REPEATS PER IMAGE] {args.repeats}; retained statistic=median")
    print("[BETWEEN-IMAGE SUMMARY] mean ± sample standard deviation")
    print(f"[OUTPUT ROOT] {output_root}")

    benchmark_start = time.time()
    all_per_image_rows: list[dict[str, Any]] = []
    combination_metadata: list[dict[str, Any]] = []

    for index, job in enumerate(jobs, start=1):
        rows, metadata = benchmark_combination(
            job,
            args,
            threshold_grid,
            combination_index=index,
            combination_count=len(jobs),
        )
        all_per_image_rows.extend(rows)
        combination_metadata.append(metadata)

        combination_long = build_long_summary(rows)
        combination_wide = build_wide_summary(combination_long)
        combination_dir = output_root / job.dataset / job.backbone
        combination_dir.mkdir(parents=True, exist_ok=True)
        write_csv(combination_dir / "runtime_per_image.csv", rows)
        write_csv(combination_dir / "runtime_summary_long.csv", combination_long)
        write_csv(combination_dir / "runtime_summary_wide.csv", combination_wide)
        save_json(combination_dir / "runtime_metadata.json", metadata)
        print_combination_summary(job.dataset, job.backbone, combination_long)

    long_summary_rows = build_long_summary(all_per_image_rows)
    wide_summary_rows = build_wide_summary(long_summary_rows)
    threshold_paper_rows = build_threshold_paper_table(long_summary_rows)
    inference_paper_rows = build_inference_paper_table(all_per_image_rows)

    write_csv(output_root / "runtime_per_image_all.csv", all_per_image_rows)
    write_csv(output_root / "runtime_summary_long.csv", long_summary_rows)
    write_csv(output_root / "runtime_summary_wide.csv", wide_summary_rows)
    write_csv(output_root / "runtime_threshold_paper_table.csv", threshold_paper_rows)
    write_csv(output_root / "runtime_inference_paper_table.csv", inference_paper_rows)

    total_elapsed = time.time() - benchmark_start
    save_json(
        output_root / "runtime_metadata.json",
        {
            "analysis": "threshold_runtime_all_dataset_backbone_combinations",
            "seed": args.seed,
            "sample_seed": args.sample_seed,
            "datasets": list(args.datasets),
            "backbones": list(args.backbones),
            "n_combinations": len(jobs),
            "required_images_per_combination": args.images,
            "total_unique_image_combination_pairs": len(jobs) * args.images,
            "total_method_rows": len(all_per_image_rows),
            "model_warmup_runs_per_combination": args.model_warmup,
            "postprocess_warmup_runs_per_image": args.postprocess_warmup,
            "timing_repeats_per_image": args.repeats,
            "within_image_timing_aggregation": "median",
            "between_image_reporting": "mean and sample standard deviation",
            "methods": list(METHODS),
            "runtime_metrics": list(RUNTIME_METRICS),
            "excluded_from_inference_timing": (
                "image decoding, mask loading, letterbox resizing, and normalization"
            ),
            "inference_timing_scope": (
                "tensor transfer to device, forward pass, sigmoid, and "
                "probability-map transfer to CPU"
            ),
            "fixed_and_validation_selection_time_ms": 0.0,
            "histogram_bins": HISTOGRAM_BINS,
            "threshold_grid_size": THRESHOLD_GRID_SIZE,
            "ecmts_implementation": (
                "exact 2001-point grid using sorting and cumulative sums"
            ),
            "binary_mask_generation_included": False,
            "inference_plus_threshold_selection_note": (
                "Derived as the sum of independently measured per-image median "
                "inference and threshold-selection times. Binary-mask generation "
                "was excluded."
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "total_elapsed_seconds": total_elapsed,
            "combination_metadata": combination_metadata,
        },
    )

    print("=" * 100)
    print("[OK] All runtime benchmarks completed.")
    print(f"[PER IMAGE] {output_root / 'runtime_per_image_all.csv'}")
    print(f"[LONG SUMMARY] {output_root / 'runtime_summary_long.csv'}")
    print(f"[WIDE SUMMARY] {output_root / 'runtime_summary_wide.csv'}")
    print(
        f"[PAPER TABLE: THRESHOLD] "
        f"{output_root / 'runtime_threshold_paper_table.csv'}"
    )
    print(
        f"[PAPER TABLE: INFERENCE] "
        f"{output_root / 'runtime_inference_paper_table.csv'}"
    )
    print(f"[METADATA] {output_root / 'runtime_metadata.json'}")


if __name__ == "__main__":
    main()
