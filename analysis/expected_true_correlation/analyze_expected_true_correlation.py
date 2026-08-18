#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze expected-versus-true F2 agreement across all dataset-backbone pairs.

By default, the script processes all 3 datasets x 3 backbones for one training
seed. It writes per-combination summaries, pooled image-level summaries, an
environment-balanced macro summary, paper-ready wide tables, and combined
figures.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common.result_analysis_common import (
    BACKBONES,
    DATASETS,
    analysis_result_dir,
    as_float,
    display_backbone,
    display_dataset,
    evaluation_result_dir,
    mean_std_median,
    read_csv,
    row_value,
    save_json,
    write_csv,
)


METRIC_SPECS = (
    ("expected_true_pearson", ("expected_true_pearson",), "Pearson r"),
    ("expected_true_spearman", ("expected_true_spearman",), "Spearman rho"),
    ("oracle_regret", ("oracle_regret",), "Oracle regret"),
    (
        "expected_f2_at_ecmts",
        ("expected_f2_at_ecmts", "expected_f2_at_proposed"),
        "Expected F2 at ECMTS threshold",
    ),
    (
        "true_f2_at_ecmts",
        ("true_f2_at_ecmts", "true_f2_at_proposed"),
        "True F2 at ECMTS threshold",
    ),
    ("oracle_true_f2", ("oracle_true_f2",), "Oracle true F2"),
    (
        "absolute_threshold_gap",
        ("absolute_threshold_gap",),
        "Absolute threshold gap",
    ),
)

PRIMARY_PLOT_METRICS = (
    "expected_true_pearson",
    "expected_true_spearman",
    "oracle_regret",
    "absolute_threshold_gap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to process. Default: all datasets.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=BACKBONES,
        default=list(BACKBONES),
        help="Backbones to process. Default: all backbones.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Combined output directory. By default, results are written under "
            "the expected_true_correlation_all analysis directory."
        ),
    )
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        help="Show boxplot outliers. They are hidden by default.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Skip combinations whose diagnostics.csv is absent. By default, a "
            "missing combination raises an error so that all requested results "
            "are guaranteed to be included."
        ),
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def metric_array(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> np.ndarray:
    return finite_array(
        as_float(row_value(row, *columns))
        for row in rows
    )


def metric_spec(metric_name: str) -> tuple[str, tuple[str, ...], str]:
    for specification in METRIC_SPECS:
        if specification[0] == metric_name:
            return specification
    raise KeyError(f"Unknown metric: {metric_name}")


def sample_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(mean) or not math.isfinite(std):
        return "NA"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def combination_label(dataset: str, backbone: str) -> str:
    return f"{display_dataset(dataset)}\n{display_backbone(backbone)}"


def summarize_combination(
    rows: list[dict[str, Any]],
    dataset: str,
    backbone: str,
    seed: int,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for metric, columns, label in METRIC_SPECS:
        statistics = mean_std_median(
            as_float(row_value(row, *columns))
            for row in rows
        )
        summary_rows.append(
            {
                "scope": "dataset_backbone",
                "dataset": dataset,
                "dataset_display": display_dataset(dataset),
                "backbone": backbone,
                "backbone_display": display_backbone(backbone),
                "seed": seed,
                "n_images_total": len(rows),
                "metric": metric,
                "metric_label": label,
                **statistics,
            }
        )
    return summary_rows


def summarize_pooled_images(
    rows: list[dict[str, Any]],
    seed: int,
    n_combinations: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for metric, columns, label in METRIC_SPECS:
        statistics = mean_std_median(
            as_float(row_value(row, *columns))
            for row in rows
        )
        result.append(
            {
                "scope": "pooled_images",
                "dataset": "ALL",
                "dataset_display": "All datasets",
                "backbone": "ALL",
                "backbone_display": "All backbones",
                "seed": seed,
                "n_combinations": n_combinations,
                "n_images_total": len(rows),
                "metric": metric,
                "metric_label": label,
                **statistics,
            }
        )
    return result


def summarize_macro_combinations(
    combination_summary_rows: list[dict[str, Any]],
    seed: int,
    n_combinations: int,
) -> list[dict[str, Any]]:
    """Summarize the nine combination-level means with equal combination weight."""
    result: list[dict[str, Any]] = []
    for metric, _, label in METRIC_SPECS:
        means = finite_array(
            as_float(row.get("mean"))
            for row in combination_summary_rows
            if row.get("metric") == metric
        )
        if means.size == 0:
            continue
        result.append(
            {
                "scope": "macro_across_combinations",
                "dataset": "ALL",
                "dataset_display": "All datasets",
                "backbone": "ALL",
                "backbone_display": "All backbones",
                "seed": seed,
                "n_combinations": n_combinations,
                "n_images_total": "",
                "metric": metric,
                "metric_label": label,
                "n": int(means.size),
                "mean": float(np.mean(means)),
                "std": sample_std(means),
                "median": float(np.median(means)),
                "minimum": float(np.min(means)),
                "maximum": float(np.max(means)),
            }
        )
    return result


def build_wide_numeric_table(
    combination_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in combination_summary_rows:
        key = (str(row["dataset"]), str(row["backbone"]))
        if key not in groups:
            groups[key] = {
                "dataset": row["dataset"],
                "dataset_display": row["dataset_display"],
                "backbone": row["backbone"],
                "backbone_display": row["backbone_display"],
                "seed": row["seed"],
                "n_images": row["n_images_total"],
            }
        metric = str(row["metric"])
        for statistic in ("mean", "std", "median", "minimum", "maximum", "n"):
            if statistic in row:
                groups[key][f"{metric}_{statistic}"] = row[statistic]
    return list(groups.values())


def build_paper_table(
    combination_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    numeric_rows = build_wide_numeric_table(combination_summary_rows)
    result: list[dict[str, Any]] = []
    for row in numeric_rows:
        output: dict[str, Any] = {
            "dataset": row["dataset_display"],
            "backbone": row["backbone_display"],
            "n_images": row["n_images"],
        }
        for metric, _, label in METRIC_SPECS:
            output[label] = format_mean_std(
                as_float(row.get(f"{metric}_mean")),
                as_float(row.get(f"{metric}_std")),
            )
        result.append(output)
    return result


def save_combination_figures(
    rows: list[dict[str, Any]],
    output_dir: Path,
    dataset: str,
    backbone: str,
    show_fliers: bool,
) -> None:
    values: list[np.ndarray] = []
    labels: list[str] = []
    for metric in PRIMARY_PLOT_METRICS:
        _, columns, label = metric_spec(metric)
        array = metric_array(rows, columns)
        if array.size:
            values.append(array)
            labels.append(label)

    if values:
        fig, axis = plt.subplots(figsize=(8.0, 4.8))
        axis.boxplot(values, tick_labels=labels, showfliers=show_fliers)
        axis.axhline(0.0, linewidth=0.8, linestyle="--")
        axis.set_title(
            f"{display_dataset(dataset)} · {display_backbone(backbone)} · "
            "Expected/true F2 diagnostics"
        )
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, output_dir, "expected_true_diagnostic_boxplots")

    ecmts = np.asarray(
        [
            as_float(
                row_value(
                    row,
                    "true_f2_at_ecmts",
                    "true_f2_at_proposed",
                )
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    oracle = np.asarray(
        [as_float(row_value(row, "oracle_true_f2")) for row in rows],
        dtype=np.float64,
    )
    mask = np.isfinite(ecmts) & np.isfinite(oracle)
    if int(mask.sum()) >= 2:
        fig, axis = plt.subplots(figsize=(5.6, 5.2))
        axis.scatter(ecmts[mask], oracle[mask], s=10, alpha=0.35)
        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("True F2 at ECMTS threshold")
        axis.set_ylabel("Oracle true F2")
        axis.set_title(
            f"{display_dataset(dataset)} · {display_backbone(backbone)} · "
            "ECMTS versus oracle"
        )
        axis.grid(alpha=0.25)
        fig.tight_layout()
        save_figure(fig, output_dir, "ecmts_vs_oracle_true_f2")


def save_combined_metric_boxplots(
    rows_by_combination: dict[tuple[str, str], list[dict[str, Any]]],
    output_dir: Path,
    show_fliers: bool,
) -> None:
    for metric in PRIMARY_PLOT_METRICS:
        _, columns, label = metric_spec(metric)
        arrays: list[np.ndarray] = []
        labels: list[str] = []
        for (dataset, backbone), rows in rows_by_combination.items():
            array = metric_array(rows, columns)
            if array.size:
                arrays.append(array)
                labels.append(combination_label(dataset, backbone))
        if not arrays:
            continue

        fig, axis = plt.subplots(figsize=(14.0, 6.0))
        axis.boxplot(arrays, tick_labels=labels, showfliers=show_fliers)
        axis.axhline(0.0, linewidth=0.8, linestyle="--")
        axis.set_ylabel(label)
        axis.set_title(f"{label} across dataset–backbone combinations")
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, output_dir, f"combined_{metric}_boxplot")


def save_combined_oracle_scatter(
    rows_by_combination: dict[tuple[str, str], list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 7.0))
    plotted = False

    for (dataset, backbone), rows in rows_by_combination.items():
        ecmts = np.asarray(
            [
                as_float(
                    row_value(
                        row,
                        "true_f2_at_ecmts",
                        "true_f2_at_proposed",
                    )
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        oracle = np.asarray(
            [as_float(row_value(row, "oracle_true_f2")) for row in rows],
            dtype=np.float64,
        )
        mask = np.isfinite(ecmts) & np.isfinite(oracle)
        if int(mask.sum()) < 2:
            continue
        axis.scatter(
            ecmts[mask],
            oracle[mask],
            s=10,
            alpha=0.28,
            label=f"{display_dataset(dataset)} · {display_backbone(backbone)}",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("True F2 at ECMTS threshold")
    axis.set_ylabel("Oracle true F2")
    axis.set_title("ECMTS versus image-wise oracle across all combinations")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "combined_ecmts_vs_oracle_true_f2")


def main() -> None:
    print("[START] Expected/true F2 and oracle analysis: all combinations", flush=True)
    args = parse_args()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else analysis_result_dir(
            "expected_true_correlation",
            "all_datasets",
            "all_backbones",
            args.seed,
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_combinations = [
        (dataset, backbone)
        for dataset in args.datasets
        for backbone in args.backbones
    ]

    rows_by_combination: dict[tuple[str, str], list[dict[str, Any]]] = {}
    combination_summary_rows: list[dict[str, Any]] = []
    all_image_rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    input_directories: dict[str, str] = {}

    for combination_index, (dataset, backbone) in enumerate(
        requested_combinations,
        start=1,
    ):
        input_dir = evaluation_result_dir(dataset, backbone, args.seed).resolve()
        diagnostics_path = input_dir / "diagnostics.csv"
        key_string = f"{dataset}/{backbone}"
        input_directories[key_string] = str(input_dir)

        print(
            f"[{combination_index}/{len(requested_combinations)}] "
            f"dataset={dataset}, backbone={backbone}",
            flush=True,
        )

        if not diagnostics_path.is_file():
            missing.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "path": str(diagnostics_path),
                }
            )
            if args.skip_missing:
                print(f"[SKIP] Missing {diagnostics_path}", flush=True)
                continue
            raise FileNotFoundError(
                "Missing diagnostics.csv for a requested combination: "
                f"{diagnostics_path}"
            )

        rows = read_csv(diagnostics_path)
        if not rows:
            raise RuntimeError(f"No diagnostic rows found in {input_dir}")

        enriched_rows: list[dict[str, Any]] = []
        for row in rows:
            enriched = dict(row)
            enriched["dataset"] = dataset
            enriched["dataset_display"] = display_dataset(dataset)
            enriched["backbone"] = backbone
            enriched["backbone_display"] = display_backbone(backbone)
            enriched["seed"] = args.seed
            enriched_rows.append(enriched)

        rows_by_combination[(dataset, backbone)] = enriched_rows
        all_image_rows.extend(enriched_rows)

        combination_summary = summarize_combination(
            enriched_rows,
            dataset,
            backbone,
            args.seed,
        )
        combination_summary_rows.extend(combination_summary)

        combination_output_dir = (
            output_dir / "by_combination" / dataset / backbone
        )
        combination_output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            combination_output_dir / "expected_true_oracle_summary.csv",
            combination_summary,
        )
        save_combination_figures(
            enriched_rows,
            combination_output_dir,
            dataset,
            backbone,
            args.show_fliers,
        )

    n_completed = len(rows_by_combination)
    if n_completed == 0:
        raise RuntimeError("No dataset-backbone combination was analyzed.")

    pooled_summary_rows = summarize_pooled_images(
        all_image_rows,
        args.seed,
        n_completed,
    )
    macro_summary_rows = summarize_macro_combinations(
        combination_summary_rows,
        args.seed,
        n_completed,
    )
    all_summary_rows = (
        combination_summary_rows
        + pooled_summary_rows
        + macro_summary_rows
    )

    write_csv(output_dir / "expected_true_oracle_per_image_all.csv", all_image_rows)
    write_csv(output_dir / "expected_true_oracle_summary_long.csv", all_summary_rows)
    write_csv(
        output_dir / "expected_true_oracle_summary_by_combination.csv",
        combination_summary_rows,
    )
    write_csv(
        output_dir / "expected_true_oracle_summary_pooled_images.csv",
        pooled_summary_rows,
    )
    write_csv(
        output_dir / "expected_true_oracle_summary_macro_combinations.csv",
        macro_summary_rows,
    )
    write_csv(
        output_dir / "expected_true_oracle_summary_wide_numeric.csv",
        build_wide_numeric_table(combination_summary_rows),
    )
    write_csv(
        output_dir / "expected_true_oracle_paper_table.csv",
        build_paper_table(combination_summary_rows),
    )

    save_combined_metric_boxplots(
        rows_by_combination,
        output_dir,
        args.show_fliers,
    )
    save_combined_oracle_scatter(rows_by_combination, output_dir)

    save_json(
        output_dir / "analysis_metadata.json",
        {
            "analysis": "expected_true_f2_correlation_and_oracle_regret_all",
            "seed": args.seed,
            "requested_datasets": list(args.datasets),
            "requested_backbones": list(args.backbones),
            "requested_combinations": len(requested_combinations),
            "completed_combinations": n_completed,
            "total_images": len(all_image_rows),
            "show_fliers": bool(args.show_fliers),
            "skip_missing": bool(args.skip_missing),
            "missing_combinations": missing,
            "input_directories": input_directories,
            "output_dir": str(output_dir),
            "summary_metrics": [metric for metric, _, _ in METRIC_SPECS],
            "pooled_summary_definition": (
                "Image-level pooling across every completed dataset-backbone "
                "combination; combinations with more test images receive more "
                "weight."
            ),
            "macro_summary_definition": (
                "Mean and sample standard deviation of the combination-level "
                "means; every dataset-backbone combination receives equal weight."
            ),
        },
    )

    print("=" * 100)
    print(f"[COMPLETED COMBINATIONS] {n_completed}/{len(requested_combinations)}")
    print(f"[TOTAL IMAGES] {len(all_image_rows)}")
    print(f"[OUTPUT] {output_dir}")
    print("[OK] Expected/true F2 and oracle analysis completed.", flush=True)


if __name__ == "__main__":
    main()
