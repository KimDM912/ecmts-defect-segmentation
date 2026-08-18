#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze the need for image-wise thresholding using oracle regret.

For each test image i, this script computes

    regret_i = F2_i(image-wise oracle) - F2_i(validation-global threshold)

The validation-global F2 is read from ``per_image_metrics.csv`` for the
``Validation F2`` method. The image-wise oracle F2 is read from the sibling
``diagnostics.csv`` file produced by ``evaluate_thresholding.py``.

Primary outputs are calculated separately for each dataset-backbone-seed
combination:
- mean validation-global F2
- mean image-wise oracle F2
- mean oracle regret and percentile-bootstrap 95% CI
- paired sign-flip permutation p-value
- per-image analysis table
- optional regret box plot

Default execution
-----------------
Run without arguments from any working directory:

    python analysis/analyze_global_oracle_regret.py

The script reads ``<project>/evaluation/results_thresholding`` and writes to
``<project>/analysis/results/global_oracle_regret/seed_2026`` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "evaluation"
    / "results_thresholding"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "results"
    / "global_oracle_regret"
)

SIGNIFICANCE_LEVEL = 0.05
VALIDATION_METHOD = "Validation F2"

REQUIRED_METRIC_COLUMNS = {
    "dataset",
    "backbone",
    "seed",
    "image_id",
    "method",
    "f2",
}

REQUIRED_DIAGNOSTIC_COLUMNS = {
    "dataset",
    "backbone",
    "seed",
    "image_id",
    "oracle_true_f2",
}

GROUP_COLUMNS = [
    "dataset",
    "backbone",
    "seed",
]

KEY_COLUMNS = [
    "dataset",
    "backbone",
    "seed",
    "image_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a validation-derived global threshold with the "
            "image-wise oracle threshold using paired image-level regret."
        )
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Optional result-root override. By default, the script uses "
            "<project>/evaluation/results_thresholding."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output-directory override. By default, outputs are "
            "written to "
            "<project>/analysis/results/global_oracle_regret/seed_<seed>."
        ),
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10_000,
        help="Number of paired image-level bootstrap resamples.",
    )

    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=10_000,
        help="Number of paired sign-flip permutation samples.",
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Percentile-bootstrap confidence level; default: 0.95.",
    )

    parser.add_argument(
        "--alternative",
        choices=("two-sided", "greater"),
        default="two-sided",
        help=(
            "'greater' tests whether mean oracle regret is greater than zero. "
            "'two-sided' tests whether the mean regret differs from zero."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Training/result seed to analyze and random seed used for "
            "bootstrap and permutation resampling; default: 2026."
        ),
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip creation of the oracle-regret box plot.",
    )

    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    path: Path,
) -> None:
    """Raise an error when required CSV columns are missing."""

    missing = sorted(set(required) - set(frame.columns))

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )


def assert_unique(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    label: str,
) -> None:
    """Ensure that one row exists for each image key."""

    duplicated = frame.duplicated(keys, keep=False)

    if duplicated.any():
        examples = (
            frame.loc[duplicated, keys]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            f"{label} contains duplicate image records "
            f"for keys {keys}. Examples: {examples}"
        )


def discover_result_pairs(
    results_root: Path,
) -> list[tuple[Path, Path]]:
    """Find matching per-image metric and diagnostic CSV files."""

    metric_paths = sorted(
        results_root.rglob("per_image_metrics.csv")
    )

    if not metric_paths:
        raise FileNotFoundError(
            "No per_image_metrics.csv files found under: "
            f"{results_root.resolve()}"
        )

    pairs: list[tuple[Path, Path]] = []
    missing_diagnostic_paths: list[Path] = []

    for metric_path in metric_paths:
        diagnostic_path = metric_path.with_name(
            "diagnostics.csv"
        )

        if diagnostic_path.is_file():
            pairs.append(
                (metric_path, diagnostic_path)
            )
        else:
            missing_diagnostic_paths.append(
                diagnostic_path
            )

    if missing_diagnostic_paths:
        preview = "\n  - ".join(
            str(path)
            for path in missing_diagnostic_paths[:10]
        )

        raise FileNotFoundError(
            "A diagnostics.csv file is missing beside one or more "
            "per_image_metrics.csv files:\n"
            f"  - {preview}"
        )

    return pairs


def load_analysis_table(
    results_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Load and merge validation-global and oracle image-wise results."""

    tables: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []

    result_pairs = discover_result_pairs(
        results_root
    )

    for metric_path, diagnostic_path in result_pairs:
        metrics = pd.read_csv(
            metric_path
        )

        diagnostics = pd.read_csv(
            diagnostic_path
        )

        require_columns(
            metrics,
            REQUIRED_METRIC_COLUMNS,
            path=metric_path,
        )

        require_columns(
            diagnostics,
            REQUIRED_DIAGNOSTIC_COLUMNS,
            path=diagnostic_path,
        )

        validation = metrics.loc[
            metrics["method"].astype(str)
            == VALIDATION_METHOD,
            KEY_COLUMNS + ["f2", "threshold"],
        ].copy()

        if validation.empty:
            raise ValueError(
                f"No method={VALIDATION_METHOD!r} rows "
                f"found in {metric_path}"
            )

        validation = validation.rename(
            columns={
                "f2": "validation_global_f2",
                "threshold": "validation_global_threshold",
            }
        )

        required_diagnostic_values = [
            "oracle_true_f2",
            "oracle_threshold",
            "defect_ratio",
            "class_separation",
            "brier_score",
            "probability_mass_bias",
        ]

        missing_diagnostic_values = sorted(
            set(required_diagnostic_values)
            - set(diagnostics.columns)
        )

        if missing_diagnostic_values:
            raise ValueError(
                f"Missing diagnostic columns in {diagnostic_path}: "
                f"{missing_diagnostic_values}"
            )

        oracle = diagnostics[
            KEY_COLUMNS
            + required_diagnostic_values
        ].copy()

        assert_unique(
            validation,
            KEY_COLUMNS,
            label=str(metric_path),
        )

        assert_unique(
            oracle,
            KEY_COLUMNS,
            label=str(diagnostic_path),
        )

        merged = validation.merge(
            oracle,
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )

        if (
            len(merged) != len(validation)
            or len(merged) != len(oracle)
        ):
            raise ValueError(
                "Metric/diagnostic image mismatch in result folder: "
                f"{metric_path.parent}. "
                f"validation={len(validation)}, "
                f"diagnostics={len(oracle)}, "
                f"merged={len(merged)}"
            )

        merged["oracle_regret"] = (
            merged["oracle_true_f2"]
            - merged["validation_global_f2"]
        )

        # Tiny negative values may arise from floating-point rounding.
        tiny_negative = merged[
            "oracle_regret"
        ].between(
            -1e-12,
            0.0,
        )

        merged.loc[
            tiny_negative,
            "oracle_regret",
        ] = 0.0

        invalid_negative = (
            merged["oracle_regret"]
            < -1e-12
        )

        if invalid_negative.any():
            examples = merged.loc[
                invalid_negative,
                KEY_COLUMNS
                + [
                    "validation_global_f2",
                    "oracle_true_f2",
                    "oracle_regret",
                ],
            ].head(10)

            raise ValueError(
                "Oracle F2 is lower than Validation-global F2 "
                "for some images. Check that both values were "
                "calculated on the same threshold grid.\n"
                f"{examples.to_string(index=False)}"
            )

        tables.append(
            merged
        )

        sources.append(
            {
                "per_image_metrics": str(
                    metric_path.resolve()
                ),
                "diagnostics": str(
                    diagnostic_path.resolve()
                ),
            }
        )

    combined = pd.concat(
        tables,
        ignore_index=True,
    )

    assert_unique(
        combined,
        KEY_COLUMNS,
        label="combined analysis table",
    )

    return combined, sources


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute a percentile-bootstrap confidence interval for the mean."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return (
            float("nan"),
            float("nan"),
        )

    if (
        values.size == 1
        or np.allclose(values, values[0])
    ):
        value = float(
            values.mean()
        )

        return value, value

    lower_quantile = (
        1.0 - confidence_level
    ) / 2.0

    upper_quantile = (
        1.0 - lower_quantile
    )

    bootstrap_means = np.empty(
        iterations,
        dtype=np.float64,
    )

    chunk_size = min(
        1_000,
        iterations,
    )

    position = 0

    while position < iterations:
        current_iterations = min(
            chunk_size,
            iterations - position,
        )

        sampled_indices = rng.integers(
            0,
            values.size,
            size=(
                current_iterations,
                values.size,
            ),
        )

        bootstrap_means[
            position:
            position + current_iterations
        ] = values[
            sampled_indices
        ].mean(
            axis=1
        )

        position += current_iterations

    low, high = np.quantile(
        bootstrap_means,
        [
            lower_quantile,
            upper_quantile,
        ],
    )

    return (
        float(low),
        float(high),
    )


def sign_flip_permutation_pvalue(
    differences: np.ndarray,
    *,
    iterations: int,
    alternative: str,
    rng: np.random.Generator,
) -> float:
    """Run a paired Monte Carlo sign-flip permutation test."""

    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if differences.size == 0:
        return float("nan")

    if np.allclose(
        differences,
        0.0,
    ):
        return 1.0

    observed_mean = float(
        differences.mean()
    )

    extreme_count = 0
    completed_iterations = 0

    chunk_size = min(
        2_000,
        iterations,
    )

    while completed_iterations < iterations:
        current_iterations = min(
            chunk_size,
            iterations - completed_iterations,
        )

        signs = rng.choice(
            np.asarray(
                [-1.0, 1.0],
                dtype=np.float64,
            ),
            size=(
                current_iterations,
                differences.size,
            ),
            replace=True,
        )

        permuted_means = (
            signs * differences
        ).mean(
            axis=1
        )

        if alternative == "greater":
            extreme_count += int(
                np.count_nonzero(
                    permuted_means
                    >= observed_mean
                )
            )
        else:
            extreme_count += int(
                np.count_nonzero(
                    np.abs(permuted_means)
                    >= abs(observed_mean)
                )
            )

        completed_iterations += (
            current_iterations
        )

    # Add-one correction prevents a zero Monte Carlo p-value.
    return float(
        (extreme_count + 1)
        / (iterations + 1)
    )


def summarize_groups(
    frame: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    permutation_iterations: int,
    confidence_level: float,
    alternative: str,
    seed: int,
) -> pd.DataFrame:
    """Summarize oracle regret for every dataset-backbone combination."""

    rows: list[
        dict[str, float | int | str]
    ] = []

    grouped = list(
        frame.groupby(
            GROUP_COLUMNS,
            sort=True,
            observed=True,
        )
    )

    seed_sequence = np.random.SeedSequence(
        seed
    )

    child_sequences = seed_sequence.spawn(
        len(grouped) * 2
    )

    for group_index, (
        group_key,
        group,
    ) in enumerate(grouped):
        dataset, backbone, training_seed = (
            group_key
        )

        regret = group[
            "oracle_regret"
        ].to_numpy(
            dtype=np.float64
        )

        bootstrap_rng = np.random.default_rng(
            child_sequences[
                2 * group_index
            ]
        )

        permutation_rng = np.random.default_rng(
            child_sequences[
                2 * group_index + 1
            ]
        )

        ci_low, ci_high = bootstrap_mean_ci(
            regret,
            iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            rng=bootstrap_rng,
        )

        p_value = sign_flip_permutation_pvalue(
            regret,
            iterations=permutation_iterations,
            alternative=alternative,
            rng=permutation_rng,
        )

        rows.append(
            {
                "dataset": str(dataset),
                "backbone": str(backbone),
                "seed": int(training_seed),
                "n_images": int(len(group)),
                "validation_global_threshold": float(
                    group[
                        "validation_global_threshold"
                    ].iloc[0]
                ),
                "validation_global_f2_mean": float(
                    group[
                        "validation_global_f2"
                    ].mean()
                ),
                "validation_global_f2_std": float(
                    group[
                        "validation_global_f2"
                    ].std(ddof=1)
                    if len(group) > 1
                    else 0.0
                ),
                "image_wise_oracle_f2_mean": float(
                    group[
                        "oracle_true_f2"
                    ].mean()
                ),
                "image_wise_oracle_f2_std": float(
                    group[
                        "oracle_true_f2"
                    ].std(ddof=1)
                    if len(group) > 1
                    else 0.0
                ),
                "mean_oracle_regret": float(
                    regret.mean()
                ),
                "oracle_regret_std": float(
                    regret.std(ddof=1)
                    if len(group) > 1
                    else 0.0
                ),
                "oracle_regret_median": float(
                    np.median(regret)
                ),
                "oracle_regret_q25": float(
                    np.quantile(
                        regret,
                        0.25,
                    )
                ),
                "oracle_regret_q75": float(
                    np.quantile(
                        regret,
                        0.75,
                    )
                ),
                "oracle_regret_ci_low": (
                    ci_low
                ),
                "oracle_regret_ci_high": (
                    ci_high
                ),
                "permutation_alternative": (
                    alternative
                ),
                "permutation_p_value": (
                    p_value
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary["significant"] = (
        summary["permutation_p_value"]
        < SIGNIFICANCE_LEVEL
    )

    summary["ci_excludes_zero"] = (
        summary["oracle_regret_ci_low"]
        > 0.0
    )

    return (
        summary
        .sort_values(
            GROUP_COLUMNS
        )
        .reset_index(
            drop=True
        )
    )


def make_boxplot(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a combined box plot for all nine combinations."""

    ordered_groups = sorted(
        frame[
            GROUP_COLUMNS
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    boxplot_values: list[np.ndarray] = []
    labels: list[str] = []

    for (
        dataset,
        backbone,
        training_seed,
    ) in ordered_groups:
        group_mask = (
            (
                frame["dataset"]
                == dataset
            )
            & (
                frame["backbone"]
                == backbone
            )
            & (
                frame["seed"]
                == training_seed
            )
        )

        boxplot_values.append(
            frame.loc[
                group_mask,
                "oracle_regret",
            ].to_numpy(
                dtype=np.float64
            )
        )

        labels.append(
            f"{dataset}\n"
            f"{backbone}\n"
            f"seed {training_seed}"
        )

    figure_width = max(
        9.0,
        1.25 * len(labels),
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            5.5,
        )
    )

    axis.boxplot(
        boxplot_values,
        tick_labels=labels,
        showmeans=True,
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    axis.set_ylabel(
        r"Oracle regret: "
        r"$F_2^{oracle} "
        r"- F_2^{validation-global}$"
    )

    axis.set_xlabel(
        "Dataset-backbone-seed combination"
    )

    axis.set_title(
        "Image-wise oracle regret of the "
        "validation-derived global threshold"
    )

    axis.tick_params(
        axis="x",
        labelrotation=25,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    args = parse_args()

    if args.bootstrap_iterations <= 0:
        raise ValueError(
            "--bootstrap-iterations must be positive."
        )

    if args.permutation_iterations <= 0:
        raise ValueError(
            "--permutation-iterations must be positive."
        )

    if not (
        0.0
        < args.confidence_level
        < 1.0
    ):
        raise ValueError(
            "--confidence-level must lie strictly "
            "between 0 and 1."
        )

    results_root = (
        args.results_root
        .expanduser()
        .resolve()
        if args.results_root is not None
        else DEFAULT_RESULTS_ROOT.resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
        if args.output_dir is not None
        else (
            DEFAULT_OUTPUT_ROOT
            / f"seed_{args.seed}"
        ).resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[START] Global-vs-oracle regret analysis",
        flush=True,
    )

    print(
        f"[RESULTS ROOT] {results_root}",
        flush=True,
    )

    print(
        f"[OUTPUT DIR] {output_dir}",
        flush=True,
    )

    print(
        f"[SEED] {args.seed}",
        flush=True,
    )

    print(
        f"[BOOTSTRAP ITERATIONS] "
        f"{args.bootstrap_iterations}",
        flush=True,
    )

    print(
        f"[PERMUTATION ITERATIONS] "
        f"{args.permutation_iterations}",
        flush=True,
    )

    per_image, sources = load_analysis_table(
        results_root
    )

    per_image = per_image.loc[
        per_image["seed"].astype(int)
        == int(args.seed)
    ].copy()

    if per_image.empty:
        raise RuntimeError(
            f"No rows for seed={args.seed} "
            f"were found under {results_root}."
        )

    combinations = (
        per_image[
            [
                "dataset",
                "backbone",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "dataset",
                "backbone",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(combinations) != 9:
        found = combinations.to_dict(
            "records"
        )

        raise RuntimeError(
            "Expected exactly 9 dataset-backbone "
            "combinations (3 datasets x 3 backbones) "
            f"for seed={args.seed}, but found "
            f"{len(combinations)}: {found}"
        )

    print(
        "[COMBINATIONS] 9 / 9",
        flush=True,
    )

    seed_token = f"seed_{args.seed}"

    sources = [
        source
        for source in sources
        if seed_token
        in Path(
            source["per_image_metrics"]
        ).parts
    ]

    summary = summarize_groups(
        per_image,
        bootstrap_iterations=(
            args.bootstrap_iterations
        ),
        permutation_iterations=(
            args.permutation_iterations
        ),
        confidence_level=(
            args.confidence_level
        ),
        alternative=(
            args.alternative
        ),
        seed=args.seed,
    )

    per_image_path = (
        output_dir
        / "global_oracle_regret_per_image.csv"
    )

    summary_path = (
        output_dir
        / "global_oracle_regret_summary.csv"
    )

    per_image.to_csv(
        per_image_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    plot_path: Path | None = None

    if not args.no_plot:
        plot_path = (
            output_dir
            / "global_oracle_regret_boxplot.png"
        )

        make_boxplot(
            per_image,
            plot_path,
        )

    metadata = {
        "analysis": (
            "validation_global_vs_"
            "image_wise_oracle_regret"
        ),
        "formula": (
            "oracle_regret = "
            "oracle_true_f2 - "
            "validation_global_f2"
        ),
        "results_root": str(
            results_root
        ),
        "output_dir": str(
            output_dir
        ),
        "validation_method": (
            VALIDATION_METHOD
        ),
        "group_columns": (
            GROUP_COLUMNS
        ),
        "bootstrap_iterations": (
            args.bootstrap_iterations
        ),
        "permutation_iterations": (
            args.permutation_iterations
        ),
        "confidence_level": (
            args.confidence_level
        ),
        "permutation_alternative": (
            args.alternative
        ),
        "significance_level": (
            SIGNIFICANCE_LEVEL
        ),
        "resampling_seed": (
            args.seed
        ),
        "n_source_pairs": (
            len(sources)
        ),
        "n_dataset_backbone_combinations": (
            int(len(combinations))
        ),
        "n_images_total": (
            int(len(per_image))
        ),
        "sources": (
            sources
        ),
        "outputs": {
            "per_image": str(
                per_image_path
            ),
            "summary": str(
                summary_path
            ),
            "boxplot": (
                str(plot_path)
                if plot_path is not None
                else None
            ),
        },
    }

    metadata_path = (
        output_dir
        / "global_oracle_regret_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "[OK] Global-vs-oracle regret "
        "analysis completed."
    )

    print(
        f"[PER IMAGE] {per_image_path}"
    )

    print(
        f"[SUMMARY] {summary_path}"
    )

    if plot_path is not None:
        print(
            f"[BOXPLOT] {plot_path}"
        )

    print(
        f"[METADATA] {metadata_path}"
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()