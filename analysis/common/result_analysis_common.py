#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for analyses of the thresholding experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ("severstal", "kolektor", "magnetic")
BACKBONES = ("resnet18", "resnet34", "efficientnet_b0")

METHODS = ("Fixed", "Validation F2", "Otsu", "Kapur", "ECMTS")
BASELINES = ("Fixed", "Validation F2", "Otsu", "Kapur")
ADAPTIVE_METHODS = ("Otsu", "Kapur", "ECMTS")

DATASET_DISPLAY = {
    "severstal": "Severstal",
    "kolektor": "KolektorSDD2",
    "magnetic": "Magnetic Tile",
}

BACKBONE_DISPLAY = {
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "efficientnet_b0": "EfficientNet-B0",
}


def _token(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


_METHOD_ALIASES = {
    "fixed": "Fixed",
    "fixed05": "Fixed",
    "validationf2": "Validation F2",
    "validation": "Validation F2",
    "otsu": "Otsu",
    "kapur": "Kapur",
    "proposed": "ECMTS",
    "ecmts": "ECMTS",
}


def canonical_method(value: object) -> str:
    """Normalize method labels while accepting the earlier ``Proposed`` label."""
    key = _token(value)
    return _METHOD_ALIASES.get(key, str(value).strip())


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY.get(str(dataset), str(dataset))


def display_backbone(backbone: str) -> str:
    return BACKBONE_DISPLAY.get(str(backbone), str(backbone))


def evaluation_result_dir(dataset: str, backbone: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "evaluation"
        / "results_thresholding"
        / dataset
        / backbone
        / f"seed_{seed}"
    )


def analysis_result_dir(
    analysis_name: str,
    dataset: str,
    backbone: str,
    seed: int,
) -> Path:
    return (
        PROJECT_ROOT
        / "analysis"
        / "results"
        / analysis_name
        / dataset
        / backbone
        / f"seed_{seed}"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"Could not decode CSV {path}: {last_error}")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def row_value(row: dict[str, Any], *names: str) -> Any:
    """Return the first present column, supporting lower/upper-case exports."""
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]

    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def mean_std_median(values: Iterable[float]) -> dict[str, float]:
    array = finite(values)
    if array.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
        }

    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def normalize_method_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows with a canonical ``method`` field and supported methods only."""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        method = canonical_method(row_value(row, "method", "Method"))
        if method not in METHODS:
            continue
        copied = dict(row)
        copied["method"] = method
        normalized.append(copied)
    return normalized
