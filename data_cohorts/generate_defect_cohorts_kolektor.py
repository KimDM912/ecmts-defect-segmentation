#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create the validation/test defect cohort manifest for KolektorSDD2.

- Reads data/splits/kolektor_split.json.
- Uses both the validation and test subsets.
- Uses the exact evaluation preprocessing and valid-region definition from
  evaluation/main_experiment/evaluation_common.py.
- Selects only images whose actual GT mask retains at least one positive pixel
  inside the valid region after preprocessing.
- Produces one defect cohort manifest; no images are copied or modified.
- The validation cohort is for the global Validation-F2 threshold baseline.
- The test cohort is for the final Fixed / Validation F2 / Otsu / Kapur / ECMTS
  comparison.

Default output:
    data/evaluation_manifests/kolektor_defect_cohorts.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

DATASET_NAME = "KolektorSDD2"
DATASET_SLUG = "kolektor"
DATA_ROOT_REL = Path("data/kolektor")
SPLIT_JSON_REL = Path("data/splits/kolektor_split.json")
OUTPUT_MANIFEST_REL = Path("data/evaluation_manifests/kolektor_defect_cohorts.json")

MASK_THRESHOLD = 127
VALID_REGION_THRESHOLD = 0.5
INPUT_HEIGHT = 640
INPUT_WIDTH = 384


def infer_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), script_path.parent, *script_path.parents]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "data").is_dir()
            and (candidate / "evaluation" / "main_experiment").is_dir()
        ):
            return candidate
    return Path.cwd().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Create validation/test defect cohorts for {DATASET_NAME}."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=infer_project_root(),
        help="Repository root containing data/ and evaluation/. Default: auto-detected.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--fail-on-label-mismatch",
        action="store_true",
        help=(
            "Fail when split JSON has_defect disagrees with the raw GT mask. "
            "Without this flag, mismatches are audited and actual GT controls selection."
        ),
    )
    return parser.parse_args()


def resolve_path(value: Path | None, project_root: Path, default_rel: Path) -> Path:
    if value is None:
        return (project_root / default_rel).resolve()
    value = value.expanduser()
    if value.is_absolute():
        return value.resolve()
    return (project_root / value).resolve()


def load_evaluation_functions(
    project_root: Path,
) -> tuple[
    Callable[..., np.ndarray],
    Callable[..., tuple[Any, np.ndarray, np.ndarray]],
]:
    module_path = project_root / "evaluation" / "main_experiment" / "evaluation_common.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"evaluation_common.py not found: {module_path}")

    module_name = f"ecmts_evaluation_common_{DATASET_SLUG}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    load_binary_mask = getattr(module, "load_binary_mask", None)
    prepare_image = getattr(module, "prepare_image", None)
    if not callable(load_binary_mask) or not callable(prepare_image):
        raise AttributeError(
            "evaluation_common.py must define callable load_binary_mask and prepare_image."
        )
    return load_binary_mask, prepare_image


def load_split_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Split JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Split JSON root must be an object: {path}")
    for subset_name in ("val", "test"):
        if subset_name not in payload:
            raise KeyError(f"Split JSON does not contain '{subset_name}': {path}")
        if not isinstance(payload[subset_name], list):
            raise TypeError(f"Split JSON field '{subset_name}' must be a list: {path}")
    return payload


def project_relative_or_absolute(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_strings(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inspect_item(
    *,
    item: dict[str, Any],
    data_root: Path,
    load_binary_mask: Callable[..., np.ndarray],
    prepare_image: Callable[..., tuple[Any, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    if "image" not in item or "mask" not in item:
        raise KeyError(
            "Each split item must contain 'image' and 'mask'. "
            f"Available keys: {sorted(item)}"
        )

    image_rel = Path(str(item["image"]))
    mask_rel = Path(str(item["mask"]))
    image_abs = (data_root / image_rel).resolve()
    mask_abs = (data_root / mask_rel).resolve()

    if not image_abs.is_file():
        raise FileNotFoundError(f"Image not found: {image_abs}")
    if not mask_abs.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_abs}")

    original_mask = load_binary_mask(mask_abs, threshold=MASK_THRESHOLD)
    _, mask_canvas, valid_canvas = prepare_image(
        image_path=image_abs,
        original_mask=original_mask,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
    )

    raw_truth = np.asarray(original_mask) > 0.5
    valid = np.asarray(valid_canvas) > VALID_REGION_THRESHOLD
    if not np.any(valid):
        raise RuntimeError(
            "Evaluation valid-region mask is empty: "
            f"dataset={DATASET_SLUG}, image={image_rel.as_posix()}"
        )

    processed_truth = (np.asarray(mask_canvas) > 0.5) & valid
    raw_positive_pixels = int(np.count_nonzero(raw_truth))
    processed_positive_pixels = int(np.count_nonzero(processed_truth))
    valid_pixels = int(np.count_nonzero(valid))

    return {
        "item": item,
        "image_rel": image_rel.as_posix(),
        "mask_rel": mask_rel.as_posix(),
        "image_abs": image_abs,
        "mask_abs": mask_abs,
        "declared_has_defect": bool(item.get("has_defect", False)),
        "raw_has_defect": raw_positive_pixels > 0,
        "selected": processed_positive_pixels > 0,
        "raw_positive_pixels": raw_positive_pixels,
        "processed_positive_pixels": processed_positive_pixels,
        "valid_pixels": valid_pixels,
        "defect_ratio_in_valid": (
            processed_positive_pixels / valid_pixels if valid_pixels > 0 else 0.0
        ),
    }


def make_selected_record(row: dict[str, Any], project_root: Path) -> dict[str, Any]:
    record = dict(row["item"])
    record["has_defect"] = True
    record["evaluable_after_preprocessing"] = True
    record["mask_path"] = project_relative_or_absolute(row["mask_abs"], project_root)
    record["raw_positive_pixels"] = int(row["raw_positive_pixels"])
    record["processed_valid_positive_pixels"] = int(row["processed_positive_pixels"])
    record["valid_pixels"] = int(row["valid_pixels"])
    record["defect_ratio_in_valid"] = float(row["defect_ratio_in_valid"])
    return record


def make_audit_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": row["image_rel"],
        "mask": row["mask_rel"],
        "declared_has_defect": bool(row["declared_has_defect"]),
        "raw_has_defect": bool(row["raw_has_defect"]),
        "selected_after_preprocessing": bool(row["selected"]),
        "raw_positive_pixels": int(row["raw_positive_pixels"]),
        "processed_valid_positive_pixels": int(row["processed_positive_pixels"]),
        "valid_pixels": int(row["valid_pixels"]),
        "defect_ratio_in_valid": float(row["defect_ratio_in_valid"]),
    }


def process_subset(
    *,
    subset_name: str,
    items: list[dict[str, Any]],
    data_root: Path,
    project_root: Path,
    load_binary_mask: Callable[..., np.ndarray],
    prepare_image: Callable[..., tuple[Any, np.ndarray, np.ndarray]],
    fail_on_label_mismatch: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        row = inspect_item(
            item=item,
            data_root=data_root,
            load_binary_mask=load_binary_mask,
            prepare_image=prepare_image,
        )
        rows.append(row)
        if index == 1 or index % 250 == 0 or index == len(items):
            selected_so_far = sum(bool(current["selected"]) for current in rows)
            print(
                f"  [{subset_name}] {index:>5}/{len(items):<5} "
                f"selected={selected_so_far:>5} last={row['image_rel']}"
            )

    label_mismatches = [
        row
        for row in rows
        if bool(row["declared_has_defect"]) != bool(row["raw_has_defect"])
    ]
    if fail_on_label_mismatch and label_mismatches:
        examples = ", ".join(row["image_rel"] for row in label_mismatches[:5])
        raise RuntimeError(
            f"{DATASET_SLUG}/{subset_name}: {len(label_mismatches)} "
            f"has_defect/GT-mask mismatches. Examples: {examples}"
        )

    selected_rows = [row for row in rows if bool(row["selected"])]
    selected_records = [make_selected_record(row, project_root) for row in selected_rows]
    declared_defect_rows = [row for row in rows if bool(row["declared_has_defect"])]
    raw_defect_rows = [row for row in rows if bool(row["raw_has_defect"])]
    declared_defect_excluded = [
        row for row in rows if bool(row["declared_has_defect"]) and not bool(row["selected"])
    ]
    raw_defect_excluded = [
        row for row in rows if bool(row["raw_has_defect"]) and not bool(row["selected"])
    ]
    recovered_actual_defects = [
        row for row in rows if not bool(row["declared_has_defect"]) and bool(row["selected"])
    ]

    counts = {
        "n_split_total": len(rows),
        "n_declared_defect": len(declared_defect_rows),
        "n_declared_normal": len(rows) - len(declared_defect_rows),
        "n_actual_raw_defect": len(raw_defect_rows),
        "n_actual_raw_normal": len(rows) - len(raw_defect_rows),
        "n_selected_valid_region_defect": len(selected_rows),
        "n_excluded_no_valid_region_defect": len(rows) - len(selected_rows),
        "n_declared_defect_lost_after_preprocessing": len(declared_defect_excluded),
        "n_raw_defect_lost_after_preprocessing": len(raw_defect_excluded),
        "n_split_label_vs_raw_mask_mismatch": len(label_mismatches),
        "n_recovered_actual_defect_from_declared_normal": len(recovered_actual_defects),
    }
    selected_ids = [str(record["image"]) for record in selected_records]
    audit = {
        "counts": counts,
        "selected_image_list_sha256": sha256_strings(selected_ids),
        "split_label_vs_raw_mask_mismatches": [make_audit_record(row) for row in label_mismatches],
        "declared_defect_excluded_after_preprocessing": [
            make_audit_record(row) for row in declared_defect_excluded
        ],
        "raw_defect_excluded_after_preprocessing": [
            make_audit_record(row) for row in raw_defect_excluded
        ],
        "actual_defects_recovered_from_declared_normal": [
            make_audit_record(row) for row in recovered_actual_defects
        ],
    }
    return selected_records, audit


def validate_no_overlap(
    val_records: list[dict[str, Any]], test_records: list[dict[str, Any]]
) -> None:
    val_ids = {Path(str(record["image"])).as_posix() for record in val_records}
    test_ids = {Path(str(record["image"])).as_posix() for record in test_records}
    overlap = val_ids & test_ids
    if overlap:
        raise RuntimeError(f"Validation/test overlap detected: {sorted(overlap)[:5]}")


def print_subset_summary(subset_name: str, audit: dict[str, Any]) -> None:
    counts = audit["counts"]
    print(
        f"  {subset_name:>4}: total={counts['n_split_total']:>5}, "
        f"declared_defect={counts['n_declared_defect']:>5}, "
        f"raw_defect={counts['n_actual_raw_defect']:>5}, "
        f"selected={counts['n_selected_valid_region_defect']:>5}, "
        f"excluded={counts['n_excluded_no_valid_region_defect']:>5}, "
        f"label_mismatch={counts['n_split_label_vs_raw_mask_mismatch']:>3}"
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_root = resolve_path(args.data_root, project_root, DATA_ROOT_REL)
    split_json = resolve_path(args.split_json, project_root, SPLIT_JSON_REL)
    manifest_path = resolve_path(args.manifest, project_root, OUTPUT_MANIFEST_REL)

    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    load_binary_mask, prepare_image = load_evaluation_functions(project_root)
    split_payload = load_split_json(split_json)

    print(f"[INFO] Dataset     : {DATASET_NAME}")
    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Data root   : {data_root}")
    print(f"[INFO] Split JSON  : {split_json}")
    print(f"[INFO] Manifest    : {manifest_path}")
    print(
        f"[INFO] Evaluation  : {INPUT_HEIGHT}x{INPUT_WIDTH}, "
        f"mask threshold={MASK_THRESHOLD}, valid threshold={VALID_REGION_THRESHOLD}"
    )
    print("[INFO] Selection   : actual GT-positive pixels inside processed valid region")
    print("")

    start = time.time()
    val_records, val_audit = process_subset(
        subset_name="val",
        items=list(split_payload["val"]),
        data_root=data_root,
        project_root=project_root,
        load_binary_mask=load_binary_mask,
        prepare_image=prepare_image,
        fail_on_label_mismatch=bool(args.fail_on_label_mismatch),
    )
    test_records, test_audit = process_subset(
        subset_name="test",
        items=list(split_payload["test"]),
        data_root=data_root,
        project_root=project_root,
        load_binary_mask=load_binary_mask,
        prepare_image=prepare_image,
        fail_on_label_mismatch=bool(args.fail_on_label_mismatch),
    )

    if not val_records:
        raise RuntimeError("No validation images retained after valid-region filtering.")
    if not test_records:
        raise RuntimeError("No test images retained after valid-region filtering.")
    validate_no_overlap(val_records, test_records)

    manifest = {
        "metadata": {
            "dataset": DATASET_NAME,
            "dataset_slug": DATASET_SLUG,
            "source_split_json": project_relative_or_absolute(split_json, project_root),
            "data_root": project_relative_or_absolute(data_root, project_root),
            "purpose": {
                "val": "global Validation-F2 threshold selection",
                "test": "final threshold-method performance evaluation",
            },
            "selection_rule": (
                "retain an image iff the actual GT mask contains at least one positive "
                "pixel inside the valid region after the exact evaluation preprocessing"
            ),
            "selection_uses_split_has_defect": False,
            "selection_uses_actual_processed_gt": True,
            "evaluation_preprocessing": {
                "mask_threshold": MASK_THRESHOLD,
                "input_height": INPUT_HEIGHT,
                "input_width": INPUT_WIDTH,
                "valid_region_threshold": VALID_REGION_THRESHOLD,
            },
            "images_copied_or_modified": False,
            "source_split_metadata": split_payload.get("metadata", {}),
            "audit": {"val": val_audit, "test": test_audit},
        },
        "val": val_records,
        "test": test_records,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.time() - start
    print("")
    print("[SUMMARY]")
    print_subset_summary("val", val_audit)
    print_subset_summary("test", test_audit)
    print(f"[OK] Defect cohort manifest saved: {manifest_path}")
    print(f"[OK] Elapsed time: {elapsed:,.1f}s")


if __name__ == "__main__":
    main()
