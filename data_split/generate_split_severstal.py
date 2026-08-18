#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a deterministic Severstal Steel Defect Detection split manifest.

Default behavior
----------------
- Reads:
    data/severstal-steel-defect-detection/train_images/
    data/severstal-steel-defect-detection/train.csv
- Generates binary masks by OR-combining class 1-4 RLE annotations:
    data/severstal-steel-defect-detection/generated_gt_masks/
- Splits all labeled images into 64% train / 16% validation / 20% test.
- Stratifies by the exact defect-class combination where possible.
- Uses seed 2026.
- Writes paths relative to the dataset root.
- Does not generate corruptions or separate post-processing splits.

Run from VS Code with "Run Python File", or:
    python data_split/generate_split_severstal.py

A different dataset location can be supplied with:
    python data_split/generate_split_severstal.py --data-root "D:/datasets/severstal-steel-defect-detection"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "severstal-steel-defect-detection"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "splits" / "severstal_split.json"


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.64
    val: float = 0.16
    test: float = 0.20

    def validate(self) -> None:
        values = (self.train, self.val, self.test)
        if any(v < 0.0 for v in values):
            raise ValueError(f"Split ratios must be non-negative: {values}")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError(f"Split ratios must sum to 1.0: {values}")


@dataclass(frozen=True)
class Item:
    image: str
    mask: str
    has_defect: bool
    defect_classes: list[int]
    class_signature: str
    original_height: int
    original_width: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic Severstal train/val/test split."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help=(
            "Directory for generated binary masks. "
            "Default: <data-root>/generated_gt_masks"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.64)
    parser.add_argument("--val-ratio", type=float, default=0.16)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument(
        "--overwrite-masks",
        action="store_true",
        help="Regenerate binary masks even when files already exist.",
    )
    return parser.parse_args()


def normalize_rle(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def parse_train_csv(csv_path: Path) -> dict[str, dict[int, list[str]]]:
    annotations: dict[str, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

        combined_format = {"ImageId_ClassId", "EncodedPixels"}.issubset(fieldnames)
        separate_format = {
            "ImageId",
            "ClassId",
            "EncodedPixels",
        }.issubset(fieldnames)

        if not combined_format and not separate_format:
            raise ValueError(
                "Unrecognized train.csv format. Expected either "
                "['ImageId_ClassId', 'EncodedPixels'] or "
                "['ImageId', 'ClassId', 'EncodedPixels']."
            )

        for row in reader:
            if combined_format:
                combined = str(row["ImageId_ClassId"])
                try:
                    image_id, class_text = combined.rsplit("_", 1)
                    class_id = int(class_text)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid ImageId_ClassId value: {combined!r}"
                    ) from exc
            else:
                image_id = str(row["ImageId"])
                class_id = int(row["ClassId"])

            if class_id not in {1, 2, 3, 4}:
                raise ValueError(
                    f"Unexpected ClassId={class_id} for image {image_id}"
                )

            rle = normalize_rle(row.get("EncodedPixels"))
            if rle is not None:
                annotations[image_id][class_id].append(rle)

    return annotations


def decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    values = np.fromstring(rle, sep=" ", dtype=np.int64)
    if values.size % 2 != 0:
        raise ValueError(f"Malformed RLE with odd token count: {rle[:80]!r}")

    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths

    flat = np.zeros(height * width, dtype=np.uint8)
    for start, end in zip(starts, ends):
        if start < 0 or end > flat.size or end < start:
            raise ValueError(
                f"RLE interval [{start}, {end}) exceeds mask size {flat.size}."
            )
        flat[start:end] = 1

    return flat.reshape((height, width), order="F")


def build_binary_mask(
    class_rles: dict[int, list[str]],
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for class_id in sorted(class_rles):
        for rle in class_rles[class_id]:
            mask = np.maximum(mask, decode_rle(rle, height, width))
    return mask * 255


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def discover_items(
    data_root: Path,
    mask_root: Path,
    overwrite_masks: bool,
) -> tuple[list[Item], dict[str, int]]:
    image_dir = data_root / "train_images"
    csv_path = data_root / "train.csv"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

    image_paths = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    annotations = parse_train_csv(csv_path)
    mask_root.mkdir(parents=True, exist_ok=True)

    items: list[Item] = []
    generated = 0
    reused = 0
    missing_csv_rows = 0

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size

        class_rles = annotations.get(image_path.name, {})
        defect_classes = sorted(
            class_id for class_id, rles in class_rles.items() if rles
        )
        has_defect = bool(defect_classes)
        signature = (
            "normal"
            if not defect_classes
            else "+".join(f"class_{class_id}" for class_id in defect_classes)
        )

        if image_path.name not in annotations:
            missing_csv_rows += 1

        mask_path = mask_root / f"{image_path.stem}_GT.png"
        if overwrite_masks or not mask_path.is_file():
            mask = build_binary_mask(class_rles, height, width)
            Image.fromarray(mask).save(mask_path)
            generated += 1
        else:
            reused += 1

        items.append(
            Item(
                image=relative_posix(image_path, data_root),
                mask=relative_posix(mask_path, data_root),
                has_defect=has_defect,
                defect_classes=defect_classes,
                class_signature=signature,
                original_height=height,
                original_width=width,
            )
        )

    stats = {
        "generated_masks": generated,
        "reused_masks": reused,
        "images_without_positive_annotations": missing_csv_rows,
    }
    return items, stats


def allocate_counts(n: int, ratios: SplitRatios) -> dict[str, int]:
    names = ("train", "val", "test")
    raw = {
        "train": n * ratios.train,
        "val": n * ratios.val,
        "test": n * ratios.test,
    }
    counts = {name: int(np.floor(raw[name])) for name in names}
    remainder = n - sum(counts.values())

    order = sorted(
        names,
        key=lambda name: (raw[name] - counts[name], -names.index(name)),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1

    return counts


def stratified_split(
    items: list[Item],
    ratios: SplitRatios,
    seed: int,
) -> dict[str, list[Item]]:
    strata: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        strata[item.class_signature].append(item)

    split = {"train": [], "val": [], "test": []}

    for stratum_index, stratum in enumerate(sorted(strata)):
        group = sorted(strata[stratum], key=lambda item: item.image)
        rng = random.Random(f"{seed}|{stratum_index}|{stratum}")
        rng.shuffle(group)

        counts = allocate_counts(len(group), ratios)
        train_end = counts["train"]
        val_end = train_end + counts["val"]

        split["train"].extend(group[:train_end])
        split["val"].extend(group[train_end:val_end])
        split["test"].extend(group[val_end:])

    for split_name in split:
        split[split_name] = sorted(split[split_name], key=lambda item: item.image)

    return split


def validate_split(items: list[Item], split: dict[str, list[Item]]) -> None:
    all_paths = {item.image for item in items}
    split_sets = {
        name: {item.image for item in split[name]}
        for name in ("train", "val", "test")
    }

    if split_sets["train"] & split_sets["val"]:
        raise RuntimeError("Train/validation overlap detected.")
    if split_sets["train"] & split_sets["test"]:
        raise RuntimeError("Train/test overlap detected.")
    if split_sets["val"] & split_sets["test"]:
        raise RuntimeError("Validation/test overlap detected.")

    union = set().union(*split_sets.values())
    if union != all_paths:
        missing = sorted(all_paths - union)
        extra = sorted(union - all_paths)
        raise RuntimeError(
            f"Split coverage mismatch. missing={missing[:5]}, extra={extra[:5]}"
        )


def sha256_strings(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize(items: list[Item]) -> dict[str, Any]:
    signatures = Counter(item.class_signature for item in items)
    class_counts = Counter(
        class_id for item in items for class_id in item.defect_classes
    )
    defect_count = sum(item.has_defect for item in items)

    return {
        "n": len(items),
        "defect": defect_count,
        "normal": len(items) - defect_count,
        "defect_ratio": defect_count / len(items) if items else 0.0,
        "class_counts": {
            str(class_id): class_counts.get(class_id, 0)
            for class_id in (1, 2, 3, 4)
        },
        "class_signatures": dict(sorted(signatures.items())),
    }


def print_summary(all_items: list[Item], split: dict[str, list[Item]]) -> None:
    print("\nSeverstal split summary")
    print("=" * 88)
    for name, current in (
        ("all", all_items),
        ("train", split["train"]),
        ("val", split["val"]),
        ("test", split["test"]),
    ):
        summary = summarize(current)
        print(
            f"{name:>5}: n={summary['n']:>5}, "
            f"defect={summary['defect']:>5}, "
            f"normal={summary['normal']:>5}, "
            f"defect_ratio={summary['defect_ratio']:.4f}"
        )
        print(f"       class_counts={summary['class_counts']}")
    print("=" * 88)


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    mask_root = (
        args.mask_root.expanduser().resolve()
        if args.mask_root is not None
        else data_root / "generated_gt_masks"
    )

    ratios = SplitRatios(
        train=args.train_ratio,
        val=args.val_ratio,
        test=args.test_ratio,
    )
    ratios.validate()

    items, mask_stats = discover_items(
        data_root=data_root,
        mask_root=mask_root,
        overwrite_masks=args.overwrite_masks,
    )
    split = stratified_split(items, ratios, args.seed)
    validate_split(items, split)

    payload = {
        "metadata": {
            "dataset": "Severstal Steel Defect Detection",
            "seed": args.seed,
            "ratios": asdict(ratios),
            "stratification": "exact defect-class combination",
            "binary_mask_rule": "OR-union of positive RLE masks from classes 1-4",
            "paths_are_relative_to": str(data_root),
            "data_root_layout": {
                "images": "train_images/",
                "annotations": "train.csv",
                "generated_masks": relative_posix(mask_root, data_root),
            },
            "mask_generation": mask_stats,
            "corruptions_generated": False,
            "separate_post_split_generated": False,
            "counts": {
                "all": summarize(items),
                "train": summarize(split["train"]),
                "val": summarize(split["val"]),
                "test": summarize(split["test"]),
            },
            "sha256_image_lists": {
                name: sha256_strings(item.image for item in split[name])
                for name in ("train", "val", "test")
            },
        },
        "train": [asdict(item) for item in split["train"]],
        "val": [asdict(item) for item in split["val"]],
        "test": [asdict(item) for item in split["test"]],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Split JSON saved: {output_path}")
    print(
        "[OK] Masks: "
        f"generated={mask_stats['generated_masks']}, "
        f"reused={mask_stats['reused_masks']}"
    )
    if mask_stats["images_without_positive_annotations"]:
        print(
            "[INFO] "
            f"{mask_stats['images_without_positive_annotations']} images had no positive RLE annotations "
            "and were treated as normal."
        )
    print_summary(items, split)


if __name__ == "__main__":
    main()
