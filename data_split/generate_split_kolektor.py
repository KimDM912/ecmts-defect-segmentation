#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a deterministic KolektorSDD2 split manifest.

Default behavior
----------------
- Reads the official folders:
    data/kolektor/train/
    data/kolektor/test/
- Splits only the official train folder into 80% train / 20% validation.
- Keeps the official test folder unchanged.
- Stratifies the train/validation split by ground-truth defect presence.
- Uses seed 2026.
- Writes paths relative to the dataset root.
- Does not generate corruptions or separate post-processing splits.

Run from VS Code with "Run Python File", or:
    python data_split/generate_split_kolektor.py

A different dataset location can be supplied with:
    python data_split/generate_split_kolektor.py --data-root "D:/datasets/kolektor"
"""

from __future__ import annotations

import argparse
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
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "kolektor"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "splits" / "kolektor_split.json"


@dataclass(frozen=True)
class TrainValRatios:
    train: float = 0.80
    val: float = 0.20

    def validate(self) -> None:
        values = (self.train, self.val)
        if any(v < 0.0 for v in values):
            raise ValueError(f"Split ratios must be non-negative: {values}")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError(f"Train/validation ratios must sum to 1.0: {values}")


@dataclass(frozen=True)
class Item:
    image: str
    mask: str
    source_split: str
    has_defect: bool
    original_height: int
    original_width: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic KolektorSDD2 split."
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--mask-threshold", type=int, default=127)
    return parser.parse_args()


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def is_gt_file(path: Path) -> bool:
    return path.stem.endswith("_GT")


def find_mask(image_path: Path) -> Path:
    candidates = [
        image_path.with_name(f"{image_path.stem}_GT.png"),
        image_path.with_name(f"{image_path.stem}_GT.jpg"),
        image_path.with_name(f"{image_path.stem}_GT.jpeg"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Ground-truth mask not found for {image_path}. "
        f"Expected one of: {[str(p) for p in candidates]}"
    )


def mask_has_defect(mask_path: Path, threshold: int) -> bool:
    with Image.open(mask_path) as mask:
        array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return bool((array > threshold).any())


def discover_split_items(
    data_root: Path,
    split_name: str,
    mask_threshold: int,
) -> list[Item]:
    split_root = data_root / split_name
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Official {split_name!r} directory not found: {split_root}"
        )

    image_paths = sorted(
        path
        for path in split_root.rglob("*")
        if is_image_file(path) and not is_gt_file(path)
    )
    if not image_paths:
        raise RuntimeError(f"No source images found in {split_root}")

    items: list[Item] = []
    for image_path in image_paths:
        mask_path = find_mask(image_path)

        with Image.open(image_path) as image:
            width, height = image.size

        items.append(
            Item(
                image=relative_posix(image_path, data_root),
                mask=relative_posix(mask_path, data_root),
                source_split=split_name,
                has_defect=mask_has_defect(mask_path, mask_threshold),
                original_height=height,
                original_width=width,
            )
        )

    return items


def allocate_counts(n: int, ratios: TrainValRatios) -> dict[str, int]:
    names = ("train", "val")
    raw = {"train": n * ratios.train, "val": n * ratios.val}
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


def stratified_train_val_split(
    items: list[Item],
    ratios: TrainValRatios,
    seed: int,
) -> tuple[list[Item], list[Item]]:
    strata: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        strata["defect" if item.has_defect else "normal"].append(item)

    train: list[Item] = []
    val: list[Item] = []

    for stratum_index, stratum in enumerate(("normal", "defect")):
        group = sorted(strata.get(stratum, []), key=lambda item: item.image)
        rng = random.Random(f"{seed}|{stratum_index}|{stratum}")
        rng.shuffle(group)

        counts = allocate_counts(len(group), ratios)
        split_point = counts["train"]
        train.extend(group[:split_point])
        val.extend(group[split_point:])

    return (
        sorted(train, key=lambda item: item.image),
        sorted(val, key=lambda item: item.image),
    )


def validate_split(
    official_train: list[Item],
    train: list[Item],
    val: list[Item],
    official_test: list[Item],
) -> None:
    official_train_paths = {item.image for item in official_train}
    train_paths = {item.image for item in train}
    val_paths = {item.image for item in val}
    test_paths = {item.image for item in official_test}

    if train_paths & val_paths:
        raise RuntimeError("Train/validation overlap detected.")
    if train_paths & test_paths:
        raise RuntimeError("Train/official-test overlap detected.")
    if val_paths & test_paths:
        raise RuntimeError("Validation/official-test overlap detected.")
    if train_paths | val_paths != official_train_paths:
        raise RuntimeError("Train/validation coverage does not match official train.")


def sha256_strings(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize(items: list[Item]) -> dict[str, Any]:
    defect_count = sum(item.has_defect for item in items)
    source_counts = Counter(item.source_split for item in items)
    return {
        "n": len(items),
        "defect": defect_count,
        "normal": len(items) - defect_count,
        "defect_ratio": defect_count / len(items) if items else 0.0,
        "source_splits": dict(sorted(source_counts.items())),
    }


def print_summary(
    all_train: list[Item],
    train: list[Item],
    val: list[Item],
    test: list[Item],
) -> None:
    print("\nKolektorSDD2 split summary")
    print("=" * 78)
    for name, current in (
        ("official_train", all_train),
        ("train", train),
        ("val", val),
        ("official_test", test),
    ):
        summary = summarize(current)
        print(
            f"{name:>14}: n={summary['n']:>4}, "
            f"defect={summary['defect']:>4}, "
            f"normal={summary['normal']:>4}, "
            f"defect_ratio={summary['defect_ratio']:.4f}"
        )
    print("=" * 78)


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    ratios = TrainValRatios(
        train=args.train_ratio,
        val=args.val_ratio,
    )
    ratios.validate()

    official_train = discover_split_items(
        data_root=data_root,
        split_name="train",
        mask_threshold=args.mask_threshold,
    )
    official_test = discover_split_items(
        data_root=data_root,
        split_name="test",
        mask_threshold=args.mask_threshold,
    )

    train, val = stratified_train_val_split(
        official_train,
        ratios=ratios,
        seed=args.seed,
    )
    validate_split(official_train, train, val, official_test)

    payload = {
        "metadata": {
            "dataset": "KolektorSDD2",
            "seed": args.seed,
            "official_protocol": (
                "Official test folder is preserved; only the official train "
                "folder is split into train and validation subsets."
            ),
            "train_val_ratios": asdict(ratios),
            "stratification": "ground-truth defect presence",
            "mask_positive_rule": (
                f"at least one grayscale mask pixel > {args.mask_threshold}"
            ),
            "paths_are_relative_to": str(data_root),
            "corruptions_generated": False,
            "separate_post_split_generated": False,
            "counts": {
                "official_train": summarize(official_train),
                "train": summarize(train),
                "val": summarize(val),
                "official_test": summarize(official_test),
            },
            "sha256_image_lists": {
                "train": sha256_strings(item.image for item in train),
                "val": sha256_strings(item.image for item in val),
                "test": sha256_strings(item.image for item in official_test),
            },
        },
        "train": [asdict(item) for item in train],
        "val": [asdict(item) for item in val],
        "test": [asdict(item) for item in official_test],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Split JSON saved: {output_path}")
    print_summary(official_train, train, val, official_test)


if __name__ == "__main__":
    main()
