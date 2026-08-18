"""Create a reproducible train/validation/test split for Magnetic Tile Defect.

The script performs only dataset discovery and split generation. It does not
resize images, generate corrupted images, or create separate post-processing
splits. Network input resizing is handled by the training pipeline.

Split policy
------------
- Default ratio: 64% train / 16% validation / 20% test.
- Stratification unit: the six dataset domains
  (five defect types and MT_Free).
- Paths stored in JSON are relative to ``--data-root``.
- Ground-truth masks are checked to determine whether each image contains
  positive defect pixels.

Example
-------
python data_split/generate_split_magnetic.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "data" / "magnetic-tile-defect-datasets.-master"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "splits" / "magnetic_tile_split.json"

DOMAINS: tuple[str, ...] = (
    "MT_Blowhole",
    "MT_Break",
    "MT_Crack",
    "MT_Fray",
    "MT_Uneven",
    "MT_Free",
)
NORMAL_DOMAIN = "MT_Free"


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.64
    val: float = 0.16
    test: float = 0.20

    def validate(self) -> None:
        values = (self.train, self.val, self.test)
        if any(value <= 0.0 for value in values):
            raise ValueError(f"All split ratios must be positive: {values}")
        if not np.isclose(sum(values), 1.0, atol=1e-12):
            raise ValueError(f"Split ratios must sum to 1.0: {values}")

    def as_tuple(self) -> tuple[float, float, float]:
        self.validate()
        return self.train, self.val, self.test


@dataclass(frozen=True)
class Item:
    image: str
    mask: str
    domain: str
    has_defect: bool
    original_height: int
    original_width: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "mask": self.mask,
            "domain": self.domain,
            "has_defect": self.has_defect,
            "original_height": self.original_height,
            "original_width": self.original_width,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a stratified Magnetic Tile train/val/test split."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Root directory containing MT_Blowhole, ..., MT_Free. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.64)
    parser.add_argument("--val-ratio", type=float, default=0.16)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="A mask pixel greater than this value is treated as defect.",
    )
    parser.add_argument(
        "--strict-domain-check",
        action="store_true",
        help=(
            "Fail when MT_Free contains positive mask pixels or a defect domain "
            "contains an empty mask. Without this flag, inconsistencies are warned."
        ),
    )
    return parser.parse_args()


def stable_seed(base_seed: int, *parts: str) -> int:
    text = "||".join([str(base_seed), *parts]).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16)


def sha256_strings(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def to_relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside data root: {path}") from exc


def find_mask(image_path: Path) -> Path:
    mask_path = image_path.with_suffix(".png")
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Mask not found for image: {image_path}\nExpected: {mask_path}"
        )
    return mask_path


def inspect_mask(mask_path: Path, threshold: int) -> bool:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return bool((mask > threshold).any())


def inspect_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return height, width


def discover_items(
    data_root: Path,
    mask_threshold: int,
    strict_domain_check: bool,
) -> list[Item]:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    items: list[Item] = []
    inconsistencies: list[str] = []

    for domain in DOMAINS:
        images_dir = data_root / domain / "Imgs"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {images_dir}")

        image_paths = sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        )
        if not image_paths:
            raise RuntimeError(f"No JPG images found in: {images_dir}")

        for image_path in image_paths:
            mask_path = find_mask(image_path)
            has_defect = inspect_mask(mask_path, mask_threshold)
            height, width = inspect_image_size(image_path)

            expected_defect = domain != NORMAL_DOMAIN
            if has_defect != expected_defect:
                inconsistencies.append(
                    f"domain={domain}, image={image_path.name}, "
                    f"mask_has_defect={has_defect}"
                )

            items.append(
                Item(
                    image=to_relative_posix(image_path, data_root),
                    mask=to_relative_posix(mask_path, data_root),
                    domain=domain,
                    has_defect=has_defect,
                    original_height=height,
                    original_width=width,
                )
            )

    if inconsistencies:
        message = (
            f"Found {len(inconsistencies)} domain/mask inconsistencies.\n"
            + "\n".join(f"  - {line}" for line in inconsistencies[:20])
        )
        if len(inconsistencies) > 20:
            message += f"\n  ... and {len(inconsistencies) - 20} more"
        if strict_domain_check:
            raise RuntimeError(message)
        print(f"[WARNING] {message}")

    return items


def allocate_counts(n: int, ratios: Sequence[float]) -> tuple[int, ...]:
    """Allocate integer split sizes using the largest-remainder method."""
    raw = [n * ratio for ratio in ratios]
    counts = [int(np.floor(value)) for value in raw]
    remainder = n - sum(counts)

    order = sorted(
        range(len(ratios)),
        key=lambda idx: (raw[idx] - counts[idx], ratios[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    # Each MTD domain is large enough for all three subsets. This guard makes
    # accidental empty domain/split cells explicit rather than silently hiding them.
    if n >= len(ratios):
        for idx, ratio in enumerate(ratios):
            if ratio > 0 and counts[idx] == 0:
                donor = max(range(len(counts)), key=counts.__getitem__)
                if counts[donor] <= 1:
                    raise RuntimeError(
                        f"Cannot allocate non-empty splits for n={n}, ratios={ratios}"
                    )
                counts[donor] -= 1
                counts[idx] += 1

    if sum(counts) != n:
        raise AssertionError("Allocated split counts do not sum to the domain size.")
    return tuple(counts)


def stratified_split(
    items: Sequence[Item],
    ratios: SplitRatios,
    seed: int,
) -> dict[str, list[Item]]:
    train_ratio, val_ratio, test_ratio = ratios.as_tuple()
    by_domain: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_domain[item.domain].append(item)

    split: dict[str, list[Item]] = {"train": [], "val": [], "test": []}

    for domain in DOMAINS:
        domain_items = list(by_domain[domain])
        rng = random.Random(stable_seed(seed, "magnetic_tile", domain))
        rng.shuffle(domain_items)

        n_train, n_val, n_test = allocate_counts(
            len(domain_items), (train_ratio, val_ratio, test_ratio)
        )
        split["train"].extend(domain_items[:n_train])
        split["val"].extend(domain_items[n_train : n_train + n_val])
        split["test"].extend(domain_items[n_train + n_val :])

        if len(domain_items) != n_train + n_val + n_test:
            raise AssertionError(f"Split count mismatch for domain: {domain}")

    # Shuffle the concatenated domain blocks while retaining reproducibility.
    for split_name, split_items in split.items():
        rng = random.Random(stable_seed(seed, "magnetic_tile", split_name, "final"))
        rng.shuffle(split_items)

    return split


def validate_split(all_items: Sequence[Item], split: dict[str, list[Item]]) -> None:
    all_ids = {item.image for item in all_items}
    split_ids = {
        name: {item.image for item in items} for name, items in split.items()
    }

    if len(all_ids) != len(all_items):
        raise RuntimeError("Duplicate image paths were found in the dataset.")

    if split_ids["train"] & split_ids["val"]:
        raise RuntimeError("Train and validation sets overlap.")
    if split_ids["train"] & split_ids["test"]:
        raise RuntimeError("Train and test sets overlap.")
    if split_ids["val"] & split_ids["test"]:
        raise RuntimeError("Validation and test sets overlap.")

    union = split_ids["train"] | split_ids["val"] | split_ids["test"]
    if union != all_ids:
        missing = sorted(all_ids - union)
        extra = sorted(union - all_ids)
        raise RuntimeError(
            f"Split coverage mismatch. Missing={missing[:5]}, extra={extra[:5]}"
        )


def summarize(items: Sequence[Item]) -> dict[str, Any]:
    domain_counts = Counter(item.domain for item in items)
    defect_count = sum(item.has_defect for item in items)
    size_counts = Counter(
        f"{item.original_height}x{item.original_width}" for item in items
    )
    return {
        "n_images": len(items),
        "n_defect": defect_count,
        "n_normal": len(items) - defect_count,
        "defect_ratio": defect_count / max(1, len(items)),
        "domain_counts": dict(sorted(domain_counts.items())),
        "original_size_counts": dict(sorted(size_counts.items())),
    }


def build_payload(
    all_items: Sequence[Item],
    split: dict[str, list[Item]],
    ratios: SplitRatios,
    seed: int,
    mask_threshold: int,
) -> dict[str, Any]:
    return {
        "__meta__": {
            "dataset": "Magnetic Tile Defect",
            "seed": seed,
            "split_strategy": "domain-stratified deterministic split",
            "stratification_domains": list(DOMAINS),
            "ratios": {
                "train": ratios.train,
                "val": ratios.val,
                "test": ratios.test,
            },
            "mask_positive_rule": f"any mask pixel > {mask_threshold}",
            "path_policy": "paths are relative to the dataset root",
            "post_split_created": False,
            "corruptions_generated": False,
            "summary": {
                "all": summarize(all_items),
                "train": summarize(split["train"]),
                "val": summarize(split["val"]),
                "test": summarize(split["test"]),
            },
            "sha256_relative_image_paths": {
                name: sha256_strings(item.image for item in items)
                for name, items in split.items()
            },
        },
        "train": [item.to_dict() for item in split["train"]],
        "val": [item.to_dict() for item in split["val"]],
        "test": [item.to_dict() for item in split["test"]],
    }


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path = output_path.expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(f"[OK] Split JSON saved: {output_path}")


def print_summary(payload: dict[str, Any]) -> None:
    print("\nMagnetic Tile split summary")
    print("=" * 72)
    summary = payload["__meta__"]["summary"]
    for split_name in ("all", "train", "val", "test"):
        stats = summary[split_name]
        print(
            f"{split_name:>5}: n={stats['n_images']:4d}, "
            f"defect={stats['n_defect']:4d}, normal={stats['n_normal']:4d}, "
            f"defect_ratio={stats['defect_ratio']:.4f}"
        )
        print(f"       domains={stats['domain_counts']}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    ratios = SplitRatios(
        train=args.train_ratio,
        val=args.val_ratio,
        test=args.test_ratio,
    )
    ratios.validate()

    all_items = discover_items(
        data_root=args.data_root,
        mask_threshold=args.mask_threshold,
        strict_domain_check=args.strict_domain_check,
    )
    split = stratified_split(all_items, ratios=ratios, seed=args.seed)
    validate_split(all_items, split)

    payload = build_payload(
        all_items=all_items,
        split=split,
        ratios=ratios,
        seed=args.seed,
        mask_threshold=args.mask_threshold,
    )
    save_json(args.output, payload)
    print_summary(payload)


if __name__ == "__main__":
    main()
