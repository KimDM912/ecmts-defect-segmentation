#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train binary defect-segmentation models with a controlled U-Net protocol.

Supported datasets
------------------
- magnetic  : Magnetic Tile Defect Dataset
- severstal : Severstal Steel Defect Detection
- kolektor  : KolektorSDD2

Supported ImageNet-pretrained encoders
--------------------------------------
- resnet18
- resnet34
- efficientnet_b0

Default paper protocol
----------------------
- U-Net decoder shared across encoders
- Valid-region binary cross-entropy only
- AdamW, learning rate 3e-4, weight decay 1e-4
- Maximum 60 epochs
- ReduceLROnPlateau on validation BCE
- Early stopping on validation BCE
- Best checkpoint selected by minimum validation BCE
- No boundary loss, logit regularization, Dice loss, or pixel-level class weight
- Natural training-set sampling; no image-level oversampling
- No training augmentation, including horizontal flipping
- ImageNet normalization and pretrained encoder weights
- Test split is not evaluated in this script

Examples
--------
python training/train_segmentation.py --dataset magnetic --backbone resnet18
python training/train_segmentation.py --dataset severstal --backbone resnet34
python training/train_segmentation.py --dataset kolektor --backbone efficientnet_b0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "magnetic": {
        "display_name": "Magnetic Tile Defect Dataset",
        "data_root": PROJECT_ROOT / "data" / "magnetic-tile-defect-datasets.-master",
        "split_json": PROJECT_ROOT / "data" / "splits" / "magnetic_tile_split.json",
        # Original aspect ratio is preserved by letterboxing.
        "input_height": 512,
        "input_width": 512,
        "resolution_note": (
            "Aspect-ratio-preserving resize to a 512x512 canvas; "
            "padded pixels are excluded from the loss and diagnostics."
        ),
    },
    "severstal": {
        "display_name": "Severstal Steel Defect Detection",
        "data_root": PROJECT_ROOT / "data" / "severstal-steel-defect-detection",
        "split_json": PROJECT_ROOT / "data" / "splits" / "severstal_split.json",
        # Native image resolution; no geometric rescaling for standard data.
        "input_height": 256,
        "input_width": 1600,
        "resolution_note": (
            "Native 256x1600 canvas; standard images are not geometrically rescaled."
        ),
    },
    "kolektor": {
        "display_name": "KolektorSDD2",
        "data_root": PROJECT_ROOT / "data" / "kolektor",
        "split_json": PROJECT_ROOT / "data" / "splits" / "kolektor_split.json",
        # Near-native portrait canvas for approximately 230x630 images.
        "input_height": 640,
        "input_width": 384,
        "resolution_note": (
            "Aspect-ratio-preserving resize to a 640x384 canvas; "
            "the approximately 630-pixel native height is retained, "
            "and padded pixels are excluded from loss and diagnostics."
        ),
    },
}

BACKBONE_CHOICES = ("resnet18", "resnet34", "efficientnet_b0")


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    backbone: str
    data_root: str
    split_json: str
    output_dir: str

    seed: int = 2026
    input_height: int = 512
    input_width: int = 512

    batch_size: int = 8
    max_epochs: int = 60
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    minimum_learning_rate: float = 1e-6

    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    mask_threshold: int = 127
    diagnostic_threshold: float = 0.5

    num_workers: int = 4
    pretrained_encoder: bool = True
    save_last_checkpoint: bool = True
    use_tensorboard: bool = True


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path
    has_defect: bool
    image_relative: str
    mask_relative: str


def parse_args(
    default_dataset: str = "magnetic",
    default_backbone: str = "resnet18",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a U-Net binary defect-segmentation model."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_SPECS),
        default=default_dataset,
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONE_CHOICES,
        default=default_backbone,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet-pretrained encoder weights.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    spec = DATASET_SPECS[args.dataset]
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else Path(spec["data_root"]).resolve()
    )
    split_json = (
        args.split_json.expanduser().resolve()
        if args.split_json is not None
        else Path(spec["split_json"]).resolve()
    )

    input_height = (
        int(args.input_height)
        if args.input_height is not None
        else int(spec["input_height"])
    )
    input_width = (
        int(args.input_width)
        if args.input_width is not None
        else int(spec["input_width"])
    )

    if input_height % 32 != 0 or input_width % 32 != 0:
        raise ValueError(
            "Input height and width must both be divisible by 32. "
            f"Received {(input_height, input_width)}."
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            PROJECT_ROOT
            / "checkpoints"
            / args.dataset
            / args.backbone
            / f"seed_{args.seed}"
        ).resolve()
    )


    return TrainConfig(
        dataset=args.dataset,
        backbone=args.backbone,
        data_root=str(data_root),
        split_json=str(split_json),
        output_dir=str(output_dir),
        seed=int(args.seed),
        input_height=input_height,
        input_width=input_width,
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        num_workers=max(0, int(args.num_workers)),
        pretrained_encoder=not bool(args.no_pretrained),
        use_tensorboard=not bool(args.no_tensorboard),
    )


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def load_split_payload(split_json: Path) -> dict[str, Any]:
    if not split_json.is_file():
        raise FileNotFoundError(f"Split JSON not found: {split_json}")
    return json.loads(split_json.read_text(encoding="utf-8"))


def resolve_item_path(
    raw_path: str,
    data_root: Path,
) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (data_root / candidate).resolve()


def item_field(item: dict[str, Any], new_key: str, legacy_key: str) -> str:
    value = item.get(new_key)
    if value is None:
        value = item.get(legacy_key)
    if not value:
        raise KeyError(
            f"Split item is missing both {new_key!r} and {legacy_key!r}: {item}"
        )
    return str(value)


def mask_contains_defect(mask_path: Path, threshold: int) -> bool:
    with Image.open(mask_path) as mask:
        array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return bool((array > threshold).any())


def load_records(
    split_payload: dict[str, Any],
    subset: str,
    data_root: Path,
    mask_threshold: int,
) -> list[SampleRecord]:
    if subset not in split_payload:
        raise KeyError(
            f"Subset {subset!r} is absent from the split JSON. "
            f"Available keys: {list(split_payload)}"
        )

    records: list[SampleRecord] = []
    for item in split_payload[subset]:
        image_raw = item_field(item, "image", "img_path")
        mask_raw = item_field(item, "mask", "gt_path")

        image_path = resolve_item_path(image_raw, data_root)
        mask_path = resolve_item_path(mask_raw, data_root)

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        has_defect = item.get("has_defect")
        if has_defect is None:
            has_defect = mask_contains_defect(mask_path, mask_threshold)

        records.append(
            SampleRecord(
                image_path=image_path,
                mask_path=mask_path,
                has_defect=bool(has_defect),
                image_relative=image_raw,
                mask_relative=mask_raw,
            )
        )

    if not records:
        raise RuntimeError(f"No records found for subset={subset!r}.")
    return records


def letterbox_image_and_mask(
    image_rgb: np.ndarray,
    mask_binary: np.ndarray,
    output_height: int,
    output_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_height, input_width = image_rgb.shape[:2]

    scale = min(
        output_width / max(input_width, 1),
        output_height / max(input_height, 1),
    )
    resized_width = max(1, int(round(input_width * scale)))
    resized_height = max(1, int(round(input_height * scale)))

    resized_image = Image.fromarray(image_rgb).resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    resized_mask = Image.fromarray(
        (mask_binary * 255).astype(np.uint8)
    ).resize(
        (resized_width, resized_height),
        resample=Image.Resampling.NEAREST,
    )

    image_canvas = np.zeros(
        (output_height, output_width, 3),
        dtype=np.uint8,
    )
    mask_canvas = np.zeros(
        (output_height, output_width),
        dtype=np.float32,
    )
    valid_canvas = np.zeros(
        (output_height, output_width),
        dtype=np.float32,
    )

    top = (output_height - resized_height) // 2
    left = (output_width - resized_width) // 2

    image_canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = np.asarray(resized_image, dtype=np.uint8)

    mask_canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = (
        np.asarray(resized_mask, dtype=np.uint8) > 127
    ).astype(np.float32)

    valid_canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = 1.0

    return image_canvas, mask_canvas, valid_canvas


class BinarySegmentationDataset(Dataset):
    IMAGENET_MEAN = np.asarray(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )[:, None, None]
    IMAGENET_STD = np.asarray(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )[:, None, None]

    def __init__(
        self,
        records: Sequence[SampleRecord],
        input_height: int,
        input_width: int,
        mask_threshold: int,
        training: bool,
    ) -> None:
        self.records = list(records)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.mask_threshold = int(mask_threshold)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        record = self.records[index]

        with Image.open(record.image_path) as image:
            image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

        with Image.open(record.mask_path) as mask:
            mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)

        if mask_array.shape != image_rgb.shape[:2]:
            raise ValueError(
                "Image and mask spatial dimensions differ: "
                f"image={image_rgb.shape[:2]}, mask={mask_array.shape}, "
                f"image_path={record.image_path}, mask_path={record.mask_path}"
            )

        mask_binary = (mask_array > self.mask_threshold).astype(np.float32)

        image_rgb, mask_binary, valid_mask = letterbox_image_and_mask(
            image_rgb=image_rgb,
            mask_binary=mask_binary,
            output_height=self.input_height,
            output_width=self.input_width,
        )


        image_chw = (
            image_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        )
        image_chw = (
            image_chw - self.IMAGENET_MEAN
        ) / self.IMAGENET_STD

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_chw)
        ).float()
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask_binary[None, ...])
        ).float()
        valid_tensor = torch.from_numpy(
            np.ascontiguousarray(valid_mask[None, ...])
        ).float()

        return image_tensor, mask_tensor, valid_tensor


class ConvNormActivation(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormActivation(input_channels, output_channels),
            ConvNormActivation(output_channels, output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.refine = DoubleConv(
            input_channels + skip_channels,
            output_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat((x, skip), dim=1)
        return self.refine(x)


class ResNetEncoder(nn.Module):
    def __init__(self, name: str, pretrained: bool) -> None:
        super().__init__()

        if name == "resnet18":
            weights = (
                torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            base = torchvision.models.resnet18(weights=weights)
        elif name == "resnet34":
            weights = (
                torchvision.models.ResNet34_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            base = torchvision.models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet encoder: {name}")

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.deep_channels = 512
        self.skip_channels = (256, 128, 64, 64)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x0 = self.stem(x)              # 1/2
        x1 = self.layer1(self.pool(x0))  # 1/4
        x2 = self.layer2(x1)           # 1/8
        x3 = self.layer3(x2)           # 1/16
        x4 = self.layer4(x3)           # 1/32
        return x4, [x3, x2, x1, x0]


class EfficientNetB0Encoder(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = (
            torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        base = torchvision.models.efficientnet_b0(weights=weights)
        self.features = base.features

        self.deep_channels = 1280
        self.skip_channels = (112, 40, 24, 16)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outputs: list[torch.Tensor] = []
        for block in self.features:
            x = block(x)
            outputs.append(x)

        skip_half = outputs[1]       # 1/2, 16
        skip_quarter = outputs[2]    # 1/4, 24
        skip_eighth = outputs[3]     # 1/8, 40
        skip_sixteenth = outputs[5]  # 1/16, 112
        deep = outputs[8]            # 1/32, 1280

        return deep, [
            skip_sixteenth,
            skip_eighth,
            skip_quarter,
            skip_half,
        ]


class EncoderUNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained_encoder: bool,
    ) -> None:
        super().__init__()

        if backbone in {"resnet18", "resnet34"}:
            self.encoder = ResNetEncoder(
                name=backbone,
                pretrained=pretrained_encoder,
            )
        elif backbone == "efficientnet_b0":
            self.encoder = EfficientNetB0Encoder(
                pretrained=pretrained_encoder,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        skip_channels = self.encoder.skip_channels
        self.center = DoubleConv(self.encoder.deep_channels, 512)
        self.decoder_16 = DecoderBlock(512, skip_channels[0], 256)
        self.decoder_8 = DecoderBlock(256, skip_channels[1], 128)
        self.decoder_4 = DecoderBlock(128, skip_channels[2], 64)
        self.decoder_2 = DecoderBlock(64, skip_channels[3], 64)
        self.segmentation_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        deep, skips = self.encoder(x)

        x = self.center(deep)
        x = self.decoder_16(x, skips[0])
        x = self.decoder_8(x, skips[1])
        x = self.decoder_4(x, skips[2])
        x = self.decoder_2(x, skips[3])
        x = self.segmentation_head(x)
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return x


def masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (valid_mask > 0.5).to(logits.dtype)
    loss_map = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    loss_sum = (loss_map * valid).sum()
    valid_count = valid.sum().clamp_min(1.0)
    loss = loss_sum / valid_count
    return loss, loss_sum.detach(), valid_count.detach()


@torch.no_grad()
def update_confusion_counts(
    totals: dict[str, float],
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
) -> None:
    predictions = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    valid = valid_mask >= 0.5

    totals["tp"] += float((predictions & truth & valid).sum().item())
    totals["fp"] += float((predictions & ~truth & valid).sum().item())
    totals["fn"] += float((~predictions & truth & valid).sum().item())
    totals["tn"] += float((~predictions & ~truth & valid).sum().item())


def metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    f2 = (5.0 * tp) / (5.0 * tp + 4.0 * fn + fp + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "dice_at_0_5": dice,
        "f2_at_0_5": f2,
        "iou_at_0_5": iou,
        "tp_at_0_5": tp,
        "fp_at_0_5": fp,
        "fn_at_0_5": fn,
        "tn_at_0_5": tn,
    }


def make_loader(
    dataset: Dataset,
    config: TrainConfig,
    training: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed + (0 if training else 10_000))

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }

    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    kwargs["shuffle"] = training

    return DataLoader(**kwargs)



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    amp_enabled = device.type == "cuda"

    total_loss_sum = 0.0
    total_valid_pixels = 0.0

    for images, masks, valid_masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid_masks = valid_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss, loss_sum, valid_count = masked_binary_cross_entropy(
                logits,
                masks,
                valid_masks,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += float(loss_sum.item())
        total_valid_pixels += float(valid_count.item())

    return total_loss_sum / max(total_valid_pixels, 1.0)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    diagnostic_threshold: float,
) -> dict[str, float]:
    model.eval()
    amp_enabled = device.type == "cuda"

    total_loss_sum = 0.0
    total_valid_pixels = 0.0
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

    for images, masks, valid_masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid_masks = valid_masks.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)
            _, loss_sum, valid_count = masked_binary_cross_entropy(
                logits,
                masks,
                valid_masks,
            )

        total_loss_sum += float(loss_sum.item())
        total_valid_pixels += float(valid_count.item())
        update_confusion_counts(
            totals=counts,
            logits=logits,
            targets=masks,
            valid_mask=valid_masks,
            threshold=diagnostic_threshold,
        )

    result = {
        "validation_bce": (
            total_loss_sum / max(total_valid_pixels, 1.0)
        )
    }
    result.update(metrics_from_counts(counts))
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    config: TrainConfig,
    epoch: int,
    best_validation_bce: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_validation_bce": float(best_validation_bce),
            "config": asdict(config),
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        path,
    )


def write_history_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_records(records: Sequence[SampleRecord]) -> dict[str, Any]:
    defect = sum(record.has_defect for record in records)
    return {
        "n": len(records),
        "defect": defect,
        "normal": len(records) - defect,
        "defect_ratio": defect / len(records) if records else 0.0,
    }


def main(
    default_dataset: str = "magnetic",
    default_backbone: str = "resnet18",
) -> None:
    args = parse_args(
        default_dataset=default_dataset,
        default_backbone=default_backbone,
    )
    config = build_config(args)

    set_reproducibility(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    data_root = Path(config.data_root)
    split_json = Path(config.split_json)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_payload = load_split_payload(split_json)
    train_records = load_records(
        split_payload=split_payload,
        subset="train",
        data_root=data_root,
        mask_threshold=config.mask_threshold,
    )
    validation_records = load_records(
        split_payload=split_payload,
        subset="val",
        data_root=data_root,
        mask_threshold=config.mask_threshold,
    )

    test_count = len(split_payload.get("test", []))

    train_dataset = BinarySegmentationDataset(
        records=train_records,
        input_height=config.input_height,
        input_width=config.input_width,
        mask_threshold=config.mask_threshold,
        training=True,
    )
    validation_dataset = BinarySegmentationDataset(
        records=validation_records,
        input_height=config.input_height,
        input_width=config.input_width,
        mask_threshold=config.mask_threshold,
        training=False,
    )

    train_loader = make_loader(
        dataset=train_dataset,
        config=config,
        training=True,
    )
    validation_loader = make_loader(
        dataset=validation_dataset,
        config=config,
        training=False,
    )

    model = EncoderUNet(
        backbone=config.backbone,
        pretrained_encoder=config.pretrained_encoder,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    env = environment_info(device)
    metadata = {
        "environment": env,
        "config": asdict(config),
        "dataset_display_name": DATASET_SPECS[config.dataset]["display_name"],
        "resolution_note": DATASET_SPECS[config.dataset]["resolution_note"],
        "data_counts": {
            "train": summarize_records(train_records),
            "validation": summarize_records(validation_records),
            "test_held_out_not_evaluated": test_count,
        },
        "model_description": (
            f"U-Net with an ImageNet-pretrained {config.backbone} encoder"
            if config.pretrained_encoder
            else f"U-Net with a randomly initialized {config.backbone} encoder"
        ),
        "checkpoint_selection": "exact minimum validation BCE",
        "early_stopping": (
            "Validation BCE with min_delta=1e-4 and patience=10 epochs."
        ),
        "training_sampling": "Natural training-set sampling without oversampling.",
        "training_augmentation": "No training augmentation.",
        "test_use": (
            "The held-out test subset is not evaluated during model training."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 88)
    print(
        f"[MODEL] U-Net encoder={config.backbone} "
        f"pretrained={config.pretrained_encoder}"
    )
    print(
        f"[DATA] dataset={config.dataset} "
        f"train={len(train_records)} "
        f"val={len(validation_records)} "
        f"test_held_out={test_count}"
    )
    print(
        f"[INPUT] {config.input_height}x{config.input_width} "
        "(aspect-ratio-preserving letterbox; padding excluded)"
    )
    print("[LOSS] valid-region BCE only")
    print("[SAMPLE] natural training-set sampling")
    print("[AUG] none")
    print(
        f"[OPT] AdamW lr={config.learning_rate:.2e} "
        f"weight_decay={config.weight_decay:.2e}"
    )
    print(
        "[SELECT] exact minimum validation BCE; "
        "threshold-0.5 metrics are diagnostic only"
    )
    print(
        f"[STOP] max_epochs={config.max_epochs}, "
        f"early_patience={config.early_stopping_patience}, "
        f"min_delta={config.early_stopping_min_delta}"
    )
    print(f"[DEVICE] {device} | GPU={env['gpu_name']}")
    print(f"[OUTPUT] {output_dir}")
    print("=" * 88)

    writer = (
        SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        if config.use_tensorboard
        else None
    )

    best_validation_bce = float("inf")
    early_stop_reference_bce = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    training_start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()

        train_bce = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        validation = validate_one_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            diagnostic_threshold=config.diagnostic_threshold,
        )

        validation_bce = float(validation["validation_bce"])
        scheduler.step(validation_bce)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        # Save the numerically lowest validation-BCE checkpoint.
        checkpoint_improved = validation_bce < best_validation_bce

        if checkpoint_improved:
            best_validation_bce = validation_bce
            best_epoch = epoch
            save_checkpoint(
                path=output_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                best_validation_bce=best_validation_bce,
            )

        # Early stopping uses an independent minimum-improvement criterion.
        early_stop_improved = (
            validation_bce
            < early_stop_reference_bce - config.early_stopping_min_delta
        )

        if early_stop_improved:
            early_stop_reference_bce = validation_bce
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.save_last_checkpoint:
            save_checkpoint(
                path=output_dir / "last.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                best_validation_bce=best_validation_bce,
            )

        epoch_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_bce": train_bce,
            **validation,
            "best_validation_bce": best_validation_bce,
            "early_stop_reference_bce": early_stop_reference_bce,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        write_history_csv(output_dir / "history.csv", history)

        if writer is not None:
            writer.add_scalar("loss/train_bce", train_bce, epoch)
            writer.add_scalar(
                "loss/validation_bce",
                validation_bce,
                epoch,
            )
            writer.add_scalar(
                "optimization/learning_rate",
                learning_rate,
                epoch,
            )
            writer.add_scalar(
                "diagnostic/f2_at_0_5",
                validation["f2_at_0_5"],
                epoch,
            )
            writer.add_scalar(
                "diagnostic/dice_at_0_5",
                validation["dice_at_0_5"],
                epoch,
            )
            writer.add_scalar(
                "diagnostic/precision_at_0_5",
                validation["precision_at_0_5"],
                epoch,
            )
            writer.add_scalar(
                "diagnostic/recall_at_0_5",
                validation["recall_at_0_5"],
                epoch,
            )
            writer.flush()

        print(
            f"[E{epoch:03d}] "
            f"lr={learning_rate:.2e} "
            f"train_bce={train_bce:.6f} "
            f"val_bce={validation_bce:.6f} "
            f"F2@0.5={validation['f2_at_0_5']:.4f} "
            f"Dice@0.5={validation['dice_at_0_5']:.4f} "
            f"P@0.5={validation['precision_at_0_5']:.4f} "
            f"R@0.5={validation['recall_at_0_5']:.4f} "
            f"best_epoch={best_epoch} "
            f"no_improve={epochs_without_improvement}/"
            f"{config.early_stopping_patience} "
            f"time={epoch_seconds / 60.0:.2f}m"
        )

        if (
            epochs_without_improvement
            >= config.early_stopping_patience
        ):
            print(
                f"[EARLY STOP] No validation-BCE improvement greater than "
                f"{config.early_stopping_min_delta} for "
                f"{config.early_stopping_patience} epochs."
            )
            break

    if writer is not None:
        writer.close()

    total_seconds = time.time() - training_start
    summary = {
        **metadata,
        "training_result": {
            "best_epoch": best_epoch,
            "best_validation_bce": best_validation_bce,
            "epochs_completed": len(history),
            "total_training_seconds": total_seconds,
            "early_stopped": len(history) < config.max_epochs,
        },
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 88)
    print("[OK] Training completed.")
    print(f"[BEST] epoch={best_epoch}, val_bce={best_validation_bce:.8f}")
    print(f"[TIME] {total_seconds / 3600.0:.2f} hours")
    print(f"[CHECKPOINT] {output_dir / 'best.pth'}")
    print("[TEST] Held-out test data were not evaluated by this script.")
    print("=" * 88)


if __name__ == "__main__":
    main()
