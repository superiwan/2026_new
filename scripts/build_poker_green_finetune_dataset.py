#!/usr/bin/env python3
"""Build v3 green-A4 fine-tune data from manually reviewed warped sources."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from build_poker_real_finetune_dataset import transform, yolo_lines


# xyxy pixels in 420x594 images created by prepare_poker_green_a4_sources.py.
# Each entry was manually drawn around a complete physical card corner mark.
# 5/6 are JOKER marks; 7 is 10H; 8 is 9S. No detector output was consulted.
GREEN_MANUAL_BOXES_PX: dict[str, list[tuple[float, float, float, float]]] = {
    "5_a4.png": [(229, 54, 296, 93), (71, 196, 148, 231)],
    "6_a4.png": [(217, 400, 291, 439), (98, 478, 192, 516)],
    "7_a4.png": [(286, 27, 331, 79), (214, 203, 257, 253)],
    "8_a4.png": [(103, 153, 148, 191), (270, 66, 312, 111)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--green-sources", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--black-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--green-train-augmentations", type=int, default=60)
    parser.add_argument("--green-val-augmentations", type=int, default=15)
    parser.add_argument("--black-keep-per-source", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260804)
    return parser.parse_args()


def copy_pair(image: Path, label: Path, output_images: Path, output_labels: Path, name: str) -> None:
    shutil.copy2(image, output_images / f"{name}{image.suffix.lower()}")
    shutil.copy2(label, output_labels / f"{name}.txt")


def write_green(
    source_dir: Path,
    count: int,
    output_images: Path,
    output_labels: Path,
    audit_dir: Path,
    seed: int,
) -> int:
    written = 0
    for source_index, (filename, boxes) in enumerate(GREEN_MANUAL_BOXES_PX.items()):
        source = source_dir / filename
        image = cv2.imread(str(source))
        if image is None or image.shape[:2] != (594, 420):
            raise SystemExit(f"expected readable 420x594 warped source: {source}")
        manual_boxes = np.asarray(boxes, dtype=np.float32)
        shutil.copy2(source, audit_dir / filename)
        (audit_dir / f"{source.stem}.txt").write_text(yolo_lines(manual_boxes, 420, 594), encoding="ascii")
        review = image.copy()
        for x1, y1, x2, y2 in manual_boxes.astype(int):
            cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.imwrite(str(audit_dir / f"{source.stem}_review.png"), review)
        for index in range(count):
            augmented, transformed_boxes = transform(
                image,
                manual_boxes,
                random.Random(seed + source_index * 10_000 + index),
            )
            name = f"green_{source.stem}_{index:03d}"
            cv2.imwrite(str(output_images / f"{name}.jpg"), augmented, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (output_labels / f"{name}.txt").write_text(yolo_lines(transformed_boxes, 420, 594), encoding="ascii")
            written += 1
    return written


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {args.output}")
    green_sources = args.green_sources.resolve()
    synthetic = args.synthetic.resolve()
    black_source = args.black_source.resolve()
    if not (synthetic / "data.yaml").is_file() or not (black_source / "data.yaml").is_file():
        raise SystemExit("synthetic and black-source datasets must contain data.yaml")
    args.output.mkdir(parents=True)
    audit_dir = args.output / "manual_audit_green_sources"
    audit_dir.mkdir()
    stats: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        output_images = args.output / "images" / split
        output_labels = args.output / "labels" / split
        output_images.mkdir(parents=True)
        output_labels.mkdir(parents=True)
        synthetic_count = 0
        for image in sorted((synthetic / "images" / split).glob("*.jpg")):
            copy_pair(image, synthetic / "labels" / split / f"{image.stem}.txt", output_images, output_labels, f"synth_{image.stem}")
            synthetic_count += 1
        green_count = black_count = 0
        if split == "train":
            green_count = write_green(green_sources / "warped_a4", args.green_train_augmentations, output_images, output_labels, audit_dir, args.seed)
            black_images = sorted((black_source / "images" / "train").glob("real_*.jpg"))
            for image in black_images[: len(GREEN_MANUAL_BOXES_PX) * args.black_keep_per_source]:
                copy_pair(image, black_source / "labels" / "train" / f"{image.stem}.txt", output_images, output_labels, f"black_{image.stem}")
                black_count += 1
        elif split == "val":
            green_count = write_green(green_sources / "warped_a4", args.green_val_augmentations, output_images, output_labels, audit_dir, args.seed + 5_000_000)
        stats[split] = {"synthetic_images": synthetic_count, "green_augmented_images": green_count, "black_retained_images": black_count, "total_images": synthetic_count + green_count + black_count}
    root = args.output.resolve().as_posix()
    (args.output / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: corner_mark\n", encoding="ascii")
    lineage = {
        "green_manual_boxes_px": GREEN_MANUAL_BOXES_PX,
        "green_source_bundle": str(green_sources),
        "synthetic_source": str(synthetic),
        "black_source": str(black_source),
        "augmentation": "global perspective, rotation, scale, gain, bias, blur, noise; no horizontal or vertical mirror",
        "warning": "All green-derived train/val images originate from four raw frames. Green validation is source-overlapping and is not independent real-scene evaluation.",
        "stats": stats,
    }
    (args.output / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="ascii")
    print(json.dumps(lineage, indent=2))


if __name__ == "__main__":
    main()
