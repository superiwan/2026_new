#!/usr/bin/env python3
"""Generate a legal, self-contained YOLO dataset for poker corner marks.

The generated card graphics are drawn from primitive text/suit glyphs.  No card
photo or third-party image asset is used.  It deliberately models the contest
geometry (green A4, white card fragments, arbitrary in-plane rotation) but is
not a substitute for MaixCAM camera images.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAME = "corner_mark"
IMAGE_SIZE = 640
CARD_W, CARD_H = 190, 270
MARK_BOX = (13, 12, 63, 77)
FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=180)
    parser.add_argument("--test", type=int, default=180)
    parser.add_argument("--seed", type=int, default=260804)
    return parser.parse_args()


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def local_mark(rank: str, suit: str, color: tuple[int, int, int]) -> Image.Image:
    mark = Image.new("RGBA", (50, 65), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mark)
    draw.text((2, -2), rank, font=font(24), fill=color, stroke_width=0)
    # A filled vector suit keeps data generation self-contained even when the
    # Windows symbol font is unavailable.
    cx, cy = 16, 43
    if suit in ("H", "D"):
        draw.ellipse((cx - 8, cy - 8, cx, cy), fill=color)
        draw.ellipse((cx, cy - 8, cx + 8, cy), fill=color)
        draw.polygon([(cx - 8, cy - 3), (cx + 8, cy - 3), (cx, cy + 11)], fill=color)
    elif suit == "C":
        for dx, dy in ((-6, 0), (6, 0), (0, -7)):
            draw.ellipse((cx + dx - 6, cy + dy - 6, cx + dx + 6, cy + dy + 6), fill=color)
        draw.polygon([(cx - 2, cy + 5), (cx + 3, cy + 5), (cx + 7, cy + 14), (cx - 6, cy + 14)], fill=color)
    else:
        draw.polygon([(cx, cy - 11), (cx + 9, cy + 1), (cx + 4, cy + 10), (cx, cy + 4), (cx - 4, cy + 10), (cx - 9, cy + 1)], fill=color)
        draw.polygon([(cx - 2, cy + 5), (cx + 3, cy + 5), (cx + 7, cy + 14), (cx - 6, cy + 14)], fill=color)
    return mark


def card_fragment(rng: random.Random, include_corner: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Return BGR+alpha fragment and the full local corner-mark quadrilateral."""
    card = Image.new("RGBA", (CARD_W, CARD_H), (249, 248, 241, 255))
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((1, 1, CARD_W - 2, CARD_H - 2), radius=13, outline=(55, 55, 55, 255), width=2)
    rank = rng.choice(list("A23456789") + ["10", "J", "Q", "K"])
    suit = rng.choice(["S", "H", "D", "C"])
    color = (195, 20, 25) if suit in ("H", "D") else (25, 25, 28)
    mark = local_mark(rank, suit, color)
    card.alpha_composite(mark, (MARK_BOX[0], MARK_BOX[1]))
    rotated = mark.rotate(180)
    card.alpha_composite(rotated, (CARD_W - MARK_BOX[2], CARD_H - MARK_BOX[3]))
    center_suit = local_mark("", suit, color).resize((92, 120))
    card.alpha_composite(center_suit, (CARD_W // 2 - 46, CARD_H // 2 - 60))

    # A contest-like piece: either a rectangle/square containing a physical
    # card corner or an irregular central/interior fragment (negative).
    mask = Image.new("L", (CARD_W, CARD_H), 0)
    mask_draw = ImageDraw.Draw(mask)
    if include_corner:
        side = rng.choice(("tl", "br"))
        if side == "tl":
            x2, y2 = rng.randint(82, 150), rng.randint(105, 190)
            polygon = [(0, 0), (x2, 0), (x2, y2), (0, y2)]
            quad = np.float32([[MARK_BOX[0], MARK_BOX[1]], [MARK_BOX[2], MARK_BOX[1]], [MARK_BOX[2], MARK_BOX[3]], [MARK_BOX[0], MARK_BOX[3]]])
        else:
            x1, y1 = rng.randint(35, 105), rng.randint(70, 155)
            polygon = [(x1, y1), (CARD_W - 1, y1), (CARD_W - 1, CARD_H - 1), (x1, CARD_H - 1)]
            quad = np.float32([[CARD_W - MARK_BOX[2], CARD_H - MARK_BOX[3]], [CARD_W - MARK_BOX[0], CARD_H - MARK_BOX[3]], [CARD_W - MARK_BOX[0], CARD_H - MARK_BOX[1]], [CARD_W - MARK_BOX[2], CARD_H - MARK_BOX[1]]])
    else:
        x1, y1 = rng.randint(50, 90), rng.randint(65, 110)
        x2, y2 = rng.randint(120, 165), rng.randint(145, 215)
        polygon = [(x1, y1), (x2, y1 + rng.randint(-25, 20)), (x2 - rng.randint(0, 28), y2), (x1 + rng.randint(-25, 20), y2 - rng.randint(0, 20))]
        quad = None
    mask_draw.polygon(polygon, fill=255)
    fragment = Image.composite(card, Image.new("RGBA", card.size, (0, 0, 0, 0)), mask)
    return cv2.cvtColor(np.asarray(fragment), cv2.COLOR_RGBA2BGRA), quad


def transform_fragment(
    canvas: np.ndarray,
    fragment: np.ndarray,
    mark_quad: np.ndarray | None,
    rng: random.Random,
) -> np.ndarray | None:
    """Perspective-paste fragment and return transformed mark quadrilateral."""
    h, w = fragment.shape[:2]
    angle = rng.uniform(-175, 175)
    scale = rng.uniform(0.72, 1.25)
    cx, cy = rng.uniform(130, 510), rng.uniform(135, 520)
    base = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    radians = math.radians(angle)
    rotate = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]], dtype=np.float32) * scale
    destination = base @ rotate.T + np.float32([cx, cy])
    destination += np.float32([[rng.uniform(-8, 8), rng.uniform(-8, 8)] for _ in range(4)])
    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(fragment, matrix, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = warped[:, :, 3].astype(np.float32) / 255.0
    shadow = cv2.GaussianBlur((alpha * 120).astype(np.uint8), (0, 0), 6)
    for channel in range(3):
        canvas[:, :, channel] = np.clip(canvas[:, :, channel].astype(np.float32) * (1.0 - shadow.astype(np.float32) / 700.0), 0, 255).astype(np.uint8)
        canvas[:, :, channel] = (warped[:, :, channel].astype(np.float32) * alpha + canvas[:, :, channel].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    if mark_quad is None:
        return None
    return cv2.perspectiveTransform(mark_quad.reshape(1, -1, 2), matrix).reshape(-1, 2)


def scene(rng: random.Random) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), (43, 101, 49), dtype=np.uint8)
    # A4 has the target green-paper aspect ratio and subtle lighting variation.
    paper = np.full((530, 375, 3), (62, 153, 74), dtype=np.uint8)
    yy, xx = np.mgrid[:530, :375]
    illumination = ((xx / 375.0 - 0.5) * rng.uniform(-18, 18) + (yy / 530.0 - 0.5) * rng.uniform(-14, 14)).astype(np.int16)
    paper = np.clip(paper.astype(np.int16) + illumination[:, :, None], 0, 255).astype(np.uint8)
    canvas[55:585, 132:507] = paper
    boxes: list[tuple[float, float, float, float]] = []
    has_positive = rng.random() >= 0.23
    fragment_count = rng.randint(1, 4)
    for index in range(fragment_count):
        include_corner = has_positive and (index == 0 or rng.random() < 0.42)
        fragment, quad = card_fragment(rng, include_corner)
        transformed = transform_fragment(canvas, fragment, quad, rng)
        if transformed is None:
            continue
        min_xy = transformed.min(axis=0)
        max_xy = transformed.max(axis=0)
        x1, y1 = np.maximum(min_xy - 2, 0)
        x2, y2 = np.minimum(max_xy + 2, IMAGE_SIZE - 1)
        if x2 - x1 >= 8 and y2 - y1 >= 8 and x1 >= 2 and y1 >= 2 and x2 < IMAGE_SIZE - 2 and y2 < IMAGE_SIZE - 2:
            boxes.append((float(x1), float(y1), float(x2), float(y2)))
    # Camera-like degradation after labels are fixed.
    if rng.random() < 0.55:
        canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.15, 1.15))
    if rng.random() < 0.70:
        gain = rng.uniform(0.78, 1.22)
        bias = rng.uniform(-18, 18)
        canvas = np.clip(canvas.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.42:
        noise = rng.normalvariate(0, 1.0)
        canvas = np.clip(canvas.astype(np.float32) + np.random.default_rng(rng.randrange(2**32)).normal(noise, rng.uniform(1.0, 4.0), canvas.shape), 0, 255).astype(np.uint8)
    return canvas, boxes


def write_split(root: Path, split: str, count: int, seed: int) -> dict[str, int]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    positive_images = 0
    boxes_total = 0
    for index in range(count):
        rng = random.Random(seed + index)
        image, boxes = scene(rng)
        image_path = image_dir / f"{split}_{index:05d}.jpg"
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(82, 97)])
        lines = []
        for x1, y1, x2, y2 in boxes:
            center_x = (x1 + x2) / (2 * IMAGE_SIZE)
            center_y = (y1 + y2) / (2 * IMAGE_SIZE)
            width = (x2 - x1) / IMAGE_SIZE
            height = (y2 - y1) / IMAGE_SIZE
            lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        (label_dir / f"{split}_{index:05d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
        positive_images += int(bool(boxes))
        boxes_total += len(boxes)
    return {"images": count, "positive_images": positive_images, "negative_images": count - positive_images, "boxes": boxes_total}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {args.output}")
    args.output.mkdir(parents=True)
    statistics = {
        "train": write_split(args.output, "train", args.train, args.seed),
        "val": write_split(args.output, "val", args.val, args.seed + 1_000_000),
        "test": write_split(args.output, "test", args.test, args.seed + 2_000_000),
    }
    dataset_root = args.output.resolve().as_posix()
    (args.output / "data.yaml").write_text(
        f"path: {dataset_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: corner_mark\n",
        encoding="ascii",
    )
    (args.output / "DATASET_CARD.md").write_text(
        "# Synthetic poker-corner dataset\n\n"
        "License: CC0-1.0 for the generated images and annotations. The generator uses only "
        "primitive drawing instructions and Windows-installed fonts; it contains no third-party card images. "
        "This data models green A4 paper and complete card-corner marks but has unavoidable camera/domain "
        "differences. It is valid for pipeline training, not a claim of real-scene accuracy.\n",
        encoding="ascii",
    )
    (args.output / "split_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="ascii")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
