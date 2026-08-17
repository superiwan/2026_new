#!/usr/bin/env python3
"""Fail-fast validation for a one-class YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    data_file = args.data.resolve()
    config = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    root = (data_file.parent / config.get("path", ".")).resolve()
    names = config.get("names", {})
    expected_names = [names[key] for key in sorted(names)] if isinstance(names, dict) else names
    report: dict[str, object] = {"data": str(data_file), "classes": expected_names, "splits": {}, "errors": []}
    for split in ("train", "val", "test"):
        image_dir = root / config[split]
        images = sorted(path for path in image_dir.glob("*.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        labels_dir = root / "labels" / split
        split_stats = {"images": len(images), "labels": 0, "empty_labels": 0, "boxes": 0}
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                report["errors"].append(f"unreadable image: {image_path}")
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                report["errors"].append(f"missing label: {label_path}")
                continue
            split_stats["labels"] += 1
            lines = [line for line in label_path.read_text(encoding="ascii").splitlines() if line.strip()]
            split_stats["empty_labels"] += int(not lines)
            for line_number, line in enumerate(lines, 1):
                fields = line.split()
                if len(fields) != 5:
                    report["errors"].append(f"malformed row {label_path}:{line_number}")
                    continue
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError:
                    report["errors"].append(f"non-numeric row {label_path}:{line_number}")
                    continue
                if class_id < 0 or class_id >= len(expected_names) or any(value <= 0 or value > 1 for value in values):
                    report["errors"].append(f"out-of-range row {label_path}:{line_number}")
                    continue
                split_stats["boxes"] += 1
        report["splits"][split] = split_stats
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(f"dataset validation failed: {len(report['errors'])} errors")


if __name__ == "__main__":
    main()
