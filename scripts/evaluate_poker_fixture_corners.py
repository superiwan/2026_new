#!/usr/bin/env python3
"""Report detector boxes against manually audited poker fixture corner marks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_poker_real_finetune_dataset import MANUAL_BOXES_PX
from ultralytics import YOLO


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.30)
    args = parser.parse_args()
    model = YOLO(str(args.model.resolve()))
    details = {}
    total_matches = total_predictions = 0
    for filename, gt_boxes in MANUAL_BOXES_PX.items():
        result = model.predict(str((args.fixtures / filename).resolve()), imgsz=640, conf=args.conf, iou=0.45, device=0, verbose=False)[0]
        predictions = [tuple(map(float, box)) for box in result.boxes.xyxy.cpu().tolist()]
        remaining = set(range(len(gt_boxes)))
        matches = 0
        for prediction in predictions:
            candidates = [(iou(prediction, gt_boxes[index]), index) for index in remaining]
            if candidates:
                best_iou, index = max(candidates)
                if best_iou >= args.match_iou:
                    matches += 1
                    remaining.remove(index)
        details[filename] = {"ground_truth": len(gt_boxes), "predictions": len(predictions), "matched": matches}
        total_matches += matches
        total_predictions += len(predictions)
    report = {
        "model": str(args.model.resolve()),
        "warning": "These five manually labelled fixtures are training-source parents for the fine-tune dataset. This is a diagnostic sanity check, not independent real-scene evaluation.",
        "confidence_threshold": args.conf,
        "match_iou": args.match_iou,
        "totals": {"ground_truth": 10, "predictions": total_predictions, "matched": total_matches},
        "per_fixture": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")
    args.output.write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
