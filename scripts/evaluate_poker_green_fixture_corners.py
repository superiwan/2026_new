#!/usr/bin/env python3
"""Diagnostic corner-mark matching on the manually labelled green A4 sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_poker_green_finetune_dataset import GREEN_MANUAL_BOXES_PX
from evaluate_poker_fixture_corners import iou
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.30)
    args = parser.parse_args()
    model = YOLO(str(args.model.resolve()))
    details, total_matches, total_predictions = {}, 0, 0
    for filename, gt_boxes in GREEN_MANUAL_BOXES_PX.items():
        result = model.predict(str((args.sources / filename).resolve()), imgsz=640, conf=args.conf, iou=0.45, device=0, verbose=False)[0]
        predictions = [tuple(map(float, box)) for box in result.boxes.xyxy.cpu().tolist()]
        remaining = set(range(len(gt_boxes)))
        matches = 0
        for prediction in predictions:
            candidates = [(iou(prediction, gt_boxes[index]), index) for index in remaining]
            if candidates:
                score, index = max(candidates)
                if score >= args.match_iou:
                    matches += 1
                    remaining.remove(index)
        details[filename] = {"ground_truth": len(gt_boxes), "predictions": len(predictions), "matched": matches}
        total_matches += matches
        total_predictions += len(predictions)
    report = {
        "model": str(args.model.resolve()),
        "warning": "These four green sources are parents of the v3 fine-tune augmentations. This is a source-overlapping diagnostic, not independent real-scene evaluation.",
        "confidence_threshold": args.conf,
        "match_iou": args.match_iou,
        "totals": {"ground_truth": 8, "predictions": total_predictions, "matched": total_matches},
        "per_fixture": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")
    args.output.write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
