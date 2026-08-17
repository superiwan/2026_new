#!/usr/bin/env python3
"""Train and export the fixed-shape poker corner YOLO11n baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--name", default="poker_corner_yolo11n_640_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve() / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"missing data.yaml: {data_yaml}")
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"missing official pretrained model: {model_path}")
    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_yaml),
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        imgsz=640,
        epochs=args.epochs,
        patience=20,
        batch=args.batch,
        workers=args.workers,
        pretrained=True,
        rect=False,
        cache=False,
        seed=260804,
        deterministic=True,
        degrees=18.0,
        translate=0.08,
        scale=0.22,
        shear=2.0,
        perspective=0.0008,
        hsv_h=0.012,
        hsv_s=0.35,
        hsv_v=0.28,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.45,
        close_mosaic=12,
        erasing=0.0,
        device=0,
    )
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    evaluator = YOLO(str(best))
    metrics = evaluator.val(data=str(data_yaml), split="test", imgsz=640, batch=args.batch, workers=args.workers, device=0)
    metric_record = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "best_pt": str(best),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(metric_record, indent=2), encoding="ascii")
    exported = evaluator.export(format="onnx", imgsz=640, dynamic=False, opset=17, simplify=True)
    print(json.dumps({**metric_record, "onnx": str(exported)}, indent=2))


if __name__ == "__main__":
    main()
