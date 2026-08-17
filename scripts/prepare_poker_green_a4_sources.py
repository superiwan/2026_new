#!/usr/bin/env python3
"""Copy raw green-A4 fixture frames and rectify them with the project detector."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import legacy_2026_new as legacy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="+", default=["5.jpg", "6.jpg", "7.jpg", "8.jpg"])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing source bundle: {args.output}")
    args.output.mkdir(parents=True)
    raw_copy_dir = args.output / "raw_originals"
    warped_dir = args.output / "warped_a4"
    raw_copy_dir.mkdir()
    warped_dir.mkdir()
    report = {"raw_dir": str(args.raw_dir.resolve()), "warp_size": [legacy.WARP_W, legacy.WARP_H], "frames": {}}
    for name in args.names:
        source = args.raw_dir.resolve() / name
        image_bgr = cv2.imread(str(source))
        if image_bgr is None:
            raise SystemExit(f"cannot read raw frame: {source}")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        quad, homography = legacy.detect_a4(rgb)
        if homography is None:
            raise SystemExit(f"project detect_a4 failed: {source}")
        warped_rgb = legacy.warp_a4(rgb, homography)
        shutil.copy2(source, raw_copy_dir / name)
        warped_path = warped_dir / f"{source.stem}_a4.png"
        cv2.imwrite(str(warped_path), cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR))
        report["frames"][name] = {"quad_px": quad.round(3).tolist(), "warped": str(warped_path)}
    (args.output / "a4_warp_lineage.json").write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
