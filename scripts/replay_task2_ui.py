#!/usr/bin/env python3
"""Replay recorded Task 2 UI screenshots through the locked A4 geometry."""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = next(
    (candidate for candidate in (
        SCRIPT_DIR, os.path.dirname(SCRIPT_DIR),
    ) if os.path.exists(os.path.join(candidate, "legacy_2026_new.py"))),
    os.path.dirname(SCRIPT_DIR),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import legacy_2026_new as legacy
from solvers import task2_config
from solvers.task2_white import Task2WhiteSolver


PRESETS = {
    "20260801111405.jpeg": {
        "classification": "device_ui_screenshot_raw_scene",
        "quad": ((102, 384), (100, 112), (484, 106), (490, 382)),
    },
    "20260801111515.jpeg": {
        "classification": "device_ui_screenshot_preview_occluded",
        "quad": ((102, 384), (100, 112), (484, 106), (490, 382)),
    },
    "20260801132747.jpeg": {
        "classification": "device_ui_screenshot_raw_scene",
        "quad": ((98, 382), (100, 110), (485, 106), (490, 384)),
    },
}


def replay(path, timeout_seconds):
    started = time.perf_counter()
    name = os.path.basename(path)
    preset = PRESETS.get(name)
    if preset is None:
        raise RuntimeError("no locked Q0-Q3 preset for %s" % name)
    result = {
        "image": name,
        "classification": preset["classification"],
        "selected_path": "locked_ui_quad",
    }
    if preset["classification"].endswith("preview_occluded"):
        result.update({
            "status": "SKIPPED",
            "reason": "LIVE preview covers the A4 pixels and creates false pieces",
            "elapsed_s": round(time.perf_counter() - started, 4),
        })
        return result

    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError("cannot read %s" % path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    auto_quad, auto_homography = legacy.detect_a4(rgb)
    destination = np.float32((
        (0, 0), (legacy.WARP_W - 1, 0),
        (legacy.WARP_W - 1, legacy.WARP_H - 1),
        (0, legacy.WARP_H - 1),
    ))
    homography = cv2.getPerspectiveTransform(
        np.float32(preset["quad"]), destination)
    rectified = legacy.warp_a4(rgb, homography)
    pieces, binary, timings = legacy.detect_pieces(rectified)
    result.update({
        "a4_auto": auto_homography is not None,
        "a4_auto_quad": (None if auto_quad is None
                         else np.round(auto_quad, 2).tolist()),
        "detected_count": len(pieces),
        "vertices": [len(piece) for piece in pieces],
        "areas_px2": [round(abs(cv2.contourArea(
            piece.astype(np.float32))), 2) for piece in pieces],
        "detect_timings_ms": timings,
    })
    configured_timeout = task2_config.SOLVE_TIMEOUT_SECONDS
    task2_config.SOLVE_TIMEOUT_SECONDS = timeout_seconds
    try:
        actions, diagnostics = Task2WhiteSolver().solve_detected(
            rectified, pieces, binary, timings)
    finally:
        task2_config.SOLVE_TIMEOUT_SECONDS = configured_timeout
    result.update({
        "status": "SOLVED",
        "action_count": len(actions),
        "fill_ratio": round(float(diagnostics["fill_ratio"]), 6),
        "topology_path": diagnostics.get("topology_path"),
        "elapsed_s": round(time.perf_counter() - started, 4),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    failed = False
    for path in args.images:
        try:
            payload = replay(path, args.timeout_seconds)
        except Exception as error:
            failed = True
            payload = {
                "image": os.path.basename(path),
                "status": "FAILED",
                "error": "%s: %s" % (type(error).__name__, error),
            }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
