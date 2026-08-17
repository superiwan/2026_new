"""Run one bounded poker replay without camera, display, touch, or UART."""

import argparse
import json
import os
import time

import cv2
import numpy as np

import legacy_2026_new as legacy

try:
    from core.poker_corner_runtime import PokerCornerRuntime
except ImportError:
    PokerCornerRuntime = None
from solvers.poker_layout_selector import (
    _render_candidate,
    build_layout_candidates,
    collect_corner_mark_evidence,
    same_shape_pairs,
)
from solvers import task2_config
from solvers.task3_poker import Task3PokerSolver, detect_poker_pieces


def _render_layout(image, pieces, transforms):
    height, width = image.shape[:2]
    canvas = np.zeros_like(image)
    coverage = np.zeros((height, width), np.uint8)
    for piece, transform in zip(pieces, transforms):
        polygon = cv2.perspectiveTransform(
            np.asarray(piece, np.float32)[None],
            np.asarray(transform, np.float32))[0]
        piece_mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(piece_mask, [np.round(polygon).astype(np.int32)], 255)
        warped = cv2.warpPerspective(
            image, np.asarray(transform, np.float32), (width, height))
        canvas[piece_mask != 0] = warped[piece_mask != 0]
        coverage[piece_mask != 0] = 255
    return canvas, coverage


def _mark_summary(mark):
    return {
        "piece_index": int(mark["piece_index"]),
        "confidence": round(float(mark["confidence"]), 4),
        "center": [round(float(value), 2) for value in mark["center"]],
    }


def _read_bgr(path):
    encoded = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--candidate-output-dir")
    parser.add_argument("--force-best-candidate", action="store_true")
    parser.add_argument("--no-timeout", action="store_true")
    args = parser.parse_args()

    if args.no_timeout:
        task2_config.SOLVE_TIMEOUT_SECONDS = None

    bgr = _read_bgr(args.image)
    if bgr is None:
        raise RuntimeError("cannot read image: %s" % args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source_quad = None
    if args.raw:
        source_quad, homography = legacy.detect_a4(rgb)
        if homography is None:
            raise RuntimeError("cannot detect green A4 in raw frame")
        rgb = legacy.warp_a4(rgb, homography)
    detector = None
    if not args.no_model:
        if PokerCornerRuntime is None:
            raise RuntimeError("poker corner runtime is unavailable")
        detector = PokerCornerRuntime.load()

    started = time.monotonic()
    pieces, mask = detect_poker_pieces(rgb)
    marks = collect_corner_mark_evidence(detector, rgb, pieces)
    payload = {
        "image": args.image,
        "pieces": len(pieces),
        "vertices": [len(piece) for piece in pieces],
        "same_shape_pairs": same_shape_pairs(pieces),
        "marks": [_mark_summary(mark) for mark in marks],
        "detect_seconds": round(time.monotonic() - started, 4),
    }
    if source_quad is not None:
        payload["a4_quad_px"] = np.round(source_quad, 2).tolist()
    if args.solve:
        actions, diagnostics = Task3PokerSolver(
            corner_evidence_detector=detector,
            require_disambiguation=True,
            return_best_candidate_on_failure=args.force_best_candidate,
        ).solve_detected(rgb, pieces, mask)
        payload.update({
            "actions": len(actions),
            "fill_ratio": diagnostics.get("fill_ratio"),
            "topology_path": diagnostics.get("topology_path"),
            "timed_out": bool(diagnostics.get("timed_out")),
            "best_effort": bool(diagnostics.get("best_effort")),
            "fallback_reason": diagnostics.get("fallback_reason"),
            "selected_assignment": diagnostics.get("selected_assignment"),
            "same_shape_pair": diagnostics.get("same_shape_pair"),
            "candidate_scores": diagnostics.get("candidate_scores"),
            "solve_seconds": round(time.monotonic() - started, 4),
        })
        if args.output:
            rendered, coverage = _render_layout(
                rgb, pieces, diagnostics["transforms"])
            rendered[coverage == 0] = (30, 30, 30)
            if not cv2.imwrite(
                    args.output,
                    cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)):
                raise RuntimeError("cannot write output: %s" % args.output)
            payload["output"] = args.output
        if args.candidate_output_dir and diagnostics.get("same_shape_pair"):
            os.makedirs(args.candidate_output_dir, exist_ok=True)
            candidates = build_layout_candidates(
                pieces,
                diagnostics.get(
                    "texture_base_transforms", diagnostics["transforms"]),
                tuple(diagnostics["same_shape_pair"]),
            )
            candidate_outputs = []
            for index, candidate in enumerate(candidates):
                rendered, rendered_mask = _render_candidate(
                    rgb, pieces, candidate["transforms"], long_side=480)
                if rendered is None:
                    continue
                rendered[rendered_mask == 0] = (30, 30, 30)
                label = candidate["assignment"]
                output_path = os.path.join(
                    args.candidate_output_dir, "%s_gapless.png" % label)
                if not cv2.imwrite(
                        output_path,
                        cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(
                        "cannot write candidate output: %s" % output_path)
                candidate_outputs.append(output_path)
            payload["candidate_outputs"] = candidate_outputs
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
