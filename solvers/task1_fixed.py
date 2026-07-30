"""Task 1 fixed-piece recognition and saved pose lookup from 2026_new."""

import math
import os

import numpy as np

try:
    import legacy_2026_new as legacy
    from core.piece_action import actions_from_transforms
except ImportError:
    from .. import legacy_2026_new as legacy
    from ..core.piece_action import actions_from_transforms


DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "task1_layout.json",
)


def _rigid_about_centers(angle_degrees, source_center, target_center):
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    transform = np.array(((cosine, -sine, 0.0),
                          (sine, cosine, 0.0),
                          (0.0, 0.0, 1.0)), dtype=np.float64)
    rotated_center = transform[:2, :2].dot(source_center)
    transform[:2, 2] = target_center - rotated_center
    return transform


class Task1FixedSolver:
    """Match exactly four known pieces to a persisted lower-half lookup table."""

    name = "题1-固定"

    def __init__(self, template_path=DEFAULT_TEMPLATE_PATH):
        self.template_path = template_path

    def calibrate(self, rectified_rgb):
        pieces, _binary, _timings = legacy.detect_pieces(
            rectified_rgb, legacy.LOWER_REGION)
        if len(pieces) != 4:
            raise RuntimeError("题1标定需要下半区恰好 4 块，当前检测到 %d 块" % len(pieces))
        layout = legacy.make_layout(pieces)
        legacy.save_layout(layout, self.template_path)
        return layout

    def solve(self, rectified_rgb):
        layout = legacy.load_layout(self.template_path)
        if layout is None:
            layout = self.calibrate(rectified_rgb)
            print("[TASK1] initialized fixed lookup: %s" % self.template_path)
        if int(layout.get("piece_count", 0)) != 4:
            raise RuntimeError("题1固定模板必须包含 4 块")

        pieces, _binary, _timings = legacy.detect_pieces(
            rectified_rgb, legacy.UPPER_REGION)
        if len(pieces) != 4:
            raise RuntimeError("题1需要上半区恰好 4 块，当前检测到 %d 块" % len(pieces))
        assignment, status, score = legacy.match_pieces_to_layout(pieces, layout)
        if status != "MATCH OK":
            raise RuntimeError("题1固定碎片匹配失败: %s" % status)

        targets = legacy.layout_polygons(layout)
        transforms = []
        for piece, slot in zip(pieces, assignment):
            target = targets[slot]
            angle = legacy.polygon_rotation_to_target(piece, target)
            transforms.append(_rigid_about_centers(
                angle,
                legacy.polygon_centroid(piece).astype(np.float64),
                legacy.polygon_centroid(target).astype(np.float64),
            ))
        actions = actions_from_transforms(
            pieces, transforms, mm_per_pixel=legacy.MM_PER_PIXEL,
            confidence=max(0.0, 1.0 - float(score)),
        )
        return actions, {
            "pieces": pieces,
            "transforms": transforms,
            "assignment": assignment,
            "match_score": score,
        }
