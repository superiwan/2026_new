"""Task 2(2): geometry candidates ranked by poker-face seam continuity.

The upstream 2026_vision_v1 repository renders poker textures but its solver is
geometry-only.  This module keeps that topology search and adds the missing
image-based seam score; it is intentionally identified as merged-project code.
"""

import cv2
import numpy as np

try:
    import legacy_2026_new as legacy
    from core.piece_action import actions_from_transforms
    from solvers import task2_config as config
    from solvers import task2_white as geometry
except ImportError:
    from .. import legacy_2026_new as legacy
    from ..core.piece_action import actions_from_transforms
    from . import task2_config as config
    from . import task2_white as geometry


def detect_poker_pieces(rectified_rgb):
    """Segment printed card fragments from the black rectified A4 plane."""
    hsv = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = np.where((value >= 65) | ((value >= 38) & (saturation >= 28)),
                    255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8), iterations=1)
    inset = max(legacy.REGION_MARGIN, 6)
    mask[:inset, :] = 0
    mask[-inset:, :] = 0
    mask[:, :inset] = 0
    mask[:, -inset:] = 0

    a4_area = mask.size
    contours = list(legacy.find_contours(mask))
    contours.sort(key=cv2.contourArea, reverse=True)
    pieces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (a4_area * config.PIECE_MIN_AREA_RATIO <= area
                <= a4_area * config.PIECE_MAX_AREA_RATIO):
            continue
        polygon = legacy.approximate_piece(contour)
        if polygon is not None:
            pieces.append(polygon.astype(np.float64))
        if len(pieces) == 4:
            break
    return pieces, mask


def _sample_rgb(image, points):
    height, width = image.shape[:2]
    points = np.round(points).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return image[points[:, 1], points[:, 0]].astype(np.float32)


def _inside_normal(start, end, polygon):
    vector = end - start
    normal = np.array((-vector[1], vector[0]), dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1e-9)
    midpoint = (start + end) * 0.5
    if np.dot(np.asarray(polygon).mean(axis=0) - midpoint, normal) < 0:
        normal = -normal
    return normal


def seam_texture_cost(rectified_rgb, pieces, match, samples=24):
    """Compare colour and inward gradients along a proposed cut seam."""
    _error, i, _ei, j, _ej = match[:5]
    ia, ib, ja, jb = geometry.match_segments(pieces, match)
    ni = _inside_normal(ia, ib, pieces[i])
    nj = _inside_normal(ja, jb, pieces[j])
    ratios = np.linspace(0.08, 0.92, samples)[:, None]
    points_i = ia + (ib - ia) * ratios
    # Assembly reverses the second edge: ia<->jb and ib<->ja.
    points_j = jb + (ja - jb) * ratios
    near_i = _sample_rgb(rectified_rgb, points_i + ni * 2.0)
    near_j = _sample_rgb(rectified_rgb, points_j + nj * 2.0)
    deep_i = _sample_rgb(rectified_rgb, points_i + ni * 5.0)
    deep_j = _sample_rgb(rectified_rgb, points_j + nj * 5.0)
    colour = np.mean(np.abs(near_i - near_j)) / 255.0
    gradient = np.mean(np.abs((deep_i - near_i) - (deep_j - near_j))) / 255.0
    return float(0.72 * colour + 0.28 * gradient)


class Task3PokerSolver:
    name = "题2-扑克"

    def solve(self, rectified_rgb):
        pieces, mask = detect_poker_pieces(rectified_rgb)
        if not 1 <= len(pieces) <= 4:
            raise RuntimeError("检测到 %d 块扑克碎片，需要 1-4 块" % len(pieces))
        paper = np.int32((((0, 0),), ((config.A4_WARP_WIDTH - 1, 0),),
                          ((config.A4_WARP_WIDTH - 1,
                            config.A4_WARP_HEIGHT - 1),),
                          ((0, config.A4_WARP_HEIGHT - 1),)))

        best = None
        source_area = max(1.0, sum(abs(cv2.contourArea(
            piece.astype(np.float32))) for piece in pieces))
        if len(pieces) == 1:
            transforms, matches, fill_ratio = geometry.solve(pieces, paper)
            texture_cost = 0.0
        else:
            for matches in geometry.matching_sets(
                    pieces, "auto",
                    config.FAST_SEARCH_FULL_CANDIDATES * 2,
                    config.FAST_SEARCH_PARTIAL_CANDIDATES):
                assembled = geometry.assemble_from_matches(pieces, matches)
                if assembled is None or assembled[1] < config.MIN_RECTANGLE_FILL:
                    continue
                geometry_score, fill_ratio, transforms = assembled
                texture_cost = float(np.mean([
                    seam_texture_cost(rectified_rgb, pieces, match)
                    for match in matches
                ]))
                score = geometry_score / source_area + texture_cost * 12.0
                if best is None or score < best[0]:
                    best = (score, fill_ratio, transforms, matches, texture_cost)
            if best is None:
                raise RuntimeError("未找到同时满足几何与扑克纹理连续性的拼接")
            _score, fill_ratio, transforms, matches, texture_cost = best
            transforms = geometry.optimize_pose_graph(pieces, matches, transforms)
            transforms = geometry._target_transform(pieces, transforms, paper)

        confidence = max(0.0, min(1.0, fill_ratio * (1.0 - texture_cost)))
        actions = actions_from_transforms(
            pieces, transforms, mm_per_pixel=0.5, confidence=confidence,
        )
        return actions, {
            "pieces": pieces,
            "transforms": transforms,
            "matches": matches,
            "fill_ratio": fill_ratio,
            "texture_cost": texture_cost,
            "piece_mask": mask,
        }
