"""Task 2(2): poker detection backed by the Task 2(1) topology solver.

Poker fragments keep their own segmentation, while assembly and confidence
use exactly the same implementation as the white-piece mode.
"""

import cv2
import numpy as np


# Follow the upstream simulator's frame-adaptive foreground threshold.  The
# black paper level moves substantially with exposure, while the relative
# card-to-paper contrast remains stable in real captures.
POKER_CARD_GRAY_MIN = 70
POKER_CARD_PAPER_DELTA = 38
POKER_WHITE_GEOMETRY_AREA_TOLERANCE = 0.15
POKER_MAX_PIECE_VERTICES = 8
POKER_WHITE_MAX_PIECE_VERTICES = 5
POKER_NOTCH_REPAIR_TARGET_VERTICES = 5
POKER_NOTCH_MAX_DEPTH_RATIO = 0.06
POKER_NOTCH_MAX_MOUTH_RATIO = 0.20
POKER_COLLINEAR_MAX_DEPTH_RATIO = 0.005
POKER_CARD_SHORT_LONG_RATIO = 5.0 / 7.0
POKER_CARD_ASPECT_REL_TOLERANCE = 0.10
POKER_TEXTURE_EDGE_WEIGHT = 0.04
POKER_TEXTURE_FINALISTS = 12
POKER_TEXTURE_MAX_FILL_LOSS = 0.015


def _sample_rgb(image, points):
    points = np.asarray(points, dtype=np.float64)
    height, width = image.shape[:2]
    x = np.clip(np.round(points[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.round(points[:, 1]).astype(np.int32), 0, height - 1)
    return image[y, x].astype(np.float32)


def _inside_normal(start, end, polygon):
    vector = np.asarray(end, dtype=np.float64) - start
    normal = np.asarray((-vector[1], vector[0]), dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    midpoint = (np.asarray(start, dtype=np.float64) + end) * 0.5
    if np.dot(np.asarray(polygon).mean(axis=0) - midpoint, normal) < 0.0:
        normal = -normal
    return normal


def poker_seam_texture_cost(rectified_rgb, pieces, match, samples=24):
    """Measure colour and inward-gradient continuity across one cut seam."""
    _error, first_index, _first_edge, second_index, _second_edge = match[:5]
    first_start, first_end, second_start, second_end = (
        geometry.match_segments(pieces, match))
    first_normal = _inside_normal(
        first_start, first_end, pieces[first_index])
    second_normal = _inside_normal(
        second_start, second_end, pieces[second_index])
    ratios = np.linspace(0.08, 0.92, samples)[:, None]
    first_points = first_start + (first_end - first_start) * ratios
    second_points = second_end + (second_start - second_end) * ratios
    first_near = _sample_rgb(
        rectified_rgb, first_points + first_normal * 2.0)
    second_near = _sample_rgb(
        rectified_rgb, second_points + second_normal * 2.0)
    first_deep = _sample_rgb(
        rectified_rgb, first_points + first_normal * 5.0)
    second_deep = _sample_rgb(
        rectified_rgb, second_points + second_normal * 5.0)
    colour = float(np.mean(np.abs(first_near - second_near)) / 255.0)
    gradient = float(np.mean(np.abs(
        (first_deep - first_near) - (second_deep - second_near))) / 255.0)
    return 0.72 * colour + 0.28 * gradient


def _poker_texture_priorities(rectified_rgb):
    cache = {}

    def texture_cost(pieces, match):
        key = tuple(match)
        if key not in cache:
            cache[key] = poker_seam_texture_cost(
                rectified_rgb, pieces, match)
        return cache[key]

    def match_priority(pieces, match):
        return (float(match[0])
                + POKER_TEXTURE_EDGE_WEIGHT * texture_cost(pieces, match))

    def topology_priority(pieces, matches):
        if not matches:
            return 0.0
        return float(np.mean([
            texture_cost(pieces, match) for match in matches
        ]))

    return match_priority, topology_priority


def _poker_target_aspect_error(target_size_mm):
    short_side_mm, long_side_mm = sorted(float(value)
                                         for value in target_size_mm)
    if long_side_mm <= 1e-6:
        return float("inf")
    aspect_ratio = short_side_mm / long_side_mm
    return abs(
        aspect_ratio - POKER_CARD_SHORT_LONG_RATIO
    ) / POKER_CARD_SHORT_LONG_RATIO


def _poker_target_aspect_valid(target_size_mm):
    return (_poker_target_aspect_error(target_size_mm)
            <= POKER_CARD_ASPECT_REL_TOLERANCE)

try:
    import legacy_2026_new as legacy
    from solvers import task2_config as config
    from solvers import task2_white as geometry
    from solvers import poker_arc_geometry
    from solvers import poker_layout_selector
except ImportError:
    from .. import legacy_2026_new as legacy
    from . import task2_config as config
    from . import task2_white as geometry
    from . import poker_arc_geometry
    from . import poker_layout_selector


def _piece_areas(pieces):
    return sorted(abs(cv2.contourArea(
        np.asarray(piece, dtype=np.float32))) for piece in pieces)


def _same_piece_geometry(white_pieces, poker_pieces):
    if not (1 <= len(white_pieces) == len(poker_pieces) <= 4):
        return False
    if any(len(piece) > POKER_WHITE_MAX_PIECE_VERTICES
           for piece in white_pieces):
        return False
    for white_area, poker_area in zip(
            _piece_areas(white_pieces), _piece_areas(poker_pieces)):
        relative_error = abs(white_area - poker_area) / max(
            white_area, poker_area, 1.0)
        if relative_error > POKER_WHITE_GEOMETRY_AREA_TOLERANCE:
            return False
    return True


def _cross_2d(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _vertex_depth(previous, current, following):
    chord = following - previous
    chord_length = float(np.linalg.norm(chord))
    if chord_length < 1e-6:
        return 0.0, chord_length
    depth = abs(_cross_2d(chord, current - previous)) / chord_length
    return depth, chord_length


def _drop_collinear_vertices(polygon):
    points = [point.copy() for point in polygon]
    while len(points) > 3:
        array = np.asarray(points, dtype=np.float64)
        perimeter = float(cv2.arcLength(array.astype(np.float32), True))
        candidates = []
        for index, current in enumerate(array):
            previous = array[(index - 1) % len(array)]
            following = array[(index + 1) % len(array)]
            depth, chord_length = _vertex_depth(
                previous, current, following)
            if chord_length < 1e-6:
                candidates.append((depth, index))
                continue
            projection = float(np.dot(
                current - previous, following - previous,
            ) / (chord_length * chord_length))
            if (0.0 <= projection <= 1.0
                    and depth <= max(
                        0.75,
                        POKER_COLLINEAR_MAX_DEPTH_RATIO * perimeter,
                    )):
                candidates.append((depth, index))
        if not candidates:
            break
        _depth, index = min(candidates)
        points.pop(index)
    return np.asarray(points, dtype=np.float64)


def _straighten_concave_vertices(polygon):
    """Project every inward run onto the chord between its hull vertices."""
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) <= 3:
        return points.copy()
    hull_indices = cv2.convexHull(
        points.astype(np.float32), returnPoints=False,
    ).reshape(-1)
    if len(hull_indices) == len(points):
        return points.copy()

    # Rotate to a hull vertex so the final hull-to-first-hull run does not
    # require special cyclic output handling.
    first = int(np.min(hull_indices))
    points = np.roll(points, -first, axis=0)
    hull_indices = np.sort(cv2.convexHull(
        points.astype(np.float32), returnPoints=False,
    ).reshape(-1))

    straightened = []
    count = len(points)
    for position, start_index in enumerate(hull_indices):
        end_index = (int(hull_indices[position + 1])
                     if position + 1 < len(hull_indices) else count)
        start = points[int(start_index)]
        end = points[end_index % count]
        straightened.append(start.copy())
        chord = end - start
        length_squared = float(np.dot(chord, chord))
        if length_squared < 1e-9:
            continue
        run = [points[index % count]
               for index in range(int(start_index) + 1, end_index)]
        perimeter = float(cv2.arcLength(
            points.astype(np.float32), True))
        depth_limit = max(
            0.75, POKER_COLLINEAR_MAX_DEPTH_RATIO * perimeter)
        if any(_vertex_depth(start, value, end)[0] > depth_limit
               for value in run):
            continue
        projected = []
        for value in run:
            ratio = float(np.dot(value - start, chord) / length_squared)
            ratio = min(max(ratio, 0.0), 1.0)
            if 1e-5 < ratio < 1.0 - 1e-5:
                projected.append((ratio, start + ratio * chord))
        for _ratio, value in sorted(projected):
            if np.linalg.norm(value - straightened[-1]) > 0.25:
                straightened.append(value)
    return np.asarray(straightened, dtype=np.float64)


def _repair_shallow_concave_notches(polygon):
    """Replace segmentation-made shallow V notches with their mouth chord."""
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    repaired = False
    while len(points) > POKER_NOTCH_REPAIR_TARGET_VERTICES:
        signed_area = float(cv2.contourArea(
            points.astype(np.float32), oriented=True))
        perimeter = float(cv2.arcLength(points.astype(np.float32), True))
        if abs(signed_area) < 1e-6 or perimeter < 1e-6:
            break
        candidates = []
        for index, current in enumerate(points):
            previous = points[(index - 1) % len(points)]
            following = points[(index + 1) % len(points)]
            turn = _cross_2d(current - previous, following - current)
            if turn * signed_area >= 0.0:
                continue
            depth, mouth = _vertex_depth(previous, current, following)
            if (depth <= POKER_NOTCH_MAX_DEPTH_RATIO * perimeter
                    and mouth <= POKER_NOTCH_MAX_MOUTH_RATIO * perimeter):
                candidates.append((depth / max(mouth, 1e-6), index))
        if not candidates:
            break
        _relative_depth, index = min(candidates)
        points = np.delete(points, index, axis=0)
        repaired = True
    if repaired:
        points = _drop_collinear_vertices(points)
    return points


def _approximate_poker_piece(contour):
    # Poker fragments are guaranteed to be convex.  Dark suits or portraits
    # touching a cut edge merge with the black A4 mask and create artificial
    # inward notches.  Preserve the established polygon fit, then replace
    # every inward notch by the straight chord between its outer endpoints.
    recovered, arc_report = poker_arc_geometry.recover_virtual_corners(
        contour)
    if (arc_report["corners"]
            and 3 <= len(recovered) <= POKER_MAX_PIECE_VERTICES):
        return legacy.ensure_clockwise(recovered).astype(np.float64)

    polygon = legacy.approximate_piece(contour)
    if polygon is not None:
        polygon = _straighten_concave_vertices(polygon)
        if 3 <= len(polygon) <= POKER_MAX_PIECE_VERTICES:
            return legacy.ensure_clockwise(polygon).astype(np.float64)

    hull = cv2.convexHull(contour)
    hull_perimeter = float(cv2.arcLength(hull, True))
    for ratio in legacy.POLY_EPSILON_RATIOS:
        approximation = cv2.approxPolyDP(
            hull, ratio * hull_perimeter, True)[:, 0, :]
        if 3 <= len(approximation) <= POKER_MAX_PIECE_VERTICES:
            return legacy.ensure_clockwise(approximation).astype(np.float64)
    return None


def _detect_poker_mask(rectified_rgb):
    green_paper = legacy.green_paper_mask(
        rectified_rgb, apply_morphology=False)
    if float(np.mean(green_paper != 0)) >= legacy.A4_GREEN_CACHE_MIN_FILL_RATIO:
        # On the green A4, every card region is foreground regardless of
        # whether a black suit touches its cut edge. Internal print therefore
        # becomes a hole instead of merging the piece with the background.
        mask = np.where(green_paper == 0, 255, 0).astype(np.uint8)
    else:
        # Retain deterministic replay for the historical rectified fixtures.
        # Raw black A4 frames are no longer accepted by detect_a4().
        gray = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2GRAY)
        paper_level = float(np.median(gray))
        card_threshold = max(
            POKER_CARD_GRAY_MIN, paper_level + POKER_CARD_PAPER_DELTA)
        mask = np.where(gray > card_threshold, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8), iterations=1)
    inset = max(legacy.REGION_MARGIN, 6)
    mask[:inset, :] = 0
    mask[-inset:, :] = 0
    mask[:, :inset] = 0
    mask[:, -inset:] = 0

    a4_area = mask.size
    contours = list(legacy.find_contours(mask))
    contours.sort(key=lambda contour: (
        cv2.boundingRect(contour)[1], cv2.boundingRect(contour)[0]))
    pieces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (a4_area * config.PIECE_MIN_AREA_RATIO <= area
                <= a4_area * config.PIECE_MAX_AREA_RATIO):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if (x <= inset or y <= inset
                or x + width >= mask.shape[1] - inset
                or y + height >= mask.shape[0] - inset):
            continue
        polygon = _approximate_poker_piece(contour)
        if polygon is not None:
            if 3 <= len(polygon) <= POKER_MAX_PIECE_VERTICES:
                pieces.append(polygon.astype(np.float64))
        if len(pieces) == 4:
            break
    pieces.sort(key=lambda piece: (
        cv2.boundingRect(np.round(piece).astype(np.int32))[1],
        cv2.boundingRect(np.round(piece).astype(np.int32))[0],
    ))
    return pieces, mask


def detect_poker_pieces(rectified_rgb):
    """Detect printed card stock with an exposure-adaptive outer contour."""
    poker_pieces, poker_mask = _detect_poker_mask(rectified_rgb)
    white_pieces, white_mask, _timings = legacy.detect_pieces(rectified_rgb)
    if _same_piece_geometry(white_pieces, poker_pieces):
        arc_reports = poker_arc_geometry.analyze_piece_arcs(
            poker_mask, poker_pieces)
        if any(report["corners"] for report in arc_reports):
            return poker_pieces, poker_mask
        return white_pieces, white_mask
    return poker_pieces, poker_mask


class Task3PokerSolver:
    name = "题2-扑克"

    def __init__(self, corner_evidence_detector=None,
                 require_disambiguation=False,
                 require_corner_evidence=False,
                 return_best_candidate_on_failure=False):
        self.corner_evidence_detector = corner_evidence_detector
        self.require_disambiguation = bool(require_disambiguation)
        self.require_corner_evidence = bool(require_corner_evidence)
        self.return_best_candidate_on_failure = bool(
            return_best_candidate_on_failure)

    def solve(self, rectified_rgb):
        pieces, mask = detect_poker_pieces(rectified_rgb)
        return self.solve_detected(rectified_rgb, pieces, mask)

    def solve_detected(self, rectified_rgb, pieces, mask=None):
        """Solve geometry, then compare only the equal pair assignments."""
        mark_evidence = poker_layout_selector.collect_corner_mark_evidence(
            self.corner_evidence_detector, rectified_rgb, pieces)
        match_priority, topology_priority = _poker_texture_priorities(
            rectified_rgb)
        best_effort_options = {}
        if self.return_best_candidate_on_failure:
            best_effort_options["return_best_candidate_on_failure"] = True
        result = geometry.Task2WhiteSolver().solve_detected(
            rectified_rgb, pieces, mask,
            solve_options={
                "match_priority": match_priority,
                "topology_priority": topology_priority,
                "finalist_count": POKER_TEXTURE_FINALISTS,
                "finalist_max_fill_loss": POKER_TEXTURE_MAX_FILL_LOSS,
                "accept_fast_best": True,
                "defer_fast_accept": True,
            },
            use_default_search=True,
            candidate_validator=_poker_target_aspect_valid,
            **best_effort_options
        )
        pairs = poker_layout_selector.same_shape_pairs(pieces)
        pair = (poker_layout_selector.closest_same_shape_pair(pieces)
                if len(pieces) == 4 else None)
        needs_selection = len(pieces) == 4 and pair is not None
        if not needs_selection:
            if self.require_disambiguation:
                base_actions, base_diagnostics = result
                diagnostics = dict(base_diagnostics)
                diagnostics.update({
                    "disambiguation_skipped": True,
                    "disambiguation_reason": (
                        "AMBIGUOUS: expected four pieces and one equal "
                        "rectangle/square pair"),
                    "same_shape_pairs": pairs,
                    "same_shape_pair": pair,
                })
                return base_actions, diagnostics
            return result

        pair_mark_evidence = tuple(
            mark for mark in mark_evidence
            if int(mark["piece_index"]) in pair)
        if self.require_corner_evidence:
            if self.corner_evidence_detector is None:
                raise poker_layout_selector.PokerLayoutAmbiguousError(
                    "poker corner detector unavailable")
            if len(pair_mark_evidence) != 2:
                raise poker_layout_selector.PokerLayoutAmbiguousError(
                    "expected exactly two complete poker corner marks on "
                    "the equal-shape pair",
                    {"corner_mark_count": len(pair_mark_evidence)})

        _base_actions, base_diagnostics = result
        arc_reports = poker_arc_geometry.analyze_piece_arcs(mask, pieces)
        try:
            selected, selection_diagnostics = (
                poker_layout_selector.select_poker_layout(
                    pieces, base_diagnostics, arc_reports,
                    mark_evidence, rectified_rgb=rectified_rgb,
                    pair=pair, force_best=True))
        except poker_layout_selector.PokerLayoutAmbiguousError as error:
            diagnostics = dict(base_diagnostics)
            diagnostics.update(getattr(error, "diagnostics", {}))
            diagnostics.update({
                "disambiguation_skipped": True,
                "disambiguation_reason": str(error),
                "arc_reports": arc_reports,
                "corner_mark_evidence": mark_evidence,
                "pair_corner_mark_evidence": pair_mark_evidence,
                "corner_marks": mark_evidence,
            })
            return _base_actions, diagnostics
        try:
            from core.piece_action import actions_from_transforms
        except ImportError:
            from ..core.piece_action import actions_from_transforms
        diagnostics = dict(base_diagnostics)
        diagnostics.update(selection_diagnostics)
        diagnostics.update({
            "transforms": selected["transforms"],
            "assignment": selected["assignment"],
            "arc_reports": arc_reports,
            "corner_mark_evidence": mark_evidence,
            "pair_corner_mark_evidence": pair_mark_evidence,
            "corner_marks": mark_evidence,
        })
        actions = actions_from_transforms(
            pieces, selected["transforms"],
            mm_per_pixel=config.A4_MM_PER_PIXEL,
            confidence=float(base_diagnostics["fill_ratio"]),
        )
        return actions, diagnostics
