"""Bounded original/swapped layout selection for poker fragments.

The shared Task 2 solver remains the only topology solver.  This module takes
its validated transforms, finds the unique equal rectangle/square pair and
compares exactly two assignments. Only marks associated with that pair affect
selection; marks on the other pieces are ignored. An optional detector adds
corner-mark center/direction evidence without introducing a YOLO dependency
here.
"""

import itertools
import math

import cv2
import numpy as np

try:
    from . import poker_arc_geometry as arc_geometry
    from . import task2_white as geometry
except ImportError:
    import poker_arc_geometry as arc_geometry
    import task2_white as geometry


RECTANGLE_MAX_ANGLE_ERROR_DEGREES = 11.0
RECTANGLE_MAX_OPPOSITE_SIDE_ERROR = 0.09
RECTANGLE_MIN_BOX_FILL = 0.91
SQUARE_MIN_SIDE_RATIO = 0.88
EQUAL_PAIR_MAX_AREA_ERROR = 0.09
EQUAL_PAIR_MAX_PERIMETER_ERROR = 0.07
EQUAL_PAIR_MAX_SIDE_ERROR = 0.08
MARK_CORNER_MAX_DISTANCE_RATIO = 0.28
MARK_MIN_CONFIDENCE = 0.20
MARK_MIN_INWARD_DOT = 0.15
MIN_SELECTION_MARGIN = 0.025
GENERIC_PAIR_MIN_BOX_FILL = 0.42
GENERIC_PAIR_SQUARE_MIN_RATIO = 0.82
TEXTURE_MARK_REJECTION_PENALTY = 0.25
TEXTURE_CORNER_WEIGHT = 1.00
CORNER_PATCH_SIZE = 32
CORNER_DARK_MIN_AREA = 3
CORNER_DARK_THRESHOLD = 42.0


class PokerLayoutAmbiguousError(RuntimeError):
    """A safe refusal that the controller must not turn into UART actions."""

    def __init__(self, message, diagnostics=None):
        super().__init__("AMBIGUOUS: %s" % message)
        self.diagnostics = dict(diagnostics or {})
        self.diagnostics["ambiguous"] = True


def _relative_error(first, second):
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), 1e-9)


def _polygon_measurements(piece):
    polygon = np.asarray(piece, dtype=np.float64).reshape(-1, 2)
    if len(polygon) != 4:
        return None
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return None
    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 1.0 or perimeter <= 4.0:
        return None
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    if float(np.min(lengths)) <= 1.0:
        return None
    unit = edges / lengths[:, None]
    corner_cosines = np.abs(np.sum(unit * np.roll(unit, 1, axis=0), axis=1))
    angle_errors = np.degrees(np.arcsin(np.clip(
        corner_cosines, 0.0, 1.0)))
    if float(np.max(angle_errors)) > RECTANGLE_MAX_ANGLE_ERROR_DEGREES:
        return None
    opposite_error = max(
        _relative_error(lengths[0], lengths[2]),
        _relative_error(lengths[1], lengths[3]),
    )
    if opposite_error > RECTANGLE_MAX_OPPOSITE_SIDE_ERROR:
        return None
    width, height = cv2.minAreaRect(contour)[1]
    short_side, long_side = sorted((float(width), float(height)))
    if short_side <= 1.0 or area / (short_side * long_side) < (
            RECTANGLE_MIN_BOX_FILL):
        return None
    kind = ("square" if short_side / long_side >= SQUARE_MIN_SIDE_RATIO
            else "rectangle")
    return {
        "area": area,
        "perimeter": perimeter,
        "side_lengths": lengths,
        "short_side": short_side,
        "long_side": long_side,
        "kind": kind,
    }


def rectangle_kind(piece):
    """Return ``square``/``rectangle`` for an eligible physical pair piece."""
    measurement = _polygon_measurements(piece)
    return None if measurement is None else measurement["kind"]


def _cyclic_side_error(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    best = float("inf")
    for reverse in (False, True):
        values = second[::-1] if reverse else second
        for shift in range(len(values)):
            shifted = np.roll(values, shift)
            error = float(np.max(np.abs(first - shifted)
                                  / np.maximum(first, shifted)))
            best = min(best, error)
    return best


def same_shape_pairs(pieces):
    """Return all geometry-qualified equal rectangle/square pairs."""
    measurements = [_polygon_measurements(piece) for piece in pieces]
    values = []
    for first, second in itertools.combinations(range(len(pieces)), 2):
        left = measurements[first]
        right = measurements[second]
        if left is None or right is None or left["kind"] != right["kind"]:
            continue
        area_error = _relative_error(left["area"], right["area"])
        perimeter_error = _relative_error(
            left["perimeter"], right["perimeter"])
        side_error = _cyclic_side_error(
            left["side_lengths"], right["side_lengths"])
        if (area_error <= EQUAL_PAIR_MAX_AREA_ERROR
                and perimeter_error <= EQUAL_PAIR_MAX_PERIMETER_ERROR
                and side_error <= EQUAL_PAIR_MAX_SIDE_ERROR):
            score = (
                area_error / EQUAL_PAIR_MAX_AREA_ERROR
                + perimeter_error / EQUAL_PAIR_MAX_PERIMETER_ERROR
                + side_error / EQUAL_PAIR_MAX_SIDE_ERROR)
            values.append((score, first, second))
    values.sort(key=lambda value: (
        value[0],
        tuple(np.asarray(pieces[value[1]]).mean(axis=0)),
        tuple(np.asarray(pieces[value[2]]).mean(axis=0)),
    ))
    return tuple((first, second) for _score, first, second in values)


def unique_same_shape_pair(pieces):
    pairs = same_shape_pairs(pieces)
    return pairs[0] if len(pairs) == 1 else None


def _broad_shape_kind(piece):
    polygon = np.asarray(piece, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 3:
        return None
    width, height = cv2.minAreaRect(polygon)[1]
    short_side, long_side = sorted((float(width), float(height)))
    box_area = short_side * long_side
    if short_side <= 1.0 or box_area <= 1.0:
        return None
    fill = abs(float(cv2.contourArea(polygon))) / box_area
    if fill < GENERIC_PAIR_MIN_BOX_FILL:
        return None
    return ("square" if short_side / long_side
            >= GENERIC_PAIR_SQUARE_MIN_RATIO else "rectangle")


def closest_same_shape_pair(pieces):
    """Return the contest-prior pair even when strict rectangle gates miss."""
    strict = same_shape_pairs(pieces)
    if strict:
        return strict[0]
    values = []
    for first, second in itertools.combinations(range(len(pieces)), 2):
        left = np.asarray(pieces[first], dtype=np.float32).reshape(-1, 2)
        right = np.asarray(pieces[second], dtype=np.float32).reshape(-1, 2)
        if len(left) != len(right):
            continue
        left_kind = _broad_shape_kind(left)
        right_kind = _broad_shape_kind(right)
        if left_kind is None or left_kind != right_kind:
            continue
        left_area = abs(float(cv2.contourArea(left)))
        right_area = abs(float(cv2.contourArea(right)))
        if min(left_area, right_area) <= 1.0:
            continue
        area_error = _relative_error(left_area, right_area)
        perimeter_error = _relative_error(
            cv2.arcLength(left, True), cv2.arcLength(right, True))
        shape_error = float(cv2.matchShapes(
            left, right, cv2.CONTOURS_MATCH_I1, 0.0))
        score = 5.0 * area_error + perimeter_error + shape_error
        values.append((score, first, second))
    if not values:
        return None
    values.sort(key=lambda value: (
        value[0],
        tuple(np.asarray(pieces[value[1]]).mean(axis=0)),
        tuple(np.asarray(pieces[value[2]]).mean(axis=0)),
    ))
    return values[0][1], values[0][2]


def _fit_rigid(source, target):
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if len(source) != len(target):
        return None
    source_center = source.mean(axis=0)
    source_local = source - source_center
    best = None
    for reverse in (False, True):
        order = np.arange(len(target))
        if reverse:
            order = order[::-1]
        for shift in range(len(target)):
            ordered = target[np.roll(order, shift)]
            target_center = ordered.mean(axis=0)
            covariance = source_local.T.dot(ordered - target_center)
            try:
                left, _singular, right = np.linalg.svd(covariance)
            except np.linalg.LinAlgError:
                continue
            rotation = right.T.dot(left.T)
            if np.linalg.det(rotation) < 0.0:
                right[-1, :] *= -1.0
                rotation = right.T.dot(left.T)
            translation = target_center - rotation.dot(source_center)
            mapped = source.dot(rotation.T) + translation
            error = float(np.sqrt(np.mean(np.sum(
                (mapped - ordered) ** 2, axis=1))))
            transform = np.eye(3, dtype=np.float64)
            transform[:2, :2] = rotation
            transform[:2, 2] = translation
            rotation_angle = abs(math.atan2(rotation[1, 0], rotation[0, 0]))
            rank = (round(error, 9), round(rotation_angle, 9),
                    tuple(np.round(transform.reshape(-1), 9)))
            if best is None or rank < best[0]:
                best = (rank, transform, error)
    return None if best is None else (best[1], best[2])


def _swap_transforms(pieces, transforms, pair):
    first, second = pair
    first_target = geometry.apply_h(pieces[first], transforms[first])
    second_target = geometry.apply_h(pieces[second], transforms[second])
    first_fit = _fit_rigid(pieces[first], second_target)
    second_fit = _fit_rigid(pieces[second], first_target)
    if first_fit is None or second_fit is None:
        return None
    values = [np.asarray(transform, dtype=np.float64).copy()
              for transform in transforms]
    values[first] = first_fit[0]
    values[second] = second_fit[0]
    scale = max(math.sqrt(abs(float(cv2.contourArea(
        np.asarray(pieces[first], dtype=np.float32))))), 1.0)
    fit_error_ratio = max(first_fit[1], second_fit[1]) / scale
    return values, float(fit_error_ratio)


def build_layout_candidates(pieces, transforms, pair):
    """Build deterministic original and swapped assignments only."""
    pieces = [np.asarray(piece, dtype=np.float64) for piece in pieces]
    transforms = [np.asarray(value, dtype=np.float64) for value in transforms]
    if len(pieces) != len(transforms):
        raise ValueError("piece/transform count mismatch")
    original = {
        "assignment": "original",
        "transforms": tuple(value.copy() for value in transforms),
        "fit_error_ratio": 0.0,
        "slot_piece_indices": tuple(range(len(pieces))),
    }
    swapped = _swap_transforms(pieces, transforms, pair)
    if swapped is None:
        return (original,)
    slot_piece_indices = list(range(len(pieces)))
    slot_piece_indices[pair[0]], slot_piece_indices[pair[1]] = (
        slot_piece_indices[pair[1]], slot_piece_indices[pair[0]])
    return (original, {
        "assignment": "swapped",
        "transforms": tuple(swapped[0]),
        "fit_error_ratio": swapped[1],
        "slot_piece_indices": tuple(slot_piece_indices),
    })


def close_matched_seams(pieces, transforms, matches):
    """Remove target safety-gap translations while preserving rigid poses."""
    count = len(pieces)
    if count == 0 or len(transforms) != count or not matches:
        return tuple(np.asarray(value, dtype=np.float64).copy()
                     for value in transforms)
    equations = []
    targets = []
    for match in matches:
        _error, first, _first_edge, second, _second_edge = match[:5]
        first_start, first_end, second_start, second_end = (
            geometry.match_segments(pieces, match))
        first_target = geometry.apply_h(
            np.asarray((first_start, first_end), dtype=np.float64),
            transforms[first])
        second_target = geometry.apply_h(
            np.asarray((second_end, second_start), dtype=np.float64),
            transforms[second])
        row = np.zeros(count, dtype=np.float64)
        row[first] = -1.0
        row[second] = 1.0
        equations.append(row)
        targets.append(np.mean(first_target - second_target, axis=0))
    anchor = np.zeros(count, dtype=np.float64)
    anchor[0] = 1.0
    equations.append(anchor)
    targets.append(np.zeros(2, dtype=np.float64))
    offsets, _residuals, _rank, _singular = np.linalg.lstsq(
        np.asarray(equations), np.asarray(targets), rcond=None)
    closed = []
    for transform, offset in zip(transforms, offsets):
        translated = np.asarray(transform, dtype=np.float64).copy()
        translated[:2, 2] += offset
        closed.append(translated)
    return tuple(closed)


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


def _source_seam_samples(image, piece, transform, target_start, target_end,
                         samples=24):
    inverse = np.linalg.inv(np.asarray(transform, dtype=np.float64))
    source_start, source_end = geometry.apply_h(
        np.asarray((target_start, target_end), dtype=np.float64), inverse)
    normal = _inside_normal(source_start, source_end, piece)
    ratios = np.linspace(0.08, 0.92, samples)[:, None]
    points = source_start + (source_end - source_start) * ratios
    near = _sample_rgb(image, points + normal * 2.0)
    deep = _sample_rgb(image, points + normal * 5.0)
    return near, deep


def layout_texture_cost(rectified_rgb, pieces, base_transforms, matches,
                        candidate):
    """Score all internal seams after assigning source pieces to slots."""
    if not matches:
        return float("inf")
    slot_piece_indices = candidate["slot_piece_indices"]
    candidate_transforms = candidate["transforms"]
    costs = []
    for match in matches:
        _error, first_slot, _first_edge, second_slot, _second_edge = match[:5]
        first_start, first_end, second_start, second_end = (
            geometry.match_segments(pieces, match))
        first_target = geometry.apply_h(
            np.asarray((first_start, first_end)),
            base_transforms[first_slot])
        second_target = geometry.apply_h(
            np.asarray((second_end, second_start)),
            base_transforms[second_slot])
        first_piece = slot_piece_indices[first_slot]
        second_piece = slot_piece_indices[second_slot]
        first_near, first_deep = _source_seam_samples(
            rectified_rgb, pieces[first_piece],
            candidate_transforms[first_piece], *first_target)
        second_near, second_deep = _source_seam_samples(
            rectified_rgb, pieces[second_piece],
            candidate_transforms[second_piece], *second_target)
        colour = float(np.mean(np.abs(first_near - second_near)) / 255.0)
        gradient = float(np.mean(np.abs(
            (first_deep - first_near) - (second_deep - second_near)))
            / 255.0)
        costs.append(0.72 * colour + 0.28 * gradient)
    return float(np.mean(costs))


def _render_candidate(image, pieces, transforms, long_side=160):
    polygons = [geometry.apply_h(piece, transform)
                for piece, transform in zip(pieces, transforms)]
    all_points = np.vstack(polygons)
    rectangle = cv2.boxPoints(cv2.minAreaRect(
        all_points.astype(np.float32))).astype(np.float64)
    edges = np.roll(rectangle, -1, axis=0) - rectangle
    long_edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    long_angle = math.atan2(long_edge[1], long_edge[0])
    canonical = geometry.rigid(math.pi * 0.5 - long_angle)
    transforms = [canonical @ transform for transform in transforms]
    polygons = [geometry.apply_h(piece, transform)
                for piece, transform in zip(pieces, transforms)]
    all_points = np.vstack(polygons)
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    size = maximum - minimum
    if np.min(size) <= 1.0:
        return None, None
    scale = float(long_side) / max(float(np.max(size)), 1.0)
    width = max(8, int(math.ceil(size[0] * scale)) + 2)
    height = max(8, int(math.ceil(size[1] * scale)) + 2)
    normalize = np.asarray((
        (scale, 0.0, 1.0 - minimum[0] * scale),
        (0.0, scale, 1.0 - minimum[1] * scale),
        (0.0, 0.0, 1.0),
    ), dtype=np.float64)
    canvas = np.zeros((height, width, 3), np.uint8)
    mask = np.zeros((height, width), np.uint8)
    for piece, transform in zip(pieces, transforms):
        matrix = normalize @ transform
        polygon = geometry.apply_h(piece, matrix)
        piece_mask = np.zeros_like(mask)
        cv2.fillPoly(piece_mask, [np.round(polygon).astype(np.int32)], 1)
        warped = cv2.warpPerspective(
            image, matrix.astype(np.float32), (width, height),
            flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        canvas[piece_mask != 0] = warped[piece_mask != 0]
        mask[piece_mask != 0] = 1
    return canvas, mask


def _corner_feature(gray, valid):
    values = gray[valid != 0]
    if values.size < CORNER_DARK_MIN_AREA:
        return None, 0.0
    background = float(np.percentile(values, 75))
    threshold = min(background - CORNER_DARK_THRESHOLD, 125.0)
    dark = ((gray <= threshold) & (valid != 0)).astype(np.uint8)
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count = int(np.count_nonzero(dark))
    if count < CORNER_DARK_MIN_AREA:
        return None, 0.0
    moments = cv2.moments(dark, binaryImage=True)
    centroid = np.asarray((
        moments["m10"] / max(moments["m00"], 1e-6),
        moments["m01"] / max(moments["m00"], 1e-6),
    ), dtype=np.float64)
    darkness = max(
        0.0, (background - float(gray[dark != 0].mean())) / 255.0)
    confidence = min(
        1.0, 0.35 + 0.55 * darkness + min(0.10, count / 400.0))
    return (gray.astype(np.float32), dark, centroid), confidence


def _normalize_gray(gray, valid):
    values = gray[valid != 0]
    if values.size == 0:
        return np.zeros_like(gray, dtype=np.float32)
    center = float(np.median(values))
    spread = max(float(np.std(values)), 12.0)
    return np.clip((gray.astype(np.float32) - center) / spread, -3.0, 3.0)


def _shift_patch(values, dx, dy):
    shifted = np.zeros_like(values)
    source_x = slice(max(0, -dx), min(values.shape[1], values.shape[1] - dx))
    target_x = slice(max(0, dx), min(values.shape[1], values.shape[1] + dx))
    source_y = slice(max(0, -dy), min(values.shape[0], values.shape[0] - dy))
    target_y = slice(max(0, dy), min(values.shape[0], values.shape[0] + dy))
    shifted[target_y, target_x] = values[source_y, source_x]
    return shifted


def _one_diagonal_corner_score(rendered, mask):
    height, width = mask.shape
    roi_width = max(4, int(round(width * 0.24)))
    roi_height = max(4, int(round(height * 0.30)))
    gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY).astype(np.float32)
    first_gray = cv2.resize(
        gray[:roi_height, :roi_width],
        (CORNER_PATCH_SIZE, CORNER_PATCH_SIZE))
    first_valid = cv2.resize(
        mask[:roi_height, :roi_width],
        (CORNER_PATCH_SIZE, CORNER_PATCH_SIZE),
        interpolation=cv2.INTER_NEAREST)
    second_gray = cv2.resize(
        gray[-roi_height:, -roi_width:],
        (CORNER_PATCH_SIZE, CORNER_PATCH_SIZE))
    second_valid = cv2.resize(
        mask[-roi_height:, -roi_width:],
        (CORNER_PATCH_SIZE, CORNER_PATCH_SIZE),
        interpolation=cv2.INTER_NEAREST)
    first, first_confidence = _corner_feature(first_gray, first_valid)
    second, second_confidence = _corner_feature(second_gray, second_valid)
    if first is None or second is None:
        return 0.0
    first_gray, first_dark, _first_centroid = first
    second_gray, second_dark, _second_centroid = second
    first_gray = _normalize_gray(first_gray, first_valid)
    second_gray = cv2.rotate(
        _normalize_gray(second_gray, second_valid), cv2.ROTATE_180)
    second_dark = cv2.rotate(second_dark, cv2.ROTATE_180)
    best_difference = 1.0
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            shifted_gray = _shift_patch(second_gray, dx, dy)
            shifted_dark = _shift_patch(second_dark, dx, dy)
            valid = (first_dark != 0) | (shifted_dark != 0)
            if np.any(valid):
                best_difference = min(best_difference, float(np.mean(
                    np.abs(first_gray[valid] - shifted_gray[valid])) / 6.0))
    confidence = min(first_confidence, second_confidence)
    return max(0.0, confidence * (1.0 - best_difference))


def _corner_opposition_score(rendered, mask):
    return max(
        _one_diagonal_corner_score(rendered, mask),
        _one_diagonal_corner_score(
            cv2.flip(rendered, 1), cv2.flip(mask, 1)),
    )


def _evidence_value(item, name, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def collect_corner_mark_evidence(detector, rectified_rgb, pieces):
    """Normalize detector output and associate each mark with one piece.

    The provider contract is deliberately small: ``detect(image)`` (or a
    callable) returns dictionaries/objects containing ``confidence`` and
    either ``center`` or ``bbox_xyxy``.  ``direction`` is an optional vector
    pointing from the mark toward the card interior.  ``piece_index`` may be
    supplied; otherwise containment provides the association.
    """
    if detector is None:
        return ()
    raw = (detector.detect(rectified_rgb)
           if hasattr(detector, "detect") else detector(rectified_rgb))
    evidence = []
    polygons = [np.asarray(piece, dtype=np.float64).reshape(-1, 2)
                for piece in pieces]
    for item in raw or ():
        center = _evidence_value(item, "center")
        if center is None:
            box = _evidence_value(item, "bbox_xyxy")
            if box is None or len(box) != 4:
                continue
            center = ((float(box[0]) + float(box[2])) * 0.5,
                      (float(box[1]) + float(box[3])) * 0.5)
        center = np.asarray(center, dtype=np.float64).reshape(2)
        confidence = float(_evidence_value(item, "confidence", 1.0))
        if confidence < MARK_MIN_CONFIDENCE:
            continue
        piece_index = _evidence_value(item, "piece_index")
        containing = [index for index, polygon in enumerate(polygons)
                      if cv2.pointPolygonTest(
                          polygon.astype(np.float32), tuple(center),
                          False) >= 0]
        if piece_index is None:
            if len(containing) != 1:
                continue
            piece_index = containing[0]
        piece_index = int(piece_index)
        if piece_index not in containing:
            continue
        direction = _evidence_value(item, "direction")
        if direction is not None:
            direction = np.asarray(direction, dtype=np.float64).reshape(2)
            if float(np.linalg.norm(direction)) <= 1e-7:
                direction = None
        normalized_box = _evidence_value(item, "bbox_xyxy")
        if normalized_box is not None and len(normalized_box) == 4:
            normalized_box = tuple(float(value) for value in normalized_box)
        evidence.append({
            "piece_index": piece_index,
            "center": center,
            "confidence": min(1.0, max(0.0, confidence)),
            "direction": direction,
            "bbox_xyxy": normalized_box,
        })
    evidence.sort(key=lambda value: (
        value["piece_index"], value["center"][1], value["center"][0],
        -value["confidence"],
    ))
    return tuple(evidence)


def _candidate_geometry(pieces, candidate, mark_evidence):
    transforms = candidate["transforms"]
    assembled = [geometry.apply_h(piece, transform)
                 for piece, transform in zip(pieces, transforms)]
    all_points = np.vstack(assembled).astype(np.float32)
    rectangle = cv2.minAreaRect(all_points)
    rectangle_corners = cv2.boxPoints(rectangle).astype(np.float64)
    short_side = min(float(rectangle[1][0]), float(rectangle[1][1]))
    card_center = np.asarray(rectangle[0], dtype=np.float64)
    result = dict(candidate)
    result.update({
        "valid": False,
        "rejection": None,
        "arc_cost": float("inf"),
        "mark_cost": 0.0,
        "score": float("inf"),
        "rectangle_corners": rectangle_corners,
    })
    mark_costs = []
    mark_corner_indices = []
    for mark in mark_evidence:
        piece_index = int(mark["piece_index"])
        mapped_center = geometry.apply_h(
            np.asarray((mark["center"],), dtype=np.float64),
            transforms[piece_index])[0]
        if cv2.pointPolygonTest(
                np.asarray(assembled[piece_index], dtype=np.float32),
                tuple(mapped_center), False) < 0:
            result["rejection"] = "mark_outside_piece"
            return result
        distances_to_card_corners = np.linalg.norm(
            rectangle_corners - mapped_center, axis=1)
        target_corner_index = int(np.argmin(distances_to_card_corners))
        target_distance = float(
            distances_to_card_corners[target_corner_index])
        if target_distance > max(
                12.0, MARK_CORNER_MAX_DISTANCE_RATIO * short_side):
            result["rejection"] = "mark_outside_corner_zone"
            return result
        mark_corner_indices.append(target_corner_index)
        direction_penalty = 0.0
        if mark["direction"] is not None:
            mapped_direction = np.asarray(
                transforms[piece_index], dtype=np.float64
            )[:2, :2].dot(mark["direction"])
            expected = card_center - rectangle_corners[target_corner_index]
            mapped_direction /= max(float(np.linalg.norm(
                mapped_direction)), 1e-9)
            expected /= max(float(np.linalg.norm(expected)), 1e-9)
            inward_dot = float(np.dot(mapped_direction, expected))
            if inward_dot < MARK_MIN_INWARD_DOT:
                result["rejection"] = "mark_direction_outward"
                return result
            direction_penalty = (1.0 - inward_dot) * 0.25
        mark_costs.append(
            target_distance / max(short_side, 1.0)
            + direction_penalty)

    if len(mark_corner_indices) >= 2:
        strongest = sorted(
            zip(mark_evidence, mark_corner_indices),
            key=lambda value: -float(value[0]["confidence"]),
        )[:2]
        first_corner = strongest[0][1]
        second_corner = strongest[1][1]
        if (first_corner - second_corner) % 4 != 2:
            result["rejection"] = "pair_marks_not_on_opposite_corners"
            return result

    mark_cost = float(np.mean(mark_costs)) if mark_costs else 0.0
    result.update({
        "valid": True,
        "arc_cost": 0.0,
        "mark_cost": mark_cost,
        "score": (mark_cost
                  + 0.20 * float(candidate["fit_error_ratio"])),
        "mark_corner_indices": tuple(mark_corner_indices),
    })
    return result


def select_poker_layout(pieces, base_diagnostics, arc_reports,
                        mark_evidence=(), rectified_rgb=None, pair=None,
                        force_best=False):
    """Select original/swapped using texture, with corner marks as evidence."""
    pieces = [np.asarray(piece, dtype=np.float64) for piece in pieces]
    pairs = same_shape_pairs(pieces)
    pair = (pairs[0] if len(pairs) == 1 else None) if pair is None else pair
    pair_mark_evidence = tuple(
        mark for mark in mark_evidence
        if pair is not None and int(mark["piece_index"]) in pair)
    common = {
        "same_shape_pairs": pairs,
        "same_shape_pair": pair,
        "physical_arc_count": len(arc_geometry.corner_points(arc_reports)),
        "corner_mark_count": len(pair_mark_evidence),
        "ignored_corner_mark_count": (
            len(mark_evidence) - len(pair_mark_evidence)),
    }
    if pair is None:
        raise PokerLayoutAmbiguousError(
            "could not choose the required same-shape pair", common)
    transforms = base_diagnostics.get("transforms", ())
    if len(transforms) != len(pieces):
        raise PokerLayoutAmbiguousError(
            "geometry result does not cover every piece", common)
    candidates = build_layout_candidates(pieces, transforms, pair)
    if len(candidates) != 2:
        raise PokerLayoutAmbiguousError(
            "could not construct both original and swapped layouts", common)
    evaluated = tuple(_candidate_geometry(
        pieces, candidate, pair_mark_evidence)
        for candidate in candidates)
    matches = tuple(base_diagnostics.get("matches", ()))
    if rectified_rgb is not None and matches:
        texture_base_transforms = close_matched_seams(
            pieces, transforms, matches)
        texture_candidates = {
            candidate["assignment"]: candidate
            for candidate in build_layout_candidates(
                pieces, texture_base_transforms, pair)
        }
        textured = []
        for candidate in evaluated:
            candidate = dict(candidate)
            texture_candidate = texture_candidates[candidate["assignment"]]
            texture_cost = layout_texture_cost(
                rectified_rgb, pieces, texture_base_transforms, matches,
                texture_candidate)
            rendered, rendered_mask = _render_candidate(
                rectified_rgb, pieces, texture_candidate["transforms"])
            corner_score = (0.0 if rendered is None else
                            _corner_opposition_score(
                                rendered, rendered_mask))
            mark_penalty = (float(candidate["mark_cost"])
                            if candidate["valid"] else
                            TEXTURE_MARK_REJECTION_PENALTY)
            candidate["texture_cost"] = texture_cost
            candidate["corner_score"] = corner_score
            candidate["score"] = (
                texture_cost - TEXTURE_CORNER_WEIGHT * corner_score
                + 0.10 * mark_penalty
                + 0.05 * float(candidate["fit_error_ratio"]))
            textured.append(candidate)
        evaluated = tuple(textured)
    valid = [candidate for candidate in evaluated if candidate["valid"]]
    diagnostics = dict(common)
    if rectified_rgb is not None and matches:
        diagnostics["texture_base_transforms"] = texture_base_transforms
    diagnostics["candidate_scores"] = tuple({
        "assignment": candidate["assignment"],
        "valid": bool(candidate["valid"]),
        "score": float(candidate["score"]),
        "arc_cost": float(candidate["arc_cost"]),
        "mark_cost": float(candidate["mark_cost"]),
        "rejection": candidate["rejection"],
        "texture_cost": float(candidate.get("texture_cost", float("inf"))),
        "corner_score": float(candidate.get("corner_score", 0.0)),
    } for candidate in evaluated)
    if rectified_rgb is not None and matches and force_best:
        valid = [candidate for candidate in evaluated
                 if math.isfinite(candidate["score"])]
    if not valid:
        raise PokerLayoutAmbiguousError(
            "neither assignment places the equal-pair marks on card edges",
            diagnostics)
    valid.sort(key=lambda candidate: (
        candidate["score"], candidate["assignment"]))
    selected = valid[0]
    runner_up_margin = (float("inf") if len(valid) == 1 else
                        valid[1]["score"] - selected["score"])
    diagnostics["runner_up_margin"] = float(runner_up_margin)
    if (len(valid) > 1 and not force_best
            and runner_up_margin < MIN_SELECTION_MARGIN):
        raise PokerLayoutAmbiguousError(
            "original/swapped evidence margin is insufficient", diagnostics)
    diagnostics.update({
        "ambiguous": False,
        "selected_assignment": selected["assignment"],
        "selected_score": float(selected["score"]),
    })
    return selected, diagnostics
