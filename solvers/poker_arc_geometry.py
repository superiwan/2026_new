"""Physical poker-corner detection and virtual sharp-corner recovery.

The implementation is intentionally independent of card texture and paper
colour.  It operates on the dense outer contour produced after the green-A4
foreground segmentation.  A rounded card corner is accepted only when a run
of one to three short chords joins two long, near-perpendicular straight
edges and the dense contour between them has measurable circular curvature.
"""

import itertools
import math

import cv2
import numpy as np


ARC_APPROX_EPSILON_RATIO = 0.0025
ARC_LONG_EDGE_MIN_RATIO = 0.055
ARC_LONG_EDGE_MIN_PX = 8.0
ARC_MAX_SHORT_CHORDS = 3
ARC_MIN_CORNER_ANGLE_DEGREES = 64.0
ARC_MIN_RADIUS_PX = 2.5
ARC_MAX_RADIUS_PERIMETER_RATIO = 0.08
ARC_MAX_TANGENT_BALANCE = 2.8
ARC_MIN_DEPTH_PX = 0.65
ARC_MIN_PATH_CHORD_RATIO = 1.015
ARC_MAX_CIRCLE_RMS_PX = 1.8
ARC_MIN_CONFIDENCE = 0.48


def _cross(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _line_intersection(first_start, first_end, second_start, second_end):
    first_vector = first_end - first_start
    second_vector = second_end - second_start
    denominator = _cross(first_vector, second_vector)
    if abs(denominator) < 1e-7:
        return None
    distance = _cross(second_start - first_start, second_vector) / denominator
    return first_start + distance * first_vector


def _forward_indices(start, end, count):
    values = [int(start)]
    while values[-1] != int(end):
        values.append((values[-1] + 1) % count)
        if len(values) > count:
            return ()
    return tuple(values)


def _dense_path(contour, start, end):
    dense = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    start_index = int(np.argmin(np.linalg.norm(dense - start, axis=1)))
    end_index = int(np.argmin(np.linalg.norm(dense - end, axis=1)))
    indices = _forward_indices(start_index, end_index, len(dense))
    return dense[np.asarray(indices, dtype=np.int32)]


def _circle_fit(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 5:
        return None
    matrix = np.column_stack((2.0 * points, np.ones(len(points))))
    values = np.sum(points * points, axis=1)
    try:
        center_x, center_y, constant = np.linalg.lstsq(
            matrix, values, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_squared = constant + center_x * center_x + center_y * center_y
    if radius_squared <= 0.0:
        return None
    center = np.array((center_x, center_y), dtype=np.float64)
    radius = math.sqrt(float(radius_squared))
    residual = np.linalg.norm(points - center, axis=1) - radius
    return center, radius, float(np.sqrt(np.mean(residual * residual)))


def _path_depth(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length < 1e-7:
        return 0.0, chord_length, 0.0
    offsets = points - points[0]
    distances = np.abs(
        chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
    ) / chord_length
    path_length = float(np.sum(np.linalg.norm(
        np.diff(points, axis=0), axis=1)))
    return float(np.max(distances)), chord_length, path_length


def _candidate_rejection(reason, previous_edge, next_edge, short_count):
    return {
        "reason": str(reason),
        "previous_edge": int(previous_edge),
        "next_edge": int(next_edge),
        "short_chord_count": int(short_count),
    }


def detect_rounded_corners(contour, piece_index=None):
    """Return accepted rounded corners and rejected geometric candidates."""
    dense = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(dense) < 8:
        return {
            "piece_index": piece_index,
            "corners": (),
            "rejected": ({"reason": "contour_too_short"},),
            "arc_polygon": dense.copy(),
        }
    contour32 = dense.astype(np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour32, True))
    approximation = cv2.approxPolyDP(
        contour32, ARC_APPROX_EPSILON_RATIO * perimeter, True,
    )[:, 0, :].astype(np.float64)
    if len(approximation) < 4:
        return {
            "piece_index": piece_index,
            "corners": (),
            "rejected": ({"reason": "approximation_too_short"},),
            "arc_polygon": approximation,
        }

    segments = np.roll(approximation, -1, axis=0) - approximation
    lengths = np.linalg.norm(segments, axis=1)
    long_limit = max(ARC_LONG_EDGE_MIN_PX,
                     ARC_LONG_EDGE_MIN_RATIO * perimeter)
    long_edges = [index for index, length in enumerate(lengths)
                  if length >= long_limit]
    corners = []
    rejected = []
    if len(long_edges) < 2:
        rejected.append({"reason": "fewer_than_two_long_edges"})
    for previous_edge in long_edges:
        later = [value for value in long_edges
                 if (value - previous_edge) % len(approximation) > 0]
        if not later:
            continue
        next_edge = min(
            later, key=lambda value: (value - previous_edge)
            % len(approximation))
        short_count = ((next_edge - previous_edge)
                       % len(approximation)) - 1
        if not 1 <= short_count <= ARC_MAX_SHORT_CHORDS:
            continue

        tangent_start_index = (previous_edge + 1) % len(approximation)
        tangent_end_index = next_edge
        tangent_start = approximation[tangent_start_index]
        tangent_end = approximation[tangent_end_index]
        intersection = _line_intersection(
            approximation[previous_edge], tangent_start,
            tangent_end, approximation[(next_edge + 1)
                                        % len(approximation)],
        )
        if intersection is None:
            rejected.append(_candidate_rejection(
                "parallel_tangent_edges", previous_edge, next_edge,
                short_count))
            continue
        previous_direction = segments[previous_edge] / max(
            lengths[previous_edge], 1e-9)
        next_direction = segments[next_edge] / max(
            lengths[next_edge], 1e-9)
        angle = math.degrees(math.acos(np.clip(
            abs(float(np.dot(previous_direction, next_direction))),
            -1.0, 1.0)))
        if not (ARC_MIN_CORNER_ANGLE_DEGREES <= angle
                <= 90.0):
            rejected.append(_candidate_rejection(
                "tangent_angle", previous_edge, next_edge, short_count))
            continue

        first_radius = float(np.linalg.norm(intersection - tangent_start))
        second_radius = float(np.linalg.norm(intersection - tangent_end))
        minimum_radius = min(first_radius, second_radius)
        maximum_radius = max(first_radius, second_radius)
        radius_limit = ARC_MAX_RADIUS_PERIMETER_RATIO * perimeter
        if (minimum_radius < ARC_MIN_RADIUS_PX
                or maximum_radius > radius_limit
                or maximum_radius / max(minimum_radius, 1e-9)
                > ARC_MAX_TANGENT_BALANCE):
            rejected.append(_candidate_rejection(
                "tangent_radius", previous_edge, next_edge, short_count))
            continue

        path = _dense_path(dense, tangent_start, tangent_end)
        depth, chord_length, path_length = _path_depth(path)
        if (depth < ARC_MIN_DEPTH_PX
                or chord_length < 1e-7
                or path_length / chord_length < ARC_MIN_PATH_CHORD_RATIO):
            rejected.append(_candidate_rejection(
                "insufficient_curvature", previous_edge, next_edge,
                short_count))
            continue
        circle = _circle_fit(path)
        if circle is None:
            rejected.append(_candidate_rejection(
                "circle_fit", previous_edge, next_edge, short_count))
            continue
        circle_center, circle_radius, circle_rms = circle
        if (not ARC_MIN_RADIUS_PX <= circle_radius <= radius_limit
                or circle_rms > ARC_MAX_CIRCLE_RMS_PX):
            rejected.append(_candidate_rejection(
                "circle_residual", previous_edge, next_edge, short_count))
            continue

        outside_distance = float(cv2.pointPolygonTest(
            contour32, tuple(intersection), True))
        if outside_distance > 1.5:
            rejected.append(_candidate_rejection(
                "virtual_corner_inside", previous_edge, next_edge,
                short_count))
            continue

        angle_score = max(0.0, 1.0 - abs(90.0 - angle) / 26.0)
        balance_score = minimum_radius / max(maximum_radius, 1e-9)
        curve_score = min(1.0, depth / max(circle_radius * 0.18, 1.0))
        fit_score = max(0.0, 1.0 - circle_rms
                        / ARC_MAX_CIRCLE_RMS_PX)
        confidence = float(
            0.30 * angle_score + 0.20 * balance_score
            + 0.25 * curve_score + 0.25 * fit_score)
        if confidence < ARC_MIN_CONFIDENCE:
            rejected.append(_candidate_rejection(
                "low_confidence", previous_edge, next_edge, short_count))
            continue
        corners.append({
            "piece_index": piece_index,
            "tangent_start": tangent_start.copy(),
            "tangent_end": tangent_end.copy(),
            "arc_points": path.copy(),
            "virtual_corner": intersection.copy(),
            "confidence": confidence,
            "radius_px": float(circle_radius),
            "circle_center": circle_center.copy(),
            "circle_rms_px": float(circle_rms),
            "short_chord_count": int(short_count),
            "start_vertex": int(tangent_start_index),
            "end_vertex": int(tangent_end_index),
        })
    return {
        "piece_index": piece_index,
        "corners": tuple(corners),
        "rejected": tuple(rejected),
        "arc_polygon": approximation,
    }


def _clockwise_points(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1],
                        points[:, 0] - center[0])
    return points[np.argsort(angles)]


def recover_virtual_corners(contour, piece_index=None):
    """Replace accepted physical arcs with intersections of tangent edges."""
    report = detect_rounded_corners(contour, piece_index=piece_index)
    approximation = np.asarray(
        report["arc_polygon"], dtype=np.float64).reshape(-1, 2)
    if not report["corners"]:
        report["solver_polygon"] = approximation.copy()
        return approximation, report

    removed = set()
    for corner in report["corners"]:
        removed.update(_forward_indices(
            corner["start_vertex"], corner["end_vertex"],
            len(approximation)))
    kept = [point.copy() for index, point in enumerate(approximation)
            if index not in removed]
    kept.extend(corner["virtual_corner"].copy()
                for corner in report["corners"])
    recovered = _clockwise_points(kept)
    recovered32 = recovered.astype(np.float32).reshape(-1, 1, 2)
    epsilon = max(0.25, 0.0015 * cv2.arcLength(recovered32, True))
    recovered = cv2.approxPolyDP(
        recovered32, epsilon, True)[:, 0, :].astype(np.float64)
    report["solver_polygon"] = recovered.copy()
    return recovered, report


def _contour_center(contour):
    moments = cv2.moments(np.asarray(contour, dtype=np.float32))
    if abs(moments["m00"]) < 1e-7:
        return np.asarray(contour, dtype=np.float64).reshape(-1, 2).mean(
            axis=0)
    return np.array((moments["m10"] / moments["m00"],
                     moments["m01"] / moments["m00"]), dtype=np.float64)


def analyze_piece_arcs(binary_mask, pieces):
    """Associate physical mask contours to solver pieces and analyze arcs."""
    if binary_mask is None:
        return tuple({
            "piece_index": index,
            "corners": (),
            "rejected": ({"reason": "missing_binary_mask"},),
            "solver_polygon": np.asarray(piece, dtype=np.float64),
        } for index, piece in enumerate(pieces))
    mask = np.asarray(binary_mask, dtype=np.uint8)
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )[-2]
    contours = [contour for contour in contours
                if abs(cv2.contourArea(contour)) > 16.0]
    contour_centers = [_contour_center(contour) for contour in contours]
    unused = set(range(len(contours)))
    reports = []
    for piece_index, piece in enumerate(pieces):
        polygon = np.asarray(piece, dtype=np.float64).reshape(-1, 2)
        center = _contour_center(polygon)
        candidates = []
        for contour_index in unused:
            contour = contours[contour_index]
            contains = cv2.pointPolygonTest(
                contour.astype(np.float32), tuple(center), False) >= 0
            distance = float(np.linalg.norm(
                contour_centers[contour_index] - center))
            candidates.append((not contains, distance, contour_index))
        if not candidates:
            reports.append({
                "piece_index": piece_index,
                "corners": (),
                "rejected": ({"reason": "physical_contour_not_found"},),
                "solver_polygon": polygon.copy(),
            })
            continue
        _outside, _distance, contour_index = min(candidates)
        unused.remove(contour_index)
        _recovered, report = recover_virtual_corners(
            contours[contour_index], piece_index=piece_index)
        report["physical_contour"] = np.asarray(
            contours[contour_index], dtype=np.float64).reshape(-1, 2)
        reports.append(report)
    # Real printed card contours can split one physical arc into many short
    # pieces after thresholding.  When strict circle fitting leaves fewer than
    # four corners, use the card-stock prior used by the reference solver:
    # rank virtual vertices by their distance from the measured contour and
    # fill only the missing slots.  Sharp cut vertices stay below the 2 px
    # minimum and are not promoted.
    accepted = corner_points(reports, confidence_threshold=0.0)
    if len(accepted) < 4:
        ranked = []
        for piece_index, (piece, report) in enumerate(zip(pieces, reports)):
            measured = report.get("physical_contour")
            if measured is None:
                continue
            contour = np.asarray(measured, dtype=np.float32).reshape(-1, 1, 2)
            for vertex in np.asarray(piece, dtype=np.float64).reshape(-1, 2):
                distance = abs(cv2.pointPolygonTest(
                    contour, tuple(map(float, vertex)), True))
                if distance < 2.0:
                    continue
                if any(int(item["piece_index"]) == piece_index
                       and np.linalg.norm(
                           np.asarray(item["virtual_corner"])
                           - vertex) < 1.5
                       for item in accepted):
                    continue
                ranked.append((float(distance), piece_index, vertex.copy()))
        ranked.sort(key=lambda item: (-item[0], item[1],
                                      tuple(np.round(item[2], 6))))
        for distance, piece_index, vertex in ranked[:4 - len(accepted)]:
            reports[piece_index].setdefault("corners", ())
            reports[piece_index]["corners"] = tuple(
                reports[piece_index]["corners"]) + ({
                    "piece_index": piece_index,
                    "virtual_corner": vertex,
                    "confidence": min(0.85, 0.50 + 0.05 * distance),
                    "radius_px": float(distance),
                    "circle_rms_px": None,
                    "fallback_distance_prior": True,
                },)
            accepted = corner_points(reports, confidence_threshold=0.0)
    return tuple(reports)


def corner_points(reports, confidence_threshold=ARC_MIN_CONFIDENCE):
    """Flatten accepted virtual corners for diagnostics and selection."""
    values = []
    for report in reports:
        for corner in report.get("corners", ()):
            if float(corner.get("confidence", 0.0)) >= confidence_threshold:
                values.append(corner)
    return tuple(values)


def best_corner_assignment(points, rectangle_corners):
    """Return the minimum-distance one-to-one assignment for four corners."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    rectangle_corners = np.asarray(
        rectangle_corners, dtype=np.float64).reshape(-1, 2)
    if len(points) != 4 or len(rectangle_corners) != 4:
        return None
    best = None
    for order in itertools.permutations(range(4)):
        distances = np.linalg.norm(
            points - rectangle_corners[np.asarray(order)], axis=1)
        value = (float(np.sum(distances)), tuple(order), distances)
        if best is None or value[0] < best[0]:
            best = value
    return best
