"""MaixCAM Pro lightweight A4 puzzle vision program.

Device mode uses MaixPy for capture/display and OpenCV for geometry.  Run
``python main.py --self-test`` on a PC to exercise the complete vision pipeline
with a synthetic frame; this mode never opens a camera.
"""

import itertools
import math
import sys

import cv2
import numpy as np

try:
    from maix import app, camera, display, image, time as maix_time, touchscreen
except ImportError:  # Allows the camera-free host self-test.
    app = camera = display = image = maix_time = touchscreen = None


# Camera / rectified A4 geometry.  420 x 594 gives exactly 2 pixels/mm.
CAM_W, CAM_H = 640, 480
WARP_W, WARP_H = 420, 594
MM_PER_PIXEL = 0.5
SKIP_FRAMES = 30

# Tune these first for the actual black paper, white pieces, lens and lighting.
A4_MIN_AREA_RATIO = 0.18
A4_RATIO_TOLERANCE = 0.35
WHITE_THRESHOLD = 165
MORPH_KERNEL = 3
PIECE_MIN_AREA_RATIO = 0.001
PIECE_MAX_AREA_RATIO = 0.25
POLY_EPSILON_RATIOS = (0.012, 0.018, 0.025, 0.035, 0.05, 0.07)
EDGE_LENGTH_TOLERANCE = 0.15
MAX_EDGE_MATCH_CANDIDATES = 40
MAX_COMPLETE_CANDIDATES = 6000
MAX_SOLVE_MS = 6000
SEARCH_PROGRESS_INTERVAL_MS = 250
TOPOLOGY_BATCH_SIZE = 64
POSE_GRAPH_ITERATIONS = 20
MAX_SPLIT_VARIANTS = 10
MAX_THREE_WAY_SPLIT_VARIANTS = 18
MAX_REFINED_TOPOLOGIES = 40
MAX_INVERSE_FIT_CANDIDATES = 60
QUICK_GEOMETRY_MARGIN = 0.03
MIN_FILL_RATE = 0.85
MAX_FINAL_OVERLAP_RATIO = 0.05
INVERSE_MIN_FILL_RATE = 0.78
INVERSE_MAX_OVERLAP_RATIO = 0.08
MIN_LONG_MM, MAX_LONG_MM = 90.0, 120.0
MIN_SHORT_MM, MAX_SHORT_MM = 50.0, 90.0
RECT_APPROX_EPSILON_RATIO = 0.02
RECT_ANGLE_TOLERANCE_DEG = 20.0
EARLY_ACCEPT_FILL_RATE = 0.86
EARLY_ACCEPT_OVERLAP_RATIO = 0.04
EARLY_ACCEPT_CORNER_ERROR_DEG = 15.0

COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_YELLOW = (255, 220, 0)
COLOR_RED = (255, 60, 60)
PIECE_COLORS = ((255, 90, 90), (90, 255, 120), (80, 170, 255), (255, 180, 60))


def ticks_ms():
    if maix_time is not None:
        return maix_time.ticks_ms()
    import time
    return int(time.perf_counter() * 1000)


def elapsed_ms(start):
    return ticks_ms() - start


def find_contours(binary):
    result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


def order_quad(points):
    """Return TL, TR, BR, BL and rotate landscape observations to portrait."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    top = np.linalg.norm(ordered[1] - ordered[0])
    left = np.linalg.norm(ordered[3] - ordered[0])
    if top > left:
        ordered = np.roll(ordered, 1, axis=0)
    return ordered


def detect_a4(rgb):
    """Locate the largest black convex A4-like quadrilateral."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    frame_area = rgb.shape[0] * rgb.shape[1]
    target_ratio = 297.0 / 210.0
    best = None
    best_score = -1.0
    for contour in find_contours(black):
        area = cv2.contourArea(contour)
        if area < frame_area * A4_MIN_AREA_RATIO:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_quad(approx[:, 0, :])
        width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) * 0.5
        height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) * 0.5
        if min(width, height) < 1.0:
            continue
        ratio_error = abs(max(width, height) / min(width, height) - target_ratio) / target_ratio
        if ratio_error > A4_RATIO_TOLERANCE:
            continue
        score = area / frame_area - ratio_error
        if score > best_score:
            best, best_score = quad, score

    if best is None:
        return None, None
    destination = np.float32(((0, 0), (WARP_W - 1, 0), (WARP_W - 1, WARP_H - 1), (0, WARP_H - 1)))
    return best, cv2.getPerspectiveTransform(best, destination)


def warp_a4(rgb, homography):
    return cv2.warpPerspective(rgb, homography, (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)


def cached_a4_is_valid(warped_rgb):
    """Cheap validation used only when SOLVE is pressed, never every frame."""
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    border = np.concatenate((gray[:8, :].ravel(), gray[-8:, :].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()))
    return float(np.mean(gray < WHITE_THRESHOLD)) > 0.55 and float(np.mean(border < WHITE_THRESHOLD)) > 0.65


def contour_polygon_quality(contour, polygon):
    """Return raster IoU and normalized area error against the source contour."""
    contour_points = contour.reshape(-1, 2)
    all_points = np.vstack((contour_points, polygon))
    minimum = np.floor(all_points.min(axis=0) - 2).astype(int)
    maximum = np.ceil(all_points.max(axis=0) + 2).astype(int)
    width, height = maximum - minimum + 1
    contour_mask = np.zeros((int(height), int(width)), np.uint8)
    polygon_mask = np.zeros_like(contour_mask)
    cv2.fillPoly(contour_mask, [np.round(contour_points - minimum).astype(np.int32)], 1)
    cv2.fillPoly(polygon_mask, [np.round(polygon - minimum).astype(np.int32)], 1)
    intersection = int(np.count_nonzero(contour_mask & polygon_mask))
    union = int(np.count_nonzero(contour_mask | polygon_mask))
    iou = intersection / max(1, union)
    contour_area = max(1.0, cv2.contourArea(contour.astype(np.float32)))
    area_error = abs(cv2.contourArea(polygon.astype(np.float32)) - contour_area) / contour_area
    return iou, area_error


def fit_contour_edge(contour_points, start_index, end_index):
    if end_index >= start_index:
        segment = contour_points[start_index:end_index + 1]
    else:
        segment = np.vstack((contour_points[start_index:], contour_points[:end_index + 1]))
    if len(segment) < 2:
        return None
    if len(segment) >= 10:
        trim = max(1, len(segment) // 12)
        segment = segment[trim:-trim]
    vx, vy, x0, y0 = cv2.fitLine(segment.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    return np.float32((x0, y0)), np.float32((vx, vy))


def intersect_lines(first, second):
    point1, direction1 = first
    point2, direction2 = second
    cross = float(direction1[0] * direction2[1] - direction1[1] * direction2[0])
    if abs(cross) < 0.08:
        return None
    delta = point2 - point1
    distance = float(delta[0] * direction2[1] - delta[1] * direction2[0]) / cross
    return point1 + direction1 * distance


def refine_polygon_vertices(contour, polygon):
    """Fit every source-contour edge and replace vertices by adjacent intersections."""
    contour_points = contour.reshape(-1, 2).astype(np.float32)
    indices = [int(np.argmin(np.sum((contour_points - vertex) ** 2, axis=1))) for vertex in polygon]
    lines = []
    for index in range(len(polygon)):
        line = fit_contour_edge(contour_points, indices[index], indices[(index + 1) % len(polygon)])
        if line is None:
            return polygon.copy()
        lines.append(line)

    refined = []
    for index, original in enumerate(polygon):
        intersection = intersect_lines(lines[index - 1], lines[index])
        previous_length = np.linalg.norm(polygon[index - 1] - original)
        next_length = np.linalg.norm(polygon[(index + 1) % len(polygon)] - original)
        max_shift = max(6.0, 0.18 * min(previous_length, next_length))
        if intersection is None or np.linalg.norm(intersection - original) > max_shift:
            intersection = original
        refined.append(intersection)
    return np.asarray(refined, dtype=np.float32)


def approximate_piece(contour):
    perimeter = cv2.arcLength(contour, True)
    best = None
    best_score = -float("inf")
    seen = set()
    for ratio in POLY_EPSILON_RATIOS:
        approximation = cv2.approxPolyDP(contour, ratio * perimeter, True)
        if not 3 <= len(approximation) <= 5:
            continue
        signature = tuple(int(value) for value in approximation.reshape(-1))
        if signature in seen:
            continue
        seen.add(signature)
        polygon = approximation[:, 0, :].astype(np.float32)
        polygon = refine_polygon_vertices(contour, polygon)
        iou, area_error = contour_polygon_quality(contour, polygon)
        score = iou - 0.35 * area_error
        if score > best_score:
            best, best_score = polygon, score
    if best is not None and cv2.contourArea(best, oriented=True) < 0:
        best = best[::-1].copy()
    return best


def detect_pieces(warped_rgb):
    timings = {}
    start = ticks_ms()
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    # Area filtering removes isolated bright noise; avoid opening, which rounds shard tips.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary[:5, :] = binary[-5:, :] = 0
    binary[:, :5] = binary[:, -5:] = 0
    timings["binary_morph"] = elapsed_ms(start)

    start = ticks_ms()
    contours = find_contours(binary)
    a4_area = WARP_W * WARP_H
    contours = [c for c in contours if a4_area * PIECE_MIN_AREA_RATIO <= cv2.contourArea(c) <= a4_area * PIECE_MAX_AREA_RATIO]
    contours.sort(key=cv2.contourArea, reverse=True)
    timings["contours"] = elapsed_ms(start)

    start = ticks_ms()
    pieces = []
    for contour in contours:
        polygon = approximate_piece(contour)
        if polygon is not None:
            pieces.append(polygon)
        if len(pieces) == 4:  # The task specifies no more than four pieces.
            break
    timings["approx_poly"] = elapsed_ms(start)
    return pieces, binary, timings


def edges(polygon):
    for index in range(len(polygon)):
        yield polygon[index], polygon[(index + 1) % len(polygon)]


def rigid_matrix(angle, tx=0.0, ty=0.0):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.float64(((cosine, -sine, tx), (sine, cosine, ty), (0.0, 0.0, 1.0)))


def apply_rigid(points, transform):
    points = np.asarray(points)
    return (points @ transform[:2, :2].T + transform[:2, 2]).astype(np.float32)


def align_directed_edge(source_a, source_b, target_a, target_b):
    source = source_b - source_a
    target = target_b - target_a
    angle = math.atan2(target[1], target[0]) - math.atan2(source[1], source[0])
    transform = rigid_matrix(angle)
    mapped_a = transform[:2, :2].dot(source_a)
    transform[:2, 2] = target_a - mapped_a
    return transform


def candidate_edge_matches(pieces):
    """Create the repository-style globally sorted cut-edge shortlist."""
    all_edges = {}
    for piece_index, polygon in enumerate(pieces):
        for edge_index, edge in enumerate(edges(polygon)):
            all_edges[(piece_index, edge_index)] = edge
    matches = []
    for (first_key, first_edge), (second_key, second_edge) in itertools.combinations(all_edges.items(), 2):
        first_piece, first_edge_index = first_key
        second_piece, second_edge_index = second_key
        if first_piece == second_piece:
            continue
        first_length = float(np.linalg.norm(first_edge[1] - first_edge[0]))
        second_length = float(np.linalg.norm(second_edge[1] - second_edge[0]))
        if max(first_length, second_length) < 1.0:
            continue
        relative_error = abs(first_length - second_length) / max(first_length, second_length)
        if relative_error <= EDGE_LENGTH_TOLERANCE:
            matches.append((relative_error, first_piece, first_edge_index, second_piece, second_edge_index))
    matches.sort(key=lambda item: item[0])
    return matches[:MAX_EDGE_MATCH_CANDIDATES]


def split_edge_piece_sets(pieces):
    """Split one long edge into two or three collinear matching segments."""
    variants = []
    three_way_variants = []
    seen = set()
    three_way_seen = set()
    for long_piece, polygon in enumerate(pieces):
        for long_edge, (start, end) in enumerate(edges(polygon)):
            long_length = float(np.linalg.norm(end - start))
            other_indices = [index for index in range(len(pieces)) if index != long_piece]
            for first_piece, second_piece in itertools.combinations(other_indices, 2):
                for _, first_endpoints in enumerate(edges(pieces[first_piece])):
                    first_length = float(np.linalg.norm(first_endpoints[1] - first_endpoints[0]))
                    for _, second_endpoints in enumerate(edges(pieces[second_piece])):
                        second_length = float(np.linalg.norm(second_endpoints[1] - second_endpoints[0]))
                        combined = first_length + second_length
                        if max(long_length, combined) < 1.0:
                            continue
                        relative_error = abs(long_length - combined) / max(long_length, combined)
                        if relative_error > EDGE_LENGTH_TOLERANCE:
                            continue
                        for ratio in (first_length / combined, second_length / combined):
                            if not 0.20 <= ratio <= 0.80:
                                continue
                            key = (long_piece, long_edge, int(round(ratio * 100)))
                            if key in seen:
                                continue
                            seen.add(key)
                            split_point = start + (end - start) * ratio
                            augmented = [piece.copy() for piece in pieces]
                            augmented[long_piece] = np.insert(polygon, long_edge + 1, split_point, axis=0).astype(np.float32)
                            variants.append((relative_error, augmented, key))

            if len(other_indices) != 3:
                continue
            edge_choices = [list(edges(pieces[index])) for index in other_indices]
            for chosen_edges in itertools.product(*edge_choices):
                lengths = [float(np.linalg.norm(edge[1] - edge[0])) for edge in chosen_edges]
                combined = sum(lengths)
                if max(long_length, combined) < 1.0:
                    continue
                relative_error = abs(long_length - combined) / max(long_length, combined)
                if relative_error > EDGE_LENGTH_TOLERANCE:
                    continue
                for order in itertools.permutations(range(3)):
                    ordered_lengths = [lengths[index] for index in order]
                    segment_ratios = [length / combined for length in ordered_lengths]
                    if min(segment_ratios) < 0.10:
                        continue
                    first_ratio = segment_ratios[0]
                    second_ratio = segment_ratios[0] + segment_ratios[1]
                    key = (
                        long_piece, long_edge,
                        int(round(first_ratio * 100)), int(round(second_ratio * 100)),
                    )
                    if key in three_way_seen:
                        continue
                    three_way_seen.add(key)
                    split_points = np.float32((
                        start + (end - start) * first_ratio,
                        start + (end - start) * second_ratio,
                    ))
                    augmented = [piece.copy() for piece in pieces]
                    augmented[long_piece] = np.concatenate((
                        polygon[:long_edge + 1], split_points, polygon[long_edge + 1:]
                    )).astype(np.float32)
                    three_way_variants.append((relative_error, augmented, key))
    variants.sort(key=lambda item: item[0])
    three_way_variants.sort(key=lambda item: item[0])
    selected = variants[:MAX_SPLIT_VARIANTS] + three_way_variants[:MAX_THREE_WAY_SPLIT_VARIANTS]
    return [(pieces, None)] + [(variant, key) for _, variant, key in selected]


def matching_topologies(pieces, should_stop=None):
    """Enumerate general connected trees and one-cycle edge-disjoint topologies."""
    piece_count = len(pieces)
    candidates = candidate_edge_matches(pieces)
    examined = 0
    for match_count in range(piece_count - 1, piece_count + 1):
        for topology in itertools.combinations(candidates, match_count):
            examined += 1
            if examined % 128 == 0 and should_stop is not None and should_stop():
                return
            used_edges = set()
            graph = [set() for _ in range(piece_count)]
            valid = True
            for _, first_piece, first_edge, second_piece, second_edge in topology:
                if (first_piece, first_edge) in used_edges or (second_piece, second_edge) in used_edges:
                    valid = False
                    break
                used_edges.add((first_piece, first_edge))
                used_edges.add((second_piece, second_edge))
                graph[first_piece].add(second_piece)
                graph[second_piece].add(first_piece)
            if not valid:
                continue
            reached = {0}
            stack = [0]
            while stack:
                for neighbor in graph[stack.pop()]:
                    if neighbor not in reached:
                        reached.add(neighbor)
                        stack.append(neighbor)
            if len(reached) == piece_count:
                yield topology


def propagate_topology(pieces, topology):
    piece_edges = [list(edges(polygon)) for polygon in pieces]
    adjacency = [[] for _ in pieces]
    for _, first_piece, first_edge, second_piece, second_edge in topology:
        adjacency[first_piece].append((second_piece, first_edge, second_edge))
        adjacency[second_piece].append((first_piece, second_edge, first_edge))

    transforms = [None] * len(pieces)
    transforms[0] = np.eye(3, dtype=np.float64)
    stack = [0]
    closure_error = 0.0
    while stack:
        source_index = stack.pop()
        for target_index, source_edge_index, target_edge_index in adjacency[source_index]:
            source_a, source_b = piece_edges[source_index][source_edge_index]
            target_a, target_b = piece_edges[target_index][target_edge_index]
            world_source = apply_rigid(np.asarray((source_a, source_b), dtype=np.float32), transforms[source_index])
            proposed = align_directed_edge(target_a, target_b, world_source[1], world_source[0])
            if transforms[target_index] is None:
                transforms[target_index] = proposed
                stack.append(target_index)
            else:
                proposed_polygon = apply_rigid(pieces[target_index], proposed)
                existing_polygon = apply_rigid(pieces[target_index], transforms[target_index])
                closure_error += float(np.linalg.norm(proposed_polygon - existing_polygon, axis=1).mean())
    if any(transform is None for transform in transforms):
        return None
    assembled = [apply_rigid(polygon, transform) for polygon, transform in zip(pieces, transforms)]
    return transforms, assembled, closure_error


def pose_graph_refine(pieces, topology, initial_transforms):
    """Numerically minimize all reversed matched-edge residuals, fixing piece zero."""
    if len(pieces) < 3:
        return initial_transforms

    def pack(transforms):
        values = []
        for transform in transforms[1:]:
            values.extend((math.atan2(transform[1, 0], transform[0, 0]), transform[0, 2], transform[1, 2]))
        return np.asarray(values, dtype=np.float64)

    def unpack(values):
        transforms = [initial_transforms[0]]
        for index in range(len(pieces) - 1):
            angle, tx, ty = values[index * 3:index * 3 + 3]
            transforms.append(rigid_matrix(angle, tx, ty))
        return transforms

    piece_edges = [list(edges(polygon)) for polygon in pieces]

    def residual(values):
        transforms = unpack(values)
        residuals = []
        for _, first_piece, first_edge, second_piece, second_edge in topology:
            first_a, first_b = piece_edges[first_piece][first_edge]
            second_a, second_b = piece_edges[second_piece][second_edge]
            world_first = apply_rigid(np.asarray((first_a, first_b), dtype=np.float32), transforms[first_piece])
            world_second = apply_rigid(np.asarray((second_b, second_a), dtype=np.float32), transforms[second_piece])
            residuals.extend((world_first - world_second).reshape(-1))
        return np.asarray(residuals, dtype=np.float64)

    values = pack(initial_transforms)
    try:
        for _ in range(POSE_GRAPH_ITERATIONS):
            base = residual(values)
            jacobian = np.empty((len(base), len(values)), dtype=np.float64)
            for index in range(len(values)):
                step = 1e-5 if index % 3 == 0 else 1e-3
                shifted = values.copy()
                shifted[index] += step
                jacobian[:, index] = (residual(shifted) - base) / step
            delta = np.linalg.lstsq(jacobian, -base, rcond=None)[0]
            if not np.all(np.isfinite(delta)):
                return initial_transforms
            values += delta
            if np.linalg.norm(delta) < 1e-7:
                break
    except (ValueError, np.linalg.LinAlgError):
        return initial_transforms
    return unpack(values)


def range_penalty(value, minimum, maximum):
    if value < minimum:
        return (minimum - value) / minimum
    if value > maximum:
        return (value - maximum) / maximum
    return 0.0


def rectangle_outline_metrics(union_mask):
    contours = find_contours((union_mask.astype(np.uint8) * 255))
    if not contours:
        return 0, float("inf"), False
    if len(contours) != 1:
        return sum(len(contour) for contour in contours), float("inf"), False
    outline = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(outline, True)
    polygon = cv2.approxPolyDP(outline, RECT_APPROX_EPSILON_RATIO * perimeter, True)
    if len(polygon) != 4 or not cv2.isContourConvex(polygon):
        return len(polygon), float("inf"), False
    points = polygon[:, 0, :].astype(np.float32)
    errors = []
    for index, point in enumerate(points):
        first = points[index - 1] - point
        second = points[(index + 1) % 4] - point
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator < 1.0:
            return 4, float("inf"), False
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        errors.append(abs(angle - 90.0))
    maximum_error = max(errors)
    return 4, maximum_error, maximum_error <= RECT_ANGLE_TOLERANCE_DEG


def solution_meets_constraints(candidate):
    long_mm, short_mm = candidate["size_mm"]
    return (
        candidate["fill_rate"] >= MIN_FILL_RATE
        and candidate["overlap_ratio"] <= MAX_FINAL_OVERLAP_RATIO
        and MIN_LONG_MM <= long_mm <= MAX_LONG_MM
        and MIN_SHORT_MM <= short_mm <= MAX_SHORT_MM
        and candidate["outline_is_rectangle"]
    )


def inverse_solution_meets_constraints(candidate):
    long_mm, short_mm = candidate["size_mm"]
    return (
        candidate["fill_rate"] >= INVERSE_MIN_FILL_RATE
        and candidate["overlap_ratio"] <= INVERSE_MAX_OVERLAP_RATIO
        and MIN_LONG_MM <= long_mm <= MAX_LONG_MM
        and MIN_SHORT_MM <= short_mm <= MAX_SHORT_MM
    )


def is_high_confidence_solution(candidate):
    return (
        solution_meets_constraints(candidate)
        and candidate["fill_rate"] >= EARLY_ACCEPT_FILL_RATE
        and candidate["overlap_ratio"] <= EARLY_ACCEPT_OVERLAP_RATIO
        and candidate["max_corner_error_deg"] <= EARLY_ACCEPT_CORNER_ERROR_DEG
    )


def constraint_failures(candidate):
    long_mm, short_mm = candidate["size_mm"]
    failures = []
    if candidate["fill_rate"] < MIN_FILL_RATE:
        failures.append("FILL")
    if candidate["overlap_ratio"] > MAX_FINAL_OVERLAP_RATIO:
        failures.append("OVERLAP")
    if not MIN_LONG_MM <= long_mm <= MAX_LONG_MM or not MIN_SHORT_MM <= short_mm <= MAX_SHORT_MM:
        failures.append("SIZE")
    if not candidate["outline_is_rectangle"]:
        failures.append("RECT")
    return failures


def constraint_violation(candidate):
    long_mm, short_mm = candidate["size_mm"]
    size_error = range_penalty(long_mm, MIN_LONG_MM, MAX_LONG_MM) + range_penalty(short_mm, MIN_SHORT_MM, MAX_SHORT_MM)
    rectangle_error = 0.0 if candidate["outline_is_rectangle"] else 0.5
    return (
        max(0.0, MIN_FILL_RATE - candidate["fill_rate"]) * 4.0
        + max(0.0, candidate["overlap_ratio"] - MAX_FINAL_OVERLAP_RATIO) * 5.0
        + size_error * 2.0
        + rectangle_error
    )


def evaluate_assembly(polygons):
    all_points = np.vstack(polygons)
    x0, y0 = np.floor(all_points.min(axis=0) - 3).astype(int)
    shifted = [polygon - np.float32((x0, y0)) for polygon in polygons]
    max_xy = np.ceil(np.vstack(shifted).max(axis=0) + 4).astype(int)
    width, height = max(1, int(max_xy[0])), max(1, int(max_xy[1]))
    accumulation = np.zeros((height, width), np.uint8)
    for polygon in shifted:
        one = np.zeros_like(accumulation)
        cv2.fillPoly(one, [np.round(polygon).astype(np.int32)], 1)
        accumulation += one
    union = accumulation > 0
    overlap_px = int(np.maximum(accumulation.astype(np.int16) - 1, 0).sum())

    rect = cv2.minAreaRect(np.vstack(shifted).astype(np.float32))
    box = cv2.boxPoints(rect)
    rect_mask = np.zeros_like(accumulation)
    cv2.fillConvexPoly(rect_mask, np.round(box).astype(np.int32), 1)
    rect_px = max(1, int(np.count_nonzero(rect_mask)))
    union_px = int(np.count_nonzero(union))
    gap_px = int(np.count_nonzero((rect_mask > 0) & ~union))
    overlap_ratio = overlap_px / max(1, sum(int(round(cv2.contourArea(p))) for p in shifted))
    gap_ratio = gap_px / rect_px
    fill_rate = min(1.0, union_px / rect_px)
    outline_vertices, max_corner_error, outline_is_rectangle = rectangle_outline_metrics(union)

    short_px, long_px = sorted(rect[1])
    short_mm, long_mm = short_px * MM_PER_PIXEL, long_px * MM_PER_PIXEL
    size_penalty = range_penalty(short_mm, 50.0, 90.0) + range_penalty(long_mm, 90.0, 120.0)
    score = 5.0 * overlap_ratio + 4.0 * gap_ratio + 2.0 * size_penalty
    return {
        "score": score,
        "polygons": [p.copy() for p in polygons],
        "overlap_mm2": overlap_px * MM_PER_PIXEL * MM_PER_PIXEL,
        "gap_mm2": gap_px * MM_PER_PIXEL * MM_PER_PIXEL,
        "overlap_ratio": overlap_ratio,
        "gap_ratio": gap_ratio,
        "fill_rate": fill_rate,
        "size_mm": (long_mm, short_mm),
        "outline_vertices": outline_vertices,
        "max_corner_error_deg": max_corner_error,
        "outline_is_rectangle": outline_is_rectangle,
    }


def quick_assembly_rejection(polygons):
    """Reject geometry that cannot pass hard constraints without raster masks."""
    points = np.vstack(polygons)
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    _, size, _ = cv2.minAreaRect(points)
    short_px, long_px = sorted(size)
    short_mm, long_mm = short_px * MM_PER_PIXEL, long_px * MM_PER_PIXEL
    size_margin_mm = MM_PER_PIXEL * 1.5
    size_possible = (
        MIN_LONG_MM - size_margin_mm <= long_mm <= MAX_LONG_MM + size_margin_mm
        and MIN_SHORT_MM - size_margin_mm <= short_mm <= MAX_SHORT_MM + size_margin_mm
    )
    size_error = range_penalty(long_mm, MIN_LONG_MM, MAX_LONG_MM) + range_penalty(short_mm, MIN_SHORT_MM, MAX_SHORT_MM)
    if not size_possible:
        return "SIZE", 2.0 * size_error + 1.0

    rectangle_area = max(1.0, short_px * long_px)
    piece_area = sum(abs(cv2.contourArea(polygon)) for polygon in polygons)
    gross_fill = piece_area / rectangle_area
    if gross_fill < MIN_FILL_RATE - QUICK_GEOMETRY_MARGIN:
        return "FILL", (MIN_FILL_RATE - gross_fill) * 4.0

    maximum_gross_fill = 1.0 / max(1e-6, 1.0 - MAX_FINAL_OVERLAP_RATIO)
    if gross_fill > maximum_gross_fill + QUICK_GEOMETRY_MARGIN:
        return "OVERLAP", (gross_fill - maximum_gross_fill) * 5.0
    return None


def solve_puzzle(pieces, progress_callback=None):
    """Search general/full and virtual-split edge topologies, then hard-validate."""
    if not 2 <= len(pieces) <= 4:
        return None, 0, False, None

    normalized = [piece if cv2.contourArea(piece, oriented=True) >= 0 else piece[::-1].copy() for piece in pieces]
    ranked = []
    best_valid = None
    best_inverse = None
    best_invalid = None
    inverse_ranked = []
    topology_count = 0
    raster_evaluations = 0
    quick_rejections = 0
    truncated = False
    solve_start = ticks_ms()
    last_progress_ms = -SEARCH_PROGRESS_INTERVAL_MS

    def time_exhausted():
        return MAX_SOLVE_MS > 0 and elapsed_ms(solve_start) >= MAX_SOLVE_MS

    def report_progress(force=False):
        nonlocal last_progress_ms
        if progress_callback is None:
            return
        elapsed = elapsed_ms(solve_start)
        if not force and elapsed - last_progress_ms < SEARCH_PROGRESS_INTERVAL_MS:
            return
        candidate_progress = topology_count / max(1.0, float(MAX_COMPLETE_CANDIDATES))
        time_progress = elapsed / float(MAX_SOLVE_MS) if MAX_SOLVE_MS > 0 else 0.0
        progress_callback(topology_count, elapsed, min(1.0, max(candidate_progress, time_progress)))
        last_progress_ms = elapsed

    def search_should_stop():
        report_progress()
        return time_exhausted()

    def decorate(metrics, topology_score, topology, closure_error, split_key):
        metrics["topology_score"] = topology_score
        metrics["closure_error_px"] = closure_error
        metrics["matched_edges"] = tuple(topology)
        metrics["split_edge"] = split_key
        metrics["failures"] = constraint_failures(metrics)
        return metrics

    def consider(metrics, selection_score):
        nonlocal best_valid, best_invalid
        if solution_meets_constraints(metrics):
            if best_valid is None or selection_score < best_valid[0]:
                best_valid = (selection_score, metrics)
        else:
            violation = constraint_violation(metrics)
            if best_invalid is None or violation < best_invalid[0]:
                best_invalid = (violation, metrics)

    def consider_inverse(metrics, selection_score):
        nonlocal best_inverse
        if not inverse_solution_meets_constraints(metrics):
            return
        metrics["inverse_fit"] = True
        if best_inverse is None or selection_score < best_inverse[0]:
            best_inverse = (selection_score, metrics)

    def remember_inverse(priority, assembled, topology, closure_error, split_key, length_error):
        inverse_ranked.append((
            priority, [polygon.copy() for polygon in assembled],
            topology, closure_error, split_key, length_error,
        ))
        if len(inverse_ranked) > MAX_INVERSE_FIT_CANDIDATES * 2:
            inverse_ranked.sort(key=lambda item: item[0])
            del inverse_ranked[MAX_INVERSE_FIT_CANDIDATES:]

    def process_topology(variant_pieces, split_key, topology):
        nonlocal topology_count, raster_evaluations, quick_rejections, high_confidence_found
        propagated = propagate_topology(variant_pieces, topology)
        if propagated is None:
            return
        topology_count += 1
        report_progress()
        transforms, assembled, closure_error = propagated
        length_error = sum(match[0] for match in topology)
        quick_rejection = quick_assembly_rejection(assembled)
        if quick_rejection is not None:
            quick_rejections += 1
            remember_inverse(
                quick_rejection[1] + length_error, assembled,
                topology, closure_error, split_key, length_error,
            )
            if len(topology) > len(variant_pieces) - 1:
                topology_score = closure_error * 5.0 + quick_rejection[1] * 5000.0 + length_error * 5000.0
                ranked.append((topology_score, variant_pieces, split_key, topology, transforms, closure_error, length_error))
            return

        metrics = evaluate_assembly(assembled)
        raster_evaluations += 1
        overlap_pixels = metrics["overlap_mm2"] / (MM_PER_PIXEL * MM_PER_PIXEL)
        gap_pixels = metrics["gap_mm2"] / (MM_PER_PIXEL * MM_PER_PIXEL)
        topology_score = closure_error * 5.0 + overlap_pixels * 3.0 + gap_pixels + length_error * 5000.0
        metrics = decorate(metrics, topology_score, topology, closure_error, split_key)
        consider(metrics, metrics["score"] + length_error)
        if not solution_meets_constraints(metrics):
            # Inverse fitting deliberately ignores seam-length error here: imperfectly
            # cut pieces should be ranked by the final rectangle gap/overlap instead.
            consider_inverse(metrics, metrics["score"])
        ranked.append((topology_score, variant_pieces, split_key, topology, transforms, closure_error, length_error))
        if is_high_confidence_solution(metrics):
            high_confidence_found = True

    stop = False
    high_confidence_found = False
    report_progress(True)
    variant_states = [
        (variant_pieces, split_key, iter(matching_topologies(variant_pieces, search_should_stop)))
        for variant_pieces, split_key in split_edge_piece_sets(normalized)
    ]
    while variant_states and not stop and not high_confidence_found:
        next_states = []
        for variant_pieces, split_key, topology_iterator in variant_states:
            exhausted = False
            for _ in range(TOPOLOGY_BATCH_SIZE):
                if topology_count >= MAX_COMPLETE_CANDIDATES or time_exhausted():
                    truncated = stop = True
                    break
                try:
                    topology = next(topology_iterator)
                except StopIteration:
                    exhausted = True
                    break
                process_topology(variant_pieces, split_key, topology)
                if high_confidence_found:
                    break
            if not exhausted:
                next_states.append((variant_pieces, split_key, topology_iterator))
            if stop or high_confidence_found:
                break
        variant_states = next_states
        if time_exhausted():
            truncated = True
            break

    ranked.sort(key=lambda item: item[0])
    for topology_score, variant_pieces, split_key, topology, transforms, closure_error, length_error in ([] if high_confidence_found else ranked[:MAX_REFINED_TOPOLOGIES]):
        if time_exhausted():
            truncated = True
            break
        if len(topology) <= len(variant_pieces) - 1:
            continue  # Tree edges are already exactly aligned; there is no loop residual to distribute.
        refined = pose_graph_refine(variant_pieces, topology, transforms)
        assembled = [apply_rigid(polygon, transform) for polygon, transform in zip(variant_pieces, refined)]
        if quick_assembly_rejection(assembled) is not None:
            quick_rejections += 1
            continue
        metrics = decorate(evaluate_assembly(assembled), topology_score, topology, closure_error, split_key)
        raster_evaluations += 1
        consider(metrics, metrics["score"] + length_error)
        if not solution_meets_constraints(metrics):
            consider_inverse(metrics, metrics["score"])

    if best_valid is None:
        inverse_ranked.sort(key=lambda item: item[0])
        for _, assembled, topology, closure_error, split_key, length_error in inverse_ranked[:MAX_INVERSE_FIT_CANDIDATES]:
            metrics = decorate(evaluate_assembly(assembled), 0.0, topology, closure_error, split_key)
            raster_evaluations += 1
            consider_inverse(metrics, metrics["score"])

    solution = best_valid[1] if best_valid is not None else (None if best_inverse is None else best_inverse[1])
    invalid = None if best_invalid is None else best_invalid[1]
    stats = {
        "topologies_tested": topology_count,
        "quick_rejections": quick_rejections,
        "raster_evaluations": raster_evaluations,
        "inverse_candidates_evaluated": min(len(inverse_ranked), MAX_INVERSE_FIT_CANDIDATES),
        "solve_elapsed_ms": elapsed_ms(solve_start),
    }
    if solution is not None:
        solution.update(stats)
    if invalid is not None:
        invalid.update(stats)
    report_progress(True)
    return solution, topology_count, truncated, invalid


def analyze_once(rgb, homography, progress_callback=None):
    timings = {}
    total_start = ticks_ms()
    start = ticks_ms()
    warped = warp_a4(rgb, homography)
    timings["warp"] = elapsed_ms(start)
    if not cached_a4_is_valid(warped):
        timings["total"] = elapsed_ms(total_start)
        return None, "A4 cache invalid", timings

    pieces, binary, piece_timings = detect_pieces(warped)
    timings.update(piece_timings)
    if not 2 <= len(pieces) <= 4:
        timings["total"] = elapsed_ms(total_start)
        return {"warped": warped, "binary": binary, "pieces": pieces, "solution": None}, "need 2-4 pieces", timings

    start = ticks_ms()
    solver_progress = None
    if progress_callback is not None:
        def solver_progress(count, elapsed, progress):
            progress_callback(warped, pieces, count, elapsed, progress, timings)
    solution, count, truncated, best_invalid = solve_puzzle(pieces, solver_progress)
    timings["solve"] = elapsed_ms(start)
    timings["total"] = elapsed_ms(total_start)
    result = {"warped": warped, "binary": binary, "pieces": pieces, "solution": solution, "best_invalid": best_invalid, "candidates": count, "truncated": truncated}
    if solution is not None:
        status = "INVERSE FIT" if solution.get("inverse_fit") else "OK"
    elif truncated:
        status = "SEARCH LIMIT"
    else:
        status = "NO VALID SOLUTION"
    return result, status, timings


def draw_a4_border(rgb, quad):
    if quad is not None:
        cv2.polylines(rgb, [np.round(quad).astype(np.int32)], True, COLOR_GREEN, 3, cv2.LINE_AA)
        for index, point in enumerate(quad):
            p = tuple(np.round(point).astype(int))
            cv2.circle(rgb, p, 5, COLOR_YELLOW, -1, cv2.LINE_AA)
            cv2.putText(rgb, str(index), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1, cv2.LINE_AA)


def draw_buttons(rgb):
    cv2.rectangle(rgb, (0, 0), (CAM_W // 2 - 1, 36), (35, 80, 120), -1)
    cv2.rectangle(rgb, (CAM_W // 2, 0), (CAM_W - 1, 36), (50, 105, 45), -1)
    cv2.putText(rgb, "FIND A4", (105, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(rgb, "SOLVE ONCE", (405, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def fit_image(source, max_width, max_height):
    scale = min(max_width / source.shape[1], max_height / source.shape[0])
    size = (max(1, int(round(source.shape[1] * scale))), max(1, int(round(source.shape[0] * scale))))
    return cv2.resize(source, size, interpolation=cv2.INTER_AREA), scale


def draw_piece_overlay(warped, pieces):
    overlay = warped.copy()
    for index, polygon in enumerate(pieces):
        points = np.round(polygon).astype(np.int32)
        color = PIECE_COLORS[index % len(PIECE_COLORS)]
        cv2.polylines(overlay, [points], True, color, 3, cv2.LINE_AA)
        for vertex_index, point in enumerate(points):
            p = tuple(point)
            cv2.circle(overlay, p, 5, COLOR_CYAN, -1, cv2.LINE_AA)
            cv2.putText(overlay, str(vertex_index), (p[0] + 5, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1, cv2.LINE_AA)
    return overlay


def draw_virtual_assembly(canvas, solution, area, searching=False):
    x, y, width, height = area
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (50, 50, 50), 1)
    inverse_fit = solution is not None and solution.get("inverse_fit", False)
    title = "INVERSE FIT" if inverse_fit else "VIRTUAL RECT"
    title_color = COLOR_YELLOW if inverse_fit else (230, 230, 230)
    cv2.putText(canvas, title, (x + 5, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, title_color, 1, cv2.LINE_AA)
    if solution is None:
        text = "SEARCHING..." if searching else "NO VALID SOLUTION"
        color = COLOR_YELLOW if searching else COLOR_RED
        offset = 72 if searching else 28
        cv2.putText(canvas, text, (x + offset, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return
    polygons = solution["polygons"]
    all_points = np.vstack(polygons)
    minimum = all_points.min(axis=0)
    span = np.maximum(all_points.max(axis=0) - minimum, 1.0)
    scale = min((width - 24) / span[0], (height - 38) / span[1])
    offset = np.float32((x + (width - span[0] * scale) * 0.5, y + 25 + (height - 32 - span[1] * scale) * 0.5))
    for index, polygon in enumerate(polygons):
        points = np.round((polygon - minimum) * scale + offset).astype(np.int32)
        color = PIECE_COLORS[index % len(PIECE_COLORS)]
        cv2.fillPoly(canvas, [points], color)
        cv2.polylines(canvas, [points], True, (255, 255, 255), 1, cv2.LINE_AA)
    rectangle = cv2.minAreaRect(np.vstack([((p - minimum) * scale + offset) for p in polygons]).astype(np.float32))
    cv2.polylines(canvas, [np.round(cv2.boxPoints(rectangle)).astype(np.int32)], True, COLOR_GREEN, 2, cv2.LINE_AA)


def draw_search_progress(canvas, candidates, elapsed, progress):
    left, top, width, height = 18, 417, 290, 18
    progress = float(np.clip(progress, 0.0, 1.0))
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (65, 65, 65), -1)
    filled = int(round(width * progress))
    if filled > 0:
        cv2.rectangle(canvas, (left, top), (left + filled, top + height), COLOR_CYAN, -1)
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (220, 220, 220), 1)
    label = "%d%%  c%d  %.1f/%.1fs" % (
        int(round(progress * 100.0)), candidates, elapsed / 1000.0, MAX_SOLVE_MS / 1000.0,
    )
    cv2.putText(canvas, label, (left, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1, cv2.LINE_AA)


def build_result_view(raw_rgb, quad, analysis, status, timings, fps, search_progress=None):
    canvas = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    raw_copy = raw_rgb.copy()
    draw_a4_border(raw_copy, quad)
    raw_small = cv2.resize(raw_copy, (320, 240), interpolation=cv2.INTER_AREA)
    canvas[38:278, :320] = raw_small

    warped_overlay = draw_piece_overlay(analysis["warped"], analysis["pieces"]) if analysis is not None else np.zeros((WARP_H, WARP_W, 3), np.uint8)
    warp_small, _ = fit_image(warped_overlay, 300, 400)
    wx = 330 + (305 - warp_small.shape[1]) // 2
    canvas[42:42 + warp_small.shape[0], wx:wx + warp_small.shape[1]] = warp_small
    draw_virtual_assembly(
        canvas, None if analysis is None else analysis.get("solution"),
        (4, 323, 318, 151), searching=search_progress is not None,
    )
    if search_progress is not None:
        draw_search_progress(canvas, *search_progress)

    draw_buttons(canvas)
    cv2.putText(canvas, "RAW + A4", (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "RECTIFIED + POLYGONS", (350, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "FPS %.1f  %s" % (fps, status), (5, 291), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    if analysis is not None and analysis.get("solution") is not None:
        solution = analysis["solution"]
        metric_color = COLOR_YELLOW if solution.get("inverse_fit") else COLOR_GREEN
        text = "%.1fx%.1fmm fill %.3f" % (solution["size_mm"][0], solution["size_mm"][1], solution["fill_rate"])
        cv2.putText(canvas, text, (330, 454), cv2.FONT_HERSHEY_SIMPLEX, 0.41, metric_color, 1, cv2.LINE_AA)
        text = "gap %.0f ov %.0f ang %.1f c%d/r%d%s" % (
            solution["gap_mm2"], solution["overlap_mm2"], solution["max_corner_error_deg"],
            analysis["candidates"], solution.get("raster_evaluations", analysis["candidates"]),
            "+" if analysis["truncated"] else "",
        )
        cv2.putText(canvas, text, (330, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1, cv2.LINE_AA)
    elif analysis is not None and analysis.get("best_invalid") is not None:
        rejected = analysis["best_invalid"]
        long_mm, short_mm = rejected["size_mm"]
        cv2.putText(canvas, "FAIL " + "/".join(rejected["failures"]), (330, 454), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_RED, 1, cv2.LINE_AA)
        text = "f%.3f ov%.3f %.0fx%.0f c%d/r%d%s" % (
            rejected["fill_rate"], rejected["overlap_ratio"], long_mm, short_mm,
            analysis["candidates"], rejected.get("raster_evaluations", analysis["candidates"]),
            "+" if analysis["truncated"] else "",
        )
        cv2.putText(canvas, text, (330, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 230), 1, cv2.LINE_AA)
    timing_line1 = "a4:%d warp:%d bin:%d cont:%d" % (timings.get("find_a4", 0), timings.get("warp", 0), timings.get("binary_morph", 0), timings.get("contours", 0))
    timing_line2 = "poly:%d solve:%d total:%d ms" % (timings.get("approx_poly", 0), timings.get("solve", 0), timings.get("total", 0))
    cv2.putText(canvas, timing_line1, (5, 306), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, timing_line2, (5, 319), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


class TouchButton:
    def __init__(self):
        self.was_pressed = False
        self.last_x = 0
        self.last_y = 0

    def read(self, touch):
        x, y, pressed = touch.read()
        action = None
        if pressed:
            self.last_x, self.last_y = x, y
            self.was_pressed = True
        elif self.was_pressed:
            self.was_pressed = False
            if self.last_y <= 60:
                action = "find" if self.last_x < CAM_W // 2 else "solve"
        return action


def run_device():
    if camera is None:
        raise RuntimeError("MaixPy is unavailable. Use --self-test on PC or run main.py on MaixCAM Pro.")

    cam = camera.Camera(CAM_W, CAM_H, image.Format.FMT_RGB888, buff_num=2)
    screen = display.Display()
    touch = touchscreen.TouchScreen()
    cam.skip_frames(SKIP_FRAMES)  # Intentionally called exactly once.

    frame = cam.read()
    rgb = image.image2cv(frame, ensure_bgr=False, copy=False)
    start = ticks_ms()
    quad, homography = detect_a4(rgb)
    last_a4_ms = elapsed_ms(start)
    timings = {"find_a4": last_a4_ms}
    status = "A4 cached" if homography is not None else "tap FIND A4"
    cached_view = None
    buttons = TouchButton()
    fps_meter = maix_time.FPS(10)
    fps = 0.0

    while not app.need_exit():
        frame = cam.read()
        rgb = image.image2cv(frame, ensure_bgr=False, copy=False)
        action = buttons.read(touch)

        if action == "find":
            start = ticks_ms()
            quad, homography = detect_a4(rgb)
            last_a4_ms = elapsed_ms(start)
            timings = {"find_a4": last_a4_ms}
            status = "A4 cached" if homography is not None else "A4 not found"
            cached_view = None
        elif action == "solve":
            if homography is None:
                start = ticks_ms()
                quad, homography = detect_a4(rgb)
                last_a4_ms = elapsed_ms(start)
                timings = {"find_a4": last_a4_ms}
            if homography is None:
                status = "A4 not found"
                cached_view = None
            else:
                def show_search_progress(warped, pieces, count, search_ms, progress, partial_timings):
                    progress_timings = dict(partial_timings)
                    progress_timings["find_a4"] = last_a4_ms
                    progress_timings["solve"] = search_ms
                    progress_timings["total"] = search_ms + sum(
                        progress_timings.get(key, 0)
                        for key in ("warp", "binary_morph", "contours", "approx_poly")
                    )
                    progress_analysis = {
                        "warped": warped, "pieces": pieces, "solution": None,
                        "best_invalid": None, "candidates": count, "truncated": False,
                    }
                    progress_view = build_result_view(
                        rgb, quad, progress_analysis, "SEARCHING", progress_timings, fps,
                        search_progress=(count, search_ms, progress),
                    )
                    screen.show(image.cv2image(progress_view, bgr=False, copy=False))

                analysis, status, timings = analyze_once(rgb, homography, show_search_progress)
                timings["find_a4"] = last_a4_ms
                if status == "A4 cache invalid":
                    start = ticks_ms()
                    quad, homography = detect_a4(rgb)
                    refind_ms = elapsed_ms(start)
                    last_a4_ms = refind_ms
                    if homography is None:
                        status = "A4 invalid; refind failed"
                        cached_view = None
                    else:
                        analysis, status, timings = analyze_once(rgb, homography, show_search_progress)
                        timings["find_a4"] = last_a4_ms
                        cached_view = build_result_view(rgb, quad, analysis, status, timings, fps)
                else:
                    cached_view = build_result_view(rgb, quad, analysis, status, timings, fps)

        if cached_view is None:
            draw_a4_border(rgb, quad)
            draw_buttons(rgb)
            cv2.putText(rgb, "FPS %.1f  %s" % (fps, status), (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            shown = image.cv2image(rgb, bgr=False, copy=False)
        else:
            shown = image.cv2image(cached_view, bgr=False, copy=False)
        screen.show(shown)
        fps = fps_meter.fps()


def affine_polygon(polygon, angle_degrees, translation):
    angle = math.radians(angle_degrees)
    rotation = np.float32(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    return polygon.dot(rotation.T) + np.float32(translation)


def synthetic_frame():
    """Make a perspective A4 scene containing three scattered white shards."""
    paper = np.zeros((WARP_H, WARP_W, 3), np.uint8)
    target = (
        np.float32(((0, 0), (190, 0), (148, 45), (0, 30))),
        np.float32(((0, 30), (148, 45), (190, 120), (0, 120))),
        np.float32(((190, 0), (190, 120), (148, 45))),
    )
    scattered = (
        affine_polygon(target[0], 8, (35, 65)),
        affine_polygon(target[1], -12, (185, 235)),
        affine_polygon(target[2], 28, (170, -20)),
    )
    for polygon in scattered:
        cv2.fillPoly(paper, [np.round(polygon).astype(np.int32)], (245, 245, 245))

    raw = np.full((CAM_H, CAM_W, 3), 205, np.uint8)
    quad = np.float32(((145, 25), (475, 48), (545, 455), (80, 430)))
    destination = np.float32(((0, 0), (WARP_W - 1, 0), (WARP_W - 1, WARP_H - 1), (0, WARP_H - 1)))
    inverse = cv2.getPerspectiveTransform(destination, quad)
    projected = cv2.warpPerspective(paper, inverse, (CAM_W, CAM_H))
    mask = cv2.warpPerspective(np.full((WARP_H, WARP_W), 255, np.uint8), inverse, (CAM_W, CAM_H))
    raw[mask > 0] = projected[mask > 0]
    return raw


def self_test():
    raw = synthetic_frame()
    start = ticks_ms()
    quad, homography = detect_a4(raw)
    find_ms = elapsed_ms(start)
    if homography is None:
        raise AssertionError("synthetic A4 was not detected")
    analysis, status, timings = analyze_once(raw, homography)
    if status != "OK" or analysis is None or analysis["solution"] is None:
        raise AssertionError("synthetic puzzle failed: %s" % status)
    solution = analysis["solution"]
    if len(analysis["pieces"]) != 3:
        raise AssertionError("expected 3 pieces, got %d" % len(analysis["pieces"]))
    if solution["fill_rate"] < 0.94 or solution["overlap_ratio"] > 0.05:
        raise AssertionError("weak synthetic assembly: %r" % solution)
    if not solution_meets_constraints(solution):
        raise AssertionError("synthetic assembly did not pass hard constraints: %r" % solution)
    constraint_mutations = (
        ("fill_rate", MIN_FILL_RATE - 0.001),
        ("overlap_ratio", MAX_FINAL_OVERLAP_RATIO + 0.001),
        ("size_mm", (MAX_LONG_MM + 1.0, solution["size_mm"][1])),
        ("size_mm", (solution["size_mm"][0], MIN_SHORT_MM - 1.0)),
        ("outline_is_rectangle", False),
    )
    for key, value in constraint_mutations:
        invalid = solution.copy()
        invalid[key] = value
        if solution_meets_constraints(invalid):
            raise AssertionError("hard constraint was not enforced: %s=%r" % (key, value))
    rectangle_mask = np.zeros((140, 220), np.uint8)
    cv2.rectangle(rectangle_mask, (20, 20), (200, 120), 1, -1)
    if not rectangle_outline_metrics(rectangle_mask)[2]:
        raise AssertionError("axis-aligned rectangle outline was rejected")
    trapezoid_mask = np.zeros_like(rectangle_mask)
    cv2.fillPoly(trapezoid_mask, [np.int32(((20, 20), (200, 20), (160, 120), (60, 120)))], 1)
    if rectangle_outline_metrics(trapezoid_mask)[2]:
        raise AssertionError("non-right-angle quadrilateral was accepted")
    too_small = [
        np.float32(((0, 0), (60, 0), (0, 60))),
        np.float32(((0, 0), (60, 0), (0, 60))),
    ]
    invalid_solution, _, _, _ = solve_puzzle(too_small)
    if invalid_solution is not None:
        raise AssertionError("invalid-size assembly was accepted: %r" % invalid_solution)
    topology_cases = (
        (
            (
                np.float32(((0, 0), (80, 0), (105, 120), (0, 120))),
                np.float32(((80, 0), (190, 0), (190, 120), (105, 120))),
            ),
            ((17, (20, 30)), (-31, (270, 180))),
        ),
        (
            (
                np.float32(((0, 0), (80, 0), (100, 60), (0, 70))),
                np.float32(((80, 0), (190, 0), (190, 50), (100, 60))),
                np.float32(((100, 60), (190, 50), (190, 120), (110, 120))),
                np.float32(((0, 70), (100, 60), (110, 120), (0, 120))),
            ),
            ((12, (20, 40)), (-28, (260, 40)), (47, (260, 240)), (-53, (60, 260))),
        ),
    )
    for source_pieces, poses in topology_cases:
        scattered = [affine_polygon(piece, angle, translation) for piece, (angle, translation) in zip(source_pieces, poses)]
        topology_solution, _, _, _ = solve_puzzle(scattered)
        if topology_solution is None or not solution_meets_constraints(topology_solution):
            raise AssertionError("%d-piece topology regression failed" % len(source_pieces))
    t_junction = (
        np.float32(((0, 0), (80, 0), (80, 120), (0, 120))),
        np.float32(((80, 0), (190, 0), (190, 50), (80, 50))),
        np.float32(((80, 50), (190, 50), (190, 120), (80, 120))),
    )
    t_poses = ((11, (30, 40)), (-23, (245, 55)), (37, (190, 250)))
    t_scattered = [affine_polygon(piece, angle, translation) for piece, (angle, translation) in zip(t_junction, t_poses)]
    t_solution, _, _, _ = solve_puzzle(t_scattered)
    if len(split_edge_piece_sets(t_scattered)) <= 1 or t_solution is None:
        raise AssertionError("T-junction split-edge regression failed")
    capture_scale = WARP_W / 165.0
    capture_replay = [
        np.float32(points) * capture_scale
        for points in (
            ((243.9, 58.8), (269.4, 78.2), (237.6, 103.8), (217.6, 82.0)),
            ((264.4, 103.4), (269.9, 153.4), (239.7, 155.2), (237.7, 112.9)),
            ((264.4, 103.4), (320.3, 78.9), (327.4, 128.6)),
            ((329.6, 49.2), (336.9, 119.5), (365.7, 96.7)),
        )
    ]
    capture_solution, _, _, _ = solve_puzzle(capture_replay)
    if capture_solution is None or capture_solution.get("split_edge") is None:
        raise AssertionError("real four-piece capture replay failed")
    if capture_solution.get("raster_evaluations", MAX_COMPLETE_CANDIDATES) > 10:
        raise AssertionError(
            "real four-piece capture replay performed too many raster evaluations: %r"
            % capture_solution.get("raster_evaluations")
        )
    search_limit_replay = [
        np.float32(points) * capture_scale
        for points in (
            ((240.0, 40.2), (281.6, 44.5), (286.2, 72.8), (238.2, 70.5)),
            ((238.8, 76.5), (267.7, 79.6), (264.1, 113.7), (233.3, 118.1)),
            ((331.3, 32.3), (361.9, 71.9), (293.0, 79.7)),
            ((273.3, 96.1), (333.3, 88.1), (338.9, 123.9)),
        )
    ]
    progress_updates = []
    search_limit_solution, _, search_limit_truncated, _ = solve_puzzle(
        search_limit_replay,
        lambda count, elapsed, progress: progress_updates.append((count, elapsed, progress)),
    )
    if search_limit_solution is None or search_limit_truncated:
        raise AssertionError("second real capture replay hit the search limit")
    if len(progress_updates) < 2 or progress_updates[0][0] != 0:
        raise AssertionError("search progress callback was not updated")
    if progress_updates[-1][0] != search_limit_solution["topologies_tested"]:
        raise AssertionError("search progress did not report the final candidate count")
    if any(after[2] < before[2] for before, after in zip(progress_updates, progress_updates[1:])):
        raise AssertionError("search progress moved backwards")
    interleaved_search_replay = [
        np.float32(points) * capture_scale
        for points in (
            ((248.1, 130.3), (290.7, 132.7), (295.6, 160.6), (247.7, 160.9)),
            ((256.1, 176.3), (285.3, 177.2), (284.0, 211.6), (253.5, 218.5)),
            ((340.0, 125.5), (366.6, 169.4), (296.1, 168.1)),
            ((292.0, 192.8), (350.9, 177.9), (361.0, 213.0)),
        )
    ]
    interleaved_solution, _, interleaved_truncated, _ = solve_puzzle(interleaved_search_replay)
    if interleaved_solution is None or interleaved_truncated:
        raise AssertionError("third real capture replay exhausted sequential variants")
    three_way_split_replay = [
        np.float32(points) * capture_scale
        for points in (
            ((284.0, 41.0), (272.0, 42.0), (247.0, 107.0), (271.0, 92.0)),
            ((298.0, 46.0), (293.0, 49.0), (281.0, 86.0), (297.0, 73.0)),
            ((315.0, 54.0), (304.0, 53.0), (302.0, 76.0), (314.0, 66.0)),
            ((321.0, 81.0), (266.0, 119.0), (301.0, 130.0)),
        )
    ]
    three_way_solution, _, three_way_truncated, _ = solve_puzzle(three_way_split_replay)
    if three_way_solution is None or three_way_truncated:
        raise AssertionError("one-long-edge to three-short-edges replay failed")
    if not three_way_solution.get("inverse_fit"):
        raise AssertionError("irregular capture should use the inverse-fit fallback")
    if not inverse_solution_meets_constraints(three_way_solution):
        raise AssertionError("inverse-fit capture violated its relaxed constraints")
    print("SELF_TEST_OK")
    print("pieces=%d candidates=%d truncated=%s" % (len(analysis["pieces"]), analysis["candidates"], analysis["truncated"]))
    print("size_mm=%.1fx%.1f fill=%.4f gap_mm2=%.1f overlap_mm2=%.1f" % (solution["size_mm"][0], solution["size_mm"][1], solution["fill_rate"], solution["gap_mm2"], solution["overlap_mm2"]))
    print("timings_ms=find_a4:%d %s" % (find_ms, " ".join("%s:%d" % item for item in timings.items())))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        run_device()
