"""Task 2(1): white-piece topology solver from 2026_vision_v1.

The edge matching and rectangle assembly below are kept aligned with
``maixcam_pro/puzzle_solver.py`` at upstream commit ``39c94d8``.  The small
adapter at the end accepts the A4 image already rectified by 2026_new and
returns the shared PieceAction contract.
"""

import itertools
import math
import time as wall_time

import cv2
import numpy as np

try:
    from . import task2_config as config
except ImportError:  # Run directly in MaixVision with this folder as project root.
    import task2_config as config


class SolveTimeoutError(RuntimeError):
    """Stop one solve cycle before it consumes the whole contest window."""

    def __init__(self):
        seconds = config.SOLVE_TIMEOUT_SECONDS
        message = ("SOLVE TIMEOUT" if seconds is None else
                   "SOLVE TIMEOUT %dS" % seconds)
        super().__init__(message)


def _solve_progress(message):
    """Emit low-volume progress logs visible in MaixVision/SSH."""
    print("[SOLVE-PROGRESS] %s" % message, flush=True)


def new_solve_deadline(timeout_seconds=None):
    if timeout_seconds is None:
        timeout_seconds = config.SOLVE_TIMEOUT_SECONDS
    if timeout_seconds is None or float(timeout_seconds) <= 0.0:
        return None
    return wall_time.perf_counter() + float(timeout_seconds)


def _check_solve_deadline(deadline):
    if deadline is not None and wall_time.perf_counter() >= deadline:
        raise SolveTimeoutError()


def rigid(angle, tx=0.0, ty=0.0):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([
        [cosine, -sine, tx],
        [sine, cosine, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def apply_h(points, transform):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = homogeneous @ transform.T
    return mapped[:, :2] / mapped[:, 2, None]


def _apply_rigid(points, transform):
    """Apply a solver rigid transform without allocating homogeneous points."""
    points = np.asarray(points)
    return points @ transform[:2, :2].T + transform[:2, 2]


def edges(polygon):
    return [(polygon[i], polygon[(i + 1) % len(polygon)])
            for i in range(len(polygon))]


def align_edge(src_a, src_b, dst_a, dst_b):
    """Rigid transform mapping src_a/src_b to dst_a/dst_b."""
    source = src_b - src_a
    target = dst_b - dst_a
    angle = (math.atan2(target[1], target[0])
             - math.atan2(source[1], source[0]))
    transform = rigid(angle)
    mapped = apply_h(np.array([src_a]), transform)[0]
    transform[:2, 2] = dst_a - mapped
    return transform


def candidate_matchings(pieces, max_candidates=None, match_priority=None,
                        deadline=None):
    """Shortlist upstream v2.1 full-edge and T-junction matches."""
    if max_candidates is None:
        max_candidates = config.MAX_EDGE_CANDIDATES
    all_edges = {(piece_index, edge_index): edge
                 for piece_index, piece in enumerate(pieces)
                 for edge_index, edge in enumerate(edges(piece))}
    candidates = []
    for (i, ei), (j, ej) in itertools.combinations(all_edges, 2):
        _check_solve_deadline(deadline)
        if i == j:
            continue
        a, b = all_edges[(i, ei)]
        c, d = all_edges[(j, ej)]
        length_a = np.linalg.norm(b - a)
        length_b = np.linalg.norm(d - c)
        relative_error = abs(length_a - length_b) / max(
            length_a, length_b, 1e-9)
        if relative_error < config.EDGE_LENGTH_TOLERANCE:
            candidates.append(
                (relative_error, i, ei, j, ej, 0.0, 1.0, 0.0, 1.0))
        ratio = min(length_a, length_b) / max(length_a, length_b, 1e-9)
        if (config.PARTIAL_EDGE_MIN_RATIO <= ratio
                <= config.PARTIAL_EDGE_MAX_RATIO):
            penalty = config.PARTIAL_EDGE_PENALTY
            if length_a > length_b:
                candidates.extend((
                    (penalty, i, ei, j, ej, 0.0, ratio, 0.0, 1.0),
                    (penalty, i, ei, j, ej,
                     1.0 - ratio, 1.0, 0.0, 1.0),
                ))
            else:
                candidates.extend((
                    (penalty, i, ei, j, ej, 0.0, 1.0, 0.0, ratio),
                    (penalty, i, ei, j, ej,
                     0.0, 1.0, 1.0 - ratio, 1.0),
                ))
    candidates.sort()
    candidates = candidates[:max_candidates]
    if match_priority is not None:
        candidates.sort(key=lambda match: (
            match_priority(pieces, match), match,
        ))
    return candidates


def match_segments(pieces, match):
    """Return the full or partial edge segments encoded by one match."""
    _, i, edge_i, j, edge_j, ia0, ia1, ja0, ja1 = match
    a, b = edges(pieces[i])[edge_i]
    c, d = edges(pieces[j])[edge_j]
    return (a + (b - a) * ia0, a + (b - a) * ia1,
            c + (d - c) * ja0, c + (d - c) * ja1)


def matching_sets(pieces, cut_mode="auto", max_full=None, max_partial=None,
                  candidate_limit=None, partial_ratio_range=None,
                  match_priority=None, deadline=None):
    """Enumerate connected v2.1 topology candidates without generator truth."""
    count = len(pieces)
    if count == 1:
        yield ()
        return

    candidates = candidate_matchings(
        pieces, candidate_limit, match_priority, deadline)
    pair_count = (count if ((cut_mode == "common" and count >= 3)
                            or (cut_mode == "concave" and count >= 2))
                  else count - 1)
    full = [match for match in candidates
            if tuple(match[5:]) == (0.0, 1.0, 0.0, 1.0)]
    partial = [match for match in candidates
               if tuple(match[5:]) != (0.0, 1.0, 0.0, 1.0)]
    if partial_ratio_range is not None:
        minimum_ratio, maximum_ratio = partial_ratio_range
        partial = [
            match for match in partial
            if minimum_ratio <= min(
                match[6] - match[5], match[8] - match[7],
            ) <= maximum_ratio
        ]
    if max_full is not None:
        full = full[:max_full]
    if max_partial is not None:
        partial = partial[:max_partial]
    if cut_mode == "multi_partial" and count == 4:
        combinations = (
            (full_match,) + partial_matches
            for full_match in full
            for partial_matches in itertools.combinations(partial, 2)
        )
    elif cut_mode == "t_junction" and count >= 3:
        combinations = (
            tuple(base) + (part,)
            for base in itertools.combinations(full, pair_count - 1)
            for part in partial)
    elif cut_mode in {
            "common", "boundary_fan", "strips", "corner", "concave",
            "equal_rectangles", "sequential"}:
        combinations = itertools.combinations(full, pair_count)
    else:
        combinations = itertools.chain(
            itertools.combinations(full, pair_count),
            (tuple(base) + (part,)
             for base in itertools.combinations(full, pair_count - 1)
             for part in partial),
        )
    for combination in combinations:
        _check_solve_deadline(deadline)
        used_edges = set()
        degree = [0] * count
        graph = [set() for _ in range(count)]
        valid = True
        for match in combination:
            _, i, ei, j, ej = match[:5]
            if (i, ei) in used_edges or (j, ej) in used_edges:
                valid = False
                break
            used_edges.update(((i, ei), (j, ej)))
            degree[i] += 1
            degree[j] += 1
            graph[i].add(j)
            graph[j].add(i)
        if not valid or any(value == 0 for value in degree):
            continue
        if (cut_mode == "common" and count >= 3
                and any(value != 2 for value in degree)):
            continue

        visited, stack = {0}, [0]
        while stack:
            for neighbour in graph[stack.pop()]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        if len(visited) == count:
            yield combination


def optimize_pose_graph(pieces, matches, initial):
    """Distribute closed-loop endpoint error; copied in simplified form."""
    if len(pieces) < 3:
        return initial
    _solve_progress("pose optimization start pieces=%d matches=%d" % (
        len(pieces), len(matches)))

    def pack(poses):
        values = []
        for pose in poses[1:]:
            values.extend((math.atan2(pose[1, 0], pose[0, 0]),
                           pose[0, 2], pose[1, 2]))
        return np.asarray(values, dtype=np.float64)

    def unpack(values):
        poses = [initial[0]]
        for index in range(len(pieces) - 1):
            theta, tx, ty = values[3 * index:3 * index + 3]
            poses.append(rigid(theta, tx, ty))
        return poses

    def residual(values):
        poses = unpack(values)
        result = []
        for match in matches:
            _, i, _ei, j, _ej = match[:5]
            ia, ib, ja, jb = match_segments(pieces, match)
            world_i = apply_h(np.array([ia, ib]), poses[i])
            world_j = apply_h(np.array([jb, ja]), poses[j])
            result.extend((world_i - world_j).ravel())
        return np.asarray(result)

    values = pack(initial)
    for iteration in range(15):
        if iteration == 0 or iteration % 5 == 0:
            _solve_progress("pose optimization iteration=%d" % (iteration + 1))
        current = residual(values)
        jacobian = np.empty((len(current), len(values)))
        for index in range(len(values)):
            step = 1e-5 if index % 3 == 0 else 1e-3
            shifted = values.copy()
            shifted[index] += step
            jacobian[:, index] = (residual(shifted) - current) / step
        delta, *_ = np.linalg.lstsq(jacobian, -current, rcond=None)
        values += delta
        if np.linalg.norm(delta) < 1e-7:
            break
    _solve_progress("pose optimization complete")
    return unpack(values)


def _assembly_quality(assembled, matches, closure_error, return_metrics=False):
    all_points = np.vstack(assembled)
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    shift = -minimum + 6
    width, height = np.ceil(maximum - minimum + 12).astype(int)
    masks = []
    for polygon in assembled:
        mask = np.zeros((height, width), np.uint8)
        points = np.round(polygon + shift).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask)

    total = sum(masks)
    overlap = float(np.count_nonzero(total > 1))
    union = (total > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    rectangle_area = max(1.0, rect[1][0] * rect[1][1])
    union_area = float(np.count_nonzero(union))
    contour_area = cv2.contourArea(contour)
    fill_error = max(0.0, rectangle_area - union_area)
    fill_ratio = min(1.0, union_area / rectangle_area)
    source_area = sum(abs(cv2.contourArea(
        polygon.astype(np.float32))) for polygon in assembled)
    overlap_ratio = overlap / max(source_area, 1.0)
    short_side_px, long_side_px = sorted(float(value) for value in rect[1])
    target_size_mm = (
        short_side_px * config.A4_MM_PER_PIXEL,
        long_side_px * config.A4_MM_PER_PIXEL,
    )
    dimension_error = _target_dimension_error(target_size_mm)
    rectangle_perimeter = 2.0 * (short_side_px + long_side_px)
    disconnected_area = sum(cv2.contourArea(item) for item in contours)
    disconnected_area -= contour_area
    perimeter_error = abs(
        cv2.arcLength(contour, True) - rectangle_perimeter)
    match_error = sum(match[0] for match in matches) * 5000.0
    score = (
        closure_error * 8.0
        + overlap * 12.0
        + fill_error * 8.0
        + abs(union_area - source_area) * 4.0
        + abs(rectangle_area - source_area) * 3.0
        + dimension_error * max(source_area, 1.0) * 0.85
        + disconnected_area * 20.0
        + perimeter_error * 25.0
        + match_error
    )
    if return_metrics:
        return score, fill_ratio, overlap_ratio, target_size_mm
    return score, fill_ratio


def _target_bbox_size_mm(assembled):
    """Return the overall placed-piece rectangle size, including gaps."""
    all_points = np.vstack(assembled).astype(np.float32)
    _center, size, _angle = cv2.minAreaRect(all_points)
    short_side_px, long_side_px = sorted(float(value) for value in size)
    return (short_side_px * config.A4_MM_PER_PIXEL,
            long_side_px * config.A4_MM_PER_PIXEL)


def _range_error(value, bounds):
    minimum, maximum = bounds
    if value < minimum:
        return (minimum - value) / max(minimum, 1e-9)
    if value > maximum:
        return (value - maximum) / max(maximum, 1e-9)
    return 0.0


def _target_dimension_error(target_size_mm):
    short_side_mm, long_side_mm = sorted(target_size_mm)
    return (
        _range_error(short_side_mm, config.TARGET_SHORT_SIDE_MM_RANGE)
        + _range_error(long_side_mm, config.TARGET_LONG_SIDE_MM_RANGE)
    )


def _target_dimensions_valid(target_size_mm):
    short_side_mm, long_side_mm = sorted(target_size_mm)
    tolerance = config.TARGET_DIMENSION_TOLERANCE_MM
    short_minimum, short_maximum = config.TARGET_SHORT_SIDE_MM_RANGE
    long_minimum, long_maximum = config.TARGET_LONG_SIDE_MM_RANGE
    return (
        short_minimum - tolerance <= short_side_mm <= short_maximum + tolerance
        and long_minimum - tolerance <= long_side_mm <= long_maximum + tolerance
    )


def _passes_final_quality(fill_ratio, overlap_ratio, target_size_mm,
                          min_rectangle_fill=None,
                          candidate_validator=None):
    if min_rectangle_fill is None:
        min_rectangle_fill = config.MIN_RECTANGLE_FILL
    valid = (
        fill_ratio >= min_rectangle_fill
        and overlap_ratio <= config.MAX_RECTANGLE_OVERLAP_RATIO
        and _target_dimensions_valid(target_size_mm)
    )
    if valid and candidate_validator is not None:
        valid = bool(candidate_validator(target_size_mm))
    return valid


def _dynamic_piece_scales(pieces, matches):
    """Estimate bounded per-piece uniform scales from matched seam lengths."""
    count = len(pieces)
    if count < 2 or not matches:
        return np.ones(count, dtype=np.float64)

    equations = []
    targets = []
    for match in matches:
        ia, ib, ja, jb = match_segments(pieces, match)
        length_i = float(np.linalg.norm(ib - ia))
        length_j = float(np.linalg.norm(jb - ja))
        if length_i <= 1e-6 or length_j <= 1e-6:
            continue
        row = np.zeros(count, dtype=np.float64)
        row[match[1]] = 1.0
        row[match[3]] = -1.0
        equations.append(row)
        targets.append(math.log(length_j / length_i))
    if not equations:
        return np.ones(count, dtype=np.float64)

    log_scales, *_ = np.linalg.lstsq(
        np.vstack(equations), np.asarray(targets), rcond=None)
    scales = np.exp(log_scales)
    scales /= math.exp(float(np.mean(np.log(scales))))
    return np.clip(
        scales, config.DYNAMIC_SCALE_MIN, config.DYNAMIC_SCALE_MAX)


def _scale_pieces_about_centres(pieces, scales):
    """Uniformly scale each polygon without changing its centre or shape."""
    scaled = []
    for piece, scale in zip(pieces, scales):
        center = piece.mean(axis=0)
        scaled.append(center + (piece - center) * float(scale))
    return scaled


def rough_assembly_from_matches(pieces, matches,
                                match_transform_cache=None):
    """Place one topology and reject implausible bounds without raster work."""
    if match_transform_cache is None:
        match_transform_cache = {}
    adjacency = [[] for _ in pieces]
    for match in matches:
        key = tuple(match)
        relative = match_transform_cache.get(key)
        if relative is None:
            ia, ib, ja, jb = match_segments(pieces, match)
            relative = (
                align_edge(ja, jb, ib, ia),
                align_edge(ia, ib, jb, ja),
            )
            match_transform_cache[key] = relative
        _, i, _ei, j, _ej = match[:5]
        adjacency[i].append((j, relative[0]))
        adjacency[j].append((i, relative[1]))

    transforms = [None] * len(pieces)
    transforms[0] = np.eye(3)
    stack = [0]
    closure_error = 0.0
    while stack:
        i = stack.pop()
        for j, relative in adjacency[i]:
            proposed = transforms[i] @ relative
            if transforms[j] is None:
                transforms[j] = proposed
                stack.append(j)
            else:
                error = _apply_rigid(
                    pieces[j], proposed) - _apply_rigid(
                        pieces[j], transforms[j])
                closure_error += np.linalg.norm(error, axis=1).mean()

    if any(transform is None for transform in transforms):
        return None
    assembled = [_apply_rigid(piece, transform)
                 for piece, transform in zip(pieces, transforms)]
    all_points = np.vstack(assembled).astype(np.float32)
    rect = cv2.minAreaRect(all_points)
    rectangle_area = max(1.0, rect[1][0] * rect[1][1])
    source_area = sum(abs(cv2.contourArea(
        piece.astype(np.float32))) for piece in pieces)
    rough_fill = source_area / rectangle_area
    if (rough_fill < config.MIN_RECTANGLE_FILL * 0.72
            or rough_fill > 1.18):
        return None
    match_error = sum(match[0] for match in matches) * 5000.0
    rough_score = (abs(1.0 - rough_fill) * source_area
                   + closure_error * 8.0 + match_error)
    return rough_score, min(1.0, rough_fill), transforms, assembled, closure_error


def assemble_from_matches(pieces, matches, min_rough_fill=None,
                          match_transform_cache=None):
    rough = rough_assembly_from_matches(
        pieces, matches, match_transform_cache)
    if rough is None:
        return None
    if (min_rough_fill is not None
            and rough[1] < min_rough_fill):
        return None
    _rough_score, _rough_fill, transforms, assembled, closure_error = rough
    score, fill_ratio = _assembly_quality(assembled, matches, closure_error)
    return score, fill_ratio, transforms


def _equal_rectangle_transforms(pieces):
    """Resolve blank equal rectangles whose piece identities are unobservable."""
    count = len(pieces)
    if count not in (2, 3, 4):
        return None
    dimensions = []
    for piece in pieces:
        contour = piece.astype(np.float32).reshape(-1, 1, 2)
        if len(piece) != 4 or not cv2.isContourConvex(contour):
            return None
        width, height = cv2.minAreaRect(contour)[1]
        rectangle_area = max(width * height, 1.0)
        if (abs(cv2.contourArea(contour)) / rectangle_area
                < config.EQUAL_RECTANGLE_MIN_FILL):
            return None
        dimensions.append((min(width, height), max(width, height)))
    dimensions = np.asarray(dimensions, dtype=np.float64)
    mean = dimensions.mean(axis=0)
    if np.any(np.ptp(dimensions, axis=0) / np.maximum(mean, 1.0)
              > config.EQUAL_RECTANGLE_SIZE_TOLERANCE):
        return None

    if count == 4:
        cell_width, cell_height = mean[1], mean[0]
        slots = ((0.0, 0.0), (cell_width, 0.0),
                 (0.0, cell_height), (cell_width, cell_height))
    else:
        cell_width, cell_height = mean[0], mean[1]
        slots = tuple((index * cell_width, 0.0)
                      for index in range(count))

    transforms = []
    for piece, slot in zip(pieces, slots):
        best = None
        for start, end in edges(piece):
            vector = end - start
            angle = -math.atan2(vector[1], vector[0])
            rotation = rigid(angle)
            rotated = apply_h(piece, rotation)
            minimum, maximum = rotated.min(axis=0), rotated.max(axis=0)
            size = maximum - minimum
            cost = abs(size[0] - cell_width) + abs(size[1] - cell_height)
            if best is None or cost < best[0]:
                best = (cost, rotation, minimum)
        _, rotation, minimum = best
        transforms.append(
            rigid(0.0, *(np.asarray(slot) - minimum)) @ rotation)
    return transforms


def _point_segment_distance(point, start, end):
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, segment)) / length_squared
    closest = start + min(1.0, max(0.0, ratio)) * segment
    return float(np.linalg.norm(point - closest))


def _segments_intersect(a, b, c, d):
    def cross(first, second, third):
        first_vector = second - first
        second_vector = third - first
        return float(first_vector[0] * second_vector[1]
                     - first_vector[1] * second_vector[0])

    ab_c, ab_d = cross(a, b, c), cross(a, b, d)
    cd_a, cd_b = cross(c, d, a), cross(c, d, b)
    epsilon = 1e-7
    if (ab_c * ab_d < -epsilon and cd_a * cd_b < -epsilon):
        return True
    if abs(ab_c) <= epsilon and _point_segment_distance(c, a, b) <= epsilon:
        return True
    if abs(ab_d) <= epsilon and _point_segment_distance(d, a, b) <= epsilon:
        return True
    if abs(cd_a) <= epsilon and _point_segment_distance(a, c, d) <= epsilon:
        return True
    if abs(cd_b) <= epsilon and _point_segment_distance(b, c, d) <= epsilon:
        return True
    return False


def _polygon_distance(first, second):
    """Return the shortest boundary distance between two simple polygons."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_cv = first.astype(np.float32)
    second_cv = second.astype(np.float32)
    if (cv2.pointPolygonTest(first_cv, tuple(second[0]), False) >= 0
            or cv2.pointPolygonTest(second_cv, tuple(first[0]), False) >= 0):
        return 0.0
    best = float("inf")
    for index, start in enumerate(first):
        end = first[(index + 1) % len(first)]
        for other_index, other_start in enumerate(second):
            other_end = second[(other_index + 1) % len(second)]
            if _segments_intersect(start, end, other_start, other_end):
                return 0.0
            best = min(
                best,
                _point_segment_distance(start, other_start, other_end),
                _point_segment_distance(end, other_start, other_end),
                _point_segment_distance(other_start, start, end),
                _point_segment_distance(other_end, start, end),
            )
    return best


def _minimum_pairwise_gap(polygons):
    if len(polygons) < 2:
        return float("inf")
    return min(
        _polygon_distance(polygons[first], polygons[second])
        for first in range(len(polygons))
        for second in range(first + 1, len(polygons))
    )


def _spread_piece_transforms(pieces, transforms, gap_pixels):
    """Add the requested clearance with minimal pairwise translations."""
    if len(pieces) < 2 or gap_pixels <= 0.0:
        return transforms
    assembled = [apply_h(piece, transform)
                 for piece, transform in zip(pieces, transforms)]
    if _minimum_pairwise_gap(assembled) >= gap_pixels:
        return transforms

    offsets = np.zeros((len(assembled), 2), dtype=np.float64)
    tolerance = 1e-3
    for _ in range(32):
        moved = False
        current = [polygon + offset
                   for polygon, offset in zip(assembled, offsets)]
        for first in range(len(current)):
            for second in range(first + 1, len(current)):
                distance = _polygon_distance(
                    current[first], current[second])
                if distance + tolerance >= gap_pixels:
                    continue
                direction = (current[first].mean(axis=0)
                             - current[second].mean(axis=0))
                length = float(np.linalg.norm(direction))
                if length <= 1e-6:
                    angle = 2.0 * math.pi * first / len(current)
                    direction = np.array(
                        (math.cos(angle), math.sin(angle)))
                else:
                    direction /= length
                shift = direction * (gap_pixels - distance) * 0.50
                offsets[first] += shift
                offsets[second] -= shift
                current[first] = assembled[first] + offsets[first]
                current[second] = assembled[second] + offsets[second]
                moved = True
        separated = [polygon + offset
                     for polygon, offset in zip(assembled, offsets)]
        if _minimum_pairwise_gap(separated) + tolerance >= gap_pixels:
            return [rigid(0.0, *offset) @ transform
                    for offset, transform in zip(offsets, transforms)]
        if not moved:
            break
    raise RuntimeError("无法为拼好碎片预留 %.1f mm 安全间距" % (
        gap_pixels * config.A4_MM_PER_PIXEL))


def _target_transform(pieces, transforms, paper, enforce_placement=True):
    assembled = [apply_h(piece, transform)
                 for piece, transform in zip(pieces, transforms)]
    all_points = np.vstack(assembled).astype(np.float32)
    _, size, angle_degrees = cv2.minAreaRect(all_points)
    if size[0] < size[1]:
        angle_degrees += 90.0

    normalize = rigid(math.radians(-angle_degrees))
    rotated = apply_h(all_points, normalize)
    minimum, maximum = rotated.min(axis=0), rotated.max(axis=0)
    if (maximum - minimum)[0] < (maximum - minimum)[1]:
        normalize = rigid(math.radians(90.0 - angle_degrees))
        rotated = apply_h(all_points, normalize)
        minimum, maximum = rotated.min(axis=0), rotated.max(axis=0)

    normalized = [normalize @ transform for transform in transforms]
    gap_pixels = config.TARGET_PIECE_GAP_MM / config.A4_MM_PER_PIXEL
    normalized = _spread_piece_transforms(pieces, normalized, gap_pixels)
    assembled = [apply_h(piece, transform)
                 for piece, transform in zip(pieces, normalized)]
    minimum = np.vstack(assembled).min(axis=0)
    maximum = np.vstack(assembled).max(axis=0)

    x, y, width, height = cv2.boundingRect(paper.astype(np.int32))
    recovered_size = maximum - minimum
    margin = min(width, height) * config.TARGET_MARGIN_RATIO
    if (enforce_placement
            and np.any(recovered_size
                       > np.array((width, height)) - 2.0 * margin)):
        raise RuntimeError("预留 %.1f mm 间距后的拼图超出 A4 可放置区域" % (
            config.TARGET_PIECE_GAP_MM))
    target_center_x = x + width * config.TARGET_CENTER_X_RATIO
    target_origin = np.array([
        target_center_x - recovered_size[0] / 2.0,
        y + height * 0.5 + (
            config.TARGET_AXIS_CLEARANCE_MM / config.A4_MM_PER_PIXEL),
    ])
    target_origin[0] = min(max(target_origin[0], x + margin),
                           x + width - margin - recovered_size[0])
    maximum_origin_y = y + height - margin - recovered_size[1]
    if enforce_placement and target_origin[1] > maximum_origin_y:
        raise RuntimeError("拼好图无法在中轴线下方预留 %.1f mm 安全距离" % (
            config.TARGET_AXIS_CLEARANCE_MM))
    translate = rigid(0.0, *(target_origin - minimum))
    return [translate @ transform for transform in normalized]


def solve(pieces, paper, cut_mode="auto", min_rectangle_fill=None,
          fast_full_candidates=None, fast_min_rough_fill=None,
          accept_fast_best=False, defer_fast_accept=False,
          candidate_limit=None,
          partial_ratio_range=None, deadline=None,
          match_priority=None, topology_priority=None, finalist_count=0,
          finalist_max_fill_loss=0.0, fast_max_topologies=None,
          candidate_recorder=None, dynamic_scaling=False,
          result_metadata=None, candidate_validator=None):
    """Return final per-piece 3x3 transforms, selected matches and fill ratio."""
    if not 1 <= len(pieces) <= 4:
        raise ValueError("碎片数量必须为 1～4")
    _solve_progress("geometry start pieces=%d cut_mode=%s" % (
        len(pieces), cut_mode))
    if min_rectangle_fill is None:
        min_rectangle_fill = config.MIN_RECTANGLE_FILL
    if fast_full_candidates is None:
        fast_full_candidates = config.FAST_SEARCH_FULL_CANDIDATES
    if fast_min_rough_fill is None:
        fast_min_rough_fill = config.FAST_SEARCH_MIN_ROUGH_FILL
    _check_solve_deadline(deadline)

    transforms = (_equal_rectangle_transforms(pieces)
                  if (not dynamic_scaling
                      and cut_mode in ("auto", "equal_rectangles")) else None)
    matches = ()
    optimized_in_search = False
    scale_by_matches = {}
    selected_scales = np.ones(len(pieces), dtype=np.float64)
    if dynamic_scaling:
        _solve_progress(
            "dynamic-scale enabled range=%.3f..%.3f" % (
                config.DYNAMIC_SCALE_MIN, config.DYNAMIC_SCALE_MAX))
    if transforms is not None:
        _solve_progress("equal-rectangle shortcut")
    if transforms is None:
        best = None
        best_rank = None
        match_transform_cache = {}
        if cut_mode == "auto":
            search_limits = ((
                fast_full_candidates,
                config.FAST_SEARCH_PARTIAL_CANDIDATES,
            ),)
        elif cut_mode == "multi_partial":
            search_limits = ((fast_full_candidates, None),)
        else:
            search_limits = ((None, None),)
        for max_full, max_partial in search_limits:
            _solve_progress("topology pass full=%s partial=%s" % (
                max_full if max_full is not None else "all",
                max_partial if max_partial is not None else "all"))
            accepted = False
            finalists = []
            for topology_index, candidate_matches in enumerate(matching_sets(
                    pieces, cut_mode, max_full, max_partial,
                    candidate_limit=candidate_limit,
                    partial_ratio_range=partial_ratio_range,
                    match_priority=match_priority, deadline=deadline)):
                if (fast_max_topologies is not None
                        and (max_full is not None or max_partial is not None)
                        and topology_index >= fast_max_topologies):
                    break
                _check_solve_deadline(deadline)
                if topology_index == 0 or topology_index % 100 == 0:
                    _solve_progress("topology candidate=%d matches=%d best=%s" % (
                        topology_index + 1, len(candidate_matches),
                        "yes" if best is not None else "no"))
                min_rough_fill = (
                    fast_min_rough_fill
                    if max_full is not None or max_partial is not None
                    else None
                )
                candidate_scales = (
                    _dynamic_piece_scales(pieces, candidate_matches)
                    if dynamic_scaling
                    else np.ones(len(pieces), dtype=np.float64)
                )
                geometry_pieces = _scale_pieces_about_centres(
                    pieces, candidate_scales)
                scale_by_matches[candidate_matches] = candidate_scales
                result = assemble_from_matches(
                    geometry_pieces, candidate_matches,
                    min_rough_fill=min_rough_fill,
                    match_transform_cache=(
                        None if dynamic_scaling else match_transform_cache))
                candidate_rank = None
                if result is not None:
                    candidate_rank = (
                        (result[0],
                         topology_priority(pieces, candidate_matches))
                        if topology_priority is not None else (result[0],)
                    )
                    if (finalist_count > 0
                            and topology_priority is not None
                            and result[1] >= min_rectangle_fill):
                        finalist = (
                            topology_priority(pieces, candidate_matches),
                            result[0], (*result, candidate_matches),
                        )
                        finalists.append(finalist)
                        finalists.sort(key=lambda item: item[:2])
                        del finalists[finalist_count:]
                        if (candidate_recorder is not None
                                and any(item is finalist
                                        for item in finalists)):
                            finalist_geometry = [
                                apply_h(piece, transform)
                                for piece, transform in zip(
                                    geometry_pieces, result[2])
                            ]
                            finalist_target_transforms = _target_transform(
                                geometry_pieces, result[2], paper,
                                enforce_placement=False)
                            candidate_recorder(
                                finalist_target_transforms,
                                candidate_matches, result[1], candidate_rank,
                                _assembly_quality(
                                    finalist_geometry, candidate_matches, 0.0,
                                    return_metrics=True)[3])
                if (result is not None
                        and (best is None or candidate_rank < best_rank)):
                    previous_best = best
                    previous_best_rank = best_rank
                    best = (*result, candidate_matches)
                    best_rank = candidate_rank
                    if candidate_recorder is not None:
                        candidate_target_transforms = _target_transform(
                            geometry_pieces, result[2], paper,
                            enforce_placement=False)
                        candidate_geometry = [
                            apply_h(piece, transform)
                            for piece, transform in zip(
                                geometry_pieces, result[2])
                        ]
                        candidate_recorder(
                            candidate_target_transforms,
                            candidate_matches, result[1], candidate_rank,
                            _assembly_quality(
                                candidate_geometry, candidate_matches, 0.0,
                                return_metrics=True)[3])
                    if (result[1] >= config.FAST_SEARCH_ACCEPT_FILL
                            or ((dynamic_scaling
                                 or candidate_recorder is not None)
                                and result[1] >= min_rectangle_fill)):
                        _check_solve_deadline(deadline)
                        candidate_transforms = optimize_pose_graph(
                            geometry_pieces, candidate_matches, result[2])
                        candidate_assembled = [
                            apply_h(piece, transform)
                            for piece, transform in zip(
                                geometry_pieces, candidate_transforms)
                        ]
                        quality = _assembly_quality(
                            candidate_assembled, candidate_matches, 0.0,
                            return_metrics=True,
                        )
                        final_rank = (
                            (quality[0], topology_priority(
                                pieces, candidate_matches))
                            if topology_priority is not None
                            else (quality[0],)
                        )
                        if candidate_recorder is not None:
                            candidate_target_transforms = _target_transform(
                                geometry_pieces, candidate_transforms, paper,
                                enforce_placement=False)
                            candidate_recorder(
                                candidate_target_transforms,
                                candidate_matches, quality[1], final_rank,
                                quality[3])
                        if _passes_final_quality(
                                *quality[1:], min_rectangle_fill,
                                candidate_validator):
                            try:
                                candidate_final = _target_transform(
                                    geometry_pieces,
                                    candidate_transforms, paper)
                            except RuntimeError:
                                candidate_final = None
                            if candidate_final is not None:
                                candidate_final_size_mm = _target_bbox_size_mm([
                                    apply_h(piece, transform)
                                    for piece, transform in zip(
                                        geometry_pieces, candidate_final)
                                ])
                                if (candidate_validator is not None
                                        and not candidate_validator(
                                            candidate_final_size_mm)):
                                    candidate_final = None
                            if (candidate_final is not None
                                    and candidate_recorder is not None):
                                candidate_recorder(
                                    candidate_final, candidate_matches,
                                    quality[1], final_rank,
                                    quality[3])
                            if (candidate_final is not None
                                    and not defer_fast_accept
                                    and (result[1]
                                         >= config.FAST_SEARCH_ACCEPT_FILL
                                         or dynamic_scaling)):
                                best = (
                                    quality[0], quality[1],
                                    candidate_transforms, candidate_matches,
                                )
                                best_rank = final_rank
                                optimized_in_search = True
                                accepted = True
                                _solve_progress(
                                    "accepted fill=%.1f%% matches=%d" % (
                                        quality[1] * 100.0,
                                        len(candidate_matches)))
                                break
                            if (candidate_final is None
                                    and candidate_validator is not None):
                                best = previous_best
                                best_rank = previous_best_rank
                        elif candidate_validator is not None:
                            best = previous_best
                            best_rank = previous_best_rank
            if (not accepted and accept_fast_best
                    and max_full is not None and best is not None):
                raw_candidates = [best]
                raw_candidates.extend(item[2] for item in finalists)
                optimized_candidates = []
                seen_matches = set()
                for raw_candidate in raw_candidates:
                    _check_solve_deadline(deadline)
                    candidate_matches = raw_candidate[3]
                    if candidate_matches in seen_matches:
                        continue
                    seen_matches.add(candidate_matches)
                    candidate_scales = scale_by_matches.get(
                        candidate_matches,
                        np.ones(len(pieces), dtype=np.float64))
                    geometry_pieces = _scale_pieces_about_centres(
                        pieces, candidate_scales)
                    candidate_transforms = optimize_pose_graph(
                        geometry_pieces, candidate_matches, raw_candidate[2])
                    candidate_assembled = [
                        apply_h(piece, transform)
                        for piece, transform in zip(
                            geometry_pieces, candidate_transforms)
                    ]
                    quality = _assembly_quality(
                        candidate_assembled, candidate_matches, 0.0,
                        return_metrics=True,
                    )
                    if _passes_final_quality(
                            *quality[1:], min_rectangle_fill,
                            candidate_validator):
                        candidate_texture = (
                            topology_priority(pieces, candidate_matches)
                            if topology_priority is not None else 0.0
                        )
                        optimized_candidates.append((
                            candidate_texture, -quality[1], quality[0],
                            candidate_transforms, candidate_matches,
                        ))
                if optimized_candidates:
                    highest_fill = max(
                        -item[1] for item in optimized_candidates)
                    optimized_candidates = [
                        item for item in optimized_candidates
                        if -item[1] >= (
                            highest_fill - finalist_max_fill_loss)
                    ]
                    selected = min(
                        optimized_candidates, key=lambda item: item[:3])
                    best = (
                        selected[2], -selected[1],
                        selected[3], selected[4],
                    )
                    best_rank = (selected[2], selected[0])
                    accepted = True
                    optimized_in_search = True
            if accepted:
                _solve_progress("topology pass accepted")
                break
        if best is None:
            raise RuntimeError("未找到满足边长配对与碎片邻接关系的拼接")
        _, _fill_ratio, transforms, matches = best
        selected_scales = scale_by_matches.get(
            matches, np.ones(len(pieces), dtype=np.float64))
        if not optimized_in_search:
            _solve_progress("optimizing best topology matches=%d" % len(matches))
            _check_solve_deadline(deadline)
            transforms = optimize_pose_graph(
                _scale_pieces_about_centres(pieces, selected_scales),
                matches, transforms)
    _solve_progress("quality check")
    _check_solve_deadline(deadline)
    geometry_pieces = _scale_pieces_about_centres(pieces, selected_scales)
    assembled = [apply_h(piece, transform)
                 for piece, transform in zip(geometry_pieces, transforms)]
    assembly_cost, fill_ratio, overlap_ratio, target_size_mm = _assembly_quality(
        assembled, matches, 0.0, return_metrics=True)
    if candidate_recorder is not None:
        rank = ((assembly_cost, topology_priority(pieces, matches))
                if topology_priority is not None else (assembly_cost,))
        final_target_transforms = _target_transform(
            geometry_pieces, transforms, paper,
            enforce_placement=False)
        candidate_recorder(
            final_target_transforms,
            matches, fill_ratio, rank,
            target_size_mm)
    if fill_ratio < min_rectangle_fill:
        raise RuntimeError("最佳拼接的矩形填充率仅 %.1f%%" % (fill_ratio * 100.0))
    if overlap_ratio > config.MAX_RECTANGLE_OVERLAP_RATIO:
        raise RuntimeError("最佳拼接的碎片重叠率达到 %.1f%%" % (
            overlap_ratio * 100.0))
    if not _target_dimensions_valid(target_size_mm):
        raise RuntimeError("最佳拼接的矩形尺寸为 %.1f mm x %.1f mm" % (
            target_size_mm[0], target_size_mm[1]))
    if (candidate_validator is not None
            and not candidate_validator(target_size_mm)):
        raise RuntimeError("最佳拼接的矩形比例不符合当前模式要求")
    final = _target_transform(geometry_pieces, transforms, paper)
    _solve_progress("geometry complete fill=%.1f%% matches=%d" % (
        fill_ratio * 100.0, len(matches)))
    if result_metadata is not None:
        result_metadata["piece_scales"] = tuple(
            float(value) for value in selected_scales)
    return final, matches, fill_ratio


def motion_commands(pieces, transforms):
    """Convert transforms to rotation and pixel translation commands."""
    commands = []
    for index, (piece, transform) in enumerate(zip(pieces, transforms)):
        current_center = piece.mean(axis=0)
        target_center = apply_h(np.array([current_center]), transform)[0]
        delta = target_center - current_center
        commands.append({
            "piece": index,
            "rotation_deg": math.degrees(
                math.atan2(transform[1, 0], transform[0, 0])),
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "distance": float(np.linalg.norm(delta)),
            "matrix_3x3": transform.tolist(),
        })
    return commands


def _stable_piece_orders(pieces):
    """Try topology anchors in a geometry-only order, independent of position."""
    canonical = sorted(
        range(len(pieces)),
        key=lambda index: (
            -abs(cv2.contourArea(pieces[index].astype(np.float32))),
            -cv2.arcLength(pieces[index].astype(np.float32), True),
            len(pieces[index]),
        ),
    )
    for anchor in canonical:
        yield [anchor] + [index for index in canonical if index != anchor]


def _restore_piece_order(values, ordered_indices):
    restored = [None] * len(ordered_indices)
    for ordered_index, original_index in enumerate(ordered_indices):
        restored[original_index] = values[ordered_index]
    return restored


def _restore_match_indices(matches, ordered_indices):
    restored = []
    for match in matches:
        values = list(match)
        values[1] = ordered_indices[values[1]]
        values[3] = ordered_indices[values[3]]
        restored.append(tuple(values))
    return tuple(restored)


class Task2WhiteSolver:
    """Detect white pieces on a rectified A4 plane and solve their topology."""

    name = "题2-纯白"

    def solve(self, rectified_rgb):
        try:
            from legacy_2026_new import detect_pieces
            from core.piece_action import actions_from_transforms
        except ImportError:  # Package-style import on PC.
            from ..legacy_2026_new import detect_pieces
            from ..core.piece_action import actions_from_transforms

        pieces, binary, timings = detect_pieces(rectified_rgb)
        return self.solve_detected(rectified_rgb, pieces, binary, timings)

    def solve_detected(self, rectified_rgb, pieces, binary=None, timings=None,
                       solve_options=None, candidate_validator=None,
                       best_effort_candidate_priority=None,
                       return_best_candidate_on_failure=False,
                       use_default_search=None):
        """Solve a DETECT-stage snapshot without segmenting it a second time."""
        try:
            from core.piece_action import actions_from_transforms
        except ImportError:
            from ..core.piece_action import actions_from_transforms
        del rectified_rgb
        timings = timings or {}
        default_search = (solve_options is None if use_default_search is None
                          else bool(use_default_search))
        solve_options = dict(solve_options or {})
        if candidate_validator is not None:
            solve_options["candidate_validator"] = candidate_validator
        _solve_progress("task2 start pieces=%d search=%s" % (
            len(pieces), "default" if default_search else "custom"))
        max_anchor_orders = solve_options.pop("max_anchor_orders", None)
        return_best_candidate_on_failure = bool(
            return_best_candidate_on_failure)
        if solve_options.get("deadline") is None:
            solve_options["deadline"] = new_solve_deadline()
        if default_search:
            search_paths = (
                ("standard", dict(
                    solve_options,
                    fast_max_topologies=(
                        config.FAST_STANDARD_MAX_TOPOLOGIES))),
                ("multi_partial", dict(
                    solve_options, cut_mode="multi_partial",
                    fast_max_topologies=(
                        config.FAST_MULTI_PARTIAL_MAX_TOPOLOGIES))),
                ("scaled_standard", dict(
                    solve_options, dynamic_scaling=True,
                    fast_full_candidates=(
                        config.DYNAMIC_SCALE_FULL_CANDIDATES),
                    fast_max_topologies=(
                        config.DYNAMIC_SCALE_STANDARD_MAX_TOPOLOGIES))),
                ("scaled_multi_partial", dict(
                    solve_options, cut_mode="multi_partial",
                    dynamic_scaling=True,
                    fast_full_candidates=(
                        config.DYNAMIC_SCALE_FULL_CANDIDATES),
                    fast_max_topologies=(
                        config.DYNAMIC_SCALE_MULTI_PARTIAL_MAX_TOPOLOGIES))),
            )
        else:
            search_paths = ((None, solve_options),)
        pieces = [np.asarray(piece, dtype=np.float64) for piece in pieces]
        if not 1 <= len(pieces) <= 4:
            raise RuntimeError("检测到 %d 块纯白碎片，需要 1-4 块" % len(pieces))
        paper = np.int32((((0, 0),), ((config.A4_WARP_WIDTH - 1, 0),),
                          ((config.A4_WARP_WIDTH - 1,
                            config.A4_WARP_HEIGHT - 1),),
                          ((0, config.A4_WARP_HEIGHT - 1),)))
        first_error = None
        last_error = None
        transforms = None
        matches = ()
        fill_ratio = None
        topology_path = None
        timed_out = False
        best_candidate = None
        timeout_candidates = []
        best_effort_candidate = None
        best_effort = False
        fallback_reason = None
        piece_scales = tuple(1.0 for _piece in pieces)
        for path_name, path_options in search_paths:
            _solve_progress("task2 path=%s" % (path_name or "custom"))
            path_max_anchor_orders = max_anchor_orders
            if (default_search
                    and path_name in ("standard", "scaled_standard")):
                path_max_anchor_orders = config.FAST_STANDARD_MAX_ANCHORS
            for order_index, ordered_indices in enumerate(
                    _stable_piece_orders(pieces)):
                if (path_max_anchor_orders is not None
                        and order_index >= path_max_anchor_orders):
                    break
                ordered_pieces = [pieces[index]
                                  for index in ordered_indices]

                def record_candidate(candidate_transforms, candidate_matches,
                                     candidate_fill_ratio, candidate_rank,
                                     candidate_target_size_mm=None,
                                     ordered_indices=ordered_indices,
                                     path_name=path_name):
                    nonlocal best_candidate, best_effort_candidate
                    candidate = (
                        candidate_rank,
                        _restore_piece_order(
                            candidate_transforms, ordered_indices),
                        _restore_match_indices(
                            candidate_matches, ordered_indices),
                        candidate_fill_ratio,
                        path_name,
                    )
                    if (best_candidate is None
                            or candidate[0] < best_candidate[0]):
                        best_candidate = candidate
                    if (len(candidate_rank) > 1
                            and solve_options.get(
                                "finalist_max_fill_loss", 0.0) > 0.0):
                        timeout_candidates.append(candidate)
                        if len(timeout_candidates) > 64:
                            highest_fill = max(
                                item[3] for item in timeout_candidates)
                            max_fill_loss = float(solve_options[
                                "finalist_max_fill_loss"])
                            near_best_fill = [
                                item for item in timeout_candidates
                                if item[3] >= highest_fill - max_fill_loss
                            ]
                            texture_best = sorted(
                                near_best_fill,
                                key=lambda item: (
                                    item[0][1:], item[0][0], -item[3]),
                            )[:48]
                            fill_best = sorted(
                                timeout_candidates,
                                key=lambda item: (-item[3], item[0]),
                            )[:16]
                            retained_ids = set()
                            timeout_candidates[:] = []
                            for item in texture_best + fill_best:
                                if id(item) in retained_ids:
                                    continue
                                retained_ids.add(id(item))
                                timeout_candidates.append(item)
                    quality_valid = (
                        candidate_validator is None
                        or (candidate_target_size_mm is not None
                            and candidate_validator(
                                candidate_target_size_mm))
                    )
                    if (best_effort_candidate_priority is not None
                            and candidate_target_size_mm is not None):
                        priority = best_effort_candidate_priority(
                            candidate_target_size_mm)
                    elif best_effort_candidate_priority is not None:
                        priority = float("inf")
                    else:
                        priority = 0.0
                    best_effort_rank = (
                        0 if quality_valid else 1,
                        float(priority),
                        -candidate_fill_ratio,
                        candidate_rank,
                    )
                    best_effort_value = (
                        best_effort_rank, candidate[1], candidate[2],
                        candidate_fill_ratio, path_name,
                        candidate_target_size_mm, quality_valid,
                    )
                    if (best_effort_candidate is None
                            or best_effort_value[0]
                            < best_effort_candidate[0]):
                        best_effort_candidate = best_effort_value

                try:
                    _check_solve_deadline(path_options["deadline"])
                    result_metadata = {}
                    transforms, matches, fill_ratio = solve(
                        ordered_pieces, paper,
                        candidate_recorder=record_candidate,
                        result_metadata=result_metadata,
                        **path_options)
                    transforms = _restore_piece_order(
                        transforms, ordered_indices)
                    matches = _restore_match_indices(
                        matches, ordered_indices)
                    piece_scales = tuple(_restore_piece_order(
                        result_metadata.get(
                            "piece_scales",
                            tuple(1.0 for _piece in ordered_pieces)),
                        ordered_indices))
                    topology_path = path_name
                    _solve_progress(
                        "task2 path complete fill=%.1f%% matches=%d" % (
                            fill_ratio * 100.0, len(matches)))
                    break
                except SolveTimeoutError:
                    _solve_progress("task2 path timeout path=%s" % path_name)
                    if (return_best_candidate_on_failure
                            and best_effort_candidate is not None):
                        (_rank, transforms, matches, fill_ratio,
                         candidate_path, best_effort_target_size_mm,
                         best_effort_quality_valid) = best_effort_candidate
                        topology_path = "%s_timeout_best_effort" % (
                            candidate_path or "custom")
                        timed_out = True
                        best_effort = True
                        fallback_reason = "SOLVE TIMEOUT"
                        break
                    timeout_candidate = None
                    if timeout_candidates:
                        highest_fill = max(
                            candidate[3] for candidate in timeout_candidates)
                        max_fill_loss = float(solve_options.get(
                            "finalist_max_fill_loss", 0.0))
                        eligible = [
                            candidate for candidate in timeout_candidates
                            if candidate[3] >= highest_fill - max_fill_loss
                        ]
                        timeout_candidate = min(
                            eligible,
                            key=lambda candidate: (
                                candidate[0][1:], candidate[0][0],
                                -candidate[3]),
                        )
                    if timeout_candidate is None:
                        timeout_candidate = best_candidate
                    if timeout_candidate is None:
                        raise
                    (_rank, transforms, matches, fill_ratio,
                     candidate_path) = timeout_candidate
                    topology_path = "%s_timeout_best" % candidate_path
                    timed_out = True
                    break
                except RuntimeError as error:
                    last_error = error
                    _solve_progress("task2 path failed path=%s error=%s" % (
                        path_name, error))
                    if first_error is None:
                        first_error = error
            if transforms is not None:
                break
        if transforms is None:
            if (not return_best_candidate_on_failure
                    or best_effort_candidate is None):
                raise last_error or first_error
            (_rank, transforms, matches, fill_ratio,
             candidate_path, best_effort_target_size_mm,
             best_effort_quality_valid) = best_effort_candidate
            topology_path = "%s_best_effort" % (candidate_path or "custom")
            best_effort = True
            fallback_reason = str(last_error or first_error or
                                  "strict quality gate rejected candidate")
            _solve_progress(
                "task2 best-effort fill=%.1f%% path=%s reason=%s" % (
                    fill_ratio * 100.0, candidate_path, fallback_reason))
        actions = actions_from_transforms(
            pieces, transforms, mm_per_pixel=config.A4_MM_PER_PIXEL,
            confidence=fill_ratio,
        )
        diagnostics = {
            "pieces": pieces,
            "piece_binary": binary,
            "timings": timings,
            "transforms": transforms,
            "matches": matches,
            "fill_ratio": fill_ratio,
            "piece_scales": piece_scales,
        }
        if topology_path is not None:
            diagnostics["topology_path"] = topology_path
        if timed_out:
            diagnostics["timed_out"] = True
        if best_effort:
            diagnostics["best_effort"] = True
            diagnostics["fallback_reason"] = fallback_reason
            diagnostics["best_effort_target_size_mm"] = (
                best_effort_target_size_mm)
            diagnostics["best_effort_quality_valid"] = bool(
                best_effort_quality_valid)
        if any(abs(value - 1.0) > 1e-4 for value in piece_scales):
            _solve_progress("task2 dynamic scales=%s" % ",".join(
                "%.3f" % value for value in piece_scales))
        _solve_progress("task2 complete actions=%d fill=%.1f%%" % (
            len(actions), fill_ratio * 100.0))
        return actions, diagnostics
