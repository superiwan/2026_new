"""Task 2(1): white-piece topology solver from 2026_vision_v1.

The edge matching and rectangle assembly below are kept aligned with
``maixcam_pro/puzzle_solver.py`` at upstream commit ``39c94d8``.  The small
adapter at the end accepts the A4 image already rectified by 2026_new and
returns the shared PieceAction contract.
"""

import itertools
import math

import cv2
import numpy as np

try:
    from . import task2_config as config
except ImportError:  # Run directly in MaixVision with this folder as project root.
    import task2_config as config


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


def candidate_matchings(pieces):
    """Shortlist upstream v2.1 full-edge and T-junction matches."""
    all_edges = {(piece_index, edge_index): edge
                 for piece_index, piece in enumerate(pieces)
                 for edge_index, edge in enumerate(edges(piece))}
    candidates = []
    for (i, ei), (j, ej) in itertools.combinations(all_edges, 2):
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
    return candidates[:config.MAX_EDGE_CANDIDATES]


def match_segments(pieces, match):
    """Return the full or partial edge segments encoded by one match."""
    _, i, edge_i, j, edge_j, ia0, ia1, ja0, ja1 = match
    a, b = edges(pieces[i])[edge_i]
    c, d = edges(pieces[j])[edge_j]
    return (a + (b - a) * ia0, a + (b - a) * ia1,
            c + (d - c) * ja0, c + (d - c) * ja1)


def matching_sets(pieces, cut_mode="auto", max_full=None, max_partial=None):
    """Enumerate connected v2.1 topology candidates without generator truth."""
    count = len(pieces)
    if count == 1:
        yield ()
        return

    candidates = candidate_matchings(pieces)
    pair_count = (count if ((cut_mode == "common" and count >= 3)
                            or (cut_mode == "concave" and count >= 2))
                  else count - 1)
    full = [match for match in candidates
            if tuple(match[5:]) == (0.0, 1.0, 0.0, 1.0)]
    partial = [match for match in candidates
               if tuple(match[5:]) != (0.0, 1.0, 0.0, 1.0)]
    if max_full is not None:
        full = full[:max_full]
    if max_partial is not None:
        partial = partial[:max_partial]
    if cut_mode == "t_junction" and count >= 3:
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
    for _ in range(15):
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
    return unpack(values)


def _assembly_quality(assembled, matches, closure_error):
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
    expected_aspect = config.TARGET_ASPECT_RATIO
    aspect = max(rect[1]) / max(1.0, min(rect[1]))
    aspect_error = abs(math.log(max(aspect, 1e-9) / expected_aspect))
    expected_width = math.sqrt(max(source_area, 1.0) * expected_aspect)
    expected_height = expected_width / expected_aspect
    expected_perimeter = 2.0 * (expected_width + expected_height)
    disconnected_area = sum(cv2.contourArea(item) for item in contours)
    disconnected_area -= contour_area
    perimeter_error = abs(cv2.arcLength(contour, True) - expected_perimeter)
    match_error = sum(match[0] for match in matches) * 5000.0
    score = (
        closure_error * 8.0
        + overlap * 12.0
        + fill_error * 8.0
        + abs(union_area - source_area) * 4.0
        + abs(rectangle_area - source_area) * 3.0
        + aspect_error * max(source_area, 1.0) * 0.85
        + disconnected_area * 20.0
        + perimeter_error * 25.0
        + match_error
    )
    return score, fill_ratio


def assemble_from_matches(pieces, matches):
    adjacency = [[] for _ in pieces]
    for match in matches:
        _, i, _ei, j, _ej = match[:5]
        adjacency[i].append((j, match, False))
        adjacency[j].append((i, match, True))

    transforms = [None] * len(pieces)
    transforms[0] = np.eye(3)
    stack = [0]
    closure_error = 0.0
    while stack:
        i = stack.pop()
        for j, match, reversed_sides in adjacency[i]:
            ia, ib, ja, jb = match_segments(pieces, match)
            if reversed_sides:
                ia, ib, ja, jb = ja, jb, ia, ib
            world_a, world_b = apply_h(
                np.array([ia, ib]), transforms[i])
            proposed = align_edge(ja, jb, world_b, world_a)
            if transforms[j] is None:
                transforms[j] = proposed
                stack.append(j)
            else:
                error = apply_h(pieces[j], proposed) - apply_h(
                    pieces[j], transforms[j])
                closure_error += np.linalg.norm(error, axis=1).mean()

    if any(transform is None for transform in transforms):
        return None
    assembled = [apply_h(piece, transform)
                 for piece, transform in zip(pieces, transforms)]
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


def _target_transform(pieces, transforms, paper):
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

    x, y, width, height = cv2.boundingRect(paper.astype(np.int32))
    target_center = np.array([
        x + width * config.TARGET_CENTER_X_RATIO,
        y + height * config.TARGET_CENTER_Y_RATIO,
    ])
    recovered_size = maximum - minimum
    margin = min(width, height) * config.TARGET_MARGIN_RATIO
    target_origin = target_center - recovered_size / 2.0
    target_origin[0] = min(max(target_origin[0], x + margin),
                           x + width - margin - recovered_size[0])
    target_origin[1] = min(max(target_origin[1], y + margin),
                           y + height - margin - recovered_size[1])
    translate = rigid(0.0, *(target_origin - minimum))
    return [translate @ normalize @ transform for transform in transforms]


def solve(pieces, paper, cut_mode="auto"):
    """Return final per-piece 3x3 transforms, selected matches and fill ratio."""
    if not 1 <= len(pieces) <= 4:
        raise ValueError("碎片数量必须为 1～4")

    transforms = (_equal_rectangle_transforms(pieces)
                  if cut_mode in ("auto", "equal_rectangles") else None)
    matches = ()
    if transforms is None:
        best = None
        search_limits = ((config.FAST_SEARCH_FULL_CANDIDATES,
                          config.FAST_SEARCH_PARTIAL_CANDIDATES),
                         (None, None)) if cut_mode == "auto" else ((None, None),)
        for max_full, max_partial in search_limits:
            for candidate_matches in matching_sets(
                    pieces, cut_mode, max_full, max_partial):
                result = assemble_from_matches(pieces, candidate_matches)
                if result is not None and (best is None or result[0] < best[0]):
                    best = (*result, candidate_matches)
            if best is not None and best[1] >= config.FAST_SEARCH_ACCEPT_FILL:
                break
        if best is None:
            raise RuntimeError("未找到满足边长配对与碎片邻接关系的拼接")
        _, _fill_ratio, transforms, matches = best
        transforms = optimize_pose_graph(pieces, matches, transforms)
    assembled = [apply_h(piece, transform)
                 for piece, transform in zip(pieces, transforms)]
    _, fill_ratio = _assembly_quality(assembled, matches, 0.0)
    if fill_ratio < config.MIN_RECTANGLE_FILL:
        raise RuntimeError("最佳拼接的矩形填充率仅 %.1f%%" % (fill_ratio * 100.0))
    final = _target_transform(pieces, transforms, paper)
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
        pieces = [np.asarray(piece, dtype=np.float64) for piece in pieces]
        if not 1 <= len(pieces) <= 4:
            raise RuntimeError("检测到 %d 块纯白碎片，需要 1-4 块" % len(pieces))
        paper = np.int32((((0, 0),), ((config.A4_WARP_WIDTH - 1, 0),),
                          ((config.A4_WARP_WIDTH - 1,
                            config.A4_WARP_HEIGHT - 1),),
                          ((0, config.A4_WARP_HEIGHT - 1),)))
        transforms, matches, fill_ratio = solve(pieces, paper)
        actions = actions_from_transforms(
            pieces, transforms, mm_per_pixel=0.5,
            confidence=fill_ratio,
        )
        return actions, {
            "pieces": pieces,
            "piece_binary": binary,
            "timings": timings,
            "transforms": transforms,
            "matches": matches,
            "fill_ratio": fill_ratio,
        }
