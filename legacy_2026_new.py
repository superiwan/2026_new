"""MaixCAM Pro saved-layout puzzle helper.

Device mode uses MaixPy for capture/display and OpenCV for geometry.  Run
``python main.py --self-test`` on a PC to exercise the camera-free pipeline.
"""

import json
import math
import os
import sys
import tempfile

import cv2
import numpy as np

try:
    from maix import app, camera, display, image, pinmap, uart, time as maix_time, touchscreen
except ImportError:  # Allows the camera-free host self-test.
    app = camera = display = image = pinmap = uart = maix_time = touchscreen = None


# Camera / rectified A4 geometry.  420 x 594 gives exactly 2 pixels/mm.
CAM_W, CAM_H = 640, 480
WARP_W, WARP_H = 420, 594
MM_PER_PIXEL = 0.5
SKIP_FRAMES = 30

# Tune these first for the actual green paper, white pieces, lens and lighting.
A4_MIN_AREA_RATIO = 0.18
A4_MAX_AREA_RATIO = 0.85
A4_RATIO_TOLERANCE = 0.35
A4_GREEN_MIN_G_MINUS_R = 12
A4_GREEN_MIN_G_MINUS_B = 6
A4_GREEN_MIN_SATURATION = 25
A4_GREEN_MIN_FILL_RATIO = 0.50
A4_GREEN_CACHE_MIN_FILL_RATIO = 0.45
A4_GREEN_CACHE_MIN_BORDER_RATIO = 0.65
WHITE_THRESHOLD = 165
WHITE_MAX_SATURATION = 55
WHITE_MIN_VALUE = 100
WHITE_GREEN_MASK_MIN_FILL_RATIO = 0.65
MORPH_KERNEL = 3
PIECE_MIN_AREA_RATIO = 0.001
PIECE_MAX_AREA_RATIO = 0.25
PIECE_MIN_THICKNESS_PX = 4.0
PIECE_MAX_ASPECT_RATIO = 12.0
PIECE_BORDER_REJECT_PX = 6
POLY_EPSILON_RATIOS = (0.012, 0.018, 0.025, 0.035, 0.05, 0.07)
MAX_PIECES = 4
MATCH_SCORE_LIMIT = 0.45
FAST_POLY_EPSILON_RATIO = 0.025
DETECTION_INTERVAL_MS = 100
A4_DETECT_SCALE = 0.5
UART_DEVICE = "/dev/ttyS0"
UART_BAUDRATE = 115200
UART_SEND_INTERVAL_MS = 100
A4_SPLIT_Y = WARP_H // 2
REGION_MARGIN = 6
UPPER_REGION = (REGION_MARGIN, A4_SPLIT_Y - REGION_MARGIN)
LOWER_REGION = (A4_SPLIT_Y + REGION_MARGIN, WARP_H - REGION_MARGIN)

_A4_CLOSE_KERNEL = np.ones((7, 7), np.uint8)
_PIECE_MORPH_KERNEL = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)

STORAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_layout.json")

COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_YELLOW = (255, 220, 0)
COLOR_RED = (255, 60, 60)
PIECE_COLORS = ((255, 90, 90), (90, 255, 120), (80, 170, 255), (255, 180, 60), (200, 120, 255), (255, 240, 80))


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


def is_piece_contour(contour, paper_area, image_shape,
                     y_offset=0, full_height=None):
    """Reject tiny blobs and filament-like contours before polygon fitting."""
    area = cv2.contourArea(contour)
    if not (paper_area * PIECE_MIN_AREA_RATIO
            <= area <= paper_area * PIECE_MAX_AREA_RATIO):
        return False
    image_height, image_width = image_shape[:2]
    full_height = image_height if full_height is None else full_height
    points = contour.reshape(-1, 2)
    if (np.any(points[:, 0] < PIECE_BORDER_REJECT_PX)
            or np.any(points[:, 0] >= image_width - PIECE_BORDER_REJECT_PX)
            or np.any(points[:, 1] + y_offset < PIECE_BORDER_REJECT_PX)
            or np.any(points[:, 1] + y_offset
                      >= full_height - PIECE_BORDER_REJECT_PX)):
        return False
    rect_width, rect_height = cv2.minAreaRect(contour)[1]
    short_side, long_side = sorted(
        (float(rect_width), float(rect_height)))
    if short_side < PIECE_MIN_THICKNESS_PX:
        return False
    return long_side / max(short_side, 1e-6) <= PIECE_MAX_ASPECT_RATIO


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


def green_paper_mask(rgb, apply_morphology=True):
    """Return pixels whose green channel dominates under varied exposure."""
    red, green, blue = cv2.split(rgb)
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    red = red.astype(np.int16)
    green = green.astype(np.int16)
    blue = blue.astype(np.int16)
    mask = np.where(
        ((green - red >= A4_GREEN_MIN_G_MINUS_R)
         & (green - blue >= A4_GREEN_MIN_G_MINUS_B)
         & (saturation >= A4_GREEN_MIN_SATURATION)),
        255, 0,
    ).astype(np.uint8)
    if apply_morphology:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, _A4_CLOSE_KERNEL, iterations=2)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, _PIECE_MORPH_KERNEL, iterations=1)
    return mask


def detect_a4(rgb):
    """Locate the largest green convex A4-like quadrilateral."""
    scale = A4_DETECT_SCALE
    working = (cv2.resize(rgb, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
               if scale < 1.0 else rgb)
    green_mask = green_paper_mask(working)

    frame_area = working.shape[0] * working.shape[1]
    target_ratio = 297.0 / 210.0
    best = None
    best_score = -1.0
    for contour in find_contours(green_mask):
        # A broad low-saturation shadow can cut into one paper edge. Try the
        # measured contour first, then reconstruct only that missing boundary
        # with its convex hull while retaining all existing A4 quality gates.
        contour_options = (contour, cv2.convexHull(contour))
        for option_index, candidate in enumerate(contour_options):
            area = cv2.contourArea(candidate)
            if not (frame_area * A4_MIN_AREA_RATIO
                    <= area <= frame_area * A4_MAX_AREA_RATIO):
                continue
            perimeter = cv2.arcLength(candidate, True)
            epsilons = ((0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
                        if option_index == 0 else
                        (0.005, 0.008, 0.01, 0.015, 0.02, 0.025,
                         0.03, 0.04, 0.05))
            accepted = False
            for epsilon in epsilons:
                approx = cv2.approxPolyDP(
                    candidate, epsilon * perimeter, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                quad = order_quad(approx[:, 0, :])
                width = (np.linalg.norm(quad[1] - quad[0])
                         + np.linalg.norm(quad[2] - quad[3])) * 0.5
                height = (np.linalg.norm(quad[3] - quad[0])
                          + np.linalg.norm(quad[2] - quad[1])) * 0.5
                if min(width, height) < 1.0:
                    continue
                ratio_error = abs(
                    max(width, height) / min(width, height) - target_ratio
                ) / target_ratio
                if ratio_error > A4_RATIO_TOLERANCE:
                    continue

                center = np.mean(approx[:, 0, :], axis=0)
                inset = np.round(
                    center + (approx[:, 0, :] - center) * 0.94,
                ).astype(np.int32)
                interior = np.zeros(green_mask.shape, np.uint8)
                cv2.fillConvexPoly(interior, inset, 255)
                green_fill = float(
                    np.mean(green_mask[interior != 0] != 0))
                if green_fill < A4_GREEN_MIN_FILL_RATIO:
                    continue
                score = (area / frame_area - ratio_error
                         + green_fill * 0.25 - option_index * 0.02)
                if score > best_score:
                    best, best_score = quad, score
                accepted = True
                break
            if accepted:
                break

    if best is None:
        return None, None
    if scale < 1.0:
        best = best / scale
    destination = np.float32(((0, 0), (WARP_W - 1, 0), (WARP_W - 1, WARP_H - 1), (0, WARP_H - 1)))
    return best, cv2.getPerspectiveTransform(best, destination)


def warp_a4(rgb, homography):
    return cv2.warpPerspective(rgb, homography, (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)


def cached_a4_is_valid(warped_rgb):
    """Cheap validation used before actions; avoids using a stale homography."""
    green_mask = green_paper_mask(warped_rgb, apply_morphology=False)
    border = np.concatenate((
        green_mask[:8, :].ravel(), green_mask[-8:, :].ravel(),
        green_mask[:, :8].ravel(), green_mask[:, -8:].ravel(),
    ))
    return (float(np.mean(green_mask != 0))
            >= A4_GREEN_CACHE_MIN_FILL_RATIO
            and float(np.mean(border != 0))
            >= A4_GREEN_CACHE_MIN_BORDER_RATIO)


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
        if not 3 <= len(approximation) <= 8:
            continue
        polygon = approximation[:, 0, :].astype(np.float32)
        key = tuple(map(tuple, np.round(polygon).astype(int)))
        if key in seen or abs(cv2.contourArea(polygon)) < 1.0:
            continue
        seen.add(key)
        refined = refine_polygon_vertices(contour, polygon)
        iou, area_error = contour_polygon_quality(contour, refined)
        score = iou - 0.7 * area_error - 0.02 * abs(len(refined) - 4)
        if score > best_score:
            best_score = score
            best = refined
    return best


def signed_area(polygon):
    return float(cv2.contourArea(np.asarray(polygon, dtype=np.float32), oriented=True))


def ensure_clockwise(polygon):
    polygon = np.asarray(polygon, dtype=np.float32)
    return polygon[::-1].copy() if signed_area(polygon) > 0 else polygon.copy()


def polygon_centroid(polygon):
    polygon = np.asarray(polygon, dtype=np.float32)
    moments = cv2.moments(polygon)
    if abs(moments["m00"]) < 1e-6:
        return np.mean(polygon, axis=0)
    return np.float32((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))


def normalize_angle(angle):
    """Normalize an image-coordinate rotation to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def polygon_rotation_to_target(current, target):
    """Estimate the clockwise image rotation from current shape to target shape."""
    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if len(current) == len(target) and len(current) >= 3:
        current_edges = np.roll(current, -1, axis=0) - current
        target_edges = np.roll(target, -1, axis=0) - target
        current_lengths = np.linalg.norm(current_edges, axis=1)
        target_lengths = np.linalg.norm(target_edges, axis=1)
        current_lengths /= max(float(np.sum(current_lengths)), 1e-6)
        target_lengths /= max(float(np.sum(target_lengths)), 1e-6)
        best = None
        for shift in range(len(current)):
            angles = []
            for index, edge in enumerate(current_edges):
                other = target_edges[(index + shift) % len(target_edges)]
                edge_length = float(np.linalg.norm(edge) * np.linalg.norm(other))
                if edge_length < 1e-6:
                    continue
                cross = float(edge[0] * other[1] - edge[1] * other[0])
                dot = float(np.dot(edge, other))
                angle = math.degrees(math.atan2(cross, dot))
                angles.append(angle)
            if not angles:
                continue
            radians = np.radians(angles)
            mean = math.degrees(math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))))
            errors = [normalize_angle(value - mean) ** 2 for value in angles]
            length_errors = [
                (current_lengths[index] - target_lengths[(index + shift) % len(target_lengths)]) ** 2
                for index in range(len(current_lengths))
            ]
            score = float(np.mean(errors)) + 10000.0 * float(np.mean(length_errors))
            candidate = score, abs(normalize_angle(mean)), mean
            if (
                best is None
                or score < best[0] - 0.5
                or (score <= best[0] + 0.5 and candidate[1] < best[1])
            ):
                best = candidate
        if best is not None:
            return normalize_angle(best[2])

    # Fallback for differing polygon approximations: use the long side of the
    # minimum-area rectangle, which is stable for the contest's sheet pieces.
    def long_side_angle(polygon):
        box = cv2.boxPoints(cv2.minAreaRect(polygon.astype(np.float32)))
        sides = np.roll(box, -1, axis=0) - box
        side = sides[int(np.argmax(np.linalg.norm(sides, axis=1)))]
        return math.degrees(math.atan2(float(side[1]), float(side[0])))

    return normalize_angle(long_side_angle(target) - long_side_angle(current))


def crc8_ascii(payload):
    value = 0
    for byte in payload.encode("ascii"):
        value ^= byte
        for _ in range(8):
            value = ((value << 1) ^ 0x07) & 0xFF if value & 0x80 else (value << 1) & 0xFF
    return value


class PoseSender:
    """UART0 sender for one matched piece pose per line."""
    def __init__(self):
        self.serial = None
        self.last_send_ms = 0
        if uart is None or pinmap is None:
            return
        pinmap.set_pin_function("A17", "UART0_RX")
        pinmap.set_pin_function("A16", "UART0_TX")
        self.serial = uart.UART(UART_DEVICE, UART_BAUDRATE)

    def send(self, pieces, layout, assignment):
        if self.serial is None or assignment is None or len(pieces) != len(assignment):
            return 0
        now = ticks_ms()
        if now - self.last_send_ms < UART_SEND_INTERVAL_MS:
            return 0
        self.last_send_ms = now
        saved = layout.get("pieces", [])
        sent = 0
        for piece, slot in zip(pieces, assignment):
            if not 0 <= slot < len(saved):
                continue
            current_center = polygon_centroid(piece) * MM_PER_PIXEL
            target_polygon = np.asarray(saved[slot]["polygon"], dtype=np.float32)
            target_center = polygon_centroid(target_polygon) * MM_PER_PIXEL
            angle = polygon_rotation_to_target(piece, target_polygon)
            payload = "P,%d,%.1f,%.1f,%.1f,%.1f,%.1f" % (
                slot, target_center[0], target_center[1],
                current_center[0], current_center[1], angle,
            )
            frame = "$%s*%02X\r\n" % (payload, crc8_ascii(payload))
            self.serial.write_str(frame)
            sent += 1
        return sent


def detect_pieces(warped_rgb, region=None):
    """Detect white pieces inside an optional vertical region of the A4."""
    timings = {}
    start = ticks_ms()
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    green_mask = green_paper_mask(warped_rgb, apply_morphology=False)
    green_fill = float(np.mean(green_mask != 0))
    if green_fill >= WHITE_GREEN_MASK_MIN_FILL_RATIO:
        # White pieces can be darker than the green paper under auto exposure,
        # but they do not retain its green-channel dominance.
        binary = np.where(green_mask == 0, 255, 0).astype(np.uint8)
    else:
        # Large illumination shadows weaken the green mask. Low saturation is
        # the more stable white-piece cue in those frames.
        hsv = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        binary = np.where(
            ((saturation <= WHITE_MAX_SATURATION)
             & (value >= WHITE_MIN_VALUE)),
            255, 0,
        ).astype(np.uint8)
        if not np.any(binary):
            _, binary = cv2.threshold(
                gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    binary[:5, :] = 0
    binary[-5:, :] = 0
    binary[:, :5] = 0
    binary[:, -5:] = 0
    if region is not None:
        top, bottom = region
        binary[:top, :] = 0
        binary[bottom:, :] = 0
    if MORPH_KERNEL > 1:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, _PIECE_MORPH_KERNEL, iterations=1)
    timings["binary_morph"] = elapsed_ms(start)

    start = ticks_ms()
    contours = list(find_contours(binary))
    timings["contours"] = elapsed_ms(start)

    start = ticks_ms()
    paper_area = WARP_W * WARP_H
    pieces = []
    for contour in contours:
        if not is_piece_contour(contour, paper_area, binary.shape):
            continue
        polygon = approximate_piece(contour)
        if polygon is None:
            continue
        pieces.append(ensure_clockwise(polygon))
    pieces.sort(key=lambda item: cv2.boundingRect(np.round(item).astype(np.int32))[1] * WARP_W + cv2.boundingRect(np.round(item).astype(np.int32))[0])
    timings["approx_poly"] = elapsed_ms(start)
    return pieces[:MAX_PIECES], binary, timings


def analyze_pieces(rgb, homography, region=None):
    timings = {}
    total_start = ticks_ms()
    start = ticks_ms()
    warped = warp_a4(rgb, homography)
    timings["warp"] = elapsed_ms(start)
    if not cached_a4_is_valid(warped):
        timings["total"] = elapsed_ms(total_start)
        return None, "A4 cache invalid", timings
    pieces, binary, piece_timings = detect_pieces(warped, region)
    timings.update(piece_timings)
    timings["total"] = elapsed_ms(total_start)
    return {"warped": warped, "binary": binary, "pieces": pieces}, "OK", timings


def build_region_remap(homography, region):
    """Precompute a fixed-point perspective map for one horizontal A4 region."""
    top, bottom = region
    height = bottom - top
    shift = np.float64(((1, 0, 0), (0, 1, -top), (0, 0, 1)))
    inverse = np.linalg.inv(shift.dot(homography))
    grid_x, grid_y = np.meshgrid(
        np.arange(WARP_W, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    denominator = inverse[2, 0] * grid_x + inverse[2, 1] * grid_y + inverse[2, 2]
    map_x = (
        inverse[0, 0] * grid_x + inverse[0, 1] * grid_y + inverse[0, 2]
    ) / denominator
    map_y = (
        inverse[1, 0] * grid_x + inverse[1, 1] * grid_y + inverse[1, 2]
    ) / denominator
    return cv2.convertMaps(
        map_x.astype(np.float32), map_y.astype(np.float32), cv2.CV_16SC2,
    )


def detect_pieces_fast(rectified_gray, y_offset):
    """Detect clean white pieces with one polygon approximation per contour."""
    timings = {}
    start = ticks_ms()
    _, binary = cv2.threshold(rectified_gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    binary[:5, :] = 0
    binary[-5:, :] = 0
    binary[:, :5] = 0
    binary[:, -5:] = 0
    if MORPH_KERNEL > 1:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, _PIECE_MORPH_KERNEL, iterations=1)
    timings["binary_morph"] = elapsed_ms(start)

    start = ticks_ms()
    contours = find_contours(binary)
    timings["contours"] = elapsed_ms(start)

    start = ticks_ms()
    paper_area = WARP_W * WARP_H
    ranked = []
    for contour in contours:
        if not is_piece_contour(
                contour, paper_area, binary.shape,
                y_offset=y_offset, full_height=WARP_H):
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(
            contour, FAST_POLY_EPSILON_RATIO * perimeter, True,
        )
        if not 3 <= len(approximation) <= 8:
            continue
        polygon = approximation[:, 0, :].astype(np.float32)
        polygon[:, 1] += y_offset
        polygon = ensure_clockwise(polygon)
        x, y, _, _ = cv2.boundingRect(approximation)
        ranked.append((y * WARP_W + x, polygon))
    ranked.sort(key=lambda item: item[0])
    timings["approx_poly"] = elapsed_ms(start)
    return [polygon for _, polygon in ranked[:MAX_PIECES]], binary, timings


def analyze_region_fast(rgb, remap, region):
    """Run the cached half-A4 grayscale pipeline without finding A4 again."""
    timings = {}
    total_start = ticks_ms()
    start = ticks_ms()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    timings["grayscale"] = elapsed_ms(start)

    start = ticks_ms()
    map1, map2 = remap
    rectified_gray = cv2.remap(
        gray, map1, map2, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    timings["remap"] = elapsed_ms(start)
    pieces, binary, piece_timings = detect_pieces_fast(rectified_gray, region[0])
    timings.update(piece_timings)
    timings["total"] = elapsed_ms(total_start)
    return {"binary": binary, "pieces": pieces}, timings


def side_signature(polygon):
    lengths = [float(np.linalg.norm(end - start)) for start, end in zip(polygon, np.roll(polygon, -1, axis=0))]
    total = sum(lengths)
    if total <= 1e-6:
        return []
    return sorted(length / total for length in lengths)


def piece_record(polygon, slot_index):
    polygon = ensure_clockwise(polygon)
    center = polygon_centroid(polygon)
    return {
        "slot": int(slot_index),
        "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in polygon],
        "area_px2": round(float(abs(cv2.contourArea(polygon))), 3),
        "centroid": [round(float(center[0]), 3), round(float(center[1]), 3)],
        "sides": [round(value, 6) for value in side_signature(polygon)],
        "vertices": int(len(polygon)),
    }


def make_layout(pieces):
    records = [piece_record(piece, index) for index, piece in enumerate(pieces)]
    areas = [record["area_px2"] for record in records]
    return {
        "version": 2,
        "saved_ms": ticks_ms(),
        "warp_size": [WARP_W, WARP_H],
        "saved_region": [LOWER_REGION[0], LOWER_REGION[1]],
        "mm_per_pixel": MM_PER_PIXEL,
        "piece_count": len(records),
        "total_area_px2": round(sum(areas), 3),
        "pieces": records,
    }


def save_layout(layout, path=STORAGE_PATH):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(layout, handle, separators=(",", ":"))
    try:
        os.replace(temp_path, path)
    except AttributeError:
        if os.path.exists(path):
            os.remove(path)
        os.rename(temp_path, path)


def load_layout(path=STORAGE_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        layout = json.load(handle)
    if layout.get("version") != 2 or not isinstance(layout.get("pieces"), list):
        return None
    return layout


def delete_layout(path=STORAGE_PATH):
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def piece_features(polygon):
    return {
        "sides": side_signature(polygon),
        "vertices": len(polygon),
        "area_px2": max(float(abs(cv2.contourArea(polygon))), 1.0),
    }


def shape_score(detected_features, saved_record):
    saved_sides = saved_record.get("sides", [])
    detected_sides = detected_features["sides"]
    side_count = min(len(saved_sides), len(detected_sides))
    if side_count:
        side_error = sum(abs(saved_sides[index] - detected_sides[index]) for index in range(side_count)) / side_count
    else:
        side_error = 1.0
    vertex_error = abs(detected_features["vertices"] - int(saved_record.get("vertices", 0))) * 0.08
    saved_area = max(float(saved_record.get("area_px2", 1.0)), 1.0)
    detected_area = detected_features["area_px2"]
    area_error = abs(math.log(detected_area / saved_area))
    return 0.75 * area_error + 1.80 * side_error + vertex_error


def match_pieces_to_layout(pieces, layout):
    saved = layout.get("pieces", [])
    if len(pieces) != len(saved):
        return None, "COUNT %d/%d" % (len(pieces), len(saved)), None
    count = len(saved)
    if count == 0:
        return [], "EMPTY", 0.0
    features = [piece_features(piece) for piece in pieces]
    scores = [[shape_score(feature, record) for record in saved] for feature in features]

    states = {0: (0.0, ())}
    for piece_index in range(count):
        next_states = {}
        for mask, (cost, assignment) in states.items():
            for slot_index in range(count):
                bit = 1 << slot_index
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_cost = cost + scores[piece_index][slot_index]
                previous = next_states.get(next_mask)
                if previous is None or next_cost < previous[0]:
                    next_states[next_mask] = (next_cost, assignment + (slot_index,))
        states = next_states
    best_score, best_assignment = states[(1 << count) - 1]
    average = best_score / count
    if average > MATCH_SCORE_LIMIT:
        return best_assignment, "WEAK MATCH %.2f" % average, average
    return best_assignment, "MATCH OK", average


def layout_polygons(layout):
    return [np.asarray(record["polygon"], dtype=np.float32) for record in layout.get("pieces", [])]


def build_arranged_polygons(pieces, layout, assignment):
    saved_polygons = layout_polygons(layout)
    if assignment is None:
        return saved_polygons
    arranged = [None] * len(saved_polygons)
    for detected_index, slot_index in enumerate(assignment):
        arranged[slot_index] = saved_polygons[slot_index]
    return arranged


def draw_a4_border(rgb, quad):
    if quad is not None:
        cv2.polylines(rgb, [np.round(quad).astype(np.int32)], True, COLOR_GREEN, 3, cv2.LINE_AA)
        for index, point in enumerate(quad):
            p = tuple(np.round(point).astype(int))
            cv2.circle(rgb, p, 5, COLOR_YELLOW, -1, cv2.LINE_AA)
            cv2.putText(rgb, str(index), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1, cv2.LINE_AA)


def draw_buttons(rgb):
    button_w = CAM_W // 3
    labels = (("FIND A4", (35, 80, 120)), ("SAVE", (50, 105, 45)), ("DELETE", (120, 65, 45)))
    for index, (label, color) in enumerate(labels):
        x0 = index * button_w
        x1 = CAM_W - 1 if index == 2 else (index + 1) * button_w - 1
        cv2.rectangle(rgb, (x0, 0), (x1, 36), color, -1)
        cv2.rectangle(rgb, (x0, 0), (x1, 36), (220, 220, 220), 1, cv2.LINE_8)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        text_x = x0 + max(4, (x1 - x0 + 1 - text_size[0]) // 2)
        cv2.putText(rgb, label, (text_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_8)


def project_to_camera(polygon, inverse_homography):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return np.round(
        cv2.perspectiveTransform(points, inverse_homography).reshape(-1, 2)
    ).astype(np.int32)


def prepare_camera_geometry(homography, layout):
    inverse = np.linalg.inv(homography)
    split = project_to_camera(
        np.float32(((0, A4_SPLIT_Y), (WARP_W - 1, A4_SPLIT_Y))),
        inverse,
    )
    saved = []
    if layout is not None:
        saved = [project_to_camera(polygon, inverse) for polygon in layout_polygons(layout)]
    return inverse, split, saved


def build_live_camera_view(
    rgb, quad, split_camera, saved_camera, live_camera,
    piece_slots, status, timings, fps,
):
    canvas = rgb.copy()
    if quad is not None:
        cv2.polylines(
            canvas, [np.round(quad).astype(np.int32)], True,
            COLOR_GREEN, 2, cv2.LINE_8,
        )
    draw_buttons(canvas)
    if split_camera is not None:
        cv2.line(
            canvas, tuple(split_camera[0]), tuple(split_camera[1]),
            COLOR_YELLOW, 2, cv2.LINE_8,
        )

    for slot_index, polygon in enumerate(saved_camera):
        color = PIECE_COLORS[slot_index % len(PIECE_COLORS)]
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_8)

    for piece_index, polygon in enumerate(live_camera):
        slot_index = piece_slots[piece_index] if piece_slots is not None else piece_index
        color = PIECE_COLORS[slot_index % len(PIECE_COLORS)]
        cv2.polylines(canvas, [polygon], True, color, 3, cv2.LINE_8)

    status_text = "FPS %.1f  %s  d:%dms" % (
        fps, status, timings.get("total", 0),
    )
    cv2.rectangle(canvas, (0, 37), (CAM_W - 1, 63), (0, 0, 0), -1)
    cv2.putText(
        canvas, status_text, (8, 58),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_8,
    )
    return canvas


def fit_image(source, max_width, max_height):
    scale = min(max_width / source.shape[1], max_height / source.shape[0])
    size = (max(1, int(round(source.shape[1] * scale))), max(1, int(round(source.shape[0] * scale))))
    return cv2.resize(source, size, interpolation=cv2.INTER_AREA), scale


def draw_piece_overlay(warped, pieces, saved_polygons=None, piece_slots=None):
    overlay = warped.copy()
    if saved_polygons:
        for index, polygon in enumerate(saved_polygons):
            points = np.round(polygon).astype(np.int32)
            color = PIECE_COLORS[index % len(PIECE_COLORS)]
            layer = overlay.copy()
            cv2.fillPoly(layer, [points], color)
            cv2.addWeighted(layer, 0.35, overlay, 0.65, 0, overlay)
            cv2.polylines(overlay, [points], True, color, 2, cv2.LINE_AA)
    for index, polygon in enumerate(pieces):
        points = np.round(polygon).astype(np.int32)
        color_index = piece_slots[index] if piece_slots is not None else index
        color = PIECE_COLORS[color_index % len(PIECE_COLORS)]
        cv2.polylines(overlay, [points], True, color, 3, cv2.LINE_AA)
        for vertex_index, point in enumerate(points):
            p = tuple(point)
            cv2.circle(overlay, p, 5, COLOR_CYAN, -1, cv2.LINE_AA)
            cv2.putText(overlay, str(vertex_index), (p[0] + 5, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.line(overlay, (0, A4_SPLIT_Y), (WARP_W - 1, A4_SPLIT_Y), COLOR_YELLOW, 2, cv2.LINE_AA)
    cv2.putText(overlay, "LIVE PIECES", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_CYAN, 2, cv2.LINE_AA)
    cv2.putText(overlay, "SAVED RECT", (8, A4_SPLIT_Y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GREEN, 2, cv2.LINE_AA)
    return overlay


def draw_saved_arrangement(canvas, polygons, area, title):
    x, y, width, height = area
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (50, 50, 50), 1)
    cv2.putText(canvas, title, (x + 5, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    if not polygons:
        cv2.putText(canvas, "NO SAVED LAYOUT", (x + 42, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_RED, 2, cv2.LINE_AA)
        return
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
    rectangle = cv2.minAreaRect(np.vstack(polygons).astype(np.float32))
    box = cv2.boxPoints(rectangle)
    display_box = np.round((box - minimum) * scale + offset).astype(np.int32)
    cv2.polylines(canvas, [display_box], True, COLOR_GREEN, 2, cv2.LINE_AA)


def build_result_view(
    raw_rgb, quad, analysis, layout, status, timings, fps,
    arranged=None, match_score=None, piece_slots=None,
):
    canvas = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    raw_copy = raw_rgb.copy()
    draw_a4_border(raw_copy, quad)
    raw_small = cv2.resize(raw_copy, (320, 240), interpolation=cv2.INTER_AREA)
    canvas[38:278, :320] = raw_small

    if analysis is not None:
        saved_polygons = [] if layout is None else layout_polygons(layout)
        warped_overlay = draw_piece_overlay(
            analysis["warped"], analysis["pieces"], saved_polygons, piece_slots,
        )
    else:
        warped_overlay = np.zeros((WARP_H, WARP_W, 3), np.uint8)
    warp_small, _ = fit_image(warped_overlay, 300, 400)
    wx = 330 + (305 - warp_small.shape[1]) // 2
    canvas[42:42 + warp_small.shape[0], wx:wx + warp_small.shape[1]] = warp_small

    preview = arranged if arranged is not None else ([] if layout is None else layout_polygons(layout))
    draw_saved_arrangement(canvas, preview, (4, 323, 318, 151), "SAVED LOWER RECT")

    draw_buttons(canvas)
    cv2.putText(canvas, "RAW + A4", (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "RECTIFIED + PIECES", (350, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "FPS %.1f  %s" % (fps, status), (5, 291), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    if layout is not None:
        cv2.putText(canvas, "saved:%d pieces" % layout.get("piece_count", 0), (330, 454), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "saved:none", (330, 454), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_RED, 1, cv2.LINE_AA)
    if analysis is not None:
        pieces = analysis["pieces"]
        area_mm2 = sum(abs(cv2.contourArea(piece)) for piece in pieces) * MM_PER_PIXEL ** 2
        score_text = "" if match_score is None else " score %.2f" % match_score
        cv2.putText(canvas, "detected:%d area:%.0f%s" % (len(pieces), area_mm2, score_text), (330, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 230), 1, cv2.LINE_AA)
    timing_line = "a4:%d warp:%d bin:%d cont:%d poly:%d total:%d ms" % (
        timings.get("find_a4", 0), timings.get("warp", 0), timings.get("binary_morph", 0),
        timings.get("contours", 0), timings.get("approx_poly", 0), timings.get("total", 0),
    )
    cv2.putText(canvas, timing_line, (5, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)
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
                third = CAM_W // 3
                if self.last_x < third:
                    action = "find"
                elif self.last_x < third * 2:
                    action = "save"
                else:
                    action = "delete"
        return action


def run_device():
    if camera is None:
        raise RuntimeError("MaixPy is unavailable. Use --self-test on PC or run main.py on MaixCAM Pro.")

    cam = camera.Camera(CAM_W, CAM_H, image.Format.FMT_RGB888, fps=30, buff_num=1)
    screen = display.Display()
    touch = touchscreen.TouchScreen()
    pose_sender = PoseSender()
    cam.skip_frames(SKIP_FRAMES)  # Intentionally called exactly once.

    layout = load_layout()
    quad = homography = None
    region_remaps = {}
    inverse_homography = None
    split_camera = None
    saved_camera = []
    pieces = []
    live_camera = []
    piece_slots = None
    timings = {}
    status = "tap FIND A4"
    buttons = TouchButton()
    fps_meter = maix_time.FPS(10)
    fps = 0.0
    last_detection_ms = 0

    while not app.need_exit():
        frame = cam.read()
        rgb = image.image2cv(frame, ensure_bgr=False, copy=False)
        action = buttons.read(touch)

        if action == "find":
            start = ticks_ms()
            quad, homography = detect_a4(rgb)
            timings = {"find_a4": elapsed_ms(start)}
            if homography is None:
                region_remaps = {}
                inverse_homography = None
                split_camera = None
                saved_camera = []
                pieces = []
                live_camera = []
                piece_slots = None
                status = "A4 not found"
            else:
                region_remaps = {
                    "upper": build_region_remap(homography, UPPER_REGION),
                    "lower": build_region_remap(homography, LOWER_REGION),
                }
                inverse_homography, split_camera, saved_camera = prepare_camera_geometry(
                    homography, layout,
                )
                pieces = []
                live_camera = []
                piece_slots = None
                last_detection_ms = 0
                status = "A4 cached"
        elif action == "save":
            if homography is None:
                status = "tap FIND A4 first"
            else:
                analysis, timings = analyze_region_fast(
                    rgb, region_remaps["lower"], LOWER_REGION,
                )
                if not analysis["pieces"]:
                    status = "NO LOWER PIECES"
                else:
                    layout = make_layout(analysis["pieces"])
                    save_layout(layout)
                    inverse_homography, split_camera, saved_camera = prepare_camera_geometry(
                        homography, layout,
                    )
                    pieces = []
                    live_camera = []
                    piece_slots = None
                    last_detection_ms = 0
                    status = "SAVED LOWER %d" % layout["piece_count"]
                    print("[PUZZLE] saved_lower_layout pieces=%d path=%s" % (layout["piece_count"], STORAGE_PATH))
        elif action == "delete":
            removed = delete_layout()
            layout = None
            saved_camera = []
            pieces = []
            live_camera = []
            piece_slots = None
            status = "DELETED" if removed else "NOT SAVED"
            print("[PUZZLE] saved_layout_deleted=%s path=%s" % (removed, STORAGE_PATH))

        if (
            homography is not None
            and layout is not None
            and (last_detection_ms == 0 or elapsed_ms(last_detection_ms) >= DETECTION_INTERVAL_MS)
        ):
            last_detection_ms = ticks_ms()
            analysis, timings = analyze_region_fast(
                rgb, region_remaps["upper"], UPPER_REGION,
            )
            pieces = analysis["pieces"]
            assignment, match_status, _ = match_pieces_to_layout(pieces, layout)
            piece_slots = assignment
            if match_status == "MATCH OK":
                pose_sender.send(pieces, layout, assignment)
            live_camera = [
                project_to_camera(polygon, inverse_homography)
                for polygon in pieces
            ]
            if action is None:
                status = match_status
        elif homography is not None and layout is None and action is None:
            status = "place rect below; SAVE"

        live_view = build_live_camera_view(
            rgb, quad, split_camera, saved_camera, live_camera,
            piece_slots, status, timings, fps,
        )
        shown = image.cv2image(live_view, bgr=False, copy=False)
        screen.show(shown)
        fps = fps_meter.fps()


def affine_polygon(polygon, angle_degrees, translation):
    angle = math.radians(angle_degrees)
    rotation = np.float32(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    return polygon.dot(rotation.T) + np.float32(translation)


def pose_polygon(polygon, angle_degrees, center):
    centered = polygon - polygon_centroid(polygon)
    return affine_polygon(centered, angle_degrees, center)


def synthetic_frame(scattered=True):
    """Make a saved lower layout or scattered upper-piece A4 scene."""
    paper = np.full((WARP_H, WARP_W, 3), (128, 160, 118), np.uint8)
    # Keep synthetic pieces separate after perspective interpolation and the
    # production 3x3 close operation.
    gap = 8.0
    target = (
        np.float32(((90, 350), (210 - gap, 350), (210 - gap, 446), (90, 446))),
        np.float32(((210 + gap, 350), (330, 350), (330, 446), (210 + gap, 446))),
        np.float32(((90, 446 + gap), (330, 446 + gap), (330, 536), (90, 536))),
    )
    if scattered:
        pieces = (
            pose_polygon(target[0], 12, (80, 92)),
            pose_polygon(target[1], -10, (330, 102)),
            pose_polygon(target[2], 8, (205, 225)),
        )
    else:
        pieces = target
    for polygon in pieces:
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
    temp_path = os.path.join(tempfile.gettempdir(), "maixcam_saved_layout_test.json")
    if os.path.exists(temp_path):
        os.remove(temp_path)

    saved_raw = synthetic_frame(scattered=False)
    quad, homography = detect_a4(saved_raw)
    if homography is None:
        raise AssertionError("synthetic A4 was not detected")
    saved_lower_remap = build_region_remap(homography, LOWER_REGION)
    saved_upper_remap = build_region_remap(homography, UPPER_REGION)
    saved_analysis, _ = analyze_region_fast(saved_raw, saved_lower_remap, LOWER_REGION)
    if len(saved_analysis["pieces"]) != 3:
        raise AssertionError("synthetic saved pieces failed")
    saved_upper, _ = analyze_region_fast(saved_raw, saved_upper_remap, UPPER_REGION)
    if saved_upper["pieces"]:
        raise AssertionError("lower saved layout leaked into upper detection")
    layout = make_layout(saved_analysis["pieces"])
    save_layout(layout, temp_path)
    loaded = load_layout(temp_path)
    if loaded is None or loaded["piece_count"] != 3:
        raise AssertionError("saved layout did not persist")

    check_raw = synthetic_frame(scattered=True)
    quad, homography = detect_a4(check_raw)
    check_upper_remap = build_region_remap(homography, UPPER_REGION)
    check_lower_remap = build_region_remap(homography, LOWER_REGION)
    check_analysis, timings = analyze_region_fast(check_raw, check_upper_remap, UPPER_REGION)
    check_lower, _ = analyze_region_fast(check_raw, check_lower_remap, LOWER_REGION)
    if check_lower["pieces"]:
        raise AssertionError("upper live pieces leaked into lower detection")
    assignment, match_status, score = match_pieces_to_layout(check_analysis["pieces"], loaded)
    if assignment is None or score is None or score > MATCH_SCORE_LIMIT:
        raise AssertionError("saved layout match failed: %s score=%r" % (match_status, score))

    target_polygons = layout_polygons(loaded)
    for detected_index, slot_index in enumerate(assignment):
        expected_angles = (-12.0, 10.0, -8.0)
        angle = polygon_rotation_to_target(
            check_analysis["pieces"][detected_index], target_polygons[slot_index],
        )
        if abs(normalize_angle(angle - expected_angles[slot_index])) > 2.5:
            raise AssertionError("piece rotation failed: slot=%d angle=%.2f" % (slot_index, angle))

    class SerialCapture:
        def __init__(self):
            self.frames = []

        def write_str(self, frame):
            self.frames.append(frame)

    sender = PoseSender()
    sender.serial = SerialCapture()
    sender.last_send_ms = -UART_SEND_INTERVAL_MS
    if sender.send(check_analysis["pieces"], loaded, assignment) != 3:
        raise AssertionError("UART pose frame count failed")
    for frame in sender.serial.frames:
        if not frame.startswith("$P,") or not frame.endswith("\r\n"):
            raise AssertionError("UART framing failed: %r" % frame)
        payload, checksum = frame[1:-2].split("*")
        if int(checksum, 16) != crc8_ascii(payload):
            raise AssertionError("UART CRC failed: %r" % frame)
    if not delete_layout(temp_path) or load_layout(temp_path) is not None:
        raise AssertionError("saved layout delete failed")

    inverse, split_camera, saved_camera = prepare_camera_geometry(homography, loaded)
    live_camera = [
        project_to_camera(polygon, inverse)
        for polygon in check_analysis["pieces"]
    ]
    preview = build_live_camera_view(
        check_raw, quad, split_camera, saved_camera, live_camera,
        assignment, "MATCH OK", timings, 0.0,
    )
    if not np.any(preview != check_raw):
        raise AssertionError("result preview was not rendered")

    print("SELF_TEST_OK")
    print("pieces=%d score=%.3f assignment=%s" % (len(check_analysis["pieces"]), score, assignment))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        run_device()
