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
MAX_PIECES = 6
MATCH_SCORE_LIMIT = 0.45
A4_REFRESH_MS = 1000

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
    """Cheap validation used before actions; avoids using a stale homography."""
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


def detect_pieces(warped_rgb):
    """Detect white pieces on the rectified black A4 surface."""
    timings = {}
    start = ticks_ms()
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    binary[:5, :] = 0
    binary[-5:, :] = 0
    binary[:, :5] = 0
    binary[:, -5:] = 0
    if MORPH_KERNEL > 1:
        kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    timings["binary_morph"] = elapsed_ms(start)

    start = ticks_ms()
    contours = list(find_contours(binary))
    timings["contours"] = elapsed_ms(start)

    start = ticks_ms()
    paper_area = WARP_W * WARP_H
    pieces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not paper_area * PIECE_MIN_AREA_RATIO <= area <= paper_area * PIECE_MAX_AREA_RATIO:
            continue
        polygon = approximate_piece(contour)
        if polygon is None:
            continue
        pieces.append(ensure_clockwise(polygon))
    pieces.sort(key=lambda item: cv2.boundingRect(np.round(item).astype(np.int32))[1] * WARP_W + cv2.boundingRect(np.round(item).astype(np.int32))[0])
    timings["approx_poly"] = elapsed_ms(start)
    return pieces[:MAX_PIECES], binary, timings


def analyze_pieces(rgb, homography):
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
    timings["total"] = elapsed_ms(total_start)
    return {"warped": warped, "binary": binary, "pieces": pieces}, "OK", timings


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
        "version": 1,
        "saved_ms": ticks_ms(),
        "warp_size": [WARP_W, WARP_H],
        "mm_per_pixel": MM_PER_PIXEL,
        "piece_count": len(records),
        "total_area_px2": round(sum(areas), 3),
        "pieces": records,
    }


def save_layout(layout, path=STORAGE_PATH):
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
    if layout.get("version") != 1 or not isinstance(layout.get("pieces"), list):
        return None
    return layout


def delete_layout(path=STORAGE_PATH):
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def shape_score(detected, saved_record):
    saved_sides = saved_record.get("sides", [])
    detected_sides = side_signature(detected)
    side_count = min(len(saved_sides), len(detected_sides))
    if side_count:
        side_error = sum(abs(saved_sides[index] - detected_sides[index]) for index in range(side_count)) / side_count
    else:
        side_error = 1.0
    vertex_error = abs(len(detected) - int(saved_record.get("vertices", 0))) * 0.08
    saved_area = max(float(saved_record.get("area_px2", 1.0)), 1.0)
    detected_area = max(float(abs(cv2.contourArea(detected))), 1.0)
    area_error = abs(math.log(detected_area / saved_area))
    return 0.75 * area_error + 1.80 * side_error + vertex_error


def match_pieces_to_layout(pieces, layout):
    saved = layout.get("pieces", [])
    if len(pieces) != len(saved):
        return None, "COUNT %d/%d" % (len(pieces), len(saved)), None
    count = len(saved)
    if count == 0:
        return [], "EMPTY", 0.0
    scores = [[shape_score(piece, record) for record in saved] for piece in pieces]
    best_assignment = None
    best_score = float("inf")
    for assignment in permutations(range(count)):
        total = sum(scores[piece_index][slot_index] for piece_index, slot_index in enumerate(assignment))
        if total < best_score:
            best_score = total
            best_assignment = assignment
    average = best_score / count
    if average > MATCH_SCORE_LIMIT:
        return best_assignment, "WEAK MATCH %.2f" % average, average
    return best_assignment, "MATCH OK", average


def permutations(values):
    values = tuple(values)
    if len(values) <= 1:
        yield tuple(values)
    else:
        for index, value in enumerate(values):
            for suffix in permutations(values[:index] + values[index + 1:]):
                yield (value,) + suffix


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
    labels = (("SAVE", (35, 80, 120)), ("DELETE", (120, 65, 45)), ("CHECK", (50, 105, 45)))
    for index, (label, color) in enumerate(labels):
        x0 = index * button_w
        x1 = CAM_W - 1 if index == 2 else (index + 1) * button_w - 1
        cv2.rectangle(rgb, (x0, 0), (x1, 36), color, -1)
        cv2.rectangle(rgb, (x0, 0), (x1, 36), (220, 220, 220), 1)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        text_x = x0 + max(4, (x1 - x0 + 1 - text_size[0]) // 2)
        cv2.putText(rgb, label, (text_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


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


def build_result_view(raw_rgb, quad, analysis, layout, status, timings, fps, arranged=None, match_score=None):
    canvas = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    raw_copy = raw_rgb.copy()
    draw_a4_border(raw_copy, quad)
    raw_small = cv2.resize(raw_copy, (320, 240), interpolation=cv2.INTER_AREA)
    canvas[38:278, :320] = raw_small

    if analysis is not None:
        warped_overlay = draw_piece_overlay(analysis["warped"], analysis["pieces"])
    else:
        warped_overlay = np.zeros((WARP_H, WARP_W, 3), np.uint8)
    warp_small, _ = fit_image(warped_overlay, 300, 400)
    wx = 330 + (305 - warp_small.shape[1]) // 2
    canvas[42:42 + warp_small.shape[0], wx:wx + warp_small.shape[1]] = warp_small

    preview = arranged if arranged is not None else ([] if layout is None else layout_polygons(layout))
    draw_saved_arrangement(canvas, preview, (4, 323, 318, 151), "SAVED RECT")

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
                    action = "save"
                elif self.last_x < third * 2:
                    action = "delete"
                else:
                    action = "check"
        return action


def ensure_a4(rgb, quad, homography):
    timings = {}
    if homography is not None and cached_a4_is_valid(warp_a4(rgb, homography)):
        return quad, homography, timings
    start = ticks_ms()
    quad, homography = detect_a4(rgb)
    timings["find_a4"] = elapsed_ms(start)
    return quad, homography, timings


def run_device():
    if camera is None:
        raise RuntimeError("MaixPy is unavailable. Use --self-test on PC or run main.py on MaixCAM Pro.")

    cam = camera.Camera(CAM_W, CAM_H, image.Format.FMT_RGB888, buff_num=2)
    screen = display.Display()
    touch = touchscreen.TouchScreen()
    cam.skip_frames(SKIP_FRAMES)  # Intentionally called exactly once.

    layout = load_layout()
    quad = homography = None
    timings = {}
    status = "loaded %d pieces" % layout.get("piece_count", 0) if layout is not None else "tap SAVE"
    cached_view = None
    buttons = TouchButton()
    fps_meter = maix_time.FPS(10)
    fps = 0.0
    last_a4_refresh = 0

    while not app.need_exit():
        frame = cam.read()
        rgb = image.image2cv(frame, ensure_bgr=False, copy=False)
        action = buttons.read(touch)

        if action in ("save", "check"):
            quad, homography, find_timings = ensure_a4(rgb, quad, homography)
            timings = dict(find_timings)
            if homography is None:
                status = "A4 not found"
                cached_view = None
            else:
                analysis, status, action_timings = analyze_pieces(rgb, homography)
                timings.update(action_timings)
                if status == "A4 cache invalid":
                    quad, homography, find_timings = ensure_a4(rgb, None, None)
                    timings.update(find_timings)
                    analysis, status, action_timings = analyze_pieces(rgb, homography) if homography is not None else (None, "A4 not found", {})
                    timings.update(action_timings)
                if analysis is not None and status == "OK":
                    if action == "save":
                        if analysis["pieces"]:
                            layout = make_layout(analysis["pieces"])
                            save_layout(layout)
                            status = "SAVED %d pieces" % layout["piece_count"]
                            cached_view = build_result_view(rgb, quad, analysis, layout, status, timings, fps)
                            print("[PUZZLE] saved_layout pieces=%d path=%s" % (layout["piece_count"], STORAGE_PATH))
                        else:
                            status = "NO PIECES"
                            cached_view = build_result_view(rgb, quad, analysis, layout, status, timings, fps)
                    else:
                        if layout is None:
                            status = "NO SAVED LAYOUT"
                            cached_view = build_result_view(rgb, quad, analysis, layout, status, timings, fps)
                        else:
                            assignment, match_status, score = match_pieces_to_layout(analysis["pieces"], layout)
                            arranged = build_arranged_polygons(analysis["pieces"], layout, assignment) if assignment is not None else None
                            status = match_status
                            cached_view = build_result_view(rgb, quad, analysis, layout, status, timings, fps, arranged, score)
                            print("[PUZZLE] check status=%s assignment=%s" % (match_status, assignment))
                else:
                    cached_view = build_result_view(rgb, quad, analysis, layout, status, timings, fps)
        elif action == "delete":
            removed = delete_layout()
            layout = None
            status = "DELETED" if removed else "NOT SAVED"
            cached_view = None
            print("[PUZZLE] saved_layout_deleted=%s path=%s" % (removed, STORAGE_PATH))

        now = ticks_ms()
        if cached_view is None and (homography is None or elapsed_ms(last_a4_refresh) >= A4_REFRESH_MS):
            start = ticks_ms()
            quad, homography = detect_a4(rgb)
            timings = {"find_a4": elapsed_ms(start)}
            last_a4_refresh = now
            if homography is not None and status in ("tap SAVE", "A4 not found"):
                status = "A4 ready"

        if cached_view is None:
            draw_a4_border(rgb, quad)
            draw_buttons(rgb)
            saved_text = " saved:%d" % layout.get("piece_count", 0) if layout is not None else " saved:none"
            cv2.putText(rgb, "FPS %.1f  %s%s" % (fps, status, saved_text), (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            shown = image.cv2image(rgb, bgr=False, copy=False)
        else:
            shown = image.cv2image(cached_view, bgr=False, copy=False)
        screen.show(shown)
        fps = fps_meter.fps()


def affine_polygon(polygon, angle_degrees, translation):
    angle = math.radians(angle_degrees)
    rotation = np.float32(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    return polygon.dot(rotation.T) + np.float32(translation)


def synthetic_frame(scattered=True):
    """Make a perspective A4 scene containing white pieces."""
    paper = np.zeros((WARP_H, WARP_W, 3), np.uint8)
    gap = 4.0
    target = (
        np.float32(((90, 110), (210 - gap, 110), (210 - gap, 210 - gap), (90, 210 - gap))),
        np.float32(((210 + gap, 110), (330, 110), (330, 210 - gap), (210 + gap, 210 - gap))),
        np.float32(((90, 210 + gap), (330, 210 + gap), (330, 300), (90, 300))),
    )
    if scattered:
        pieces = (
            affine_polygon(target[0], 12, (10, -20)),
            affine_polygon(target[1], -10, (-10, 120)),
            affine_polygon(target[2], 8, (-10, 80)),
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
    saved_analysis, status, _ = analyze_pieces(saved_raw, homography)
    if status != "OK" or saved_analysis is None or len(saved_analysis["pieces"]) != 3:
        raise AssertionError("synthetic saved pieces failed: %s" % status)
    layout = make_layout(saved_analysis["pieces"])
    save_layout(layout, temp_path)
    loaded = load_layout(temp_path)
    if loaded is None or loaded["piece_count"] != 3:
        raise AssertionError("saved layout did not persist")

    check_raw = synthetic_frame(scattered=True)
    quad, homography = detect_a4(check_raw)
    check_analysis, status, _ = analyze_pieces(check_raw, homography)
    if status != "OK" or check_analysis is None:
        raise AssertionError("synthetic check failed: %s" % status)
    assignment, match_status, score = match_pieces_to_layout(check_analysis["pieces"], loaded)
    if assignment is None or score is None or score > MATCH_SCORE_LIMIT:
        raise AssertionError("saved layout match failed: %s score=%r" % (match_status, score))
    arranged = build_arranged_polygons(check_analysis["pieces"], loaded, assignment)
    if len(arranged) != 3:
        raise AssertionError("arranged preview was not built")
    if not delete_layout(temp_path) or load_layout(temp_path) is not None:
        raise AssertionError("saved layout delete failed")

    preview = build_result_view(check_raw, quad, check_analysis, loaded, "MATCH OK", {}, 0.0, arranged, score)
    if not np.any(preview[323:474, 4:322]):
        raise AssertionError("result preview was not rendered")

    print("SELF_TEST_OK")
    print("pieces=%d score=%.3f assignment=%s" % (len(check_analysis["pieces"]), score, assignment))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        run_device()
