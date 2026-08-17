"""Interactive two-mode MaixCAM Pro puzzle application."""

import os
import sys
import time as wall_time

import cv2
import numpy as np

try:
    from maix import app, camera, display, image, time as maix_time, touchscreen
except ImportError:
    app = camera = display = image = maix_time = touchscreen = None

import legacy_2026_new as vision
from core.poker_corner_runtime import PokerCornerRuntime
from core.serial_protocol import (
    PositionSender, is_valid_pixel_point,
)
from solvers import Task2WhiteSolver, Task3PokerSolver
from solvers import poker_arc_geometry, poker_layout_selector
from solvers.task2_white import apply_h
from solvers.task3_poker import detect_poker_pieces


CAM_W, CAM_H = vision.CAM_W, vision.CAM_H
TOP_H = 54
STATUS_H = 30
BOTTOM_H = 58
BUTTON_LABELS = ("TASK2 WHITE", "TASK2 POKER")
MODE_COLORS = ((70, 190, 90), (190, 105, 225))
PIECE_COLORS = ((255, 90, 90), (90, 255, 120),
                (80, 170, 255), (255, 190, 70))


def build_device_solvers(runtime_loader=PokerCornerRuntime.load):
    """Build device solvers while keeping white mode usable without YOLO."""
    provider = None
    try:
        provider = runtime_loader()
        print("[BOOT] poker corner model ready path=%s" % (
            provider.model_path,), flush=True)
    except Exception as error:
        print("[BOOT] poker corner model unavailable: %s" % error,
              flush=True)
    return (
        Task2WhiteSolver(),
        Task3PokerSolver(
            corner_evidence_detector=provider,
            require_disambiguation=True,
            require_corner_evidence=False,
        ),
    )


def _print_solve_telemetry(mode, actions, diagnostics, solve_time_s,
                           sent_count):
    """Print compact solve results immediately for MaixVision/SSH terminals."""
    diagnostics = diagnostics or {}
    pieces = diagnostics.get("pieces")
    piece_count = len(pieces) if pieces is not None else 0
    action_count = len(actions) if actions is not None else 0
    matches = diagnostics.get("matches")
    fill_ratio = diagnostics.get("fill_ratio")
    summary = (
        "[SOLVE] mode=%s pieces=%d actions=%d sent=%d time=%s"
        % (mode, piece_count, action_count, sent_count,
           "--" if solve_time_s is None else "%.3fs" % solve_time_s)
    )
    if fill_ratio is not None:
        summary += " fill=%.1f%%" % (float(fill_ratio) * 100.0)
    if matches is not None:
        summary += " matches=%d" % len(matches)
    topology_path = diagnostics.get("topology_path")
    if topology_path:
        summary += " path=%s" % topology_path
    piece_scales = diagnostics.get("piece_scales")
    if (piece_scales is not None
            and any(abs(float(value) - 1.0) > 1e-4
                    for value in piece_scales)):
        summary += " scales=%s" % ",".join(
            "%.3f" % float(value) for value in piece_scales)
    if diagnostics.get("timed_out"):
        summary += " timed_out=1"
    print(summary, flush=True)

    assignment = diagnostics.get("assignment")
    if assignment is not None:
        print("[SOLVE] assignment=%s" % (assignment,), flush=True)
    timings = diagnostics.get("timings")
    if isinstance(timings, dict) and timings:
        print("[SOLVE] timings=%s" % (timings,), flush=True)
    for index, action in enumerate(actions if actions is not None else ()):
        print(
            "[SOLVE] action[%d] piece=%d pick=(%.1f,%.1f,%.1f) "
            "place=(%.1f,%.1f,%.1f) conf=%.3f"
            % (index, action.piece_id, action.pick_x, action.pick_y,
               action.pick_angle, action.place_x, action.place_y,
               action.place_angle, action.confidence),
            flush=True)


def screen_text(value, fallback="SOLVER ERROR"):
    """Keep all text rendered on the device display ASCII-only."""
    value = str(value)
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return fallback
    return value


class TouchRouter:
    """Route a complete press/release to a mode tab or bottom control."""

    def __init__(self):
        self.pressed_target = None

    @staticmethod
    def target(x, y):
        if 0 <= y < TOP_H:
            return "mode", min(1, max(0, int(x * 2 // CAM_W)))
        if CAM_H - BOTTOM_H <= y < CAM_H:
            return "control", min(3, max(0, int(x * 4 // CAM_W)))
        return None

    def update_values(self, x, y, pressed):
        current = self.target(x, y)
        if pressed:
            self.pressed_target = current
            return None
        selected = self.pressed_target
        self.pressed_target = None
        # MaixPy may report reset coordinates on release. Match the trusted
        # 2026_new behavior by routing from the last pressed position.
        return selected

    def update(self, touch):
        return self.update_values(*touch.read())


class MergedController:
    """User-driven A4, detection and solve interaction state."""

    CONTROL_SETS = (("FIND A4", "DETECT", "SOLVE", "RESET"),) * 2

    def __init__(self, solvers=None, sender=None):
        self.solvers = solvers or (Task2WhiteSolver(), Task3PokerSolver())
        self.sender = sender
        self.mode = 0
        self.stage = "READY"
        self.message = ("WAIT STM32 OK" if sender is not None
                        else "TAP FIND A4")
        self.quad = None
        self.homography = None
        self.inverse_homography = None
        self.split_camera = None
        self.rectified = None
        self.detected_pieces = []
        self.detected_camera = []
        self.target_camera = []
        self.actions = []
        self.sent_frames = []
        self.pending_position_pairs = []
        self.next_position_index = 0
        self.diagnostics = {}
        self.solve_time_s = None
        self._preview_cache = None

    @property
    def control_labels(self):
        return self.CONTROL_SETS[self.mode]

    def _clear_results(self):
        self.rectified = None
        self.detected_pieces = []
        self.detected_camera = []
        self.target_camera = []
        self.actions = []
        self.sent_frames = []
        self.pending_position_pairs = []
        self.next_position_index = 0
        self.diagnostics = {}
        self.solve_time_s = None
        self._preview_cache = None

    def _invalidate_preview(self):
        self._preview_cache = None

    def _refresh_projection(self):
        if self.homography is None:
            self.inverse_homography = None
            self.split_camera = None
            return
        self.inverse_homography, self.split_camera, _ = (
            vision.prepare_camera_geometry(
                self.homography, None))

    def select_mode(self, mode):
        self.mode = int(mode)
        self._clear_results()
        self._refresh_projection()
        self.stage = "A4 LOCKED" if self.homography is not None else "READY"
        self.message = ("WAIT STM32 OK" if self.sender is not None else
                        ("SELECT ACTION" if self.homography is not None
                         else "TAP FIND A4"))

    def _warp_current(self, rgb):
        if self.homography is None:
            raise RuntimeError("FIND A4 FIRST")
        # FIND A4 locks the geometry. Keep using that transform until the user
        # explicitly runs FIND A4 again; piece coverage/exposure must not
        # invalidate an otherwise fixed A4.
        rectified = vision.warp_a4(rgb, self.homography)
        self.rectified = rectified
        self._invalidate_preview()
        return rectified

    def find_a4(self, rgb):
        self.quad, self.homography = vision.detect_a4(rgb)
        self._clear_results()
        if self.homography is None:
            self._refresh_projection()
            raise RuntimeError("A4 NOT FOUND")
        self._refresh_projection()
        self.rectified = vision.warp_a4(rgb, self.homography)
        self._invalidate_preview()
        self.stage = "A4 LOCKED"
        self.message = "A4 READY - SELECT ACTION"

    def detect(self, rgb):
        if self.homography is None:
            raise RuntimeError("FIND A4 FIRST")
        rectified = self._warp_current(rgb)
        self.target_camera = []
        self.actions = []
        self.sent_frames = []
        self.pending_position_pairs = []
        self.next_position_index = 0
        self.solve_time_s = None
        if self.mode == 0:
            pieces, binary, timings = vision.detect_pieces(rectified)
            extra = {"piece_binary": binary, "timings": timings}
            self.message = "WHITE PIECES %d" % len(pieces)
        else:
            pieces, binary = detect_poker_pieces(rectified)
            extra = {"piece_binary": binary}
            detector = getattr(self.solvers[1],
                               "corner_evidence_detector", None)
            if detector is not None:
                extra["corner_marks"] = tuple(
                    poker_layout_selector.collect_corner_mark_evidence(
                        detector, rectified, pieces))
            extra["arc_reports"] = tuple(
                poker_arc_geometry.analyze_piece_arcs(binary, pieces))
            self.message = "POKER PIECES %d" % len(pieces)
        self.detected_pieces = [np.asarray(piece, dtype=np.float64)
                                for piece in pieces]
        self.detected_camera = [
            vision.project_to_camera(piece, self.inverse_homography)
            for piece in self.detected_pieces
        ]
        self.diagnostics = {"pieces": self.detected_pieces, **extra}
        self.stage = "DETECTED"
        self._invalidate_preview()

    def _apply_solution(self, rectified, reuse_detected=False):
        self.actions = []
        self.sent_frames = []
        self.pending_position_pairs = []
        self.next_position_index = 0
        solver = self.solvers[self.mode]
        solve_started = wall_time.perf_counter()
        try:
            if (reuse_detected and self.detected_pieces
                    and hasattr(solver, "solve_detected")):
                self.actions, self.diagnostics = solver.solve_detected(
                    rectified, self.detected_pieces,
                    self.diagnostics.get("piece_binary"))
            else:
                self.actions, self.diagnostics = solver.solve(rectified)
        finally:
            self.solve_time_s = wall_time.perf_counter() - solve_started
        self.detected_pieces = [
            np.asarray(piece, dtype=np.float64)
            for piece in self.diagnostics.get("pieces", ())
        ]
        self.detected_camera = [
            vision.project_to_camera(piece, self.inverse_homography)
            for piece in self.detected_pieces
        ]
        transforms = self.diagnostics.get("transforms") or ()
        self.target_camera = [
            vision.project_to_camera(apply_h(piece, transform),
                                     self.inverse_homography)
            for piece, transform in zip(self.detected_pieces, transforms)
        ]
        position_pairs = []
        if self.inverse_homography is not None:
            for action in self.actions:
                points_a4 = np.float64((
                    (action.pick_x / vision.MM_PER_PIXEL,
                     action.pick_y / vision.MM_PER_PIXEL),
                    (action.place_x / vision.MM_PER_PIXEL,
                     action.place_y / vision.MM_PER_PIXEL),
                ))
                points_camera = vision.project_to_camera(
                    points_a4, self.inverse_homography)
                clockwise_rotation = action.place_angle - action.pick_angle
                position_pairs.append((
                    points_camera[0], clockwise_rotation, points_camera[1],
                ))
        all_valid = (position_pairs
                     and len(position_pairs) == len(self.actions)
                     and all(is_valid_pixel_point(green)
                             and is_valid_pixel_point(red)
                             for green, _degree, red in position_pairs))
        if all_valid:
            self.pending_position_pairs = position_pairs
        self.stage = "SOLVED"
        if self.sender is not None:
            self.message = ("WAIT ACK %d" % len(position_pairs)
                            if all_valid else "UART NO VALID POINTS")
        else:
            self.message = "SOLUTION %d PIECES" % len(self.detected_pieces)
        _print_solve_telemetry(
            BUTTON_LABELS[self.mode], self.actions, self.diagnostics,
            self.solve_time_s, len(self.sent_frames))
        self._invalidate_preview()

    def _send_next_position(self):
        if (self.sender is None
                or self.next_position_index >= len(
                    self.pending_position_pairs)):
            return False
        pair = self.pending_position_pairs[self.next_position_index]
        frames = self.sender.send((pair,))
        if not frames:
            self.pending_position_pairs = []
            self.next_position_index = 0
            self.stage = "SOLVED"
            self.message = "UART NO VALID POINTS"
            return False
        self.sent_frames.extend(frames)
        self.next_position_index += 1
        self.stage = "SENT"
        self.message = "UART SENT %d/%d" % (
            self.next_position_index, len(self.pending_position_pairs))
        return True

    def _all_positions_sent(self):
        return (bool(self.pending_position_pairs)
                and self.next_position_index >= len(
                    self.pending_position_pairs))

    def handle_ack(self, rgb):
        """Advance exactly one coordinate pair for one STM32 ``<ok>``."""
        try:
            if self._send_next_position():
                return True
            self.find_a4(rgb)
            self.detect(rgb)
            self.solve(rgb)
            if hasattr(self.sender, "discard_input"):
                self.sender.discard_input()
            return self._send_next_position()
        except Exception as error:
            self.pending_position_pairs = []
            self.next_position_index = 0
            self.stage = "ERROR"
            self.message = screen_text(error)
            print("[MERGE] ACK ERROR mode=%s: %s" % (
                BUTTON_LABELS[self.mode], error))
            return False

    def process_uart(self, rgb):
        """Handle ACK for the touch-selected mode without changing it."""
        if self.sender is None or not hasattr(self.sender, "poll_ack"):
            return False
        try:
            if self.sender.poll_ack():
                return self.handle_ack(rgb)
            if (self._all_positions_sent()
                    and hasattr(self.sender, "send_over")):
                self.sender.send_over()
                return True
            return False
        except Exception as error:
            self.stage = "ERROR"
            self.message = "UART READ ERROR"
            print("[MERGE] UART READ ERROR: %s" % error)
            return False

    def solve(self, rgb):
        if self.stage == "DETECTED" and self.rectified is not None:
            self._apply_solution(self.rectified, reuse_detected=True)
        else:
            self._apply_solution(self._warp_current(rgb))

    def reset(self):
        self._clear_results()
        self.stage = "A4 LOCKED" if self.homography is not None else "READY"
        self.message = ("WAIT STM32 OK" if self.sender is not None else
                        ("SELECT ACTION" if self.homography is not None
                         else "TAP FIND A4"))

    def handle_control(self, index, rgb):
        try:
            if index == 0:
                self.find_a4(rgb)
            elif index == 1:
                self.detect(rgb)
            elif index == 2:
                self.solve(rgb)
            else:
                self.reset()
        except Exception as error:
            self.stage = "ERROR"
            self.message = screen_text(error)
            print("[MERGE] ERROR mode=%s: %s" % (
                BUTTON_LABELS[self.mode], error))


def _draw_centered(canvas, label, x0, x1, y, scale=0.50):
    size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = x0 + max(3, (x1 - x0 - size[0]) // 2)
    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_8)


def piece_point_label(piece_index, placed=False):
    """Return paired point labels: scattered even, placed odd."""
    return "P%d" % (int(piece_index) * 2 + int(bool(placed)))


def polygon_label_origin(polygon, label, scale=0.55, thickness=2):
    """Place an OpenCV text baseline where its box fits inside a polygon."""
    points = np.round(np.asarray(polygon)).astype(np.int32)
    x, y, width, height = cv2.boundingRect(points)
    local = points - (x, y)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [local], 255)

    size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness,
    )
    padding = max(2, thickness)
    box_width = size[0] + 2 * padding
    box_height = size[1] + baseline + 2 * padding
    kernel_width = 2 * ((box_width + 1) // 2) + 1
    kernel_height = 2 * ((box_height + 1) // 2) + 1
    available = cv2.erode(
        mask, np.ones((kernel_height, kernel_width), np.uint8),
    )
    if np.any(available):
        distance = cv2.distanceTransform(available, cv2.DIST_L2, 3)
    else:
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    _, _, _, center = cv2.minMaxLoc(distance)
    center_x = x + center[0]
    center_y = y + center[1]
    return (
        int(round(center_x - size[0] * 0.5)),
        int(round(center_y + (size[1] - baseline) * 0.5)),
    )


def _draw_polygon_label(canvas, polygon, label, color,
                        scale=0.55, thickness=2):
    origin = polygon_label_origin(
        polygon, label, scale=scale, thickness=thickness,
    )
    cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_8)


def orientation_marker_point(polygon):
    """Return a stable, slightly inset point associated with vertex zero."""
    points = np.asarray(polygon, dtype=np.float64)
    if len(points) == 0:
        return None
    vertex = points[0]
    candidate = vertex * 0.8 + points.mean(axis=0) * 0.2
    if cv2.pointPolygonTest(points.astype(np.float32), tuple(candidate), False) < 0:
        candidate = vertex
    rounded = np.round(candidate).astype(np.int32)
    return int(rounded[0]), int(rounded[1])


def _draw_orientation_marker(canvas, polygon):
    point = orientation_marker_point(polygon)
    if point is None:
        return
    cv2.circle(canvas, point, 6, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, point, 3, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_quad_coordinates(canvas, quad):
    """Draw the detected A4 corner and center camera coordinates."""
    rounded_quad = np.round(quad).astype(np.int32)
    for index, point in enumerate(rounded_quad):
        x, y = (int(point[0]), int(point[1]))
        label = "Q%d (%d,%d)" % (index, x, y)
        size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1,
        )
        text_x = min(max(4, x + 6), canvas.shape[1] - size[0] - 4)
        text_y = min(max(TOP_H + STATUS_H + size[1] + 4, y - 6),
                     canvas.shape[0] - BOTTOM_H - 4)
        origin = (text_x, text_y)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (0, 0, 0), 3, cv2.LINE_8)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (40, 255, 80), 1, cv2.LINE_8)

    center = np.round(np.mean(np.asarray(quad), axis=0)).astype(np.int32)
    center_x, center_y = (int(center[0]), int(center[1]))
    label = "C (%d,%d)" % (center_x, center_y)
    size, _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1,
    )
    text_x = min(max(4, center_x + 6), canvas.shape[1] - size[0] - 4)
    text_y = min(max(TOP_H + STATUS_H + size[1] + 4, center_y - 6),
                 canvas.shape[0] - BOTTOM_H - 4)
    origin = (text_x, text_y)
    cv2.circle(canvas, (center_x, center_y), 4, (255, 220, 0), -1,
               cv2.LINE_8)
    cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (0, 0, 0), 3, cv2.LINE_8)
    cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (255, 220, 0), 1, cv2.LINE_8)


def _build_rectified_preview(controller):
    if controller.rectified is None:
        return None
    preview = controller.rectified.copy()
    cv2.line(preview, (0, vision.A4_SPLIT_Y),
             (vision.WARP_W - 1, vision.A4_SPLIT_Y),
             (255, 220, 0), 3, cv2.LINE_8)
    for index, polygon in enumerate(controller.detected_pieces):
        color = PIECE_COLORS[index % len(PIECE_COLORS)]
        points = np.round(polygon).astype(np.int32)
        cv2.polylines(preview, [points], True, color, 4, cv2.LINE_8)
        _draw_orientation_marker(preview, polygon)
        _draw_polygon_label(
            preview, polygon, piece_point_label(index), color, scale=0.65,
        )
    for mark in controller.diagnostics.get("corner_marks", ()):
        box = mark.get("bbox_xyxy")
        if box is not None:
            x0, y0, x1, y1 = np.round(box).astype(np.int32)
            cv2.rectangle(preview, (int(x0), int(y0)), (int(x1), int(y1)),
                          (0, 140, 255), 2, cv2.LINE_8)
        center = np.round(mark["center"]).astype(np.int32)
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(preview, (cx, cy), 6, (0, 140, 255), -1, cv2.LINE_8)
        cv2.circle(preview, (cx, cy), 9, (255, 255, 255), 2, cv2.LINE_8)
        cv2.putText(preview, "Y%.2f" % float(mark["confidence"]),
                    (cx + 8, max(16, cy - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 140, 255), 1, cv2.LINE_8)
    for report in controller.diagnostics.get("arc_reports", ()):
        for corner in report.get("corners", ()):
            arc_points = corner.get("arc_points")
            if arc_points is not None and len(arc_points) >= 2:
                cv2.polylines(
                    preview,
                    [np.round(arc_points).astype(np.int32)],
                    False, (255, 80, 220), 2, cv2.LINE_8)
            virtual = corner.get("virtual_corner")
            if virtual is None:
                continue
            vx, vy = np.round(virtual).astype(np.int32)
            cv2.drawMarker(preview, (int(vx), int(vy)), (255, 80, 220),
                           cv2.MARKER_CROSS, 14, 2, cv2.LINE_8)
            cv2.putText(preview, "V", (int(vx) + 7, int(vy) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 220),
                        1, cv2.LINE_8)
    transforms = controller.diagnostics.get("transforms") or ()
    for index, (piece, transform) in enumerate(
            zip(controller.detected_pieces, transforms)):
        target = np.round(apply_h(piece, transform)).astype(np.int32)
        cv2.polylines(preview, [target], True, (40, 255, 80),
                      3, cv2.LINE_8)
        _draw_orientation_marker(preview, target)
        _draw_polygon_label(
            preview, target, piece_point_label(index, placed=True),
            (40, 255, 80), scale=0.65,
        )
    cv2.putText(preview, "LIVE", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (80, 220, 255), 2, cv2.LINE_8)
    cv2.putText(preview, "TARGET",
                (8, vision.A4_SPLIT_Y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 255, 100),
                2, cv2.LINE_8)

    available_h = CAM_H - TOP_H - STATUS_H - BOTTOM_H - 12
    height = min(300, available_h)
    width = max(1, int(round(height * vision.WARP_W / vision.WARP_H)))
    return cv2.resize(preview, (width, height), interpolation=cv2.INTER_AREA)


def _draw_rectified_preview(canvas, controller):
    if controller.rectified is None:
        return
    if controller._preview_cache is None:
        controller._preview_cache = _build_rectified_preview(controller)
    scaled = controller._preview_cache
    height, width = scaled.shape[:2]
    x0 = CAM_W - width - 8
    y0 = TOP_H + STATUS_H + 6
    cv2.rectangle(canvas, (x0 - 3, y0 - 3),
                  (x0 + width + 2, y0 + height + 2),
                  (15, 15, 15), -1)
    canvas[y0:y0 + height, x0:x0 + width] = scaled
    cv2.rectangle(canvas, (x0 - 1, y0 - 1),
                  (x0 + width, y0 + height), (230, 230, 230), 1)


def draw_ui(rgb, controller, fps=0.0):
    canvas = rgb.copy()
    tab_w = CAM_W // 2
    for index, label in enumerate(BUTTON_LABELS):
        x0 = index * tab_w
        x1 = CAM_W if index == 1 else (index + 1) * tab_w
        color = MODE_COLORS[index] if controller.mode == index else (55, 55, 55)
        cv2.rectangle(canvas, (x0, 0), (x1 - 1, TOP_H - 1), color, -1)
        cv2.rectangle(canvas, (x0, 0), (x1 - 1, TOP_H - 1),
                      (220, 220, 220), 1)
        _draw_centered(canvas, label, x0, x1, 35, 0.52)

    if controller.quad is not None:
        cv2.polylines(canvas, [np.round(controller.quad).astype(np.int32)],
                      True, (40, 255, 80), 2, cv2.LINE_8)
    if controller.split_camera is not None:
        cv2.line(canvas, tuple(controller.split_camera[0]),
                 tuple(controller.split_camera[1]),
                 (255, 220, 0), 2, cv2.LINE_8)

    for index, polygon in enumerate(controller.detected_camera):
        color = PIECE_COLORS[index % len(PIECE_COLORS)]
        cv2.polylines(canvas, [polygon], True, color, 3, cv2.LINE_8)
        _draw_orientation_marker(canvas, polygon)
        _draw_polygon_label(
            canvas, polygon, piece_point_label(index), color,
        )
    for index, polygon in enumerate(controller.target_camera):
        cv2.polylines(canvas, [polygon], True, (40, 255, 80),
                      3, cv2.LINE_8)
        _draw_orientation_marker(canvas, polygon)
        _draw_polygon_label(
            canvas, polygon, piece_point_label(index, placed=True),
            (40, 255, 80),
        )
    _draw_rectified_preview(canvas, controller)
    if controller.quad is not None:
        _draw_quad_coordinates(canvas, controller.quad)

    cv2.rectangle(canvas, (0, TOP_H), (CAM_W - 1, TOP_H + STATUS_H),
                  (0, 0, 0), -1)
    solve_time = ("--" if controller.solve_time_s is None
                  else "%.3fs" % controller.solve_time_s)
    status = "%s | %s | DET:%d | %.1f FPS | SOLVE:%s" % (
        controller.stage, controller.message[:30],
        len(controller.detected_pieces), fps, solve_time)
    cv2.putText(canvas, status, (7, TOP_H + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255),
                1, cv2.LINE_8)

    control_y = CAM_H - BOTTOM_H
    control_w = CAM_W // 4
    for index, label in enumerate(controller.control_labels):
        x0 = index * control_w
        x1 = CAM_W if index == 3 else (index + 1) * control_w
        cv2.rectangle(canvas, (x0, control_y), (x1 - 1, CAM_H - 1),
                      (48, 70, 82), -1)
        cv2.rectangle(canvas, (x0, control_y), (x1 - 1, CAM_H - 1),
                      (220, 220, 220), 1)
        _draw_centered(canvas, label, x0, x1, control_y + 36, 0.48)
    return canvas


def run_device(frame_limit=None):
    if camera is None:
        raise RuntimeError("MaixPy is unavailable; run the PC tests instead")
    print("[BOOT] imports ready", flush=True)
    cam = camera.Camera(CAM_W, CAM_H, image.Format.FMT_RGB888,
                        fps=30, buff_num=2)
    print("[BOOT] camera ready", flush=True)
    screen = display.Display()
    print("[BOOT] display ready", flush=True)
    touch = touchscreen.TouchScreen()
    print("[BOOT] touch ready", flush=True)
    touch_router = TouchRouter()
    controller = MergedController(
        solvers=build_device_solvers(), sender=PositionSender())
    print("[BOOT] controller ready", flush=True)
    cam.skip_frames(vision.SKIP_FRAMES)
    print("[BOOT] camera warmup ready", flush=True)
    fps_meter = maix_time.FPS(10)
    fps = 0.0
    frame_count = 0

    while True:
        if app.need_exit():
            break
        frame = cam.read()
        rgb = image.image2cv(frame, ensure_bgr=False, copy=False)
        event = touch_router.update(touch)
        if event is not None:
            kind, index = event
            if kind == "mode":
                controller.select_mode(index)
            else:
                controller.handle_control(index, rgb)
        controller.process_uart(rgb)
        # cv2image(copy=False) borrows the NumPy storage. Keep canvas alive
        # until display.show() has consumed the Maix image.
        canvas = draw_ui(rgb, controller, fps)
        shown = image.cv2image(canvas, bgr=False, copy=False)
        screen.show(shown)
        fps = fps_meter.fps()
        frame_count += 1
        if frame_limit is not None and frame_count >= frame_limit:
            print("[BOOT] bounded smoke complete frames=%d fps=%.2f" % (
                frame_count, fps), flush=True)
            break


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.discover("tests"))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    smoke_arg = next((value for value in sys.argv
                      if value.startswith("--device-smoke")), None)
    if smoke_arg is not None:
        frame_limit = int(smoke_arg.split("=", 1)[1]) if "=" in smoke_arg else 120
        run_device(frame_limit=frame_limit)
    else:
        run_device()
