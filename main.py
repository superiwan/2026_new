"""Interactive three-mode MaixCAM Pro puzzle application."""

import os
import sys

import cv2
import numpy as np

try:
    from maix import app, camera, display, image, time as maix_time, touchscreen
except ImportError:
    app = camera = display = image = maix_time = touchscreen = None

import legacy_2026_new as vision
from solvers import Task1FixedSolver, Task2WhiteSolver, Task3PokerSolver
from solvers.task2_white import apply_h
from solvers.task3_poker import detect_poker_pieces


CAM_W, CAM_H = vision.CAM_W, vision.CAM_H
TOP_H = 54
STATUS_H = 30
BOTTOM_H = 58
BUTTON_LABELS = ("TASK1 FIXED", "TASK2 WHITE", "TASK2 POKER")
MODE_COLORS = ((48, 150, 255), (70, 190, 90), (190, 105, 225))
PIECE_COLORS = ((255, 90, 90), (90, 255, 120),
                (80, 170, 255), (255, 190, 70))


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
            return "mode", min(2, max(0, int(x * 3 // CAM_W)))
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
    """User-driven A4, template, detection and solve interaction state."""

    CONTROL_SETS = (
        ("FIND A4", "SAVE TPL", "DETECT", "CLEAR"),
        ("FIND A4", "DETECT", "SOLVE", "RESET"),
        ("FIND A4", "DETECT", "SOLVE", "RESET"),
    )

    def __init__(self, solvers=None):
        self.solvers = solvers or (
            Task1FixedSolver(), Task2WhiteSolver(), Task3PokerSolver(),
        )
        self.mode = 0
        self.stage = "READY"
        self.message = "TAP FIND A4"
        self.quad = None
        self.homography = None
        self.inverse_homography = None
        self.split_camera = None
        self.region_remaps = {}
        self.rectified = None
        self.detected_pieces = []
        self.detected_camera = []
        self.target_camera = []
        self.saved_camera = []
        self.assignment = None
        self.actions = []
        self.diagnostics = {}
        self.template_layout = None
        self.last_auto_detect_ms = 0
        self._load_template()

    @property
    def control_labels(self):
        return self.CONTROL_SETS[self.mode]

    def _load_template(self):
        solver = self.solvers[0]
        path = getattr(solver, "template_path", None)
        self.template_layout = vision.load_layout(path) if path else None

    def _clear_results(self):
        self.rectified = None
        self.detected_pieces = []
        self.detected_camera = []
        self.target_camera = []
        self.assignment = None
        self.actions = []
        self.diagnostics = {}

    def _refresh_projection(self):
        if self.homography is None:
            self.inverse_homography = None
            self.split_camera = None
            self.region_remaps = {}
            self.saved_camera = []
            return
        self.region_remaps = {
            "upper": vision.build_region_remap(
                self.homography, vision.UPPER_REGION),
            "lower": vision.build_region_remap(
                self.homography, vision.LOWER_REGION),
        }
        self.inverse_homography, self.split_camera, self.saved_camera = (
            vision.prepare_camera_geometry(
                self.homography, self.template_layout if self.mode == 0 else None))

    def select_mode(self, mode):
        self.mode = int(mode)
        self._clear_results()
        self._load_template()
        self._refresh_projection()
        self.last_auto_detect_ms = 0
        self.stage = "A4 LOCKED" if self.homography is not None else "READY"
        self.message = ("SELECT ACTION" if self.homography is not None
                        else "TAP FIND A4")

    def _warp_current(self, rgb):
        if self.homography is None:
            raise RuntimeError("FIND A4 FIRST")
        rectified = vision.warp_a4(rgb, self.homography)
        if not vision.cached_a4_is_valid(rectified):
            raise RuntimeError("A4 CACHE INVALID")
        self.rectified = rectified
        return rectified

    def _refresh_upper_preview(self, rgb):
        """Update the live half of the A4 preview using the cached remap."""
        if self.rectified is None or "upper" not in self.region_remaps:
            return
        map1, map2 = self.region_remaps["upper"]
        upper = cv2.remap(
            rgb, map1, map2, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        top, bottom = vision.UPPER_REGION
        self.rectified[top:bottom] = upper

    def find_a4(self, rgb):
        self.quad, self.homography = vision.detect_a4(rgb)
        self._clear_results()
        if self.homography is None:
            self._refresh_projection()
            raise RuntimeError("A4 NOT FOUND")
        self._refresh_projection()
        self.rectified = vision.warp_a4(rgb, self.homography)
        self.stage = "A4 LOCKED"
        self.message = "A4 READY - SELECT ACTION"
        self.last_auto_detect_ms = 0

    def save_template(self, rgb):
        rectified = self._warp_current(rgb)
        layout = self.solvers[0].calibrate(rectified)
        self.template_layout = layout
        self._refresh_projection()
        self.stage = "TEMPLATE SAVED"
        self.message = "SAVED 4 LOWER PIECES"
        self.last_auto_detect_ms = 0

    def clear_template(self):
        solver = self.solvers[0]
        path = getattr(solver, "template_path", None)
        removed = vision.delete_layout(path) if path else False
        self.template_layout = None
        self.saved_camera = []
        self._clear_results()
        self.stage = "A4 LOCKED" if self.homography is not None else "READY"
        self.message = "TEMPLATE CLEARED" if removed else "NO TEMPLATE"

    def detect(self, rgb, refresh_rectified=True):
        if self.homography is None:
            raise RuntimeError("FIND A4 FIRST")
        rectified = (self._warp_current(rgb) if refresh_rectified
                     else self.rectified)
        if not refresh_rectified:
            self._refresh_upper_preview(rgb)
        self.target_camera = []
        self.actions = []
        self.assignment = None
        if self.mode == 0:
            analysis, timings = vision.analyze_region_fast(
                rgb, self.region_remaps["upper"], vision.UPPER_REGION)
            pieces = analysis["pieces"]
            binary = analysis["binary"]
            extra = {"piece_binary": binary, "timings": timings}
            if self.template_layout is not None:
                assignment, status, score = vision.match_pieces_to_layout(
                    pieces, self.template_layout)
                self.assignment = assignment
                extra.update({"match_status": status, "match_score": score})
                self.message = status
            else:
                self.message = "NO TEMPLATE - SAVE LOWER"
        elif self.mode == 1:
            pieces, binary, timings = vision.detect_pieces(rectified)
            extra = {"piece_binary": binary, "timings": timings}
            self.message = "WHITE PIECES %d" % len(pieces)
        else:
            pieces, binary = detect_poker_pieces(rectified)
            extra = {"piece_binary": binary}
            self.message = "POKER PIECES %d" % len(pieces)
        self.detected_pieces = [np.asarray(piece, dtype=np.float64)
                                for piece in pieces]
        self.detected_camera = [
            vision.project_to_camera(piece, self.inverse_homography)
            for piece in self.detected_pieces
        ]
        self.diagnostics = {"pieces": self.detected_pieces, **extra}
        self.stage = "DETECTED"

    def auto_update(self, rgb, now_ms=None):
        """Restore 2026_new's 10 Hz upper-piece preview for task 1."""
        if (self.mode != 0 or self.homography is None
                or self.template_layout is None):
            return False
        now_ms = vision.ticks_ms() if now_ms is None else int(now_ms)
        if (self.last_auto_detect_ms
                and now_ms - self.last_auto_detect_ms
                < vision.DETECTION_INTERVAL_MS):
            return False
        self.last_auto_detect_ms = now_ms
        try:
            self.detect(rgb, refresh_rectified=False)
            return True
        except Exception as error:
            self.stage = "ERROR"
            self.message = screen_text(error)
            print("[MERGE] AUTO DETECT ERROR: %s" % error)
            return False

    def solve(self, rgb):
        rectified = self._warp_current(rgb)
        self.actions, self.diagnostics = self.solvers[self.mode].solve(rectified)
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
        self.stage = "SOLVED"
        self.message = "SOLUTION %d PIECES" % len(self.detected_pieces)
        print("[MERGE] mode=%s solved=%d" % (
            BUTTON_LABELS[self.mode], len(self.detected_pieces)))

    def reset(self):
        self._clear_results()
        self.stage = "A4 LOCKED" if self.homography is not None else "READY"
        self.message = ("SELECT ACTION" if self.homography is not None
                        else "TAP FIND A4")

    def handle_control(self, index, rgb):
        try:
            if index == 0:
                self.find_a4(rgb)
            elif self.mode == 0 and index == 1:
                self.save_template(rgb)
            elif self.mode == 0 and index == 2:
                self.detect(rgb)
            elif self.mode == 0 and index == 3:
                self.clear_template()
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


def _draw_rectified_preview(canvas, controller):
    if controller.rectified is None:
        return
    preview = controller.rectified.copy()
    cv2.line(preview, (0, vision.A4_SPLIT_Y),
             (vision.WARP_W - 1, vision.A4_SPLIT_Y),
             (255, 220, 0), 3, cv2.LINE_8)
    if controller.mode == 0 and controller.template_layout is not None:
        for index, polygon in enumerate(
                vision.layout_polygons(controller.template_layout)):
            color = PIECE_COLORS[index % len(PIECE_COLORS)]
            cv2.polylines(preview,
                          [np.round(polygon).astype(np.int32)],
                          True, color, 2, cv2.LINE_8)
    for index, polygon in enumerate(controller.detected_pieces):
        color = PIECE_COLORS[index % len(PIECE_COLORS)]
        points = np.round(polygon).astype(np.int32)
        cv2.polylines(preview, [points], True, color, 4, cv2.LINE_8)
        center = np.round(polygon.mean(axis=0)).astype(int)
        cv2.putText(preview, "P%d" % index, tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_8)
    transforms = controller.diagnostics.get("transforms") or ()
    for piece, transform in zip(controller.detected_pieces, transforms):
        target = np.round(apply_h(piece, transform)).astype(np.int32)
        cv2.polylines(preview, [target], True, (40, 255, 80),
                      3, cv2.LINE_8)
    cv2.putText(preview, "LIVE", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (80, 220, 255), 2, cv2.LINE_8)
    cv2.putText(preview, "TEMPLATE",
                (8, vision.A4_SPLIT_Y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 255, 100),
                2, cv2.LINE_8)

    available_h = CAM_H - TOP_H - STATUS_H - BOTTOM_H - 12
    height = min(300, available_h)
    width = max(1, int(round(height * vision.WARP_W / vision.WARP_H)))
    scaled = cv2.resize(preview, (width, height), interpolation=cv2.INTER_AREA)
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
    tab_w = CAM_W // 3
    for index, label in enumerate(BUTTON_LABELS):
        x0 = index * tab_w
        x1 = CAM_W if index == 2 else (index + 1) * tab_w
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

    for index, polygon in enumerate(controller.saved_camera):
        color = PIECE_COLORS[index % len(PIECE_COLORS)]
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_8)
    for index, polygon in enumerate(controller.detected_camera):
        color_index = (controller.assignment[index]
                       if controller.assignment is not None
                       and index < len(controller.assignment) else index)
        color = PIECE_COLORS[color_index % len(PIECE_COLORS)]
        cv2.polylines(canvas, [polygon], True, color, 3, cv2.LINE_8)
        center = np.round(polygon.mean(axis=0)).astype(int)
        cv2.putText(canvas, "P%d" % index, tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_8)
    for polygon in controller.target_camera:
        cv2.polylines(canvas, [polygon], True, (40, 255, 80),
                      3, cv2.LINE_8)
    _draw_rectified_preview(canvas, controller)

    cv2.rectangle(canvas, (0, TOP_H), (CAM_W - 1, TOP_H + STATUS_H),
                  (0, 0, 0), -1)
    template = ("TPL:%d" % controller.template_layout.get("piece_count", 0)
                if controller.template_layout is not None else "TPL:NONE")
    status = "%s | %s | DET:%d | %s | %.1f FPS" % (
        controller.stage, controller.message[:30],
        len(controller.detected_pieces), template, fps)
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
    controller = MergedController()
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
        elif controller.mode == 0:
            controller.auto_update(rgb)
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
