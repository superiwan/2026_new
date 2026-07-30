"""Merged three-mode MaixCAM Pro puzzle application."""

import sys

import cv2

try:
    from maix import app, camera, display, image, time as maix_time, touchscreen
except ImportError:
    app = camera = display = image = maix_time = touchscreen = None

import legacy_2026_new as vision
from core.serial_protocol import ActionSender
from solvers import Task1FixedSolver, Task2WhiteSolver, Task3PokerSolver


CAM_W, CAM_H = vision.CAM_W, vision.CAM_H
BUTTON_H = 64
BUTTON_LABELS = ("题1-固定", "题2-纯白", "题2-扑克")
BUTTON_FALLBACKS = ("1 FIXED", "2 WHITE", "2 POKER")
MODE_COLORS = ((48, 150, 255), (70, 190, 90), (190, 105, 225))


class ModeTouch:
    def __init__(self):
        self.pressed_mode = None

    def update(self, touch):
        x, y, pressed = touch.read()
        mode = min(2, max(0, int(x * 3 // CAM_W))) if y < BUTTON_H else None
        if pressed:
            self.pressed_mode = mode
            return None
        selected = self.pressed_mode
        self.pressed_mode = None
        return selected if selected is not None and selected == mode else None


class MergedController:
    IDLE = "IDLE"
    LOCATE = "LOCATE A4"
    WARP = "RECTIFY A4"
    SOLVE = "SOLVE"
    DONE = "DONE"
    ERROR = "ERROR"

    def __init__(self, sender=None, solvers=None):
        self.sender = sender or ActionSender()
        self.solvers = solvers or (
            Task1FixedSolver(), Task2WhiteSolver(), Task3PokerSolver(),
        )
        self.mode = None
        self.stage = self.IDLE
        self.message = "SELECT MODE"
        self.quad = None
        self.homography = None
        self.rectified = None
        self.actions = []
        self.diagnostics = {}

    def select_mode(self, mode):
        self.mode = int(mode)
        self.stage = self.LOCATE
        self.message = BUTTON_LABELS[self.mode]
        self.quad = self.homography = self.rectified = None
        self.actions = []
        self.diagnostics = {}

    def advance(self, rgb):
        """Advance at most one expensive stage for the current camera frame."""
        try:
            if self.stage == self.LOCATE:
                self.quad, self.homography = vision.detect_a4(rgb)
                if self.homography is None:
                    raise RuntimeError("A4 NOT FOUND")
                self.stage = self.WARP
                self.message = "A4 LOCKED"
            elif self.stage == self.WARP:
                self.rectified = vision.warp_a4(rgb, self.homography)
                if not vision.cached_a4_is_valid(self.rectified):
                    raise RuntimeError("A4 CACHE INVALID")
                self.stage = self.SOLVE
                self.message = "RECTIFIED 0.5MM/PX"
            elif self.stage == self.SOLVE:
                self.actions, self.diagnostics = self.solvers[self.mode].solve(
                    self.rectified)
                self.sender.send(self.actions)
                self.stage = self.DONE
                self.message = "DONE %d ACTIONS" % len(self.actions)
                print("[MERGE] mode=%s actions=%d" % (
                    BUTTON_LABELS[self.mode], len(self.actions)))
                for action in self.actions:
                    print("[ACTION] %s" % action)
        except Exception as error:
            self.stage = self.ERROR
            self.message = str(error)
            print("[MERGE] ERROR mode=%s: %s" % (
                BUTTON_LABELS[self.mode] if self.mode is not None else "-",
                error))


def draw_ui(rgb, controller, fps=0.0):
    canvas = rgb.copy()
    third = CAM_W // 3
    for index in range(3):
        x0 = index * third
        x1 = CAM_W if index == 2 else (index + 1) * third
        selected = controller.mode == index
        color = MODE_COLORS[index] if selected else (65, 65, 65)
        cv2.rectangle(canvas, (x0, 0), (x1 - 1, BUTTON_H - 1), color, -1)
        cv2.rectangle(canvas, (x0, 0), (x1 - 1, BUTTON_H - 1), (235, 235, 235), 1)
        if image is None:
            cv2.putText(canvas, BUTTON_FALLBACKS[index], (x0 + 24, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                        cv2.LINE_AA)
    if controller.quad is not None:
        cv2.polylines(canvas, [controller.quad.astype("int32")], True,
                      (40, 255, 80), 3, cv2.LINE_AA)
    status = "%s | %s | %.1f FPS" % (
        controller.stage, controller.message[:42], fps)
    cv2.rectangle(canvas, (0, CAM_H - 38), (CAM_W, CAM_H), (0, 0, 0), -1)
    cv2.putText(canvas, status, (10, CAM_H - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return canvas


def draw_chinese_labels(maix_image):
    """Use MaixPy's Unicode renderer, falling back to ASCII on the device."""
    try:
        color = image.Color.from_rgb(255, 255, 255)
        third = CAM_W // 3
        for index, label in enumerate(BUTTON_LABELS):
            maix_image.draw_string(index * third + 48, 17, label,
                                   color=color, scale=1.0, thickness=1)
    except Exception:
        try:
            color = image.Color.from_rgb(255, 255, 255)
            third = CAM_W // 3
            for index, label in enumerate(BUTTON_FALLBACKS):
                maix_image.draw_string(index * third + 48, 17, label,
                                       color=color, scale=1.0, thickness=1)
        except Exception:
            pass


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
    touch_state = ModeTouch()
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
        mode = touch_state.update(touch)
        if mode is not None:
            controller.select_mode(mode)
        controller.advance(rgb)
        # cv2image(copy=False) borrows the NumPy storage. Keep this reference
        # alive until display.show() has consumed the Maix image; passing the
        # temporary draw_ui(...) result directly can leave a dangling pointer.
        canvas = draw_ui(rgb, controller, fps)
        shown = image.cv2image(canvas, bgr=False, copy=False)
        draw_chinese_labels(shown)
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
