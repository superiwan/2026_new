import math
import os
import tempfile
import unittest

import cv2
import numpy as np

import legacy_2026_new as legacy
from core.piece_action import PieceAction
from core.serial_protocol import crc8_ascii, encode_action
from main import BUTTON_LABELS, MergedController, screen_text
from solvers.task1_fixed import Task1FixedSolver
from solvers.task2_white import Task2WhiteSolver
from solvers.task3_poker import Task3PokerSolver, seam_texture_cost


def pose(polygon, angle_degrees, center):
    polygon = np.asarray(polygon, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    local = polygon - polygon.mean(axis=0)
    angle = math.radians(angle_degrees)
    rotation = np.float32(((math.cos(angle), -math.sin(angle)),
                           (math.sin(angle), math.cos(angle))))
    return local.dot(rotation.T) + center


class SerialCapture:
    def __init__(self):
        self.frames = []

    def write(self, value):
        self.frames.append(value.decode("ascii"))


class SenderCapture:
    def __init__(self):
        self.actions = None

    def send(self, actions):
        self.actions = list(actions)


class SolverCapture:
    def solve(self, _rectified):
        return [PieceAction(0, 1, 2, 3, 4, 5, 6)], {"ok": True}


class MergedProjectTest(unittest.TestCase):
    def test_device_screen_text_is_ascii_only(self):
        self.assertTrue(all(label.isascii() for label in BUTTON_LABELS))
        self.assertEqual(screen_text("题1识别失败"), "SOLVER ERROR")
        self.assertEqual(screen_text("A4 NOT FOUND"), "A4 NOT FOUND")

    def test_uart_frame_contains_pick_and_place_pose_with_crc(self):
        action = PieceAction(2, 10.5, 20.25, -30, 100, 120, 45)
        frame = encode_action(action)
        self.assertTrue(frame.startswith("$A,2,10.50,20.25,-30.00,"))
        payload, checksum = frame[1:-2].split("*")
        self.assertEqual(int(checksum, 16), crc8_ascii(payload))

    def test_controller_advances_only_one_heavy_stage_per_frame(self):
        frame = legacy.synthetic_frame(scattered=True)
        sender = SenderCapture()
        controller = MergedController(
            sender=sender,
            solvers=(SolverCapture(), SolverCapture(), SolverCapture()),
        )
        controller.select_mode(1)
        controller.advance(frame)
        self.assertEqual(controller.stage, controller.WARP)
        controller.advance(frame)
        self.assertEqual(controller.stage, controller.SOLVE)
        controller.advance(frame)
        self.assertEqual(controller.stage, controller.DONE)
        self.assertEqual(len(sender.actions), 1)

    def test_task1_fixed_lookup_returns_four_actions(self):
        shapes = (
            np.float32(((0, 0), (52, 0), (16, 42))),
            np.float32(((0, 0), (65, 0), (65, 34), (0, 34))),
            np.float32(((0, 0), (58, 0), (46, 46), (8, 38))),
            np.float32(((10, 0), (58, 12), (50, 48), (18, 55), (0, 22))),
        )
        template = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        current = np.zeros_like(template)
        lower_centers = ((55, 360), (155, 360), (255, 365), (355, 370))
        upper_centers = ((65, 90), (165, 205), (270, 95), (355, 210))
        for index, shape in enumerate(shapes):
            target = pose(shape, 0, lower_centers[index])
            scattered = pose(shape, (12, -24, 35, -48)[index],
                             upper_centers[index])
            cv2.fillPoly(template, [np.round(target).astype(np.int32)],
                         (245, 245, 245))
            cv2.fillPoly(current, [np.round(scattered).astype(np.int32)],
                         (245, 245, 245))

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "task1_layout.json")
            solver = Task1FixedSolver(path)
            layout = solver.calibrate(template)
            self.assertEqual(layout["piece_count"], 4)
            actions, diagnostics = solver.solve(current)
            self.assertEqual(len(actions), 4)
            self.assertEqual(len(diagnostics["assignment"]), 4)
            self.assertTrue(all(0 <= action.place_y <= 297 for action in actions))

    def test_task2_white_returns_millimetre_actions(self):
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        pieces = (
            pose(np.float32(((0, 0), (70, 0), (70, 60), (0, 60))),
                 18, (95, 120)),
            pose(np.float32(((0, 0), (70, 0), (70, 60), (0, 60))),
                 -27, (280, 180)),
        )
        for polygon in pieces:
            cv2.fillPoly(rectified, [np.round(polygon).astype(np.int32)],
                         (245, 245, 245))
        actions, diagnostics = Task2WhiteSolver().solve(rectified)
        self.assertEqual(len(actions), 2)
        self.assertGreater(diagnostics["fill_ratio"], 0.9)
        self.assertTrue(all(0 <= action.pick_x <= 210 for action in actions))
        self.assertTrue(all(0 <= action.pick_y <= 297 for action in actions))

    def test_poker_seam_score_prefers_continuous_texture(self):
        image = np.zeros((160, 240, 3), np.uint8)
        first = np.float64(((20, 20), (70, 20), (70, 140), (20, 140)))
        second = np.float64(((150, 20), (200, 20), (200, 140), (150, 140)))
        for y in range(20, 141):
            colour = (60 + y, 220 - y // 2, 80 + y // 3)
            cv2.line(image, (20, y), (70, y), colour, 1)
            cv2.line(image, (150, y), (200, y), colour, 1)
        pieces = (first, second)
        continuous = (0.0, 0, 1, 1, 3, 0.0, 1.0, 0.0, 1.0)
        wrong = (0.0, 0, 1, 1, 1, 0.0, 1.0, 0.0, 1.0)
        self.assertLess(seam_texture_cost(image, pieces, continuous),
                        seam_texture_cost(image, pieces, wrong))

    def test_task3_poker_full_solver_returns_actions(self):
        image = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        for y in range(80, 180):
            colour = (120 + y % 60, 80 + (y * 2) % 100, 170)
            cv2.line(image, (40, y), (120, y), colour, 1)
            cv2.line(image, (230, y + 45), (310, y + 45), colour, 1)
        cv2.circle(image, (75, 125), 18, (250, 250, 245), -1)
        cv2.circle(image, (265, 170), 18, (250, 250, 245), -1)
        actions, diagnostics = Task3PokerSolver().solve(image)
        self.assertEqual(len(actions), 2)
        self.assertGreater(diagnostics["fill_ratio"], 0.85)
        self.assertGreaterEqual(diagnostics["texture_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
