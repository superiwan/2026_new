import inspect
import unittest

import numpy as np

from core.poker_corner_runtime import CornerMark, PokerCornerRuntime
from main import build_device_solvers


class FakeImageModule:
    def __init__(self):
        self.calls = []

    def cv2image(self, rgb, bgr=False, copy=False):
        self.calls.append((rgb, bgr, copy))
        return "maix-image"


class FakeDetector:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def detect(self, image, conf_th, iou_th):
        self.calls.append((image, conf_th, iou_th))
        return self.results


class PokerCornerRuntimeTest(unittest.TestCase):
    def test_runtime_disables_dual_buffer_for_one_shot_solves(self):
        parameter = inspect.signature(PokerCornerRuntime.load).parameters[
            "dual_buff"]
        self.assertIs(parameter.default, False)

    def test_corner_mark_center_and_dictionary(self):
        mark = CornerMark(10, 20, 8, 12, 0.75)
        self.assertEqual(mark.center, (14.0, 26.0))
        self.assertEqual(mark.confidence, 0.75)
        self.assertEqual(mark.bbox_xyxy, (10, 20, 18, 32))
        self.assertEqual(mark.as_dict()["class_id"], 0)

    def test_runtime_converts_rgb_and_sorts_results(self):
        image_module = FakeImageModule()
        detector = FakeDetector((
            {"x": 4, "y": 5, "w": 6, "h": 7,
             "score": 0.6, "class_id": 0},
            {"x": 10, "y": 12, "width": 8, "height": 9,
             "confidence": 0.9, "class_idx": 0},
        ))
        runtime = PokerCornerRuntime(detector, image_module)

        results = runtime.detect_rgb(
            np.zeros((32, 24, 3), np.uint8), max_detections=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score, 0.9)
        self.assertEqual(detector.calls, [("maix-image", 0.35, 0.45)])
        self.assertFalse(image_module.calls[0][1])
        self.assertFalse(image_module.calls[0][2])

        aliased = runtime.detect(np.zeros((32, 24, 3), np.uint8))
        self.assertEqual(len(aliased), 2)

    def test_runtime_rejects_non_rgb_input(self):
        runtime = PokerCornerRuntime(FakeDetector(()), FakeImageModule())
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            runtime.detect_rgb(np.zeros((32, 24), np.uint8))

    def test_device_solvers_inject_loaded_runtime(self):
        runtime = PokerCornerRuntime(FakeDetector(()), FakeImageModule())

        solvers = build_device_solvers(runtime_loader=lambda: runtime)

        self.assertIs(solvers[1].corner_evidence_detector, runtime)
        self.assertTrue(solvers[1].require_disambiguation)
        self.assertFalse(solvers[1].require_corner_evidence)

    def test_missing_runtime_keeps_both_modes_available(self):
        def fail_load():
            raise RuntimeError("model missing")

        white_solver, poker_solver = build_device_solvers(
            runtime_loader=fail_load)

        self.assertIsNotNone(white_solver)
        self.assertIsNotNone(poker_solver)
        self.assertIsNone(poker_solver.corner_evidence_detector)
        self.assertFalse(poker_solver.require_corner_evidence)


if __name__ == "__main__":
    unittest.main()
