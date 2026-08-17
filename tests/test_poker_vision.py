import itertools
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import legacy_2026_new as legacy
from main import MergedController
from solvers import poker_arc_geometry
from solvers import poker_layout_selector
from solvers import task2_white as geometry
from solvers.task3_poker import (
    Task3PokerSolver, detect_poker_pieces, poker_seam_texture_cost,
)


def translation(dx, dy):
    value = np.eye(3, dtype=np.float64)
    value[:2, 2] = (dx, dy)
    return value


def rounded_rectangle_mask(shape=(220, 280), first=(36, 34),
                           second=(224, 166), radius=14):
    mask = np.zeros(shape, dtype=np.uint8)
    left, top = first
    right, bottom = second
    cv2.rectangle(mask, (left + radius, top),
                  (right - radius, bottom), 255, -1)
    cv2.rectangle(mask, (left, top + radius),
                  (right, bottom - radius), 255, -1)
    for center in (
            (left + radius, top + radius),
            (right - radius, top + radius),
            (left + radius, bottom - radius),
            (right - radius, bottom - radius)):
        cv2.circle(mask, center, radius, 255, -1)
    return mask


def dense_contour(mask):
    return max(cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )[-2], key=cv2.contourArea)


def layout_fixture():
    pieces = (
        np.float64(((10, 10), (50, 10), (50, 50), (10, 50))),
        np.float64(((100, 10), (140, 10), (140, 50), (100, 50))),
        np.float64(((100, 140), (160, 140), (100, 200))),
        np.float64(((140, 140), (200, 140), (200, 200))),
    )
    transforms = (
        translation(150, 90),
        translation(0, 90),
        np.eye(3, dtype=np.float64),
        np.eye(3, dtype=np.float64),
    )
    arc_points = ((10, 10), (140, 10), (100, 200), (200, 200))
    reports = tuple({
        "piece_index": index,
        "corners": ({
            "piece_index": index,
            "virtual_corner": np.asarray(
                arc_points[index], dtype=np.float64),
            "confidence": 1.0,
        },),
        "rejected": (),
    } for index in range(4))
    diagnostics = {
        "pieces": pieces,
        "transforms": transforms,
        "matches": (),
        "fill_ratio": 0.96,
    }
    return pieces, transforms, reports, diagnostics


def corner_mark_layout_fixture():
    pieces = (
        np.float64(((0, 0), (40, 0), (40, 40), (0, 40))),
        np.float64(((100, 100), (140, 100), (140, 140), (100, 140))),
        np.float64(((40, 0), (140, 0), (140, 100), (40, 40))),
        np.float64(((0, 40), (40, 40), (100, 140), (0, 140))),
    )
    transforms = tuple(np.eye(3, dtype=np.float64) for _piece in pieces)
    reports = tuple({
        "piece_index": index,
        "corners": (),
        "rejected": (),
    } for index in range(4))
    marks = (
        {"piece_index": 0, "center": np.float64((5, 5)),
         "confidence": 0.9, "direction": None, "bbox_xyxy": None},
        {"piece_index": 1, "center": np.float64((135, 135)),
         "confidence": 0.9, "direction": None, "bbox_xyxy": None},
        {"piece_index": 2, "center": np.float64((130, 10)),
         "confidence": 1.0, "direction": None, "bbox_xyxy": None},
    )
    diagnostics = {
        "pieces": pieces,
        "transforms": transforms,
        "matches": (),
        "fill_ratio": 0.96,
    }
    return pieces, transforms, reports, marks, diagnostics


def permuted_layout(pieces, transforms, reports, order):
    reordered_pieces = tuple(pieces[index] for index in order)
    reordered_transforms = tuple(transforms[index] for index in order)
    reordered_reports = []
    for new_index, old_index in enumerate(order):
        corners = []
        for corner in reports[old_index]["corners"]:
            value = dict(corner)
            value["piece_index"] = new_index
            corners.append(value)
        reordered_reports.append({
            "piece_index": new_index,
            "corners": tuple(corners),
            "rejected": (),
        })
    return reordered_pieces, reordered_transforms, tuple(reordered_reports)


class SenderCapture:
    def __init__(self):
        self.calls = []

    def send(self, position_pairs):
        self.calls.append(tuple(position_pairs))
        return ()

    def discard_input(self):
        pass


class PokerVisionTest(unittest.TestCase):
    def test_closest_same_shape_pair_recovers_two_matching_trapezoids(self):
        pieces = (
            np.float64(((0, 0), (50, 0), (42, 80), (8, 80))),
            np.float64(((100, 0), (155, 5), (145, 60), (105, 55))),
            np.float64(((200, 0), (255, 5), (245, 60), (205, 55))),
            np.float64(((300, 0), (380, 0), (370, 35), (310, 35))),
        )

        self.assertEqual(
            poker_layout_selector.closest_same_shape_pair(pieces), (1, 2))

    def test_close_matched_seams_removes_target_safety_gap(self):
        pieces = (
            np.float64(((0, 0), (40, 0), (40, 40), (0, 40))),
            np.float64(((60, 0), (100, 0), (100, 40), (60, 40))),
        )
        transforms = tuple(np.eye(3, dtype=np.float64) for _ in pieces)
        match = (0.0, 0, 1, 1, 3, 0.0, 1.0, 0.0, 1.0)

        closed = poker_layout_selector.close_matched_seams(
            pieces, transforms, (match,))
        first_start, first_end, second_start, second_end = (
            geometry.match_segments(pieces, match))
        first_target = geometry.apply_h(
            np.asarray((first_start, first_end)), closed[0])
        second_target = geometry.apply_h(
            np.asarray((second_end, second_start)), closed[1])

        np.testing.assert_allclose(first_target, second_target, atol=1e-6)

    def test_corner_opposition_prefers_matching_diagonal_marks(self):
        correct = np.full((120, 80, 3), 235, np.uint8)
        wrong = correct.copy()
        cv2.rectangle(correct, (3, 3), (18, 24), (20, 20, 20), -1)
        cv2.rectangle(correct, (61, 95), (76, 116), (20, 20, 20), -1)
        cv2.rectangle(wrong, (3, 3), (18, 24), (20, 20, 20), -1)
        cv2.rectangle(wrong, (61, 3), (76, 24), (20, 20, 20), -1)
        mask = np.ones(correct.shape[:2], np.uint8)

        correct_score = poker_layout_selector._corner_opposition_score(
            correct, mask)
        wrong_score = poker_layout_selector._corner_opposition_score(
            wrong, mask)

        self.assertGreater(correct_score, wrong_score + 0.25)

    def test_poker_seam_texture_prefers_continuous_print(self):
        pieces = (
            np.float64(((10, 10), (50, 10), (50, 50), (10, 50))),
            np.float64(((60, 10), (100, 10), (100, 50), (60, 50))),
        )
        match = (0.0, 0, 1, 1, 3, 0.0, 1.0, 0.0, 1.0)
        continuous = np.zeros((120, 120, 3), np.uint8)
        continuous[10:51, 10:51] = (220, 30, 30)
        continuous[10:51, 60:101] = (220, 30, 30)
        broken = continuous.copy()
        broken[10:51, 60:101] = (30, 30, 220)

        continuous_cost = poker_seam_texture_cost(
            continuous, pieces, match)
        broken_cost = poker_seam_texture_cost(broken, pieces, match)

        self.assertLess(continuous_cost, broken_cost * 0.25)

    def test_green_a4_rounded_piece_recovers_four_virtual_corners(self):
        image = np.full(
            (legacy.WARP_H, legacy.WARP_W, 3), (70, 170, 65),
            dtype=np.uint8)
        card_mask = rounded_rectangle_mask(
            shape=image.shape[:2], first=(86, 104), second=(274, 236),
            radius=14)
        image[card_mask != 0] = (235, 235, 230)

        pieces, binary = detect_poker_pieces(image)
        reports = poker_arc_geometry.analyze_piece_arcs(binary, pieces)

        self.assertEqual(len(pieces), 1)
        self.assertEqual(len(pieces[0]), 4)
        self.assertEqual(len(reports[0]["corners"]), 4)
        actual = np.asarray(sorted(
            (tuple(np.round(corner["virtual_corner"]).astype(int))
             for corner in reports[0]["corners"])))
        expected = np.asarray(sorted(((86, 104), (274, 104),
                                      (86, 236), (274, 236))))
        self.assertTrue(np.allclose(actual, expected, atol=2.0))

    def test_real_poker_fixtures_keep_four_corner_prior(self):
        fixture_dir = "tests/fixtures"
        for index in range(1, 6):
            with self.subTest(index=index):
                image = cv2.cvtColor(cv2.imread(
                    "%s/poker_real_%d.png" % (fixture_dir, index)),
                    cv2.COLOR_BGR2RGB)
                pieces, mask = detect_poker_pieces(image)
                reports = poker_arc_geometry.analyze_piece_arcs(
                    mask, pieces)
                self.assertEqual(
                    len(poker_arc_geometry.corner_points(reports)), 4)

    def test_sharp_rectangle_is_not_reported_as_rounded(self):
        mask = np.zeros((180, 240), dtype=np.uint8)
        cv2.rectangle(mask, (35, 30), (205, 150), 255, -1)

        recovered, report = poker_arc_geometry.recover_virtual_corners(
            dense_contour(mask))

        self.assertEqual(report["corners"], ())
        self.assertEqual(len(recovered), 4)

    def test_same_shape_pair_requires_exactly_one_rectangle_pair(self):
        pieces, _transforms, _reports, _diagnostics = layout_fixture()

        self.assertEqual(
            poker_layout_selector.same_shape_pairs(pieces), ((0, 1),))
        self.assertEqual(
            poker_layout_selector.unique_same_shape_pair(pieces), (0, 1))
        four_squares = pieces[:2] + tuple(
            pieces[index] + (250, 0) for index in range(2))
        self.assertIsNone(
            poker_layout_selector.unique_same_shape_pair(four_squares))

    def test_detector_evidence_interface_associates_bbox_to_piece(self):
        pieces, _transforms, _reports, _diagnostics = layout_fixture()

        class Detector:
            def detect(self, image):
                self.shape = image.shape
                return ({
                    "bbox_xyxy": (15, 15, 31, 35),
                    "confidence": 0.91,
                    "direction": (1, 1),
                },)

        detector = Detector()
        evidence = poker_layout_selector.collect_corner_mark_evidence(
            detector, np.zeros((594, 420, 3), dtype=np.uint8), pieces)

        self.assertEqual(detector.shape, (594, 420, 3))
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["piece_index"], 0)
        self.assertAlmostEqual(evidence[0]["confidence"], 0.91)
        self.assertEqual(evidence[0]["bbox_xyxy"],
                         (15.0, 15.0, 31.0, 35.0))

    def test_only_equal_pair_marks_select_edge_assignment(self):
        pieces, _transforms, reports, marks, diagnostics = (
            corner_mark_layout_fixture())

        selected, result = poker_layout_selector.select_poker_layout(
            pieces, diagnostics, reports, marks)

        self.assertEqual(selected["assignment"], "original")
        self.assertEqual(result["same_shape_pair"], (0, 1))
        self.assertEqual(result["corner_mark_count"], 2)
        self.assertEqual(result["ignored_corner_mark_count"], 1)
        self.assertEqual(
            result["candidate_scores"][1]["rejection"],
            "mark_outside_corner_zone")
        self.assertFalse(result["ambiguous"])

    def test_task3_regenerates_actions_from_mark_selected_transforms(self):
        pieces, _transforms, reports, marks, diagnostics = (
            corner_mark_layout_fixture())
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        binary = np.zeros(rectified.shape[:2], np.uint8)
        with (
            patch.object(
                geometry.Task2WhiteSolver, "solve_detected",
                return_value=((), dict(diagnostics))),
            patch.object(
                poker_arc_geometry, "analyze_piece_arcs",
                return_value=reports),
            patch.object(
                poker_layout_selector, "collect_corner_mark_evidence",
                return_value=marks),
        ):
            actions, result = Task3PokerSolver().solve_detected(
                rectified, pieces, binary)

        self.assertEqual(result["selected_assignment"], "original")
        self.assertEqual(len(actions), 4)
        self.assertEqual(result["corner_mark_count"], 2)
        self.assertEqual(result["ignored_corner_mark_count"], 1)

    def test_layout_selection_is_invariant_for_all_piece_permutations(self):
        pieces, transforms, reports, marks, _diagnostics = (
            corner_mark_layout_fixture())
        expected_centers = None
        for order in itertools.permutations(range(4)):
            with self.subTest(order=order):
                values = permuted_layout(
                    pieces, transforms, reports, order)
                reordered_pieces, reordered_transforms, reordered_reports = (
                    values)
                old_to_new = {old: new for new, old in enumerate(order)}
                reordered_marks = tuple(
                    dict(mark, piece_index=old_to_new[mark["piece_index"]])
                    for mark in marks)
                selected, result = poker_layout_selector.select_poker_layout(
                    reordered_pieces,
                    {"transforms": reordered_transforms},
                    reordered_reports, reordered_marks)
                centers = {}
                for new_index, old_index in enumerate(order):
                    source_center = reordered_pieces[new_index].mean(axis=0)
                    centers[old_index] = geometry.apply_h(
                        np.asarray((source_center,)),
                        selected["transforms"][new_index])[0]
                ordered_centers = np.asarray(
                    [centers[index] for index in range(4)])
                if expected_centers is None:
                    expected_centers = ordered_centers
                self.assertTrue(np.allclose(
                    ordered_centers, expected_centers, atol=1e-6))
                self.assertEqual(result["selected_assignment"], "original")

    def test_best_effort_task2_returns_recorded_rectangle_after_strict_failure(self):
        pieces = (
            np.float64(((10, 10), (50, 10), (50, 50), (10, 50))),
            np.float64(((60, 10), (100, 10), (100, 50), (60, 50))),
        )
        transforms = (translation(120, 300), translation(120, 300))

        def fail_strict(_pieces, _paper, candidate_recorder=None, **_options):
            candidate_recorder(transforms, (), 0.81, (10.0,))
            raise RuntimeError("strict quality failed")

        with patch.object(geometry, "solve", side_effect=fail_strict):
            actions, diagnostics = geometry.Task2WhiteSolver().solve_detected(
                np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8),
                pieces,
                solve_options={"max_anchor_orders": 1},
                return_best_candidate_on_failure=True,
            )

        self.assertEqual(len(actions), 2)
        self.assertTrue(diagnostics["best_effort"])
        self.assertEqual(diagnostics["fill_ratio"], 0.81)

    def test_poker_best_effort_prefers_card_ratio_over_higher_fill(self):
        pieces = (
            np.float64(((10, 10), (50, 10), (50, 50), (10, 50))),
            np.float64(((60, 10), (100, 10), (100, 50), (60, 50))),
        )
        wrong_ratio = (translation(20, 250), translation(20, 250))
        card_ratio = (translation(120, 300), translation(120, 300))

        def fail_strict(_pieces, _paper, candidate_recorder=None, **_options):
            candidate_recorder(
                wrong_ratio, (), 0.95, (5.0,), (40.0, 120.0))
            candidate_recorder(
                card_ratio, (), 0.90, (8.0,), (70.0, 100.0))
            raise RuntimeError("strict quality failed")

        with patch.object(geometry, "solve", side_effect=fail_strict):
            actions, diagnostics = geometry.Task2WhiteSolver().solve_detected(
                np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8),
                pieces,
                solve_options={"max_anchor_orders": 1},
                candidate_validator=(
                    lambda size: abs(min(size) / max(size) - 5.0 / 7.0)
                    / (5.0 / 7.0) <= 0.10),
                best_effort_candidate_priority=(
                    lambda size: abs(min(size) / max(size) - 5.0 / 7.0)),
                return_best_candidate_on_failure=True,
            )

        self.assertEqual(len(actions), 2)
        self.assertEqual(diagnostics["fill_ratio"], 0.90)
        self.assertEqual(
            diagnostics["best_effort_target_size_mm"], (70.0, 100.0))
        self.assertTrue(diagnostics["best_effort_quality_valid"])

    def test_task3_forces_same_shape_assignment_when_marks_are_ambiguous(self):
        pieces, transforms, reports, diagnostics = layout_fixture()
        ambiguous_reports = tuple(
            dict(report, corners=(() if index else report["corners"]))
            for index, report in enumerate(reports))
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        binary = np.zeros(rectified.shape[:2], np.uint8)
        base_actions = (object(), object(), object(), object())

        with (
            patch.object(
                geometry.Task2WhiteSolver, "solve_detected",
                return_value=(base_actions, dict(diagnostics))),
            patch.object(
                poker_arc_geometry, "analyze_piece_arcs",
                return_value=ambiguous_reports),
        ):
            actions, result = Task3PokerSolver(
                require_disambiguation=True,
            ).solve_detected(rectified, pieces, binary)

        self.assertEqual(len(actions), 4)
        self.assertNotIn("best_effort", result)
        self.assertEqual(result["same_shape_pair"], (0, 1))
        self.assertEqual(result["selected_assignment"], "original")
        self.assertNotIn("disambiguation_skipped", result)


if __name__ == "__main__":
    unittest.main()
