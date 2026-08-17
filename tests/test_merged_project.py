import itertools
import math
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import legacy_2026_new as legacy
from core.piece_action import PieceAction
from core.serial_protocol import (
    OVER_FRAME, PositionSender, encode_position_pair, is_valid_pixel_point,
)
from main import (
    BUTTON_LABELS, MergedController, TouchRouter, draw_ui,
    _draw_quad_coordinates, orientation_marker_point, piece_point_label,
    polygon_label_origin, screen_text,
)
from solvers import task2_white as geometry
from solvers.task2_white import Task2WhiteSolver
from solvers.task3_poker import (
    Task3PokerSolver, _approximate_poker_piece,
    _poker_target_aspect_error,
    _poker_target_aspect_valid,
    _repair_shallow_concave_notches,
    _same_piece_geometry, detect_poker_pieces,
)


def pose(polygon, angle_degrees, center):
    polygon = np.asarray(polygon, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    local = polygon - polygon.mean(axis=0)
    angle = math.radians(angle_degrees)
    rotation = np.float32(((math.cos(angle), -math.sin(angle)),
                           (math.sin(angle), math.cos(angle))))
    return local.dot(rotation.T) + center


class SerialCapture:
    def __init__(self, incoming=()):
        self.frames = []
        self.incoming = list(incoming)

    def write(self, value):
        self.frames.append(value.decode("ascii"))

    def read(self):
        return self.incoming.pop(0) if self.incoming else None


class SenderCapture:
    def __init__(self):
        self.position_pairs = []

    def send(self, position_pairs):
        self.position_pairs.extend(position_pairs)
        return [encode_position_pair(green, degree, red)
                for green, degree, red in position_pairs
                if (is_valid_pixel_point(green)
                    and is_valid_pixel_point(red))]


class SolverCapture:
    def __init__(self):
        self.solve_count = 0

    def solve(self, _rectified):
        self.solve_count += 1
        action = PieceAction(0, 100, 100, 3, 120, 150, 6)
        piece = np.float64(((10, 10), (30, 10), (30, 30), (10, 30)))
        return [action], {"pieces": (piece,), "transforms": (np.eye(3),)}


class FailingSolver:
    def solve(self, _rectified):
        raise RuntimeError("QUALITY GATE FAILED")


class TimeoutSolver:
    def solve(self, _rectified):
        raise geometry.SolveTimeoutError()


class TwoActionSolver(SolverCapture):
    def solve(self, _rectified):
        self.solve_count += 1
        actions = (
            PieceAction(0, 100, 100, 3, 120, 150, 6),
            PieceAction(1, 80, 100, -4, 100, 140, 8),
        )
        pieces = (
            np.float64(((10, 10), (30, 10), (30, 30), (10, 30))),
            np.float64(((40, 10), (60, 10), (60, 30), (40, 30))),
        )
        return list(actions), {
            "pieces": pieces,
            "transforms": (np.eye(3), np.eye(3)),
        }


class ReuseSolver(SolverCapture):
    def __init__(self):
        super().__init__()
        self.reused = False

    def solve_detected(self, rectified, pieces, binary):
        self.reused = True
        self.reused_piece_count = len(pieces)
        self.reused_binary = binary
        return self.solve(rectified)


class MergedProjectTest(unittest.TestCase):
    def test_a4_quad_and_center_coordinates_are_drawn_on_screen(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        quad = np.float32(((20, 90), (610, 95), (615, 400), (15, 405)))

        _draw_quad_coordinates(canvas, quad)

        self.assertGreater(np.count_nonzero(canvas), 100)
        self.assertTrue(np.any(canvas[248, 315] != 0))

    def test_piece_point_labels_pair_scattered_and_placed_coordinates(self):
        self.assertEqual(piece_point_label(0), "P0")
        self.assertEqual(piece_point_label(0, placed=True), "P1")
        self.assertEqual(piece_point_label(1), "P2")
        self.assertEqual(piece_point_label(1, placed=True), "P3")

    def test_polygon_label_stays_inside_irregular_piece(self):
        polygon = np.float32((
            (171.6, 57.8), (105.0, 59.6),
            (105.5, 166.0), (174.3, 120.5),
        ))
        label = "P2"
        scale = 0.65
        thickness = 2

        origin = polygon_label_origin(
            polygon, label, scale=scale, thickness=thickness,
        )
        size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness,
        )
        left, bottom = origin
        corners = (
            (left, bottom - size[1]),
            (left + size[0], bottom - size[1]),
            (left + size[0], bottom + baseline),
            (left, bottom + baseline),
        )

        self.assertTrue(all(
            cv2.pointPolygonTest(polygon, corner, False) >= 0
            for corner in corners
        ))

    def test_orientation_marker_tracks_the_same_corner_after_rotation(self):
        square = np.float32(((40, 40), (100, 40), (100, 100), (40, 100)))
        rotated = pose(square, 90, (180, 180))

        source_marker = orientation_marker_point(square)
        target_marker = orientation_marker_point(rotated)

        self.assertGreater(
            cv2.pointPolygonTest(square, source_marker, False), 0)
        self.assertGreater(
            cv2.pointPolygonTest(rotated, target_marker, False), 0)
        self.assertLess(source_marker[0], 70)
        self.assertGreater(target_marker[0], 180)

    def test_detect_green_a4_and_validate_cached_warp(self):
        rgb = legacy.synthetic_frame(scattered=True)

        quad, homography = legacy.detect_a4(rgb)
        rectified = legacy.warp_a4(rgb, homography)

        self.assertIsNotNone(homography)
        self.assertEqual(quad.shape, (4, 2))
        self.assertTrue(legacy.cached_a4_is_valid(rectified))

    def test_black_paper_is_not_accepted_as_green_a4(self):
        rgb = legacy.synthetic_frame(scattered=True)
        green = np.full(rgb.shape, (128, 160, 118), dtype=np.uint8)
        is_green = np.all(rgb == green, axis=2)
        rgb[is_green] = 35

        _quad, homography = legacy.detect_a4(rgb)

        self.assertIsNone(homography)

    def test_poker_black_print_stays_inside_piece_on_green_a4(self):
        image = np.full(
            (legacy.WARP_H, legacy.WARP_W, 3),
            (128, 160, 118), dtype=np.uint8,
        )
        cards = (
            np.int32(((40, 70), (140, 70), (140, 170), (40, 170))),
            np.int32(((210, 90), (330, 90), (330, 210), (210, 210))),
        )
        cv2.fillPoly(image, cards, (220, 220, 215))
        cv2.circle(image, (90, 120), 20, (20, 20, 20), -1)
        cv2.line(image, (300, 120), (330, 150), (15, 15, 15), 10)

        pieces, mask = detect_poker_pieces(image)

        self.assertEqual(len(pieces), 2)
        self.assertEqual(int(mask[20, 20]), 0)
        self.assertTrue(all(len(piece) == 4 for piece in pieces))

    def test_mode_specific_interaction_controls(self):
        controller = MergedController()
        self.assertEqual(BUTTON_LABELS, ("TASK2 WHITE", "TASK2 POKER"))
        self.assertEqual(controller.control_labels,
                         ("FIND A4", "DETECT", "SOLVE", "RESET"))
        controller.select_mode(1)
        self.assertEqual(controller.control_labels,
                         ("FIND A4", "DETECT", "SOLVE", "RESET"))

    def test_touch_router_triggers_mode_and_control_on_release(self):
        router = TouchRouter()
        self.assertIsNone(router.update_values(30, 20, True))
        self.assertEqual(router.update_values(0, 0, False), ("mode", 0))
        self.assertIsNone(router.update_values(620, 20, True))
        self.assertEqual(router.update_values(0, 0, False), ("mode", 1))
        self.assertIsNone(router.update_values(250, 450, True))
        self.assertEqual(router.update_values(0, 0, False),
                         ("control", 1))

    def test_a4_geometry_includes_split_line_and_piece_overlays(self):
        frame = legacy.synthetic_frame(scattered=True)
        controller = MergedController()
        controller.select_mode(0)
        controller.handle_control(0, frame)
        self.assertIsNotNone(controller.homography)
        self.assertIsNotNone(controller.split_camera)
        controller.handle_control(2, frame)
        self.assertGreater(len(controller.detected_pieces), 0)
        self.assertEqual(len(controller.detected_camera),
                         len(controller.detected_pieces))
        rendered = draw_ui(frame, controller, 0.0)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertTrue(np.any(rendered != frame))
        preview_x = 430
        self.assertGreater(np.count_nonzero(
            rendered[90:390, preview_x:] != frame[90:390, preview_x:]), 1000)
        cached = controller._preview_cache
        draw_ui(frame, controller, 0.0)
        self.assertIs(controller._preview_cache, cached)

    def test_piece_detection_rejects_small_blobs_and_thin_filaments(self):
        rectified = np.zeros(
            (legacy.WARP_H, legacy.WARP_W, 3), dtype=np.uint8)
        valid_shapes = (
            np.int32(((55, 80), (145, 92), (130, 165), (48, 150))),
            np.int32(((225, 105), (315, 80), (350, 155), (255, 180))),
            np.int32(((80, 350), (250, 365), (105, 390))),
        )
        for polygon in valid_shapes:
            cv2.fillPoly(rectified, [polygon], (245, 245, 245))
        cv2.fillPoly(rectified, [np.int32(
            ((18, 40), (30, 40), (43, 245), (31, 245)))],
            (245, 245, 245))
        cv2.rectangle(rectified, (350, 300), (358, 308),
                      (245, 245, 245), -1)

        pieces, _binary, _timings = legacy.detect_pieces(rectified)
        gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
        fast_pieces, _fast_binary, _fast_timings = (
            legacy.detect_pieces_fast(gray, 0))

        self.assertEqual(len(pieces), len(valid_shapes))
        self.assertEqual(len(fast_pieces), len(valid_shapes))
        detected_areas = sorted(abs(cv2.contourArea(piece))
                                for piece in pieces)
        expected_areas = sorted(abs(cv2.contourArea(piece))
                                for piece in valid_shapes)
        np.testing.assert_allclose(detected_areas, expected_areas,
                                   rtol=0.08, atol=25)

    def test_piece_detection_keeps_small_and_near_border_fragments(self):
        rectified = np.zeros(
            (legacy.WARP_H, legacy.WARP_W, 3), dtype=np.uint8)
        small_near_border = np.int32(
            ((7, 90), (37, 100), (14, 125)))
        narrow_fragment = np.int32(
            ((80, 180), (86, 180), (91, 240), (85, 240)))
        expected = (small_near_border, narrow_fragment)
        for polygon in expected:
            cv2.fillPoly(rectified, [polygon], (245, 245, 245))
        cv2.circle(rectified, (200, 300), 3, (245, 245, 245), -1)

        pieces, _binary, _timings = legacy.detect_pieces(rectified)
        gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
        fast_pieces, _fast_binary, _fast_timings = (
            legacy.detect_pieces_fast(gray, 0))

        self.assertEqual(len(pieces), len(expected))
        self.assertEqual(len(fast_pieces), len(expected))

    def test_piece_detection_rejects_a4_boundary_contours(self):
        rectified = np.zeros(
            (legacy.WARP_H, legacy.WARP_W, 3), dtype=np.uint8)
        valid_shapes = (
            np.int32(((55, 80), (145, 92), (130, 165), (48, 150))),
            np.int32(((225, 105), (315, 80), (350, 155), (255, 180))),
            np.int32(((80, 350), (250, 365), (105, 390))),
        )
        for polygon in valid_shapes:
            cv2.fillPoly(rectified, [polygon], (245, 245, 245))
        cv2.line(rectified, (75, 2), (417, 2), (245, 245, 245), 14)
        cv2.line(rectified, (417, 2), (417, 585), (245, 245, 245), 14)

        pieces, _binary, _timings = legacy.detect_pieces(rectified)
        gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
        fast_pieces, _fast_binary, _fast_timings = (
            legacy.detect_pieces_fast(gray, 0))

        self.assertEqual(len(pieces), len(valid_shapes))
        self.assertEqual(len(fast_pieces), len(valid_shapes))

    def test_piece_detection_handles_dim_white_on_green_paper(self):
        rectified = np.full(
            (legacy.WARP_H, legacy.WARP_W, 3),
            (142, 176, 126), dtype=np.uint8,
        )
        expected = (
            np.int32(((35, 55), (120, 70), (105, 155), (25, 135))),
            np.int32(((190, 45), (315, 60), (295, 145), (175, 125))),
            np.int32(((50, 255), (165, 235), (145, 350), (35, 330))),
            np.int32(((230, 260), (365, 245), (350, 355), (215, 365))),
        )
        for polygon in expected:
            # These low-saturation pieces are dimmer than WHITE_THRESHOLD,
            # matching the exposure in the supplied green-A4 photos.
            cv2.fillPoly(rectified, [polygon], (154, 153, 158))

        pieces, _binary, _timings = legacy.detect_pieces(rectified)

        self.assertEqual(len(pieces), len(expected))

    def test_a4_detection_recovers_from_a_desaturated_edge_shadow(self):
        paper = np.full(
            (legacy.WARP_H, legacy.WARP_W, 3),
            (142, 176, 126), dtype=np.uint8,
        )
        cv2.fillPoly(paper, [np.int32((
            (legacy.WARP_W - 1, 115),
            (legacy.WARP_W - 1, 500),
            (245, 385),
            (275, 205),
        ))], (145, 148, 132))

        raw = np.full((legacy.CAM_H, legacy.CAM_W, 3), 205, np.uint8)
        quad = np.float32(((145, 25), (475, 48), (545, 455), (80, 430)))
        destination = np.float32((
            (0, 0), (legacy.WARP_W - 1, 0),
            (legacy.WARP_W - 1, legacy.WARP_H - 1),
            (0, legacy.WARP_H - 1),
        ))
        inverse = cv2.getPerspectiveTransform(destination, quad)
        projected = cv2.warpPerspective(
            paper, inverse, (legacy.CAM_W, legacy.CAM_H))
        paper_mask = cv2.warpPerspective(
            np.full((legacy.WARP_H, legacy.WARP_W), 255, np.uint8),
            inverse, (legacy.CAM_W, legacy.CAM_H))
        raw[paper_mask > 0] = projected[paper_mask > 0]

        _quad, homography = legacy.detect_a4(raw)

        self.assertIsNotNone(homography)

    def test_task2_modes_show_detect_solve_and_reset_results(self):
        frame = legacy.synthetic_frame(scattered=True)
        for mode in (0, 1):
            with self.subTest(mode=mode):
                controller = MergedController()
                controller.select_mode(mode)
                controller.handle_control(0, frame)
                controller.handle_control(1, frame)
                self.assertEqual(controller.stage, "DETECTED")
                self.assertGreater(len(controller.detected_camera), 0)
                self.assertEqual(controller.target_camera, [])

                controller.handle_control(2, frame)
                self.assertEqual(controller.stage, "SOLVED")
                self.assertEqual(len(controller.target_camera),
                                 len(controller.detected_camera))

                controller.handle_control(3, frame)
                self.assertEqual(controller.stage, "A4 LOCKED")
                self.assertEqual(controller.detected_camera, [])
                self.assertEqual(controller.target_camera, [])
                self.assertIsNotNone(controller.homography)

    def test_solve_reuses_detect_stage_snapshot_when_supported(self):
        frame = legacy.synthetic_frame(scattered=True)
        solver = ReuseSolver()
        controller = MergedController(solvers=(solver, solver))
        controller.select_mode(1)
        controller.handle_control(0, frame)
        controller.detected_pieces = [np.float64(
            ((10, 10), (30, 10), (30, 30), (10, 30)))]
        controller.diagnostics = {"piece_binary": np.ones((2, 2), np.uint8)}
        controller.stage = "DETECTED"
        with patch("main.wall_time.perf_counter", side_effect=(10.0, 10.456)):
            controller.handle_control(2, frame)
        self.assertTrue(solver.reused)
        self.assertEqual(solver.reused_piece_count, 1)
        self.assertAlmostEqual(controller.solve_time_s, 0.456)

        controller.reset()
        self.assertIsNone(controller.solve_time_s)

    def test_task2_reuses_locked_a4_without_cache_revalidation(self):
        frame = legacy.synthetic_frame(scattered=True)
        controller = MergedController()
        controller.select_mode(1)
        controller.handle_control(0, frame)

        with patch.object(legacy, "cached_a4_is_valid", return_value=False):
            controller.handle_control(1, frame)

        self.assertEqual(controller.stage, "DETECTED")
        self.assertNotEqual(controller.message, "A4 CACHE INVALID")

    def test_device_screen_text_is_ascii_only(self):
        self.assertTrue(all(label.isascii() for label in BUTTON_LABELS))
        self.assertEqual(screen_text("识别失败"), "SOLVER ERROR")
        self.assertEqual(screen_text("A4 NOT FOUND"), "A4 NOT FOUND")

    def test_uart_pair_includes_clockwise_degree_between_pixel_lines(self):
        frame = encode_position_pair(
            (261.9, 272.8), -90.0, (320.7, 300.4))
        self.assertEqual(
            frame, "gre:(261,272)\ndeg:(270)\nred:(320,300)\n")
        serial = SerialCapture()
        frames = PositionSender(serial=serial).send((
            ((261.9, 272.8), -90.0, (320.7, 300.4)),
        ))
        self.assertEqual(frames, [frame])
        self.assertEqual(serial.frames, [frame])

    def test_uart_skips_pairs_outside_calibrated_a4_pixel_range(self):
        serial = SerialCapture()
        frames = PositionSender(serial=serial).send((
            ((26, 272), 90, (320, 300)),
            ((261, 272), 90, (496, 300)),
        ))
        self.assertEqual(frames, [])
        self.assertEqual(serial.frames, [])

    def test_uart_over_frame_has_required_newline(self):
        serial = SerialCapture()
        sender = PositionSender(serial=serial)

        self.assertEqual(sender.send_over(), "<over>\r\n")
        self.assertEqual(OVER_FRAME, "<over>\r\n")
        self.assertEqual(serial.frames, ["<over>\r\n"])

    def test_uart_ack_parser_accepts_fragments_and_drops_duplicate_ack(self):
        serial = SerialCapture((b"noise<o", b"k>\r\n<ok>\r\n"))
        sender = PositionSender(serial=serial)

        self.assertFalse(sender.poll_ack())
        self.assertTrue(sender.poll_ack())
        self.assertFalse(sender.poll_ack())

    def test_uart_discards_ack_retries_accumulated_during_recognition(self):
        serial = SerialCapture((b"<ok>\r\n", b"<ok>\r\n", b"<ok>\r\n"))
        sender = PositionSender(serial=serial)

        self.assertTrue(sender.poll_ack())
        sender.discard_input()

        self.assertFalse(sender.poll_ack())

    def test_successful_solve_waits_for_ack_before_sending(self):
        frame = legacy.synthetic_frame(scattered=True)
        sender = SenderCapture()
        solvers = (SolverCapture(), SolverCapture(), SolverCapture())
        controller = MergedController(solvers=solvers, sender=sender)
        controller.select_mode(1)
        controller.handle_control(0, frame)
        controller.handle_control(2, frame)
        self.assertEqual(controller.stage, "SOLVED")
        self.assertEqual(controller.message, "WAIT ACK 1")
        self.assertEqual(sender.position_pairs, [])

        self.assertTrue(controller.handle_ack(frame))
        self.assertEqual(controller.stage, "SENT")
        self.assertEqual(controller.message, "UART SENT 1/1")
        self.assertEqual(len(sender.position_pairs), 1)
        self.assertEqual(len(controller.sent_frames), 1)
        self.assertTrue(controller.sent_frames[0].startswith("gre:("))
        self.assertIn("\ndeg:(3)\n", controller.sent_frames[0])
        self.assertIn("\nred:(", controller.sent_frames[0])
        action = controller.actions[0]
        expected = legacy.project_to_camera(np.float64((
            (action.pick_x / legacy.MM_PER_PIXEL,
             action.pick_y / legacy.MM_PER_PIXEL),
            (action.place_x / legacy.MM_PER_PIXEL,
             action.place_y / legacy.MM_PER_PIXEL),
        )), controller.inverse_homography)
        np.testing.assert_allclose(sender.position_pairs[0][0], expected[0])
        self.assertEqual(sender.position_pairs[0][1], 3)
        np.testing.assert_allclose(sender.position_pairs[0][2], expected[1])

    def test_first_ack_runs_pipeline_then_each_ack_sends_one_piece(self):
        frame = legacy.synthetic_frame(scattered=True)
        sender = SenderCapture()
        solver = TwoActionSolver()
        controller = MergedController(
            solvers=(solver, solver), sender=sender)
        controller.select_mode(1)

        self.assertTrue(controller.handle_ack(frame))
        self.assertEqual(solver.solve_count, 1)
        self.assertEqual(len(sender.position_pairs), 1)
        self.assertEqual(controller.message, "UART SENT 1/2")

        self.assertTrue(controller.handle_ack(frame))
        self.assertEqual(solver.solve_count, 1)
        self.assertEqual(len(sender.position_pairs), 2)
        self.assertEqual(controller.message, "UART SENT 2/2")

        self.assertTrue(controller.handle_ack(frame))
        self.assertEqual(solver.solve_count, 2)
        self.assertEqual(len(sender.position_pairs), 3)
        self.assertEqual(controller.message, "UART SENT 1/2")

    def test_uart_poll_drives_automatic_pipeline_end_to_end(self):
        frame = legacy.synthetic_frame(scattered=True)
        serial = SerialCapture((b"<ok>\r\n",))
        sender = PositionSender(serial=serial)
        solver = SolverCapture()
        controller = MergedController(
            solvers=(solver, solver), sender=sender)
        controller.select_mode(1)

        self.assertTrue(controller.process_uart(frame))
        self.assertEqual(solver.solve_count, 1)
        self.assertEqual(len(serial.frames), 1)
        self.assertTrue(serial.frames[0].startswith("gre:("))
        self.assertIn("\ndeg:(3)\n", serial.frames[0])
        self.assertIn("\nred:(", serial.frames[0])
        self.assertEqual(controller.message, "UART SENT 1/1")

    def test_uart_ack_never_changes_touch_selected_mode(self):
        frame = legacy.synthetic_frame(scattered=True)
        for selected_mode in range(2):
            serial = SerialCapture((b"<ok>\r\n",))
            controller = MergedController(sender=PositionSender(serial=serial))
            controller.select_mode(selected_mode)
            controller.pending_position_pairs = [
                ((261, 272), 270, (320, 300)),
            ]

            self.assertTrue(controller.process_uart(frame))
            self.assertEqual(controller.mode, selected_mode)

    def test_every_mode_repeats_over_after_all_piece_frames(self):
        frame = legacy.synthetic_frame(scattered=True)
        pair = ((261, 272), 270, (320, 300))

        for selected_mode in range(2):
            serial = SerialCapture((b"<ok>\r\n",))
            controller = MergedController(
                sender=PositionSender(serial=serial))
            controller.select_mode(selected_mode)
            controller.pending_position_pairs = [pair]

            self.assertTrue(controller.process_uart(frame))
            self.assertEqual(len(serial.frames), 1)
            self.assertTrue(serial.frames[0].startswith("gre:("))

            self.assertTrue(controller.process_uart(frame))
            self.assertTrue(controller.process_uart(frame))
            self.assertEqual(
                serial.frames[-2:], ["<over>\r\n", "<over>\r\n"])

            controller.select_mode((selected_mode + 1) % 2)
            self.assertFalse(controller.process_uart(frame))
            self.assertEqual(serial.frames[-1], "<over>\r\n")

    def test_first_ack_automates_poker_mode(self):
        frame = legacy.synthetic_frame(scattered=True)
        sender = SenderCapture()
        solver = TwoActionSolver()
        controller = MergedController(
            solvers=(solver, solver), sender=sender)
        controller.select_mode(1)

        self.assertTrue(controller.handle_ack(frame))
        self.assertEqual(solver.solve_count, 1)
        self.assertEqual(len(sender.position_pairs), 1)
        self.assertEqual(controller.message, "UART SENT 1/2")

    def test_ack_failure_sends_no_placeholder_and_waits_for_retry(self):
        frame = legacy.synthetic_frame(scattered=True)
        sender = SenderCapture()
        solver = FailingSolver()
        controller = MergedController(
            solvers=(solver, solver), sender=sender)
        controller.select_mode(1)

        self.assertFalse(controller.handle_ack(frame))
        self.assertEqual(controller.stage, "ERROR")
        self.assertEqual(sender.position_pairs, [])
        self.assertEqual(controller.sent_frames, [])

    def test_failed_solver_does_not_send_uart_actions(self):
        frame = legacy.synthetic_frame(scattered=True)
        sender = SenderCapture()
        solvers = (FailingSolver(), FailingSolver())
        controller = MergedController(solvers=solvers, sender=sender)
        controller.select_mode(1)
        controller.handle_control(0, frame)
        controller.handle_control(2, frame)
        self.assertEqual(controller.stage, "ERROR")
        self.assertEqual(sender.position_pairs, [])
        self.assertEqual(controller.sent_frames, [])

    def test_solver_without_a_complete_timeout_candidate_sends_no_uart_actions(self):
        frame = legacy.synthetic_frame(scattered=True)
        sender = SenderCapture()
        solver = TimeoutSolver()
        controller = MergedController(
            solvers=(solver, solver), sender=sender)
        controller.select_mode(1)
        controller.handle_control(0, frame)

        controller.handle_control(2, frame)

        self.assertEqual(controller.stage, "ERROR")
        self.assertEqual(controller.message, "SOLVE TIMEOUT 80S")
        self.assertEqual(controller.actions, [])
        self.assertEqual(controller.pending_position_pairs, [])
        self.assertEqual(sender.position_pairs, [])

    def test_task2_timeout_uses_best_candidate_even_below_quality_gate(self):
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        solver = Task2WhiteSolver()
        pieces = [np.float64(((100, 180), (180, 180), (140, 250)))]

        def fake_solve(pieces, _paper, **options):
            options["candidate_recorder"](
                [np.eye(3) for _piece in pieces], (), 0.20, (0.1,))
            raise geometry.SolveTimeoutError()

        with patch.object(geometry, "solve", side_effect=fake_solve):
            actions, diagnostics = solver.solve_detected(rectified, pieces)

        self.assertEqual(len(actions), 1)
        self.assertEqual(diagnostics["fill_ratio"], 0.20)
        self.assertTrue(diagnostics["timed_out"])
        self.assertTrue(diagnostics["topology_path"].endswith(
            "_timeout_best"))

    def test_task2_timeout_uses_texture_within_poker_fill_window(self):
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        pieces = [np.float64(((100, 180), (180, 180), (140, 250)))]
        high_fill = [geometry.rigid(0.0, 20, 200)]
        continuous_seam = [geometry.rigid(0.0, 120, 250)]
        low_fill = [geometry.rigid(0.0, 220, 300)]

        def fake_solve(_pieces, _paper, **options):
            record = options["candidate_recorder"]
            record(high_fill, (), 0.94, (1.0, 0.80), (70.0, 100.0))
            record(continuous_seam, (), 0.93, (2.0, 0.10),
                   (70.0, 100.0))
            record(low_fill, (), 0.90, (0.5, 0.01), (70.0, 100.0))
            raise geometry.SolveTimeoutError()

        with patch.object(geometry, "solve", side_effect=fake_solve):
            actions, diagnostics = Task2WhiteSolver().solve_detected(
                rectified, pieces,
                solve_options={
                    "finalist_max_fill_loss": 0.015,
                    "max_anchor_orders": 1,
                },
            )

        self.assertEqual(len(actions), 1)
        self.assertAlmostEqual(diagnostics["fill_ratio"], 0.93)
        self.assertTrue(diagnostics["timed_out"])
        np.testing.assert_allclose(
            diagnostics["transforms"][0], continuous_seam[0])

    def test_task2_white_returns_millimetre_actions(self):
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        pieces = (
            pose(np.float32(((0, 0), (90, 0), (90, 100), (0, 100))),
                 18, (95, 120)),
            pose(np.float32(((0, 0), (90, 0), (90, 100), (0, 100))),
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

    def test_task2_white_is_invariant_to_detected_piece_order(self):
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        pieces = (
            np.float64((
                (192.347, 36.400), (188.767, 77.662),
                (264.904, 68.866), (230.563, 38.880),
            )),
            np.float64((
                (379.788, 56.812), (319.583, 46.071),
                (284.707, 246.704), (331.569, 201.502),
            )),
            np.float64((
                (273.293, 83.386), (151.593, 117.485),
                (186.619, 274.870),
            )),
            np.float64((
                (74.551, 88.231), (54.074, 91.970),
                (85.585, 249.712), (113.994, 149.824),
            )),
        )
        solver = Task2WhiteSolver()

        for order in itertools.permutations(range(len(pieces))):
            with self.subTest(order=order):
                ordered = [pieces[index] for index in order]
                actions, diagnostics = solver.solve_detected(
                    rectified, ordered,
                )

                self.assertEqual([action.piece_id for action in actions],
                                 list(range(len(pieces))))
                for actual, expected in zip(
                        diagnostics["pieces"], ordered):
                    np.testing.assert_array_equal(actual, expected)
                self.assertGreaterEqual(
                    diagnostics["fill_ratio"],
                    geometry.config.MIN_RECTANGLE_FILL,
                )

    def test_task2_white_solves_real_multi_partial_piece_set(self):
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        pieces = (
            np.float64((
                (29.000, 41.500), (77.500, 139.300),
                (128.900, 139.400), (153.900, 86.400),
            )),
            np.float64((
                (32.400, 308.300), (70.100, 316.900),
                (179.400, 196.300), (38.600, 179.500),
            )),
            np.float64((
                (172.500, 324.100), (66.700, 347.800),
                (71.800, 426.300), (141.400, 410.000),
            )),
            np.float64((
                (170.500, 427.100), (69.400, 470.400),
                (131.500, 553.000),
            )),
        )

        actions, diagnostics = Task2WhiteSolver().solve_detected(
            rectified, pieces,
        )

        self.assertEqual([action.piece_id for action in actions],
                         list(range(len(pieces))))
        self.assertEqual(diagnostics["topology_path"], "multi_partial")
        self.assertGreaterEqual(
            diagnostics["fill_ratio"],
            geometry.config.MIN_RECTANGLE_FILL,
        )
        self.assertEqual(
            sum(tuple(match[5:]) != (0.0, 1.0, 0.0, 1.0)
                for match in diagnostics["matches"]),
            2,
        )

    def test_task2_piture1_contours_survive_pose_and_subpixel_jitter(self):
        captured = (
            np.float64(((389.584, 40.878), (320.820, 109.015),
                        (319.918, 181.550), (357.678, 218.082))),
            np.float64(((201.125, 45.649), (195.264, 99.938),
                        (280.214, 253.096), (292.042, 54.306))),
            np.float64(((117.967, 101.268), (129.523, 200.606),
                        (189.842, 244.159), (173.105, 94.285))),
            np.float64(((86.503, 98.468), (28.204, 186.157),
                        (117.203, 241.550))),
        )
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        solver = Task2WhiteSolver()

        for seed in range(4):
            random = np.random.default_rng(seed)
            pieces = []
            for polygon in captured:
                center = polygon.mean(axis=0)
                angle = random.uniform(-math.pi, math.pi)
                rotation = np.float64((
                    (math.cos(angle), -math.sin(angle)),
                    (math.sin(angle), math.cos(angle)),
                ))
                translation = random.uniform((60, 60), (360, 280))
                jitter = random.normal(0.0, 0.35, polygon.shape)
                pieces.append(
                    (polygon - center).dot(rotation.T)
                    + translation + jitter)

            with self.subTest(seed=seed):
                actions, diagnostics = solver.solve_detected(
                    rectified, pieces)
                self.assertEqual(len(actions), 4)
                self.assertEqual(
                    diagnostics["topology_path"], "multi_partial")
                self.assertGreaterEqual(
                    diagnostics["fill_ratio"],
                    geometry.config.MIN_RECTANGLE_FILL,
                )

    def test_dynamic_piece_scaling_preserves_centres_and_shape(self):
        pieces = [
            np.float64(((0, 0), (90, 0), (90, 60), (0, 60))),
            np.float64(((200, 0), (300, 0), (300, 60), (200, 60))),
        ]
        match = (0.10, 0, 0, 1, 0, 0.0, 1.0, 0.0, 1.0)

        scales = geometry._dynamic_piece_scales(pieces, (match,))
        scaled = geometry._scale_pieces_about_centres(pieces, scales)

        self.assertAlmostEqual(scales[0], geometry.config.DYNAMIC_SCALE_MAX)
        self.assertAlmostEqual(scales[1], geometry.config.DYNAMIC_SCALE_MIN)
        for original, adjusted in zip(pieces, scaled):
            np.testing.assert_allclose(
                adjusted.mean(axis=0), original.mean(axis=0))
            original_edges = np.linalg.norm(
                np.roll(original, -1, axis=0) - original, axis=1)
            adjusted_edges = np.linalg.norm(
                np.roll(adjusted, -1, axis=0) - adjusted, axis=1)
            np.testing.assert_allclose(
                adjusted_edges / original_edges,
                adjusted_edges[0] / original_edges[0])

    def test_dynamic_scale_recovers_a_four_percent_contour_size_error(self):
        pieces = [
            np.float64(((389.584, 40.878), (320.820, 109.015),
                        (319.918, 181.550), (357.678, 218.082))),
            np.float64(((201.125, 45.649), (195.264, 99.938),
                        (280.214, 253.096), (292.042, 54.306))),
            np.float64(((117.967, 101.268), (129.523, 200.606),
                        (189.842, 244.159), (173.105, 94.285))),
            np.float64(((86.503, 98.468), (28.204, 186.157),
                        (117.203, 241.550))),
        ]
        center = pieces[0].mean(axis=0)
        pieces[0] = center + (pieces[0] - center) * 1.04
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)

        actions, diagnostics = Task2WhiteSolver().solve_detected(
            rectified, pieces)

        self.assertEqual(len(actions), 4)
        self.assertEqual(
            diagnostics["topology_path"], "scaled_multi_partial")
        self.assertGreaterEqual(
            diagnostics["fill_ratio"], geometry.config.MIN_RECTANGLE_FILL)
        self.assertLess(diagnostics["piece_scales"][0], 1.0)
        self.assertTrue(any(
            value > 1.0 for value in diagnostics["piece_scales"][1:]))
        self.assertTrue(all(
            np.allclose(transform[:2, :2].T @ transform[:2, :2], np.eye(2))
            for transform in diagnostics["transforms"]
        ))

    def test_failed_topologies_use_only_bounded_dynamic_scale_fallback(self):
        pieces = [
            np.float64(((10, 10), (70, 10), (70, 60), (10, 60))),
            np.float64(((110, 10), (175, 10), (175, 60), (110, 60))),
        ]
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        match = ((0.0, 0, 0, 1, 0, 0.0, 1.0, 0.0, 1.0),)
        calls = []

        def fake_solve(_pieces, _paper, **options):
            calls.append(options)
            if not options.get("dynamic_scaling"):
                raise RuntimeError("force bounded scale fallback")
            options["result_metadata"]["piece_scales"] = (1.04, 0.96)
            return [np.eye(3), np.eye(3)], match, 0.90

        with (
            patch.object(geometry, "solve", side_effect=fake_solve),
            patch.object(
                geometry, "_stable_piece_orders",
                side_effect=lambda _pieces: iter(([0, 1],))),
        ):
            actions, diagnostics = Task2WhiteSolver().solve_detected(
                rectified, pieces)

        self.assertEqual(len(actions), 2)
        self.assertEqual(diagnostics["topology_path"], "scaled_standard")
        self.assertEqual(diagnostics["piece_scales"], (1.04, 0.96))
        self.assertEqual(
            [call.get("dynamic_scaling", False) for call in calls],
            [False, False, True])
        self.assertTrue(all(
            "complete_fallback" not in call for call in calls))
        self.assertEqual(
            calls[0]["fast_max_topologies"],
            geometry.config.FAST_STANDARD_MAX_TOPOLOGIES)
        self.assertEqual(
            calls[1]["fast_max_topologies"],
            geometry.config.FAST_MULTI_PARTIAL_MAX_TOPOLOGIES)
        self.assertEqual(
            calls[2]["fast_max_topologies"],
            geometry.config.DYNAMIC_SCALE_STANDARD_MAX_TOPOLOGIES)
        self.assertEqual(
            calls[2]["fast_full_candidates"],
            geometry.config.DYNAMIC_SCALE_FULL_CANDIDATES)
        self.assertTrue(all(
            np.allclose(transform[:2, :2].T @ transform[:2, :2], np.eye(2))
            for transform in diagnostics["transforms"]
        ))

    def test_task2_quality_does_not_prefer_fixed_10_by_6_aspect(self):
        ten_by_six = np.float64((
            (0, 0), (200, 0), (200, 120), (0, 120),
        ))
        other_valid_ratio = np.float64((
            (0, 0), (192, 0), (192, 125), (0, 125),
        ))

        first_score = geometry._assembly_quality(
            [ten_by_six], (), 0.0, return_metrics=True,
        )[0]
        second_score = geometry._assembly_quality(
            [other_valid_ratio], (), 0.0, return_metrics=True,
        )[0]

        self.assertAlmostEqual(first_score, second_score, delta=100.0)

    def test_task2_final_quality_uses_contest_dimension_ranges(self):
        self.assertTrue(geometry._target_dimensions_valid((50.0, 90.0)))
        self.assertTrue(geometry._target_dimensions_valid((90.0, 120.0)))
        self.assertTrue(geometry._target_dimensions_valid((60.0, 90.0)))
        self.assertFalse(geometry._target_dimensions_valid((40.0, 90.0)))
        self.assertFalse(geometry._target_dimensions_valid((50.0, 130.0)))

    def test_task2_target_leaves_configured_gap_between_pieces(self):
        pieces = (
            np.float64(((0, 0), (70, 0), (70, 60), (0, 60))),
            np.float64(((0, 0), (70, 0), (70, 60), (0, 60))),
        )
        assembled = (
            np.eye(3),
            geometry.rigid(0.0, 70.0, 0.0),
        )
        paper = np.int32((((0, 0),), ((419, 0),),
                          ((419, 593),), ((0, 593),)))

        transforms = geometry._target_transform(pieces, assembled, paper)
        targets = [geometry.apply_h(piece, transform)
                   for piece, transform in zip(pieces, transforms)]
        gap_pixels = geometry._minimum_pairwise_gap(targets)

        self.assertAlmostEqual(
            gap_pixels * geometry.config.A4_MM_PER_PIXEL,
            geometry.config.TARGET_PIECE_GAP_MM,
            delta=0.05,
        )
        target_centers_mm = [
            geometry.apply_h(
                np.asarray((piece.mean(axis=0),)), transform,
            )[0] * geometry.config.A4_MM_PER_PIXEL
            for piece, transform in zip(pieces, transforms)
        ]
        self.assertAlmostEqual(
            abs(target_centers_mm[1][0] - target_centers_mm[0][0]),
            35.0 + geometry.config.TARGET_PIECE_GAP_MM,
            delta=0.05,
        )

        target_minimum_y = min(float(polygon[:, 1].min())
                               for polygon in targets)
        _, paper_y, _, paper_height = cv2.boundingRect(paper)
        paper_axis_y = paper_y + paper_height * 0.5
        self.assertAlmostEqual(
            (target_minimum_y - paper_axis_y)
            * geometry.config.A4_MM_PER_PIXEL,
            geometry.config.TARGET_AXIS_CLEARANCE_MM,
            delta=0.05,
        )

    def test_fast_topology_stops_at_first_high_fill_candidate(self):
        pieces = [
            np.float64(((0, 0), (40, 0), (0, 30))),
            np.float64(((80, 0), (120, 0), (120, 30))),
        ]
        paper = np.int32((((0, 0),), ((419, 0),),
                          ((419, 593),), ((0, 593),)))
        first_matches = ("first",)
        later_matches = ("later",)
        identities = [np.eye(3), np.eye(3)]
        with (
            patch.object(geometry, "_equal_rectangle_transforms",
                         return_value=None),
            patch.object(geometry, "matching_sets",
                         return_value=iter((first_matches, later_matches))),
            patch.object(geometry, "assemble_from_matches", side_effect=(
                (100.0, 0.99, identities),
                (10.0, 0.98, identities),
            )) as assemble,
            patch.object(geometry, "optimize_pose_graph",
                         return_value=identities),
            patch.object(geometry, "_assembly_quality",
                         return_value=(0.0, 0.98, 0.0, (60.0, 100.0))),
            patch.object(geometry, "_target_transform",
                         return_value=identities),
        ):
            _transforms, matches, fill_ratio = geometry.solve(pieces, paper)
        self.assertEqual(matches, first_matches)
        self.assertEqual(assemble.call_count, 1)
        self.assertEqual(fill_ratio, 0.98)

    def test_fast_topology_skips_candidate_that_fails_final_quality(self):
        pieces = [
            np.float64(((0, 0), (40, 0), (0, 30))),
            np.float64(((80, 0), (120, 0), (120, 30))),
        ]
        paper = np.int32((((0, 0),), ((419, 0),),
                          ((419, 593),), ((0, 593),)))
        first_matches = ("overlap",)
        later_matches = ("valid",)
        identities = [np.eye(3), np.eye(3)]
        with (
            patch.object(geometry, "_equal_rectangle_transforms",
                         return_value=None),
            patch.object(geometry, "matching_sets",
                         return_value=iter((first_matches, later_matches))),
            patch.object(geometry, "assemble_from_matches", side_effect=(
                (20.0, 0.95, identities),
                (10.0, 0.94, identities),
            )),
            patch.object(geometry, "optimize_pose_graph",
                         return_value=identities),
            patch.object(geometry, "_assembly_quality", side_effect=(
                (20.0, 0.95, 0.20, (60.0, 100.0)),
                (10.0, 0.94, 0.0, (60.0, 100.0)),
                (10.0, 0.94, 0.0, (60.0, 100.0)),
            )),
            patch.object(geometry, "_target_transform",
                         return_value=identities),
        ):
            _transforms, matches, fill_ratio = geometry.solve(pieces, paper)
        self.assertEqual(matches, later_matches)
        self.assertEqual(fill_ratio, 0.94)

    def test_geometry_does_not_run_a_complete_topology_fallback(self):
        pieces = [
            np.float64(((0, 0), (40, 0), (0, 30))),
            np.float64(((80, 0), (120, 0), (120, 30))),
        ]
        paper = np.int32((((0, 0),), ((419, 0),),
                          ((419, 593),), ((0, 593),)))
        fast_matches = ("invalid-fast",)
        identities = [np.eye(3), np.eye(3)]
        with (
            patch.object(geometry, "_equal_rectangle_transforms",
                         return_value=None),
            patch.object(geometry, "matching_sets",
                         return_value=iter((fast_matches,))) as matching,
            patch.object(geometry, "assemble_from_matches",
                         return_value=(20.0, 0.95, identities)),
            patch.object(geometry, "optimize_pose_graph",
                         return_value=identities),
            patch.object(geometry, "_assembly_quality",
                         return_value=(20.0, 0.95, 0.20,
                                       (60.0, 100.0))),
        ):
            with self.assertRaisesRegex(RuntimeError, "重叠率"):
                geometry.solve(pieces, paper)
        self.assertEqual(matching.call_count, 1)

    def test_texture_finalist_may_trade_at_most_configured_fill_loss(self):
        pieces = [
            np.float64(((0, 0), (40, 0), (0, 30))),
            np.float64(((80, 0), (120, 0), (120, 30))),
        ]
        paper = np.int32((((0, 0),), ((419, 0),),
                          ((419, 593),), ((0, 593),)))
        geometry_best = ("geometry",)
        texture_best = ("texture",)
        identities = [np.eye(3), np.eye(3)]

        def texture_priority(_pieces, matches):
            return 0.30 if matches == geometry_best else 0.10

        with (
            patch.object(geometry, "_equal_rectangle_transforms",
                         return_value=None),
            patch.object(geometry, "matching_sets", return_value=iter((
                geometry_best, texture_best,
            ))),
            patch.object(geometry, "assemble_from_matches", side_effect=(
                (10.0, 0.85, identities),
                (20.0, 0.84, identities),
            )),
            patch.object(geometry, "optimize_pose_graph",
                         return_value=identities),
            patch.object(geometry, "_assembly_quality", side_effect=(
                (10.0, 0.85, 0.0, (60.0, 100.0)),
                (20.0, 0.84, 0.0, (60.0, 100.0)),
                (20.0, 0.84, 0.0, (60.0, 100.0)),
            )),
            patch.object(geometry, "_target_transform",
                         return_value=identities),
        ):
            _transforms, matches, fill_ratio = geometry.solve(
                pieces, paper,
                accept_fast_best=True,
                min_rectangle_fill=0.80,
                topology_priority=texture_priority,
                finalist_count=2,
                finalist_max_fill_loss=0.015,
            )

        self.assertEqual(matches, texture_best)
        self.assertEqual(fill_ratio, 0.84)

    def test_task2_solver_uses_80_second_shared_deadline(self):
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        pieces = [np.float64(((10, 10), (90, 10), (50, 70)))]
        captured = {}

        def fake_solve(_pieces, _paper, **options):
            captured.update(options)
            return [np.eye(3)], (), 1.0

        with (
            patch.object(geometry.wall_time, "perf_counter",
                         return_value=100.0),
            patch.object(geometry, "solve", side_effect=fake_solve),
        ):
            Task2WhiteSolver().solve_detected(rectified, pieces)

        self.assertEqual(geometry.config.SOLVE_TIMEOUT_SECONDS, 80.0)
        self.assertEqual(captured["deadline"], 180.0)
        self.assertNotIn("complete_fallback", captured)
        self.assertEqual(
            captured["fast_max_topologies"],
            geometry.config.FAST_STANDARD_MAX_TOPOLOGIES)

    def test_task2_default_topology_paths_share_one_deadline(self):
        rectified = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        base_piece = np.float64(((10, 10), (70, 10), (40, 60)))
        pieces = [base_piece + np.float64((index * 80, 0))
                  for index in range(4)]
        calls = []

        def fake_solve(_pieces, _paper, **options):
            calls.append(options)
            if len(calls) == 1:
                raise RuntimeError("STANDARD FAILED")
            return [np.eye(3) for _piece in _pieces], (), 0.95

        with (
            patch.object(geometry, "new_solve_deadline", return_value=123.0),
            patch.object(geometry.wall_time, "perf_counter",
                         return_value=100.0),
            patch.object(geometry, "solve", side_effect=fake_solve),
        ):
            _actions, diagnostics = Task2WhiteSolver().solve_detected(
                rectified, pieces,
            )

        self.assertEqual([call["deadline"] for call in calls],
                         [123.0, 123.0])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("cut_mode", calls[0])
        self.assertEqual(calls[1]["cut_mode"], "multi_partial")
        self.assertEqual(diagnostics["topology_path"], "multi_partial")

    def test_geometry_solver_stops_when_deadline_expires(self):
        pieces = [np.float64(((0, 0), (80, 0), (40, 60)))]
        paper = np.int32((((0, 0),), ((419, 0),),
                          ((419, 593),), ((0, 593),)))

        with patch.object(geometry.wall_time, "perf_counter",
                          return_value=10.0):
            with self.assertRaisesRegex(
                    geometry.SolveTimeoutError, "SOLVE TIMEOUT"):
                geometry.solve(pieces, paper, deadline=9.0)

    def test_geometry_records_candidate_before_final_quality_gate(self):
        pieces = [np.float64(((0, 0), (80, 0), (40, 60)))]
        paper = np.int32((((0, 0),), ((419, 0),),
                          ((419, 593),), ((0, 593),)))
        recorded = []

        with (
            patch.object(geometry, "_equal_rectangle_transforms",
                         return_value=None),
            patch.object(geometry, "matching_sets", return_value=[()]),
            patch.object(geometry, "assemble_from_matches",
                         return_value=(0.2, 0.2, [np.eye(3)])),
            patch.object(geometry, "optimize_pose_graph",
                         return_value=[np.eye(3)]),
            patch.object(geometry, "_assembly_quality",
                         return_value=(0.2, 0.2, 0.5, (40.0, 30.0))),
        ):
            with self.assertRaisesRegex(RuntimeError, "填充率"):
                geometry.solve(
                    pieces, paper,
                    candidate_recorder=lambda *candidate: recorded.append(
                        candidate),
                )

        self.assertTrue(recorded)
        self.assertEqual(recorded[-1][2], 0.2)
        self.assertEqual(recorded[-1][4], (40.0, 30.0))

    def test_poker_prefers_white_geometry_when_piece_areas_agree(self):
        white = (
            np.float64(((0, 0), (80, 0), (80, 50), (0, 50))),
            np.float64(((0, 0), (60, 0), (60, 40), (0, 40))),
        )
        poker = (
            white[0] * np.float64((1.02, 1.02)),
            white[1] * np.float64((0.98, 0.98)),
        )
        self.assertTrue(_same_piece_geometry(white, poker))
        self.assertFalse(_same_piece_geometry(white, poker[:1]))

    def test_poker_rejects_white_geometry_with_too_many_edges(self):
        white = (np.float64((
            (0, 0), (80, 0), (80, 50),
            (45, 50), (40, 45), (35, 50), (0, 50),
        )),)
        poker = (np.float64(((0, 0), (80, 0), (80, 50), (0, 50))),)

        self.assertFalse(_same_piece_geometry(white, poker))

    def test_poker_detection_returns_white_mode_geometry_when_usable(self):
        image = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        white = [np.float64(((10, 10), (90, 10), (90, 60), (10, 60)))]
        poker = [np.float64(((11, 10), (91, 10), (91, 61), (11, 61)))]
        white_mask = np.full(image.shape[:2], 255, np.uint8)
        poker_mask = np.zeros(image.shape[:2], np.uint8)
        with (
            patch("solvers.task3_poker._detect_poker_mask",
                  return_value=(poker, poker_mask)),
            patch.object(legacy, "detect_pieces",
                         return_value=(white, white_mask, {})),
        ):
            pieces, mask = detect_poker_pieces(image)
        self.assertIs(pieces, white)
        self.assertIs(mask, white_mask)

    def test_poker_solver_uses_shared_task2_with_texture_and_aspect_gate(self):
        image = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        pieces = [
            np.float64(((20, 20), (80, 20), (80, 100), (20, 100))),
            np.float64(((180, 20), (240, 20), (240, 100), (180, 100))),
        ]
        mask = np.zeros(image.shape[:2], np.uint8)
        expected = (["actions"], {"topology_path": "standard"})
        with patch.object(
                geometry.Task2WhiteSolver, "solve_detected",
                return_value=expected) as solve_detected:
            actual = Task3PokerSolver().solve_detected(image, pieces, mask)
        self.assertIs(actual, expected)
        args, kwargs = solve_detected.call_args
        self.assertEqual(args, (image, pieces, mask))
        self.assertIs(
            kwargs["candidate_validator"], _poker_target_aspect_valid)
        self.assertTrue(kwargs["use_default_search"])
        self.assertTrue(kwargs["solve_options"]["defer_fast_accept"])
        self.assertEqual(kwargs["solve_options"]["finalist_count"], 12)
        self.assertNotIn("return_best_candidate_on_failure", kwargs)
        self.assertTrue(_poker_target_aspect_valid((70.0, 100.0)))
        self.assertFalse(_poker_target_aspect_valid((60.0, 100.0)))

    def test_poker_aspect_gate_rejects_wrong_ratio_candidate(self):
        paper = np.int32((
            ((0, 0),), ((legacy.WARP_W - 1, 0),),
            ((legacy.WARP_W - 1, legacy.WARP_H - 1),),
            ((0, legacy.WARP_H - 1),),
        ))
        pieces = [
            np.float64(((20, 20), (120, 20), (120, 140), (20, 140))),
            np.float64(((240, 300), (340, 300),
                        (340, 420), (240, 420))),
        ]

        geometry.solve(pieces, paper)
        with self.assertRaisesRegex(RuntimeError, "矩形比例"):
            geometry.solve(
                pieces, paper,
                candidate_validator=_poker_target_aspect_valid)

    def test_task3_poker_full_solver_returns_actions(self):
        image = np.zeros((legacy.WARP_H, legacy.WARP_W, 3), np.uint8)
        cv2.rectangle(image, (30, 60), (130, 199), (235, 235, 230), -1)
        cv2.rectangle(image, (220, 250), (320, 389), (235, 235, 230), -1)
        for y in range(60, 200):
            colour = (120 + y % 60, 80 + (y * 2) % 100, 170)
            cv2.line(image, (35, y), (125, y), colour, 1)
            cv2.line(image, (225, y + 190), (315, y + 190), colour, 1)
        cv2.circle(image, (80, 130), 18, (250, 250, 245), -1)
        cv2.circle(image, (270, 320), 18, (250, 250, 245), -1)
        actions, diagnostics = Task3PokerSolver().solve(image)
        self.assertEqual(len(actions), 2)
        self.assertGreater(diagnostics["fill_ratio"], 0.85)

    def test_poker_detection_rejects_exposed_gray_a4_background(self):
        image = np.full(
            (legacy.WARP_H, legacy.WARP_W, 3), 80, dtype=np.uint8,
        )
        shapes = (
            np.int32(((35, 70), (145, 95), (70, 235))),
            np.int32(((205, 70), (325, 70), (325, 155), (270, 125),
                      (205, 155))),
            np.int32(((220, 275), (350, 245), (335, 400))),
        )
        cv2.fillPoly(image, shapes, (235, 235, 230))
        cv2.line(image, (55, 105), (105, 175), (45, 35, 90), 9)
        cv2.circle(image, (275, 105), 18, (170, 35, 45), -1)
        cv2.line(image, (255, 290), (325, 340), (40, 85, 55), 8)

        pieces, mask = detect_poker_pieces(image)

        self.assertEqual(len(pieces), 3)
        self.assertEqual(int(mask[20, 20]), 0)

    def test_poker_detection_does_not_bridge_nearby_colour_prints(self):
        image = np.full(
            (legacy.WARP_H, legacy.WARP_W, 3), 55, dtype=np.uint8,
        )
        cards = (
            np.int32(((35, 60), (125, 60), (125, 145), (35, 145))),
            np.int32(((135, 60), (225, 60), (225, 145), (135, 145))),
            np.int32(((35, 180), (125, 180), (125, 265), (35, 265))),
            np.int32(((135, 180), (225, 180), (225, 265), (135, 265))),
        )
        cv2.fillPoly(image, cards, (225, 225, 220))
        # Saturated dark printing approaches across the black gap.  It is
        # card texture, not evidence that the two physical pieces touch.
        cv2.line(image, (115, 100), (131, 100), (45, 35, 90), 5)
        cv2.line(image, (115, 220), (131, 220), (40, 85, 55), 5)

        pieces, _mask = detect_poker_pieces(image)

        self.assertEqual(len(pieces), 4)
        self.assertTrue(all(len(piece) == 4 for piece in pieces))

    def test_poker_repairs_shallow_triangular_notch(self):
        polygon = np.float64((
            (20, 20), (120, 20), (120, 100),
            (78, 100), (70, 88), (62, 100), (20, 100),
        ))

        repaired = _repair_shallow_concave_notches(polygon)

        self.assertEqual(len(repaired), 4)
        self.assertTrue(np.allclose(
            repaired, ((20, 20), (120, 20), (120, 100), (20, 100)),
        ))

    def test_poker_replaces_every_concave_notch_with_a_straight_edge(self):
        notched_rectangle = np.float32((
            (20, 20), (120, 20), (120, 100),
            (90, 100), (70, 55), (50, 100), (20, 100),
        ))
        placements = ((0, (80, 80)), (17, (180, 150)), (-73, (260, 320)))

        for angle, center in placements:
            with self.subTest(angle=angle, center=center):
                contour = np.round(pose(
                    notched_rectangle, angle, center,
                )).astype(np.int32).reshape(-1, 1, 2)
                polygon = _approximate_poker_piece(contour)

                self.assertIsNotNone(polygon)
                self.assertLessEqual(len(polygon), 5)
                rounded = np.round(polygon).astype(np.int32)
                hull_area = abs(cv2.contourArea(cv2.convexHull(contour)))
                self.assertAlmostEqual(
                    abs(cv2.contourArea(rounded)),
                    hull_area,
                    delta=max(1.0, hull_area * 0.002),
                )

    def test_real_poker_frame_never_emits_more_than_five_edges(self):
        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        image = cv2.cvtColor(cv2.imread(os.path.join(
            fixture_dir, "poker_real_2.png")), cv2.COLOR_BGR2RGB)

        pieces, _mask = detect_poker_pieces(image)

        self.assertEqual(len(pieces), 4)
        self.assertTrue(all(3 <= len(piece) <= 8 for piece in pieces))

    def test_real_poker_frames_are_detected_and_solved(self):
        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        solver = Task3PokerSolver()

        for index in range(1, 5):
            with self.subTest(index=index):
                image = cv2.cvtColor(cv2.imread(os.path.join(
                    fixture_dir, "poker_real_%d.png" % index,
                )), cv2.COLOR_BGR2RGB)
                pieces, mask = detect_poker_pieces(image)
                actions, diagnostics = solver.solve_detected(
                    image, pieces, mask,
                )

                self.assertEqual(len(pieces), 4)
                self.assertEqual(len(actions), 4)
                self.assertGreaterEqual(diagnostics["fill_ratio"], 0.83)

    def test_real_poker_fifth_frame_uses_task2_search(self):
        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        image = cv2.cvtColor(cv2.imread(os.path.join(
            fixture_dir, "poker_real_5.png",
        )), cv2.COLOR_BGR2RGB)
        pieces, mask = detect_poker_pieces(image)

        actions, diagnostics = Task3PokerSolver().solve_detected(
            image, pieces, mask)

        self.assertEqual(len(actions), 4)
        self.assertEqual(diagnostics["topology_path"], "multi_partial")

    def test_real_poker_solver_caches_repeated_edge_alignments(self):
        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        image = cv2.cvtColor(cv2.imread(os.path.join(
            fixture_dir, "poker_real_4.png",
        )), cv2.COLOR_BGR2RGB)
        pieces, mask = detect_poker_pieces(image)

        with patch.object(
                geometry, "align_edge", wraps=geometry.align_edge) as align:
            actions, diagnostics = Task3PokerSolver().solve_detected(
                image, pieces, mask,
            )

        self.assertEqual(len(actions), 4)
        self.assertGreaterEqual(diagnostics["fill_ratio"], 0.83)
        self.assertLessEqual(
            align.call_count,
            2 * geometry.config.MAX_EDGE_CANDIDATES,
        )

    def test_historical_black_a4_photos_are_rejected(self):
        photo_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "piture",
        )
        for index in range(5):
            with self.subTest(index=index):
                raw = cv2.cvtColor(cv2.imread(os.path.join(
                    photo_dir, "%d.jpg" % index,
                )), cv2.COLOR_BGR2RGB)
                _quad, homography = legacy.detect_a4(raw)

                self.assertIsNone(homography)

    def test_poker_detection_rejects_rectified_border_artifact(self):
        image = np.full(
            (legacy.WARP_H, legacy.WARP_W, 3), 45, dtype=np.uint8,
        )
        cards = (
            np.int32(((45, 80), (125, 80), (125, 150), (45, 150))),
            np.int32(((165, 80), (245, 80), (245, 150), (165, 150))),
            np.int32(((45, 210), (125, 210), (125, 280), (45, 280))),
            np.int32(((165, 210), (245, 210), (245, 280), (165, 280))),
        )
        cv2.fillPoly(image, cards, (230, 230, 225))
        # Perspective interpolation can bring the bright area outside the A4
        # into the rectified border.  It must not consume one of four slots.
        cv2.rectangle(image, (0, 0), (300, 45), (235, 235, 235), -1)

        pieces, _mask = detect_poker_pieces(image)

        self.assertEqual(len(pieces), 4)
        self.assertTrue(all(piece[:, 1].min() > legacy.REGION_MARGIN
                            for piece in pieces))


if __name__ == "__main__":
    unittest.main()
