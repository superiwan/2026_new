"""Solver-independent pick/place action contract in physical millimetres."""

import math

import cv2
import numpy as np


def normalize_angle(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def polygon_center(polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    moments = cv2.moments(polygon.astype(np.float32))
    if abs(moments["m00"]) < 1e-9:
        return polygon.mean(axis=0)
    return np.array((moments["m10"] / moments["m00"],
                     moments["m01"] / moments["m00"]), dtype=np.float64)


def polygon_angle(polygon):
    """Return an undirected principal-edge angle in image coordinates."""
    polygon = np.asarray(polygon, dtype=np.float64)
    edges = np.roll(polygon, -1, axis=0) - polygon
    vector = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    return normalize_angle(angle if angle < 90.0 else angle - 180.0)


class PieceAction:
    """One physical pick-to-place command returned by every solver."""

    __slots__ = (
        "piece_id", "pick_x", "pick_y", "pick_angle",
        "place_x", "place_y", "place_angle", "confidence",
    )

    def __init__(self, piece_id, pick_x, pick_y, pick_angle,
                 place_x, place_y, place_angle, confidence=1.0):
        self.piece_id = int(piece_id)
        self.pick_x = float(pick_x)
        self.pick_y = float(pick_y)
        self.pick_angle = normalize_angle(pick_angle)
        self.place_x = float(place_x)
        self.place_y = float(place_y)
        self.place_angle = normalize_angle(place_angle)
        self.confidence = float(confidence)

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self):
        return (
            "PieceAction(piece_id=%d, pick=(%.1f, %.1f, %.1f), "
            "place=(%.1f, %.1f, %.1f), confidence=%.3f)"
            % (self.piece_id, self.pick_x, self.pick_y, self.pick_angle,
               self.place_x, self.place_y, self.place_angle, self.confidence)
        )


def actions_from_transforms(pieces, transforms, mm_per_pixel=0.5,
                            confidence=1.0):
    """Convert rectified-plane transforms to physical millimetre actions."""
    actions = []
    for index, (piece, transform) in enumerate(zip(pieces, transforms)):
        transform = np.asarray(transform, dtype=np.float64)
        pick_center = polygon_center(piece)
        homogeneous = np.array((pick_center[0], pick_center[1], 1.0))
        place_center = transform.dot(homogeneous)
        place_center = place_center[:2] / place_center[2]
        rotation = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        pick_angle = polygon_angle(piece)
        actions.append(PieceAction(
            index,
            pick_center[0] * mm_per_pixel,
            pick_center[1] * mm_per_pixel,
            pick_angle,
            place_center[0] * mm_per_pixel,
            place_center[1] * mm_per_pixel,
            pick_angle + rotation,
            confidence,
        ))
    return actions
