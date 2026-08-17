"""MaixPy YOLO11 adapter for poker corner-mark detection."""

import os

import numpy as np


DEFAULT_MODEL_PATH = "/root/models/poker_corner_yolo11n_640_int8_v2.mud"
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.45
DEFAULT_MAX_DETECTIONS = 8


class CornerMark:
    """Device-independent corner-mark detection result."""

    def __init__(self, x, y, width, height, score, class_id=0):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        self.score = float(score)
        self.class_id = int(class_id)

    @property
    def center(self):
        return (self.x + self.width * 0.5,
                self.y + self.height * 0.5)

    @property
    def confidence(self):
        return self.score

    @property
    def bbox_xyxy(self):
        return (self.x, self.y,
                self.x + self.width, self.y + self.height)

    def as_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "score": self.score,
            "class_id": self.class_id,
        }


def _read_field(obj, *names):
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError("detection has no field %s" % "/".join(names))


class PokerCornerRuntime:
    """Wrap ``maix.nn.YOLO11`` behind a NumPy RGB interface."""

    def __init__(self, detector, image_module, model_path=DEFAULT_MODEL_PATH):
        self.detector = detector
        self.image_module = image_module
        self.model_path = model_path

    @classmethod
    def load(cls, model_path=None, dual_buff=False):
        path = model_path or os.environ.get(
            "POKER_CORNER_MODEL", DEFAULT_MODEL_PATH)
        try:
            from maix import image, nn
        except ImportError as error:
            raise RuntimeError("MaixPy nn/image modules are unavailable") from error
        try:
            detector = nn.YOLO11(path, dual_buff=bool(dual_buff))
        except Exception as error:
            raise RuntimeError(
                "failed to load poker corner model %s: %s" % (
                    path, error)) from error
        return cls(detector, image, path)

    def detect_rgb(self, rectified_rgb, conf_th=DEFAULT_CONFIDENCE,
                   iou_th=DEFAULT_IOU,
                   max_detections=DEFAULT_MAX_DETECTIONS):
        rgb = np.ascontiguousarray(rectified_rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rectified_rgb must be an HxWx3 RGB image")
        maix_image = self.image_module.cv2image(
            rgb, bgr=False, copy=False)
        objects = self.detector.detect(
            maix_image, conf_th=float(conf_th), iou_th=float(iou_th))
        results = []
        for obj in objects:
            results.append(CornerMark(
                _read_field(obj, "x"),
                _read_field(obj, "y"),
                _read_field(obj, "w", "width"),
                _read_field(obj, "h", "height"),
                _read_field(obj, "score", "confidence"),
                _read_field(obj, "class_id", "class_idx", "class_index"),
            ))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:max(0, int(max_detections))]

    def detect(self, rectified_rgb):
        """Provider interface used by the poker layout selector."""
        return self.detect_rgb(rectified_rgb)
