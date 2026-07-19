"""Image preprocessing utilities."""
import cv2
import numpy as np


def resize_frame(frame: np.ndarray, width: int = 640, height: int = 640) -> np.ndarray:
    return cv2.resize(frame, (width, height))


def normalize(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) / 255.0


def enhance_lighting(frame: np.ndarray) -> np.ndarray:
    """CLAHE-based lighting enhancement for varying conditions."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def flip_horizontal(frame: np.ndarray) -> np.ndarray:
    return cv2.flip(frame, 1)
