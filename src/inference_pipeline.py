"""
Real-Time Inference Pipeline
Author: Soham Ashok Karpe, M. Eng.
"""

import cv2
import numpy as np
import time
from utils.logger import setup_logger

logger = setup_logger(__name__)


class InferencePipeline:
    """
    End-to-end inference pipeline:
    Capture → Preprocess → Detect → Postprocess → Output
    """

    def __init__(self, detector, camera_manager, visualizer, config: dict = None):
        self.detector = detector
        self.camera_manager = camera_manager
        self.visualizer = visualizer
        self.config = config or {}
        self.frame_count = 0
        self.fps = 0
        self._last_time = time.time()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame before inference."""
        if frame is None:
            return None

        # Resize if needed
        target_size = self.config.get("inference", {}).get("img_size", 640)

        # Normalize brightness for varying lighting conditions
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame

    def postprocess(self, detections: list) -> list:
        """
        Filter and sort detections by confidence.
        Apply minimum confidence threshold.
        """
        min_conf = self.config.get("inference", {}).get("min_confidence", 0.5)
        filtered = [d for d in detections if d["confidence"] >= min_conf]
        filtered.sort(key=lambda x: x["confidence"], reverse=True)
        return filtered

    def compute_fps(self) -> float:
        """Calculate current frames per second."""
        now = time.time()
        elapsed = now - self._last_time
        self.fps = 1.0 / elapsed if elapsed > 0 else 0
        self._last_time = now
        return self.fps

    def run_single_frame(self, frame: np.ndarray, camera_idx: int = 0) -> dict:
        """
        Run full pipeline on a single frame.

        Returns:
            dict with annotated frame and detections
        """
        start = time.time()

        # Preprocess
        processed = self.preprocess(frame)
        if processed is None:
            return {"frame": frame, "detections": [], "fps": 0}

        # Detect
        detections = self.detector.detect(processed)

        # Postprocess
        detections = self.postprocess(detections)

        # Visualize
        annotated = self.visualizer.draw(processed, detections, camera_idx)

        # FPS
        elapsed_ms = (time.time() - start) * 1000
        fps = self.compute_fps()
        self.frame_count += 1

        # Draw FPS
        cv2.putText(annotated, f"FPS: {fps:.1f}  {elapsed_ms:.1f}ms",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 1)

        return {
            "frame": annotated,
            "detections": detections,
            "fps": fps,
            "inference_ms": elapsed_ms,
        }
