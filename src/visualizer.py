"""
Visualization Module
Author: Soham Ashok Karpe, M. Eng.
"""

import cv2
import numpy as np
from datetime import datetime


class Visualizer:
    """Draws bounding boxes, labels and stats on frames."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame: np.ndarray, detections: list, camera_idx: int = 0) -> np.ndarray:
        """Draw detections on frame."""
        output = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = det["class"].upper()
            conf = det["confidence"]
            color = det["color"]

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, self.font, 0.6, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

            # Draw label text
            cv2.putText(output, text, (x1 + 3, y1 - 5),
                        self.font, 0.6, (0, 0, 0), 2)

        # Overlay status bar
        self._draw_status(output, detections, camera_idx)
        return output

    def _draw_status(self, frame: np.ndarray, detections: list, camera_idx: int):
        """Draw status overlay at top of frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Camera index
        cv2.putText(frame, f"CAM {camera_idx}", (10, 27),
                    self.font, 0.6, (0, 212, 255), 2)

        # Timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 90, 27),
                    self.font, 0.6, (180, 180, 180), 1)

        # Detection count
        n = len(detections)
        live = sum(1 for d in detections if d["class"] == "live")
        spoof = n - live
        status_text = f"Live: {live}  Spoof: {spoof}"
        color = (0, 255, 100) if spoof == 0 else (0, 0, 255)
        cv2.putText(frame, status_text, (w // 2 - 80, 27),
                    self.font, 0.6, color, 2)
