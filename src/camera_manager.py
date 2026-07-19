"""
Multi-Camera Manager
Author: Soham Ashok Karpe, M. Eng.
"""

import cv2
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CameraManager:
    """
    Manages multiple camera inputs with synchronized frame capture.
    Supports USB cameras, IP cameras, and GigE Vision industrial cameras.
    """

    def __init__(self, camera_indices: list, config: dict = None):
        self.camera_indices = camera_indices
        self.config = config or {}
        self.captures = []

        cam_cfg = self.config.get("camera", {})
        self.width = cam_cfg.get("width", 1280)
        self.height = cam_cfg.get("height", 720)
        self.fps = cam_cfg.get("fps", 30)

    def open(self) -> bool:
        """Open all cameras."""
        self.captures = []
        for idx in self.camera_indices:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                logger.error(f"Cannot open camera {idx}")
                self.release()
                return False

            # Set resolution and FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            self.captures.append(cap)
            logger.info(f"Camera {idx} opened: {self.width}x{self.height} @ {self.fps}fps")

        return True

    def read(self) -> list:
        """Read synchronized frames from all cameras."""
        frames = []
        for i, cap in enumerate(self.captures):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                logger.warning(f"Failed to read from camera {self.camera_indices[i]}")
                frames.append(None)
        return frames

    def release(self):
        """Release all cameras."""
        for cap in self.captures:
            if cap and cap.isOpened():
                cap.release()
        self.captures = []
        logger.info("All cameras released.")

    def get_camera_info(self) -> list:
        """Get properties of all cameras."""
        info = []
        for i, cap in enumerate(self.captures):
            if cap and cap.isOpened():
                info.append({
                    "index": self.camera_indices[i],
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                })
        return info

    def __del__(self):
        self.release()
