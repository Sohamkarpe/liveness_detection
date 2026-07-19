"""
YOLOv8-based Liveness Detector
Author: Soham Ashok Karpe, M. Eng.
"""

import numpy as np
import cv2
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Class labels
CLASS_LABELS = {0: "live", 1: "spoof"}
CLASS_COLORS = {
    "live":  (0, 255, 100),   # Green
    "spoof": (0, 0, 255),     # Red
}


class LivenessDetector:
    """
    Real-time liveness detection using YOLOv8.
    Detects whether a face is a live person or a spoof attempt.
    """

    def __init__(
        self,
        model_path: str = "models/liveness_model.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        use_trt: bool = False,
        img_size: int = 640,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size
        self.use_trt = use_trt
        self.model = None

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            path = Path(model_path)

            if not path.exists():
                logger.warning(
                    f"Model not found at {model_path}. "
                    "Using YOLOv8n as placeholder for demo purposes."
                )
                # For demo: load base YOLOv8 (replace with trained model)
                self.model = YOLO("yolov8n.pt")
            else:
                self.model = YOLO(str(path))
                logger.info(f"Model loaded from: {model_path}")

            if self.use_trt:
                logger.info("Exporting to TensorRT for faster inference...")
                self.model.export(format="engine", device=self.device)

        except ImportError:
            logger.error(
                "ultralytics not installed. Run: pip install ultralytics"
            )
            raise

    def detect(self, frame: np.ndarray) -> list:
        """
        Run liveness detection on a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detections: [{"class": str, "confidence": float, "bbox": list}]
        """
        if self.model is None:
            return []

        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                imgsz=self.img_size,
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    label = CLASS_LABELS.get(class_id, "unknown")
                    detections.append({
                        "class": label,
                        "confidence": confidence,
                        "bbox": bbox,
                        "color": CLASS_COLORS.get(label, (255, 255, 255)),
                    })

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def detect_batch(self, frames: list) -> list:
        """
        Run detection on multiple frames (batch inference).

        Args:
            frames: List of BGR images

        Returns:
            List of detection lists (one per frame)
        """
        if not frames or self.model is None:
            return [[] for _ in frames]

        try:
            results_batch = self.model(
                frames,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                imgsz=self.img_size,
            )

            all_detections = []
            for results in results_batch:
                detections = []
                boxes = results.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        label = CLASS_LABELS.get(class_id, "unknown")
                        detections.append({
                            "class": label,
                            "confidence": confidence,
                            "bbox": bbox,
                            "color": CLASS_COLORS.get(label, (255, 255, 255)),
                        })
                all_detections.append(detections)

            return all_detections

        except Exception as e:
            logger.error(f"Batch detection error: {e}")
            return [[] for _ in frames]
