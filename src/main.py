"""
Liveness Detection System - Main Entry Point
Author: Soham Ashok Karpe, M. Eng.
Institution: Technische Hochschule Deggendorf (DIT), Campus Cham
Period: Mar 2024 – Jun 2024 | DIT HiWi Project
"""

import argparse
import sys
import cv2
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from detector import LivenessDetector
from camera_manager import CameraManager
from visualizer import Visualizer
from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Liveness Detection using YOLOv8"
    )
    parser.add_argument(
        "--config", type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model", type=str,
        default=None,
        help="Path to YOLOv8 model (.pt file)"
    )
    parser.add_argument(
        "--camera", type=int,
        default=0,
        help="Single camera index"
    )
    parser.add_argument(
        "--cameras", type=int, nargs="+",
        default=None,
        help="Multiple camera indices (e.g. --cameras 0 1 2)"
    )
    parser.add_argument(
        "--device", type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device"
    )
    parser.add_argument(
        "--conf", type=float,
        default=0.5,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--trt", action="store_true",
        help="Use TensorRT for faster inference (NVIDIA Jetson)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save output video"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("Liveness Detection System — Starting")
    logger.info("Author: Soham Karpe, M. Eng. | DIT Campus Cham")
    logger.info("=" * 60)

    # Load config
    try:
        config = load_config(args.config)
        logger.info(f"Config loaded from: {args.config}")
    except FileNotFoundError:
        logger.warning(f"Config not found at {args.config}, using defaults")
        config = {}

    # Override config with CLI args
    model_path = args.model or config.get("model", {}).get("path", "models/liveness_model.pt")
    conf_threshold = args.conf or config.get("model", {}).get("confidence", 0.5)
    device = args.device or config.get("model", {}).get("device", "cpu")

    # Determine cameras
    camera_indices = args.cameras if args.cameras else [args.camera]
    logger.info(f"Camera indices: {camera_indices}")
    logger.info(f"Device: {device}")
    logger.info(f"Confidence threshold: {conf_threshold}")

    # Initialize detector
    logger.info(f"Loading model from: {model_path}")
    detector = LivenessDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
        use_trt=args.trt
    )

    # Initialize camera manager
    cam_manager = CameraManager(camera_indices, config)

    # Initialize visualizer
    visualizer = Visualizer(config)

    # Open cameras
    if not cam_manager.open():
        logger.error("Failed to open cameras. Exiting.")
        sys.exit(1)

    logger.info("System ready. Press 'q' to quit.")

    try:
        while True:
            # Capture frames from all cameras
            frames = cam_manager.read()

            if not frames:
                logger.error("No frames captured")
                break

            results_all = []
            for idx, frame in enumerate(frames):
                if frame is None:
                    continue

                # Run liveness detection
                detections = detector.detect(frame)

                # Visualize results on frame
                annotated = visualizer.draw(frame, detections, camera_idx=idx)
                results_all.append((annotated, detections))

                # Display
                window_name = f"Liveness Detection - Camera {camera_indices[idx]}"
                cv2.imshow(window_name, annotated)

            # Log summary
            for idx, (_, detections) in enumerate(results_all):
                for det in detections:
                    status = "✅ LIVE" if det["class"] == "live" else "⚠️  SPOOF"
                    logger.info(
                        f"Camera {camera_indices[idx]}: {status} "
                        f"(conf: {det['confidence']:.2f})"
                    )

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User requested exit.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cam_manager.release()
        cv2.destroyAllWindows()
        logger.info("System shut down cleanly.")


if __name__ == "__main__":
    main()
