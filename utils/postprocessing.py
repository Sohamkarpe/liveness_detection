"""Detection postprocessing utilities."""


def filter_by_confidence(detections: list, threshold: float = 0.5) -> list:
    return [d for d in detections if d["confidence"] >= threshold]


def sort_by_confidence(detections: list) -> list:
    return sorted(detections, key=lambda x: x["confidence"], reverse=True)


def get_dominant_class(detections: list) -> str:
    """Return the class with highest confidence."""
    if not detections:
        return "unknown"
    best = max(detections, key=lambda x: x["confidence"])
    return best["class"]


def is_live(detections: list, threshold: float = 0.5) -> bool:
    """Check if any detection is live above threshold."""
    for d in detections:
        if d["class"] == "live" and d["confidence"] >= threshold:
            return True
    return False
