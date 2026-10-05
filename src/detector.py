"""YOLOv8 pedestrian detector wrapper."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    """A single person detection in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def pixel_height(self) -> int:
        return max(1, self.y2 - self.y1)


class PedestrianDetector:
    """Load a pretrained Ultralytics model and return only COCO person detections."""

    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = confidence
        self.model_path = Path(model_path)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc
        self.model = YOLO(model_path)

    def detect(self, frame) -> list[Detection]:
        """Run inference on one BGR OpenCV frame and keep only class 0 (person)."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()
                x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
                detections.append(
                    Detection(x1, y1, x2, y2, float(box.conf[0].item()))
                )
        return detections

