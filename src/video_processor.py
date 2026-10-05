"""Video reading, inference, annotation, and output writing."""

from dataclasses import dataclass
from pathlib import Path
import time

import cv2

from .detector import Detection, PedestrianDetector
from .distance_estimator import estimate_distance_m


@dataclass
class ProcessingStats:
    total_frames: int = 0
    total_detections: int = 0
    max_pedestrians: int = 0
    processing_seconds: float = 0.0

    @property
    def average_detections_per_frame(self) -> float:
        return self.total_detections / self.total_frames if self.total_frames else 0.0

    @property
    def processing_fps(self) -> float:
        return self.total_frames / self.processing_seconds if self.processing_seconds else 0.0


def _draw_detection(frame, detection: Detection, distance_m: float) -> None:
    """Draw a readable bounding box and label on a BGR frame."""
    color = (0, 220, 0)
    cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
    label = f"Pedestrian | {detection.confidence:.2f} | ~{distance_m:.1f} m"
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    top = max(detection.y1, text_height + baseline + 4)
    cv2.rectangle(frame, (detection.x1, top - text_height - baseline - 4),
                  (detection.x1 + text_width + 6, top), color, -1)
    cv2.putText(frame, label, (detection.x1 + 3, top - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    detector: PedestrianDetector,
    real_height_m: float = 1.7,
    focal_length_px: float = 700.0,
    progress_callback=None,
) -> ProcessingStats:
    """Process a video while preserving its FPS and frame dimensions."""
    input_path, output_path = Path(input_path), Path(output_path)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Input video has invalid dimensions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {output_path}")

    stats = ProcessingStats()
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame)
            for detection in detections:
                distance = estimate_distance_m(
                    detection.pixel_height, real_height_m, focal_length_px
                )
                _draw_detection(frame, detection, distance)
            cv2.putText(frame, f"Pedestrians: {len(detections)}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            stats.total_frames += 1
            stats.total_detections += len(detections)
            stats.max_pedestrians = max(stats.max_pedestrians, len(detections))
            if progress_callback and frame_count > 0:
                progress_callback(min(stats.total_frames / frame_count, 1.0))
    finally:
        stats.processing_seconds = time.perf_counter() - started
        capture.release()
        writer.release()
    if stats.total_frames == 0:
        raise ValueError("Input video contained no readable frames")
    return stats

