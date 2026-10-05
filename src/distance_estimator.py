"""Monocular, pinhole-camera distance approximation."""


def estimate_distance_m(
    pixel_height: int,
    real_height_m: float = 1.7,
    focal_length_px: float = 700.0,
) -> float:
    """Estimate distance in metres using D = (H * F) / h."""
    if pixel_height <= 0:
        raise ValueError("pixel_height must be positive")
    if real_height_m <= 0 or focal_length_px <= 0:
        raise ValueError("real_height_m and focal_length_px must be positive")
    return (real_height_m * focal_length_px) / pixel_height


def calibrate_focal_length_px(
    pixel_height: int, known_distance_m: float, known_height_m: float
) -> float:
    """Calculate focal length from one known person-distance observation."""
    if pixel_height <= 0 or known_distance_m <= 0 or known_height_m <= 0:
        raise ValueError("all calibration values must be positive")
    return (pixel_height * known_distance_m) / known_height_m

