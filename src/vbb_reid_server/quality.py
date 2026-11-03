"""Face quality ``qf`` in [0, 1], shaped like the board's ``face/scoring.py``
(det score x power-mean of pose, size, blur and alignment penalties, p = -2),
so face-size and blur buckets mean the same here and on the board. The pose term
is an approximation from the five landmarks (the board gets yaw elsewhere);
thresholds on ``qf`` are re-measured on this server, never copied."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityPolicy:
    max_yaw_ratio: float = 0.45  # |nose offset| / inter-eye distance at which pose_pen hits 0
    size_floor_px: float = 40.0  # face height giving size_pen = 1
    blur_min: float = 30.0  # Laplacian variance on the aligned 112 gray image
    blur_max: float = 150.0
    residual_min: float = 1.0
    residual_max: float = 6.0
    aggregation_p: float = -2.0


def _ramp01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _power_mean(vals: list[float], p: float) -> float:
    eps = 1e-6
    arr = np.clip(np.asarray(vals, dtype=np.float64), eps, 1.0)
    if p == 0:
        return float(np.exp(np.log(arr).mean()))
    return float(np.mean(arr ** p) ** (1.0 / p))


def yaw_ratio(kps: np.ndarray) -> float:
    """Signed nose offset from the eye midpoint, normalised by inter-eye distance."""
    le, re, nose = kps[0], kps[1], kps[2]
    eye_mid = (le + re) / 2.0
    d = float(np.linalg.norm(re - le))
    if d < 1e-3:
        return 1.0
    axis = (re - le) / d
    return float(np.dot(nose - eye_mid, axis) / d)


def blur_var(aligned_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality(det_score: float, face_h_px: float, kps: np.ndarray, aligned_bgr: np.ndarray,
                 residual: float, policy: QualityPolicy = QualityPolicy()) -> float:
    pose_pen = max(0.0, 1.0 - abs(yaw_ratio(kps)) / policy.max_yaw_ratio)
    size_pen = min(1.0, face_h_px / policy.size_floor_px)
    blur_pen = _ramp01(blur_var(aligned_bgr), policy.blur_min, policy.blur_max)
    residual_pen = 1.0 - _ramp01(residual, policy.residual_min, policy.residual_max)
    combined = _power_mean([pose_pen, size_pen, blur_pen, residual_pen], policy.aggregation_p)
    return float(det_score) * combined
