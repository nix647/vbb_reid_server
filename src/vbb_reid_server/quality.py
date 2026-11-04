"""Face quality, split into MEASURES (stored at embed time) and a POLICY (applied at read time).

Measures per detected face: detector score, face height (px, in the crop), Laplacian blur
variance of the aligned 112 face, mean alignment residual, left/right residual asymmetry,
and a signed yaw ratio from the five landmarks. They are facts about the crop and never go
stale. ``qf`` is computed from them by a :class:`QualityPolicy` whenever the cache is read,
so a policy change needs no re-embed.

The default policy follows the board's ``face/scoring.py`` shape — det score x power mean
(p = -2) of pose, size, blur and an asymmetry penalty — with the board's gating: the
asymmetry penalty applies only when the mean residual is small (<= ``residual_mean_cap``);
a large mean residual is pose, which the pose term already covers. (Until 0.1.1 the residual
itself was penalised to zero at 6 px, which zeroed most non-frontal faces: measured on
gpu-ubun 2026-11-03, 54 % of faces had residual >= 6 while same-person pairs in the 6-10
band still had median similarity 0.447 against 0.046 for different people.)

The pose term is an approximation from landmarks; every threshold here is re-measured on
the server (``vbb-reid-server quality-report``), never copied from the board.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityPolicy:
    use_pose: bool = True
    max_yaw_ratio: float = 0.6  # |nose offset| / inter-eye distance at which pose_pen hits 0
    use_size: bool = True
    size_floor_px: float = 40.0  # face height giving size_pen = 1
    use_blur: bool = True
    blur_min: float = 30.0  # Laplacian variance on the aligned 112 gray image
    blur_max: float = 150.0
    use_asym: bool = True  # board-style: asymmetry ramp, only when the mean residual is small
    asym_min: float = 1.0
    asym_max: float = 6.0
    residual_mean_cap: float = 3.0
    use_residual: bool = False  # the 0.1.0 behaviour (penalise the residual itself); off by default
    residual_min: float = 1.0
    residual_max: float = 6.0
    use_det: bool = True
    aggregation_p: float = -2.0

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path | None) -> "QualityPolicy":
        if not path:
            return cls()
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = sorted(k for k in spec if k not in known and not k.startswith("_"))
        if unknown:
            raise ValueError(f"unknown QualityPolicy key(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
        return replace(cls(), **{k: v for k, v in spec.items() if k in known})


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if hi <= lo:
        return (x >= hi).astype(np.float64)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _power_mean(cols: list[np.ndarray], p: float) -> np.ndarray:
    if not cols:
        return np.ones(0)
    arr = np.clip(np.stack(cols, axis=0), 1e-6, 1.0)
    if p == 0:
        return np.exp(np.log(arr).mean(axis=0))
    return np.mean(arr ** p, axis=0) ** (1.0 / p)


def qf_from_measures(det, face_h, blur, residual, asym, yaw, policy: QualityPolicy = QualityPolicy()) -> np.ndarray:
    """Vectorised qf in [0, 1] from stored measures (arrays of equal length)."""
    det = np.asarray(det, dtype=np.float64)
    n = det.shape[0]
    cols: list[np.ndarray] = []
    if policy.use_pose:
        cols.append(np.maximum(0.0, 1.0 - np.abs(np.asarray(yaw, dtype=np.float64)) / policy.max_yaw_ratio))
    if policy.use_size:
        cols.append(np.minimum(1.0, np.asarray(face_h, dtype=np.float64) / policy.size_floor_px))
    if policy.use_blur:
        cols.append(_ramp(blur, policy.blur_min, policy.blur_max))
    if policy.use_asym:
        pen = 1.0 - _ramp(asym, policy.asym_min, policy.asym_max)
        gated = np.asarray(residual, dtype=np.float64) <= policy.residual_mean_cap
        cols.append(np.where(gated, pen, 1.0))
    if policy.use_residual:
        cols.append(1.0 - _ramp(residual, policy.residual_min, policy.residual_max))
    combined = _power_mean(cols, policy.aggregation_p) if cols else np.ones(n)
    return (det if policy.use_det else np.ones(n)) * combined


def yaw_ratio(kps: np.ndarray) -> float:
    """Signed nose offset from the eye midpoint, along the eye axis, over the inter-eye
    distance. Large magnitudes are profile faces (eyes nearly overlap in the image)."""
    le, re, nose = kps[0], kps[1], kps[2]
    eye_mid = (le + re) / 2.0
    d = float(np.linalg.norm(re - le))
    if d < 1e-3:
        return 99.0
    axis = (re - le) / d
    return float(np.dot(nose - eye_mid, axis) / d)


def blur_var(aligned_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def asymmetry(per_point_residual: np.ndarray) -> float:
    """|mean residual of the left points (eye, mouth corner) - right points|, in aligned px."""
    r = np.asarray(per_point_residual, dtype=np.float64)
    return float(abs((r[0] + r[3]) / 2.0 - (r[1] + r[4]) / 2.0))


def face_quality(det_score: float, face_h_px: float, kps: np.ndarray, aligned_bgr: np.ndarray,
                 residual: float, policy: QualityPolicy = QualityPolicy(), asym: float = 0.0) -> float:
    """Scalar convenience wrapper (tests, ad-hoc use)."""
    return float(qf_from_measures([det_score], [face_h_px], [blur_var(aligned_bgr)], [residual], [asym],
                                  [yaw_ratio(kps)], policy)[0])
