"""Five-point face alignment to the 112x112 ArcFace template (same template as
the board, ReID ``preprocess_rknn.py`` ``ARCFACE_DST``), so crops fed to AdaFace /
ArcFace models here match what the board's recognizer saw."""

from __future__ import annotations

import cv2
import numpy as np

ARCFACE_DST = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]], dtype=np.float32)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity transform (rotation, uniform scale, translation) mapping
    ``src`` onto ``dst`` in the least-squares sense (Umeyama 1991)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[1, 1] = -1
    r = u @ s @ vt
    var_s = (sc ** 2).sum() / n
    scale = (d * np.diag(s)).sum() / var_s if var_s > 0 else 1.0
    t = mu_d - scale * r @ mu_s
    m = np.zeros((2, 3), dtype=np.float64)
    m[:, :2] = scale * r
    m[:, 2] = t
    return m


def norm_crop(img: np.ndarray, kps: np.ndarray, size: int = 112) -> tuple[np.ndarray, float]:
    """Warp a face to ``size``x``size``. Returns (aligned BGR image, mean landmark
    residual in pixels after alignment — a misalignment signal)."""
    dst = ARCFACE_DST * (size / 112.0)
    m = umeyama_similarity(kps, dst)
    aligned = cv2.warpAffine(img, m, (size, size), borderValue=0.0)
    proj = (m[:, :2] @ np.asarray(kps, dtype=np.float64).T).T + m[:, 2]
    residual = float(np.linalg.norm(proj - dst, axis=1).mean())
    return aligned, residual
