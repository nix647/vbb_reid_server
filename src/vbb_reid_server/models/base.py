from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    bbox: np.ndarray  # (4,) x1 y1 x2 y2 in crop pixels
    score: float
    kps: np.ndarray  # (5, 2) left eye, right eye, nose, left mouth, right mouth

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class FaceDetector(Protocol):
    model_id: str

    def detect(self, img_bgr: np.ndarray) -> list[FaceDetection]: ...


class FaceEmbedder(Protocol):
    model_id: str

    def embed(self, aligned_bgr: list[np.ndarray]) -> np.ndarray:
        """(N, 112, 112, 3) BGR uint8 aligned faces -> (N, D) L2-normalised float32."""
        ...


class BodyEmbedder(Protocol):
    model_id: str

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Person crops (any size, BGR uint8) -> (N, D) L2-normalised float32."""
        ...


def file_model_id(path: str | Path, hash_bytes: int = 1 << 30) -> str:
    """``<file name>:sha256:<first 16 hex>`` — the same idea as the board's gallery stamp."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        remaining = hash_bytes
        while remaining > 0:
            chunk = fh.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return f"{p.name}:sha256:{h.hexdigest()[:16]}"


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)
