"""Build a model set from a JSON config (see ``configs/models.example.json``).

{
  "name": "adaface_ir101+siglip2",
  "face_detector": {"type": "scrfd", "path": "...det_10g.onnx"},
  "face": {"type": "onnx_face", "path": "...", "color": "BGR", "flip": true},
  "body": {"type": "hf_vision", "repo": "google/siglip2-so400m-patch14-384"}
}
Either ``face`` or ``body`` may be omitted (a face-only or body-only candidate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelSet:
    name: str
    face_detector: Any | None
    face: Any | None
    body: Any | None

    @property
    def face_id(self) -> str | None:
        if self.face is None or self.face_detector is None:
            return None
        return f"{self.face_detector.model_id}|{self.face.model_id}"

    @property
    def body_id(self) -> str | None:
        return None if self.body is None else self.body.model_id


def _build(spec: dict[str, Any] | None, base: Path):
    if not spec:
        return None
    spec = dict(spec)
    kind = spec.pop("type")
    if "path" in spec:
        p = Path(spec["path"])
        spec["path"] = p if p.is_absolute() else base / p
    if kind == "scrfd":
        from .onnx_models import SCRFDDetector
        return SCRFDDetector(**spec)
    if kind == "onnx_face":
        from .onnx_models import OnnxFaceEmbedder
        return OnnxFaceEmbedder(**spec)
    if kind == "onnx_body":
        from .onnx_models import OnnxBodyEmbedder
        if "mean" in spec:
            spec["mean"] = tuple(spec["mean"])
        if "std" in spec:
            spec["std"] = tuple(spec["std"])
        return OnnxBodyEmbedder(**spec)
    if kind == "hf_vision":
        from .hf_models import HFVisionEmbedder
        return HFVisionEmbedder(**spec)
    raise ValueError(f"unknown model type {kind!r}")


def load_model_set(config_path: str | Path) -> ModelSet:
    path = Path(config_path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    return ModelSet(
        name=cfg.get("name") or path.stem,
        face_detector=_build(cfg.get("face_detector"), base),
        face=_build(cfg.get("face"), base),
        body=_build(cfg.get("body"), base),
    )
