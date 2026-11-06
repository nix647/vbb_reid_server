"""ONNX Runtime adapters: SCRFD face detector, 112x112 face recognisers
(ArcFace / AdaFace exports, the board's AdaFace IR-50 via the gpu-ubun ONNX port),
and a generic person-ReID / vision encoder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import FaceDetection, file_model_id, l2n


def _session(path: str | Path, providers: list[str] | None = None):
    import onnxruntime as ort  # optional dependency
    if hasattr(ort, "preload_dlls"):  # ORT >= 1.21: use the CUDA/cuDNN libraries pip installed (e.g. with torch)
        try:
            ort.preload_dlls()
        except Exception:
            pass
    avail = ort.get_available_providers()
    wanted = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    use = [p for p in wanted if p in avail] or ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=use)


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][ovr <= thresh]
    return keep


class SCRFDDetector:
    """InsightFace SCRFD with keypoints (``*_bnkps`` / ``det_10g.onnx``), strides 8/16/32,
    two anchors per location. Letterboxes into ``input_size`` top-left."""

    def __init__(self, path: str | Path, input_size: int = 640, score_thresh: float = 0.5,
                 nms_thresh: float = 0.4, providers: list[str] | None = None):
        self.path = Path(path)
        self.model_id = f"{file_model_id(self.path)}:in{int(input_size)}:s{score_thresh:g}:n{nms_thresh:g}"
        self.session = _session(self.path, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(input_size)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        n_out = len(self.session.get_outputs())
        if n_out != 9:
            raise ValueError(f"SCRFD export with {n_out} outputs: need the 9-output export with landmarks "
                             f"(*_bnkps / det_10g.onnx); alignment needs the five keypoints")
        self.fmc = 3
        self.strides = (8, 16, 32)
        self.use_kps = n_out == 9
        self.num_anchors = 2
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        c = self._centers.get(key)
        if c is None:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            c = np.stack([c] * self.num_anchors, axis=1).reshape(-1, 2)
            self._centers[key] = c
        return c

    def detect(self, img_bgr: np.ndarray) -> list[FaceDetection]:
        size = self.input_size
        ih, iw = img_bgr.shape[:2]
        scale = min(size / ih, size / iw)
        nh, nw = max(1, int(round(ih * scale))), max(1, int(round(iw * scale)))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})
        outs = [o[0] if o.ndim == 3 else o for o in outs]
        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(self.strides):
            scores = outs[idx].reshape(-1)
            bbox = outs[idx + self.fmc].reshape(-1, 4) * stride
            h, w = size // stride, size // stride
            centers = self._anchor_centers(h, w, stride)
            pos = np.where(scores >= self.score_thresh)[0]
            if pos.size == 0:
                continue
            c = centers[pos]
            b = bbox[pos]
            boxes = np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], axis=1)
            scores_l.append(scores[pos])
            boxes_l.append(boxes)
            if self.use_kps:
                k = outs[idx + 2 * self.fmc].reshape(-1, 10)[pos] * stride
                kp = np.empty((pos.size, 5, 2), dtype=np.float32)
                kp[:, :, 0] = c[:, 0:1] + k[:, 0::2]
                kp[:, :, 1] = c[:, 1:2] + k[:, 1::2]
                kps_l.append(kp)
        if not scores_l:
            return []
        scores = np.concatenate(scores_l)
        boxes = np.concatenate(boxes_l) / scale
        kps = np.concatenate(kps_l) / scale if kps_l else np.zeros((scores.size, 5, 2), np.float32)
        keep = _nms(boxes, scores, self.nms_thresh)
        return [FaceDetection(bbox=boxes[i].astype(np.float32), score=float(scores[i]),
                              kps=kps[i].astype(np.float32)) for i in keep]


class OnnxFaceEmbedder:
    """112x112 recogniser. ``color``: channel order the network expects (AdaFace: BGR,
    InsightFace ArcFace: RGB). ``flip``: average with the horizontally flipped face."""

    def __init__(self, path: str | Path, color: str = "RGB", mean: float = 127.5, std: float = 127.5,
                 layout: str = "NCHW", flip: bool = False, batch: int = 64, providers: list[str] | None = None):
        self.path = Path(path)
        # Every preprocessing choice is part of the id: two colour orders of one weights file
        # are two models, and must never share a cache.
        self.model_id = (f"{file_model_id(self.path)}:{color.upper()}:m{float(mean):g}:s{float(std):g}:"
                         f"{layout.upper()}" + (":flip" if flip else ""))
        self.session = _session(self.path, providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.color = color.upper()
        self.mean, self.std = float(mean), float(std)
        self.layout = layout.upper()
        self.flip = flip
        static = inp.shape[0] if isinstance(inp.shape[0], int) else None
        self.batch = static or batch

    def _prep(self, faces: list[np.ndarray]) -> np.ndarray:
        arr = np.stack([cv2.resize(f, (112, 112)) if f.shape[:2] != (112, 112) else f for f in faces])
        if self.color == "RGB":
            arr = arr[..., ::-1]
        arr = (arr.astype(np.float32) - self.mean) / self.std
        if self.layout == "NCHW":
            arr = arr.transpose(0, 3, 1, 2)
        return np.ascontiguousarray(arr)

    def _run(self, faces: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(faces), self.batch):
            chunk = faces[i:i + self.batch]
            pad = self.batch - len(chunk) if self.batch and len(chunk) < self.batch and self._static() else 0
            x = self._prep(chunk + [chunk[-1]] * pad)
            y = self.session.run(None, {self.input_name: x})[0]
            out.append(y[:len(chunk)])
        return np.concatenate(out).reshape(len(faces), -1)

    def _static(self) -> bool:
        return isinstance(self.session.get_inputs()[0].shape[0], int)

    def embed_with_norm(self, aligned_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Unit embeddings and the raw feature norm before normalisation (with flip: of the
        mean of both passes). ArcFace/AdaFace-style models give a low norm to unrecognisable
        inputs (hair, the back of a head), so the norm is stored as a quality measure."""
        if not aligned_bgr:
            return np.zeros((0, 0), dtype=np.float32), np.zeros(0)
        e = self._run(aligned_bgr)
        if self.flip:
            e = (e + self._run([f[:, ::-1].copy() for f in aligned_bgr])) / 2.0
        return l2n(e), np.linalg.norm(e, axis=1).astype(np.float64)

    def embed(self, aligned_bgr: list[np.ndarray]) -> np.ndarray:
        return self.embed_with_norm(aligned_bgr)[0]


class OnnxBodyEmbedder:
    """Generic person-crop encoder (the board's ViT-S/16+ICS ONNX twin, CLIP-ReID or
    SOLIDER exports). Resizes to ``height`` x ``width`` (no letterbox, as ReID models
    are trained), normalises with per-channel ``mean``/``std`` in 0..255 units.

    ``resize``: ``"cv2"`` (INTER_LINEAR) or ``"pil"`` (PIL bilinear, which antialiases when it
    shrinks; torchvision's Resize on a PIL image, which made the CLIP-ReID vectors the owner's
    test was scored on). Needs pillow (``.[onnx]`` declares it). A threshold measured on
    vectors from one resize does not transfer to the other (2026-11-06: 3 body names changed
    person on facility 3), so it is part of the model id."""

    def __init__(self, path: str | Path, height: int = 256, width: int = 128, color: str = "RGB",
                 mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
                 std: tuple[float, float, float] = (58.395, 57.12, 57.375),
                 layout: str = "NCHW", flip: bool = False, batch: int = 64, providers: list[str] | None = None,
                 resize: str = "cv2"):
        if resize not in ("cv2", "pil"):
            raise ValueError(f"resize must be 'cv2' or 'pil', not {resize!r}")
        if resize == "pil":
            from PIL import Image  # pillow: only needed for this option
            self._pil = Image
        self.resize = resize
        self.path = Path(path)
        self.model_id = (f"{file_model_id(self.path)}:{int(height)}x{int(width)}:{color.upper()}:"
                         f"m{','.join(f'{x:g}' for x in mean)}:s{','.join(f'{x:g}' for x in std)}:{layout.upper()}"
                         + (":flip" if flip else "") + (":pil" if resize == "pil" else ""))
        self.session = _session(self.path, providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.h, self.w = int(height), int(width)
        self.color = color.upper()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.layout = layout.upper()
        self.flip = flip
        self.static_batch = inp.shape[0] if isinstance(inp.shape[0], int) else None
        self.batch = self.static_batch or batch

    def _prep(self, crops: list[np.ndarray]) -> np.ndarray:
        if self.resize == "pil":  # PIL resizes each channel alike, so the BGR order is kept here
            arr = np.stack([np.asarray(self._pil.fromarray(np.ascontiguousarray(c)).resize(
                (self.w, self.h), self._pil.Resampling.BILINEAR)) for c in crops])
        else:
            arr = np.stack([cv2.resize(c, (self.w, self.h), interpolation=cv2.INTER_LINEAR) for c in crops])
        if self.color == "RGB":
            arr = arr[..., ::-1]
        arr = (arr.astype(np.float32) - self.mean) / self.std
        if self.layout == "NCHW":
            arr = arr.transpose(0, 3, 1, 2)
        return np.ascontiguousarray(arr)

    def _run(self, crops: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(crops), self.batch):
            chunk = crops[i:i + self.batch]
            pad = (self.static_batch - len(chunk)) if self.static_batch and len(chunk) < self.static_batch else 0
            y = self.session.run(None, {self.input_name: self._prep(chunk + [chunk[-1]] * pad)})[0]
            out.append(y[:len(chunk)])
        return np.concatenate(out).reshape(len(crops), -1)

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, 0), dtype=np.float32)
        e = self._run(crops_bgr)
        if self.flip:
            e = e + self._run([c[:, ::-1].copy() for c in crops_bgr])
        return l2n(e)
