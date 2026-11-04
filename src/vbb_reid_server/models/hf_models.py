"""Hugging Face vision encoders used as body/appearance embedders (SigLIP2,
DINOv3, CLIP). Optional: needs the ``hf`` extra (torch, transformers, pillow).

Person crops are tall; square-input encoders would squash them, so by default the
crop is padded to a square with neutral grey before the processor resizes it."""

from __future__ import annotations

import numpy as np

from .base import l2n


class HFVisionEmbedder:
    def __init__(self, repo: str, revision: str | None = None, pooling: str = "auto",
                 pad_square: bool = True, batch: int = 32, device: str | None = None,
                 dtype: str = "float16"):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.repo = repo
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if self.device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
        self.model = AutoModel.from_pretrained(repo, revision=revision, torch_dtype=self.dtype).to(self.device).eval()
        rev = revision or getattr(self.model.config, "_commit_hash", None) or "main"
        self.model_id = f"hf:{repo}@{str(rev)[:12]}:{pooling}:{dtype if self.device == 'cuda' else 'float32'}" \
            + (":pad" if pad_square else "")
        self.pooling = pooling
        self.pad_square = pad_square
        self.batch = batch

    @staticmethod
    def _pad(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        s = max(h, w)
        out = np.full((s, s, 3), 127, dtype=np.uint8)
        y, x = (s - h) // 2, (s - w) // 2
        out[y:y + h, x:x + w] = img
        return out

    def _features(self, pixel_values):
        m = self.model
        if self.pooling in ("auto", "image_features") and hasattr(m, "get_image_features"):
            return m.get_image_features(pixel_values=pixel_values)
        out = m(pixel_values=pixel_values)
        if self.pooling in ("auto", "pooler") and getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        hs = out.last_hidden_state
        cfg = getattr(m.config, "vision_config", m.config)
        has_cls = "siglip" not in str(getattr(cfg, "model_type", "")).lower()  # SigLIP has no CLS token
        if self.pooling == "mean":
            if not has_cls:
                return hs.mean(dim=1)
            skip = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)  # CLS + DINOv2/v3 registers
            return hs[:, skip:].mean(dim=1)
        if not has_cls:
            raise ValueError(f"pooling={self.pooling!r} needs a CLS token; {self.repo} has none - use image_features or mean")
        return hs[:, 0]

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, 0), dtype=np.float32)
        torch = self.torch
        outs = []
        for i in range(0, len(crops_bgr), self.batch):
            imgs = [c[..., ::-1] for c in crops_bgr[i:i + self.batch]]  # BGR -> RGB
            if self.pad_square:
                imgs = [self._pad(np.ascontiguousarray(c)) for c in imgs]
            inputs = self.processor(images=[np.ascontiguousarray(c) for c in imgs], return_tensors="pt")
            pv = inputs["pixel_values"].to(self.device, dtype=self.dtype)
            with torch.inference_mode():
                f = self._features(pv)
            outs.append(f.float().cpu().numpy())
        return l2n(np.concatenate(outs))
