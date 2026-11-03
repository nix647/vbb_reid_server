"""Embed every crop of a facility-day and cache the result per model.

Cache layout (``--emb`` root):
  <root>/face/<face_slug>/<facility>/<day>.npz   sighting_id, crop_idx, source, vec, qf,
                                                 face_h, det, residual   (one row per crop
                                                 with a usable face)
  <root>/face/<face_slug>/<facility>/<day>.done  sighting_id:crop_idx lines already processed
  <root>/body/<body_slug>/<facility>/<day>.npz   sighting_id, crop_idx, source, vec
  <root>/<kind>/<slug>/model.json                the full model id

Rebuildable from crops at any time; ``prune`` deletes day files older than N days
(decision 14: 14 days).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .align import norm_crop
from .crops import CropRef, CropStore
from .data import Sighting
from .quality import QualityPolicy, face_quality


def slug(model_id: str) -> str:
    return hashlib.sha256(model_id.encode()).hexdigest()[:16]


@dataclass
class FaceTable:
    sighting_id: np.ndarray  # (N,) str
    crop_idx: np.ndarray  # (N,) int
    source: np.ndarray  # (N,) str
    vec: np.ndarray  # (N, D) float32
    qf: np.ndarray
    face_h: np.ndarray
    det: np.ndarray
    residual: np.ndarray

    @classmethod
    def empty(cls, dim: int = 0) -> "FaceTable":
        z = np.zeros(0)
        return cls(np.zeros(0, dtype=object), np.zeros(0, dtype=int), np.zeros(0, dtype=object),
                   np.zeros((0, dim), dtype=np.float32), z, z, z, z)

    def __len__(self) -> int:
        return int(self.sighting_id.shape[0])


@dataclass
class BodyTable:
    sighting_id: np.ndarray
    crop_idx: np.ndarray
    source: np.ndarray
    vec: np.ndarray

    @classmethod
    def empty(cls, dim: int = 0) -> "BodyTable":
        return cls(np.zeros(0, dtype=object), np.zeros(0, dtype=int), np.zeros(0, dtype=object),
                   np.zeros((0, dim), dtype=np.float32))

    def __len__(self) -> int:
        return int(self.sighting_id.shape[0])


def _dir(root: Path, kind: str, model_id: str, facility: str) -> Path:
    d = root / kind / slug(model_id) / facility
    d.mkdir(parents=True, exist_ok=True)
    meta = d.parent / "model.json"
    if not meta.exists():
        meta.write_text(json.dumps({"model_id": model_id, "kind": kind}, ensure_ascii=False))
    return d


def load_face(root: Path, model_id: str, facility: str, day: str) -> FaceTable:
    f = root / "face" / slug(model_id) / facility / f"{day}.npz"
    if not f.exists():
        return FaceTable.empty()
    z = np.load(f, allow_pickle=True)
    return FaceTable(z["sighting_id"], z["crop_idx"], z["source"], z["vec"], z["qf"],
                     z["face_h"], z["det"], z["residual"])


def load_body(root: Path, model_id: str, facility: str, day: str) -> BodyTable:
    f = root / "body" / slug(model_id) / facility / f"{day}.npz"
    if not f.exists():
        return BodyTable.empty()
    z = np.load(f, allow_pickle=True)
    return BodyTable(z["sighting_id"], z["crop_idx"], z["source"], z["vec"])


def _read(ref: CropRef) -> np.ndarray | None:
    data = np.fromfile(str(ref.path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _best_face(dets, img_h: int):
    """Largest confident face; prefers faces in the top 60 % of a person crop."""
    best, best_key = None, -1.0
    for d in dets:
        cy = (d.bbox[1] + d.bbox[3]) / 2.0
        top_bonus = 1.0 if cy <= 0.6 * img_h else 0.5
        key = d.score * max(d.height, 1.0) * top_bonus
        if key > best_key:
            best, best_key = d, key
    return best


def embed_day(sightings: list[Sighting], store: CropStore, models, emb_root: str | Path,
              facility: str, day: str, *, policy: QualityPolicy = QualityPolicy(),
              face_min_px: float = 12.0, log=print) -> dict[str, int]:
    """Embed every crop of the given sightings (one facility-day) and merge into the cache."""
    root = Path(emb_root)
    stats = {"sightings": len(sightings), "crops": 0, "faces": 0, "bodies": 0, "unreadable": 0}
    t0 = time.time()

    face_id, body_id = models.face_id, models.body_id
    done: set[str] = set()
    old_face = FaceTable.empty()
    old_body = BodyTable.empty()
    if face_id:
        fdir = _dir(root, "face", face_id, facility)
        done_path = fdir / f"{day}.done"
        if done_path.exists():
            done = set(done_path.read_text().split())
        old_face = load_face(root, face_id, facility, day)
    if body_id:
        _dir(root, "body", body_id, facility)
        old_body = load_body(root, body_id, facility, day)
    body_done = {f"{s}:{c}" for s, c in zip(old_body.sighting_id, old_body.crop_idx)}

    f_rows, f_imgs = [], []
    b_rows, b_imgs = [], []
    for s in sightings:
        for ref in store.crops(s):
            key = f"{s.sighting_id}:{ref.index}"
            need_face = bool(face_id) and key not in done
            need_body = bool(body_id) and key not in body_done
            if not (need_face or need_body):
                continue
            img = _read(ref)
            stats["crops"] += 1
            if img is None:
                stats["unreadable"] += 1
                continue
            if need_body:
                b_rows.append((s.sighting_id, ref.index, ref.source))
                b_imgs.append(img)
            if need_face:
                done.add(key)
                det = _best_face(models.face_detector.detect(img), img.shape[0])
                if det is None or det.height < face_min_px:
                    continue
                aligned, residual = norm_crop(img, det.kps)
                qf = face_quality(det.score, det.height, det.kps, aligned, residual, policy)
                f_rows.append((s.sighting_id, ref.index, ref.source, qf, det.height, det.score, residual))
                f_imgs.append(aligned)

    if face_id:
        vecs = models.face.embed(f_imgs) if f_imgs else np.zeros((0, old_face.vec.shape[1] if len(old_face) else 0), np.float32)
        stats["faces"] = len(f_rows)
        new = FaceTable(
            np.array([r[0] for r in f_rows], dtype=object), np.array([r[1] for r in f_rows], dtype=int),
            np.array([r[2] for r in f_rows], dtype=object), vecs.astype(np.float32),
            np.array([r[3] for r in f_rows], dtype=float), np.array([r[4] for r in f_rows], dtype=float),
            np.array([r[5] for r in f_rows], dtype=float), np.array([r[6] for r in f_rows], dtype=float))
        merged = _concat_face(old_face, new)
        fdir = root / "face" / slug(face_id) / facility
        np.savez(fdir / f"{day}.npz", sighting_id=merged.sighting_id, crop_idx=merged.crop_idx,
                 source=merged.source, vec=merged.vec, qf=merged.qf, face_h=merged.face_h,
                 det=merged.det, residual=merged.residual)
        (fdir / f"{day}.done").write_text("\n".join(sorted(done)))
    if body_id:
        vecs = models.body.embed(b_imgs) if b_imgs else np.zeros((0, old_body.vec.shape[1] if len(old_body) else 0), np.float32)
        stats["bodies"] = len(b_rows)
        new_b = BodyTable(np.array([r[0] for r in b_rows], dtype=object), np.array([r[1] for r in b_rows], dtype=int),
                          np.array([r[2] for r in b_rows], dtype=object), vecs.astype(np.float32))
        merged_b = _concat_body(old_body, new_b)
        bdir = root / "body" / slug(body_id) / facility
        np.savez(bdir / f"{day}.npz", sighting_id=merged_b.sighting_id, crop_idx=merged_b.crop_idx,
                 source=merged_b.source, vec=merged_b.vec)
    stats["seconds"] = round(time.time() - t0, 1)
    log(f"[embed] facility {facility} {day}: {stats}")
    return stats


def _concat_face(a: FaceTable, b: FaceTable) -> FaceTable:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return FaceTable(*(np.concatenate([x, y]) for x, y in (
        (a.sighting_id, b.sighting_id), (a.crop_idx, b.crop_idx), (a.source, b.source), (a.vec, b.vec),
        (a.qf, b.qf), (a.face_h, b.face_h), (a.det, b.det), (a.residual, b.residual))))


def _concat_body(a: BodyTable, b: BodyTable) -> BodyTable:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return BodyTable(*(np.concatenate([x, y]) for x, y in (
        (a.sighting_id, b.sighting_id), (a.crop_idx, b.crop_idx), (a.source, b.source), (a.vec, b.vec))))


def prune(emb_root: str | Path, older_than_days: float) -> int:
    root = Path(emb_root)
    cutoff = time.time() - older_than_days * 86400
    n = 0
    for f in list(root.rglob("*.npz")) + list(root.rglob("*.done")):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            n += 1
    return n
