"""Embed every crop of a facility-day and cache the result per model.

Cache layout (``--emb`` root):
  <root>/face/<face_slug>/<facility>/<day>.npz   schema=2; sighting_id, crop_idx, source, vec,
                                                 and the quality MEASURES face_h, det, blur,
                                                 residual, asym, yaw (one row per crop with a
                                                 detected face). qf is NOT stored: it is computed
                                                 on read from a QualityPolicy, so changing the
                                                 policy never needs a re-embed.
                                                 ``processed`` lists every crop the detector
                                                 has seen ("sid:source:idx"), face or not, so
                                                 done-ness lives in the same file as the rows.
  <root>/body/<body_slug>/<facility>/<day>.npz   sighting_id, crop_idx, source, vec
  <root>/<kind>/<slug>/model.json                the full model id

Rebuildable from crops at any time; ``prune`` deletes day files older than N days
(decision 14: 14 days).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from .align import norm_crop_points
from .crops import CropRef, CropStore
from .data import Sighting
from .quality import QualityPolicy, asymmetry, blur_var, qf_from_measures, yaw_ratio


def slug(model_id: str) -> str:
    return hashlib.sha256(model_id.encode()).hexdigest()[:16]


FACE_SCHEMA = 3  # 1 = 0.1.0 (stored qf); 2 = measures + .done file; 3 = measures + processed in the npz
EMBED_FACE_MIN_PX = 8.0  # faces below this are not stored; the decision's face_min_px filters further


def _key(sid, source, idx) -> str:
    return f"{sid}:{source}:{int(idx)}"


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class StaleCache(RuntimeError):
    pass


@dataclass
class FaceTable:
    sighting_id: np.ndarray  # (N,) str
    crop_idx: np.ndarray  # (N,) int
    source: np.ndarray  # (N,) str
    vec: np.ndarray  # (N, D) float32
    face_h: np.ndarray  # quality MEASURES (stored)
    det: np.ndarray
    blur: np.ndarray
    residual: np.ndarray
    asym: np.ndarray
    yaw: np.ndarray
    qf: np.ndarray  # derived on read from a QualityPolicy (never stored)

    MEASURES = ("face_h", "det", "blur", "residual", "asym", "yaw")

    @classmethod
    def empty(cls, dim: int = 0) -> "FaceTable":
        z = np.zeros(0)
        return cls(np.zeros(0, dtype=object), np.zeros(0, dtype=int), np.zeros(0, dtype=object),
                   np.zeros((0, dim), dtype=np.float32), z, z, z, z, z, z, z)

    def __len__(self) -> int:
        return int(self.sighting_id.shape[0])

    def with_policy(self, policy: QualityPolicy) -> "FaceTable":
        self.qf = qf_from_measures(self.det, self.face_h, self.blur, self.residual, self.asym, self.yaw, policy) \
            if len(self) else np.zeros(0)
        return self


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


def _face_path(root: Path, model_id: str, facility: str, day: str) -> Path:
    return root / "face" / slug(model_id) / facility / f"{day}.npz"


def load_face(root: Path, model_id: str, facility: str, day: str,
              policy: QualityPolicy | None = None) -> FaceTable:
    f = _face_path(root, model_id, facility, day)
    if not f.exists():
        return FaceTable.empty()
    z = np.load(f, allow_pickle=True)
    schema = int(z["schema"]) if "schema" in z.files else 1
    if schema < FACE_SCHEMA:
        raise StaleCache(f"{f} has cache schema {schema} (current {FACE_SCHEMA}); `embed` re-embeds such a day "
                         f"automatically (crops are cached, so only the GPU work repeats)")
    t = FaceTable(z["sighting_id"], z["crop_idx"], z["source"], z["vec"], z["face_h"], z["det"], z["blur"],
                  z["residual"], z["asym"], z["yaw"], np.zeros(len(z["sighting_id"])))
    return t.with_policy(policy or QualityPolicy())


def load_processed(root: Path, model_id: str, facility: str, day: str) -> set[str]:
    f = _face_path(root, model_id, facility, day)
    if not f.exists():
        return set()
    z = np.load(f, allow_pickle=True)
    return set(str(x) for x in z["processed"]) if "processed" in z.files else set()


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


def _drop(table, keep: np.ndarray):
    return type(table)(*(getattr(table, f.name)[keep] for f in fields(table)))


def embed_day(sightings: list[Sighting], store: CropStore, models, emb_root: str | Path,
              facility: str, day: str, *, refresh: bool = False, log=print) -> dict[str, int]:
    """Embed every crop of the given sightings (one facility-day) and merge into the cache.

    Idempotent: a crop already in the cache (face: ``processed``; body: its row) is skipped.
    A sighting whose LAN crops are now available drops any CDN row it had: the LAN set holds
    the uploaded best crop too, so keeping both would store one image twice."""
    root = Path(emb_root)
    stats = {"sightings": len(sightings), "crops": 0, "faces": 0, "bodies": 0, "unreadable": 0,
             "replaced_cdn": 0}
    t0 = time.time()

    face_id, body_id = models.face_id, models.body_id
    processed: set[str] = set()
    old_face = FaceTable.empty()
    old_body = BodyTable.empty()
    if face_id:
        _dir(root, "face", face_id, facility)
        if not refresh:
            try:
                old_face = load_face(root, face_id, facility, day)
                processed = load_processed(root, face_id, facility, day)
            except StaleCache:
                log(f"[embed] facility {facility} {day}: older cache schema - re-embedding the day")
                old_face, processed = FaceTable.empty(), set()
    if body_id:
        _dir(root, "body", body_id, facility)
        if not refresh:
            old_body = load_body(root, body_id, facility, day)

    crops_of = {s.sighting_id: store.crops(s) for s in sightings}
    # LAN supersedes CDN, per sighting
    lan_sids = {sid for sid, refs in crops_of.items() if refs and refs[0].source == "lan"}
    if lan_sids:
        if len(old_face):
            keep = np.array([not (str(sid) in lan_sids and src == "cdn")
                             for sid, src in zip(old_face.sighting_id, old_face.source)], dtype=bool)
            stats["replaced_cdn"] += int((~keep).sum())
            old_face = _drop(old_face, keep)
        processed = {k for k in processed if not (k.split(":")[0] in lan_sids and k.split(":")[1] == "cdn")}
        if len(old_body):
            keep = np.array([not (str(sid) in lan_sids and src == "cdn")
                             for sid, src in zip(old_body.sighting_id, old_body.source)], dtype=bool)
            old_body = _drop(old_body, keep)
    body_done = {_key(s_, src, c) for s_, src, c in zip(old_body.sighting_id, old_body.source, old_body.crop_idx)}

    f_rows, f_imgs = [], []
    b_rows, b_imgs = [], []
    for s in sightings:
        for ref in crops_of[s.sighting_id]:
            key = _key(s.sighting_id, ref.source, ref.index)
            need_face = bool(face_id) and key not in processed
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
                processed.add(key)
                det = _best_face(models.face_detector.detect(img), img.shape[0])
                if det is None or det.height < EMBED_FACE_MIN_PX:
                    continue
                aligned, per_point = norm_crop_points(img, det.kps)
                f_rows.append((s.sighting_id, ref.index, ref.source, det.height, det.score, blur_var(aligned),
                               float(per_point.mean()), asymmetry(per_point), yaw_ratio(det.kps)))
                f_imgs.append(aligned)

    if face_id:
        vecs = models.face.embed(f_imgs) if f_imgs else np.zeros((0, old_face.vec.shape[1] if len(old_face) else 0), np.float32)
        stats["faces"] = len(f_rows)
        col = lambda i: np.array([r[i] for r in f_rows], dtype=float)  # noqa: E731
        new = FaceTable(
            np.array([r[0] for r in f_rows], dtype=object), np.array([r[1] for r in f_rows], dtype=int),
            np.array([r[2] for r in f_rows], dtype=object), vecs.astype(np.float32),
            col(3), col(4), col(5), col(6), col(7), col(8), np.zeros(len(f_rows)))
        merged = _dedupe(_concat_face(old_face, new))
        fdir = root / "face" / slug(face_id) / facility
        _atomic_savez(fdir / f"{day}.npz", schema=np.int64(FACE_SCHEMA), sighting_id=merged.sighting_id,
                      crop_idx=merged.crop_idx, source=merged.source, vec=merged.vec,
                      processed=np.array(sorted(processed), dtype=object),
                      **{m: getattr(merged, m) for m in FaceTable.MEASURES})
        stale_done = fdir / f"{day}.done"  # schema 2 leftovers
        if stale_done.exists():
            stale_done.unlink()
    if body_id:
        vecs = models.body.embed(b_imgs) if b_imgs else np.zeros((0, old_body.vec.shape[1] if len(old_body) else 0), np.float32)
        stats["bodies"] = len(b_rows)
        new_b = BodyTable(np.array([r[0] for r in b_rows], dtype=object), np.array([r[1] for r in b_rows], dtype=int),
                          np.array([r[2] for r in b_rows], dtype=object), vecs.astype(np.float32))
        merged_b = _dedupe(_concat_body(old_body, new_b))
        bdir = root / "body" / slug(body_id) / facility
        _atomic_savez(bdir / f"{day}.npz", sighting_id=merged_b.sighting_id, crop_idx=merged_b.crop_idx,
                      source=merged_b.source, vec=merged_b.vec)
    stats["seconds"] = round(time.time() - t0, 1)
    log(f"[embed] facility {facility} {day}: {stats}")
    return stats


def _dedupe(table):
    """One row per (sighting, source, crop): the last one written wins."""
    if len(table) == 0:
        return table
    seen: dict[str, int] = {}
    for i, (sid, src, c) in enumerate(zip(table.sighting_id, table.source, table.crop_idx)):
        seen[_key(sid, src, c)] = i
    keep = np.zeros(len(table), dtype=bool)
    keep[list(seen.values())] = True
    return _drop(table, keep)


def _concat_face(a: FaceTable, b: FaceTable) -> FaceTable:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    names = ("sighting_id", "crop_idx", "source", "vec") + FaceTable.MEASURES + ("qf",)
    return FaceTable(*(np.concatenate([getattr(a, n), getattr(b, n)]) for n in names))


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
    for f in list(root.rglob("*.npz")) + list(root.rglob("*.done")) + list(root.rglob("*.tmp.npz")):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            n += 1
    return n
