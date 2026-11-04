"""Synthetic exports and embedding caches — no models, no network."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from vbb_reid_server.data import Sighting
from vbb_reid_server.embed import _dir, slug

D = 32
FACE_ID = "test-det|test-face"
BODY_ID = "test-body"


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def basis(i: int, d: int = D) -> np.ndarray:
    v = np.zeros(d, dtype=np.float32)
    v[i] = 1.0
    return v


def near(base: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    return unit(base + noise * rng.standard_normal(base.shape[0]).astype(np.float32))


def sighting(sid: str, *, did: str = "45", facility: str = "3", day: str = "2026-10-20",
             board_state: str = "stranger", board_pid: str | None = None, t0: int = 0, t1: int | None = None,
             human_state: str | None = None, human_pid: str | None = None, corrected_ts: int | None = None,
             crop_is_face: bool = True) -> Sighting:
    base = 1_792_000_000_000
    return Sighting(
        sighting_id=sid, did=did, room_id="9", facility_id=facility, track_id=None,
        board_state=board_state, board_person_id=board_pid if board_state == "known" else None,
        provisional_person_id=None, best_match_score=None, crop_quality=None, crop_is_face=crop_is_face,
        crop_count=1, image_url=f"https://cdn.example/{sid}.jpg", capture_ts=base + (t1 if t1 is not None else t0),
        track_started_ts=base + t0, track_ended_ts=base + (t1 if t1 is not None else t0 + 1000),
        human_state=human_state, human_person_id=human_pid if human_state == "known" else None,
        corrected_ts=corrected_ts, day=day)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


SIGHTING_HEADER = [
    "id", "sighting_id", "did", "room_id", "facility_id", "track_id", "camera", "camera_name",
    "identity_state", "person_id", "provisional_person_id", "best_match_score", "crop_quality",
    "crop_is_face", "bbox", "crop_count", "image_url", "evidence_pending", "capture_ts", "captured_at",
    "track_started_ts", "track_ended_ts", "corrected_state", "corrected_person_id", "corrected_by",
    "corrected_at", "corrected_ts", "reconcile_status", "reconciled_at", "extras", "state", "created_at",
    "matched_via"]


def write_export(root: Path, sightings: list[dict], people: list[tuple[str, str, str]],
                 facilities=(("2", "2"), ("3", "2"))) -> None:
    """people: (person_id, facility_id, person_type)."""
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "facilities.csv", ["id", "org_id", "time_zone", "state", "created_at"],
              [[f, o, "Asia/Shanghai", 1, ""] for f, o in facilities])
    write_csv(root / "face_gallery.csv",
              ["id", "facility_id", "person_id", "person_type", "photo_url", "tentative", "merged_into", "state",
               "created_at", "updated_at"],
              [[i, f, pid, t, "", 0, "", 1, "", ""] for i, (pid, f, t) in enumerate(people)])
    write_csv(root / "person_roster.csv",
              ["id", "did", "facility_id", "person_id", "merged_into", "state", "created_at", "updated_at"], [])
    write_csv(root / "person_identity_role.csv",
              ["id", "facility_id", "person_id", "role", "state", "created_at", "updated_at"], [])
    rows = []
    for i, s in enumerate(sightings):
        r = {k: "" for k in SIGHTING_HEADER}
        r.update({"id": i + 1, "state": 1, "crop_count": 1, "crop_is_face": 1, "did": "45", "room_id": "9",
                  "identity_state": "stranger"})
        r.update(s)
        rows.append([r[k] for k in SIGHTING_HEADER])
    write_csv(root / "person_sightings.csv", SIGHTING_HEADER, rows)


def write_face_cache(emb: Path, facility: str, day: str, rows: list[tuple[str, int, np.ndarray, float, float]]) -> None:
    """rows: (sid, crop, vec, qf, face_h). Measures are chosen so the default policy reproduces
    ``qf`` exactly for face_h >= 40 (det = qf; blur, residual, asym, yaw neutral)."""
    d = _dir(emb, "face", FACE_ID, facility)
    n = len(rows)
    np.savez(d / f"{day}.npz", schema=np.int64(2),
             sighting_id=np.array([r[0] for r in rows], dtype=object), crop_idx=np.array([r[1] for r in rows]),
             source=np.array(["cdn"] * n, dtype=object),
             vec=np.stack([r[2] for r in rows]).astype(np.float32) if rows else np.zeros((0, D), np.float32),
             face_h=np.array([r[4] for r in rows], dtype=float), det=np.array([r[3] for r in rows], dtype=float),
             blur=np.full(n, 1000.0), residual=np.zeros(n), asym=np.zeros(n), yaw=np.zeros(n))


def write_body_cache(emb: Path, facility: str, day: str, rows: list[tuple[str, int, np.ndarray]]) -> None:
    d = _dir(emb, "body", BODY_ID, facility)
    np.savez(d / f"{day}.npz",
             sighting_id=np.array([r[0] for r in rows], dtype=object), crop_idx=np.array([r[1] for r in rows]),
             source=np.array(["cdn"] * len(rows), dtype=object),
             vec=np.stack([r[2] for r in rows]).astype(np.float32) if rows else np.zeros((0, D), np.float32))


__all__ = ["D", "FACE_ID", "BODY_ID", "unit", "basis", "near", "sighting", "write_export",
           "write_face_cache", "write_body_cache", "slug"]
