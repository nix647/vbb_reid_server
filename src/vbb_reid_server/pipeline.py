"""Assemble decision inputs from the export + embedding cache.

References (plan §5.3): face embeddings of crops of human-labeled ``known``
sightings, never machine labels (no self-training in v1). Which labels count is
the caller's choice through :class:`RefPolicy` — production uses "labels that
existed at run time"; the evaluation also offers "leave one day out" to compare
models with the most references possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from .data import Export, Sighting
from .embed import BodyTable, FaceTable, load_body, load_face
from .models.base import l2n


@dataclass(frozen=True)
class RefPolicy:
    window_days: int = 60  # only labels on sightings captured within this many days before the probe day
    ref_q_min: float = 0.40  # reference faces need at least this quality
    max_refs: int = 300  # most recent per person
    include_same_day: bool = False  # production: True (labels made before the run); eval temporal: False


class Workspace:
    def __init__(self, export: Export, emb_root: str | Path, face_id: str | None, body_id: str | None):
        self.export = export
        self.emb_root = Path(emb_root)
        self.face_id = face_id
        self.body_id = body_id
        self._face: dict[tuple[str, str], FaceTable] = {}
        self._body: dict[tuple[str, str], BodyTable] = {}
        self._by_id = {s.sighting_id: s for s in export.sightings}

    # ------------------------------------------------------------- tables --
    def face_table(self, facility: str, day: str) -> FaceTable:
        k = (facility, day)
        if k not in self._face:
            self._face[k] = load_face(self.emb_root, self.face_id, facility, day) if self.face_id else FaceTable.empty()
        return self._face[k]

    def body_table(self, facility: str, day: str) -> BodyTable:
        k = (facility, day)
        if k not in self._body:
            self._body[k] = load_body(self.emb_root, self.body_id, facility, day) if self.body_id else BodyTable.empty()
        return self._body[k]

    def sighting(self, sid: str) -> Sighting | None:
        return self._by_id.get(sid)

    # ------------------------------------------------------------- probes --
    def probe_inputs(self, facility: str, day: str):
        """-> (probe_faces sid->[(crop, vec, qf, face_h)], body sid->unit vec, body_crops sid->[vec])."""
        ft = self.face_table(facility, day)
        probe_faces: dict[str, list[tuple[int, np.ndarray, float, float]]] = {}
        for i in range(len(ft)):
            probe_faces.setdefault(str(ft.sighting_id[i]), []).append(
                (int(ft.crop_idx[i]), ft.vec[i], float(ft.qf[i]), float(ft.face_h[i])))
        bt = self.body_table(facility, day)
        body_crops: dict[str, list[np.ndarray]] = {}
        for i in range(len(bt)):
            body_crops.setdefault(str(bt.sighting_id[i]), []).append(bt.vec[i])
        body = {sid: l2n(np.mean(v, axis=0)) for sid, v in body_crops.items()}
        return probe_faces, body, body_crops

    # --------------------------------------------------------- references --
    def references(self, facilities: list[str], probe_day: str, *, label_cutoff_ms: int | None,
                   policy: RefPolicy = RefPolicy(), exclude_keys: set[str] | None = None,
                   exclude_days: set[str] | None = None, allow_after: bool = False,
                   label_filter: Callable[[Sighting], bool] | None = None) -> dict[str, np.ndarray]:
        """Face reference vectors per identity key.

        - Labels: human ``known`` only, ``corrected_ts < label_cutoff_ms`` when given.
        - Sightings captured on days ``< probe_day`` (``<=`` with ``include_same_day``),
          within ``window_days``; ``allow_after`` lifts the upper bound (leave-one-day-out).
        - ``exclude_keys``: people removed (leave-one-person-out). ``exclude_days``: days removed.
        """
        exclude_keys = exclude_keys or set()
        exclude_days = exclude_days or set()
        pd = date.fromisoformat(probe_day)
        lo = (pd - timedelta(days=policy.window_days)).isoformat()
        rows: dict[str, list[tuple[int, np.ndarray]]] = {}
        wanted: dict[tuple[str, str], list[Sighting]] = {}
        for s in self.export.sightings:
            if s.facility_id not in facilities or s.human_state != "known":
                continue
            if label_cutoff_ms is not None and (s.corrected_ts is None or s.corrected_ts >= label_cutoff_ms):
                continue
            if s.day in exclude_days or s.day < lo:
                continue
            if not allow_after:
                if s.day > probe_day or (s.day == probe_day and not policy.include_same_day):
                    continue
            if label_filter is not None and not label_filter(s):
                continue
            k = self.export.key(s.human_person_id)
            if k is None or k in exclude_keys:
                continue
            wanted.setdefault((s.facility_id, s.day), []).append(s)
        for (fac, day), ss in wanted.items():
            ft = self.face_table(fac, day)
            if len(ft) == 0:
                continue
            pos: dict[str, list[int]] = {}
            for i, sid in enumerate(ft.sighting_id):
                pos.setdefault(str(sid), []).append(i)
            for s in ss:
                for i in pos.get(s.sighting_id, []):
                    if ft.qf[i] >= policy.ref_q_min:
                        rows.setdefault(self.export.key(s.human_person_id), []).append((s.capture_ts, ft.vec[i]))
        out: dict[str, np.ndarray] = {}
        for k, lst in rows.items():
            lst.sort(key=lambda t: -t[0])
            out[k] = np.stack([v for _, v in lst[:policy.max_refs]]).astype(np.float32)
        return out

    def human_seeds(self, facility: str, day: str, label_cutoff_ms: int | None,
                    exclude_keys: set[str] | None = None) -> dict[str, str]:
        exclude_keys = exclude_keys or set()
        out = {}
        for s in self.export.by_facility_day(facility, day):
            if s.human_state == "known" and (label_cutoff_ms is None or (s.corrected_ts or 0) < label_cutoff_ms):
                k = self.export.key(s.human_person_id)
                if k and k not in exclude_keys:
                    out[s.sighting_id] = k
        return out


def org_facilities(export: Export, facility: str) -> list[str]:
    org = export.facility_org.get(facility)
    return export.facility_of_org(org) if org is not None else [facility]
