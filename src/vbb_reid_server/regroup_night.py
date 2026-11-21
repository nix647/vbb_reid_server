"""The sorter's inputs and rows for one facility-night (docs/DESIGN_nightly_regroup.md, "Wiring", W2).

``regroup_night`` is what the nightly run calls: it builds the items the sorter works on from the export, the face
cache, the check-page answers and the earlier nights, runs ``regroup.regroup``, and turns each probe the sorter
names into a machine row. ``build_items`` is the item building alone, so the same items can be rebuilt elsewhere (the
evaluation harness). Files only: no network, the caller passes everything it pulled from the backend.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping

from .decide import MachineLabel
from .embed import StaleCache
from .pipeline import RefPolicy, Workspace
from .regroup import Item, Params, regroup
from .regroup_inputs import EarlierPolicy, earlier_items, usable_faces
from .vetting import VetPolicy


def _staff_name(s, ex, roster) -> str | None:
    """The name staff gave without anyone confirming it (a ``known`` label without the reviewed mark)."""
    if s is not None and s.human_state == "known" and not s.human_reviewed:
        k = ex.key(s.human_person_id)
        return k if k in roster else None
    return None


def _board_name(s, ex, roster) -> str | None:
    k = ex.key(s.board_person_id) if s.board_state == "known" and s.board_person_id else None
    return k if k in roster else None


def _human_check(sid, s, ex, roster, trusted, info, cutoff_ms):
    """The human check that existed before the cutoff: ("person", key), ("nobody", None) for an answer that says
    the picture is not one of the roster, or (None, None) when nobody had checked it yet."""
    if trusted is not None and sid in trusted:
        at = (info.get(sid) or (None, None, None))[2]
        if at is not None and at < cutoff_ms:  # an answer saved after the run started is not known to it
            person = trusted[sid]
            k = ex.key(person) if person is not None else None
            return ("person", k) if k is not None and k in roster else ("nobody", None)
    if s is not None and s.human_state == "known" and s.human_reviewed \
            and s.corrected_ts is not None and s.corrected_ts < cutoff_ms:
        k = ex.key(s.human_person_id)
        if k in roster:
            return "person", k
    return None, None


class _Faces:
    """sid -> (vector, norm) of each night's usable faces, one face table read per night."""

    def __init__(self, ws: Workspace, facility: str, ref_policy: RefPolicy, vet: VetPolicy):
        self.ws, self.facility, self.ref_policy, self.vet, self.cache = ws, facility, ref_policy, vet, {}

    def night(self, day: str) -> dict:
        if day not in self.cache:
            try:
                self.cache[day] = usable_faces(self.ws.face_table(self.facility, day), self.ref_policy, self.vet)
            except (FileNotFoundError, OSError, KeyError, ValueError, StaleCache):
                self.cache[day] = {}
        return self.cache[day]


def build_items(ws: Workspace, facility: str, day: str, *, label_cutoff_ms: int, ref_policy: RefPolicy,
                vet: VetPolicy | None = None, trusted=None):
    """The sorter's items for the night: every probe (anchor, weak name, face, body) and the earlier confirmed pictures
    of the facility inside the reference window. -> (items, nobody) where ``nobody`` holds the probes a human answered
    "not on the list" (they get no machine row)."""
    ex, vet = ws.export, vet or VetPolicy()
    roster = ex.roster(facility)
    info = getattr(trusted, "info", {})
    faces = _Faces(ws, facility, ref_policy, vet)
    probes = ex.by_facility_day(facility, day)
    body = ws.probe_inputs(facility, day)[1]
    tonight = faces.night(day)
    items: list[Item] = []
    nobody: set[str] = set()
    for s in probes:
        kind, anchor = _human_check(s.sighting_id, s, ex, roster, trusted, info, label_cutoff_ms)
        if kind == "nobody":
            nobody.add(s.sighting_id)
        weak = _staff_name(s, ex, roster) or _board_name(s, ex, roster)
        face = tonight.get(s.sighting_id)
        items.append(Item(s.sighting_id, True, weak, anchor, None if face is None else face[0],
                          body.get(s.sighting_id), None if face is None else face[1]))
    # earlier confirmed pictures: the export's references and the answers found through Trusted.info
    start = (date.fromisoformat(day) - timedelta(days=ref_policy.window_days)).isoformat()
    seen = {s.sighting_id for s in probes}
    earlier: dict[str, tuple[str, str, str | None]] = {}  # sid -> (day, anchor, weak)
    for s in ex.sightings:
        if s.sighting_id in seen or s.facility_id != facility or not start <= s.day < day:
            continue
        kind, anchor = _human_check(s.sighting_id, s, ex, roster, trusted, info, label_cutoff_ms)
        if kind == "person":
            earlier[s.sighting_id] = (s.day, anchor, _staff_name(s, ex, roster))
    for sid in sorted(trusted or ()):
        inf = info.get(sid)
        if sid in seen or sid in earlier or inf is None or inf[0] != facility or not start <= inf[1] < day:
            continue
        kind, anchor = _human_check(sid, None, ex, roster, trusted, info, label_cutoff_ms)
        if kind == "person":
            earlier[sid] = (inf[1], anchor, None)
    for sid, (d, anchor, weak) in sorted(earlier.items()):
        face = faces.night(d).get(sid)
        if face is not None:
            items.append(Item(sid, False, weak, anchor, face[0], None, face[1]))
    return items, nobody


def _earlier_inputs(ws, facility, day, trusted, overlays, earlier: EarlierPolicy):
    """confirmed / touched / capture_ts / skip_days for ``earlier_items`` (E3, E5 and the fail-closed rule)."""
    ex = ws.export
    info = getattr(trusted, "info", {})
    confirmed: dict[str, str | None] = {}
    for s in ex.sightings:
        if s.facility_id == facility and s.human_state == "known" and s.human_reviewed:
            confirmed[s.sighting_id] = ex.key(s.human_person_id)
    for sid, person in (trusted or {}).items():  # an answer settles the picture it covers
        if (info.get(sid) or (facility,))[0] == facility:
            confirmed[sid] = ex.key(person) if person is not None else None
    d0 = date.fromisoformat(day)
    nights = [(d0 - timedelta(days=k)).isoformat() for k in range(1, earlier.nights + 1)]
    touched = set(getattr(trusted, "cant_tell", ()))  # "can't tell" is a human check too
    skip_days, capture_ts = set(), {}
    for d in nights:
        rows = overlays.get(d)
        if rows is None:  # the pull failed (or was never made): its overlays are unknown
            skip_days.add(d)
            continue
        for r in rows:
            sid = str(r["sighting_id"])
            capture_ts[sid] = int(r.get("capture_ts") or 0)
            if r.get("human_labeled"):
                touched.add(sid)
    return confirmed, touched, capture_ts, skip_days


def _row(s, name, ex, roster) -> MachineLabel:
    person = name.person
    board = ex.key(s.board_person_id) if s.board_state == "known" else None
    relation = "agree" if board == person else "relabel" if s.board_state == "known" else "promote"
    score = None if name.score is None else float(name.score)
    return MachineLabel(
        s.sighting_id, "known", relation, person, roster[person], name.via, 0.0 if score is None else score, None,
        score if name.via == "face" else None, score if name.via == "body" else None,
        {"reason": name.reason, "floor": None if name.floor is None else float(name.floor),
         "others": list(name.others)})


def _summary(result, report, n_rows: int) -> dict:
    names = result.names.values()
    return {
        "reasons": dict(sorted(Counter(n.reason for n in names).items())),
        "via": dict(sorted(Counter(str(n.via) for n in names).items())),
        "rows": n_rows,
        "face_groups": {k: {"members": g.n_members, "anchors": g.n_anchor, "floor": g.floor,
                            "floor_hard": g.floor_hard, "active": g.active, "anchor_seeded": g.anchor_seeded}
                        for k, g in sorted(result.face_groups.items())},
        "body_groups": {k: {"members": g.n_members, "floor": g.floor, "active": g.active}
                        for k, g in sorted(result.body_groups.items())},
        "disputes": [[a, b] for a, b, _, _ in result.disputes],
        "dissolved": [[w, list(p)] for w, p in result.dissolved],
        "displaced": [list(x) for x in result.displaced],
        "refuted": list(result.refuted),
        "rounds": result.rounds,
        "earlier": {
            "nights": [{"day": n.day, "status": n.status, "face_names": n.face_names, "added": n.added}
                       for n in report.nights],
            "added": dict(sorted(report.added.items())),
            "checked": {k: list(v) for k, v in sorted(report.checked.items())},
            "dropped": list(report.dropped),
        },
    }


def regroup_night(ws: Workspace, facility: str, day: str, *, label_cutoff_ms: int, ref_policy: RefPolicy,
                  vet: VetPolicy | None = None, trusted=None, runs_dir: str | Path,
                  overlays: Mapping[str, list[dict] | None], earlier: EarlierPolicy = EarlierPolicy(),
                  params: Params = Params()) -> dict:
    """Sort one facility-night. ``overlays``: earlier night -> that night's pulled sighting rows (``None`` = the pull
    failed). -> {"labels", "sightings" (the probes), "result", "earlier" (the report), "summary"}."""
    ex = ws.export
    roster = ex.roster(facility)
    vet = vet or VetPolicy()
    items, nobody = build_items(ws, facility, day, label_cutoff_ms=label_cutoff_ms, ref_policy=ref_policy, vet=vet,
                                trusted=trusted)
    confirmed, touched, capture_ts, skip_days = _earlier_inputs(ws, facility, day, trusted, overlays, earlier)
    extra, report = earlier_items(facility=facility, day=day, runs_dir=runs_dir, face_table=ws.face_table, key=ex.key,
                                  roster=set(roster), tonight=items, confirmed=confirmed, touched=touched,
                                  capture_ts=capture_ts, skip_days=skip_days, policy=earlier, ref_policy=ref_policy,
                                  vet=vet)
    result = regroup([*items, *extra], params)
    probes = ex.by_facility_day(facility, day)
    labels = []
    for s in probes:
        name = result.names.get(s.sighting_id)
        if name is None or name.person is None or name.via not in ("face", "body") or s.sighting_id in nobody:
            continue  # an unnamed probe, an anchor (a human's own answer), or a picture a human said is nobody
        labels.append(_row(s, name, ex, roster))
    return {"labels": labels, "sightings": probes, "result": result, "earlier": report,
            "summary": _summary(result, report, len(labels))}
