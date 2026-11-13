"""Learn only from pictures someone looked at (owner 2026-11-11/12, plan decision 34).

A staff rename covers a whole row or a whole person, but the person looked at only some of its pictures. The
backend keeps a "reviewed" mark per picture (``v_person_sightings.extras.reviewed``, served as ``reviewed`` on
``references/pull``, read into ``Sighting.human_reviewed``):

- a row renamed with nothing ticked: only its main photo;
- ticked pictures: those;
- an action at the top of a person's page: only the profile photo.

Everything else counts as not reviewed, renames from before the mark existed included.

A staff ``known`` label is trusted when:

1. **You answered the picture** on the check page (``refs.json`` or a finished night). Your answer settles it:
   trusted when it names the label's person (``answered``), dropped otherwise (``answer_disagrees``), whatever the
   staff mark says.
2. Otherwise **it carries the reviewed mark** (``reviewed``). Without it: ``not_reviewed``, not learned from.

A picture answered on the check page that no user renamed is learned from too (owner 2026-11-13: every confirmed
picture is): :func:`answered_labels` turns each answer naming a person into a label, found in that night's cache.

A trusted label seeds the same day's body clusters. It also becomes a face reference only with one clear face: at
least one crop with exactly one face (``n_faces == 1``) and a feature norm of at least ``min_norm``. With two faces
the machine can't tell which one is the person; such a label is ``two_faces`` (or ``no_face`` / ``poor_face``)
and teaches only the body match. The rule holds per crop (:func:`clear_crop`): a cache built from board crops has
several crops per sighting, and only the clear ones of a trusted sighting become references.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from .data import Sighting

FACE_OK = ("answered", "reviewed")  # trusted, with a clear crop: teaches the face and seeds the day
NO_CLEAR_FACE = ("no_face", "two_faces", "poor_face")  # trusted, no clear crop: seeds the day only
# Settled: someone looked at it or you answered it. Never put on the check page as a disagreement.
SETTLED = FACE_OK + NO_CLEAR_FACE + ("answer_disagrees",)


@dataclass(frozen=True)
class VetPolicy:
    min_norm: float = 14.0  # a face example needs a crop with one face and at least this feature norm


@dataclass
class Vetting:
    status: dict[str, str] = field(default_factory=dict)  # sid -> one of FACE_OK / NO_CLEAR_FACE / answer_disagrees / not_reviewed

    @property
    def trusted(self) -> set[str]:
        """Labels the run may learn from: each seeds its day's body clusters, and its clear crops
        (:func:`clear_crop`) are face references. The status only reports whether it has any."""
        return {sid for sid, st in self.status.items() if st in FACE_OK or st in NO_CLEAR_FACE}

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for st in self.status.values():
            out[st] = out.get(st, 0) + 1
        return dict(sorted(out.items()))


class Trusted(dict):
    """sid -> person_id (None = someone not on the list); ``info[sid]`` = (facility, day, answered at, ms): where the
    picture's face is cached and when the answer existed (a run learns only from answers given before it started)."""

    def __init__(self, *a, info: dict | None = None, **k):
        super().__init__(*a, **k)
        self.info: dict[str, tuple[str, str, int]] = dict(info or {})


def load_trusted(verify_root: str | Path | None) -> Trusted:
    """sid -> person_id you said it shows (None = someone not on the list), from ``refs.json`` and every finished
    night's ``marks.json``. A picture given two different answers is left out."""
    if verify_root is None:
        return Trusted()
    root = Path(verify_root)
    seen: dict[str, set] = {}
    info: dict[str, tuple[str, str, int]] = {}
    refs = root / "refs.json"
    if refs.exists():
        rj = json.loads(refs.read_text(encoding="utf-8"))
        for p in rj.get("people") or []:
            for ph in p.get("photos") or []:
                seen.setdefault(str(ph["sighting_id"]), set()).add(str(p["person_id"]))
                if rj.get("facility_id") and ph.get("day"):
                    info[str(ph["sighting_id"])] = (str(rj["facility_id"]), str(ph["day"]), int(rj.get("built_at") or 0) * 1000)
    for fz_path in sorted(root.glob("*/frozen.json")):
        mk_path = fz_path.with_name("marks.json")
        if not mk_path.exists():
            continue
        fz = json.loads(fz_path.read_text(encoding="utf-8"))
        mk = json.loads(mk_path.read_text(encoding="utf-8"))
        if not mk.get("submitted"):  # a night still being answered may change
            continue
        answers = mk.get("answers") or {}
        by_key = {p["key"]: str(p["person_id"]) for p in fz.get("people") or []}
        for i, it in enumerate(fz.get("items") or []):
            a = answers.get(str(i))
            if a is None or a == "cant_tell":
                continue
            seen.setdefault(str(it["sighting_id"]), set()).add(by_key.get(a))  # "other" -> None
            if fz.get("facility_id") and fz.get("day"):
                info[str(it["sighting_id"])] = (str(fz["facility_id"]), str(fz["day"]), int(mk.get("updated_at") or 0) * 1000)
    keep = {sid: next(iter(v)) for sid, v in seen.items() if len(v) == 1}
    return Trusted(keep, info={sid: info[sid] for sid in keep if sid in info})


def answered_labels(ex, trusted: dict) -> list[Sighting]:
    """Pictures answered on the check page with a person, as labels to learn from (owner 2026-11-13: every confirmed
    picture is learned from): those no user renamed, and those a user renamed to someone else (the answer wins;
    :func:`learned_as` keeps the rename out). A rename the answer agrees with is learned as it is. The face comes from
    that night's cache (``Trusted.info``); ``corrected_ts`` is when the answer was saved, so a run never learns an
    answer given after it started."""
    renamed = {s.sighting_id: ex.key(s.human_person_id) for s in ex.sightings if s.human_state == "known"}
    info = getattr(trusted, "info", {})
    out = []
    for sid, pid in sorted(trusted.items()):
        if pid is None or sid not in info or renamed.get(sid, "") == ex.key(pid):
            continue
        fac, day, at_ms = info[sid]
        midnight = int(datetime.combine(date.fromisoformat(day), datetime.min.time(), timezone.utc).timestamp() * 1000)
        out.append(Sighting(sighting_id=sid, did="", room_id="", facility_id=fac, track_id=None,
                            board_state="uncertain", board_person_id=None, provisional_person_id=None,
                            best_match_score=None, crop_quality=None, crop_is_face=True, crop_count=1, image_url=None,
                            capture_ts=midnight + 43_200_000, track_started_ts=None, track_ended_ts=None,
                            human_state="known", human_person_id=pid, corrected_ts=at_ms, day=day))
    return out


def learned_as(vetting: Vetting, trusted: dict, key) -> "callable":
    """(sid, identity key) -> whether the run learns that picture as that person: a trusted label, and for a picture
    you answered only under your answer (a rename to someone else shares the sighting id and must stay out)."""
    ok = vetting.trusted

    def check(sid: str, k) -> bool:
        return sid in ok and (sid not in trusted or key(trusted[sid]) == k)
    return check


def clear_crop(ft, i: int, policy: VetPolicy) -> bool:
    """Crop ``i`` shows exactly one face with a usable feature (a nan norm fails: unknown is not clear)."""
    return int(ft.n_faces[i]) == 1 and float(ft.norm[i]) >= policy.min_norm


def _clear_face(ft, rows: list[int], policy: VetPolicy) -> str:
    """'' when one of the picture's crops is a :func:`clear_crop`, else the reason."""
    if not rows:
        return "no_face"
    if not any(int(ft.n_faces[i]) == 1 for i in rows):
        return "two_faces"
    if not any(clear_crop(ft, i, policy) for i in rows):
        return "poor_face"
    return ""


def vet_labels(ws, labels: list[Sighting], trusted: dict[str, str | None], policy: VetPolicy) -> Vetting:
    """``labels``: the human ``known`` labels the run may learn from. ``trusted``: :func:`load_trusted`."""
    ex = ws.export
    out = Vetting()
    tables: dict[tuple[str, str], tuple] = {}

    def face_status(s: Sighting) -> str:
        k = (s.facility_id, s.day)
        if k not in tables:
            try:
                ft = ws.face_table(*k)
            except (FileNotFoundError, OSError, KeyError, ValueError):
                ft = None
            rows: dict[str, list[int]] = {}
            for i, sid in enumerate(ft.sighting_id if ft is not None else []):
                rows.setdefault(str(sid), []).append(i)
            tables[k] = (ft, rows)
        ft, rows = tables[k]
        return _clear_face(ft, rows.get(s.sighting_id, []), policy) if ft is not None else "no_face"

    for s in labels:
        key = ex.key(s.human_person_id)
        if key is None:
            continue
        if s.sighting_id in trusted:
            t = trusted[s.sighting_id]
            if t is None or ex.key(t) != key:
                out.status[s.sighting_id] = "answer_disagrees"
                continue
            ok = "answered"
        elif s.human_reviewed:
            ok = "reviewed"
        else:
            out.status[s.sighting_id] = "not_reviewed"
            continue
        out.status[s.sighting_id] = face_status(s) or ok
    return out
