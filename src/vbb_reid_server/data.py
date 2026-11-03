"""Load the read-only production export (``export_machine_eval.php`` CSVs).

Vocabulary follows the nursery DB:
- board prediction: ``identity_state`` in {known, uncertain, stranger}, ``person_id``,
  ``provisional_person_id`` — immutable.
- human overlay: ``corrected_state`` in {known, stranger}, ``corrected_person_id``,
  ``corrected_ts`` (ms). ``corrected_at`` set with ``corrected_state`` empty = reset
  to pending (an un-label).

People are compared through an *identity key*: by default the facility-local
``person_id``; with an alias file (``person_id,human[,name]``) the same human
registered in several facilities shares one key (owner 2026-11-03: facilities
2 and 3 hold the same four people under different ids).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BOARD_STATES = ("known", "uncertain", "stranger")


def _int(v: str | None) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _str(v: str | None) -> str | None:
    return v if v not in (None, "") else None


@dataclass(frozen=True)
class Sighting:
    sighting_id: str
    did: str
    room_id: str
    facility_id: str
    track_id: int | None
    board_state: str
    board_person_id: str | None
    provisional_person_id: str | None
    best_match_score: float | None
    crop_quality: float | None
    crop_is_face: bool
    crop_count: int
    image_url: str | None
    capture_ts: int  # ms
    track_started_ts: int | None  # ms
    track_ended_ts: int | None  # ms
    human_state: str | None  # known | stranger | reset | None
    human_person_id: str | None
    corrected_ts: int | None  # ms
    day: str  # facility-local YYYY-MM-DD of capture_ts

    @property
    def human_labeled(self) -> bool:
        return self.human_state in ("known", "stranger")

    @property
    def interval_ms(self) -> tuple[int, int]:
        start = self.track_started_ts if self.track_started_ts is not None else self.capture_ts
        end = self.track_ended_ts if self.track_ended_ts is not None else self.capture_ts
        if end < start:
            start, end = end, start
        return start, end


@dataclass
class Person:
    person_id: str
    facility_id: str
    person_type: str | None = None  # face_gallery: nurse | visitor
    role: str | None = None
    tentative: bool = False
    merged_into: str | None = None


@dataclass
class Export:
    root: Path
    sightings: list[Sighting]
    people: dict[str, Person]  # person_id -> Person
    facility_tz: dict[str, str]
    facility_org: dict[str, str]
    aliases: dict[str, str] = field(default_factory=dict)  # person_id -> human key
    names: dict[str, str] = field(default_factory=dict)  # human key -> display name (local only)

    # ------------------------------------------------------------ identity --
    def key(self, person_id: str | None) -> str | None:
        """Identity key of a person id: alias if one is mapped, else the id itself.
        Merges (``merged_into``) are followed one hop, as the backend does."""
        if not person_id:
            return None
        p = self.people.get(person_id)
        if p is not None and p.merged_into:
            person_id = p.merged_into
        return self.aliases.get(person_id, person_id)

    def roster(self, facility_id: str) -> dict[str, str]:
        """Closed roster of a facility: identity key -> that facility's person_id.

        Every non-tentative, non-merged registered id of the facility. Visitors are
        included: in the test facilities every registered person is a ``visitor``
        (export 2026-11-03), so excluding them would empty the roster."""
        out: dict[str, str] = {}
        for pid, p in sorted(self.people.items()):
            if p.facility_id != facility_id or p.tentative or p.merged_into:
                continue
            k = self.key(pid)
            if k is not None:
                out.setdefault(k, pid)
        return out

    def facility_of_org(self, org_id: str) -> list[str]:
        return sorted(f for f, o in self.facility_org.items() if o == org_id)

    def by_facility_day(self, facility_id: str, day: str) -> list[Sighting]:
        return [s for s in self.sightings if s.facility_id == facility_id and s.day == day]

    def days(self, facility_id: str) -> list[str]:
        return sorted({s.day for s in self.sightings if s.facility_id == facility_id})


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def local_day(ms: int, tz_name: str) -> str:
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(ms / 1000.0, tz).strftime("%Y-%m-%d")


def load_aliases(path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    """``person_id,human[,name]`` -> ({person_id: human}, {human: name})."""
    if path is None:
        return {}, {}
    aliases: dict[str, str] = {}
    names: dict[str, str] = {}
    for r in _rows(Path(path)):
        pid, human = (r.get("person_id") or "").strip(), (r.get("human") or "").strip()
        if pid and human:
            aliases[pid] = human
            if (r.get("name") or "").strip():
                names[human] = r["name"].strip()
    return aliases, names


def load_export(root: str | Path, aliases_csv: str | Path | None = None) -> Export:
    root = Path(root)
    facility_tz: dict[str, str] = {}
    facility_org: dict[str, str] = {}
    for r in _rows(root / "facilities.csv"):
        facility_tz[r["id"]] = r.get("time_zone") or "Asia/Shanghai"
        facility_org[r["id"]] = r.get("org_id") or "0"

    people: dict[str, Person] = {}
    for r in _rows(root / "face_gallery.csv"):
        if r.get("state", "1") != "1":
            continue
        people[r["person_id"]] = Person(
            person_id=r["person_id"], facility_id=r["facility_id"],
            person_type=_str(r.get("person_type")), tentative=r.get("tentative") == "1",
            merged_into=_str(r.get("merged_into")))
    for r in _rows(root / "person_roster.csv"):
        if r.get("state", "1") != "1":
            continue
        p = people.get(r["person_id"])
        if p is None:
            people[r["person_id"]] = Person(
                person_id=r["person_id"], facility_id=r.get("facility_id") or "",
                merged_into=_str(r.get("merged_into")))
        elif not p.merged_into and _str(r.get("merged_into")):
            p.merged_into = r["merged_into"]
    role_path = root / "person_identity_role.csv"
    if role_path.exists():
        for r in _rows(role_path):
            if r.get("state", "1") != "1":
                continue
            p = people.get(r["person_id"])
            if p is None:
                p = people[r["person_id"]] = Person(person_id=r["person_id"], facility_id=r["facility_id"])
            p.role = _str(r.get("role"))

    sightings: list[Sighting] = []
    for r in _rows(root / "person_sightings.csv"):
        if r.get("state", "1") != "1":
            continue
        cap = _int(r.get("capture_ts"))
        if cap is None:
            continue
        fac = r.get("facility_id") or ""
        if r.get("corrected_at"):
            hs = r.get("corrected_state") or "reset"
        else:
            hs = None
        board_state = r.get("identity_state") or "uncertain"
        if board_state not in BOARD_STATES:
            board_state = "uncertain"
        sightings.append(Sighting(
            sighting_id=r["sighting_id"], did=r.get("did") or "", room_id=r.get("room_id") or "",
            facility_id=fac, track_id=_int(r.get("track_id")),
            board_state=board_state,
            board_person_id=_str(r.get("person_id")) if board_state == "known" else None,
            provisional_person_id=_str(r.get("provisional_person_id")),
            best_match_score=_float(r.get("best_match_score")),
            crop_quality=_float(r.get("crop_quality")),
            crop_is_face=r.get("crop_is_face") == "1",
            crop_count=_int(r.get("crop_count")) or 1,
            image_url=_str(r.get("image_url")),
            capture_ts=cap,
            track_started_ts=_int(r.get("track_started_ts")),
            track_ended_ts=_int(r.get("track_ended_ts")),
            human_state=hs,
            human_person_id=_str(r.get("corrected_person_id")) if hs == "known" else None,
            corrected_ts=_int(r.get("corrected_ts")),
            day=local_day(cap, facility_tz.get(fac, "Asia/Shanghai")),
        ))
    sightings.sort(key=lambda s: (s.facility_id, s.capture_ts, s.sighting_id))
    aliases, names = load_aliases(Path(aliases_csv) if aliases_csv else None)
    return Export(root=root, sightings=sightings, people=people,
                  facility_tz=facility_tz, facility_org=facility_org,
                  aliases=aliases, names=names)
