"""Earlier nights for the nightly regroup (docs/DESIGN_nightly_regroup.md, "Earlier nights (item 122)", E1-E7).

A quiet night leaves some people with too few pictures for a bar. This reads the face names the nightly job gave on
the nights before and offers them, as weak pictures, to the people who are thin tonight. It reads files only: run
records, label files and the face cache through a ``face_table`` function the caller passes. No network.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from .embed import StaleCache
from .pipeline import RefPolicy, ref_usable
from .regroup import Item
from .vetting import VetPolicy, clear_crop


@dataclass(frozen=True)
class EarlierPolicy:
    nights: int = 7  # look back this many nights before the probe night; 0 = off
    top_up: int = 30  # a person with fewer pictures than this tonight is topped up to it
    namers: tuple[str, ...] = ("decide",)  # whose label files may be read (E6)
    min_checked: int = 5  # E5: a person's earlier names need this many human-checked pictures ...
    max_disagree: float = 0.2  # ... and are not used when more than this share of them disagree

    def __post_init__(self):
        if self.nights < 0:
            raise ValueError(f"nights must not be negative, got {self.nights}")
        if self.top_up < 1:
            raise ValueError(f"top_up must be at least 1, got {self.top_up}")
        if self.min_checked < 1:
            raise ValueError(f"min_checked must be at least 1, got {self.min_checked}")
        if not 0 <= self.max_disagree <= 1:
            raise ValueError(f"max_disagree must be inside [0, 1], got {self.max_disagree}")


@dataclass(frozen=True)
class NightReport:
    day: str
    status: str  # "used" | "no_run" | "no_labels" | "namer" | "no_faces"
    face_names: int  # face names left after E2 and E3 (roster and the skips), before the top-up limit
    added: int  # pictures this night contributed


@dataclass(frozen=True)
class EarlierReport:
    nights: tuple[NightReport, ...]  # newest first
    added: dict[str, int]  # person -> pictures added
    checked: dict[str, tuple[int, int]]  # person -> (human-checked pictures, of which disagree), E5
    dropped: tuple[str, ...]  # people E5 kept out


def usable_faces(ft, ref_policy: RefPolicy, vet: VetPolicy) -> dict[str, tuple[np.ndarray, float]]:
    """sid -> (face vector, raw feature norm) of the first row of the table that is a clear crop and usable as a
    reference. The same rule makes every face the sorter sees, tonight's included."""
    out: dict[str, tuple[np.ndarray, float]] = {}
    for i in range(len(ft)):
        sid = str(ft.sighting_id[i])
        if sid not in out and clear_crop(ft, i, vet) and ref_usable(ft, i, ref_policy):
            out[sid] = (ft.vec[i].astype(np.float64), float(ft.norm[i]))
    return out


def night_run(runs_dir: str | Path, facility: str, day: str,
              namers: Iterable[str] | None = None) -> tuple[dict, Path] | None:
    """The night's real run record and the path of its labels file (which may not exist), or None. Real = ``dry_run``
    exactly false; with ``namers``, also a ``namer`` (absent = "decide") among them, checked before the latest is
    chosen; the latest ``started_at`` wins; a record that cannot be read is skipped."""
    namers = None if namers is None else tuple(namers)
    runs_dir = Path(runs_dir)
    found = []
    for f in runs_dir.glob(f"run_{facility}_{day}_*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("dry_run") is not False:
            continue
        if namers is not None and rec.get("namer", "decide") not in namers:
            continue
        started = rec.get("started_at")
        started = float(started) if isinstance(started, (int, float)) and not isinstance(started, bool) else 0.0
        found.append((started, f.name, rec, f))
    if not found:
        return None
    _, _, rec, f = max(found, key=lambda t: t[:2])
    return rec, runs_dir / f"labels_{f.stem[len('run_'):]}.csv"


def _label_rows(path: Path) -> list[dict] | None:
    """The rows of a labels file, None when the file cannot be read; a row that cannot be read is skipped."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    rows, reader = [], iter(csv.DictReader(io.StringIO(text)))
    while True:
        try:
            rows.append(next(reader))
        except StopIteration:
            return rows
        except csv.Error:
            continue


def _night_faces(face_table: Callable, facility: str, day: str, ref_policy: RefPolicy, vet: VetPolicy):
    """The night's usable faces, or None when its face table cannot be read or is empty."""
    try:
        ft = face_table(facility, day)
    except (FileNotFoundError, OSError, KeyError, ValueError, StaleCache):
        return None
    return usable_faces(ft, ref_policy, vet) if len(ft) else None


def _face_names(rows: list[dict], key: Callable, roster: set[str]) -> list[tuple[str, str]]:
    """(sid, person) of the rows that are face names (E2) of a person on tonight's roster, folded with ``key``."""
    out = []
    for row in rows:
        sid, pid = (row.get("sighting_id") or "").strip(), (row.get("person_id") or "").strip()
        if row.get("state") != "known" or row.get("via") != "face" or not sid or not pid:
            continue
        try:
            person = key(pid)
        except Exception:  # a row that cannot be folded is skipped, not fatal
            continue
        if person in roster:
            out.append((sid, person))
    return out


def earlier_items(*, facility: str, day: str, runs_dir: str | Path, face_table: Callable, key: Callable,
                  roster: Iterable[str], tonight: Iterable[Item], confirmed: Mapping[str, str | None],
                  touched: Iterable[str], capture_ts: Mapping[str, int] | None = None,
                  policy: EarlierPolicy = EarlierPolicy(), ref_policy: RefPolicy = RefPolicy(),
                  vet: VetPolicy = VetPolicy()) -> tuple[list[Item], EarlierReport]:
    """The earlier nights' face names of the nightly job, as weak ``Item``s for the people thin tonight (E1-E7)."""
    roster, touched, tonight = set(roster), set(touched), list(tonight)
    capture_ts = capture_ts or {}
    in_tonight = {it.sid for it in tonight}
    d0 = date.fromisoformat(day)

    # E1, E2, E6: the nights read, newest first, with their face names (each picture once, E5)
    nights: list[tuple[str, str, list[tuple[str, str]], dict]] = []  # (day, status, face names, usable faces)
    seen: set[str] = set()
    for back in range(1, policy.nights + 1):  # never the probe night itself or later
        d = (d0 - timedelta(days=back)).isoformat()
        run = night_run(runs_dir, facility, d, policy.namers)  # E6: only an accepted namer's runs are candidates
        names, faces, status = [], {}, "used"
        if run is None:  # real runs that only other namers made are "namer", no real run at all is "no_run"
            status = "no_run" if night_run(runs_dir, facility, d) is None else "namer"
        elif not run[1].is_file():
            status = "no_labels"
        else:
            rows = _label_rows(run[1])
            faces = _night_faces(face_table, facility, d, ref_policy, vet) if rows is not None else None
            if rows is None:
                status = "no_labels"
            elif faces is None:
                status = "no_faces"
            else:
                once: dict[str, str] = {}
                for s, p in _face_names(rows, key, roster):  # one name per picture: the first row, the newest night
                    if s not in seen:
                        once.setdefault(s, p)
                names = list(once.items())
                seen.update(once)
        nights.append((d, status, names, faces))

    # E5: drift check, per person over the face names of the nights read
    checked: dict[str, list[int]] = {}
    for _, _, names, _ in nights:
        for sid, person in names:
            if sid in confirmed:
                c = checked.setdefault(person, [0, 0])
                c[0] += 1
                c[1] += confirmed[sid] != person  # "not on the list" (None) is a disagreement too
    dropped = tuple(sorted(p for p, (n, bad) in checked.items()
                           if n >= policy.min_checked and bad / n > policy.max_disagree))

    # E4: who is thin tonight
    have: Counter = Counter()
    for it in tonight:
        if it.face is None:
            continue
        if it.anchor is not None:
            have[it.anchor] += 1
        elif it.probe and it.weak is not None:
            have[it.weak] += 1
    need = {p: 0 if p in dropped else max(0, policy.top_up - have[p]) for p in roster}

    items: list[Item] = []
    reports: list[NightReport] = []
    added: Counter = Counter()
    for d, status, names, faces in nights:
        # E3: weak, never final. Newest first within the night, ties by sid.
        left = [(s, p) for s, p in names if s not in in_tonight and s not in confirmed and s not in touched
                and s in faces]
        left.sort(key=lambda sp: (-(capture_ts.get(sp[0]) or 0), sp[0]))
        n_added = 0
        for sid, person in left:
            if need.get(person, 0) > 0:
                vec, norm = faces[sid]
                items.append(Item(sid, False, person, None, vec, None, norm))
                need[person] -= 1
                added[person] += 1
                n_added += 1
        reports.append(NightReport(d, status, len(left), n_added))
    return items, EarlierReport(tuple(reports), dict(sorted(added.items())),
                                {p: (n, bad) for p, (n, bad) in sorted(checked.items())}, dropped)
