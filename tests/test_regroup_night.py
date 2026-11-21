"""The sorter's inputs and rows for one facility-night (docs/DESIGN_nightly_regroup.md, "Wiring", W2).

Small worlds on purpose: two people, eight confirmed pictures each on an earlier day, probes of the night on top. The
sorter runs with ``min_impostors=5`` so a group of eight can have a bar.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from helpers import FACE_ID, basis, near, sighting, write_face_cache
from vbb_reid_server.data import Export, Person
from vbb_reid_server.pipeline import RefPolicy, Workspace
from vbb_reid_server.quality import QualityPolicy
from vbb_reid_server.regroup import Params
from vbb_reid_server.regroup_inputs import EarlierPolicy
from vbb_reid_server.regroup_night import build_items, regroup_night
from vbb_reid_server.vetting import Trusted, load_trusted

TZ = ZoneInfo("Asia/Shanghai")
FAC, DAY, EARLY = "3", "2026-10-20", "2026-10-18"
PARAMS = Params(min_impostors=5)


def ms(y, mo, d, h=10):
    return int(datetime(y, mo, d, h, 0, tzinfo=TZ).timestamp() * 1000)


NOW = ms(2026, 10, 21, 2)  # the run's start: the label cutoff
BEFORE, AFTER = NOW - 3_600_000, NOW + 3_600_000


class World:
    """An export, a face cache and a runs folder for facility 3."""

    def __init__(self, tmp_path, *, aliases=None):
        self.tmp, self.rng = tmp_path, np.random.default_rng(5)
        self.emb, self.runs = tmp_path / "emb", tmp_path / "runs"
        self.emb.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        self.sightings, self.faces, self.aliases = [], {}, aliases or {}
        self.people = {p: Person(p, FAC) for p in ("pA", "pB")}
        self.people["pZ"] = Person("pZ", "2")  # a person of another facility: not on this roster
        for who, axis in (("pA", 0), ("pB", 1)):  # eight confirmed pictures each, on an earlier day
            for i in range(8):
                self.add(f"c{who}{i}", EARLY, axis, human_state="known", human_pid=who, reviewed=True,
                         corrected_ts=BEFORE)

    def add(self, sid, day, axis, *, face=True, reviewed=False, **kw):
        s = sighting(sid, day=day, facility=kw.pop("facility", FAC), **kw)
        self.sightings.append(replace(s, human_reviewed=reviewed))
        if face:
            base = basis(axis) if axis is not None else -(basis(0) + basis(1)) / np.sqrt(2)  # None: unlike both
            self.faces.setdefault(day, []).append((sid, 0, near(base, 0.05, self.rng), 0.9, 80.0))
        return sid

    def build(self, **trusted_kw):
        for day, rows in self.faces.items():
            write_face_cache(self.emb, FAC, day, rows)
        if DAY not in self.faces:
            write_face_cache(self.emb, FAC, DAY, [])
        ex = Export(root=None, sightings=self.sightings, people=self.people, facility_tz={FAC: "Asia/Shanghai"},
                    facility_org={FAC: "brand"}, aliases=self.aliases)
        return Workspace(ex, self.emb, FACE_ID, None, QualityPolicy())

    def night(self, ws, trusted=None, overlays=None, earlier=EarlierPolicy(nights=0), **kw):
        return regroup_night(ws, FAC, DAY, label_cutoff_ms=NOW, ref_policy=RefPolicy(), trusted=trusted,
                             runs_dir=self.runs, overlays=overlays or {}, earlier=earlier, params=PARAMS, **kw)


def answers(**by_sid):
    """A Trusted answer set: sid -> (person id or None, saved at ms)."""
    return Trusted({sid: p for sid, (p, at) in by_sid.items()},
                   info={sid: (FAC, DAY, at) for sid, (p, at) in by_sid.items()})


def by_sid(items):
    return {i.sid: i for i in items}


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


# anchors ----------------------------------------------------------------------------------------------------------


def test_an_answer_saved_before_the_cutoff_is_an_anchor_and_one_saved_after_is_not(w):
    w.add("before", DAY, 0, board_state="known", board_pid="pB")
    w.add("after", DAY, 0, board_state="known", board_pid="pB")
    ws = w.build()
    items, _ = build_items(ws, FAC, DAY, label_cutoff_ms=NOW, ref_policy=RefPolicy(),
                           trusted=answers(before=("pA", BEFORE), after=("pA", AFTER)))
    got = by_sid(items)
    assert (got["before"].anchor, got["before"].weak) == ("pA", "pB")
    assert (got["after"].anchor, got["after"].weak) == (None, "pB")  # the run did not know that answer yet


def test_a_reviewed_staff_label_before_the_cutoff_is_an_anchor_an_unreviewed_one_is_a_weak_name(w):
    w.add("reviewed", DAY, 0, human_state="known", human_pid="pA", reviewed=True, corrected_ts=BEFORE)
    w.add("late", DAY, 0, human_state="known", human_pid="pA", reviewed=True, corrected_ts=AFTER)
    w.add("unreviewed", DAY, 0, human_state="known", human_pid="pB", corrected_ts=BEFORE)
    w.add("untimed", DAY, 0, human_state="known", human_pid="pA", reviewed=True)  # no timestamp: not known to predate
    got = by_sid(build_items(w.build(), FAC, DAY, label_cutoff_ms=NOW, ref_policy=RefPolicy())[0])
    assert (got["reviewed"].anchor, got["reviewed"].weak) == ("pA", None)
    assert got["late"].anchor is None and got["untimed"].anchor is None
    assert (got["unreviewed"].anchor, got["unreviewed"].weak) == (None, "pB")


def test_an_answer_of_not_on_the_list_gives_no_anchor_and_no_machine_row(w):
    w.add("nobody", DAY, 0)  # it looks exactly like pA, and a human said it is nobody on the list
    w.add("alien", DAY, 0)  # a human named someone of another facility
    w.add("plain", DAY, 0)
    ws = w.build()
    out = w.night(ws, trusted=answers(nobody=(None, BEFORE), alien=("pZ", BEFORE)))
    items = by_sid(build_items(ws, FAC, DAY, label_cutoff_ms=NOW, ref_policy=RefPolicy(),
                               trusted=answers(nobody=(None, BEFORE), alien=("pZ", BEFORE)))[0])
    assert items["nobody"].anchor is None and items["alien"].anchor is None
    rows = {l.sighting_id for l in out["labels"]}
    assert "plain" in rows and "nobody" not in rows and "alien" not in rows  # the sorter is not asked to overrule


def test_an_answer_overrules_the_staff_label_it_covers(w):
    w.add("both", DAY, 0, human_state="known", human_pid="pB", reviewed=True, corrected_ts=BEFORE)
    got = by_sid(build_items(w.build(), FAC, DAY, label_cutoff_ms=NOW, ref_policy=RefPolicy(),
                             trusted=answers(both=("pA", BEFORE)))[0])
    assert got["both"].anchor == "pA"


# weak names -------------------------------------------------------------------------------------------------------


def test_the_weak_name_is_an_unreviewed_staff_name_else_the_boards_and_only_on_the_roster(w):
    w.add("staff", DAY, 0, human_state="known", human_pid="pB", board_state="known", board_pid="pA")
    w.add("board", DAY, 0, board_state="known", board_pid="pA")
    w.add("offroster_staff", DAY, 0, human_state="known", human_pid="pZ", board_state="known", board_pid="pA")
    w.add("offroster_board", DAY, 0, board_state="known", board_pid="pZ")
    w.add("stranger_board", DAY, 0, board_state="stranger")
    got = by_sid(build_items(w.build(), FAC, DAY, label_cutoff_ms=NOW, ref_policy=RefPolicy())[0])
    assert got["staff"].weak == "pB"  # the staff name before the board's
    assert got["board"].weak == "pA"
    assert got["offroster_staff"].weak == "pA"  # a staff name off the roster is no name: the board's is next
    assert got["offroster_board"].weak is None and got["stranger_board"].weak is None


def test_every_probe_is_an_item_with_its_face_norm_and_body(w):
    w.add("noface", DAY, 0, face=False)
    w.add("withface", DAY, 0)
    got = by_sid(build_items(w.build(), FAC, DAY, label_cutoff_ms=NOW, ref_policy=RefPolicy())[0])
    assert got["noface"].probe and got["noface"].face is None and got["noface"].face_quality is None
    assert got["withface"].probe and got["withface"].face_quality == 20.0 and got["withface"].body is None


# earlier confirmed pictures ----------------------------------------------------------------------------------------


def test_earlier_confirmed_pictures_inside_the_window_are_anchors_with_their_own_nights_faces(w):
    w.add("old", "2026-07-01", 0, human_state="known", human_pid="pA", reviewed=True, corrected_ts=BEFORE)  # window
    w.add("other_fac", EARLY, 0, facility="2", human_state="known", human_pid="pA", reviewed=True,
          corrected_ts=BEFORE)
    w.add("offroster", EARLY, 0, human_state="known", human_pid="pZ", reviewed=True, corrected_ts=BEFORE)
    w.add("late_label", EARLY, 0, human_state="known", human_pid="pA", reviewed=True, corrected_ts=AFTER)
    w.add("noface", EARLY, 0, face=False, human_state="known", human_pid="pA", reviewed=True, corrected_ts=BEFORE)
    w.add("answered", EARLY, 1)  # no staff label: only a check-page answer
    ghost = ("ghosted", 0, near(basis(0), 0.05, w.rng), 0.9, 80.0)  # answered, but not in the export
    w.faces.setdefault(EARLY, []).append(ghost)
    w.add("future", "2026-10-22", 0, human_state="known", human_pid="pA", reviewed=True, corrected_ts=BEFORE)
    ws = w.build()
    items, _ = build_items(ws, FAC, DAY, label_cutoff_ms=NOW, ref_policy=RefPolicy(),
                           trusted=Trusted({"answered": "pB", "nobody": None, "ghosted": "pA"},
                                           info={"answered": (FAC, EARLY, BEFORE), "nobody": (FAC, EARLY, BEFORE),
                                                 "ghosted": (FAC, EARLY, BEFORE), "ghost": (FAC, EARLY, BEFORE)}))
    got = by_sid(items)
    earlier = {s: i for s, i in got.items() if not i.probe}
    assert sorted(earlier) == sorted([f"c{p}{i}" for p in ("pA", "pB") for i in range(8)] + ["answered", "ghosted"])
    assert all(i.anchor is not None and i.body is None and i.face is not None for i in earlier.values())
    assert earlier["answered"].anchor == "pB" and earlier["cpA0"].anchor == "pA" and earlier["ghosted"].anchor == "pA"
    assert earlier["cpA0"].face_quality == 20.0
    assert not any(i.probe for s, i in got.items() if s.startswith("c")) and "old" not in earlier


# rows -------------------------------------------------------------------------------------------------------------


def test_a_named_probe_gets_a_row_with_the_right_relation_person_and_evidence(w):
    w.add("new", DAY, 0)  # board has no name: promote
    w.add("same", DAY, 0, board_state="known", board_pid="pA")  # agree
    w.add("other", DAY, 0, board_state="known", board_pid="pB")  # the board says B, the face is A's: relabel
    w.add("noone", DAY, None)  # a face like nobody: no row
    w.add("noface", DAY, 0, face=False)  # nothing to go on: no row
    out = w.night(w.build())
    rows = {l.sighting_id: l for l in out["labels"]}
    assert set(rows) == {"new", "same", "other"}
    assert [(rows[s].state, rows[s].relation, rows[s].person_key, rows[s].person_id, rows[s].via)
            for s in ("new", "same", "other")] == [("known", "promote", "pA", "pA", "face"),
                                                    ("known", "agree", "pA", "pA", "face"),
                                                    ("known", "relabel", "pA", "pA", "face")]
    r = rows["other"]
    assert r.confidence == r.face_sim and r.body_sim is None and r.margin is None and r.confidence > 0.5
    assert r.evidence["reason"] == "named" and r.evidence["others"] == [] and r.evidence["floor"] is not None
    json.dumps(out["summary"], allow_nan=False)
    assert out["summary"]["rows"] == 3 and out["summary"]["reasons"]["named"] >= 3
    assert {s.sighting_id for s in out["sightings"]} == {"new", "same", "other", "noone", "noface"}


def test_the_row_carries_the_facilitys_person_id_for_the_key(tmp_path):
    w = World(tmp_path, aliases={"pA": "H1", "pB": "H2"})  # the keys are alias names, the ids are the facility's own
    w.add("q", DAY, 0)
    (row,) = w.night(w.build())["labels"]
    assert (row.person_key, row.person_id) == ("H1", "pA")


def test_an_anchored_probe_gets_no_row_but_a_probe_with_an_overlay_does(w):
    w.add("anchored", DAY, 0, human_state="known", human_pid="pA", reviewed=True, corrected_ts=BEFORE)
    w.add("stranger", DAY, 0, human_state="stranger")  # an overlay: the run keeps its row out of the push, not here
    w.add("reset", DAY, 0, human_state="reset")
    out = w.night(w.build())
    rows = {l.sighting_id for l in out["labels"]}
    assert "anchored" not in rows and {"stranger", "reset"} <= rows
    assert all(s.has_overlay for s in out["sightings"] if s.sighting_id in ("stranger", "reset"))
    assert out["result"].names["anchored"].via == "anchor"


# earlier nights ---------------------------------------------------------------------------------------------------


def earlier_world(w, *, namer=None):
    """Night 10-19: the nightly job named 4 pictures of pA from the face. Tonight pA has the 8 confirmed ones only."""
    prior = "2026-10-19"
    for i in range(4):
        w.add(f"e{i}", prior, 0)
    (w.runs / f"run_{FAC}_{prior}_aaaa1111.json").write_text(json.dumps(
        {"run_id": "aaaa1111", "dry_run": False, "started_at": 1, **({"namer": namer} if namer else {})}))
    with (w.runs / f"labels_{FAC}_{prior}_aaaa1111.csv").open("w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["sighting_id", "state", "via", "person_id"])
        cw.writerows([[f"e{i}", "known", "face", "pA"] for i in range(4)])
    return prior


def test_the_earlier_nights_top_up_a_thin_person_with_the_jobs_face_names(w):
    prior = earlier_world(w, namer="regroup")
    out = w.night(w.build(), overlays={prior: []}, earlier=EarlierPolicy(nights=1, top_up=12))
    rep = out["earlier"]
    assert [(n.day, n.status, n.added) for n in rep.nights] == [(prior, "used", 4)] and rep.added == {"pA": 4}
    assert out["summary"]["earlier"]["added"] == {"pA": 4}


def test_a_night_whose_overlays_could_not_be_pulled_is_skipped_and_a_missing_entry_counts_as_failed(w):
    prior = earlier_world(w)
    for overlays in ({prior: None}, {}):
        out = w.night(w.build(), overlays=overlays, earlier=EarlierPolicy(nights=1, top_up=12))
        assert [(n.status, n.added) for n in out["earlier"].nights] == [("no_overlays", 0)]


def test_a_picture_with_an_overlay_on_its_night_is_not_topped_up(w):
    prior = earlier_world(w)
    overlays = {prior: [{"sighting_id": "e0", "capture_ts": 5, "human_labeled": True, "human_state": "stranger"},
                        {"sighting_id": "e1", "capture_ts": 9, "human_labeled": False}]}
    out = w.night(w.build(), overlays=overlays, earlier=EarlierPolicy(nights=1, top_up=12))
    assert out["earlier"].nights[0].added == 3  # e0 was rejected by staff since; e1, e2, e3 may come back


def test_a_picture_answered_cant_tell_is_not_topped_up(w, tmp_path):
    prior = earlier_world(w)
    trusted = Trusted(cant_tell={"e2"})
    out = w.night(w.build(), trusted=trusted, overlays={prior: []}, earlier=EarlierPolicy(nights=1, top_up=12))
    assert out["earlier"].nights[0].added == 3


def test_load_trusted_remembers_the_pictures_answered_cant_tell(tmp_path):
    night = tmp_path / "3_2026-10-20"
    night.mkdir()
    (night / "frozen.json").write_text(json.dumps({
        "facility_id": "3", "day": "2026-10-20", "people": [{"key": "k1", "person_id": "pA"}],
        "items": [{"sighting_id": "s0"}, {"sighting_id": "s1"}, {"sighting_id": "s2"}]}))
    (night / "marks.json").write_text(json.dumps({"submitted": True, "updated_at": 5,
                                                  "answers": {"0": "k1", "1": "cant_tell", "2": "other"}}))
    t = load_trusted(tmp_path)
    assert t == {"s0": "pA", "s2": None} and t.cant_tell == {"s1"}


def test_load_trusted_remembers_the_correction_lists_cant_tell_answers_too(tmp_path):
    from vbb_reid_server.corrections import answers_path
    path = answers_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"answers": {
        "s1": {"verdict": "cant_tell", "person_id": None, "facility_id": "3", "day": DAY, "at": 5},
        "s2": {"verdict": "right", "person_id": "pA", "facility_id": "3", "day": DAY, "at": 5}}}))
    t = load_trusted(tmp_path)
    assert t == {"s2": "pA"} and t.cant_tell == {"s1"}


def test_the_earlier_names_come_from_the_nights_before_the_probe_night_only(w):
    prior = earlier_world(w)
    out = w.night(w.build(), overlays={prior: []}, earlier=EarlierPolicy(nights=3, top_up=12))
    assert [n.day for n in out["earlier"].nights] == ["2026-10-19", "2026-10-18", "2026-10-17"]
