"""Earlier nights for the nightly regroup (docs/DESIGN_nightly_regroup.md, "Earlier nights (item 122)", E1-E7).

Everything is built under ``tmp_path``: run records, labels CSVs and fake face tables. A night's labels name pictures
with sids like ``14-a1``; each gets a face vector of its own, so the items that come back can be told apart.
"""

from __future__ import annotations

import csv
import json
import random
import zlib

import numpy as np
import pytest
from vbb_reid_server.embed import FaceTable, StaleCache
from vbb_reid_server.pipeline import RefPolicy
from vbb_reid_server.regroup import Item
from vbb_reid_server.regroup_inputs import EarlierPolicy, earlier_items, night_run, usable_faces
from vbb_reid_server.vetting import VetPolicy

DAY, FAC, DIM = "2026-11-15", "3", 8


def vec(sid: str) -> np.ndarray:
    v = np.random.default_rng(zlib.crc32(sid.encode())).normal(size=DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def table(rows) -> FaceTable:
    """rows: dicts with sid (and optionally vec, norm, n_faces, qf, second_ratio, second_det); the defaults are a clear,
    usable face."""
    n = len(rows)
    col = lambda name, default: np.array([r.get(name, default) for r in rows], dtype=float)  # noqa: E731
    z = np.zeros(n)
    return FaceTable(
        sighting_id=np.array([r["sid"] for r in rows], dtype=object), crop_idx=np.zeros(n, dtype=int),
        source=np.array(["t"] * n, dtype=object),
        vec=np.array([r.get("vec", vec(r["sid"])) for r in rows], dtype=np.float32).reshape(n, DIM),
        face_h=z, det=z, blur=z, residual=z, asym=z, yaw=z, norm=col("norm", 20.0), n_faces=col("n_faces", 1),
        second_ratio=col("second_ratio", 0.0), second_det=col("second_det", 0.0), qf=col("qf", 0.9))


class World:
    """A runs directory plus the face cache of a few nights."""

    def __init__(self, tmp_path):
        self.runs, self.faces, self.table_calls = tmp_path, {}, []

    def night(self, day, rows, *, tag="aaaa1111", dry=False, started=1, namer=None, labels=True, faces=True,
              record=None):
        """rows: (sid, person_id) or (sid, person_id, state, via). Writes the run record and the labels file."""
        rec = {"run_id": tag, "started_at": started} if record is None else record
        if dry is not None and record is None:
            rec["dry_run"] = dry
        if namer is not None and record is None:
            rec["namer"] = namer
        (self.runs / f"run_{FAC}_{day}_{tag}.json").write_text(json.dumps(rec))
        if labels:
            with (self.runs / f"labels_{FAC}_{day}_{tag}.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["run_id", "sighting_id", "state", "via", "person_id", "person_key"])
                for r in rows:
                    sid, pid, state, via = (tuple(r) + ("known", "face"))[:4] if len(r) == 2 else r
                    w.writerow([tag, sid, state, via, pid, "IGNORED"])
        if faces:
            self.faces.setdefault(day, {}).update({r[0]: None for r in rows})

    def face_table(self, facility, day):
        self.table_calls.append(day)
        if day not in self.faces:
            raise FileNotFoundError(day)
        return table([{"sid": s} for s in self.faces[day]])

    def run(self, **over):
        args = dict(facility=FAC, day=DAY, runs_dir=self.runs, face_table=self.face_table, key=lambda p: p,
                    roster={"A", "B"}, tonight=[], confirmed={}, touched=set(), policy=EarlierPolicy())
        args.update(over)
        return earlier_items(**args)


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


def sids(items):
    return [i.sid for i in items]


def probe(sid, weak="A", *, face=True, anchor=None, is_probe=True):
    return Item(sid, is_probe, weak, anchor, vec(sid) if face else None, None)


def shape(items):
    return [(i.sid, i.probe, i.weak, i.anchor, None if i.face is None else i.face.tobytes(), i.body, i.face_quality)
            for i in items]


# usable_faces -----------------------------------------------------------------------------------------------------


def test_usable_faces_takes_the_first_clear_usable_row_of_the_table_and_returns_its_norm():
    rows = [
        {"sid": "s1", "n_faces": 2},  # two faces in the crop: not clear
        {"sid": "s1", "norm": 5.0},  # raw norm below VetPolicy.min_norm (14)
        {"sid": "s1", "qf": 0.1},  # quality below RefPolicy.ref_q_min (0.40)
        {"sid": "s1", "norm": 15.0, "vec": vec("first")},  # the first row that passes
        {"sid": "s1", "norm": 30.0, "vec": vec("second")},  # a better one later in the table does not win
        {"sid": "s2", "n_faces": 2},  # nothing usable: absent
        {"sid": "s3", "second_ratio": 0.8, "second_det": 0.9},  # skipped only when the policy says so
    ]
    got = usable_faces(table(rows), RefPolicy(ref_two_face_ratio=0.5), VetPolicy())
    assert set(got) == {"s1"}
    v, norm = got["s1"]
    assert norm == 15.0 and v.dtype == np.float64 and np.allclose(v, vec("first"))
    assert set(usable_faces(table(rows), RefPolicy(), VetPolicy())) == {"s1", "s3"}


# night_run --------------------------------------------------------------------------------------------------------


def test_night_run_returns_the_record_and_its_labels_path_even_when_that_file_is_missing(w):
    w.night("2026-11-14", [], labels=False)
    rec, path = night_run(w.runs, FAC, "2026-11-14")
    assert rec["run_id"] == "aaaa1111" and path == w.runs / f"labels_{FAC}_2026-11-14_aaaa1111.csv"
    assert not path.exists()
    assert night_run(w.runs, FAC, "2026-11-13") is None


def test_night_run_wants_a_real_run_and_the_latest_started_one(w):
    w.night("2026-11-14", [], tag="dry00000", dry=True, started=999)
    w.night("2026-11-14", [], tag="nokey000", dry=None, started=998)  # dry_run missing: not a real run
    w.night("2026-11-14", [], tag="text0000", record={"dry_run": "false", "started_at": 997})  # not exactly false
    assert night_run(w.runs, FAC, "2026-11-14") is None
    w.night("2026-11-14", [], tag="early000", started=5)
    w.night("2026-11-14", [], tag="late0000", started=9)
    w.night("2026-11-14", [], tag="mid00000", started=7)
    assert night_run(w.runs, FAC, "2026-11-14")[0]["run_id"] == "late0000"


def test_night_run_skips_a_record_that_cannot_be_read(w):
    (w.runs / f"run_{FAC}_2026-11-14_broken00.json").write_text("{not json")
    (w.runs / f"run_{FAC}_2026-11-14_list0000.json").write_text("[1, 2]")
    assert night_run(w.runs, FAC, "2026-11-14") is None
    w.night("2026-11-14", [], tag="good0000")
    assert night_run(w.runs, FAC, "2026-11-14")[0]["run_id"] == "good0000"


def test_night_run_looks_at_one_facility_and_one_day(w):
    w.night("2026-11-14", [], tag="aaaa1111")
    assert night_run(w.runs, "4", "2026-11-14") is None and night_run(w.runs, FAC, "2026-11-13") is None


@pytest.mark.parametrize("bad", [dict(nights=-1), dict(top_up=0), dict(min_checked=0), dict(max_disagree=-0.1),
                                 dict(max_disagree=1.1)])
def test_a_bad_policy_raises(bad):
    with pytest.raises(ValueError):
        EarlierPolicy(**bad)


def test_the_boundary_values_of_a_policy_are_allowed():
    EarlierPolicy(nights=0, top_up=1, min_checked=1, max_disagree=0)
    EarlierPolicy(max_disagree=1)


# E1: nights -------------------------------------------------------------------------------------------------------


def test_the_nights_before_the_probe_night_are_read_newest_first_and_never_the_probe_night_or_later(w):
    for d in ("2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16"):
        w.night(d, [(f"{d[-2:]}-a1", "A")])
    items, rep = w.run(policy=EarlierPolicy(nights=3, top_up=10))
    assert [n.day for n in rep.nights] == ["2026-11-14", "2026-11-13", "2026-11-12"]
    assert sids(items) == ["14-a1", "13-a1", "12-a1"]
    assert sorted(set(w.table_calls)) == ["2026-11-12", "2026-11-13", "2026-11-14"]  # 15 and 16 never opened


def test_nights_counts_calendar_days_across_a_month_end(w):
    w.night("2026-02-28", [("n28", "A")])
    w.night("2026-02-27", [("n27", "A")])
    items, rep = w.run(day="2026-03-02", policy=EarlierPolicy(nights=3, top_up=10))
    assert [(n.day, n.status) for n in rep.nights] == [("2026-03-01", "no_run"), ("2026-02-28", "used"),
                                                       ("2026-02-27", "used")]
    assert sids(items) == ["n28", "n27"]


def test_zero_nights_reads_nothing(w):
    w.night("2026-11-14", [("14-a1", "A")])
    items, rep = w.run(policy=EarlierPolicy(nights=0))
    assert items == [] and rep.nights == () and w.table_calls == []
    assert rep.added == {} and rep.checked == {} and rep.dropped == ()


def test_a_night_says_why_it_contributes_nothing(w):
    w.night("2026-11-14", [("14-a1", "A")], dry=True)  # only a dry run
    w.night("2026-11-13", [("13-a1", "A")], labels=False)  # no labels file
    w.night("2026-11-12", [("12-a1", "A")], namer="someone-else")  # not a namer we read
    w.night("2026-11-11", [("11-a1", "A")], faces=False)  # no face cache: the table call raises
    w.night("2026-11-10", [("10-a1", "A")])
    items, rep = w.run(policy=EarlierPolicy(nights=6, top_up=10))
    status = {n.day[-2:]: n.status for n in rep.nights}
    assert status == {"14": "no_run", "13": "no_labels", "12": "namer", "11": "no_faces", "10": "used", "09": "no_run"}
    assert sids(items) == ["10-a1"]
    assert [(n.face_names, n.added) for n in rep.nights if n.status != "used"] == [(0, 0)] * 5


@pytest.mark.parametrize("error", [FileNotFoundError, OSError, KeyError, ValueError])
def test_a_face_table_that_cannot_be_read_makes_the_night_no_faces(w, error):
    w.night("2026-11-14", [("14-a1", "A")])

    def broken(facility, day):
        raise error("x")
    items, rep = w.run(face_table=broken, policy=EarlierPolicy(nights=1, top_up=10))
    assert items == [] and rep.nights[0].status == "no_faces"


def test_an_empty_face_table_makes_the_night_no_faces(w):
    w.night("2026-11-14", [("14-a1", "A")])
    items, rep = w.run(face_table=lambda f, d: table([]), policy=EarlierPolicy(nights=1, top_up=10))
    assert items == [] and rep.nights[0].status == "no_faces"


# E2: face names only ----------------------------------------------------------------------------------------------


def test_only_known_face_names_with_a_person_are_read(w):
    w.night("2026-11-14", [("ok", "A"), ("body", "A", "known", "body"), ("stranger", "A", "stranger", "face"),
                           ("unsure", "A", "uncertain", "face"), ("nobody", "", "known", "face")])
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["ok"] and rep.nights[0].face_names == 1


# E3: weak, never final --------------------------------------------------------------------------------------------


def test_an_earlier_picture_becomes_a_weak_probe_free_item_with_its_face_and_norm_as_quality(w):
    w.night("2026-11-14", [("s1", "A")])
    (it,), _ = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert (it.sid, it.probe, it.weak, it.anchor, it.body, it.face_quality) == ("s1", False, "A", None, None, 20.0)
    assert np.allclose(it.face, vec("s1"))


def test_the_person_is_the_id_folded_with_tonights_merge_chain_and_person_key_is_ignored(w):
    w.night("2026-11-14", [("s1", "old-id")])  # the person_key column says "IGNORED"
    items, _ = w.run(key=lambda p: {"old-id": "A"}.get(p, p), policy=EarlierPolicy(nights=1, top_up=10))
    assert [(i.sid, i.weak) for i in items] == [("s1", "A")]


def test_a_person_not_on_tonights_roster_is_skipped(w):
    w.night("2026-11-14", [("s1", "A"), ("s2", "Z")])
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["s1"] and rep.nights[0].face_names == 1


def test_a_picture_already_in_tonights_items_is_skipped(w):
    w.night("2026-11-14", [("s1", "A"), ("s2", "A")])
    items, rep = w.run(tonight=[probe("s1", "B")], policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["s2"]


def test_a_human_checked_picture_is_skipped_including_a_not_on_the_list_answer(w):
    w.night("2026-11-14", [("s1", "A"), ("s2", "A"), ("s3", "A"), ("s4", "A")])
    items, _ = w.run(confirmed={"s1": "A", "s2": "B", "s3": None}, policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["s4"]


def test_a_picture_with_any_human_overlay_now_is_skipped(w):
    w.night("2026-11-14", [("s1", "A"), ("s2", "A")])
    items, _ = w.run(touched={"s1"}, policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["s2"]


def test_a_picture_without_a_usable_face_is_skipped(w):
    w.night("2026-11-14", [("good", "A"), ("twofaces", "A"), ("blurred", "A"), ("absent", "A")])
    rows = [{"sid": "good"}, {"sid": "twofaces", "n_faces": 2}, {"sid": "blurred", "qf": 0.1}]  # "absent": no row
    items, rep = w.run(face_table=lambda f, d: table(rows), policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["good"] and rep.nights[0].face_names == 1


def test_face_names_counts_what_is_left_after_the_skips_but_before_the_top_up_limit(w):
    w.night("2026-11-14", [(f"s{k}", "A") for k in range(6)] + [("out", "Z"), ("seen", "A")])
    items, rep = w.run(tonight=[probe("seen")], policy=EarlierPolicy(nights=1, top_up=3))  # A has 1 tonight: needs 2
    assert len(items) == 2 and rep.nights[0].face_names == 6 and rep.nights[0].added == 2


# E4: top-up -------------------------------------------------------------------------------------------------------


def test_a_thin_person_is_filled_exactly_to_top_up_and_a_full_one_gets_none(w):
    w.night("2026-11-14", [(f"a{k}", "A") for k in range(10)] + [(f"b{k}", "B") for k in range(10)])
    tonight = [probe(f"ta{k}", "A") for k in range(2)] + [probe(f"tb{k}", "B") for k in range(5)]
    items, rep = w.run(tonight=tonight, policy=EarlierPolicy(nights=1, top_up=5))
    assert sorted(sids(items)) == ["a0", "a1", "a2"]  # A: 2 tonight, 3 more; B: 5 = top_up, nothing
    assert rep.added == {"A": 3} and rep.nights[0].added == 3


def test_a_person_with_nothing_tonight_gets_top_up_pictures(w):
    w.night("2026-11-14", [(f"a{k}", "A") for k in range(10)])
    items, _ = w.run(policy=EarlierPolicy(nights=1, top_up=4))
    assert len(items) == 4


def test_only_anchored_pictures_and_probes_with_their_name_count_as_tonights_pictures(w):
    w.night("2026-11-14", [(f"a{k}", "A") for k in range(10)])
    tonight = [
        probe("anchored", "B", anchor="A", is_probe=False),  # confirmed, any day: counts
        probe("probe", "A"),  # a probe with the weak name: counts
        probe("context", "A", is_probe=False),  # an earlier weak picture: does not count
        probe("noface", "A", face=False),  # no face: does not count
        probe("anchor-noface", "B", face=False, anchor="A", is_probe=False),
    ]
    items, _ = w.run(tonight=tonight, policy=EarlierPolicy(nights=1, top_up=4))
    assert len(items) == 2  # 4 - 2


def test_nights_newest_first_then_capture_time_newest_first_then_sid(w):
    w.night("2026-11-14", [("n14-a", "A"), ("n14-b", "A"), ("n14-c", "A")])
    w.night("2026-11-13", [("n13-a", "A"), ("n13-b", "A")])
    ts = {"n14-a": 100, "n14-b": 300, "n14-c": 200, "n13-a": 900, "n13-b": 800}  # night 13 is later in the clock
    items, _ = w.run(capture_ts=ts, policy=EarlierPolicy(nights=2, top_up=5))
    assert sids(items) == ["n14-b", "n14-c", "n14-a", "n13-a", "n13-b"]  # the night comes first, then newest first


def test_without_capture_times_the_order_is_by_sid_and_equal_times_tie_by_sid(w):
    w.night("2026-11-14", [("s3", "A"), ("s1", "A"), ("s2", "A"), ("s0", "A")])
    items, _ = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["s0", "s1", "s2", "s3"]
    items, _ = w.run(capture_ts={"s1": 5, "s2": 5, "s3": 5}, policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["s1", "s2", "s3", "s0"]  # known times first (newest first), the missing one last


def test_the_limit_cuts_the_oldest_pictures(w):
    w.night("2026-11-14", [("n14-a", "A"), ("n14-b", "A")])
    w.night("2026-11-13", [("n13-a", "A"), ("n13-b", "A")])
    items, rep = w.run(policy=EarlierPolicy(nights=2, top_up=3))
    assert sids(items) == ["n14-a", "n14-b", "n13-a"]
    assert [(n.day[-2:], n.face_names, n.added) for n in rep.nights] == [("14", 2, 2), ("13", 2, 1)]


def test_a_picture_in_two_nights_is_added_once(w):
    w.night("2026-11-14", [("dup", "A"), ("x1", "A")])
    w.night("2026-11-13", [("dup", "A"), ("x2", "A")])
    items, rep = w.run(policy=EarlierPolicy(nights=2, top_up=10))
    assert sids(items) == ["dup", "x1", "x2"] and rep.added == {"A": 3}


# E5: drift check --------------------------------------------------------------------------------------------------


def drift_world(w, agree, other, none, extra=6, night="2026-11-14"):
    """A's names: ``agree`` checked pictures confirmed A, ``other`` confirmed B, ``none`` confirmed "not on the list",
    plus ``extra`` unchecked ones."""
    rows, confirmed = [], {}
    for k in range(agree + other + none):
        sid = f"c{k:02d}"
        rows.append((sid, "A"))
        confirmed[sid] = "A" if k < agree else "B" if k < agree + other else None
    rows += [(f"u{k}", "A") for k in range(extra)]
    w.night(night, rows)
    return confirmed


POL = EarlierPolicy(nights=2, top_up=100, min_checked=5, max_disagree=0.2)


def test_a_share_of_disagreement_exactly_at_max_disagree_is_not_dropped(w):
    items, rep = w.run(confirmed=drift_world(w, agree=8, other=2, none=0), policy=POL)
    assert rep.checked == {"A": (10, 2)} and rep.dropped == () and len(items) == 6


def test_more_disagreement_than_max_disagree_drops_the_person(w):
    items, rep = w.run(confirmed=drift_world(w, agree=7, other=3, none=0), policy=POL)
    assert rep.checked == {"A": (10, 3)} and rep.dropped == ("A",)
    assert items == [] and rep.added == {}


def test_a_not_on_the_list_answer_counts_as_a_disagreement(w):
    items, rep = w.run(confirmed=drift_world(w, agree=7, other=0, none=3), policy=POL)
    assert rep.checked == {"A": (10, 3)} and rep.dropped == ("A",) and items == []


def test_fewer_than_min_checked_pictures_are_never_enough_to_drop(w):
    items, rep = w.run(confirmed=drift_world(w, agree=0, other=4, none=0), policy=POL)  # 4 of 4 disagree, but only 4
    assert rep.checked == {"A": (4, 4)} and rep.dropped == () and len(items) == 6


def test_a_picture_in_two_nights_is_checked_once(w):
    rows = [(f"c{k}", "A") for k in range(3)] + [("u1", "A")]
    w.night("2026-11-14", rows)
    w.night("2026-11-13", rows)
    items, rep = w.run(confirmed={f"c{k}": "B" for k in range(3)}, policy=POL)  # 3 pictures, twice in the files
    assert rep.checked == {"A": (3, 3)} and rep.dropped == () and sids(items) == ["u1"]


def test_only_the_person_that_drifts_is_dropped(w):
    rows = [(f"a{k}", "A") for k in range(10)] + [(f"b{k}", "B") for k in range(10)]
    w.night("2026-11-14", rows)
    confirmed = {f"a{k}": "B" for k in range(10)} | {f"b{k}": "B" for k in range(5)}
    items, rep = w.run(confirmed=confirmed, policy=POL)
    assert rep.dropped == ("A",) and rep.checked == {"A": (10, 10), "B": (5, 0)}
    assert sorted(sids(items)) == [f"b{k}" for k in range(5, 10)] and rep.added == {"B": 5}


# E6: provenance ---------------------------------------------------------------------------------------------------


def test_a_run_record_without_a_namer_counts_as_decide(w):
    w.night("2026-11-14", [("s1", "A")])  # no "namer" key
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert rep.nights[0].status == "used" and sids(items) == ["s1"]


def test_the_sorters_own_runs_are_read_by_default_and_an_unknown_namer_is_refused(w):
    assert EarlierPolicy().namers == ("decide", "regroup") and EarlierPolicy().test_runs is False
    w.night("2026-11-14", [("s1", "A")], namer="regroup")
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["s1"] and rep.nights[0].status == "used"
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10, namers=("decide",)))
    assert items == [] and rep.nights[0].status == "namer"
    w.night("2026-11-13", [("s2", "A")], namer="someone-else")
    items, rep = w.run(policy=EarlierPolicy(nights=2, top_up=10))
    assert [(n.day[-2:], n.status) for n in rep.nights] == [("14", "used"), ("13", "namer")]


# E7: deterministic ------------------------------------------------------------------------------------------------


def test_the_same_inputs_in_any_order_give_the_same_items_and_report(w):
    for d, tag in (("2026-11-14", "aaaa1111"), ("2026-11-13", "bbbb2222")):
        w.night(d, [(f"{d[-2:]}-{k}", "AB"[k % 2]) for k in range(8)], tag=tag)
    tonight = [probe(f"t{k}", "A") for k in range(3)]
    confirmed = {"14-0": "A", "13-1": "B", "14-5": None}
    touched = {"14-2", "13-3"}
    ts = {f"{d}-{k}": random.Random(k).randint(0, 3) for d in ("14", "13") for k in range(8)}
    pol = EarlierPolicy(nights=2, top_up=6)
    base_items, base_rep = w.run(tonight=tonight, confirmed=confirmed, touched=touched, capture_ts=ts, policy=pol)
    for seed in (1, 2, 3):
        rnd = random.Random(seed)
        shuffled = list(tonight)
        rnd.shuffle(shuffled)
        items, rep = w.run(tonight=shuffled, confirmed=dict(rnd.sample(sorted(confirmed.items()), len(confirmed))),
                           touched=rnd.sample(sorted(touched), len(touched)), capture_ts=ts, policy=pol,
                           roster=rnd.sample(["A", "B"], 2))
        assert shape(items) == shape(base_items) and rep == base_rep


def test_a_labels_row_that_cannot_be_read_is_skipped_not_fatal(w):
    w.night("2026-11-14", [("s1", "A"), ("s2", "bad"), ("s3", "A")])

    def key(pid):
        if pid == "bad":
            raise ValueError("cannot fold")
        return pid
    items, rep = w.run(key=key, policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["s1", "s3"] and rep.nights[0].status == "used"


def test_a_row_with_missing_fields_is_skipped_not_fatal(w):
    w.night("2026-11-14", [("s1", "A")])
    path = w.runs / f"labels_{FAC}_2026-11-14_aaaa1111.csv"
    path.write_text(path.read_text() + "r,s2,known\nr,,known,face,A,x\n")  # a short row, a row with no sid
    items, _ = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["s1"]


def test_a_labels_file_that_cannot_be_decoded_makes_the_night_no_labels(w):
    w.night("2026-11-14", [("s1", "A")])
    (w.runs / f"labels_{FAC}_2026-11-14_aaaa1111.csv").write_bytes(b"\xff\xfe\x00bad")
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert items == [] and rep.nights[0].status == "no_labels"


def test_a_face_table_in_an_older_cache_format_makes_the_night_no_faces(w):
    w.night("2026-11-14", [("14-a1", "A")])

    def stale(facility, day):
        raise StaleCache("cache from an older model")
    items, rep = w.run(face_table=stale, policy=EarlierPolicy(nights=1, top_up=10))
    assert items == [] and rep.nights[0].status == "no_faces"


def test_night_run_with_namers_takes_the_latest_run_of_an_accepted_namer(w):
    w.night("2026-11-14", [], tag="dec00000", started=5)
    w.night("2026-11-14", [], tag="reg00000", started=9, namer="regroup")
    assert night_run(w.runs, FAC, "2026-11-14")[0]["run_id"] == "reg00000"  # no filter: the latest real run
    assert night_run(w.runs, FAC, "2026-11-14", ("decide",))[0]["run_id"] == "dec00000"
    assert night_run(w.runs, FAC, "2026-11-14", ("decide", "regroup"))[0]["run_id"] == "reg00000"
    assert night_run(w.runs, FAC, "2026-11-14", ("someone-else",)) is None


def test_a_later_run_of_another_namer_does_not_hide_an_accepted_run(w):
    w.night("2026-11-14", [("dec-1", "A")], tag="dec00000", started=5)
    w.night("2026-11-14", [("reg-1", "A")], tag="reg00000", started=9, namer="regroup")
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10, namers=("decide",)))
    assert sids(items) == ["dec-1"] and rep.nights[0].status == "used"  # the later sorter run is not accepted here
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["reg-1"]  # by default both are accepted and the latest wins
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10, namers=("regroup",)))
    assert sids(items) == ["reg-1"]


def test_the_status_is_namer_only_when_the_night_has_real_runs_but_none_with_an_accepted_namer(w):
    w.night("2026-11-14", [("s1", "A")], tag="reg00000", namer="other")
    w.night("2026-11-13", [("s2", "A")], tag="dry00000", namer="other", dry=True)  # a dry run is no real run
    w.night("2026-11-12", [("s3", "A")], tag="dec00000", dry=True)
    w.night("2026-11-12", [("s4", "A")], tag="reg00001", namer="other")
    items, rep = w.run(policy=EarlierPolicy(nights=3, top_up=10))
    assert [(n.day[-2:], n.status) for n in rep.nights] == [("14", "namer"), ("13", "no_run"), ("12", "namer")]
    assert items == []


def test_night_run_counts_a_dry_run_only_when_test_runs_is_asked_for(w):
    w.night("2026-11-14", [], tag="real0000", started=5)
    w.night("2026-11-14", [], tag="test0000", started=9, dry=True)
    assert night_run(w.runs, FAC, "2026-11-14")[0]["run_id"] == "real0000"
    assert night_run(w.runs, FAC, "2026-11-14", None, True)[0]["run_id"] == "test0000"  # the latest of both kinds
    only_dry = w.runs / f"run_{FAC}_2026-11-13_dry00000.json"
    only_dry.write_text(json.dumps({"run_id": "dry00000", "dry_run": True, "started_at": 1}))
    assert night_run(w.runs, FAC, "2026-11-13") is None
    assert night_run(w.runs, FAC, "2026-11-13", None, True)[0]["run_id"] == "dry00000"
    only_dry.write_text(json.dumps({"run_id": "dry00000", "started_at": 1}))  # dry_run missing: never a run
    assert night_run(w.runs, FAC, "2026-11-13", None, True) is None


def test_a_dry_run_night_is_read_only_with_test_runs(w):
    w.night("2026-11-14", [("t1", "A")], dry=True)
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert items == [] and rep.nights[0].status == "no_run"
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10, test_runs=True))
    assert sids(items) == ["t1"] and rep.nights[0].status == "used"


def test_a_night_whose_overlays_could_not_be_pulled_is_not_read_at_all(w):
    w.night("2026-11-14", [("a1", "A")])
    w.night("2026-11-13", [("b1", "A")])
    items, rep = w.run(skip_days={"2026-11-14"}, policy=EarlierPolicy(nights=2, top_up=10))
    assert [(n.day[-2:], n.status, n.face_names, n.added) for n in rep.nights] == [("14", "no_overlays", 0, 0),
                                                                                  ("13", "used", 1, 1)]
    assert sids(items) == ["b1"] and w.table_calls == ["2026-11-13"]  # no face table, no labels read for 14
    assert rep.checked == {} and rep.dropped == ()


def test_skip_days_defaults_to_nothing_skipped(w):
    w.night("2026-11-14", [("a1", "A")])
    items, rep = w.run(policy=EarlierPolicy(nights=1, top_up=10))
    assert sids(items) == ["a1"] and rep.nights[0].status == "used"
