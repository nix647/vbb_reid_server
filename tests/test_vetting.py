"""Learn only from pictures someone looked at (owner 2026-11-11/12, plan decision 34), and the disagreement queue."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import numpy as np
from helpers import FACE_ID, basis, near, sighting, write_export, write_face_cache
from test_v020 import ROSTER, FakeBackend, _probe, _ref, client_for, ms
from vbb_reid_server.api import export_from_backend
from vbb_reid_server.data import Export, Person, load_export
from vbb_reid_server.decide import Thresholds
from vbb_reid_server.embed import _dir
from vbb_reid_server.pipeline import RefPolicy, Workspace
from vbb_reid_server.shadow import decide_night, disagreements, shadow_run
from vbb_reid_server.verify import IMAGE_PREFIX, build_night, page_data, score_night
from vbb_reid_server.vetting import VetPolicy, load_trusted, vet_labels

DAY = "2026-10-20"
A, B, X = basis(0), basis(1), basis(5)  # X: someone not on the roster
RNG = np.random.default_rng(7)
TH, POL = Thresholds(), RefPolicy(include_same_day=True)


def _ws(tmp_path, rows, sightings, people=("pA", "pB"), patch=None):
    """rows: (sid, vec); ``patch``: sid -> {field: value} written over the cache's measures."""
    emb = tmp_path / "emb"
    write_face_cache(emb, "3", DAY, [(sid, 0, v, 0.9, 80.0) for sid, v in rows])
    if patch:
        f = _dir(emb, "face", FACE_ID, "3") / f"{DAY}.npz"
        z = dict(np.load(f, allow_pickle=True))
        for i, sid in enumerate(z["sighting_id"]):
            for k, v in (patch.get(str(sid)) or {}).items():
                z[k][i] = v
        np.savez(f, **z)
    ex = Export(root=None, sightings=sightings, people={p: Person(p, "3") for p in people},
                facility_tz={"3": "Asia/Shanghai"}, facility_org={"3": "brand"})
    return Workspace(ex, emb, FACE_ID, None)


def _lab(sid, pid, ts, state="known", reviewed=False):
    s = sighting(sid, day=DAY, human_state=state, human_pid=pid, corrected_ts=ts)
    return replace(s, image_url=IMAGE_PREFIX + f"x/{sid}.jpg", human_reviewed=reviewed)


def _night(tmp_path):
    """Staff renamed rows to pA and pB, looking only at the reviewed pictures.
    u0-u2 (someone else, X) and q0 (pB's face) were swept into pA unseen; s0 (pA's face) was marked a stranger."""
    labs = ([_lab(f"a{j}", "pA", 100, reviewed=True) for j in range(3)] + [_lab(f"u{j}", "pA", 100) for j in range(3)]
            + [_lab(f"b{j}", "pB", 300, reviewed=True) for j in range(3)]
            + [_lab("q0", "pA", 500), _lab("s0", None, 500, state="stranger")])
    rows = ([(f"a{j}", near(A, 0.05, RNG)) for j in range(3)] + [(f"u{j}", near(X, 0.05, RNG)) for j in range(3)]
            + [(f"b{j}", near(B, 0.05, RNG)) for j in range(3)]
            + [("q0", near(B, 0.05, RNG)), ("s0", near(A, 0.05, RNG))])
    return _ws(tmp_path, rows, labs)


def test_only_pictures_someone_looked_at_are_learned_from(tmp_path):
    ws = _night(tmp_path)
    off = decide_night(ws, "3", DAY, TH, POL, label_cutoff_ms=10_000)
    assert off["vetting"] is None and off["refs"].n_sightings("pA") == 7  # off: everything staff named pA
    on = decide_night(ws, "3", DAY, TH, POL, label_cutoff_ms=10_000, vet=VetPolicy())
    st = on["vetting"].status
    assert {st[f"a{j}"] for j in range(3)} == {"reviewed"}
    assert {st[f"u{j}"] for j in range(3)} | {st["q0"]} == {"not_reviewed"}
    assert (on["refs"].n_sightings("pA"), on["refs"].n_sightings("pB")) == (3, 3)
    assert set(on["seeds"]) == {"a0", "a1", "a2", "b0", "b1", "b2"}


def test_a_reviewed_picture_without_one_clear_face_seeds_the_day_but_never_teaches_a_face(tmp_path):
    labs = [_lab(f"a{j}", "pA", 100, reviewed=True) for j in range(3)] + [
        _lab(sid, "pA", 100, reviewed=True) for sid in ("nf", "two", "dim")]
    rows = [(f"a{j}", near(A, 0.05, RNG)) for j in range(3)] + [("two", near(A, 0.05, RNG)), ("dim", near(A, 0.05, RNG))]
    ws = _ws(tmp_path, rows, labs, patch={"two": {"n_faces": 2, "second_ratio": 0.8}, "dim": {"norm": 9.0}})
    on = decide_night(ws, "3", DAY, TH, POL, label_cutoff_ms=10_000, vet=VetPolicy())
    v = on["vetting"]
    assert (v.status["nf"], v.status["two"], v.status["dim"]) == ("no_face", "two_faces", "poor_face")
    assert on["refs"].n_sightings("pA") == 3  # "two" and "dim" would teach the face without the check
    assert {"nf", "two", "dim"} <= set(on["seeds"])


def test_only_the_clear_crops_of_a_reviewed_picture_teach_the_face(tmp_path):
    """A cache built from board crops holds several crops per sighting: one clear crop must not carry a
    two-face crop of the same sighting into the references."""
    labs = [_lab(f"a{j}", "pA", 100, reviewed=True) for j in range(2)] + [_lab("m0", "pA", 100, reviewed=True)]
    emb = tmp_path / "emb"
    write_face_cache(emb, "3", DAY, [(f"a{j}", 0, near(A, 0.05, RNG), 0.9, 80.0) for j in range(2)]
                     + [("m0", 0, near(A, 0.05, RNG), 0.9, 80.0), ("m0", 1, near(X, 0.05, RNG), 0.9, 80.0)])
    f = _dir(emb, "face", FACE_ID, "3") / f"{DAY}.npz"
    z = dict(np.load(f, allow_pickle=True))
    z["n_faces"][3] = 2  # m0's second crop shows two faces
    np.savez(f, **z)
    ex = Export(root=None, sightings=labs, people={p: Person(p, "3") for p in ("pA", "pB")},
                facility_tz={"3": "Asia/Shanghai"}, facility_org={"3": "brand"})
    on = decide_night(Workspace(ex, emb, FACE_ID, None), "3", DAY, TH, POL, label_cutoff_ms=10_000, vet=VetPolicy())
    assert on["vetting"].status["m0"] == "reviewed"  # it has a clear crop
    vecs = on["refs"]["pA"]
    assert len(vecs) == 3 and float((vecs @ X).max()) < 0.5  # the two-face crop never became a reference


def test_your_answer_on_the_check_page_settles_a_label_either_way(tmp_path):
    labs = [_lab("ans", "pA", 100), _lab("wrong", "pA", 100, reviewed=True), _lab("other", "pA", 100, reviewed=True)]
    labs += [_lab(f"a{j}", "pA", 100, reviewed=True) for j in range(3)]
    ws = _ws(tmp_path, [(s.sighting_id, near(A, 0.05, RNG)) for s in labs], labs)
    v = vet_labels(ws, labs, {"ans": "pA", "wrong": "pB", "other": None}, VetPolicy())
    assert (v.status["ans"], v.status["wrong"], v.status["other"]) == ("answered", "answer_disagrees", "answer_disagrees")
    assert v.trusted == {"ans", "a0", "a1", "a2"}  # answered pA though nobody ticked it; a staff look loses to yours


def test_trusted_pictures_come_from_refs_and_finished_nights_only(tmp_path):
    root = tmp_path / "verify"
    (root / "3_2026-10-20").mkdir(parents=True)
    (root / "3_2026-10-21").mkdir(parents=True)
    people = [{"key": "pA", "person_id": "pA", "photos": [{"sighting_id": "r1"}, {"sighting_id": "both"}]},
              {"key": "pB", "person_id": "pB", "photos": []}]
    (root / "refs.json").write_text(json.dumps({"people": people}))
    for night, submitted, last in (("3_2026-10-20", True, "n3"), ("3_2026-10-21", False, "late")):
        items = [{"sighting_id": s} for s in ("n1", "n2", "both", last)]
        (root / night / "frozen.json").write_text(json.dumps({"people": people, "items": items}))
        (root / night / "marks.json").write_text(json.dumps(
            {"answers": {"0": "pB", "1": "other", "2": "pB", "3": "cant_tell" if submitted else "pA"},
             "submitted": submitted}))
    t = load_trusted(root)
    assert t == {"r1": "pA", "n1": "pB", "n2": None}  # "both" got two answers; can't tell and "late" (unfinished) skipped


def test_labels_nobody_looked_at_are_queued_when_the_machine_disagrees(tmp_path):
    ws = _night(tmp_path)
    assert disagreements(decide_night(ws, "3", DAY, TH, POL, label_cutoff_ms=10_000), ws.export.key) == []
    on = decide_night(ws, "3", DAY, TH, POL, label_cutoff_ms=10_000, vet=VetPolicy())
    d = {x["sighting_id"]: x for x in disagreements(on, ws.export.key)}
    assert (d["q0"]["staff"], d["q0"]["machine"], d["q0"]["check"]) == ("pA", "pB", "not_reviewed")
    assert d["s0"]["staff"] is None and d["s0"]["machine"] == "pA"  # a stranger label the machine names
    assert not {"a0", "b0"} & set(d)  # pictures someone looked at are never queued


def test_a_picture_you_already_answered_is_never_queued_again(tmp_path):
    ws = _night(tmp_path)
    on = decide_night(ws, "3", DAY, TH, POL, label_cutoff_ms=10_000, vet=VetPolicy(), trusted={"q0": "pB"})
    assert on["vetting"].status["q0"] == "answer_disagrees"
    assert "q0" not in {x["sighting_id"] for x in disagreements(on, ws.export.key)}


def test_labels_made_after_the_run_are_not_checked_or_used(tmp_path):
    ws = _night(tmp_path)
    on = decide_night(ws, "3", DAY, TH, POL, label_cutoff_ms=450, vet=VetPolicy())
    assert "q0" not in on["vetting"].status


def test_disagreements_reach_the_page_blind_and_are_scored_apart(tmp_path):
    ws = _night(tmp_path)
    runs = tmp_path / "runs"
    shadow_run(ws, "3", DAY, TH, runs, policy=POL, label_cutoff_ms=10_000, run_id="r" * 32, vet=VetPolicy(),
               extra={"dry_run": False})
    meta = json.loads(next(runs.glob("run_*.json")).read_text())
    assert meta["label_check"]["counts"]["not_reviewed"] == 4 and len(meta["disagreements"]) == 2
    with next(runs.glob("labels_*.csv")).open() as fh:
        labels = {r["sighting_id"]: r for r in csv.DictReader(fh)}
    row = labels["q0"]
    assert (row["human_state"], row["human_person_id"], row["label_check"], row["human_reviewed"]) == (
        "known", "pA", "not_reviewed", "0")
    assert labels["a0"]["human_reviewed"] == "1"
    # every staff label the run saw, with its mark: the correction list re-checks older pictures against it
    with next(runs.glob("staff_*.csv")).open() as fh:
        staff = {r["sighting_id"]: r for r in csv.DictReader(fh)}
    assert set(staff) == {s.sighting_id for s in ws.export.sightings if s.human_state}  # u0-u2 too: never named
    assert {"u0", "u1", "u2"} <= set(staff) - set(labels)
    assert (staff["a0"]["human_reviewed"], staff["q0"]["human_reviewed"], staff["s0"]["human_state"]) == ("1", "0", "stranger")
    refs = tmp_path / "refs.json"
    refs.write_text(json.dumps({"people": [{"key": "pA", "person_id": "pA", "photos": [{"url": IMAGE_PREFIX + "r"}]},
                                           {"key": "pB", "person_id": "pB", "photos": [{"url": IMAGE_PREFIX + "r"}]}]}))
    plain = json.loads(build_night(runs, refs, tmp_path / "plain", "3", DAY).read_text())
    assert [it["kind"] for it in plain["items"]].count("check") == 0  # by default they go to the correction list
    fz = json.loads(build_night(runs, refs, tmp_path / "verify", "3", DAY, n_check=None).read_text())
    checks = [i for i, it in enumerate(fz["items"]) if it["kind"] == "check"]
    assert fz["disagreements"] == 2 and len(checks) == 2
    assert "kind" not in json.dumps(page_data(fz)) and "staff" not in json.dumps(page_data(fz))
    q = next(i for i in checks if fz["items"][i]["sighting_id"] == "q0")
    s = next(i for i in checks if fz["items"][i]["sighting_id"] == "s0")
    sc = score_night(fz, {"answers": {str(q): "pA", str(s): "other"}})
    assert sc["checks"]["staff"] == 1 and sc["checks"]["neither"] == 1
    assert sc["right"] + sc["wrong"] == 0  # disagreements never count toward the machine's precision


def test_disagreements_are_mixed_into_the_sample(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    from vbb_reid_server.shadow import LABEL_FIELDS
    with (runs / f"labels_3_{DAY}_rrrrrrrr.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for i in range(20):
            w.writerow({k: "" for k in LABEL_FIELDS} | {"sighting_id": f"p{i}", "state": "known", "relation": "promote",
                                                        "via": "face", "person_id": "pA", "human_labeled": 0,
                                                        "image_url": IMAGE_PREFIX + f"p{i}"})
    dis = [{"sighting_id": f"d{i}", "url": IMAGE_PREFIX + f"d{i}", "machine": "pB", "staff": "pA"} for i in range(40)]
    (runs / f"run_3_{DAY}_rrrrrrrr.json").write_text(json.dumps({"run_id": "r" * 32, "dry_run": False,
                                                                 "disagreements": dis}))
    refs = tmp_path / "refs.json"
    refs.write_text(json.dumps({"people": []}))
    fz = json.loads(build_night(runs, refs, tmp_path / "verify", "3", DAY, n_check=None).read_text())
    kinds = [it["kind"] for it in fz["items"]]
    assert kinds.count("sample") == 20 and kinds.count("check") == 40 and fz["disagreements"] == 40  # all of them, past any old cap of 30
    capped = json.loads(build_night(runs, refs, tmp_path / "v2", "3", DAY, n_check=5).read_text())
    assert [it["kind"] for it in capped["items"]].count("check") == 5
    assert kinds != sorted(kinds, reverse=True)  # not all samples first: the reviewer can't tell them apart


def test_the_reviewed_mark_arrives_with_the_backends_references():
    sightings = [_probe("p1", ms(2026, 10, 20, 11), human_labeled=True, human_state="known")]
    refs = [_ref("r1", "pA", ms(2026, 10, 18)) | {"reviewed": True}, _ref("r2", "pA", ms(2026, 10, 18, 11)),
            _ref("p1", "pA", ms(2026, 10, 20, 11), day="2026-10-20") | {"reviewed": True}]
    ex = export_from_backend(client_for(FakeBackend(sightings, refs, ROSTER)), 3, DAY, window_days=60)
    by = {s.sighting_id: s for s in ex.sightings}
    assert (by["r1"].human_reviewed, by["r2"].human_reviewed, by["p1"].human_reviewed) == (True, False, True)


def test_the_reviewed_mark_arrives_with_the_backends_sightings_too():
    """Stranger and can't-tell overlays never come through references/pull: their mark rides on sightings/pull."""
    sightings = [_probe("s1", ms(2026, 10, 20, 11), human_labeled=True, human_state="stranger", human_reviewed=True),
                 _probe("s2", ms(2026, 10, 20, 12), human_labeled=True, human_state="reset"),
                 _probe("s3", ms(2026, 10, 20, 13), human_reviewed=True)]  # no overlay: a stray mark means nothing
    ex = export_from_backend(client_for(FakeBackend(sightings, [], ROSTER)), 3, DAY, window_days=60)
    by = {s.sighting_id: s for s in ex.sightings}
    assert (by["s1"].human_reviewed, by["s2"].human_reviewed, by["s3"].human_reviewed) == (True, False, False)


def test_the_csv_export_reads_the_reviewed_mark(tmp_path):
    base = {"facility_id": "3", "capture_ts": 40, "corrected_state": "known", "corrected_person_id": "f3A",
            "corrected_at": "2026-10-19 12:00:00", "corrected_ts": 50}
    box = '"crop_face_box": [0.1, 0.1, 0.2, 0.2]'
    rows = [{**base, "sighting_id": "seen", "extras": "{" + box + ', "reviewed": "50:known:f3A"}'},
            {**base, "sighting_id": "later", "extras": '{"reviewed": "49:known:f3A"}'},  # renamed again since
            {**base, "sighting_id": "other", "extras": '{"reviewed": "50:known:f3B"}'},  # the name under it changed
            {**base, "sighting_id": "bare", "extras": '{"reviewed": true}'},
            {**base, "sighting_id": "swept", "extras": "{" + box + "}"},
            {**base, "sighting_id": "broken", "extras": "not json"},
            {**base, "sighting_id": "plain"}]
    write_export(tmp_path / "export", rows, [("f3A", "3", "visitor"), ("f3B", "3", "visitor")])
    by = {s.sighting_id: s for s in load_export(tmp_path / "export").sightings}
    # the mark is "<corrected_ts>:<state>:<person>" of the correction that set it (backend ReviewedMark);
    # it counts only while all three still match the row
    assert [r["sighting_id"] for r in rows if by[r["sighting_id"]].human_reviewed] == ["seen"]
