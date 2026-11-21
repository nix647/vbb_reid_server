"""The nightly run with the sorter as its namer (docs/DESIGN_nightly_regroup.md, "Wiring", W3, W4, W5, W6): the
production run against the in-memory backend of test_v020, the check page's pool, and the option plumbing."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import basis, near, write_face_cache
from test_v020 import DAY, NOW, TOKEN, _night, _run, ms
from vbb_reid_server.cli import build_parser, cmd_run
from vbb_reid_server.regroup_inputs import EarlierPolicy
from vbb_reid_server.shadow import LABEL_FIELDS
from vbb_reid_server.verify import IMAGE_PREFIX, build_night

PRIOR = "2026-10-19"


def pulled_days(fake):
    """The days the run asked sightings for (a day of several pages counts once)."""
    return sorted({c[1]["day"] for c in fake.calls if c[0] == "worker/sightings/pull"})


# W3: run_night ----------------------------------------------------------------------------------------------------


def test_a_machine_name_that_differs_from_the_boards_is_pushed_as_a_name(tmp_path):
    fake = _night(tmp_path)
    fake.sightings[0].update(board_state="known", board_person_id="pB")  # the board says pB, the face is pA's
    out = _run(tmp_path, fake)
    row = next(r for (sid, _), r in fake.labels.items() if sid == "q1")
    assert (row["state"], row["relation"], row["person_id"]) == ("known", "relabel", "pA")
    counts = fake.runs[out["run_id"]]["counts"]
    assert counts["pushed_relabel"] == 1 and counts["accepted"] == 1


def test_the_run_record_names_the_sorter_and_summarises_its_work(tmp_path):
    fake = _night(tmp_path)
    out = _run(tmp_path, fake)
    record = json.loads(next((tmp_path / "out").glob("run_*.json")).read_text())
    assert record["namer"] == "regroup" and record["dry_run"] is False and record["run_id"] == out["run_id"]
    summary = record["regroup"]
    assert summary["rows"] == 2 and summary["reasons"]["named"] >= 1  # q1 and q2: the file keeps the overlay's row
    assert summary["via"]["face"] >= 1
    assert {"face_groups", "disputes", "dissolved", "displaced", "refuted", "earlier", "rounds"} <= set(summary)
    assert out["regroup"] == summary and set(summary["face_groups"]) == {"pA", "pB"}
    assert all(g["active"] for g in summary["face_groups"].values())
    reg = fake.calls[fake.paths().index("worker/runs/register")][1]
    assert reg["thresholds"]["namer"] == "regroup"


def test_the_local_files_keep_their_columns_and_the_overlay_decision(tmp_path):
    fake = _night(tmp_path)
    out = _run(tmp_path, fake)
    path = next((tmp_path / "out").glob(f"labels_3_{DAY}_{out['run_id'][:8]}.csv"))
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = {r["sighting_id"]: r for r in reader}
    assert reader.fieldnames == LABEL_FIELDS
    assert rows["q2"]["human_labeled"] == "1" and rows["q1"]["human_labeled"] == "0" and rows["q1"]["label_check"] == ""
    assert "q3" not in rows  # an unnamed probe has no row
    assert list((tmp_path / "out").glob("staff_*.csv"))


def test_the_earlier_nights_are_pulled_for_their_overlays(tmp_path):
    fake = _night(tmp_path)
    _run(tmp_path, fake)
    assert pulled_days(fake) == [f"2026-10-{d}" for d in range(13, 21)]  # the probe night and the 7 nights before
    fake = _night(tmp_path)
    _run(tmp_path, fake, earlier=EarlierPolicy(nights=0))
    assert pulled_days(fake) == [DAY]


def test_a_night_whose_pull_failed_is_left_out_of_the_look_back(tmp_path):
    from test_v020 import Refused
    fake = _night(tmp_path)
    real = fake._worker_sightings_pull

    def pull(p):
        if p["day"] == PRIOR:
            raise Refused("backend busy")
        return real(p)
    fake._worker_sightings_pull = pull
    out = _run(tmp_path, fake)
    nights = {n["day"]: n["status"] for n in out["regroup"]["earlier"]["nights"]}
    assert nights[PRIOR] == "no_overlays" and nights["2026-10-18"] == "no_run"
    assert fake.runs[out["run_id"]]["status"] == "done"  # one failed pull does not fail the run


def test_the_earlier_nights_names_top_up_a_thin_person_in_the_run(tmp_path):
    fake = _night(tmp_path)
    rng = np.random.default_rng(9)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / f"run_3_{PRIOR}_aaaa1111.json").write_text(json.dumps(
        {"run_id": "aaaa1111", "dry_run": False, "started_at": 1, "namer": "regroup"}))
    with (out_dir / f"labels_3_{PRIOR}_aaaa1111.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sighting_id", "state", "via", "person_id"])
        w.writerows([[f"e{i}", "known", "face", "pA"] for i in range(5)])
    write_face_cache(tmp_path / "emb", "3", PRIOR,
                     [(f"e{i}", 0, near(basis(0), 0.05, rng), 0.9, 80.0) for i in range(5)])
    out = _run(tmp_path, fake, earlier=EarlierPolicy(top_up=40))  # pA has 35 confirmed pictures: thin at 40
    rep = out["regroup"]["earlier"]
    assert rep["added"] == {"pA": 5} and rep["nights"][0]["status"] == "used"
    assert fake.runs[out["run_id"]]["counts"]["earlier_added"] == 5


def test_a_dry_run_writes_the_sorters_files_and_pushes_nothing(tmp_path):
    fake = _night(tmp_path)
    out = _run(tmp_path, fake, dry_run=True)
    assert not fake.runs and out["counts"]["pushed_promote"] == 1
    record = json.loads(next((tmp_path / "out").glob("run_*.json")).read_text())
    assert record["namer"] == "regroup" and record["dry_run"] is True


def test_the_guards_still_fire_and_the_thin_references_guard_is_gone(tmp_path):
    fake = _night(tmp_path, reject={"q1": "not_in_roster"})
    out = _run(tmp_path, fake)
    assert out["guard_failures"] == ["1 row(s) rejected as not_in_roster"]
    fake = _night(tmp_path)
    out = _run(tmp_path, fake)
    assert out["guard_failures"] == [], "no 'named person below N references' any more: the sorter has no references"


# W4: the check page -----------------------------------------------------------------------------------------------


def _files(runs: Path, run_id: str, dry: bool, rows, finished=1.0):
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"run_3_2026-11-06_{run_id[:8]}.json").write_text(json.dumps(
        {"run_id": run_id, "finished_at": finished, "dry_run": dry, "model": "m"}))
    with (runs / f"labels_3_2026-11-06_{run_id[:8]}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for sid, relation in rows:
            row = {"sighting_id": sid, "state": "known", "relation": relation, "via": "face", "person_id": "pA3",
                   "human_labeled": 0, "image_url": IMAGE_PREFIX + f"x/{sid}.jpg"}
            w.writerow({k: row.get(k, "") for k in LABEL_FIELDS})


def _refs(tmp_path):
    p = tmp_path / "refs.json"
    p.write_text(json.dumps({"built_at": 1, "people": [{"person_id": "pA3", "key": "H1", "name": "", "photos": []}]}))
    return p


def test_the_check_page_draws_from_promotions_and_corrections_alike(tmp_path):
    runs = tmp_path / "runs"
    _files(runs, "a" * 32, False,
           [(f"p{i}", "promote") for i in range(10)] + [(f"r{i}", "relabel") for i in range(10)] + [("same", "agree")])
    f = build_night(runs, _refs(tmp_path), tmp_path / "v", "3", "2026-11-06", n=20, log=lambda *_: None)
    data = json.loads(f.read_text())
    sids = {it["sighting_id"] for it in data["items"]}
    assert data["pool"] == 20 and sids == {f"p{i}" for i in range(10)} | {f"r{i}" for i in range(10)}


def test_test_runs_draws_from_the_newest_dry_run_and_otherwise_from_the_real_one(tmp_path):
    runs = tmp_path / "runs"
    _files(runs, "a" * 32, False, [(f"real{i}", "promote") for i in range(5)], finished=5.0)
    _files(runs, "b" * 32, True, [(f"old{i}", "promote") for i in range(5)], finished=6.0)
    _files(runs, "c" * 32, True, [(f"dry{i}", "relabel") for i in range(5)], finished=9.0)
    f = build_night(runs, _refs(tmp_path), tmp_path / "t", "3", "2026-11-06", n=20, test_runs=True,
                    log=lambda *_: None)
    data = json.loads(f.read_text())
    assert data["run_id"] == "c" * 32 and {it["sighting_id"] for it in data["items"]} == {f"dry{i}" for i in range(5)}
    g = build_night(runs, _refs(tmp_path), tmp_path / "r", "3", "2026-11-06", n=20, log=lambda *_: None)
    assert json.loads(g.read_text())["run_id"] == "a" * 32
    only_real = tmp_path / "runs2"
    _files(only_real, "d" * 32, False, [("x", "promote")])
    assert build_night(only_real, _refs(tmp_path), tmp_path / "n", "3", "2026-11-06", test_runs=True,
                       log=lambda *_: None) is None  # a test line draws only from dry runs


# W6: options ------------------------------------------------------------------------------------------------------


def _run_args(tmp_path, **kw):
    tok = tmp_path / "tok"
    tok.write_text(TOKEN + "\n")
    point = tmp_path / "point.json"
    point.write_text(json.dumps({"thresholds": {}, "quality": {}, "ref_policy": {}}))
    base = dict(backend="https://b.example/api", token_file=str(tok), point=str(point), max_refs=None,
                window_days=None, models=None, emb=str(tmp_path), face_model=None, body_model=None, lan_crops=None,
                cdn_cache=str(tmp_path / "cdn"), day="2026-10-20", tz="Asia/Shanghai", facility="3",
                out=str(tmp_path / "o"), dry_run=False, max_promote_share=None, keep_crops=False)
    return SimpleNamespace(**{**base, **kw})


def _captured(tmp_path, monkeypatch, **kw):
    import vbb_reid_server.run as runmod
    seen = {}
    monkeypatch.setattr(runmod, "run_night", lambda client, facility, day, **k: seen.update(k) or {})
    cmd_run(_run_args(tmp_path, **kw))
    return seen["earlier"]


def test_the_run_options_set_the_earlier_nights_policy(tmp_path, monkeypatch):
    assert _captured(tmp_path, monkeypatch) == EarlierPolicy(nights=7, test_runs=False)  # no options: the defaults
    got = _captured(tmp_path, monkeypatch, earlier_nights=3, earlier_test_runs=True)
    assert (got.nights, got.test_runs) == (3, True)
    assert _captured(tmp_path, monkeypatch, earlier_nights=0).nights == 0


def test_an_answered_picture_is_an_anchor_in_the_run(tmp_path):
    fake = _night(tmp_path)
    root = tmp_path / "verify"
    root.mkdir()
    (root / "refs.json").write_text(json.dumps({  # q3 was answered as pB before the run started
        "facility_id": "3", "built_at": 1,
        "people": [{"person_id": "pB", "photos": [{"sighting_id": "q3", "day": DAY}]}]}))
    out = _run(tmp_path, fake, verify_root=root)
    assert out["regroup"]["reasons"].get("anchor") == 1
    assert "q3" not in {sid for (sid, _), r in fake.labels.items()}  # a human's answer needs no machine row


def test_verify_build_gets_the_test_runs_option(tmp_path, monkeypatch):
    import vbb_reid_server.verify as vmod
    from vbb_reid_server.cli import cmd_verify_build
    seen = []
    monkeypatch.setattr(vmod, "build_night", lambda *a, **k: seen.append(k))
    args = SimpleNamespace(runs=str(tmp_path), refs=str(tmp_path / "r.json"), out=str(tmp_path), facility="3",
                           day="2026-10-20", n=20, run_id=None)
    cmd_verify_build(args)
    cmd_verify_build(SimpleNamespace(**{**vars(args), "test_runs": True}))
    assert [k["test_runs"] for k in seen] == [False, True]


def test_the_parser_knows_the_new_options():
    p = build_parser()
    base = ["run", "--backend", "b", "--token-file", "t", "--facility", "3", "--point", "p", "--emb", "e",
            "--cdn-cache", "c", "--out", "o"]
    a = p.parse_args(base)
    assert (a.earlier_nights, a.earlier_test_runs) == (7, False)
    a = p.parse_args(base + ["--earlier-nights", "0", "--earlier-test-runs"])
    assert (a.earlier_nights, a.earlier_test_runs) == (0, True)
    v = ["verify-build", "--runs", "r", "--refs", "f", "--out", "o", "--facility", "3"]
    assert p.parse_args(v).test_runs is False and p.parse_args(v + ["--test-runs"]).test_runs is True


# W5: scripts/nightly_run.sh ----------------------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nightly_run.sh"


NEEDS_BOX = pytest.mark.skipif(
    not SCRIPT.exists() or sys.platform == "darwin"
    or not (Path("/usr/bin/flock").exists() or Path("/bin/flock").exists()),
    reason="the script needs flock and GNU timeout (the GPU box has them) and is not in every test copy")


def _run_script(tmp_path, *args, test_line=False):
    """The real script against a stub `vbb-reid-server` that records its arguments, one line per call."""
    venv, root = tmp_path / "venv", tmp_path / "root"
    (venv / "bin").mkdir(parents=True)
    stub = venv / "bin" / "vbb-reid-server"
    stub.write_text('#!/bin/bash\necho "$@" >> "$STUB_LOG"\n')
    stub.chmod(0o755)
    (root / "verify").mkdir(parents=True)
    (root / "verify" / "refs.json").write_text("{}")
    env = {**os.environ, "VBB_RUN_VENV": str(venv), "VBB_RUN_ROOT": str(root), "STUB_LOG": str(tmp_path / "calls.txt"),
           "HOME": str(tmp_path)}
    env.pop("VBB_RUN_TEST", None)
    if test_line:
        env["VBB_RUN_TEST"] = "1"
    done = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)
    calls = (tmp_path / "calls.txt").read_text().splitlines() if (tmp_path / "calls.txt").exists() else []
    return done, [c.split() for c in calls]


@NEEDS_BOX
def test_the_script_runs_the_normal_path_when_not_asked_for_the_test_line(tmp_path):
    done, calls = _run_script(tmp_path)
    assert done.returncode == 0, done.stderr
    assert [c[0] for c in calls] == ["run", "verify-build", "corrections"]
    assert "--dry-run" not in calls[0] and "--earlier-test-runs" not in calls[0] and "--test-runs" not in calls[1]


@NEEDS_BOX
def test_the_test_line_is_a_dry_run_with_earlier_test_runs_a_dry_check_page_and_no_corrections(tmp_path):
    done, calls = _run_script(tmp_path, test_line=True)
    assert done.returncode == 0, done.stderr
    assert [c[0] for c in calls] == ["run", "verify-build"]  # no correction list
    assert "--dry-run" in calls[0] and "--earlier-test-runs" in calls[0]
    assert "--test-runs" in calls[1]
    (tmp_path / "b").mkdir()
    done, calls = _run_script(tmp_path / "b", "--day", "2026-11-02", test_line=True)
    assert [c[0] for c in calls] == ["run"] and calls[0].count("--day") == 2  # an argument run skips the check page
