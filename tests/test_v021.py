"""0.2.1: fixes from the adversarial review of 0.2.0 (2026-11-05). One test per finding; each was
red against 0.2.0 or locks a behaviour 0.2.0 lacked."""

from __future__ import annotations

import http.server
import json
import threading
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from helpers import FACE_ID, write_export, write_face_cache
from test_v020 import (ADMIN, DAY, ROSTER, TOKEN, FakeBackend, FakeStore, _night, _probe, _ref, _run, client_for,
                       ms)
from vbb_reid_server.api import BackendClient, BackendError, check_base_url, export_from_backend
from vbb_reid_server.run import Guards, guard_failures


# 1. lost answers: register / finish replayed safely, the run always ends
def test_a_register_whose_answer_was_lost_is_replayed_and_the_run_finishes(tmp_path):
    fake = _night(tmp_path)
    fake.lose_answer["worker/runs/register"] = 1
    out = _run(tmp_path, fake)
    assert fake.paths().count("worker/runs/register") == 2
    assert fake.runs[out["run_id"]]["status"] == "done"


def test_maintenance_mode_rewriting_a_handled_register_is_safe(tmp_path):
    """ApiResponse.php rewrites code to 5003 AFTER the controller ran: the register happened."""
    fake = _night(tmp_path)
    real = fake.__call__

    calls = {"n": 0}

    def maint(url, body, headers, timeout):
        status, raw = real(url, body, headers, timeout)
        if url.endswith("worker/runs/register") and calls["n"] == 0:
            calls["n"] += 1
            env = json.loads(raw)
            env["code"] = 5003
            return 200, json.dumps(env).encode()
        return status, raw

    c = BackendClient("https://backend.example/api", TOKEN, transport=maint, sleep=lambda s: None)
    from vbb_reid_server.decide import Thresholds
    from vbb_reid_server.pipeline import RefPolicy
    from vbb_reid_server.quality import QualityPolicy
    from vbb_reid_server.run import run_night
    out = run_night(c, 3, DAY, th=Thresholds(face_sim=0.4, face_margin=0.05, face_q_min=0.0, body_enabled=False),
                    quality=QualityPolicy(), policy=RefPolicy(max_refs=50), emb_root=tmp_path / "emb",
                    out_dir=tmp_path / "out", face_id=FACE_ID, worker_id="w", now_ms=lambda: ms(2026, 10, 21, 2, 30),
                    log=lambda *a: None)
    assert fake.runs[out["run_id"]]["status"] == "done"


def test_a_refused_register_still_tries_to_finish_the_run(tmp_path):
    fake = _night(tmp_path)
    fake.fail["worker/runs/register"] = [(200, {"code": 1002, "message": "lease held", "action": None, "data": None})]
    with pytest.raises(BackendError):
        _run(tmp_path, fake)
    assert fake.paths()[-1] == "worker/runs/finish", "the worker cannot tell an applied register from a refused one"


def test_a_signal_during_the_run_finishes_it_as_failed(tmp_path):
    fake = _night(tmp_path)
    real = fake.__call__

    def stop(url, body, headers, timeout):
        if url.endswith("worker/labels/push"):
            raise SystemExit("stopped by signal 15")
        return real(url, body, headers, timeout)

    c = BackendClient("https://backend.example/api", TOKEN, transport=stop, sleep=lambda s: None)
    from vbb_reid_server.decide import Thresholds
    from vbb_reid_server.pipeline import RefPolicy
    from vbb_reid_server.quality import QualityPolicy
    from vbb_reid_server.run import run_night
    with pytest.raises(SystemExit):
        run_night(c, 3, DAY, th=Thresholds(face_sim=0.4, face_margin=0.05, face_q_min=0.0, body_enabled=False),
                  quality=QualityPolicy(), policy=RefPolicy(), emb_root=tmp_path / "emb",
                  out_dir=tmp_path / "out", face_id=FACE_ID, worker_id="w", log=lambda *a: None)
    run = next(iter(fake.runs.values()))
    assert run["status"] == "failed" and "signal 15" in run["error"]


# 2. the token never leaks
def test_a_token_with_a_second_line_is_refused_without_echoing_it():
    with pytest.raises(ValueError) as e:
        BackendClient("https://backend.example/api", "s3cret-AAAAAAAAAAAA\r\nBBB")
    assert "s3cret" not in str(e.value)


def test_an_unexpected_client_error_is_redacted():
    def boom(url, body, headers, timeout):
        raise ValueError(f"Invalid header value {headers['Authorization']!r}")

    c = BackendClient("https://backend.example/api", TOKEN, transport=boom, sleep=lambda s: None)
    with pytest.raises(BackendError) as e:
        c.call("worker/runs/finish", {})
    assert TOKEN not in str(e.value) and "***" in str(e.value)


# 3. redirects and plain http
class _Redirect(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        _Redirect.seen.append((self.path, self.headers.get("Authorization")))
        if self.path.startswith("/api/"):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/elsewhere")
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"code":200,"message":"ok","action":null,"data":{}}')

    do_GET = do_POST

    def log_message(self, *a):
        pass


def test_a_redirect_is_refused_and_the_token_goes_nowhere_else():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Redirect)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        _Redirect.seen = []
        c = BackendClient(f"http://127.0.0.1:{srv.server_port}/api", TOKEN, sleep=lambda s: None, retries=0)
        with pytest.raises(BackendError, match="redirect refused"):
            c.call("worker/runs/finish", {})
        assert [p for p, _ in _Redirect.seen] == ["/api/worker/runs/finish"], "the redirect target was never asked"
    finally:
        srv.shutdown()


def test_only_https_except_to_this_machine():
    assert check_base_url("https://b.example/api/") == "https://b.example/api"
    assert check_base_url("http://127.0.0.1:8080/api") == "http://127.0.0.1:8080/api"
    assert check_base_url("http://localhost/api") == "http://localhost/api"
    with pytest.raises(ValueError):
        check_base_url("http://b.example/api")


# 4. no overlay is ever pushed, reset included
def test_a_sighting_reset_to_pending_is_not_pushed(tmp_path):
    fake = _night(tmp_path)
    fake.sightings[0].update(human_labeled=True, human_state="reset")  # q1 would otherwise be promoted
    _run(tmp_path, fake)
    assert "q1" not in fake.pushed_ever and "q2" not in fake.pushed_ever


# 5. the time zone is checked against what the backend says
def test_the_references_time_zone_is_checked_when_sightings_omit_it():
    fake = FakeBackend([_probe("p1", ms(2026, 10, 20))], [], ROSTER, tz="Europe/Berlin", sightings_tz=False)
    with pytest.raises(BackendError, match="--tz Europe/Berlin"):
        export_from_backend(client_for(fake), 3, "2026-10-20", window_days=60)


def test_a_backend_that_sends_no_time_zone_is_refused():
    class NoTz(FakeBackend):
        def _worker_references_pull(self, p):
            out = super()._worker_references_pull(p)
            out.pop("time_zone", None)
            return out
    fake = NoTz([_probe("p1", ms(2026, 10, 20))], [], ROSTER, sightings_tz=False)
    with pytest.raises(BackendError, match="no time_zone"):
        export_from_backend(client_for(fake), 3, "2026-10-20", window_days=60)


# 6 and 13. guards: thin references, crop failures, an empty run
def test_guards_for_thin_people_failed_crops_and_empty_runs():
    ok = guard_failures({"labels": 5}, Counter(), 30, Guards(), crop_stats={"cdn_hit": 100, "cdn_failed": 2})
    assert ok == []
    assert guard_failures({"labels": 5}, Counter(), 30, Guards(), thin_people={"pB": 0}, min_refs=3)
    assert guard_failures({"labels": 5}, Counter(), 30, Guards(), crop_stats={"cdn_hit": 50, "cdn_failed": 10})
    assert guard_failures({"labels": 0}, Counter(), 30, Guards())
    assert guard_failures({"labels": 0}, Counter(), 5, Guards()) == [], "a small quiet day is not suspect"
    assert guard_failures({"labels": 1}, Counter({"human_labeled": 1}), 30, Guards()), \
        "the worker never pushes overlays, so a human_labeled refusal means something is off"


# 7. one bad day does not stop a range
def test_a_crash_on_one_day_does_not_stop_the_range(tmp_path, monkeypatch):
    import vbb_reid_server.run as runmod
    from vbb_reid_server.cli import cmd_run
    seen = []

    def fake_run_night(client, facility, day, **kw):
        seen.append(day)
        if day == "2026-10-20":
            raise KeyError("sighting_id")
        return {}

    monkeypatch.setattr(runmod, "run_night", fake_run_night)
    tok = tmp_path / "tok"
    tok.write_text(TOKEN + "\n")
    point = tmp_path / "point.json"
    point.write_text(json.dumps({"thresholds": {}, "quality": {}, "ref_policy": {}}))
    args = SimpleNamespace(backend="https://b.example/api", token_file=str(tok),
                           point=str(point), max_refs=None, window_days=None, models=None, emb=str(tmp_path),
                           face_model=None, body_model=None, lan_crops=None, cdn_cache=str(tmp_path / "cdn"),
                           day="2026-10-20..2026-10-21", tz="Asia/Shanghai", facility="3", out=str(tmp_path / "o"),
                           dry_run=False, max_promote_share=None, keep_crops=False)
    with pytest.raises(SystemExit, match="2026-10-20"):
        cmd_run(args)
    assert seen == ["2026-10-20", "2026-10-21"]


# 8. embedding: only what is missing, references face only, no future days
def test_the_second_night_embeds_nothing_new_and_references_skip_the_body_model(tmp_path):
    from test_embed import StubDetector, StubEmbed, jpg
    from vbb_reid_server.crops import CropStore
    from vbb_reid_server.models.registry import ModelSet
    from vbb_reid_server.run import embed_missing
    lan = tmp_path / "lan"
    refs = [_ref(f"r{i}", "pA", ms(2026, 10, 18, 9 + i)) for i in range(3)]
    refs.append(_ref("late", "pA", ms(2026, 10, 22, 9), day="2026-10-22"))  # after the probe day
    sightings = [_probe("q1", ms(2026, 10, 20, 9))]
    for r in refs:
        jpg(lan / "45" / r["day"] / r["sighting_id"] / "0.jpg", (200, 30, 30))
    jpg(lan / "45" / DAY / "q1" / "0.jpg", (200, 30, 30))
    ex = export_from_backend(client_for(FakeBackend(sightings, refs, ROSTER)), 3, DAY, window_days=60)
    face, body = StubEmbed("stub-face"), StubEmbed("stub-body")
    models = ModelSet("stub", StubDetector(), face, body)
    store = CropStore(lan, tmp_path / "cdn", fetch=False)
    first = embed_missing(ex, "3", DAY, models, store, tmp_path / "emb", log=lambda *a: None)
    assert first["embedded_sightings"] == 4, "3 references + 1 probe; the reference after the probe day is skipped"
    assert body.calls == 1, "the body model runs on the probe day only"
    face_calls = face.calls
    second = embed_missing(ex, "3", DAY, models, store, tmp_path / "emb", log=lambda *a: None)
    assert second["embedded_sightings"] == 0 and face.calls == face_calls


# 9. a board prediction under a merged-away id agrees with its survivor
def test_a_board_name_under_a_merged_id_counts_as_agreement(tmp_path):
    fake = _night(tmp_path)
    fake.roster = ROSTER + [{"person_id": "pA_old", "person_type": "visitor", "role": None, "merged_into": "pA",
                             "tentative": False}]
    fake.sightings[0].update(board_state="known", board_person_id="pA_old")
    _run(tmp_path, fake)
    row = next(r for (sid, _), r in fake.labels.items() if sid == "q1")
    assert row["relation"] == "agree", "the roster's merged ids let the worker fold the board's id"


# 10. --roster-only-refs decides without the alias file, reports alias keys
def test_roster_only_shadow_keeps_alias_split_ids_apart_but_reports_alias_keys(tmp_path):
    from vbb_reid_server.cli import main
    rng = np.random.default_rng(3)
    exp, emb = tmp_path / "export", tmp_path / "emb"
    base = ms(2026, 10, 18, 9)
    rows, cache = [], []
    # two roster ids of facility 3 that the alias file says are one human "H"; two refs each
    for i, pid in enumerate(["pX", "pX", "pY", "pY"]):
        sid = f"r{i}"
        rows.append({"sighting_id": sid, "facility_id": "3", "capture_ts": base + i, "identity_state": "uncertain",
                     "corrected_state": "known", "corrected_person_id": pid, "corrected_at": "x", "corrected_ts": 1,
                     "extras": json.dumps({"reviewed": f"1:known:{pid}"})})  # looked at: learned from by default
        cache.append((sid, 0, np.eye(32, dtype=np.float32)[0], 0.9, 80.0))
    rows.append({"sighting_id": "q1", "facility_id": "3", "capture_ts": ms(2026, 10, 20, 9)})
    write_export(exp, rows, [("pX", "3", "visitor"), ("pY", "3", "visitor")])
    (tmp_path / "aliases.csv").write_text("person_id,human\npX,H\npY,H\n")
    write_face_cache(emb, "3", "2026-10-18", cache)
    write_face_cache(emb, "3", DAY, [("q1", 0, np.eye(32, dtype=np.float32)[0], 0.9, 80.0)])
    (tmp_path / "th.json").write_text(json.dumps({"face_sim": 0.4, "face_margin": 0.05, "face_q_min": 0.0,
                                                  "min_refs": 2, "body_enabled": False}))
    common = ["shadow", "--export", str(exp), "--aliases", str(tmp_path / "aliases.csv"), "--emb", str(emb),
              "--face-model", "test-face", "--facility", "3", "--days", DAY, "--thresholds", str(tmp_path / "th.json"),
              "--labels-until", "all"]
    main(common + ["--out", str(tmp_path / "pooled")])
    main(common + ["--roster-only-refs", "--out", str(tmp_path / "prod")])
    import csv, glob
    def labels(d):
        return list(csv.DictReader(open(glob.glob(str(tmp_path / d / "labels_*.csv"))[0], encoding="utf-8")))
    def run(d):
        return json.load(open(glob.glob(str(tmp_path / d / "run_*.json"))[0], encoding="utf-8"))
    assert run("pooled")["references"] == {"H": 4}
    assert run("prod")["references"] == {"pX": 2, "pY": 2}, "production keeps the two ids apart"
    assert [l["state"] for l in labels("pooled") if l["sighting_id"] == "q1"] == ["known"]
    assert not [l for l in labels("prod") if l["sighting_id"] == "q1" and l["state"] == "known"], \
        "the two ids tie, so production abstains - what 6g must show, not hide"


# 12. a repeating cursor stops
def test_a_repeating_cursor_is_refused():
    class Loop(FakeBackend):
        def _worker_sightings_pull(self, p):
            return {"sightings": [], "next_cursor": "same", "time_zone": "Asia/Shanghai"}
    with pytest.raises(BackendError, match="repeats"):
        client_for(Loop([], [], [])).pull_sightings(3, DAY)
