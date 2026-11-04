"""0.2.0: the /worker API client, the backend export, the roster-only reference rule, and the
production run end to end against an in-memory backend (no network, no GPU)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from helpers import FACE_ID, basis, near, sighting, write_face_cache
from vbb_reid_server.api import BackendClient, BackendError, export_from_backend
from vbb_reid_server.data import Export, Person
from vbb_reid_server.decide import Thresholds
from vbb_reid_server.pipeline import RefPolicy, Workspace
from vbb_reid_server.quality import QualityPolicy
from vbb_reid_server.run import Guards, label_row, model_id_of, run_night

TZ = ZoneInfo("Asia/Shanghai")
TOKEN = "s3cret-worker-token"
ADMIN = "s3cret-admin-token-1"


def ms(y, mo, d, h=10, mi=0):
    return int(datetime(y, mo, d, h, mi, tzinfo=TZ).timestamp() * 1000)


# ---------------------------------------------------------------- fake backend --
class Refused(Exception):
    pass


class FakeBackend:
    """The /worker API as machine_labels.md §3 describes it, in memory. Pages of ``page`` rows."""

    def __init__(self, sightings, references, roster, *, tz="Asia/Shanghai", page=2, reject=None,
                 sightings_tz=True):
        self.sightings, self.references, self.roster, self.tz, self.page = sightings, references, roster, tz, page
        self.sightings_tz = sightings_tz
        self.reject = reject or {}  # sighting_id -> reason
        self.calls: list[tuple[str, dict]] = []
        self.runs: dict[str, dict] = {}
        self.labels: dict[tuple[str, str], dict] = {}
        self.fail: dict[str, list] = {}  # path -> queue of (status, body) to return before the real answer
        self.lose_answer: dict[str, int] = {}  # path -> how many handled requests lose their answer
        self.pushed_ever: set[str] = set()

    def _page(self, rows, cursor):
        start = int(cursor or 0)
        chunk = rows[start:start + self.page]
        nxt = str(start + self.page) if start + self.page < len(rows) else None
        return chunk, nxt

    def __call__(self, url, body, headers, timeout):
        path = url.split("/api/", 1)[1]
        payload = json.loads(body)
        self.calls.append((path, payload))
        if self.fail.get(path):
            status, out = self.fail[path].pop(0)
            return status, json.dumps(out).encode() if isinstance(out, dict) else out
        auth = headers.get("Authorization")
        if auth not in (f"Bearer {TOKEN}", f"Bearer {ADMIN}"):
            return 200, self._env(401, "bad token")
        if path.endswith(("activate", "revert")) and auth != f"Bearer {ADMIN}":
            return 200, self._env(1002, "admin scope required")
        try:
            data = getattr(self, "_" + path.replace("/", "_"))(payload)
        except Refused as e:
            return 200, self._env(1002, str(e))
        out = self._env(200, "ok", data)
        if self.lose_answer.get(path):  # the request was handled; the answer never arrives
            self.lose_answer[path] -= 1
            return 504, b"gateway timeout"
        return 200, out

    lose_answer: dict = {}

    @staticmethod
    def _env(code, message, data=None):
        return json.dumps({"code": code, "message": message, "action": None, "data": data}).encode()

    def _worker_runs_register(self, p):
        if p["run_id"] in self.runs:  # idempotent: the same run again returns it (machine_labels.md §3)
            return {"run_id": p["run_id"], "lease_until": 0}
        self.runs[p["run_id"]] = {"status": "running", **p}
        return {"run_id": p["run_id"], "lease_until": 0}

    def _worker_sightings_pull(self, p):
        rows = [s for s in self.sightings if s["_day"] == p["day"]]
        rows = [{k: v for k, v in s.items() if not k.startswith("_")} for s in rows]
        chunk, nxt = self._page(rows, p.get("cursor"))
        out = {"sightings": chunk, "next_cursor": nxt}
        if self.sightings_tz:
            out["time_zone"] = self.tz
        return out

    def _worker_references_pull(self, p):
        rows = [r for r in self.references if r["capture_ts"] >= p["since_ts"]]
        chunk, nxt = self._page(rows, p.get("cursor"))
        out = {"references": chunk, "next_cursor": nxt}
        if not p.get("cursor"):
            out.update(roster=self.roster, time_zone=self.tz)
        return out

    def _worker_labels_push(self, p):
        if self.runs.get(p["run_id"], {}).get("status") != "running":
            raise Refused("run is not running")
        assert 1 <= len(p["rows"]) <= 500
        overlay = {s["sighting_id"] for s in self.sightings if s.get("human_labeled")}
        rejected = []
        for r in p["rows"]:
            self.pushed_ever.add(r["sighting_id"])
            reason = self.reject.get(r["sighting_id"]) or ("human_labeled" if r["sighting_id"] in overlay
                                                           and r["state"] == "known" else None)
            if reason:
                rejected.append({"sighting_id": r["sighting_id"], "reason": reason})
            else:
                self.labels[(r["sighting_id"], p["run_id"])] = r
        return {"accepted": len(p["rows"]) - len(rejected), "rejected": rejected}

    def _worker_runs_finish(self, p):
        run = self.runs.get(p["run_id"])
        if run is None:
            raise Refused("unknown run")
        if run["status"] == p["status"]:  # idempotent
            return {"run_id": p["run_id"], "status": run["status"]}
        if run["status"] != "running":
            raise Refused(f"run is {run['status']}")
        run.update(status=p["status"], counts=p["counts"], error=p.get("error"))
        return {"run_id": p["run_id"], "status": p["status"]}

    def _worker_runs_activate(self, p):
        run = self.runs[p["run_id"]]
        if run["status"] not in ("done", "activated"):
            raise Refused(f"run is {run['status']}")
        run["status"] = "activated"
        return {"run_id": p["run_id"], "status": "activated"}

    def paths(self):
        return [c[0] for c in self.calls]


def client_for(fake, token=TOKEN, **kw):
    return BackendClient("https://backend.example/api", token, transport=fake, sleep=lambda s: None, **kw)


# ----------------------------------------------------------------- the client --
def test_envelope_errors_are_raised_at_once_and_service_errors_are_retried():
    fake = FakeBackend([], [], [])
    fake.fail["worker/runs/register"] = [
        (502, b"bad gateway"),
        (200, {"code": 500, "message": "busy", "action": None, "data": None}),
    ]
    c = client_for(fake)
    assert c.register("a" * 32, 3, "2026-10-20", "m", "w", {})["run_id"] == "a" * 32
    assert fake.paths().count("worker/runs/register") == 3, "two retries, then the real answer"

    fake.fail["worker/runs/finish"] = [(200, {"code": 1002, "message": "run is done", "action": None, "data": None})]
    with pytest.raises(BackendError) as e:
        c.finish("a" * 32, "done", {})
    assert e.value.code == 1002 and fake.paths().count("worker/runs/finish") == 1, "a refusal is not retried"


def test_a_transport_failure_is_retried_and_then_given_up():
    attempts = []

    def down(url, body, headers, timeout):
        attempts.append(1)
        raise OSError("connection refused")

    c = BackendClient("https://backend.example/api", TOKEN, transport=down, retries=3, sleep=lambda s: None)
    with pytest.raises(BackendError, match="retries exhausted"):
        c.call("worker/runs/finish", {})
    assert len(attempts) == 4


def test_the_token_never_appears_in_errors_or_repr():
    fake = FakeBackend([], [], [])
    c = BackendClient("https://backend.example/api", "wrong-token-0123456", transport=fake, sleep=lambda s: None)
    with pytest.raises(BackendError) as e:
        c.call("worker/runs/finish", {})
    assert e.value.code == 401
    assert "wrong-token" not in str(e.value) and "wrong-token" not in repr(c)


def test_pages_follow_the_cursor_to_the_end():
    rows = [{"sighting_id": f"s{i}", "did": "45", "room_id": 9, "board_state": "uncertain", "capture_ts": i,
             "human_labeled": False, "_day": "2026-10-20"} for i in range(5)]
    fake = FakeBackend(rows, [], [], page=2)
    got, tz = client_for(fake).pull_sightings(3, "2026-10-20")
    assert [r["sighting_id"] for r in got] == ["s0", "s1", "s2", "s3", "s4"] and tz == "Asia/Shanghai"
    assert [c[1]["cursor"] for c in fake.calls] == [None, "2", "4"]
    assert all(c[1]["limit"] == 500 for c in fake.calls)


# ------------------------------------------------------- the backend export --
def _probe(sid, t, **kw):
    row = {"sighting_id": sid, "did": "45", "room_id": 9, "track_id": 1, "board_state": "uncertain",
           "board_person_id": None, "provisional_person_id": None, "crop_is_face": True, "crop_quality": 0.6,
           "image_url": f"https://cdn.example/{sid}.jpg", "capture_ts": t, "track_started_ts": t - 1000,
           "track_ended_ts": t, "human_labeled": False, "_day": "2026-10-20"}
    row.update(kw)
    return row


def _ref(sid, pid, t, *, facility=3, day="2026-10-18", corrected=None):
    return {"sighting_id": sid, "facility_id": facility, "person_id": pid, "did": "45",
            "image_url": f"https://cdn.example/{sid}.jpg", "crop_is_face": True, "capture_ts": t,
            "corrected_ts": corrected or t + 3_600_000, "day": day}


ROSTER = [{"person_id": "pA", "person_type": "visitor", "role": None, "merged_into": None, "tentative": False},
          {"person_id": "pB", "person_type": "visitor", "role": None, "merged_into": None, "tentative": False}]


def test_the_export_keeps_overlays_and_puts_same_day_labels_on_their_probe():
    sightings = [_probe("p1", ms(2026, 10, 20)),
                 _probe("p2", ms(2026, 10, 20, 11), human_labeled=True, human_state="known"),
                 _probe("p3", ms(2026, 10, 20, 12), human_labeled=True, human_state="reset"),
                 _probe("p4", ms(2026, 10, 20, 13), human_labeled=True)]  # an older backend without human_state
    refs = [_ref("r1", "pA", ms(2026, 10, 18)), _ref("p2", "pA", ms(2026, 10, 20, 11), day="2026-10-20"),
            _ref("x1", "pA", ms(2026, 10, 19), facility=2, day="2026-10-19")]
    ex = export_from_backend(client_for(FakeBackend(sightings, refs, ROSTER)), 3, "2026-10-20", window_days=60)

    by = {s.sighting_id: s for s in ex.sightings}
    assert by["p1"].human_state is None and not by["p1"].human_labeled
    assert (by["p2"].human_state, by["p2"].human_person_id, by["p2"].day) == ("known", "pA", "2026-10-20")
    assert by["p3"].human_state == "reset"
    assert by["p4"].human_labeled, "labeled with an unknown kind still counts as labeled (never pushed)"
    assert (by["x1"].facility_id, by["x1"].day, by["x1"].human_person_id) == ("2", "2026-10-19", "pA")
    assert ex.roster_ids("3") == {"pA", "pB"}
    assert [s.sighting_id for s in ex.by_facility_day("3", "2026-10-20")] == ["p1", "p2", "p3", "p4"]


def test_a_facility_in_another_time_zone_is_refused():
    fake = FakeBackend([_probe("p1", ms(2026, 10, 20))], [], ROSTER, tz="Europe/Berlin")
    with pytest.raises(BackendError, match="--tz Europe/Berlin"):
        export_from_backend(client_for(fake), 3, "2026-10-20", window_days=60)


def test_references_start_at_local_midnight_of_the_window():
    fake = FakeBackend([_probe("p1", ms(2026, 10, 20))], [], ROSTER)
    export_from_backend(client_for(fake), 3, "2026-10-20", window_days=60)
    since = [c[1]["since_ts"] for c in fake.calls if c[0] == "worker/references/pull"][0]
    assert since == ms(2026, 8, 21, 0, 0)


# --------------------------------------------------- roster-only references --
def test_roster_only_drops_a_sibling_campus_id_the_alias_file_had_unified(tmp_path):
    """Phase A unified facility 2's and 3's ids of one human with an alias file. Production has none,
    so references/pull sends only labels of this facility's roster; --roster-only-refs measures that."""
    people = {"pA3": Person("pA3", "3", "visitor"), "pA2": Person("pA2", "2", "visitor")}
    s3 = [sighting(f"a3_{i}", facility="3", day="2026-10-18", human_state="known", human_pid="pA3",
                   corrected_ts=1) for i in range(3)]
    s2 = [sighting(f"a2_{i}", facility="2", day="2026-10-18", human_state="known", human_pid="pA2",
                   corrected_ts=1) for i in range(3)]
    ex = Export(root=tmp_path, sightings=s3 + s2, people=people, facility_tz={"2": "Asia/Shanghai", "3": "Asia/Shanghai"},
                facility_org={"2": "o", "3": "o"}, aliases={"pA3": "H1", "pA2": "H1"})
    rng = np.random.default_rng(0)
    for fac, ss in (("3", s3), ("2", s2)):
        write_face_cache(tmp_path, fac, "2026-10-18", [(s.sighting_id, 0, near(basis(0), 0.05, rng), 0.9, 80.0) for s in ss])
    ws = Workspace(ex, tmp_path, FACE_ID, None, QualityPolicy())
    every = ws.references(["2", "3"], "2026-10-20", label_cutoff_ms=None, policy=RefPolicy())
    own = ws.references(["2", "3"], "2026-10-20", label_cutoff_ms=None, policy=RefPolicy(roster_only=True),
                        roster_ids=ex.roster_ids("3"))
    assert every.n_sightings("H1") == 6 and own.n_sightings("H1") == 3
    with pytest.raises(ValueError):
        ws.references(["3"], "2026-10-20", label_cutoff_ms=None, policy=RefPolicy(roster_only=True))


def test_fold_follows_merges_but_not_aliases(tmp_path):
    people = {"old": Person("old", "3", merged_into="new"), "new": Person("new", "3")}
    ex = Export(root=tmp_path, sightings=[], people=people, facility_tz={}, facility_org={}, aliases={"new": "H1"})
    assert ex.fold("old") == "new" and ex.key("old") == "H1" and ex.roster_ids("3") == {"new"}


# ------------------------------------------------------------ the full run --
DAY = "2026-10-20"
NOW = ms(2026, 10, 21, 2, 30)


def _night(tmp_path, *, reject=None):
    """pA and pB have 4 labeled references each on 10-18. The probe day holds: q1 (pA's face, board
    unsure -> promote), q2 (pA's face, already human-labeled -> decided, never pushed), q3 (a face
    like nobody -> abstain)."""
    rng = np.random.default_rng(1)
    refs, cache_ref = [], []
    for who, axis in (("pA", 0), ("pB", 1)):
        for i in range(4):
            sid = f"r{who}{i}"
            refs.append(_ref(sid, who, ms(2026, 10, 18, 9 + i)))
            cache_ref.append((sid, 0, near(basis(axis), 0.05, rng), 0.9, 80.0))
    write_face_cache(tmp_path / "emb", "3", "2026-10-18", cache_ref)
    sightings = [_probe("q1", ms(2026, 10, 20, 9)), _probe("q2", ms(2026, 10, 20, 10), human_labeled=True,
                                                         human_state="stranger"),
                 _probe("q3", ms(2026, 10, 20, 11))]
    write_face_cache(tmp_path / "emb", "3", DAY, [("q1", 0, near(basis(0), 0.05, rng), 0.9, 80.0),
                                                  ("q2", 0, near(basis(0), 0.05, rng), 0.9, 80.0),
                                                  ("q3", 0, near(basis(5), 0.05, rng), 0.9, 80.0)])
    return FakeBackend(sightings, refs, ROSTER, reject=reject)


class FakeStore:
    def __init__(self, stats=None):
        self.deleted = 0
        self.pruned = []
        self.stats = stats or {"lan": 0, "cdn_hit": 3, "cdn_fetched": 0, "cdn_failed": 0, "missing": 0}

    def delete_fetched(self):
        self.deleted += 1
        return 0

    def prune_cdn(self, days):
        self.pruned.append(days)
        return 0


def _run(tmp_path, fake, **kw):
    th = Thresholds(face_sim=0.40, face_margin=0.05, face_q_min=0.0, min_face_norm=14.0, body_enabled=False)
    return run_night(client_for(fake), 3, DAY, th=th, quality=QualityPolicy(), policy=RefPolicy(max_refs=50),
                     emb_root=tmp_path / "emb", out_dir=tmp_path / "out", face_id=FACE_ID, worker_id="gpu-ubun",
                     now_ms=lambda: NOW, log=lambda *a: None, **kw)


def test_a_night_registers_pushes_only_unlabeled_sightings_and_finishes(tmp_path):
    fake = _night(tmp_path)
    store = FakeStore()
    out = _run(tmp_path, fake, store=store)

    paths = fake.paths()
    assert paths[-1] == "worker/runs/finish"
    assert paths.index("worker/runs/register") > max(i for i, p in enumerate(paths) if p.endswith("/pull")), \
        "register comes after the pulls and the decision, so the lease covers only push and finish"
    assert paths.index("worker/runs/register") < paths.index("worker/labels/push")
    reg = fake.calls[paths.index("worker/runs/register")][1]
    assert reg["facility_id"] == 3 and reg["day"] == DAY and reg["worker_id"] == "gpu-ubun"
    from vbb_reid_server import __version__
    assert reg["model_id"] == model_id_of(FACE_ID, None) and reg["model_id"].endswith(f"vbb-reid-server {__version__}")
    assert reg["thresholds"]["ref_policy"]["roster_only"] is True, "production always uses the roster-only rule"
    pushed = {sid: r for (sid, run), r in fake.labels.items()}
    assert set(pushed) == {"q1"}, "q2 is human-labeled (never pushed); q3 looks like nobody (no row)"
    assert (pushed["q1"]["state"], pushed["q1"]["relation"], pushed["q1"]["person_id"]) == ("known", "promote", "pA")
    run = fake.runs[out["run_id"]]
    assert run["status"] == "done" and run["counts"]["pushed_promote"] == 1 and run["counts"]["accepted"] == 1
    assert store.deleted == 1 and store.pruned == [1.0], "downloaded crops and day-old leftovers are deleted"
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert any(f.startswith(f"labels_3_{DAY}_{out['run_id'][:8]}") for f in files)
    with open(next((tmp_path / "out").glob("labels_*.csv")), newline="", encoding="utf-8") as fh:
        local = {r["sighting_id"]: r for r in csv.DictReader(fh)}
    assert local["q2"]["human_labeled"] == "1", "the local file still shows the decision on q2"


def test_the_worker_never_activates_and_reports_its_guards(tmp_path):
    fake = _night(tmp_path)
    out = _run(tmp_path, fake)
    assert "worker/runs/activate" not in fake.paths() and fake.runs[out["run_id"]]["status"] == "done"
    assert fake.runs[out["run_id"]]["counts"]["guard_failures"] == 0

    fake = _night(tmp_path, reject={"q1": "not_in_roster"})
    out = _run(tmp_path, fake)
    assert out["guard_failures"] and fake.runs[out["run_id"]]["counts"]["guard_failures"] == 1
    assert fake.runs[out["run_id"]]["counts"]["rejected_not_in_roster"] == 1

    fake = _night(tmp_path)
    out = _run(tmp_path, fake, guards=Guards(max_promote_share=0.2))
    assert fake.runs[out["run_id"]]["counts"]["guard_failures"] == 1, "1 promotion of 3 sightings is above 20 %"


def test_a_failure_after_register_marks_the_run_failed(tmp_path):
    fake = _night(tmp_path)
    fake.fail["worker/labels/push"] = [(200, {"code": 1002, "message": "run is not running", "action": None,
                                              "data": None})]
    with pytest.raises(BackendError):
        _run(tmp_path, fake)
    run = next(iter(fake.runs.values()))
    assert run["status"] == "failed" and "run is not running" in run["error"]


def test_a_dry_run_pulls_and_decides_but_writes_nothing_to_the_backend(tmp_path):
    fake = _night(tmp_path)
    out = _run(tmp_path, fake, dry_run=True)
    assert set(fake.paths()) == {"worker/sightings/pull", "worker/references/pull"}
    assert out["counts"]["pushed_promote"] == 1 and not fake.runs
    assert list((tmp_path / "out").glob("labels_*.csv"))


def test_pushed_rows_are_json_safe():
    from vbb_reid_server.decide import MachineLabel
    l = MachineLabel("s1", "known", "promote", "H1", "pA", "face", np.float32(0.71), float("nan"),
                     np.float64(0.71), None, {"q": np.float32(0.5), "bad": float("inf"), "n": np.int64(3)})
    row = label_row(l)
    json.dumps(row, allow_nan=False)
    assert row["margin"] is None and row["evidence"] == {"q": 0.5, "bad": None, "n": 3}
    assert "person_key" not in row, "identity keys (possibly alias names) never leave the host"


def test_run_days_parsing():
    from types import SimpleNamespace
    from vbb_reid_server.cli import _run_days
    assert _run_days(SimpleNamespace(day="2026-10-30..2026-11-02", tz="Asia/Shanghai")) == [
        "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02"]
    assert len(_run_days(SimpleNamespace(day="auto", tz="Asia/Shanghai"))) == 1
    with pytest.raises(SystemExit):
        _run_days(SimpleNamespace(day="2026-11-02..2026-10-30", tz="Asia/Shanghai"))
