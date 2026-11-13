"""Nightly verification page (0.2.3): references, the nightly draw, the server's guards, scoring."""

from __future__ import annotations

import csv
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import FACE_ID, sighting, write_export
from vbb_reid_server.data import Export, Person, load_export
from vbb_reid_server.decide import Thresholds
from vbb_reid_server.pipeline import Workspace
from vbb_reid_server.shadow import LABEL_FIELDS, shadow_run
from vbb_reid_server.verify import (IMAGE_PREFIX, Store, build_night, index_html, make_handler, night_html,
                                    page_data, pick_references, score_night)

URL = IMAGE_PREFIX + "x/{}.jpg"


# --- references ---------------------------------------------------------------------------

def _export():
    days = ["2026-10-20", "2026-10-21"]
    ss = []
    for i in range(8):
        s = sighting(f"a{i}", day=days[i % 2])
        ss.append(replace(s, image_url=URL.format(s.sighting_id)))
    ss.append(replace(sighting("lab_ok", day=days[0], human_state="known", human_pid="pA3"), image_url=URL.format("lab_ok")))
    ss.append(replace(sighting("lab_bad", day=days[0], human_state="known", human_pid="pB3"), image_url=URL.format("lab_bad")))
    ss.append(replace(sighting("lab_str", day=days[0], human_state="stranger"), image_url=URL.format("lab_str")))
    ss.append(sighting("cdn_other", day=days[0]))  # https://cdn.example/...: not an uploaded picture
    ss.append(replace(sighting("two", day=days[0]), image_url=URL.format("two")))
    ss.append(replace(sighting("b0", day=days[1]), image_url=URL.format("b0")))
    people = {p: Person(p, "3") for p in ("pA3", "pB3", "pC3")}
    return Export(root=None, sightings=ss, people=people, facility_tz={"3": "Asia/Shanghai"},
                  facility_org={"3": "brand"}, aliases={"pA3": "H1", "pB3": "H2", "pC3": "H3"},
                  names={"H1": "Name One"})


def _answers(tmp_path, rows, name="a.csv"):
    p = tmp_path / name
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sighting_id", "answer", "note"])
        w.writerows(rows)
    return p


def test_references_keep_only_safe_one_at_a_time_answers(tmp_path):
    ex = _export()
    f1 = _answers(tmp_path, [[f"a{i}", "H1", ""] for i in range(8)] + [
        ["lab_ok", "H1", ""], ["lab_bad", "H1", ""], ["lab_str", "H1", ""], ["cdn_other", "H1", ""],
        ["two", "H1", ""], ["b0", "H2", ""], ["a7", "H1", ""], ["zz", "H9", ""]])
    f2 = tmp_path / "marks.json"
    f2.write_text(json.dumps({"answers": {"a7": "H2", "b0": "H2"}}))  # a7 now has two answers
    refs = pick_references(ex, "3", [f1, f2], one_face=lambda s: s.sighting_id != "two",
                           exclude={"a6"}, per_person=5)
    by = {p["key"]: p for p in refs["people"]}
    h1 = [x["sighting_id"] for x in by["H1"]["photos"]]
    assert set(h1) <= {"a0", "a1", "a2", "a3", "a4", "a5", "lab_ok"} and len(h1) == 5
    assert {x["day"] for x in by["H1"]["photos"]} == {"2026-10-20", "2026-10-21"}  # spread over days
    assert [x["sighting_id"] for x in by["H2"]["photos"]] == ["b0"]
    assert by["H3"]["photos"] == [] and by["H1"]["person_id"] == "pA3" and by["H1"]["name"] == "Name One"
    assert refs["dropped"] == {"answers disagree": 1, "disagrees with its human label": 2, "excluded": 1,
                               "no uploaded picture": 1, "not a roster person": 1, "not exactly one face": 1}


# --- nightly draw -------------------------------------------------------------------------

def _run_files(runs: Path, run_id: str, dry: bool | None, rows: list[dict], finished: float = 1.0):
    runs.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "finished_at": finished}
    if dry is not None:
        meta["dry_run"] = dry
    (runs / f"run_3_2026-11-06_{run_id[:8]}.json").write_text(json.dumps(meta))
    with (runs / f"labels_3_2026-11-06_{run_id[:8]}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LABEL_FIELDS})


def _row(sid, state="known", relation="promote", via="face", human=0, pid="pA3"):
    return {"sighting_id": sid, "state": state, "relation": relation, "via": via, "human_labeled": human,
            "person_id": pid, "face_sim": "0.5", "body_sim": "", "image_url": URL.format(sid)}


def _refs(tmp_path):
    p = tmp_path / "refs.json"
    p.write_text(json.dumps({"built_at": 1, "people": [
        {"person_id": "pA3", "key": "H1", "name": "", "photos": [{"sighting_id": "r1", "day": "d", "url": URL.format("r1")}]},
        {"person_id": "pB3", "key": "H2", "name": "", "photos": []}]}))
    return p


def test_the_draw_takes_only_pushed_new_names_from_the_real_run_and_never_redraws(tmp_path):
    runs = tmp_path / "runs"
    real = [_row(f"p{i}") for i in range(30)] + [
        _row("agree", relation="agree"), _row("flag", state="flag", relation="relabel"),
        _row("human", human=1), _row("body1", via="body")]
    _run_files(runs, "a" * 32, False, real, finished=2.0)
    _run_files(runs, "b" * 32, True, [_row(f"dry{i}") for i in range(30)], finished=3.0)  # newer, but dry
    _run_files(runs, "c" * 32, None, [_row(f"old{i}") for i in range(30)], finished=4.0)  # no marker: skipped
    out = tmp_path / "verify"
    f = build_night(runs, _refs(tmp_path), out, "3", "2026-11-06", n=20, log=lambda *_: None)
    data = json.loads(f.read_text())
    sids = [it["sighting_id"] for it in data["items"]]
    assert data["run_id"] == "a" * 32 and data["pool"] == 31 and len(sids) == 20 == len(set(sids))
    assert set(sids) <= {f"p{i}" for i in range(30)} | {"body1"}
    assert data["pool_by_via"] == {"body": 1, "face": 30}
    assert not os.access(f, os.W_OK)
    before = f.read_bytes()
    build_night(runs, _refs(tmp_path), out, "3", "2026-11-06", n=5, log=lambda *_: None)
    assert f.read_bytes() == before  # drawn once; a second call changes nothing
    again = build_night(runs, _refs(tmp_path), tmp_path / "v2", "3", "2026-11-06", n=20, log=lambda *_: None)
    assert [it["sighting_id"] for it in json.loads(again.read_text())["items"]] == sids  # seeded by the run id


def test_no_real_run_means_no_draw(tmp_path):
    runs = tmp_path / "runs"
    _run_files(runs, "b" * 32, True, [_row("x")])
    assert build_night(runs, _refs(tmp_path), tmp_path / "v", "3", "2026-11-06", log=lambda *_: None) is None


def test_production_run_files_carry_the_picture_and_the_dry_run_marker(tmp_path):
    exp = tmp_path / "export"
    write_export(exp, [{"sighting_id": "s1", "facility_id": "3", "capture_ts": 1_792_000_000_000}],
                 [("pA", "3", "visitor")], facilities=(("3", "2"),))
    ws = Workspace(load_export(exp), tmp_path / "emb", FACE_ID, None)
    run = shadow_run(ws, "3", "2026-10-20", Thresholds(min_refs=1, body_enabled=False), tmp_path / "out",
                     extra={"dry_run": False}, log=lambda *_: None)
    assert run["dry_run"] is False
    assert json.loads(next((tmp_path / "out").glob("run_*.json")).read_text())["dry_run"] is False
    with next((tmp_path / "out").glob("labels_*.csv")).open(encoding="utf-8") as fh:
        assert next(csv.reader(fh))[-4:] == ["image_url", "human_state", "human_person_id", "label_check"]


# --- page and server ----------------------------------------------------------------------

def _night(tmp_path):
    runs = tmp_path / "runs"
    _run_files(runs, "a" * 32, False, [_row("p0", pid="pA3"), _row("p1", pid="pB3", via="body"), _row("p2", pid="pA3")])
    root = tmp_path / "verify"
    build_night(runs, _refs(tmp_path), root, "3", "2026-11-06", n=3, log=lambda *_: None)
    return root, "3_2026-11-06"


def test_the_reviewers_page_carries_no_machine_output_and_no_links(tmp_path):
    root, night = _night(tmp_path)
    fz = Store(root).frozen(night)
    page = night_html(night, page_data(fz))
    for leak in ("pA3", "pB3", "promote", '"via"', "body_sim", "face_sim", "0.5", IMAGE_PREFIX, "p0", "p1"):
        assert leak not in page, leak


@pytest.fixture
def server(tmp_path):
    root, night = _night(tmp_path)
    fetched = []
    store = Store(root, fetch=lambda url: fetched.append(url) or b"JPEG")
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", night, root, fetched
    httpd.shutdown()


def _req(url, data=None, headers=None, method=None):
    r = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(r, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(base, night, body, headers=None):
    h = {"Content-Type": "application/json", "X-Review": "1"} if headers is None else headers
    return _req(f"{base}/n/{night}/marks", json.dumps(body).encode(), h, "POST")


def test_server_saves_only_valid_answers_from_its_own_page(server):
    base, night, root, _ = server
    assert _post(base, night, {"answers": {"0": "H1"}}, headers={"Content-Type": "application/json"})[0] == 403
    assert _post(base, night, {"answers": {"0": "H9"}})[0] == 400
    assert _post(base, night, {"answers": {"7": "H1"}})[0] == 400
    assert _post(base, night, {"answers": {"0": "H1"}}, headers={"Content-Type": "application/json", "X-Review": "1",
                                                                  "Host": "evil.example"})[0] == 403
    assert not (root / night / "marks.json").exists()
    code, body = _post(base, night, {"answers": {"0": "H1", "1": "other", "2": "cant_tell"}, "seen": [0, 1, 2, 9],
                                     "reviewer": "me"})
    assert code == 200
    saved = json.loads((root / night / "marks.json").read_text())
    assert saved["answers"] == {"0": "H1", "1": "other", "2": "cant_tell"} and saved["seen"] == [0, 1, 2]
    assert len((root / night / "marks.log").read_text().splitlines()) == 1


def test_pictures_come_only_from_the_frozen_links(server):
    base, night, root, fetched = server
    assert _req(f"{base}/img/{night}/0") == (200, b"JPEG")
    assert _req(f"{base}/ref/{night}/0/0") == (200, b"JPEG")
    assert _req(f"{base}/img/{night}/3")[0] == 404
    assert _req(f"{base}/ref/{night}/1/0")[0] == 404
    assert _req(f"{base}/img/..%2F..%2Fetc/0")[0] == 404
    assert all(u.startswith(IMAGE_PREFIX) for u in fetched) and len(fetched) == 2


def test_results_show_only_after_done(server):
    base, night, root, _ = server
    _post(base, night, {"answers": {"0": "H1", "1": "H1", "2": "cant_tell"}})
    assert "shown when done" in _req(f"{base}/")[1].decode()
    _post(base, night, {"answers": {"0": "H1", "1": "H1", "2": "cant_tell"}, "submitted": True})
    page = _req(f"{base}/")[1].decode()
    assert "1 right" in page and "1 wrong" in page and "1 can't tell" in page


def test_scoring():
    fz = {"people": [{"key": "H1", "person_id": "pA3"}, {"key": "H2", "person_id": "pB3"}],
          "items": [{"person_id": "pA3", "via": "face", "face_sim": "", "body_sim": ""},
                    {"person_id": "pA3", "via": "body", "face_sim": "", "body_sim": ""},
                    {"person_id": "pB3", "via": "body", "face_sim": "", "body_sim": ""},
                    {"person_id": "pB3", "via": "face", "face_sim": "", "body_sim": ""}]}
    sc = score_night(fz, {"answers": {"0": "H1", "1": "other", "2": "cant_tell"}})
    assert (sc["right"], sc["wrong"], sc["cant_tell"], sc["unanswered"]) == (1, 1, 1, 1)
    assert sc["per_via"] == {"face": [1, 0], "body": [0, 1]} and sc["wrong_items"][0]["i"] == 1


def test_a_link_off_the_picture_site_is_never_fetched(tmp_path):
    root, night = _night(tmp_path)
    f = root / night / "frozen.json"
    data = json.loads(f.read_text())
    data["people"][0]["photos"][0]["url"] = "http://169.254.169.254/latest/meta-data"  # a hand-edited refs.json
    os.chmod(f, 0o644)
    f.write_text(json.dumps(data))
    fetched = []
    assert Store(root, fetch=lambda u: fetched.append(u) or b"x").image(night, "ref", 0, 0) is None
    assert fetched == []


def test_pages_refuse_a_foreign_host_name(server):
    base, night, _, fetched = server  # a DNS-rebinding page would arrive with its own Host
    for path in ("/", f"/n/{night}", f"/n/{night}/marks", f"/img/{night}/0"):
        assert _req(base + path, headers={"Host": "evil.example"})[0] == 403, path
    assert fetched == []


def test_people_without_reference_pictures_are_not_offered(tmp_path):
    root, night = _night(tmp_path)  # H2 has no reference pictures in _refs
    data = page_data(Store(root).frozen(night))
    assert [p["key"] for p in data["people"]] == ["H1"]


def test_a_hidden_person_does_not_shift_the_others_reference_pictures(server):
    base, night, root, fetched = server
    f = root / night / "frozen.json"
    data = json.loads(f.read_text())
    data["people"].insert(0, {"person_id": "p0", "key": "H0", "name": "", "photos": []})  # empty, first
    os.chmod(f, 0o644)
    f.write_text(json.dumps(data))
    pd = page_data(Store(root).frozen(night))
    assert [(p["key"], p["ref"]) for p in pd["people"]] == [("H1", 1)]
    assert _req(f"{base}/ref/{night}/1/0") == (200, b"JPEG") and fetched == [URL.format("r1")]
