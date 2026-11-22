"""A reviewer removes a reference picture that doesn't show its person (owner 2026-11-22, plan decision 39): hidden on
every page, never picked again, never learned as that person, and it can be put back. Every picture opens enlarged."""

from __future__ import annotations

import json
import re
import threading
from http.server import ThreadingHTTPServer

import pytest
from test_verify import REVIEW, URL, _req
from test_verify_refs_board import people as ref_people
from test_verify_refs_board import run_cli, world
from test_vetting import A, RNG, _lab, _ws, near
from vbb_reid_server.regroup_night import _human_check
from vbb_reid_server.verify import Store, load_removed, make_handler, removed_path, removed_photos
from vbb_reid_server.vetting import VetPolicy, load_trusted, vet_labels

NIGHT = "3_2026-11-06"


def _photo(sid, confirmed=True, day="2026-11-01"):
    return {"sighting_id": sid, "day": day, "url": URL.format(sid), "confirmed": confirmed}


def _root(tmp_path, facility="3"):
    """One night (p0, the machine says H1) and refs.json: H1 = r1, r2 and a board picture b1; H2 = r3."""
    root = tmp_path / "verify"
    (root / NIGHT).mkdir(parents=True)
    people = [{"person_id": "pA3", "key": "H1", "name": "", "photos": [_photo("r1"), _photo("r2"), _photo("b1", False)]},
              {"person_id": "pB3", "key": "H2", "name": "", "photos": [_photo("r3")]}]
    item = {"sighting_id": "p0", "url": URL.format("p0"), "person_id": "pA3", "via": "face", "face_sim": "0.5",
            "body_sim": ""}
    (root / NIGHT / "frozen.json").write_text(json.dumps({"facility_id": "3", "day": "2026-11-06", "run_id": "a" * 32,
                                                          "pool": 1, "items": [item], "people": people}))
    (root / "refs.json").write_text(json.dumps({"facility_id": facility, "built_at": 5, "people": people}))
    return root


@pytest.fixture
def srv(tmp_path):
    root = _root(tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Store(root, fetch=lambda url: b"JPEG")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", root
    httpd.shutdown()


def _post(base, path, body, headers=REVIEW):
    code, out = _req(base + path, json.dumps(body).encode(), headers, "POST")
    return code, (json.loads(out) if code in (200, 400) else out)


def _data(page: bytes) -> dict:
    """The page's own data, as its script receives it."""
    return json.loads(re.search(r"const D = (\{.*\})(?:, NIGHT = |;)", page.decode()).group(1))


def test_a_removed_reference_is_hidden_on_every_page_and_can_be_put_back(srv, monkeypatch):
    base, root = srv
    monkeypatch.setattr("vbb_reid_server.verify.time.time", lambda: 100)
    assert _post(base, f"/n/{NIGHT}/refs", {"ref": 0, "j": 1, "removed": True, "reviewer": "me"}) == (200, {"removed": [1]})
    assert [p["removed"] for p in _data(_req(f"{base}/n/{NIGHT}")[1])["people"]] == [[1], []]
    batch = _data(_req(f"{base}/fix/p/pA3")[1])
    assert batch["me"]["removed"] == [1] and batch["built_at"] == 5  # the batch page hides it too, and can send one
    assert _data(_req(f"{base}/fix/p/pB3")[1])["people"][0]["removed"] == [1]  # and among the other choices
    assert load_removed(root) == {"r2": {"pA3": {"key": "H1", "facility_id": "3", "day": "2026-11-01", "confirmed": True,
                                                 "at": 100, "reviewer": "me"}}}
    assert "1 reference picture(s) removed" in _req(f"{base}/")[1].decode()
    assert removed_photos({"person_id": "pB3", "photos": [_photo("r2")]}, load_removed(root)) == []  # removed from H1 only
    monkeypatch.setattr("vbb_reid_server.verify.time.time", lambda: 200)
    assert _post(base, f"/n/{NIGHT}/refs", {"ref": 0, "j": 1, "removed": True})[1] == {"removed": [1]}
    assert load_removed(root)["r2"]["pA3"]["at"] == 100  # removed twice: the first time stands
    # from a batch page, by refs.json's numbering: H2's only picture; the night's page hides it as well
    assert _post(base, "/fix/refs", {"ref": 1, "j": 0, "removed": True, "built_at": 5}) == (200, {"removed": [0]})
    assert [p["removed"] for p in _data(_req(f"{base}/n/{NIGHT}")[1])["people"]] == [[1], [0]]
    assert _post(base, f"/n/{NIGHT}/refs", {"ref": 0, "j": 1, "removed": False}) == (200, {"removed": []})
    assert _post(base, "/fix/refs", {"ref": 1, "j": 0, "removed": False, "built_at": 5}) == (200, {"removed": []})
    assert load_removed(root) == {}
    log = [json.loads(x) for x in (root / "refs_removed.log").read_text().splitlines()]
    assert [(x["sighting_id"], x["removed"]) for x in log] == [("r2", True), ("r2", True), ("r3", True), ("r2", False),
                                                               ("r3", False)]


def test_a_removal_names_a_picture_on_its_own_page_or_nothing_is_saved(srv, tmp_path):
    base, root = srv
    ok = {"ref": 0, "j": 0, "removed": True}
    assert _post(base, f"/n/{NIGHT}/refs", ok, {"Content-Type": "application/json"})[0] == 403  # not from the page
    assert _post(base, f"/n/{NIGHT}/refs", ok, REVIEW | {"Host": "evil.example"})[0] == 403
    assert _post(base, "/n/3_2026-11-07/refs", ok)[0] == 404
    for bad in ({"ref": 0, "j": 3, "removed": True}, {"ref": 2, "j": 0, "removed": True},
                {"ref": -1, "j": 0, "removed": True}, {"ref": 0, "j": -1, "removed": True},
                {"ref": "0", "j": 0, "removed": True}, {"ref": True, "j": 0, "removed": True},
                {"ref": 0, "j": 0, "removed": "yes"}, {"ref": 0, "j": 0}, [0, 0]):
        assert _post(base, f"/n/{NIGHT}/refs", bad)[0] == 400, bad
    for stale in ({**ok, "built_at": 4}, ok):  # refs.json was rebuilt since the batch page loaded, or no date at all
        code, out = _post(base, "/fix/refs", stale)
        assert code == 400 and "reload" in out["error"], stale
    assert not removed_path(root).exists()
    blank = _root(tmp_path / "blank", facility="")  # refs.json without a facility: the job could not date the removal
    assert "no facility" in str(pytest.raises(ValueError, Store(blank).fix_set_removed, {**ok, "built_at": 5}).value)
    assert not removed_path(blank).exists()


def test_a_removed_reference_is_never_learned_as_that_person(tmp_path):
    """r1: a confirmed reference of pA, removed, also answered pA on a night. x1: removed from pA, answered pB. x2: removed
    from pA, answered pB on one night and "someone else" on another. b1: a board-named reference of pB, removed."""
    root = tmp_path / "verify"
    photos = [{"sighting_id": s, "day": "2026-10-01"} for s in ("r1", "r2", "x1", "x2")]
    (root / "3_2026-10-20").mkdir(parents=True)
    (root / "3_2026-10-21").mkdir()
    (root / "refs.json").write_text(json.dumps({"facility_id": "3", "built_at": 1_700_000_000, "people": [
        {"key": "pA", "person_id": "pA", "photos": photos},
        {"key": "pB", "person_id": "pB", "photos": [{"sighting_id": "b1", "day": "2026-10-02", "confirmed": False}]}]}))
    for night, items, answers in (("3_2026-10-20", ["x1", "x2", "r1"], {"0": "pB", "1": "pB", "2": "pA"}),
                                  ("3_2026-10-21", ["x2"], {"0": "other"})):
        (root / night / "frozen.json").write_text(json.dumps({
            "facility_id": "3", "day": night[2:], "people": [{"key": "pA", "person_id": "pA"}, {"key": "pB", "person_id": "pB"}],
            "items": [{"sighting_id": s} for s in items]}))
        (root / night / "marks.json").write_text(json.dumps({"answers": answers, "submitted": True,
                                                             "updated_at": 1_700_000_500}))
    row = {"key": "pA", "facility_id": "3", "day": "2026-10-01", "confirmed": True, "reviewer": ""}
    removed_path(root).write_text(json.dumps({"removed": {
        "r1": {"pA": row | {"at": 1_700_000_100}}, "x1": {"pA": row | {"at": 1_700_000_100}},
        "x2": {"pA": row | {"at": 1_700_000_300}},
        "b1": {"pB": row | {"key": "pB", "day": "2026-10-02", "confirmed": False, "at": 1_700_000_200}},
        "junk": "not a row", "junk2": {"pA": 7}}}))
    t = load_trusted(root)
    assert t == {"r1": None, "r2": "pA", "x1": "pB", "x2": None, "b1": None}
    assert t.info["r1"] == ("3", "2026-10-01", 1_700_000_100_000)  # dated by the removal: the run knows it from then
    assert t.info["x2"] == ("3", "2026-10-01", 1_700_000_300_000) and t.info["b1"][2] == 1_700_000_200_000
    assert t.info["x1"] == ("3", "2026-10-20", 1_700_000_500_000)  # the answer that names who it is
    # a staff label naming pA, with the reviewed mark: dropped for r1, learned for r2
    labs = [_lab("r1", "pA", 100, reviewed=True), _lab("r2", "pA", 100, reviewed=True)]
    ws = _ws(tmp_path, [(s.sighting_id, near(A, 0.05, RNG)) for s in labs], labs)
    st = vet_labels(ws, labs, t, VetPolicy()).status
    assert (st["r1"], st["r2"]) == ("answer_disagrees", "answered")
    ex = ws.export
    check = [_human_check(s.sighting_id, s, ex, ex.roster("3"), t, t.info, 1_800_000_000_000) for s in labs]
    assert check == [("nobody", None), ("person", "pA")]  # the sorter's anchor: not pA, whatever staff said
    removed_path(root).write_text(json.dumps({"removed": {}}))  # all put back
    assert load_trusted(root) == {"r1": "pA", "r2": "pA"}  # x1, x2: two answers again; b1: never an answer


def test_an_unreadable_removal_file_stops_the_run_and_names_the_file(srv):
    """Written atomically, so only a damaged disk truncates it. The nightly job must then fail, not learn the removed
    pictures as their people again; the page refuses removals with the same message."""
    base, root = srv
    removed_path(root).write_text('{"removed": {"r2": ')
    with pytest.raises(ValueError, match="refs_removed.json"):
        load_trusted(root)
    code, out = _post(base, f"/n/{NIGHT}/refs", {"ref": 0, "j": 0, "removed": True})
    assert code == 400 and "refs_removed.json" in out["error"]
    assert removed_path(root).read_text() == '{"removed": {"r2": '  # left for a person to look at


def test_a_rebuild_never_picks_a_removed_picture_again(tmp_path, monkeypatch):
    fake = world(tmp_path)  # pA: nine confirmed pictures r0..r8; pB: none, so the board's best faces
    removed_path(tmp_path).write_text(json.dumps({"removed": {  # beside the refs.json run_cli writes
        "r0": {"pA": {"at": 1}}, "s16a": {"pB": {"at": 1}}}}))
    refs = run_cli(tmp_path, monkeypatch, fake, "--per-person", "12")
    a, b = ref_people(refs)["pA"], ref_people(refs)["pB"]
    assert len(a) == 8 and "r0" not in {p["sighting_id"] for p in a} and refs["dropped"] == {"excluded": 1}
    assert [p["sighting_id"] for p in b] == ["s14a", "s15a", "s14b"] and refs["board"]["dropped"] == {"excluded": 1}


def test_every_picture_opens_enlarged_and_the_shared_script_is_declared_once(srv):
    base, root = srv
    assert _post(base, f"/n/{NIGHT}/marks", {"answers": {"0": "H2"}, "submitted": True})[0] == 200  # one wrong name
    index = _req(f"{base}/")[1].decode()
    assert 'id="zoom"' in index and "<img class=zoomable" in index
    for path in (f"/n/{NIGHT}", "/fix/p/pA3"):
        page = _req(base + path)[1].decode()
        assert 'id="zoom"' in page, path
        for decl in ("const NOTE ", "const Z ", "function zoom(", "function refRow(", "function zoomRef("):
            assert page.count(decl) == 1, (path, decl)  # one global scope: a second const stops the page's script
    assert "zbtn" in _req(f"{base}/fix/p/pA3")[1].decode()  # a batch tile's own enlarge button
