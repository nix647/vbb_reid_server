"""Board-named reference pictures for people nobody confirmed (docs/DESIGN_nightly_regroup.md, W8): picked from the
board's own sightings, marked unconfirmed, shown in a colored frame, and never an answer."""

from __future__ import annotations

import json

import numpy as np
import pytest
from helpers import _dir, FACE_ID, basis, near, write_face_cache
from test_v020 import ROSTER, TOKEN, FakeBackend, Refused, _probe, _ref, client_for, ms
from vbb_reid_server.cli import main
from vbb_reid_server.corrections import _check_pages
from vbb_reid_server.verify import (IMAGE_PREFIX, Store, add_board_references, fix_batch_html, fix_page_data,
                                    night_html, page_data, unconfirmed_photos)
from vbb_reid_server.vetting import load_trusted

DAY = "2026-10-20"
CONFIRMED_DAYS = ["2026-10-10", "2026-10-11", "2026-10-12"]
NOTE = "board's name, not confirmed"


def board(sid, day, pid, norm=20.0, **kw):
    row = _probe(sid, ms(2026, 10, int(day[-2:]), 9), board_state="known", board_person_id=pid, _day=day,
                 image_url=IMAGE_PREFIX + f"x/{sid}.jpg")
    row.update(kw)
    return row, norm


def write_cache(emb, day, rows):
    """rows: (sid, norm, n_faces). The standard test cache, with the norm and the face count of each row set."""
    rng = np.random.default_rng(len(rows) + int(day[-2:]))
    write_face_cache(emb, "3", day, [(sid, 0, near(basis(0), 0.05, rng), 0.9, 80.0) for sid, _, _ in rows])
    f = _dir(emb, "face", FACE_ID, "3") / f"{day}.npz"
    z = dict(np.load(f, allow_pickle=True))
    z["norm"] = np.array([n for _, n, _ in rows], dtype=float)
    z["n_faces"] = np.array([k for _, _, k in rows], dtype=float)
    np.savez(f, **z)


def world(tmp_path, extra=()):
    """pA has 9 confirmed (reviewed) pictures; pB has none. The board named pB on 10-14, 10-15 and 10-16 and pA on
    10-15. Face norms: s14a 30, s14b 10, s15a 25, s16a 40."""
    refs = [_ref(f"r{i}", "pA", ms(2026, 10, 10 + i % 3, 9) + i, day=CONFIRMED_DAYS[i % 3], reviewed=True)
            for i in range(9)]
    for r in refs:
        r["image_url"] = IMAGE_PREFIX + f"x/{r['sighting_id']}.jpg"
    for d in CONFIRMED_DAYS:
        write_cache(tmp_path / "emb", d, [(r["sighting_id"], 20.0, 1) for r in refs if r["day"] == d])
    cands = [board("s14a", "2026-10-14", "pB", 30), board("s14b", "2026-10-14", "pB", 10),
             board("s15a", "2026-10-15", "pB", 25), board("s16a", "2026-10-16", "pB", 40),
             board("a15", "2026-10-15", "pA", 35), *extra]
    per_day: dict = {}
    for row, norm in cands:
        per_day.setdefault(row["_day"], []).append((row["sighting_id"], norm, 1))
    for d, rows in per_day.items():
        write_cache(tmp_path / "emb", d, [(sid, n, k) for sid, n, k in rows])
    return FakeBackend([_probe("q1", ms(2026, 10, 20, 9))] + [r for r, _ in cands], refs, ROSTER)


def run_cli(tmp_path, monkeypatch, fake, *more):
    import vbb_reid_server.api as api
    monkeypatch.setattr(api, "BackendClient", lambda url, token, **kw: client_for(fake, token=token))
    tok = tmp_path / "tok"
    tok.write_text(TOKEN + "\n")
    out = tmp_path / "refs.json"
    main(["verify-refs", "--facility", "3", "--emb", str(tmp_path / "emb"), "--face-model", "test-face",
          "--out", str(out), "--from-backend", "--backend", "https://b.example/api", "--token-file", str(tok),
          "--day", DAY, *more])
    return json.loads(out.read_text())


def people(refs):
    return {p["person_id"]: p["photos"] for p in refs["people"]}


# picking -----------------------------------------------------------------------------------------------------------


def test_people_with_confirmed_pictures_get_no_board_pictures(tmp_path, monkeypatch):
    refs = run_cli(tmp_path, monkeypatch, world(tmp_path), "--per-person", "12")
    a = people(refs)["pA"]
    assert len(a) == 9 and all(p["confirmed"] is True for p in a)  # all 9 confirmed; short of 12, but no board pictures
    assert "a15" not in {p["sighting_id"] for p in a}


def test_a_person_with_none_gets_the_boards_best_faces_one_day_in_turn_marked_unconfirmed(tmp_path, monkeypatch):
    fake = world(tmp_path)
    refs = run_cli(tmp_path, monkeypatch, fake, "--per-person", "4")
    b = people(refs)["pB"]
    # the day of the best face first (10-16: 40, 10-14: 30, 10-15: 25), then the next best of 10-14
    assert [p["sighting_id"] for p in b] == ["s16a", "s14a", "s15a", "s14b"]
    assert all(p["confirmed"] is False and p["url"].startswith(IMAGE_PREFIX) for p in b)
    assert [p["sighting_id"] for p in people(run_cli(tmp_path, monkeypatch, fake, "--per-person", "2"))["pB"]] \
        == ["s16a", "s14a"]  # fewer pictures than days: the best ones, not the earliest
    assert refs["board"]["days"][0] == "2026-10-07" and refs["board"]["days"][-1] == DAY  # 14 days up to --day


def test_candidates_that_cannot_serve_as_a_reference_are_left_out(tmp_path, monkeypatch):
    extra = [board("overlay", "2026-10-16", "pB", 99, human_labeled=True, human_state="stranger"),
             board("labelled", "2026-10-16", "pB", 98, human_labeled=True, human_state="known"),
             board("reset", "2026-10-16", "pB", 97, human_labeled=True, human_state="reset"),
             board("offroster", "2026-10-16", "pZ", 96),
             board("noimage", "2026-10-16", "pB", 95, image_url="https://cdn.example/x.jpg"),
             board("uncertain", "2026-10-16", "pB", 94, board_state="uncertain", board_person_id=None)]
    fake = world(tmp_path, extra)
    # two faces in one crop: the cache says so
    write_cache(tmp_path / "emb", "2026-10-16", [("s16a", 40.0, 1), ("overlay", 99.0, 1), ("labelled", 98.0, 1),
                                                 ("reset", 97.0, 1), ("offroster", 96.0, 1), ("noimage", 95.0, 1),
                                                 ("uncertain", 94.0, 1), ("twofaces", 93.0, 2)])
    fake.sightings.append(board("twofaces", "2026-10-16", "pB", 93)[0])
    refs = run_cli(tmp_path, monkeypatch, fake, "--per-person", "20")
    assert [p["sighting_id"] for p in people(refs)["pB"]] == ["s16a", "s14a", "s15a", "s14b"]
    assert refs["board"]["dropped"] == {"has a human overlay": 3, "no uploaded picture": 1, "not a roster person": 1,
                                        "not exactly one face": 1}


def test_a_day_whose_pull_failed_is_skipped_and_said_so_without_the_token(tmp_path, monkeypatch, capsys):
    fake = world(tmp_path)
    real = fake._worker_sightings_pull

    def pull(p):
        if p["day"] == "2026-10-16":
            raise Refused(f"busy {TOKEN}")
        return real(p)
    fake._worker_sightings_pull = pull
    refs = run_cli(tmp_path, monkeypatch, fake, "--per-person", "6")
    assert [p["sighting_id"] for p in people(refs)["pB"]] == ["s14a", "s15a", "s14b"]  # no s16a: that day was skipped
    out = capsys.readouterr()
    assert "skipped 2026-10-16" in out.out and "2026-10-16" not in refs["board"]["days"]
    assert TOKEN not in out.out + out.err


def test_board_days_sets_how_far_back_the_board_is_read(tmp_path, monkeypatch):
    refs = run_cli(tmp_path, monkeypatch, world(tmp_path), "--board-days", "5")  # 10-16 .. 10-20: only s16a is in it
    assert [p["sighting_id"] for p in people(refs)["pB"]] == ["s16a"]


def test_add_board_references_never_reads_the_sorters_labels():
    from vbb_reid_server.data import Export, Person
    ex = Export(root=None, sightings=[], people={"pB": Person("pB", "3")}, facility_tz={}, facility_org={})
    refs = {"people": [{"person_id": "pB", "key": "pB", "name": "", "photos": []}]}
    row = {"sighting_id": "m1", "board_state": "uncertain", "board_person_id": None,
           "image_url": IMAGE_PREFIX + "x/m1.jpg", "machine_person_id": "pB"}  # only the sorter named it
    out = add_board_references(refs, ex, "3", {"2026-10-15": [row]}, one_face=lambda s: True, quality=lambda s, d: 1.0)
    assert out["people"][0]["photos"] == []


# the flag and the guard --------------------------------------------------------------------------------------------


def _refs_json(root, people):
    root.mkdir(parents=True, exist_ok=True)
    (root / "refs.json").write_text(json.dumps({"facility_id": "3", "built_at": 1, "people": people}))


def photo(sid, **kw):
    return {"sighting_id": sid, "day": "2026-10-18", "url": IMAGE_PREFIX + f"x/{sid}.jpg", **kw}


def test_an_old_refs_json_without_the_flag_still_loads_as_confirmed(tmp_path):
    _refs_json(tmp_path, [{"person_id": "pA", "key": "pA", "name": "", "photos": [photo("old1"), photo("old2")]}])
    assert load_trusted(tmp_path) == {"old1": "pA", "old2": "pA"}
    mixed = {"photos": [photo("old1"), photo("c", confirmed=True), photo("u", confirmed=False)]}
    assert unconfirmed_photos(mixed) == [2]


def test_load_trusted_never_reads_an_unconfirmed_picture_but_a_confirmed_copy_elsewhere_counts(tmp_path):
    _refs_json(tmp_path, [
        {"person_id": "pB", "key": "pB", "name": "", "photos": [photo("only", confirmed=False),
                                                                 photo("both", confirmed=False)]},
        {"person_id": "pA", "key": "pA", "name": "", "photos": [photo("both", confirmed=True),
                                                                 photo("mine", confirmed=True)]}])
    t = load_trusted(tmp_path)
    assert t == {"both": "pA", "mine": "pA"} and "only" not in t and "only" not in t.info


def test_the_correction_list_does_not_count_an_unconfirmed_picture_as_checked(tmp_path):
    _refs_json(tmp_path, [{"person_id": "pB", "key": "pB", "name": "", "photos": [
        photo("board", confirmed=False), photo("human", confirmed=True), photo("legacy")]}])
    answered, _ = _check_pages(tmp_path)
    assert set(answered) == {"human", "legacy"}


# the pages ---------------------------------------------------------------------------------------------------------


def _frozen():
    return {"day": DAY, "items": [{"sighting_id": "x", "url": IMAGE_PREFIX + "x"}],
            "people": [{"person_id": "pA", "key": "pA", "name": "",
                        "photos": [photo("c1", confirmed=True), photo("l")]},
                       {"person_id": "pB", "key": "pB", "name": "",
                        "photos": [photo("b1", confirmed=False), photo("b2", confirmed=False)]}]}


def test_the_check_page_marks_the_boards_pictures():
    data = page_data(_frozen())
    assert [p["unconf"] for p in data["people"]] == [[], [0, 1]]
    html = night_html("3_" + DAY, data)
    assert NOTE in html and 'im.className = "unconf"' in html and "unconf-note" in html


def test_the_frame_is_colored_in_the_light_and_the_dark_theme():
    html = night_html("3_" + DAY, page_data(_frozen()))
    assert "img.unconf{outline:3px solid var(--warn)" in html
    assert "--warn:#b45309" in html and "--warn:#fbbf24" in html  # one color for each theme, away from --accent


def test_the_batch_view_of_the_correction_list_marks_them_too(tmp_path):
    _refs_json(tmp_path, [
        {"person_id": "pB", "key": "pB", "name": "",
         "photos": [photo("b1", confirmed=False), photo("c1", confirmed=True)]},
        {"person_id": "pA", "key": "pA", "name": "", "photos": [photo("a1", confirmed=False)]}])
    data = fix_page_data(Store(tmp_path, fetch=lambda url: b""), "pB")
    assert data["me"]["unconf"] == [0] and [p["unconf"] for p in data["people"]] == [[0]]
    html = fix_batch_html(data)
    assert NOTE in html and "(D.me.unconf || []).includes(j)" in html and "(c.p.unconf || []).includes(j)" in html
    assert html.count('im.className = "unconf"') == 2  # the references and the choice thumbnails
    assert "--warn:#b45309" in html and "--warn:#fbbf24" in html
