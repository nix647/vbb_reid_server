"""``verify-refs --from-backend`` (docs/DESIGN_nightly_regroup.md, W7): the reference pictures of the check page
picked from the backend's own staff labels, for a facility with no offline export (DHW)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from helpers import basis, near, sighting, write_face_cache
from test_v020 import ROSTER, TOKEN, FakeBackend, _probe, _ref, client_for, ms
from vbb_reid_server.cli import main
from vbb_reid_server.data import Export, Person
from vbb_reid_server.verify import IMAGE_PREFIX, pick_references, reviewed_staff_answers

DAYS = ["2026-10-16", "2026-10-17", "2026-10-18"]


def labelled(sid, day, pid, *, reviewed=True, state="known", url=True):
    s = sighting(sid, day=day, human_state=state, human_pid=pid)
    picture = (IMAGE_PREFIX + f"x/{sid}.jpg") if url else "https://cdn.example/x.jpg"
    return replace(s, human_reviewed=reviewed, image_url=picture)


def export(*extra):
    ss = [labelled(f"a{i}", DAYS[i % 3], "pA") for i in range(9)]
    ss += list(extra)
    people = {p: Person(p, "3") for p in ("pA", "pB")}
    people["pZ"] = Person("pZ", "2")  # another facility: not on this roster
    return Export(root=None, sightings=ss, people=people, facility_tz={"3": "Asia/Shanghai"}, facility_org={"3": "b"})


def pick(ex, **kw):
    return pick_references(ex, "3", [], one_face=kw.pop("one_face", lambda s: True), exclude=set(),
                           answers=reviewed_staff_answers(ex, "3"), **kw)


def photos(refs, pid="pA"):
    return next(p for p in refs["people"] if p["person_id"] == pid)["photos"]


def test_only_reviewed_known_labels_are_answers():
    ex = export(labelled("unreviewed", DAYS[0], "pB", reviewed=False),
                labelled("stranger", DAYS[0], None, state="stranger"),
                replace(labelled("other_fac", DAYS[0], "pA"), facility_id="2"))
    answers = reviewed_staff_answers(ex, "3")
    assert set(answers) == {f"a{i}" for i in range(9)} and set(answers.values()) == {"pA"}


def test_reviewed_labels_become_references_one_picture_per_day_in_turn_and_names_stay_empty():
    refs = pick(export(), per_person=6)
    got = photos(refs)
    assert len(got) == 6 and [p["day"] for p in got[:3]] == sorted(DAYS)  # a round of one per day first
    assert all(p["url"].startswith(IMAGE_PREFIX) for p in got)
    assert [p["name"] for p in refs["people"]] == ["", ""] and photos(refs, "pB") == []


def test_unreviewed_stranger_and_off_roster_labels_do_not_become_references():
    ex = export(labelled("unreviewed", DAYS[0], "pB", reviewed=False),
                labelled("stranger", DAYS[0], None, state="stranger"),
                labelled("offroster", DAYS[1], "pZ"), labelled("nopicture", DAYS[1], "pB", url=False))
    refs = pick(ex, per_person=20)
    assert photos(refs, "pB") == [] and {p["sighting_id"] for p in photos(refs)} == {f"a{i}" for i in range(9)}
    assert refs["dropped"] == {"no uploaded picture": 1, "not a roster person": 1}


def test_the_one_face_check_still_applies():
    refs = pick(export(), one_face=lambda s: s.sighting_id != "a0", per_person=20)
    assert "a0" not in {p["sighting_id"] for p in photos(refs)} and refs["dropped"] == {"not exactly one face": 1}


def test_answer_files_and_given_answers_count_together(tmp_path):
    ex = export()
    f = tmp_path / "marks.json"
    f.write_text(json.dumps({"answers": {"a0": "pB"}}))  # a file disagrees with the reviewed label for a0
    refs = pick_references(ex, "3", [f], one_face=lambda s: True, exclude=set(), per_person=20,
                           answers=reviewed_staff_answers(ex, "3"))
    assert "a0" not in {p["sighting_id"] for p in photos(refs)} and refs["dropped"] == {"answers disagree": 1}


# the command -------------------------------------------------------------------------------------------------------


def _backend(tmp_path):
    rng = np.random.default_rng(2)
    refs = [_ref(f"r{i}", "pA", ms(2026, 10, 16 + i % 3, 9) + i, day=DAYS[i % 3], reviewed=True) for i in range(9)]
    refs += [_ref(f"u{i}", "pB", ms(2026, 10, 17, 9) + i, day=DAYS[1], reviewed=False) for i in range(3)]  # unreviewed
    for r in refs:
        r["image_url"] = IMAGE_PREFIX + f"x/{r['sighting_id']}.jpg"
    for d in DAYS:
        write_face_cache(tmp_path / "emb", "3", d, [(r["sighting_id"], 0, near(basis(0), 0.05, rng), 0.9, 80.0)
                                                    for r in refs if r["day"] == d])
    return FakeBackend([_probe("q1", ms(2026, 10, 20, 9))], refs, ROSTER)


def _args(tmp_path, *more):
    tok = tmp_path / "tok"
    tok.write_text(TOKEN + "\n")
    return ["verify-refs", "--facility", "3", "--emb", str(tmp_path / "emb"), "--face-model", "test-face",
            "--out", str(tmp_path / "refs.json"), *more], str(tok)


@pytest.fixture
def patched(tmp_path, monkeypatch):
    import vbb_reid_server.api as api
    fake = _backend(tmp_path)
    monkeypatch.setattr(api, "BackendClient", lambda url, token, **kw: client_for(fake, token=token))
    return fake


def test_from_backend_builds_refs_json_without_answers_or_an_export(tmp_path, patched, capsys):
    argv, tok = _args(tmp_path, "--from-backend", "--backend", "https://b.example/api", "--day", "2026-10-20",
                      "--per-person", "4")
    main([*argv, "--token-file", tok])
    refs = json.loads((tmp_path / "refs.json").read_text())
    a, b = refs["people"]
    assert (a["person_id"], b["person_id"]) == ("pA", "pB") and b["photos"] == []  # pB's are unreviewed
    assert len(a["photos"]) == 4 and {p["day"] for p in a["photos"]} == set(DAYS)
    assert a["name"] == "" and refs["facility_id"] == "3" and refs["per_person"] == 4
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err and "written to" in out.out


def test_the_window_reaches_back_from_the_day(tmp_path, patched):
    argv, tok = _args(tmp_path, "--from-backend", "--backend", "https://b.example/api", "--day", "2026-10-20",
                      "--token-file", tok_of(tmp_path))
    main(argv)
    since = next(c[1]["since_ts"] for c in patched.calls if c[0] == "worker/references/pull")
    assert since == ms(2026, 8, 21, 0)  # 60 days before the day, at local midnight (the run's reference window)
    main([*argv, "--window-days", "5", "--out", str(tmp_path / "r5.json")])
    assert [c[1]["since_ts"] for c in patched.calls if c[0] == "worker/references/pull"][-1] == ms(2026, 10, 15, 0)


def tok_of(tmp_path):
    return str(tmp_path / "tok")


def test_neither_flag_is_an_error_and_from_backend_needs_no_answers(tmp_path, patched):
    argv, tok = _args(tmp_path)
    with pytest.raises(SystemExit, match="--from-backend"):
        main(argv)  # neither --from-backend nor --export/--answers
    with pytest.raises(SystemExit, match="--from-backend"):
        main([*argv, "--export", str(tmp_path)])  # an export but no answers
    with pytest.raises(SystemExit, match="--backend and --token-file"):
        main([*argv, "--from-backend"])
    main([*argv, "--from-backend", "--backend", "https://b.example/api", "--token-file", tok, "--day", "2026-10-20"])
    assert (tmp_path / "refs.json").exists()


def test_a_backend_failure_is_reported_without_the_token(tmp_path, patched):
    patched.fail["worker/sightings/pull"] = [(200, {"code": 1002, "message": f"refused {TOKEN}", "action": None,
                                                    "data": None})]
    argv, tok = _args(tmp_path, "--from-backend", "--backend", "https://b.example/api", "--day", "2026-10-20")
    with pytest.raises(SystemExit) as e:
        main([*argv, "--token-file", tok])
    assert TOKEN not in str(e.value) and "BackendError" in str(e.value)


def test_a_range_of_days_is_refused_with_from_backend(tmp_path, patched):
    argv, tok = _args(tmp_path, "--from-backend", "--backend", "https://b.example/api",
                      "--day", "2026-10-19..2026-10-20")
    with pytest.raises(SystemExit, match="one --day"):
        main([*argv, "--token-file", tok])
