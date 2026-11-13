"""The correction list (owner 2026-11-13, plan decision 35): one list that grows night by night.

A picture joins when the machine disagrees with any name nobody confirmed (the board's, or a staff name nobody
looked at). It leaves once a person confirms it: any answer on the check page (can't tell included), or the
reviewed mark. How reviewers work through it is a follow-up; this is the list itself."""

from __future__ import annotations

import csv
import json

from vbb_reid_server.corrections import list_path, open_items, update_list
from vbb_reid_server.shadow import LABEL_FIELDS, STAFF_FIELDS
from vbb_reid_server.verify import IMAGE_PREFIX

FAC, DAY, RUN = "3", "2026-11-12", "a" * 32


def _row(sid, relation="promote", machine="pA", *, board_state="stranger", board=None, human_state="",
         human=None, reviewed=False, url=None, state="known"):
    return {k: "" for k in LABEL_FIELDS} | {
        "run_id": RUN, "sighting_id": sid, "facility_id": FAC, "day": DAY, "state": state, "relation": relation,
        "person_id": machine or "", "via": "face", "board_state": board_state, "board_person_id": board or "",
        "human_labeled": int(bool(human_state)), "human_state": human_state, "human_person_id": human or "",
        "human_reviewed": int(reviewed), "image_url": url if url is not None else IMAGE_PREFIX + f"x/{sid}.jpg"}


def _run(runs, rows, *, day=DAY, run_id=RUN, finished=100, staff=None):
    runs.mkdir(parents=True, exist_ok=True)
    tag = f"{FAC}_{day}_{run_id[:8]}"
    with (runs / f"labels_{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r | {"day": day, "run_id": run_id})
    (runs / f"run_{tag}.json").write_text(json.dumps({"run_id": run_id, "dry_run": False, "finished_at": finished}))
    if staff is not None:
        with (runs / f"staff_{tag}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=STAFF_FIELDS)
            w.writeheader()
            for sid, (state, pid, reviewed) in staff.items():
                w.writerow({"sighting_id": sid, "facility_id": FAC, "day": DAY, "human_state": state,
                            "human_person_id": pid or "", "human_reviewed": int(reviewed)})


def _night(verify, items, *, day=DAY, answers=None, submitted=False, run_id=RUN):
    d = verify / f"{FAC}_{day}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "frozen.json").write_text(json.dumps({
        "facility_id": FAC, "day": day, "run_id": run_id, "items": [{"kind": "sample", "sighting_id": s} for s in items],
        "people": [{"key": "pA", "person_id": "pA"}, {"key": "pB", "person_id": "pB"}]}))
    if answers is not None or submitted:
        (d / "marks.json").write_text(json.dumps({"answers": answers or {}, "submitted": submitted}))


def _items(verify):
    return json.loads(list_path(verify).read_text())["items"]


def test_a_picture_joins_when_the_machine_disagrees_with_a_name_nobody_confirmed(tmp_path):
    runs, verify = tmp_path / "runs", tmp_path / "verify"
    _run(runs, [
        _row("board"),                                                      # camera: stranger; machine: pA
        _row("agree", "agree", board_state="known", board="pA"),            # camera pA, machine pA
        _row("staff", "agree", board_state="known", board="pA", human_state="known", human="pB"),  # staff pB, unchecked
        _row("confirm", human_state="known", human="pA"),                   # camera stranger, staff pA unchecked, machine pA
        _row("looked", human_state="known", human="pB", reviewed=True),     # a person looked: confirmed
        _row("stranger", "agree", board_state="known", board="pA", human_state="stranger"),  # staff: stranger, unchecked
        _row("unsure", "agree", board_state="known", board="pA", human_state="reset"),       # "can't tell" is no name
        _row("relabel", "relabel", board_state="known", board="pB", state="flag"),  # camera pB, machine pA
        _row("mixed", "mixed", machine=None, state="flag"),                 # the machine names nobody
        _row("conflict", "conflict", state="flag", human_state="known", human="pB"),  # face and body disagree:
                                                                            # the machine doesn't name one person
        _row("nopic", url=""),                                              # no uploaded picture to show
    ])
    out = update_list(runs, verify, FAC, DAY)
    items = _items(verify)
    assert sorted(items) == ["board", "confirm", "relabel", "staff", "stranger"]
    assert items["board"]["against"] == ["board"] and items["staff"]["against"] == ["staff"]
    assert items["confirm"]["against"] == ["board"]  # nothing to correct, but nobody confirmed the staff name yet
    assert (items["relabel"]["machine"], items["relabel"]["board"]) == ("pA", "pB")
    assert items["stranger"]["staff"] == "stranger" and items["board"]["staff"] is None
    assert {i["status"] for i in items.values()} == {"open"} and out["added"] == 5 and out["open"] == 5
    assert [i["sighting_id"] for i in open_items(verify)] == sorted(items)


def test_pictures_already_answered_on_the_check_page_never_join(tmp_path):
    runs, verify = tmp_path / "runs", tmp_path / "verify"
    _night(verify, ["s_named", "s_cant", "s_left"], answers={"0": "pA", "1": "cant_tell"}, submitted=True)
    (verify / "refs.json").write_text(json.dumps({"people": [{"key": "pA", "person_id": "pA",
                                                              "photos": [{"sighting_id": "r_ref"}]}]}))
    _run(runs, [_row(s) for s in ("s_named", "s_cant", "s_left", "r_ref", "new")])
    update_list(runs, verify, FAC, DAY)
    assert sorted(_items(verify)) == ["new", "s_left"]  # s_left: the night is finished and it was never answered


def test_a_picture_on_an_open_check_page_waits_until_that_night_is_finished(tmp_path):
    runs, verify = tmp_path / "runs", tmp_path / "verify"
    (verify).mkdir()
    (verify / "refs.json").write_text(json.dumps({"people": []}))
    _night(verify, ["s_seen", "s_skip"])
    _run(runs, [_row("s_seen"), _row("s_skip"), _row("new")])
    out = update_list(runs, verify, FAC, DAY)
    assert sorted(_items(verify)) == ["new"] and out["held"] == 2  # never on both lists before it's answered blind

    (verify / f"{FAC}_{DAY}" / "marks.json").write_text(json.dumps({"answers": {"0": "pB"}, "submitted": True}))
    out = update_list(runs, verify, FAC, DAY)
    assert sorted(_items(verify)) == ["new", "s_skip"] and out["released"] == 1  # s_seen was answered: confirmed


def test_an_item_leaves_once_a_person_confirms_it(tmp_path):
    runs, verify = tmp_path / "runs", tmp_path / "verify"
    _run(runs, [_row(s) for s in ("ans", "cant", "look", "keep")]
         + [_row("fixed", "agree", board_state="known", board="pA", human_state="known", human="pB")])
    update_list(runs, verify, FAC, DAY)
    assert len(_items(verify)) == 5

    # a later night: answers on the check page, and the backend's current labels (that run's staff snapshot)
    _night(verify, ["ans", "cant"], day="2026-11-13", answers={"0": "pB", "1": "cant_tell"}, submitted=True,
           run_id="b" * 32)
    _run(runs, [], day="2026-11-13", run_id="b" * 32, finished=200,
         staff={"look": ("known", "pA", True), "fixed": ("known", "pA", False), "keep": ("stranger", None, False)})
    out = update_list(runs, verify, FAC, "2026-11-13")
    items = _items(verify)
    reasons = {sid: (i["status"], i.get("closed_reason")) for sid, i in items.items()}
    assert reasons == {"ans": ("closed", "answered"), "cant": ("closed", "cant_tell"), "look": ("closed", "reviewed"),
                       "fixed": ("closed", "agrees_now"),  # staff renamed it to the camera's and machine's pA
                       "keep": ("open", None)}             # a stranger label nobody looked at: still disagrees
    assert items["keep"]["staff"] == "stranger" and out["open"] == 1
    assert out["closed"] == {"agrees_now": 1, "answered": 1, "cant_tell": 1, "reviewed": 1}


def test_running_a_day_again_adds_nothing_twice_and_never_reopens(tmp_path):
    runs, verify = tmp_path / "runs", tmp_path / "verify"
    _run(runs, [_row("a"), _row("b")])
    update_list(runs, verify, FAC, DAY, now=1)
    _night(verify, ["b"], day="2026-11-13", answers={"0": "pA"}, submitted=True, run_id="b" * 32)
    _run(runs, [], day="2026-11-13", run_id="b" * 32, finished=200)
    update_list(runs, verify, FAC, "2026-11-13", now=2)
    first = _items(verify)
    out = update_list(runs, verify, FAC, DAY, now=3)  # the same night again
    assert _items(verify) == first and out["added"] == 0 and out["closed"] == {}  # b stays closed as it was
    assert (first["b"]["status"], first["b"]["closed_at"]) == ("closed", 2)


def test_the_list_waits_for_the_nights_check_page_and_uses_its_run(tmp_path):
    runs, verify = tmp_path / "runs", tmp_path / "verify"
    verify.mkdir()
    (verify / "refs.json").write_text(json.dumps({"people": []}))
    _run(runs, [_row("a")], run_id="a" * 32, finished=100)
    _run(runs, [_row("b")], run_id="c" * 32, finished=300)  # a newer run the check page didn't use
    assert update_list(runs, verify, FAC, DAY) is None and not list_path(verify).exists()

    _night(verify, [], run_id="a" * 32)
    update_list(runs, verify, FAC, DAY)
    assert sorted(_items(verify)) == ["a"]  # both lists come from the same run
