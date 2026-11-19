"""The correction list (owner 2026-11-13, plan decision 35): one list that grows night by night.

- **Who joins:** a picture whose machine name disagrees with any name nobody confirmed. That is the board's
  name (``relation`` promote or relabel), or a staff name nobody looked at (a ``known`` label of someone else, or
  a ``stranger`` label). Example: the camera said H2, staff renamed it H1 without looking, the machine says H2.
  The machine must name one person (no ``mixed`` / ``conflict``), and the picture must have an uploaded image.
- **Who never joins:** a picture a person already confirmed: answered on the check page (``refs.json`` or any
  answer on a finished night, can't tell included: "close it"), or a staff label carrying the reviewed mark.
- **Who waits:** a picture drawn on a check page still being answered. It joins once that night is finished, if
  it was left unanswered. Before that it is never on both lists, so the nightly measurement stays blind.
- **Who leaves:** an open item, once a person confirms it (as above, from any later night's answers or the latest
  run's staff labels, or an answer in the batch review below), or once a later staff rename makes every name agree
  with the machine (``agrees_now``). Closed items stay in the file with their reason; they are never reopened.

The list lives at ``<verify root>/corrections/list.json``, keyed by sighting id and written atomically, and only
the nightly step writes it.

**Batch review** (owner 2026-11-19, plan decision 37): the check page's server shows a person's open pictures in
batches of :data:`BATCH`, under that person's reference pictures. A picture that reached the reviewer's screen and
was left unmarked confirms the machine's name; a mark says who it is instead (another person, someone else, can't
tell). A picture that never reached the screen records nothing: a confirmation is never wider than what was looked
at (the app's own rule). Answers go to ``corrections/answers.json`` (written only by the page server, one save at
a time); the nightly step reads them and closes each answered item (``review_right`` / ``review_wrong`` /
``review_cant_tell``), and the job learns the answered person from it (``vetting.load_trusted``). Nothing is
written to the backend: the app's names stay as they are until a write route exists. These answers never count
toward the check page's accuracy: the list is chosen for disagreeing, not drawn at random.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

from .verify import CANT, IMAGE_PREFIX, NIGHT_RE, OTHER, _atomic_json, _night_run

BOARD_DISAGREES = ("promote", "relabel")  # decide.py: the machine names someone other than the board's name
NAMES_ONE = BOARD_DISAGREES + ("agree",)  # the machine names one person
BATCH = 24  # pictures per review batch ("certainly not one by one, most likely ... a batch", owner 2026-11-13)
VERDICTS = ("right", "wrong", "cant_tell")  # a review answer: the machine's name, another one, or nobody can tell
_REVIEW_LOCK = threading.Lock()  # the page server is threaded: two tabs must not save over each other


def list_path(verify_root: str | Path) -> Path:
    return Path(verify_root) / "corrections" / "list.json"


def _against(relation: str, machine: str, staff_state: str | None, staff_pid: str | None, reviewed: bool) -> list[str]:
    """The names nobody confirmed that the machine disagrees with: ``board`` and/or ``staff``."""
    if staff_state is not None and reviewed:
        return []  # a person looked at this picture: its label is confirmed
    out = []
    if relation in BOARD_DISAGREES:
        out.append("board")
    if (staff_state == "known" and staff_pid and staff_pid != machine) or staff_state == "stranger":
        out.append("staff")
    return out


def _staff_name(state: str | None, pid: str | None) -> str | None:
    return pid if state == "known" else state


def _check_pages(root: Path) -> tuple[dict[str, str], set[str]]:
    """``answered``: sid -> "answered" | "cant_tell" (refs.json pictures and every answer on a finished night);
    ``waiting``: pictures drawn on a night still being answered."""
    answered: dict[str, str] = {}
    waiting: set[str] = set()
    refs = root / "refs.json"
    if refs.exists():
        for p in json.loads(refs.read_text(encoding="utf-8")).get("people") or []:
            for ph in p.get("photos") or []:
                if ph.get("sighting_id"):
                    answered[str(ph["sighting_id"])] = "answered"
    for fz_path in sorted(root.glob("*/frozen.json")):
        if not NIGHT_RE.match(fz_path.parent.name):
            continue
        items = json.loads(fz_path.read_text(encoding="utf-8")).get("items") or []
        mk_path = fz_path.with_name("marks.json")
        mk = json.loads(mk_path.read_text(encoding="utf-8")) if mk_path.exists() else {}
        if not mk.get("submitted"):
            waiting |= {str(it["sighting_id"]) for it in items}
            continue
        answers = mk.get("answers") or {}
        for i, it in enumerate(items):
            a = answers.get(str(i))
            if a is not None:
                answered[str(it["sighting_id"])] = "cant_tell" if a == CANT else "answered"
    return answered, waiting


def _staff(labels_csv: Path) -> dict[str, tuple[str | None, str | None, bool]]:
    """The run's snapshot of every staff label it saw (``shadow.STAFF_FIELDS``); {} for runs written before it."""
    f = labels_csv.with_name(labels_csv.name.replace("labels_", "staff_", 1))
    if not f.exists():
        return {}
    with f.open(newline="", encoding="utf-8") as fh:
        return {r["sighting_id"]: (r["human_state"] or None, r["human_person_id"] or None, r["human_reviewed"] == "1")
                for r in csv.DictReader(fh)}


def _load(root: Path) -> dict:
    f = list_path(root)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"items": {}, "held": {}}


def _save(root: Path, data: dict) -> None:
    f = list_path(root)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, f)


def open_items(verify_root: str | Path) -> list[dict]:
    """The open items, by sighting id (never by a machine number)."""
    items = _load(Path(verify_root))["items"]
    return [items[sid] for sid in sorted(items) if items[sid]["status"] == "open"]


def answers_path(verify_root: str | Path) -> Path:
    return Path(verify_root) / "corrections" / "answers.json"


def load_answers(verify_root: str | Path) -> dict:
    """sid -> the latest batch-review answer: ``verdict`` (:data:`VERDICTS`), ``person_id`` (who it shows; None for
    someone else or can't tell), ``answer`` (the mark; "" when left unmarked), ``machine``, ``facility_id``,
    ``day``, ``at`` (seconds), ``reviewer``. Rows are read as they are: callers skip any that aren't a dict."""
    f = answers_path(verify_root)
    return (json.loads(f.read_text(encoding="utf-8")).get("answers") or {}) if f.exists() else {}


def review_people(verify_root: str | Path) -> list[dict]:
    """The people a reviewer can choose, from ``refs.json`` (key, person_id, name, photos); [] without one."""
    f = Path(verify_root) / "refs.json"
    return (json.loads(f.read_text(encoding="utf-8")).get("people") or []) if f.exists() else []


def to_review(verify_root: str | Path) -> dict[str, list[dict]]:
    """machine person_id -> its open items nobody has answered yet, oldest first (never the held ones)."""
    answered = load_answers(verify_root)
    out: dict[str, list[dict]] = defaultdict(list)
    for it in open_items(verify_root):
        if it["sighting_id"] not in answered:
            out[it["machine"]].append(it)
    return {pid: sorted(v, key=lambda i: (i["day"], i["sighting_id"])) for pid, v in out.items()}


def list_summary(verify_root: str | Path) -> dict:
    """For the review page: open items, of which answered (they close at the next nightly update), held ones,
    and closed ones by reason."""
    data = _load(Path(verify_root))
    answered = load_answers(verify_root)
    closed: dict[str, int] = defaultdict(int)
    n_open = n_answered = 0
    for it in data["items"].values():
        if it["status"] == "open":
            n_open += 1
            n_answered += it["sighting_id"] in answered
        else:
            closed[it.get("closed_reason") or "?"] += 1
    return {"open": n_open, "answered": n_answered, "held": len(data["held"]), "closed": dict(sorted(closed.items()))}


def save_review(verify_root: str | Path, machine: str, body: dict, *, now: int | None = None) -> dict:
    """One submitted batch of the machine's person ``machine``. ``body``: ``seen`` (sighting ids that reached the
    screen), ``marks`` (sid -> a person key offered on the page, ``other`` or ``cant_tell``), ``reviewer``.
    A seen picture without a mark confirms the machine's name; an unseen one records nothing. The whole batch is
    refused (``ValueError``) if a picture isn't this person's on the list, or a mark is on an unseen picture or
    isn't a choice the page offers. -> counts by verdict."""
    root = Path(verify_root)
    if not isinstance(body, dict):
        raise ValueError("the batch must be an object")
    seen, marks = body.get("seen"), body.get("marks") or {}
    if not isinstance(seen, list) or not all(isinstance(x, str) for x in seen):
        raise ValueError("seen must be a list of picture ids")
    seen = list(dict.fromkeys(seen))  # a picture counted twice is still one answer
    if not isinstance(marks, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in marks.items()):
        raise ValueError("marks must map picture ids to answers")
    people = [p for p in review_people(root) if p.get("photos")]  # the page offers only these (verify.page_data)
    by_key = {p["key"]: str(p["person_id"]) for p in people}
    if set(marks) - set(seen):
        raise ValueError("a picture you marked never reached the screen")
    if bad := {v for v in marks.values() if v not in by_key and v not in (OTHER, CANT)}:
        raise ValueError(f"not a choice on the page: {sorted(bad)}")
    reviewer = str(body.get("reviewer") or "")[:40]
    stamp = int(time.time()) if now is None else now
    with _REVIEW_LOCK:
        items = _load(root)["items"]  # held pictures are not in it: they wait for their check page
        for sid in seen:
            if sid not in items or items[sid].get("machine") != machine:
                raise ValueError(f"{sid} is not one of this person's pictures on the list")
        f = answers_path(root)
        data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"answers": {}}
        counts = {v: 0 for v in VERDICTS}
        for sid in seen:
            a = marks.get(sid, "")
            pid = machine if a == "" else by_key.get(a)  # other / cant_tell: nobody on the list
            verdict = "cant_tell" if a == CANT else "right" if pid == machine else "wrong"
            counts[verdict] += 1
            it = items[sid]
            data["answers"][sid] = {"verdict": verdict, "person_id": pid, "answer": a, "machine": machine,
                                    "facility_id": it["facility_id"], "day": it["day"], "at": stamp,
                                    "reviewer": reviewer}
        f.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(f, data)
        with f.with_name("answers.log").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": stamp, "machine": machine, "seen": len(seen), **counts,
                                 "reviewer": reviewer}) + "\n")
    return counts


def update_list(runs_dir: str | Path, verify_root: str | Path, facility: str, day: str, *, run_id: str | None = None,
                now: int | None = None, log=print) -> dict | None:
    """Add the night's pictures, release the ones whose check-page night is finished, close the confirmed ones.
    Uses the run the night's check page was drawn from, so both lists come from one run; when the check page is in
    use (``refs.json`` exists) and tonight's isn't drawn yet, nothing happens. -> counts, or None when skipped."""
    root, facility = Path(verify_root), str(facility)
    frozen = root / f"{facility}_{day}" / "frozen.json"
    if run_id is None and frozen.exists():
        run_id = json.loads(frozen.read_text(encoding="utf-8"))["run_id"]
    elif run_id is None and (root / "refs.json").exists():
        log(f"[corrections] {facility} {day}: tonight's check page isn't drawn yet; the list waits for it")
        return None
    found = _night_run(Path(runs_dir), facility, day, run_id)
    if found is None:
        log(f"[corrections] {facility} {day}: no finished non-dry run in {runs_dir}")
        return None
    labels_csv, meta = found
    with labels_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "human_state" not in rows[0]:
        raise SystemExit(f"{labels_csv} has no staff-label columns (written before 0.2.4); use a later night")
    answered, waiting = _check_pages(root)
    reviews = load_answers(root)
    staff = _staff(labels_csv)
    data = _load(root)
    items, held = data["items"], data["held"]
    stamp = int(time.time()) if now is None else now
    out = {"facility_id": facility, "day": day, "run_id": meta["run_id"], "added": 0, "held": 0, "released": 0,
           "closed": {}}

    for r in rows:  # tonight's pictures
        sid = r["sighting_id"]
        if sid in items or sid in held or sid in answered or r["relation"] not in NAMES_ONE or not r["person_id"]:
            continue
        if not (r.get("image_url") or "").startswith(IMAGE_PREFIX):
            continue
        hs, hp = r.get("human_state") or None, r.get("human_person_id") or None
        against = _against(r["relation"], r["person_id"], hs, hp, r.get("human_reviewed") == "1")
        if not against:
            continue
        item = {"sighting_id": sid, "facility_id": r["facility_id"] or facility, "day": r["day"] or day,
                "url": r["image_url"], "machine": r["person_id"], "relation": r["relation"], "via": r["via"],
                "board": r["board_person_id"] if r["board_state"] == "known" else r["board_state"],
                "staff": _staff_name(hs, hp), "against": against, "run_id": meta["run_id"], "added_at": stamp,
                "status": "open"}
        if sid in waiting:
            held[sid] = item
            out["held"] += 1
        else:
            items[sid] = item
            out["added"] += 1

    for sid in [s for s in held if s not in waiting]:  # their check-page night is finished
        item = held.pop(sid)
        if sid not in answered:  # answered there: a person confirmed it
            items[sid] = item
            out["released"] += 1

    for sid, item in items.items():  # confirmed since, or every name agrees now
        if item["status"] != "open":
            continue
        reason = answered.get(sid)
        if reason is None and isinstance(rv := reviews.get(sid), dict) and rv.get("verdict") in VERDICTS:
            reason = f"review_{rv['verdict']}"  # the batch review
        if reason is None and sid in staff:
            hs, hp, reviewed = staff[sid]
            if hs is not None and reviewed:
                reason = "reviewed"
            else:
                item["staff"] = _staff_name(hs, hp)
                item["against"] = _against(item["relation"], item["machine"], hs, hp, False)
                reason = None if item["against"] else "agrees_now"
        if reason:
            item.update(status="closed", closed_reason=reason, closed_at=stamp)
            out["closed"][reason] = out["closed"].get(reason, 0) + 1

    out["open"] = sum(1 for i in items.values() if i["status"] == "open")
    out["closed"] = dict(sorted(out["closed"].items()))
    _save(root, data)
    log(f"[corrections] {facility} {day} run {meta['run_id'][:8]}: +{out['added']} added, {out['held']} waiting for "
        f"the check page, {out['released']} released, closed {out['closed'] or 'none'}; {out['open']} open")
    return out
