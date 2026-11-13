"""A production-shaped run for one facility-day, written to files (Phase A shadow).

References: every human ``known`` label that existed when the nightly run would have
started for that day (end of the day + 2.5 h, facility time), same-day labels included;
same-day human labels made by then also seed the body clusters. Labels made later are
NOT used (``label_cutoff_ms``): an export taken weeks afterwards would otherwise let the
shadow see the future and report precision production cannot reach.
Sightings that carry a human label are decided too but marked ``human_labeled`` —
production never emits a machine row for them, and scoring excludes them."""

from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .decide import BodyIndex, MachineLabel, Thresholds, decide_day, face_scores
from .pipeline import RefPolicy, Workspace, org_facilities
from .vetting import SETTLED, VetPolicy, clear_crop, vet_labels

RUN_AT_DAY_END = "run"  # label cutoff = the nightly run's start for that day


def _cutoff(ws: Workspace, facility: str, day: str, label_cutoff_ms):
    if label_cutoff_ms == RUN_AT_DAY_END:
        from .evaluate import run_cutoff_ms
        return run_cutoff_ms(day, ws.export.facility_tz.get(facility, "Asia/Shanghai"))
    return label_cutoff_ms

LABEL_FIELDS = ["run_id", "sighting_id", "facility_id", "day", "did", "room_id", "board_state",
                "board_person_id", "state", "relation", "person_id", "person_key", "via", "confidence",
                "margin", "face_sim", "body_sim", "human_labeled", "evidence",
                "image_url",  # the verification page shows this picture (added 0.2.3)
                "human_state", "human_person_id", "label_check",  # the staff label and its check (0.2.4)
                "human_reviewed"]  # someone looked at that staff label (backend reviewed mark, 0.2.4)
# Every staff label the run saw (probe-day overlays and the references of the window), with its reviewed mark:
# the correction list re-checks older pictures against the latest one (corrections.py).
STAFF_FIELDS = ["sighting_id", "facility_id", "day", "human_state", "human_person_id", "human_reviewed"]


def decide_night(ws: Workspace, facility: str, day: str, th: Thresholds, policy: RefPolicy,
                 label_cutoff_ms: int | None, vet: VetPolicy | None = None,
                 trusted: dict[str, str | None] | None = None) -> dict:
    """The decision for one facility-day, shared by the offline shadow and the production run.
    ``vet``: learn only from staff labels someone looked at (``vetting``); ``trusted`` are your check-page
    answers (``vetting.load_trusted``). -> {labels, sightings, refs, seeds, vetting}."""
    ex = ws.export
    facs = org_facilities(ex, facility)
    roster = ex.roster(facility)
    vetting = label_filter = crop_filter = None
    if vet is not None:
        cands = [s for s in ex.sightings if s.facility_id in facs and s.human_state == "known" and s.day <= day
                 and (label_cutoff_ms is None or (s.corrected_ts is not None and s.corrected_ts < label_cutoff_ms))]
        vetting = vet_labels(ws, cands, trusted or {}, vet)
        trusted_ids = vetting.trusted
        label_filter = lambda s: s.sighting_id in trusted_ids  # noqa: E731
        crop_filter = lambda ft, i: clear_crop(ft, i, vet)  # noqa: E731
    refs = ws.references(facs, day, label_cutoff_ms=label_cutoff_ms, policy=policy, label_filter=label_filter,
                         crop_filter=crop_filter, roster_ids=ex.roster_ids(facility) if policy.roster_only else None)
    seeds = ws.human_seeds(facility, day, label_cutoff_ms)
    if vetting is not None:
        seeds = {sid: k for sid, k in seeds.items() if sid in trusted_ids}
    sightings = ex.by_facility_day(facility, day)
    probe_faces, body, body_crops = ws.probe_inputs(facility, day)
    fs = face_scores(probe_faces, refs, th)
    labels: list[MachineLabel] = decide_day(sightings, fs, body, body_crops, roster, ex.key, th, human_seeds=seeds,
                                            body_index=BodyIndex.build(sightings, body) if body else None)
    return {"labels": labels, "sightings": sightings, "refs": refs, "seeds": seeds, "vetting": vetting}


def disagreements(decided: dict, key) -> list[dict]:
    """Staff labels of the night the machine names someone else for, decided without them: ``known`` labels
    nobody looked at (one someone looked at, or you answered, is settled; the job learned from it, so agreement
    there means nothing) and stranger labels (never learned from). ``key``: person_id -> identity key."""
    vetting = decided.get("vetting")
    if vetting is None:
        return []
    by_sid = {s.sighting_id: s for s in decided["sightings"]}
    out = []
    for l in decided["labels"]:
        s = by_sid[l.sighting_id]
        if l.state != "known" or s.human_state not in ("known", "stranger"):
            continue
        if s.human_state == "known":
            if vetting.status.get(s.sighting_id) in SETTLED or key(s.human_person_id) == l.person_key:
                continue
        out.append({"sighting_id": s.sighting_id, "url": s.image_url or "", "machine": l.person_id,
                    "staff": s.human_person_id if s.human_state == "known" else None,
                    "check": vetting.status.get(s.sighting_id, "stranger"), "via": l.via})
    return out


def shadow_run(ws: Workspace, facility: str, day: str, th: Thresholds, out_dir: str | Path, *,
               policy: RefPolicy = RefPolicy(include_same_day=True), label_cutoff_ms: int | str | None = RUN_AT_DAY_END,
               model_name: str = "", run_id: str | None = None, decided: dict | None = None,
               report_key=None, extra: dict | None = None, vet: VetPolicy | None = None,
               trusted: dict[str, str | None] | None = None, log=print) -> dict:
    """``label_cutoff_ms``: ``"run"`` (default, production-like), an epoch-ms value, or ``None``
    (every label in the export — only for debugging; it leaks the future).
    ``decided``: a ``decide_night`` result to write instead of deciding again (the production run).
    ``report_key``: person_id -> key written in ``person_key`` (default: the decision's own key).
    ``extra``: more fields for the run JSON (the production run records ``dry_run``)."""
    ex = ws.export
    run_id = run_id or uuid.uuid4().hex
    started = time.time()
    label_cutoff_ms = _cutoff(ws, facility, day, label_cutoff_ms)
    if decided is None:
        decided = decide_night(ws, facility, day, th, policy, label_cutoff_ms, vet=vet, trusted=trusted)
    labels, sightings, refs, seeds = decided["labels"], decided["sightings"], decided["refs"], decided["seeds"]
    vetting = decided.get("vetting")
    by_sid = {s.sighting_id: s for s in sightings}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"sightings": len(sightings)}
    with (out / f"labels_{facility}_{day}_{run_id[:8]}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for l in labels:
            s = by_sid[l.sighting_id]
            human = s.human_labeled
            key = f"{l.state}/{l.relation}/{l.via}" + ("/human_labeled" if human else "")
            counts[key] = counts.get(key, 0) + 1
            w.writerow({
                "run_id": run_id, "sighting_id": l.sighting_id, "facility_id": facility, "day": day,
                "did": s.did, "room_id": s.room_id, "board_state": s.board_state,
                "board_person_id": s.board_person_id or "", "state": l.state, "relation": l.relation,
                "person_id": l.person_id or "",
                "person_key": (report_key(l.person_id) if report_key and l.person_id else l.person_key) or "",
                "via": l.via,
                "confidence": l.confidence, "margin": "" if l.margin is None else l.margin,
                "face_sim": "" if l.face_sim is None else l.face_sim,
                "body_sim": "" if l.body_sim is None else l.body_sim,
                "human_labeled": int(human), "evidence": json.dumps(l.evidence, ensure_ascii=False),
                "image_url": s.image_url or "",
                "human_state": s.human_state or "", "human_person_id": s.human_person_id or "",
                "label_check": vetting.status.get(l.sighting_id, "") if vetting is not None else "",
                "human_reviewed": int(s.human_reviewed),
            })
    with (out / f"staff_{facility}_{day}_{run_id[:8]}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=STAFF_FIELDS)
        w.writeheader()
        for s in ex.sightings:
            if s.human_state is not None:
                w.writerow({"sighting_id": s.sighting_id, "facility_id": s.facility_id, "day": s.day,
                            "human_state": s.human_state, "human_person_id": s.human_person_id or "",
                            "human_reviewed": int(s.human_reviewed)})
    run = {
        "run_id": run_id, "facility_id": facility, "day": day, "model": model_name,
        "face_model_id": ws.face_id, "body_model_id": ws.body_id, "thresholds": th.as_dict(),
        "ref_policy": policy.__dict__, "label_cutoff_ms": label_cutoff_ms,
        "references": {k: refs.n_sightings(k) for k in refs}, "same_day_seeds": len(seeds),
        "counts": counts, "started_at": started, "finished_at": time.time(),
        "label_check": None if vetting is None else {"policy": asdict(vet) if vet else None, "counts": vetting.counts()},
        "disagreements": disagreements(decided, ws.export.key),
        **(extra or {}),
    }
    (out / f"run_{facility}_{day}_{run_id[:8]}.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    log(f"[shadow] {facility} {day} run {run_id[:8]}: {counts}")
    return run
