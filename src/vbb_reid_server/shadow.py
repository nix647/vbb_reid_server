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
from pathlib import Path

from .decide import BodyIndex, MachineLabel, Thresholds, decide_day, face_scores
from .pipeline import RefPolicy, Workspace, org_facilities

RUN_AT_DAY_END = "run"  # label cutoff = the nightly run's start for that day


def _cutoff(ws: Workspace, facility: str, day: str, label_cutoff_ms):
    if label_cutoff_ms == RUN_AT_DAY_END:
        from .evaluate import run_cutoff_ms
        return run_cutoff_ms(day, ws.export.facility_tz.get(facility, "Asia/Shanghai"))
    return label_cutoff_ms

LABEL_FIELDS = ["run_id", "sighting_id", "facility_id", "day", "did", "room_id", "board_state",
                "board_person_id", "state", "relation", "person_id", "person_key", "via", "confidence",
                "margin", "face_sim", "body_sim", "human_labeled", "evidence"]


def shadow_run(ws: Workspace, facility: str, day: str, th: Thresholds, out_dir: str | Path, *,
               policy: RefPolicy = RefPolicy(include_same_day=True), label_cutoff_ms: int | str | None = RUN_AT_DAY_END,
               model_name: str = "", log=print) -> dict:
    """``label_cutoff_ms``: ``"run"`` (default, production-like), an epoch-ms value, or ``None``
    (every label in the export — only for debugging; it leaks the future)."""
    ex = ws.export
    run_id = uuid.uuid4().hex
    started = time.time()
    facs = org_facilities(ex, facility)
    label_cutoff_ms = _cutoff(ws, facility, day, label_cutoff_ms)
    roster = ex.roster(facility)
    refs = ws.references(facs, day, label_cutoff_ms=label_cutoff_ms, policy=policy)
    seeds = ws.human_seeds(facility, day, label_cutoff_ms)
    sightings = ex.by_facility_day(facility, day)
    probe_faces, body, body_crops = ws.probe_inputs(facility, day)
    fs = face_scores(probe_faces, refs, th)
    labels: list[MachineLabel] = decide_day(sightings, fs, body, body_crops, roster, ex.key, th, human_seeds=seeds,
                                            body_index=BodyIndex.build(sightings, body) if body else None)
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
                "person_id": l.person_id or "", "person_key": l.person_key or "", "via": l.via,
                "confidence": l.confidence, "margin": "" if l.margin is None else l.margin,
                "face_sim": "" if l.face_sim is None else l.face_sim,
                "body_sim": "" if l.body_sim is None else l.body_sim,
                "human_labeled": int(human), "evidence": json.dumps(l.evidence, ensure_ascii=False),
            })
    run = {
        "run_id": run_id, "facility_id": facility, "day": day, "model": model_name,
        "face_model_id": ws.face_id, "body_model_id": ws.body_id, "thresholds": th.as_dict(),
        "ref_policy": policy.__dict__, "label_cutoff_ms": label_cutoff_ms,
        "references": {k: refs.n_sightings(k) for k in refs}, "same_day_seeds": len(seeds),
        "counts": counts, "started_at": started, "finished_at": time.time(),
    }
    (out / f"run_{facility}_{day}_{run_id[:8]}.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    log(f"[shadow] {facility} {day} run {run_id[:8]}: {counts}")
    return run
