"""The production run for one facility-day (Phase B): register → pull → embed → decide → push → finish.

Same decision code as the offline shadow (``shadow.decide_night``); what differs is where the data
comes from (the ``/worker/*`` API, ``api.export_from_backend``) and where the labels go (the backend,
inactive until activated). The local CSV/JSON of every run are written as in shadow, so a run can be
audited and scored with the Phase A tools.

Rules this module owns:
- References are the labels that exist when the run starts (``label_cutoff_ms`` = start time) and,
  as in production's references/pull, only those of this facility's roster (``RefPolicy.roster_only``).
- Human-labeled sightings are decided (for the local file) but never pushed.
- A run that fails after registering is finished as ``failed`` with the error, so its lease does not
  block the next attempt for an hour.
- Activation is a separate, deliberate step (``--activate``, admin token) and is refused when a guard
  trips (owner 2026-11-05: manual through the trial, automatic with guards after the Phase B gate).
- Crops downloaded for the run are deleted when it ends (decision 14), unless asked to keep them.
"""

from __future__ import annotations

import math
import socket
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from . import __version__
from .api import PAGE, BackendClient, BackendError, export_from_backend
from .decide import MachineLabel, Thresholds
from .pipeline import RefPolicy, Workspace
from .quality import QualityPolicy
from .shadow import decide_night, shadow_run


@dataclass(frozen=True)
class Guards:
    """Checks a run must pass to be activated. ``max_promote_share``: promotions as a share of the
    day's sightings (None = not checked; the owner sets the number before automatic activation)."""
    max_promote_share: float | None = None


def model_id_of(face_id: str | None, body_id: str | None) -> str:
    mid = f"{face_id or 'none'}|{body_id or 'none'}|vbb-reid-server {__version__}"
    if len(mid) > 256:
        raise ValueError(f"model_id is {len(mid)} characters; the contract allows 256: {mid}")
    return mid


def _clean(v):
    """JSON-safe: numpy scalars to Python, non-finite floats to None (the API refuses NaN)."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def label_row(l: MachineLabel) -> dict:
    row = {"sighting_id": l.sighting_id, "state": l.state, "relation": l.relation, "via": l.via,
           "confidence": float(l.confidence), "evidence": _clean(l.evidence or {})}
    if l.person_id is not None:
        row["person_id"] = l.person_id
    for k in ("margin", "face_sim", "body_sim"):
        v = getattr(l, k)
        row[k] = None if v is None or not math.isfinite(float(v)) else float(v)
    return row


def guard_failures(counts: dict[str, int], rejected: Counter, n_sightings: int, guards: Guards) -> list[str]:
    out = []
    for reason in ("not_in_roster", "not_in_run_scope", "bad_row", "run_not_running"):
        if rejected.get(reason):
            out.append(f"{rejected[reason]} row(s) rejected as {reason}")
    if guards.max_promote_share is not None and n_sightings:
        share = counts.get("pushed_promote", 0) / n_sightings
        if share > guards.max_promote_share:
            out.append(f"promotions are {share:.1%} of the day's sightings (limit {guards.max_promote_share:.1%})")
    return out


def run_night(client: BackendClient, facility_id: int, day: str, *, th: Thresholds, quality: QualityPolicy,
              policy: RefPolicy, emb_root: str | Path, out_dir: str | Path, models=None,
              face_id: str | None = None, body_id: str | None = None, store=None,
              tz_name: str = "Asia/Shanghai", dry_run: bool = False, activate: bool = False,
              guards: Guards = Guards(), keep_crops: bool = False, worker_id: str | None = None,
              now_ms=lambda: int(time.time() * 1000), log=print) -> dict:
    """``models``: a loaded model set (embeds what the cache lacks); ``None`` = use the cache only,
    with ``face_id`` / ``body_id`` naming the cached models (tests, re-runs)."""
    if models is not None:
        face_id, body_id = models.face_id, models.body_id
    if not face_id:
        raise ValueError("no face model: pass a model set or a cached face_id")
    policy = replace(policy, include_same_day=True, roster_only=True)
    fac = str(facility_id)
    run_id = uuid.uuid4().hex
    started = now_ms()
    model_id = model_id_of(face_id, body_id)
    worker_id = worker_id or socket.gethostname()[:64]
    thresholds = _clean({**th.as_dict(), "ref_policy": asdict(policy), "quality": asdict(quality)})
    registered = False
    summary: dict = {"run_id": run_id, "facility_id": fac, "day": day, "dry_run": dry_run, "model_id": model_id}
    try:
        if not dry_run:
            client.register(run_id, facility_id, day, model_id, worker_id, thresholds)
            registered = True
            log(f"[run] {fac} {day}: registered run {run_id}")
        ex = export_from_backend(client, facility_id, day, window_days=policy.window_days, tz_name=tz_name)
        n_probe = len(ex.by_facility_day(fac, day))
        log(f"[run] {fac} {day}: {n_probe} sightings, {sum(1 for s in ex.sightings if s.human_state == 'known')} "
            f"human-labeled references in the window, roster {len(ex.roster_ids(fac))}")
        if models is not None:
            from .embed import embed_day
            for (f, d) in sorted({(s.facility_id, s.day) for s in ex.sightings}):
                embed_day(ex.by_facility_day(f, d), store, models, emb_root, f, d, log=log)
        ws = Workspace(ex, emb_root, face_id, body_id, quality)
        decided = decide_night(ws, fac, day, th, policy, label_cutoff_ms=started)
        local = shadow_run(ws, fac, day, th, out_dir, policy=policy, label_cutoff_ms=started,
                           model_name=model_id, run_id=run_id, decided=decided, log=log)
        by_sid = {s.sighting_id: s for s in decided["sightings"]}
        rows = [label_row(l) for l in decided["labels"] if not by_sid[l.sighting_id].human_labeled]
        counts: dict[str, int] = {"sightings": n_probe, "labels": len(rows),
                                  "references": sum(decided["refs"].n_sightings(k) for k in decided["refs"]),
                                  "references_people": len(decided["refs"])}
        for r in rows:
            k = f"pushed_{r['relation']}"
            counts[k] = counts.get(k, 0) + 1
        rejected: Counter = Counter()
        accepted = 0
        if not dry_run:
            for i in range(0, len(rows), PAGE):
                res = client.push_labels(run_id, rows[i:i + PAGE]) or {}
                accepted += int(res.get("accepted", 0))
                rejected.update(str(x.get("reason")) for x in res.get("rejected") or [])
        counts["accepted"] = accepted
        for reason, n in rejected.items():
            counts[f"rejected_{reason}"] = n
        failures = guard_failures(counts, rejected, n_probe, guards)
        counts["guard_failures"] = len(failures)
        summary.update(counts=counts, guard_failures=failures, local=local)
        if not dry_run:
            client.finish(run_id, "done", counts)
            log(f"[run] {fac} {day}: done {counts}")
            if activate:
                if failures:
                    log(f"[run] {fac} {day}: NOT activated: " + "; ".join(failures))
                else:
                    client.activate(run_id, "worker --activate")
                    summary["activated"] = True
                    log(f"[run] {fac} {day}: activated")
        else:
            log(f"[run] {fac} {day}: dry run, nothing pushed: {counts}")
        return summary
    except Exception as e:
        if registered:
            try:
                client.finish(run_id, "failed", {}, f"{e.__class__.__name__}: {e}"[:255])
            except BackendError as e2:
                log(f"[run] {fac} {day}: could not mark run {run_id} failed: {e2}")
        raise
    finally:
        if store is not None and not keep_crops:
            n = store.delete_fetched()
            if n:
                log(f"[run] deleted {n} downloaded crop(s)")
