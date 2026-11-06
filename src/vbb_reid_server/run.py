"""The production run for one facility-day (Phase B): pull → embed → decide → register → push → finish.

Same decision code as the offline shadow (``shadow.decide_night``); what differs is where the data
comes from (the ``/worker/*`` API, ``api.export_from_backend``) and where the labels go (the backend,
inactive until activated). The local CSV/JSON of every run are written as in shadow, so a run can be
audited and scored with the Phase A tools.

Rules this module owns:
- References are the labels that exist when the run starts (``label_cutoff_ms`` = start time) and,
  as in production's references/pull, only those of this facility's roster (``RefPolicy.roster_only``).
- A sighting with ANY human overlay (known, stranger, or reset to pending) is decided for the local
  file but never pushed.
- Register comes after the decision, so the lease covers only push and finish (minutes), not the
  embedding of a cold cache. Once register has been attempted the run is always finished: ``done`` on
  success, ``failed`` with the error otherwise (also on SIGTERM / Ctrl-C). The contract makes
  register and finish idempotent, so a lost answer is safe to replay.
- The worker never activates. Activation lives on the app host (owner 2026-11-05: manual through the
  trial, automatic with guards after the Phase B gate); the guards are computed here and reported in
  the run's ``counts`` (``guard_failures``) for that step to read.
- Relabels are flags in v1 whatever the operating point says (``allow_relabel`` forced off).
- Embedding: only what the cache lacks, face only for references (their body vectors are never read),
  never days after the probe day.
- Crops downloaded for the run are deleted when it ends, and leftovers older than a day (from a run
  that was killed) with them (decision 14), unless asked to keep them. ``--cdn-cache`` is the run's
  own folder; never point it at the Phase A cache.
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
    """Checks a run must pass to be activated.

    ``max_promote_share``: promotions as a share of the day's sightings (None = not checked; the owner
    sets the number before automatic activation). ``max_crop_fail_share``: crops that could not be
    downloaded, as a share of crops looked up. ``empty_run_min_sightings``: a run that names nobody out
    of at least this many sightings is suspect (a CDN outage, a broken model), and activating it would
    supersede a good earlier run of the same day."""
    max_promote_share: float | None = None
    max_crop_fail_share: float = 0.05
    empty_run_min_sightings: int = 20


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


def guard_failures(counts: dict[str, int], rejected: Counter, n_sightings: int, guards: Guards, *,
                   thin_people: dict[str, int] | None = None, min_refs: int = 0,
                   crop_stats: dict[str, int] | None = None) -> list[str]:
    out = []
    for reason in ("not_in_roster", "not_in_run_scope", "bad_row", "run_not_running", "human_labeled"):
        if rejected.get(reason):
            out.append(f"{rejected[reason]} row(s) rejected as {reason}")
    if guards.max_promote_share is not None and n_sightings:
        share = counts.get("pushed_promote", 0) / n_sightings
        if share > guards.max_promote_share:
            out.append(f"promotions are {share:.1%} of the day's sightings (limit {guards.max_promote_share:.1%})")
    if thin_people:
        out.append(f"{len(thin_people)} named person(s) below {min_refs} face references: "
                   + ", ".join(f"{k} ({n})" for k, n in sorted(thin_people.items())))
    if crop_stats:
        looked = sum(crop_stats.get(k, 0) for k in ("lan", "cdn_hit", "cdn_fetched", "cdn_failed"))
        failed = crop_stats.get("cdn_failed", 0)
        if looked and failed / looked > guards.max_crop_fail_share:
            out.append(f"{failed} of {looked} crops could not be downloaded")
    if n_sightings >= guards.empty_run_min_sightings and counts.get("labels", 0) == 0:
        out.append(f"no labels from {n_sightings} sightings")
    return out


def _processed_sids(emb_root, face_id, body_id, facility: str, day: str) -> set[str] | None:
    """Sighting ids the cache already holds for every model in use; None = unknown (embed all)."""
    from .embed import StaleCache, load_body, load_face, load_processed
    try:
        if face_id:
            load_face(Path(emb_root), face_id, facility, day)  # raises StaleCache for an old format: embed all
        done = {k.split(":")[0] for k in load_processed(Path(emb_root), face_id, facility, day)} if face_id else None
    except (StaleCache, FileNotFoundError, OSError, KeyError, ValueError):
        return None
    if body_id:
        try:
            b = {str(x) for x in load_body(Path(emb_root), body_id, facility, day).sighting_id}
        except (FileNotFoundError, OSError, KeyError, ValueError):
            return None
        done = b if done is None else done & b
    return done or set()


def embed_missing(ex, facility: str, day: str, models, store, emb_root, log=print) -> dict[str, int]:
    """Embed what the cache lacks: the probe day with every model, reference days face only, no day
    after the probe day. Returns how many sightings were sent to the models."""
    from .embed import embed_day
    face_only = replace(models, body=None)
    sent = 0
    for (f, d) in sorted({(s.facility_id, s.day) for s in ex.sightings}):
        if d > day:
            continue
        probe = (f, d) == (facility, day)
        m = models if probe else face_only
        ss = ex.by_facility_day(f, d)
        done = _processed_sids(emb_root, m.face_id, m.body_id, f, d)
        todo = ss if done is None else [s for s in ss if s.sighting_id not in done]
        if todo:
            embed_day(todo, store, m, emb_root, f, d, log=log)
            sent += len(todo)
    return {"embedded_sightings": sent}


def run_night(client: BackendClient, facility_id: int, day: str, *, th: Thresholds, quality: QualityPolicy,
              policy: RefPolicy, emb_root: str | Path, out_dir: str | Path, models=None,
              face_id: str | None = None, body_id: str | None = None, store=None,
              tz_name: str = "Asia/Shanghai", dry_run: bool = False, guards: Guards = Guards(),
              keep_crops: bool = False, worker_id: str | None = None,
              now_ms=lambda: int(time.time() * 1000), log=print) -> dict:
    """``models``: a loaded model set (embeds what the cache lacks); ``None`` = use the cache only,
    with ``face_id`` / ``body_id`` naming the cached models (tests, re-runs)."""
    if models is not None:
        face_id, body_id = models.face_id, models.body_id
    if not face_id:
        raise ValueError("no face model: pass a model set or a cached face_id")
    policy = replace(policy, include_same_day=True, roster_only=True)
    th = replace(th, allow_relabel=False)
    fac = str(facility_id)
    run_id = uuid.uuid4().hex
    started = now_ms()
    model_id = model_id_of(face_id, body_id)
    worker_id = worker_id or socket.gethostname()[:64]
    thresholds = _clean({**th.as_dict(), "ref_policy": asdict(policy), "quality": asdict(quality)})
    attempted = False
    summary: dict = {"run_id": run_id, "facility_id": fac, "day": day, "dry_run": dry_run, "model_id": model_id}
    try:
        ex = export_from_backend(client, facility_id, day, window_days=policy.window_days, tz_name=tz_name)
        n_probe = len(ex.by_facility_day(fac, day))
        log(f"[run] {fac} {day}: {n_probe} sightings, {sum(1 for s in ex.sightings if s.human_state == 'known')} "
            f"human-labeled references in the window, roster {len(ex.roster_ids(fac))}")
        counts: dict[str, int] = {"sightings": n_probe}
        if models is not None:
            counts.update(embed_missing(ex, fac, day, models, store, emb_root, log=log))
        ws = Workspace(ex, emb_root, face_id, body_id, quality)
        decided = decide_night(ws, fac, day, th, policy, label_cutoff_ms=started)
        local = shadow_run(ws, fac, day, th, out_dir, policy=policy, label_cutoff_ms=started,
                           model_name=model_id, run_id=run_id, decided=decided,
                           extra={"dry_run": dry_run}, log=log)
        by_sid = {s.sighting_id: s for s in decided["sightings"]}
        keep = [l for l in decided["labels"] if not by_sid[l.sighting_id].has_overlay]
        rows = [label_row(l) for l in keep]
        refs = decided["refs"]
        thin = {l.person_key: refs.n_sightings(l.person_key) if l.person_key in refs else 0
                for l in keep if l.state == "known" and l.person_key is not None
                and (l.person_key not in refs or refs.n_sightings(l.person_key) < th.min_refs)}
        counts.update(labels=len(rows), references=sum(refs.n_sightings(k) for k in refs),
                      references_people=len(refs))
        for r in rows:
            k = f"pushed_{r['relation']}"
            counts[k] = counts.get(k, 0) + 1
        rejected: Counter = Counter()
        accepted = 0
        if not dry_run:
            attempted = True  # from here on the run is always finished, even if register's answer was lost
            client.register(run_id, facility_id, day, model_id, worker_id, thresholds)
            log(f"[run] {fac} {day}: registered run {run_id}")
            for i in range(0, len(rows), PAGE):
                res = client.push_labels(run_id, rows[i:i + PAGE]) or {}
                accepted += int(res.get("accepted", 0))
                rejected.update(str(x.get("reason")) for x in res.get("rejected") or [])
        counts["accepted"] = accepted
        for reason, n in rejected.items():
            counts[f"rejected_{reason}"] = n
        crop_stats = dict(store.stats) if store is not None and getattr(store, "stats", None) else None
        failures = guard_failures(counts, rejected, n_probe, guards, thin_people=thin, min_refs=th.min_refs,
                                  crop_stats=crop_stats)
        counts["guard_failures"] = len(failures)
        summary.update(counts=counts, guard_failures=failures, local=local)
        if dry_run:
            log(f"[run] {fac} {day}: dry run, nothing pushed: {counts}")
            return summary
        client.finish(run_id, "done", counts)
        attempted = False  # finished
        log(f"[run] {fac} {day}: done {counts}" + (" GUARDS: " + "; ".join(failures) if failures else ""))
        return summary
    except BaseException as e:  # also SIGTERM (raised as SystemExit by the CLI) and Ctrl-C
        if attempted:
            try:
                client.finish(run_id, "failed", {}, client.redact(f"{e.__class__.__name__}: {e}")[:255])
            except BackendError as e2:
                log(f"[run] {fac} {day}: could not mark run {run_id} failed: {e2}")
        raise
    finally:
        if store is not None and not keep_crops:
            n = store.delete_fetched() + store.prune_cdn(1.0)
            if n:
                log(f"[run] deleted {n} downloaded crop(s)")
