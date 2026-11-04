"""Offline evaluation (plan §6, T5).

Targets: human-labeled ``known`` sightings of one facility, day by day. Two
reference modes:
- ``temporal`` (production-like): references are labels on EARLIER days that
  existed by the end of the probe day + 2.5 h (when the nightly run would start).
- ``lodo`` (leave one day out): references are labels on every OTHER day; the
  most references possible, for comparing models, not for predicting production.
Same-day human labels are never used as seeds here: they are the targets.

Open set: leave-one-person-out. Each person's references are removed in turn and
any ``known`` label on that person's targets counts as a false accept.
"""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .decide import Thresholds, decide_day, face_scores
from .metrics import Tally, bootstrap_precision, outcome
from .pipeline import RefPolicy, Workspace, org_facilities

QUICK_GRID = {
    "face_sim": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
    "face_margin": [0.05, 0.10, 0.15, 0.20],
    "face_q_min": [0.0, 0.15, 0.30],
    "body_enabled": [False, True],
}


def load_grid(spec: str | None) -> dict[str, list]:
    if not spec or spec == "quick":
        return dict(QUICK_GRID)
    return json.loads(Path(spec).read_text(encoding="utf-8"))


def combos(grid: dict[str, list], base: Thresholds) -> list[Thresholds]:
    keys = sorted(grid)
    return [replace(base, **dict(zip(keys, vals))) for vals in itertools.product(*(grid[k] for k in keys))]


def run_cutoff_ms(day: str, tz: str, hour: float = 26.5) -> int:
    """End of ``day`` + 2.5 h, local time, in epoch ms (the nightly run's start)."""
    d = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo(tz))
    return int((d.timestamp() + hour * 3600) * 1000)


@dataclass
class ComboResult:
    th: Thresholds
    total: Tally = field(default_factory=Tally)
    by_state: dict[str, Tally] = field(default_factory=dict)
    by_via: dict[str, Tally] = field(default_factory=dict)
    by_face: dict[str, Tally] = field(default_factory=dict)
    per_day: dict[str, Tally] = field(default_factory=dict)
    loo: Tally = field(default_factory=Tally)
    unlabeled_promotions: int = 0
    unlabeled_promotions_body: int = 0
    unlabeled_pool: int = 0


def _bucket(d: dict[str, Tally], k: str) -> Tally:
    if k not in d:
        d[k] = Tally()
    return d[k]


def evaluate_facility(ws: Workspace, facility: str, *, refs_mode: str = "temporal",
                      grid: dict[str, list] | None = None, base: Thresholds = Thresholds(),
                      policy: RefPolicy = RefPolicy(), days: list[str] | None = None,
                      loo: bool = True, log=print) -> list[ComboResult]:
    ex = ws.export
    facs = org_facilities(ex, facility)
    roster = ex.roster(facility)
    tz = ex.facility_tz.get(facility, "Asia/Shanghai")
    results = [ComboResult(th) for th in combos(grid or QUICK_GRID, base)]
    all_days = days or ex.days(facility)
    for day in all_days:
        sightings = ex.by_facility_day(facility, day)
        targets = {s.sighting_id: ex.key(s.human_person_id) for s in sightings if s.human_state == "known"}
        unlabeled_pool = {s.sighting_id for s in sightings
                          if s.human_state is None and s.board_state in ("uncertain", "stranger")}
        if refs_mode == "temporal":
            refs = ws.references(facs, day, label_cutoff_ms=run_cutoff_ms(day, tz), policy=policy)
        elif refs_mode == "lodo":
            refs = ws.references(facs, day, label_cutoff_ms=None, policy=policy,
                                 exclude_days={day}, allow_after=True)
        else:
            raise ValueError(refs_mode)
        probe_faces, body, body_crops = ws.probe_inputs(facility, day)
        log(f"[eval] {facility} {day}: {len(sightings)} sightings, {len(targets)} targets, "
            f"refs {{{', '.join(f'{k[-6:]}:{v.shape[0]}' for k, v in sorted(refs.items()))}}}, "
            f"faces {len(probe_faces)}, bodies {len(body)}")
        if not refs:
            for r in results:
                for sid, truth in targets.items():
                    s = ws.sighting(sid)
                    for bucket in (r.total, _bucket(r.per_day, day), _bucket(r.by_state, s.board_state),
                                   _bucket(r.by_face, "face" if sid in probe_faces else "noface")):
                        bucket.add("abstain")
                r.unlabeled_pool += len(unlabeled_pool)
            continue
        # face_scores depends on the quality floor, face size floor and top-k, which a grid may vary:
        # compute it once per distinct (face_q_min, face_min_px, topk).
        fs_cache: dict[tuple, object] = {}

        def fs_for(th: Thresholds, refs_used: dict, tag: str):
            key = (tag, th.face_q_min, th.face_min_px, th.topk)
            if key not in fs_cache:
                fs_cache[key] = face_scores(probe_faces, refs_used, th)
            return fs_cache[key]

        loo_sets = []
        if loo:
            for k in sorted(refs):
                k_targets = {sid for sid, t in targets.items() if t == k}
                if not k_targets:
                    continue
                refs_k = {p: v for p, v in refs.items() if p != k}
                if refs_k:
                    loo_sets.append((k, k_targets, refs_k))
        for r in results:
            fs = fs_for(r.th, refs, "all")
            labels = {l.sighting_id: l for l in decide_day(sightings, fs, body, body_crops, roster, ex.key, r.th)}
            for sid, truth in targets.items():
                s = ws.sighting(sid)
                lab = labels.get(sid)
                o = outcome(lab, truth)
                for bucket in (r.total, _bucket(r.per_day, day), _bucket(r.by_state, s.board_state),
                               _bucket(r.by_face, "face" if sid in probe_faces else "noface")):
                    bucket.add(o, sid)
                if lab is not None and lab.state == "known":
                    _bucket(r.by_via, lab.via).add(o, sid)
            r.unlabeled_pool += len(unlabeled_pool)
            for sid in unlabeled_pool:
                lab = labels.get(sid)
                if lab is not None and lab.state == "known" and lab.relation == "promote":
                    r.unlabeled_promotions += 1
                    if lab.via == "body":
                        r.unlabeled_promotions_body += 1
            for k, k_targets, refs_k in loo_sets:
                fs_k = fs_for(r.th, refs_k, f"loo:{k}")
                lab_k = {l.sighting_id: l for l in decide_day(sightings, fs_k, body, body_crops, roster, ex.key, r.th)}
                for sid in k_targets:
                    lab = lab_k.get(sid)
                    r.loo.add("wrong" if (lab is not None and lab.state == "known") else "abstain", sid)
    return results


CURVE_FIELDS = [
    "setup", "refs_mode", "facility", "face_sim", "face_margin", "face_q_min", "body_enabled", "body_link",
    "n_targets", "correct", "wrong", "flagged", "abstain", "precision", "precision_lo95_days", "coverage",
    "promote_n", "promote_precision", "promote_coverage",
    "face_precision", "body_precision", "body_labeled",
    "noface_n", "noface_coverage", "noface_precision",
    "loo_n", "loo_false_accepts", "loo_fa_rate",
    "unlabeled_pool", "unlabeled_promotions", "unlabeled_promotions_body",
]


def _f(x):
    return "" if x is None else (round(x, 5) if isinstance(x, float) else x)


def curve_rows(results: list[ComboResult], setup: str, refs_mode: str, facility: str) -> list[dict]:
    rows = []
    for r in results:
        promote = Tally()
        for st in ("uncertain", "stranger"):
            if st in r.by_state:
                promote.merge(r.by_state[st])
        nf = r.by_face.get("noface", Tally())
        rows.append({
            "setup": setup, "refs_mode": refs_mode, "facility": facility,
            "face_sim": r.th.face_sim, "face_margin": r.th.face_margin, "face_q_min": r.th.face_q_min,
            "body_enabled": int(r.th.body_enabled), "body_link": r.th.body_link,
            "n_targets": r.total.n, "correct": r.total.correct, "wrong": r.total.wrong,
            "flagged": r.total.flagged, "abstain": r.total.abstain,
            "precision": _f(r.total.precision), "precision_lo95_days": _f(bootstrap_precision(r.per_day)),
            "coverage": _f(r.total.coverage),
            "promote_n": promote.n, "promote_precision": _f(promote.precision), "promote_coverage": _f(promote.coverage),
            "face_precision": _f(r.by_via.get("face", Tally()).precision),
            "body_precision": _f(r.by_via.get("body", Tally()).precision),
            "body_labeled": r.by_via.get("body", Tally()).labeled,
            "noface_n": nf.n, "noface_coverage": _f(nf.coverage), "noface_precision": _f(nf.precision),
            "loo_n": r.loo.n, "loo_false_accepts": r.loo.wrong,
            "loo_fa_rate": _f(r.loo.wrong / r.loo.n if r.loo.n else None),
            "unlabeled_pool": r.unlabeled_pool, "unlabeled_promotions": r.unlabeled_promotions,
            "unlabeled_promotions_body": r.unlabeled_promotions_body,
        })
    return rows


def write_curve(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CURVE_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def operating_point(rows: list[dict], min_precision_lo: float, max_loo_fa: float) -> dict | None:
    """Highest coverage among rows whose day-bootstrap lower bound and leave-one-out
    false-accept rate both pass."""
    ok = [r for r in rows
          if r["precision_lo95_days"] != "" and float(r["precision_lo95_days"]) >= min_precision_lo
          and (r["loo_fa_rate"] == "" or float(r["loo_fa_rate"]) <= max_loo_fa)]
    if not ok:
        return None
    return max(ok, key=lambda r: (float(r["coverage"] or 0), r["unlabeled_promotions"]))


def summarize(rows: list[dict], point: dict | None) -> str:
    lines = [f"rows: {len(rows)}"]
    if point is None:
        lines.append("no threshold pair meets the precision / false-accept targets")
    else:
        lines.append("operating point: " + ", ".join(f"{k}={point[k]}" for k in (
            "face_sim", "face_margin", "face_q_min", "body_enabled", "precision", "precision_lo95_days", "coverage",
            "promote_precision", "promote_coverage", "noface_coverage", "loo_fa_rate", "unlabeled_promotions")))
    return "\n".join(lines)


def per_sighting_wrongs(results: list[ComboResult], point: dict | None) -> list[str]:
    if point is None:
        return []
    for r in results:
        if (r.th.face_sim == point["face_sim"] and r.th.face_margin == point["face_margin"]
                and r.th.face_q_min == point["face_q_min"] and int(r.th.body_enabled) == int(point["body_enabled"])):
            return r.total.wrong_ids
    return []


__all__ = ["evaluate_facility", "curve_rows", "write_curve", "operating_point", "summarize",
           "load_grid", "run_cutoff_ms", "per_sighting_wrongs"]
