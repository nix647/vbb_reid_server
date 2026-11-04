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
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .decide import BodyIndex, Thresholds, decide_day, face_scores
from .metrics import Tally, bootstrap_precision, cp_lower, cp_upper, outcome
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
    grid = {k: v for k, v in json.loads(Path(spec).read_text(encoding="utf-8")).items() if not k.startswith("_")}
    known = {f.name for f in fields(Thresholds)}
    unknown = sorted(set(grid) - known)
    if unknown:
        raise ValueError(f"unknown grid key(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
    return grid


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
    promote_per_day: dict[str, Tally] = field(default_factory=dict)
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
        body_index = BodyIndex.build(sightings, body) if body else None
        log(f"[eval] {facility} {day}: {len(sightings)} sightings, {len(targets)} targets, "
            f"refs {{{', '.join(f'{k[-6:]}:{v.shape[0]}' for k, v in sorted(refs.items()))}}}, "
            f"faces {len(probe_faces)}, bodies {len(body)}")
        if not refs:
            for r in results:
                for sid, truth in targets.items():
                    s = ws.sighting(sid)
                    buckets = [r.total, _bucket(r.per_day, day), _bucket(r.by_state, s.board_state),
                               _bucket(r.by_face, "face" if sid in probe_faces else "noface")]
                    if s.board_state in ("uncertain", "stranger"):
                        buckets.append(_bucket(r.promote_per_day, day))
                    for bucket in buckets:
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
                refs_k = refs.without(k)
                if refs_k:
                    loo_sets.append((k, k_targets, refs_k))
        for r in results:
            fs = fs_for(r.th, refs, "all")
            labels = {l.sighting_id: l for l in decide_day(sightings, fs, body, body_crops, roster, ex.key, r.th,
                                                           body_index=body_index)}
            for sid, truth in targets.items():
                s = ws.sighting(sid)
                lab = labels.get(sid)
                o = outcome(lab, truth)
                buckets = [r.total, _bucket(r.per_day, day), _bucket(r.by_state, s.board_state),
                           _bucket(r.by_face, "face" if sid in probe_faces else "noface")]
                if s.board_state in ("uncertain", "stranger"):
                    buckets.append(_bucket(r.promote_per_day, day))
                for bucket in buckets:
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
                lab_k = {l.sighting_id: l for l in decide_day(sightings, fs_k, body, body_crops, roster, ex.key, r.th,
                                                              body_index=body_index)}
                for sid in k_targets:
                    lab = lab_k.get(sid)
                    r.loo.add("wrong" if (lab is not None and lab.state == "known") else "abstain", sid)
    return results


CURVE_FIELDS = [
    "setup", "refs_mode", "facility", "combo", "face_sim", "face_margin", "face_q_min", "body_enabled", "body_link",
    "n_targets", "correct", "wrong", "flagged", "abstain", "precision", "precision_lo95_days", "precision_cp_lo",
    "coverage",
    "promote_n", "promote_labeled", "promote_precision", "promote_lo95_days", "promote_cp_lo", "promote_coverage",
    "face_precision", "body_precision", "body_labeled",
    "noface_n", "noface_coverage", "noface_precision",
    "loo_n", "loo_false_accepts", "loo_fa_rate", "loo_fa_cp_hi",
    "unlabeled_pool", "unlabeled_promotions", "unlabeled_promotions_body",
    "thresholds",
]


def _f(x):
    return "" if x is None else (round(x, 5) if isinstance(x, float) else x)


def curve_rows(results: list[ComboResult], setup: str, refs_mode: str, facility: str) -> list[dict]:
    rows = []
    for i, r in enumerate(results):
        promote = Tally()
        for st in ("uncertain", "stranger"):
            if st in r.by_state:
                promote.merge(r.by_state[st])
        nf = r.by_face.get("noface", Tally())
        rows.append({
            "setup": setup, "refs_mode": refs_mode, "facility": facility, "combo": i,
            "face_sim": r.th.face_sim, "face_margin": r.th.face_margin, "face_q_min": r.th.face_q_min,
            "body_enabled": int(r.th.body_enabled), "body_link": r.th.body_link,
            "n_targets": r.total.n, "correct": r.total.correct, "wrong": r.total.wrong,
            "flagged": r.total.flagged, "abstain": r.total.abstain,
            "precision": _f(r.total.precision), "precision_lo95_days": _f(bootstrap_precision(r.per_day)),
            "precision_cp_lo": _f(cp_lower(r.total.correct, r.total.labeled)),
            "coverage": _f(r.total.coverage),
            "promote_n": promote.n, "promote_labeled": promote.labeled,
            "promote_precision": _f(promote.precision),
            "promote_lo95_days": _f(bootstrap_precision(r.promote_per_day)),
            "promote_cp_lo": _f(cp_lower(promote.correct, promote.labeled)),
            "promote_coverage": _f(promote.coverage),
            "face_precision": _f(r.by_via.get("face", Tally()).precision),
            "body_precision": _f(r.by_via.get("body", Tally()).precision),
            "body_labeled": r.by_via.get("body", Tally()).labeled,
            "noface_n": nf.n, "noface_coverage": _f(nf.coverage), "noface_precision": _f(nf.precision),
            "loo_n": r.loo.n, "loo_false_accepts": r.loo.wrong,
            "loo_fa_rate": _f(r.loo.wrong / r.loo.n if r.loo.n else None),
            "loo_fa_cp_hi": _f(cp_upper(r.loo.wrong, r.loo.n)),
            "unlabeled_pool": r.unlabeled_pool, "unlabeled_promotions": r.unlabeled_promotions,
            "unlabeled_promotions_body": r.unlabeled_promotions_body,
            "thresholds": json.dumps(r.th.as_dict(), sort_keys=True),
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


def _num(x) -> float | None:
    return None if x in ("", None) else float(x)


def gate_reasons(row: dict, *, min_precision_lo: float, max_loo_fa: float, min_labeled: int,
                 gate: str = "promote") -> list[str]:
    """Why a curve row fails the gate (empty = passes). The gate is on PROMOTIONS by default —
    the only machine labels v1 ever shows. Precision must clear ``min_precision_lo`` on BOTH the
    day bootstrap (labels are clustered) and the exact binomial bound (few labels are weak
    evidence); leave-one-person-out must have been measured and its false-accept upper bound
    must stay under ``max_loo_fa``."""
    pre = "promote_" if gate == "promote" else "precision_"
    labeled = row["promote_labeled"] if gate == "promote" else row["correct"] + row["wrong"]
    days = _num(row[f"{pre}lo95_days"] if gate == "promote" else row["precision_lo95_days"])
    cp = _num(row[f"{pre}cp_lo"] if gate == "promote" else row["precision_cp_lo"])
    out = []
    if labeled < min_labeled:
        out.append(f"only {labeled} labels (< {min_labeled})")
    if days is None or days < min_precision_lo:
        out.append(f"day-bootstrap lower bound {days} < {min_precision_lo}")
    if cp is None or cp < min_precision_lo:
        out.append(f"binomial lower bound {cp} < {min_precision_lo}")
    fa_hi = _num(row["loo_fa_cp_hi"])
    if not row["loo_n"]:
        out.append("leave-one-person-out not measured")
    elif fa_hi is None or fa_hi > max_loo_fa:
        out.append(f"false-accept upper bound {fa_hi} > {max_loo_fa}")
    return out


def needed_trials(events: int, n: int, *, rate_bound: float, alpha: float = 0.025, upper: bool) -> int | None:
    """Smallest total N (>= n) that would pass at the CURRENT event rate.

    upper=False: ``events`` are successes and the binomial LOWER bound must reach ``rate_bound``
    (precision). upper=True: ``events`` are failures and the UPPER bound must stay under it
    (false accepts). None when the current rate itself is on the wrong side (more data at the
    same rate never passes)."""
    rate = events / n if n else (0.0 if upper else 1.0)
    if (upper and rate >= rate_bound) or (not upper and rate <= rate_bound):
        return None

    def ok(N: int) -> bool:
        k = round(rate * N)
        return (cp_upper(k, N, alpha) or 1.0) <= rate_bound if upper else (cp_lower(k, N, alpha) or 0.0) >= rate_bound

    lo, hi = max(n, 1), max(n, 1)
    while not ok(hi):
        hi *= 2
        if hi > 10_000_000:
            return None
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return hi


def verdict(row: dict, *, min_precision_lo: float, max_loo_fa: float, min_labeled: int,
            gate: str = "promote") -> tuple[str, list[str], dict]:
    """``pass`` | ``fail`` | ``insufficient``.

    - fail: the data show precision below the target (binomial UPPER bound under it) or false
      accepts above the ceiling (LOWER bound over it) — more labels will not rescue this row.
    - insufficient: nothing contradicts the targets, but the sample is too small to prove them.
      ``needs`` says how many labeled promotions / leave-one-out trials would pass at the
      current error rates. An insufficient row never enables display.
    """
    reasons = gate_reasons(row, min_precision_lo=min_precision_lo, max_loo_fa=max_loo_fa,
                           min_labeled=min_labeled, gate=gate)
    if not reasons:
        return "pass", [], {}
    labeled = row["promote_labeled"] if gate == "promote" else row["correct"] + row["wrong"]
    if gate == "promote":
        # promote correct = labeled * precision (exact integers are not in the row; recover them)
        p = _num(row["promote_precision"])
        correct = round(labeled * p) if p is not None else 0
    else:
        correct = row["correct"]
    loo_n, loo_fa = int(row["loo_n"] or 0), int(row["loo_false_accepts"] or 0)
    fail = []
    if labeled and (cp_upper(correct, labeled) or 1.0) < min_precision_lo:
        fail.append(f"precision demonstrably below {min_precision_lo} ({labeled - correct} wrong of {labeled})")
    if loo_n and (cp_lower(loo_fa, loo_n) or 0.0) > max_loo_fa:
        fail.append(f"false accepts demonstrably above {max_loo_fa} ({loo_fa} of {loo_n})")
    if fail:
        return "fail", fail + reasons, {}
    needs = {}
    n_prec = needed_trials(correct, labeled, rate_bound=min_precision_lo, upper=False)
    needs["labeled_promotions"] = max(min_labeled, n_prec) if n_prec is not None else None
    needs["loo_trials"] = needed_trials(loo_fa, loo_n, rate_bound=max_loo_fa, upper=True)
    return "insufficient", reasons, needs


def operating_point(rows: list[dict], min_precision_lo: float, max_loo_fa: float, *, min_labeled: int = 50,
                    gate: str = "promote") -> dict | None:
    """Highest coverage among rows that pass :func:`gate_reasons`. Note: the chosen row's own
    numbers are measured on the data used to choose it (optimistic); the independent check is
    the adjudication set scored with ``shadow`` + ``adjudicate-score``."""
    ok = [r for r in rows if not gate_reasons(r, min_precision_lo=min_precision_lo, max_loo_fa=max_loo_fa,
                                               min_labeled=min_labeled, gate=gate)]
    if not ok:
        return None
    cov = "promote_coverage" if gate == "promote" else "coverage"
    return max(ok, key=lambda r: (_num(r[cov]) or 0.0, r["unlabeled_promotions"]))


def summarize(rows: list[dict], point: dict | None) -> str:
    lines = [f"rows: {len(rows)}"]
    if point is None:
        lines.append("no grid row passes the gate")
    else:
        lines.append("operating point: " + ", ".join(f"{k}={point[k]}" for k in (
            "combo", "face_sim", "face_margin", "face_q_min", "body_enabled",
            "promote_labeled", "promote_precision", "promote_lo95_days", "promote_cp_lo", "promote_coverage",
            "noface_coverage", "loo_n", "loo_fa_cp_hi", "unlabeled_promotions")))
        lines.append("(chosen on the same labels it is scored on: optimistic; confirm on the adjudication set)")
    return "\n".join(lines)


def per_sighting_wrongs(results: list[ComboResult], point: dict | None) -> list[str]:
    if point is None:
        return []
    return results[int(point["combo"])].total.wrong_ids


def write_point(point: dict, path: Path, *, ref_policy, quality) -> None:
    """The chosen operating point, in the form ``shadow --point`` reads back."""
    from dataclasses import asdict
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "thresholds": json.loads(point["thresholds"]),
        "ref_policy": {k: v for k, v in asdict(ref_policy).items() if k != "include_same_day"},
        "quality": quality.as_dict(),
        "chosen_from": {k: point[k] for k in ("setup", "refs_mode", "facility", "combo", "promote_labeled",
                                               "promote_precision", "promote_cp_lo", "promote_lo95_days",
                                               "promote_coverage", "loo_n", "loo_fa_cp_hi")},
    }, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["verdict", "needed_trials", "evaluate_facility", "curve_rows", "write_curve", "operating_point", "summarize",
           "load_grid", "run_cutoff_ms", "per_sighting_wrongs", "gate_reasons", "write_point"]
