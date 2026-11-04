"""Which quality measures actually predict a reliable face match — measured on human labels.

Pairs of human-labeled faces (same identity key = genuine, different = impostor) are binned
by each quality measure, taking the WORSE of the two faces (a pair is only as good as its
weaker face). For each bin:

  genuine / impostor median similarity, and TAR — the share of genuine pairs above the global
  impostor threshold (the impostor score at ``far``, 1e-3 by default). A measure is useful
  when TAR falls in its bad bins and stays high in its good ones; a measure whose bins all
  look alike should not be in the quality policy.

The same table for the policy's ``qf`` shows whether the policy as a whole separates easy
from hard faces, and where a cutoff would land.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .pipeline import Workspace

# measure -> (column, worse-of-pair reducer, take absolute value)
MEASURES = {
    "qf": ("qf", np.minimum, False),
    "det": ("det", np.minimum, False),
    "face_h": ("face_h", np.minimum, False),
    "blur": ("blur", np.minimum, False),
    "residual": ("residual", np.maximum, False),
    "asym": ("asym", np.maximum, False),
    "abs_yaw": ("yaw", np.maximum, True),
    "norm": ("norm", np.minimum, False),
    "second_ratio": ("second_ratio", np.maximum, False),
    "n_faces": ("n_faces", np.maximum, False),
}


def labeled_faces(ws: Workspace, facility: str, days: list[str], max_faces: int, seed: int = 0):
    ex = ws.export
    keys, vecs, cols = [], [], {m: [] for m in ("qf", "det", "face_h", "blur", "residual", "asym", "yaw", "norm",
                                                 "second_ratio", "n_faces")}
    label = {s.sighting_id: ex.key(s.human_person_id) for s in ex.sightings
             if s.facility_id == facility and s.human_state == "known" and s.day in days}
    for day in days:
        ft = ws.face_table(facility, day)
        for i in range(len(ft)):
            k = label.get(str(ft.sighting_id[i]))
            if k is None:
                continue
            keys.append(k)
            vecs.append(ft.vec[i])
            for m in cols:
                cols[m].append(float(getattr(ft, m)[i]))
    if not keys:
        return np.zeros(0, dtype=object), np.zeros((0, 0), np.float32), {m: np.zeros(0) for m in cols}
    idx = np.arange(len(keys))
    if len(idx) > max_faces:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_faces, replace=False))
    keys_a = np.array(keys, dtype=object)[idx]
    V = np.stack(vecs)[idx].astype(np.float32)
    return keys_a, V, {m: np.asarray(v)[idx] for m, v in cols.items()}


def report(ws: Workspace, facility: str, days: list[str], *, bins: int = 5, far: float = 1e-3,
           max_faces: int = 4000, out_csv: Path | None = None, log=print) -> list[dict]:
    keys, V, cols = labeled_faces(ws, facility, days, max_faces)
    n = len(keys)
    if n < 2:
        log("[quality] fewer than two labeled faces; nothing to measure")
        return []
    iu, ju = np.triu_indices(n, k=1)
    sims = (V @ V.T)[iu, ju]  # one (n, n) product; never gather (pairs, D) rows
    same = keys[iu] == keys[ju]
    imp = np.sort(sims[~same])
    thr = float(imp[min(len(imp) - 1, int(np.ceil((1.0 - far) * len(imp))))]) if len(imp) else 1.0
    log(f"[quality] facility {facility}, {len(days)} day(s): {n} labeled faces, "
        f"{int(same.sum())} genuine / {int((~same).sum())} impostor pairs; impostor threshold at FAR {far:g} = {thr:.3f}")
    rows: list[dict] = []
    for name, (col, reduce, absval) in MEASURES.items():
        x = cols[col]
        if absval:
            x = np.abs(x)
        if not np.isfinite(x).all():  # e.g. norm from a model that does not expose it
            continue
        pair_x = reduce(x[iu], x[ju])
        values = np.unique(pair_x)
        if len(values) <= bins:  # few distinct values: one bin per value
            which = np.searchsorted(values, pair_x)
            ranges = [(v, v) for v in values]
        else:
            edges = np.unique(np.quantile(pair_x, np.linspace(0, 1, bins + 1)))
            which = np.clip(np.searchsorted(edges, pair_x, side="right") - 1, 0, len(edges) - 2)
            ranges = [(edges[b], edges[b + 1]) for b in range(len(edges) - 1)]
        for b, (lo, hi) in enumerate(ranges):
            m = which == b
            g, i_ = sims[m & same], sims[m & ~same]
            rows.append({
                "measure": name, "bin": b, "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                "genuine_pairs": int(g.size), "impostor_pairs": int(i_.size),
                "genuine_median": round(float(np.median(g)), 4) if g.size else "",
                "impostor_median": round(float(np.median(i_)), 4) if i_.size else "",
                "tar_at_far": round(float((g >= thr).mean()), 4) if g.size else "",
                "far_in_bin": round(float((i_ >= thr).mean()), 5) if i_.size else "",
            })
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    cur = None
    for r in rows:
        if r["measure"] != cur:
            cur = r["measure"]
            log(f"\n  {cur:9s} {'range':>17s} {'gen':>8s} {'imp':>8s} {'gen_med':>8s} {'imp_med':>8s} {'TAR':>6s} {'FAR':>8s}")
        log(f"  {'':9s} {r['lo']:>8.3g}-{r['hi']:<8.3g} {r['genuine_pairs']:8d} {r['impostor_pairs']:8d} "
            f"{r['genuine_median']!s:>8s} {r['impostor_median']!s:>8s} {r['tar_at_far']!s:>6s} {r['far_in_bin']!s:>8s}")
    return rows
