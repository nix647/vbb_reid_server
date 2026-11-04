"""Command line: ``vbb-reid-server <command> ...`` (or ``python -m vbb_reid_server``).

  summary            what the export contains (per facility / day / person)
  embed              crops -> embedding cache, for one model set
  eval               threshold curves on human labels (+ leave-one-person-out)
  shadow             production-shaped labels for facility-days, written to files
  adjudicate-sample  draw the stratified adjudication sample
  adjudicate-page    build the HTML page for a sample
  adjudicate-score   score a shadow run against exported answers
  prune              delete cached crops / embeddings older than N days
  quality-report     which face-quality measures predict a reliable match (on human labels)
  inspect            every measure, reference status and face score of given sightings
  doctor             check GPU, ONNX Runtime providers, torch, and (optionally) load a model set
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

from .data import load_export
from .decide import Thresholds
from .pipeline import RefPolicy, Workspace
from .quality import QualityPolicy


def strict_replace(obj, spec: dict, what: str):
    """dataclasses.replace that refuses unknown keys (a typo must not silently mean 'default').
    Keys starting with '_' are comments."""
    known = {f.name for f in fields(obj)}
    unknown = sorted(k for k in spec if k not in known and not k.startswith("_"))
    if unknown:
        raise SystemExit(f"unknown {what} key(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
    return replace(obj, **{k: v for k, v in spec.items() if k in known})


def _thresholds(args) -> Thresholds:
    if not getattr(args, "thresholds", None):
        return Thresholds()
    spec = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    return strict_replace(Thresholds(), spec, "Thresholds")


def _ids(models_cfg: str | None):
    """Model ids from a model config, without loading heavy weights twice."""
    if not models_cfg:
        return None, None, None
    from .models import load_model_set
    ms = load_model_set(models_cfg)
    return ms, ms.face_id, ms.body_id


def _ids_from_cache(emb_root: str, kind: str, name: str | None) -> str | None:
    """Pick a cached model id (eval/shadow need no GPU): an exact slug or exact id wins;
    otherwise the substring must match exactly ONE cached model."""
    if not name:
        return None
    base = Path(emb_root) / kind
    cached = [(meta.parent.name, json.loads(meta.read_text())["model_id"]) for meta in sorted(base.glob("*/model.json"))]
    exact = [mid for sl, mid in cached if name in (sl, mid)]
    if exact:
        return exact[0]
    hits = [(sl, mid) for sl, mid in cached if name in mid]
    if len(hits) == 1:
        return hits[0][1]
    if not hits:
        raise SystemExit(f"no cached {kind} model matching {name!r} under {base}")
    listing = "\n".join(f"  {sl}  {mid}" for sl, mid in hits)
    raise SystemExit(f"{name!r} matches {len(hits)} cached {kind} models; pass a slug or a longer substring:\n{listing}")


def cmd_summary(args) -> None:
    ex = load_export(args.export, args.aliases)
    for fac in sorted({s.facility_id for s in ex.sightings}):
        ss = [s for s in ex.sightings if s.facility_id == fac]
        print(f"facility {fac} (org {ex.facility_org.get(fac)}): {len(ss)} sightings, "
              f"days {ex.days(fac)[0]}..{ex.days(fac)[-1]}, roster {sorted(ex.roster(fac))}")
        lab = Counter(ex.key(s.human_person_id) for s in ss if s.human_state == "known")
        print(f"   human-labeled known per identity: {dict(lab)}")
        print(f"   board states: {dict(Counter(s.board_state for s in ss))}")


def cmd_embed(args) -> None:
    from .crops import CropStore
    from .embed import embed_day
    ex = load_export(args.export, args.aliases)
    ms, _, _ = _ids(args.models)
    store = CropStore(args.lan_crops, args.cdn_cache, fetch=not args.no_fetch)
    days = args.days.split(",") if args.days else None
    for fac in args.facility:
        for day in ex.days(fac):
            if days and day not in days:
                continue
            embed_day(ex.by_facility_day(fac, day), store, ms, args.emb, fac, day, refresh=args.refresh)
    print(f"[embed] crop sources: {store.stats}")
    print(f"[embed] face model: {ms.face_id}\n[embed] body model: {ms.body_id}")


def cmd_eval(args) -> None:
    from .evaluate import (curve_rows, evaluate_facility, load_grid, operating_point, per_sighting_wrongs,
                           summarize, verdict, write_curve, write_point)
    ex = load_export(args.export, args.aliases)
    face_id = _ids_from_cache(args.emb, "face", args.face_model)
    body_id = _ids_from_cache(args.emb, "body", args.body_model)
    quality = QualityPolicy.from_file(args.quality)
    ws = Workspace(ex, args.emb, face_id, body_id, quality)
    base = _thresholds(args)
    policy = RefPolicy(window_days=args.window_days, ref_q_min=args.ref_q_min, max_refs=args.max_refs,
                       ref_min_norm=args.ref_min_norm, ref_two_face_ratio=args.ref_two_face_ratio)
    grid = load_grid(args.grid)
    if not body_id:
        grid["body_enabled"] = [False]
    out = Path(args.out)
    setup = args.setup or f"{(face_id or '-').split('|')[-1]} + {body_id or '-'}"
    for fac in args.facility:
        for mode in args.refs_mode:
            results = evaluate_facility(ws, fac, refs_mode=mode, grid=grid, base=base, policy=policy,
                                        loo=not args.no_loo)
            rows = curve_rows(results, setup, mode, fac)
            write_curve(rows, out / "curve.csv")
            gate = dict(min_precision_lo=args.min_precision_lo, max_loo_fa=args.max_loo_fa,
                        min_labeled=args.min_labeled, gate=args.gate)
            point = operating_point(rows, **gate)
            text = summarize(rows, point)
            if point is None and rows:
                verdicts = [(r, *verdict(r, **gate)) for r in rows]
                insuff = [(r, why, nd) for r, v, why, nd in verdicts if v == "insufficient"]
                if insuff:
                    def cost(t):
                        nd = t[2]
                        lab = nd.get("labeled_promotions")
                        loo = nd.get("loo_trials")
                        need = max((lab or 10**9) - t[0]["promote_labeled"], (loo or 10**9) - int(t[0]["loo_n"] or 0), 0)
                        return (need, -(float(t[0]["promote_coverage"] or 0)))
                    r, why, nd = min(insuff, key=cost)
                    text += (f"\nVERDICT: NOT ENOUGH EVIDENCE YET ({len(insuff)} of {len(rows)} rows; none contradicts the "
                             f"targets). Closest: combo {r['combo']} (face_sim={r['face_sim']}, margin={r['face_margin']}, "
                             f"face_q_min={r['face_q_min']}): {r['promote_labeled']} labeled promotions at precision "
                             f"{r['promote_precision']}, coverage {r['promote_coverage']}, "
                             f"loo {r['loo_false_accepts']}/{r['loo_n']}."
                             f"\n  needs about {nd.get('labeled_promotions')} labeled promotions and "
                             f"{nd.get('loo_trials')} leave-one-out trials at the current error rates "
                             f"(None = the current rate itself is on the wrong side)\n  why: " + "; ".join(why))
                else:
                    text += "\nVERDICT: FAIL - every row is contradicted by the data:\n" + "\n".join(
                        f"  combo {r['combo']}: " + "; ".join(why) for r, v, why, nd in
                        sorted(verdicts, key=lambda t: -(float(t[0]["promote_coverage"] or 0)))[:3])
            else:
                text += "\nVERDICT: PASS"
            print(f"[eval] facility {fac} refs={mode}\n{text}")
            with (out / "report.md").open("a", encoding="utf-8") as fh:
                fh.write(f"## {setup} · facility {fac} · refs {mode}\n\n```\n{text}\n```\n")
                wrongs = per_sighting_wrongs(results, point)
                if wrongs:
                    fh.write(f"Wrong at the operating point ({len(wrongs)}): {', '.join(wrongs[:50])}\n\n")
            if point is not None:
                pth = out / f"point_{fac}_{mode}.json"
                write_point(point, pth, ref_policy=policy, quality=quality)
                print(f"[eval] operating point written to {pth}")


def cmd_shadow(args) -> None:
    from .shadow import shadow_run
    ex = load_export(args.export, args.aliases)
    face_id = _ids_from_cache(args.emb, "face", args.face_model)
    body_id = _ids_from_cache(args.emb, "body", args.body_model)
    th, quality = _thresholds(args), QualityPolicy.from_file(args.quality)
    policy = RefPolicy(include_same_day=True)
    if args.point:
        pt = json.loads(Path(args.point).read_text(encoding="utf-8"))
        if args.thresholds or args.quality:
            raise SystemExit("--point already carries thresholds and quality; do not combine it with --thresholds/--quality")
        th = strict_replace(Thresholds(), pt["thresholds"], "Thresholds")
        quality = strict_replace(QualityPolicy(), pt.get("quality", {}), "QualityPolicy")
        policy = strict_replace(policy, pt.get("ref_policy", {}), "RefPolicy")
    for name in ("window_days", "ref_q_min", "max_refs", "ref_min_norm", "ref_two_face_ratio"):
        if getattr(args, name) is not None:
            policy = replace(policy, **{name: getattr(args, name)})
    ws = Workspace(ex, args.emb, face_id, body_id, quality)
    days = args.days.split(",") if args.days else ex.days(args.facility)
    cutoff = "run" if args.labels_until == "run" else None
    for day in days:
        shadow_run(ws, args.facility, day, th, args.out, policy=policy, label_cutoff_ms=cutoff,
                   model_name=args.setup or "")


def cmd_adj_sample(args) -> None:
    from .adjudicate import sample, write_sample
    ex = load_export(args.export, args.aliases)
    exclude = set()
    for p in args.exclude or []:
        with open(p, newline="", encoding="utf-8") as fh:
            exclude |= {r["sighting_id"] for r in csv.DictReader(fh)}
    days = args.days.split(",") if args.days else None
    only = None
    if args.named_by:
        named: set[str] = set()
        for p in args.named_by:
            with open(p, newline="", encoding="utf-8") as fh:
                named |= {r["sighting_id"] for r in csv.DictReader(fh)
                          if r.get("state") == "known" and str(r.get("human_labeled", "0")) in ("0", "")}
        only = named
    if args.norm_below is not None or args.norm_at_least is not None:
        if not (args.emb and args.face_model):
            raise SystemExit("--norm-below / --norm-at-least need --emb and --face-model")
        face_id = _ids_from_cache(args.emb, "face", args.face_model)
        ws = Workspace(ex, args.emb, face_id, None)
        lo = args.norm_at_least if args.norm_at_least is not None else float("-inf")
        hi = args.norm_below if args.norm_below is not None else float("inf")
        best: dict[str, float] = {}
        for d in (days or ex.days(args.facility)):
            ft = ws.face_table(args.facility, d)
            for sid, nrm in zip(ft.sighting_id, ft.norm):
                best[str(sid)] = max(best.get(str(sid), float("-inf")), float(nrm))
        band = {sid for sid, nrm in best.items() if lo <= nrm < hi}
        only = band if only is None else (only & band)
    rows = sample(ex, args.facility, args.n, seed=args.seed, days=days, exclude=exclude, only=only)
    write_sample(rows, Path(args.out))
    print(f"[adjudicate] {len(rows)} sightings -> {args.out}; strata {dict(Counter(r.board_state + '/' + ('face' if r.crop_is_face else 'noface') for r in rows))}")


def cmd_adj_page(args) -> None:
    from .adjudicate import read_sample, write_page
    ex = load_export(args.export, args.aliases)
    ids = read_sample(Path(args.sample))
    write_page(ex, args.facility, ids, Path(args.out), set_id=Path(args.sample).stem,
               lan_root=Path(args.lan_crops) if args.lan_crops else None)
    print(f"[adjudicate] page with {len(ids)} items -> {args.out}")


def cmd_adj_score(args) -> None:
    from .adjudicate import read_answers, score
    ex = load_export(args.export, args.aliases)
    answers = read_answers(Path(args.answers))
    labels: dict[str, dict] = {}
    for p in args.labels:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                labels[r["sighting_id"]] = r
    res = score(ex, answers, labels)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_prune(args) -> None:
    from .crops import CropStore
    from .embed import prune
    n_c = CropStore(None, args.cdn_cache, fetch=False).prune_cdn(args.days) if args.cdn_cache else 0
    n_e = prune(args.emb, args.days) if args.emb else 0
    print(f"[prune] removed {n_c} cached crops, {n_e} embedding files older than {args.days} days")


def cmd_quality_report(args) -> None:
    from .quality_report import report
    ex = load_export(args.export, args.aliases)
    face_id = _ids_from_cache(args.emb, "face", args.face_model)
    ws = Workspace(ex, args.emb, face_id, None, QualityPolicy.from_file(args.quality))
    days = args.days.split(",") if args.days else ex.days(args.facility)
    report(ws, args.facility, days, bins=args.bins, far=args.far, max_faces=args.max_faces,
           out_csv=Path(args.out) if args.out else None)


def cmd_inspect(args) -> None:
    from .inspect_tool import inspect
    ex = load_export(args.export, args.aliases)
    face_id = _ids_from_cache(args.emb, "face", args.face_model)
    th, quality, policy = _thresholds(args), QualityPolicy.from_file(args.quality), RefPolicy()
    if args.point:
        pt = json.loads(Path(args.point).read_text(encoding="utf-8"))
        th = strict_replace(Thresholds(), pt["thresholds"], "Thresholds")
        quality = strict_replace(QualityPolicy(), pt.get("quality", {}), "QualityPolicy")
        policy = strict_replace(policy, pt.get("ref_policy", {}), "RefPolicy")
    ws = Workspace(ex, args.emb, face_id, None, quality)
    inspect(ws, args.facility, [x.strip() for x in args.sightings.split(",") if x.strip()], th, policy,
            refs_mode=args.refs_mode)


def cmd_doctor(args) -> None:
    import platform
    print(f"python {platform.python_version()} on {platform.platform()}")
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as e:
                print(f"onnxruntime preload_dlls: {e}")
        prov = ort.get_available_providers()
        print(f"onnxruntime {ort.__version__}: providers {prov}")
        if "CUDAExecutionProvider" not in prov:
            print("  WARNING: no CUDAExecutionProvider - ONNX models will run on CPU (install onnxruntime-gpu)")
    except ImportError:
        print("onnxruntime: not installed (pip install -e '.[onnx]')")
    try:
        import torch
        print(f"torch {torch.__version__}: cuda available={torch.cuda.is_available()}"
              + (f", device {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    except ImportError:
        print("torch: not installed (only needed for hf_vision body models: pip install -e '.[hf]')")
    import os
    print(f"HF_HOME={os.environ.get('HF_HOME', '(unset: models download to ~/.cache/huggingface)')}")
    if args.models:
        from .models import load_model_set
        ms = load_model_set(args.models)
        for part in ("face_detector", "face", "body"):
            m = getattr(ms, part)
            if m is None:
                print(f"{part}: -")
                continue
            sess = getattr(m, "session", None)
            where = sess.get_providers()[0] if sess is not None else getattr(m, "device", "?")
            print(f"{part}: {m.model_id}  [{where}]")
        print(f"face model id: {ms.face_id}\nbody model id: {ms.body_id}")
        import numpy as np
        rng = np.random.default_rng(0)
        if ms.face is not None:
            faces = [(rng.random((112, 112, 3)) * 255).astype(np.uint8) for _ in range(5)]
            batched = ms.face.embed(faces)
            single = np.concatenate([ms.face.embed([f]) for f in faces])
            print(f"face batched vs one-by-one: max |diff| = {float(np.abs(batched - single).max()):.2e}, "
                  f"min cosine = {float((batched * single).sum(1).min()):.6f}  (fine: max |diff| below 1e-3 and min "
                  f"cosine >= 0.9999; GPU kernels differ by batch size, measured 1.7e-4 on gpu-ubun 2026-11-04)")
        if ms.body is not None:
            crops = [(rng.random((256, 128, 3)) * 255).astype(np.uint8) for _ in range(5)]
            batched = ms.body.embed(crops)
            single = np.concatenate([ms.body.embed([c]) for c in crops])
            print(f"body batched vs one-by-one: min cosine = {float((batched * single).sum(1).min()):.6f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vbb-reid-server", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, emb=True):
        sp.add_argument("--export", required=True, help="folder with the export CSVs")
        sp.add_argument("--aliases", help="person_id,human[,name] CSV (same human across facilities)")
        if emb:
            sp.add_argument("--emb", required=True, help="embedding cache root")

    sp = sub.add_parser("summary")
    common(sp, emb=False)
    sp.set_defaults(fn=cmd_summary)

    sp = sub.add_parser("embed")
    common(sp)
    sp.add_argument("--models", required=True, help="model config JSON")
    sp.add_argument("--facility", nargs="+", required=True)
    sp.add_argument("--days", help="comma list; default every day in the export")
    sp.add_argument("--lan-crops", help="root of rsynced board crops (<did>/<day>/<sid>/<idx>.jpg)")
    sp.add_argument("--cdn-cache", help="where CDN crops are cached")
    sp.add_argument("--no-fetch", action="store_true", help="never download; use caches only")
    sp.add_argument("--refresh", action="store_true", help="re-embed days already in the cache")
    sp.set_defaults(fn=cmd_embed)

    sp = sub.add_parser("eval")
    common(sp)
    sp.add_argument("--facility", nargs="+", required=True)
    sp.add_argument("--face-model", help="cached face model (slug or substring of its id)")
    sp.add_argument("--body-model", help="cached body model (slug or substring)")
    sp.add_argument("--setup", help="name for this setup in the curve file")
    sp.add_argument("--refs-mode", nargs="+", default=["temporal", "lodo"], choices=["temporal", "lodo"])
    sp.add_argument("--grid", help="'quick' or a JSON file {param: [values]}")
    sp.add_argument("--thresholds", help="JSON with base Thresholds overrides")
    sp.add_argument("--quality", help="JSON QualityPolicy overrides (applied when the cache is read)")
    sp.add_argument("--window-days", type=int, default=60)
    sp.add_argument("--ref-q-min", type=float, default=0.40)
    sp.add_argument("--max-refs", type=int, default=300)
    sp.add_argument("--ref-min-norm", type=float, default=0.0, help="reference faces need this raw feature norm")
    sp.add_argument("--ref-two-face-ratio", type=float, default=None,
                    help="skip reference crops with a second face at least this share of the chosen one")
    sp.add_argument("--min-precision-lo", type=float, default=0.99,
                    help="gate: lower bound (day bootstrap AND exact binomial) on precision")
    sp.add_argument("--max-loo-fa", type=float, default=0.005,
                    help="gate: upper bound on the leave-one-person-out false-accept rate")
    sp.add_argument("--min-labeled", type=int, default=50, help="gate: minimum labels the machine gave")
    sp.add_argument("--gate", choices=["promote", "all"], default="promote",
                    help="gate on promotions (what v1 shows) or on every labeled target")
    sp.add_argument("--no-loo", action="store_true")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("shadow")
    common(sp)
    sp.add_argument("--facility", required=True)
    sp.add_argument("--days")
    sp.add_argument("--face-model")
    sp.add_argument("--body-model")
    sp.add_argument("--point", help="point_<facility>_<mode>.json written by eval (thresholds, quality, reference policy)")
    sp.add_argument("--thresholds", help="JSON with Thresholds (instead of --point)")
    sp.add_argument("--quality", help="JSON QualityPolicy overrides (instead of --point)")
    sp.add_argument("--window-days", type=int)
    sp.add_argument("--ref-q-min", type=float)
    sp.add_argument("--max-refs", type=int)
    sp.add_argument("--ref-min-norm", type=float)
    sp.add_argument("--ref-two-face-ratio", type=float)
    sp.add_argument("--labels-until", choices=["run", "all"], default="run",
                    help="run (default): only labels that existed at that night's run; all: every label in the "
                         "export (debugging only - it leaks the future)")
    sp.add_argument("--setup")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_shadow)

    sp = sub.add_parser("adjudicate-sample")
    common(sp, emb=False)
    sp.add_argument("--facility", required=True)
    sp.add_argument("--n", type=int, default=300)
    sp.add_argument("--days")
    sp.add_argument("--seed", type=int, default=20261103)
    sp.add_argument("--exclude", nargs="*", help="earlier sample CSVs to skip")
    sp.add_argument("--named-by", nargs="*", help="only sightings these shadow labels CSVs named (state=known)")
    sp.add_argument("--norm-below", type=float, help="only sightings whose best face has a raw norm below this")
    sp.add_argument("--norm-at-least", type=float, help="only sightings whose best face has a raw norm at least this")
    sp.add_argument("--emb", help="embedding cache (for the norm filters)")
    sp.add_argument("--face-model", help="cached face model (for the norm filters)")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_adj_sample)

    sp = sub.add_parser("adjudicate-page")
    common(sp, emb=False)
    sp.add_argument("--facility", required=True)
    sp.add_argument("--sample", required=True)
    sp.add_argument("--lan-crops")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_adj_page)

    sp = sub.add_parser("adjudicate-score")
    common(sp, emb=False)
    sp.add_argument("--answers", required=True)
    sp.add_argument("--labels", nargs="+", required=True, help="shadow labels CSV(s)")
    sp.set_defaults(fn=cmd_adj_score)

    sp = sub.add_parser("quality-report")
    common(sp)
    sp.add_argument("--facility", required=True)
    sp.add_argument("--days", help="comma list; default every day in the export")
    sp.add_argument("--face-model", required=True, help="cached face model (slug or substring)")
    sp.add_argument("--quality", help="JSON QualityPolicy overrides for the qf rows")
    sp.add_argument("--bins", type=int, default=5)
    sp.add_argument("--far", type=float, default=1e-3)
    sp.add_argument("--max-faces", type=int, default=4000)
    sp.add_argument("--out", help="CSV of the table")
    sp.set_defaults(fn=cmd_quality_report)

    sp = sub.add_parser("inspect", help="every measure, reference status and face score of given sightings")
    common(sp)
    sp.add_argument("--facility", required=True)
    sp.add_argument("--sightings", required=True, help="comma list of sighting ids")
    sp.add_argument("--face-model", required=True)
    sp.add_argument("--point")
    sp.add_argument("--thresholds")
    sp.add_argument("--quality")
    sp.add_argument("--refs-mode", choices=["temporal", "lodo"], default="temporal")
    sp.set_defaults(fn=cmd_inspect)

    sp = sub.add_parser("doctor")
    sp.add_argument("--models", help="model config JSON to load as a test")
    sp.set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("prune")
    sp.add_argument("--emb")
    sp.add_argument("--cdn-cache")
    sp.add_argument("--days", type=float, default=14.0)
    sp.set_defaults(fn=cmd_prune)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
