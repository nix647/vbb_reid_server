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


def _thresholds(args) -> Thresholds:
    th = Thresholds()
    if getattr(args, "thresholds", None):
        spec = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        known = {f.name for f in fields(Thresholds)}
        th = replace(th, **{k: v for k, v in spec.items() if k in known})
    return th


def _ids(models_cfg: str | None):
    """Model ids from a model config, without loading heavy weights twice."""
    if not models_cfg:
        return None, None, None
    from .models import load_model_set
    ms = load_model_set(models_cfg)
    return ms, ms.face_id, ms.body_id


def _ids_from_cache(emb_root: str, kind: str, name: str | None) -> str | None:
    """Pick a cached model id by slug or substring (eval/shadow need no GPU)."""
    if not name:
        return None
    base = Path(emb_root) / kind
    for meta in sorted(base.glob("*/model.json")):
        mid = json.loads(meta.read_text())["model_id"]
        if name == meta.parent.name or name in mid:
            return mid
    raise SystemExit(f"no cached {kind} model matching {name!r} under {base}")


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
            embed_day(ex.by_facility_day(fac, day), store, ms, args.emb, fac, day, face_min_px=args.face_min_px,
                      refresh=args.refresh)
    print(f"[embed] crop sources: {store.stats}")
    print(f"[embed] face model: {ms.face_id}\n[embed] body model: {ms.body_id}")


def cmd_eval(args) -> None:
    from .evaluate import (curve_rows, evaluate_facility, load_grid, operating_point, per_sighting_wrongs,
                           summarize, write_curve)
    ex = load_export(args.export, args.aliases)
    face_id = _ids_from_cache(args.emb, "face", args.face_model)
    body_id = _ids_from_cache(args.emb, "body", args.body_model)
    ws = Workspace(ex, args.emb, face_id, body_id, QualityPolicy.from_file(args.quality))
    base = _thresholds(args)
    policy = RefPolicy(window_days=args.window_days, ref_q_min=args.ref_q_min, max_refs=args.max_refs)
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
            point = operating_point(rows, args.min_precision_lo, args.max_loo_fa)
            text = summarize(rows, point)
            print(f"[eval] facility {fac} refs={mode}\n{text}")
            with (out / "report.md").open("a", encoding="utf-8") as fh:
                fh.write(f"## {setup} · facility {fac} · refs {mode}\n\n```\n{text}\n```\n")
                wrongs = per_sighting_wrongs(results, point)
                if wrongs:
                    fh.write(f"Wrong at the operating point ({len(wrongs)}): {', '.join(wrongs[:50])}\n\n")


def cmd_shadow(args) -> None:
    from .shadow import shadow_run
    ex = load_export(args.export, args.aliases)
    face_id = _ids_from_cache(args.emb, "face", args.face_model)
    body_id = _ids_from_cache(args.emb, "body", args.body_model)
    ws = Workspace(ex, args.emb, face_id, body_id, QualityPolicy.from_file(args.quality))
    th = _thresholds(args)
    days = args.days.split(",") if args.days else ex.days(args.facility)
    for day in days:
        shadow_run(ws, args.facility, day, th, args.out, model_name=args.setup or "")


def cmd_adj_sample(args) -> None:
    from .adjudicate import sample, write_sample
    ex = load_export(args.export, args.aliases)
    exclude = set()
    for p in args.exclude or []:
        with open(p, newline="", encoding="utf-8") as fh:
            exclude |= {r["sighting_id"] for r in csv.DictReader(fh)}
    days = args.days.split(",") if args.days else None
    rows = sample(ex, args.facility, args.n, seed=args.seed, days=days, exclude=exclude)
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
                  f"min cosine = {float((batched * single).sum(1).min()):.6f}  (expect ~1e-6 and 1.000000)")
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
    sp.add_argument("--face-min-px", type=float, default=12.0)
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
    sp.add_argument("--min-precision-lo", type=float, default=0.99)
    sp.add_argument("--max-loo-fa", type=float, default=0.005)
    sp.add_argument("--no-loo", action="store_true")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("shadow")
    common(sp)
    sp.add_argument("--facility", required=True)
    sp.add_argument("--days")
    sp.add_argument("--face-model")
    sp.add_argument("--body-model")
    sp.add_argument("--thresholds", help="JSON with the chosen operating point")
    sp.add_argument("--quality", help="JSON QualityPolicy overrides")
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
