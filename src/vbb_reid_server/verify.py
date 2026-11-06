"""Nightly verification page (owner 2026-11-06): a blind sample of each night's new names, judged one
picture at a time against reference pictures of each person, with a running tally.

Three steps, each its own command:

- ``verify-refs`` (once, or when the roster changes): reference pictures per person. Only pictures the
  owner named one at a time, blind (the adjudication answer files), never the bulk labels: on facility 3
  most human labels were whole board groups named in one click, and at least one shows the wrong person.
  A picture must show one face (``n_faces == 1`` on the uploaded picture), must agree with its human
  label when it has one, and must not be on the exclude list.
- ``verify-build`` (nightly, after the run): draws ``n`` of the night's new names (pushed ``known``
  promotions, uniform, seeded by the run id) into ``<root>/<facility>_<day>/frozen.json``, read-only.
  It never draws twice: if the night exists, it does nothing.
- ``verify-serve``: a loopback-only page reached by an ssh tunnel, for as long as the ssh session
  lasts. The reviewer's page never receives the machine's name, the via, any score, the board's
  state or picture links: pictures come through ``/img`` from links in ``frozen.json`` only. Answers
  save to ``marks.json`` (atomic) plus a line in ``marks.log``. Results are shown only for nights
  marked done, so judging the next picture never follows feedback on the last.
"""

from __future__ import annotations

import csv
import html
import json
import os
import random
import re
import time
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .metrics import cp_lower

IMAGE_PREFIX = "https://upvbbv.yuexingren.cn/"  # the uploaded pictures; nothing else is fetched
OTHER, CANT = "other", "cant_tell"
MAX_POST = 1 << 20
NIGHT_RE = re.compile(r"^\d+_\d{4}-\d{2}-\d{2}$")


# ------------------------------------------------------------------------ references --
def read_any_answers(path: Path) -> dict[str, str]:
    """An adjudication CSV (``read_answers``: a two-people note turns into ``mixed``) or a review page's
    ``marks.json`` (``{"answers": {sid: answer}}``)."""
    if path.suffix == ".json":
        return {str(k): str(v) for k, v in (json.loads(path.read_text(encoding="utf-8")).get("answers") or {}).items()}
    from .adjudicate import read_answers
    return read_answers(path)


def pick_references(ex, facility: str, answer_files: list[Path], *, one_face, exclude: set[str],
                    per_person: int = 6, seed: int = 1) -> dict:
    """``one_face(sighting) -> bool``: the uploaded picture shows exactly one face. Answers are identity
    keys (the alias file's H1..); a picture with two different answers across files is dropped."""
    roster = ex.roster(facility)  # key -> this facility's person_id
    answers: dict[str, set[str]] = defaultdict(set)
    for f in answer_files:
        for sid, a in read_any_answers(Path(f)).items():
            answers[sid].add(a)
    by_id = {s.sighting_id: s for s in ex.sightings}
    pool: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    dropped: dict[str, int] = defaultdict(int)
    for sid, given in sorted(answers.items()):
        s = by_id.get(sid)
        why = None
        if len(given) != 1:
            why = "answers disagree"
        elif sid in exclude:
            why = "excluded"
        elif (a := next(iter(given))) not in roster:
            why = "not a roster person"
        elif s is None or not (s.image_url or "").startswith(IMAGE_PREFIX):
            why = "no uploaded picture"
        elif s.human_state == "stranger" or (s.human_state == "known" and ex.key(s.human_person_id) != a):
            why = "disagrees with its human label"
        elif not one_face(s):
            why = "not exactly one face"
        if why:
            dropped[why] += 1
            continue
        pool[a][s.day].append(s)
    rng = random.Random(seed)
    people = []
    for key in sorted(roster):
        days = sorted(pool.get(key, {}))
        for d in days:
            rng.shuffle(pool[key][d])
        photos = []
        while len(photos) < per_person and any(pool[key][d] for d in days):
            for d in days:  # one per day in turn, so a reviewer doesn't learn one outfit
                if pool[key][d] and len(photos) < per_person:
                    s = pool[key][d].pop()
                    photos.append({"sighting_id": s.sighting_id, "day": s.day, "url": s.image_url})
        people.append({"person_id": roster[key], "key": key, "name": ex.names.get(key, ""), "photos": photos})
    return {"facility_id": facility, "built_at": int(time.time()), "per_person": per_person,
            "dropped": dict(sorted(dropped.items())), "people": people}


# ------------------------------------------------------------------------ nightly draw --
def _night_run(runs_dir: Path, facility: str, day: str, run_id: str | None):
    """The labels CSV and run JSON of the newest finished, non-dry run of the night (or ``run_id``)."""
    best = None
    for rj in runs_dir.glob(f"run_{facility}_{day}_*.json"):
        meta = json.loads(rj.read_text(encoding="utf-8"))
        if run_id is not None and meta.get("run_id") != run_id:
            continue
        if run_id is None and meta.get("dry_run") is not False:  # older files lack the marker: skipped
            continue
        if best is None or meta.get("finished_at", 0) > best[1].get("finished_at", 0):
            best = (rj, meta)
    if best is None:
        return None
    rj, meta = best
    return rj.with_name(rj.name.replace("run_", "labels_", 1).replace(".json", ".csv")), meta


def build_night(runs_dir: Path, refs_path: Path, out_root: Path, facility: str, day: str, *, n: int = 20,
                run_id: str | None = None, log=print) -> Path | None:
    night = out_root / f"{facility}_{day}"
    frozen = night / "frozen.json"
    if frozen.exists():
        log(f"[verify] {night.name}: already drawn, nothing changed")
        return frozen
    found = _night_run(Path(runs_dir), facility, day, run_id)
    if found is None:
        log(f"[verify] {facility} {day}: no finished non-dry run in {runs_dir}")
        return None
    labels_csv, meta = found
    with labels_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "image_url" not in rows[0]:
        raise SystemExit(f"{labels_csv} has no image_url column (written before 0.2.3); draw a later night")
    pool = sorted((r for r in rows if r["state"] == "known" and r["relation"] == "promote"
                   and r["human_labeled"] in ("0", "") and r["image_url"].startswith(IMAGE_PREFIX)),
                  key=lambda r: r["sighting_id"])
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["human_labeled"] in ("0", ""):
            counts[f"{r['state']}/{r['relation']}/{r['via']}"] += 1
    rng = random.Random(f"verify:{facility}:{day}:{meta['run_id']}")
    drawn = rng.sample(pool, min(n, len(pool)))
    refs = json.loads(Path(refs_path).read_text(encoding="utf-8"))
    data = {
        "facility_id": facility, "day": day, "run_id": meta["run_id"], "model": meta.get("model", ""),
        "drawn_at": int(time.time()), "n": n, "pool": len(pool), "pool_by_via": _by(pool, "via"),
        "counts": dict(sorted(counts.items())), "refs_built_at": refs.get("built_at"), "people": refs["people"],
        "items": [{"sighting_id": r["sighting_id"], "url": r["image_url"], "person_id": r["person_id"],
                   "via": r["via"], "face_sim": r["face_sim"], "body_sim": r["body_sim"]} for r in drawn],
    }
    night.mkdir(parents=True, exist_ok=True)
    tmp = night / "frozen.json.tmp"
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, frozen)
    os.chmod(frozen, 0o444)
    log(f"[verify] {night.name}: drew {len(drawn)} of {len(pool)} new names (run {meta['run_id'][:8]})")
    return frozen


def _by(rows, field) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[r[field]] += 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------------ scoring --
def score_night(frozen: dict, marks: dict) -> dict:
    """right: the answer is the machine's person; wrong: another person, or someone not on the list;
    can't tell is left out of precision."""
    by_key = {p["key"]: p["person_id"] for p in frozen["people"]}
    t = {"right": 0, "wrong": 0, "cant_tell": 0, "unanswered": 0}
    per_via: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_person: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    wrong = []
    answers = marks.get("answers") or {}
    for i, it in enumerate(frozen["items"]):
        a = answers.get(str(i))
        if a is None:
            t["unanswered"] += 1
            continue
        if a == CANT:
            t["cant_tell"] += 1
            continue
        ok = by_key.get(a) == it["person_id"]
        t["right" if ok else "wrong"] += 1
        per_via[it["via"]][0 if ok else 1] += 1
        per_person[it["person_id"]][0 if ok else 1] += 1
        if not ok:
            wrong.append({"i": i, "answer": a, "machine": it["person_id"], "via": it["via"],
                          "face_sim": it["face_sim"], "body_sim": it["body_sim"]})
    return {**t, "per_via": dict(per_via), "per_person": dict(per_person), "wrong_items": wrong}


# ------------------------------------------------------------------------ server --
def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _direct_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never a login shell's proxy


class Store:
    def __init__(self, root: Path, fetch=None):
        self.root = Path(root)
        self.fetch = fetch or (lambda url: _direct_opener().open(url, timeout=15).read())

    def nights(self) -> list[str]:
        return sorted((p.name for p in self.root.iterdir() if NIGHT_RE.match(p.name) and (p / "frozen.json").exists()),
                      reverse=True) if self.root.exists() else []

    def frozen(self, night: str) -> dict | None:
        if not NIGHT_RE.match(night):
            return None
        f = self.root / night / "frozen.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def marks(self, night: str) -> dict:
        f = self.root / night / "marks.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        return {"answers": {}, "seen": [], "failed": [], "submitted": False, "reviewer": "", "updated_at": None}

    def save_marks(self, night: str, frozen: dict, body: dict) -> dict:
        """Validates against the frozen night: item indexes in range, answers among the page's choices."""
        n = len(frozen["items"])
        allowed = {p["key"] for p in frozen["people"]} | {OTHER, CANT}
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("answers must be an object")
        clean = {}
        for k, v in answers.items():
            if not (isinstance(k, str) and k.isdigit() and int(k) < n) or v not in allowed:
                raise ValueError(f"bad answer {k!r}: {v!r}")
            clean[k] = v
        idx = lambda xs: sorted({int(x) for x in xs if isinstance(x, int) and 0 <= x < n})  # noqa: E731
        old = self.marks(night)
        seen = sorted(set(old.get("seen") or []) | set(idx(body.get("seen") or [])))
        failed = [i for i in idx(body.get("failed") or []) if i not in seen]
        reviewer = str(body.get("reviewer") or "")[:40]
        out = {"answers": clean, "seen": seen, "failed": failed, "submitted": bool(body.get("submitted")),
               "reviewer": reviewer, "updated_at": int(time.time())}
        _atomic_json(self.root / night / "marks.json", out)
        with (self.root / night / "marks.log").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": out["updated_at"], "answered": len(clean), "seen": len(seen),
                                 "submitted": out["submitted"], "reviewer": reviewer}) + "\n")
        return out

    def image(self, night: str, kind: str, a: int, b: int = 0) -> bytes | None:
        fz = self.frozen(night)
        if fz is None:
            return None
        try:
            url = fz["items"][a]["url"] if kind == "item" else fz["people"][a]["photos"][b]["url"]
        except (IndexError, KeyError):
            return None
        if not url.startswith(IMAGE_PREFIX):
            return None
        return self.fetch(url)


def page_data(fz: dict) -> dict:
    """What the reviewer's page receives: labels, names and picture counts. No machine output, no links.
    Only people with reference pictures are offered: one nobody can recognise is no use as a choice, and a
    person without a single labeled picture has no references, so the machine cannot name them either.
    ``ref`` keeps the person's place in ``frozen.json`` for ``/ref/<night>/<ref>/<j>``."""
    people = [{"key": p["key"], "name": p.get("name", ""), "photos": len(p["photos"]), "ref": i}
              for i, p in enumerate(fz["people"]) if p["photos"]]
    return {"day": fz["day"], "n": len(fz["items"]), "people": people}


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the ssh session quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def _loopback(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in ("127.0.0.1", "localhost", "[::1]")

        def do_GET(self):
            if not self._loopback():
                return self._send(403, b"loopback only", "text/plain")
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if not parts:
                return self._send(200, index_html(store).encode(), "text/html; charset=utf-8")
            if parts[0] == "n" and len(parts) in (2, 3):
                fz = store.frozen(parts[1])
                if fz is None:
                    return self._send(404, b"no such night", "text/plain")
                if len(parts) == 3 and parts[2] == "marks":
                    return self._json(200, store.marks(parts[1]))
                if len(parts) == 2:
                    return self._send(200, night_html(parts[1], page_data(fz)).encode(), "text/html; charset=utf-8")
            if parts[0] in ("img", "ref") and len(parts) in (3, 4) and all(p.isdigit() for p in parts[2:]):
                nums = [int(p) for p in parts[2:]]
                try:
                    data = store.image(parts[1], "item" if parts[0] == "img" else "ref", *nums)
                except Exception:
                    return self._send(502, b"picture could not be fetched", "text/plain")
                if data is None:
                    return self._send(404, b"no such picture", "text/plain")
                return self._send(200, data, "image/jpeg", "private, max-age=3600")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if not (self._loopback() and self.headers.get("X-Review") == "1"
                    and (self.headers.get("Content-Type") or "").startswith("application/json")):
                return self._send(403, b"refused", "text/plain")
            if len(parts) != 3 or parts[0] != "n" or parts[2] != "marks":
                return self._send(404, b"not found", "text/plain")
            fz = store.frozen(parts[1])
            if fz is None:
                return self._send(404, b"no such night", "text/plain")
            size = int(self.headers.get("Content-Length") or 0)
            if size > MAX_POST:
                return self._send(413, b"too large", "text/plain")
            try:
                out = store.save_marks(parts[1], fz, json.loads(self.rfile.read(size) or b"{}"))
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, out)
    return H


def serve(root: Path, port: int, log=print) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Store(root)))
    log(f"[verify] serving {root} on http://127.0.0.1:{port}/ (Ctrl-C or closing ssh stops it)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------------ pages --
_STYLE = """<style>
:root{--bg:#fafafa;--fg:#1c1c1c;--muted:#666;--line:#ddd;--card:#fff;--accent:#2563eb;--bad:#b91c1c;--ok:#15803d}
@media (prefers-color-scheme: dark){:root{--bg:#141414;--fg:#eee;--muted:#aaa;--line:#333;--card:#1e1e1e;--accent:#60a5fa;--bad:#f87171;--ok:#4ade80}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:12px 16px}
table{border-collapse:collapse;margin:8px 0 16px}td,th{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}
.muted{color:var(--muted);font-size:13px}.bad{color:var(--bad)}.ok{color:var(--ok)}
a{color:var(--accent)}
</style>"""


def _pct(x):
    return "" if x is None else f"{x:.1%}"


def index_html(store: Store) -> str:
    e = html.escape
    rows, tot_via, tot_person, wrong_rows = [], defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), []
    names = {}
    for night in store.nights():
        fz, mk = store.frozen(night), store.marks(night)
        names.update({p["person_id"]: (p["key"] + (f" ({p['name']})" if p.get("name") else "")) for p in fz["people"]})
        done = bool(mk.get("submitted"))
        sc = score_night(fz, mk) if done else None
        state = "done" if done else (f"{len(mk.get('answers') or {})} of {len(fz['items'])} answered" if mk.get("answers") else "not started")
        res = (f"<span class=ok>{sc['right']} right</span> · <span class=bad>{sc['wrong']} wrong</span> · "
               f"{sc['cant_tell']} can't tell") if sc else "<span class=muted>shown when done</span>"
        rows.append(f"<tr><td><a href='/n/{e(night)}'>{e(fz['day'])}</a></td><td>{fz['pool']}</td>"
                    f"<td>{len(fz['items'])}</td><td>{e(state)}</td><td>{res}</td>"
                    f"<td class=muted>{e(mk.get('reviewer') or '')}</td></tr>")
        if sc:
            for v, (r, w) in sc["per_via"].items():
                tot_via[v][0] += r
                tot_via[v][1] += w
            for p, (r, w) in sc["per_person"].items():
                tot_person[p][0] += r
                tot_person[p][1] += w
            for x in sc["wrong_items"]:
                wrong_rows.append((night, fz["day"], x))

    def tally(d, label):
        out = []
        for k, (r, w) in sorted(d.items()):
            n = r + w
            out.append(f"<tr><td>{e(label(k))}</td><td>{r}</td><td>{w}</td><td>{_pct(r / n if n else None)}</td>"
                       f"<td>{_pct(cp_lower(r, n))}</td></tr>")
        return "".join(out) or "<tr><td colspan=5 class=muted>no finished nights yet</td></tr>"

    head = "<tr><th>{}</th><th>right</th><th>wrong</th><th>share right</th><th>at least (97.5 %)</th></tr>"
    wrong_html = "".join(
        f"<tr><td>{e(day)}</td><td><img src='/img/{e(night)}/{x['i']}' style='height:120px'></td>"
        f"<td>machine: {e(names.get(x['machine'], x['machine'][-6:]))}<br>answer: {e(x['answer'])}</td>"
        f"<td class=muted>{e(x['via'])} face {e(str(x['face_sim']))} body {e(str(x['body_sim']))}</td></tr>"
        for night, day, x in wrong_rows) or "<tr><td class=muted>none</td></tr>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Nightly name check</title>{_STYLE}</head>
<body><main><h2>Nightly name check</h2>
<p class=muted>Each night, a random sample of the new names the machine gave (labels are still hidden in the app).
Open a night, say who each picture shows, then press "I'm done". Results appear only after that.</p>
<table><tr><th>night</th><th>new names</th><th>sample</th><th>status</th><th>result</th><th>reviewer</th></tr>{''.join(rows)}</table>
<h3>All finished nights, by how the name was made</h3><table>{head.format('via')}{tally(tot_via, lambda k: k)}</table>
<h3>By the person the machine named</h3><table>{head.format('person')}{tally(tot_person, lambda k: names.get(k, k[-6:]))}</table>
<h3>Wrong names</h3><table>{wrong_html}</table>
<p class=muted>"At least" is the one-sided 97.5 % lower bound: with 0 wrong it reaches 98 % only after 183 answers.</p>
</main></body></html>"""


def _js(obj) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def night_html(night: str, data: dict) -> str:
    return _NIGHT.replace("__DATA__", _js(data)).replace("__NIGHT__", _js(night))


_NIGHT = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Name check</title>""" + _STYLE + """
<style>
.refs{display:flex;flex-direction:column;gap:6px;border-bottom:1px solid var(--line);padding-bottom:8px}
.ref{display:flex;gap:6px;align-items:center;overflow-x:auto}.ref b{min-width:9em}.ref img{height:84px;border-radius:4px}
.stage{display:flex;gap:20px;flex-wrap:wrap;margin-top:12px}.stage img{max-height:420px;max-width:100%;border-radius:6px;background:#888}
.choices button{display:block;width:100%;text-align:left;margin:4px 0;padding:8px 12px;font-size:15px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
.choices button.on{outline:3px solid var(--accent)}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px}
#status.bad{color:var(--bad)}
</style></head><body><main>
<div class="bar"><a href="/">← all nights</a><b id="title"></b><span id="progress"></span>
<label class="muted">Your name <input id="reviewer" maxlength="40" style="width:9em"></label><span id="status" class="muted"></span></div>
<p class="muted">Compare each picture with the reference pictures below and pick who it is. If it shows two people, answer
for the main one. Keys: 1–9 pick a person, o = someone else, x = can't tell, ← → move.</p>
<div class="refs" id="refs"></div>
<div class="stage"><div><img id="pic" alt="picture"></div><div class="choices" id="choices" style="min-width:16em"></div></div>
<div class="bar"><button id="prev">← Back</button><button id="next">Next →</button><button id="done">I'm done</button></div>
<script>
const D = __DATA__, NIGHT = __NIGHT__;
let marks = {answers:{}, seen:[], failed:[], submitted:false, reviewer:""}, loaded = false, cur = 0, timer = null, pending = false;
const seen = new Set(), failed = new Set();
const $ = id => document.getElementById(id);
$("title").textContent = "Night " + D.day;
const choices = D.people.map((p, i) => ({v: p.key, label: (i + 1) + ". " + p.key + (p.name ? " · " + p.name : "")}))
  .concat([{v: "other", label: "o. Stranger or someone else"}, {v: "cant_tell", label: "x. Can't tell"}]);
D.people.forEach((p, i) => { const row = document.createElement("div"); row.className = "ref";
  const b = document.createElement("b"); b.textContent = (i + 1) + ". " + p.key + (p.name ? " · " + p.name : ""); row.appendChild(b);
  for (let j = 0; j < p.photos; j++) { const im = document.createElement("img"); im.loading = "lazy"; im.src = "/ref/" + NIGHT + "/" + p.ref + "/" + j; row.appendChild(im); }
  $("refs").appendChild(row); });
choices.forEach(c => { const b = document.createElement("button"); b.textContent = c.label; b.dataset.v = c.v;
  b.onclick = () => { b.blur(); answer(c.v); }; $("choices").appendChild(b); });  // blur: Enter must not repeat it
function status(t, bad){ $("status").textContent = t; $("status").className = bad ? "bad" : "muted"; }
function progress(){ $("progress").textContent = "Picture " + (cur + 1) + " of " + D.n + " · " + Object.keys(marks.answers).length + " answered"; }
function show(){ const pic = $("pic"), i = cur;  // i: a late load of an earlier picture must not mark this one
  pic.onload = () => { if (i === cur) { seen.add(i); failed.delete(i); } };
  pic.onerror = () => { if (i === cur) { if (!seen.has(i)) failed.add(i); status("Picture " + (i + 1) + " did not load", true); } };
  pic.src = "/img/" + NIGHT + "/" + cur;
  document.querySelectorAll(".choices button").forEach(b => b.classList.toggle("on", marks.answers[String(cur)] === b.dataset.v));
  progress(); }
function answer(v){ if (!loaded) { status("Still loading saved answers…", true); return; }
  if (!seen.has(cur)) { status("Wait for the picture to show before answering", true); return; }
  marks.answers[String(cur)] = v; queue(); if (cur < D.n - 1) { cur++; } show(); }
function queue(){ pending = true; clearTimeout(timer); timer = setTimeout(save, 400); }
async function save(){ if (!loaded) return false; const body = {answers: marks.answers, seen: [...seen], failed: [...failed],
    submitted: marks.submitted, reviewer: $("reviewer").value};
  try { const r = await fetch("/n/" + NIGHT + "/marks", {method: "POST", headers: {"Content-Type": "application/json", "X-Review": "1"}, body: JSON.stringify(body)});
    if (!r.ok) throw new Error(await r.text()); pending = false; status("Saved"); return true; }
  catch (e) { status("Not saved: " + e.message, true); return false; } }
$("prev").onclick = () => { if (cur > 0) { cur--; show(); } };
$("next").onclick = () => { if (cur < D.n - 1) { cur++; show(); } };
$("done").onclick = () => { const left = D.n - Object.keys(marks.answers).length;
  if (left && !confirm(left + " picture(s) have no answer. Finish anyway?")) return;
  marks.submitted = true; clearTimeout(timer); save().then(ok => { if (ok) location.href = "/"; }); };
$("reviewer").onchange = () => { try { localStorage.setItem("reviewer", $("reviewer").value); } catch (e) {} queue(); };
try { $("reviewer").value = localStorage.getItem("reviewer") || ""; } catch (e) {}
document.addEventListener("keydown", ev => { if (ev.target.tagName === "INPUT") return; const k = ev.key.toLowerCase();
  if (k === "arrowleft") return $("prev").onclick(); if (k === "arrowright") return $("next").onclick();
  if (/^[1-9]$/.test(k) && D.people[+k - 1]) return answer(D.people[+k - 1].key);
  if (k === "o") return answer("other"); if (k === "x") return answer("cant_tell"); });
window.addEventListener("beforeunload", ev => { if (pending) { ev.preventDefault(); ev.returnValue = ""; } });
fetch("/n/" + NIGHT + "/marks").then(r => r.json()).then(m => { marks = m; marks.answers = m.answers || {};
  (m.seen || []).forEach(i => seen.add(i)); if (!$("reviewer").value) $("reviewer").value = m.reviewer || "";
  loaded = true; cur = Math.min(Object.keys(marks.answers).length, D.n - 1); show();
  if (m.submitted) status("This night is marked done; changes still save."); })
  .catch(() => status("Could not load saved answers; reload the page", true));
show();
</script></main></body></html>"""
