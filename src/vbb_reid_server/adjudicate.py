"""Adjudication set (plan §6): a stratified random sample of sightings the owner
labels on purpose, in a self-contained HTML page. The page never shows any machine
output or score (07-24 ruling as clarified; also keeps the check blind).

Answers are kept in the browser (localStorage) and exported as CSV:
``sighting_id,answer,note`` with ``answer`` = identity key | ``stranger`` | ``cant_tell``.
"""

from __future__ import annotations

import csv
import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .data import Export, Sighting
from .decide import MachineLabel
from .metrics import Tally, bootstrap_precision, cp_lower


def stratum(s: Sighting) -> str:
    return f"{s.board_state}/{'face' if s.crop_is_face else 'noface'}"


def sample(ex: Export, facility: str, n: int, *, seed: int = 20261103, days: list[str] | None = None,
           exclude: set[str] | None = None, only: set[str] | None = None) -> list[Sighting]:
    """Equal share per stratum (board state x face/noface), spread evenly over days.
    Already human-labeled sightings and ``exclude`` are skipped; ``only`` restricts the pool
    (e.g. sightings a shadow run named from low-norm faces: a TARGETED set, whose precision
    describes that band, not the population)."""
    rng = random.Random(seed)
    exclude = exclude or set()
    pool = [s for s in ex.sightings if s.facility_id == facility and s.human_state is None
            and s.image_url and s.sighting_id not in exclude and (days is None or s.day in days)
            and (only is None or s.sighting_id in only)]
    by_stratum: dict[str, dict[str, list[Sighting]]] = defaultdict(lambda: defaultdict(list))
    for s in pool:
        by_stratum[stratum(s)][s.day].append(s)
    strata = sorted(by_stratum)
    if not strata:
        return []
    per = max(1, n // len(strata))
    picked: list[Sighting] = []
    for st in strata:
        days_map = by_stratum[st]
        day_keys = sorted(days_map)
        for d in day_keys:
            rng.shuffle(days_map[d])
        taken = 0
        while taken < per and any(days_map[d] for d in day_keys):
            for d in day_keys:
                if taken >= per:
                    break
                if days_map[d]:
                    picked.append(days_map[d].pop())
                    taken += 1
    rng.shuffle(picked)
    return picked[:n] if len(picked) > n else picked


def write_sample(rows: list[Sighting], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sighting_id", "facility_id", "day", "did", "board_state", "crop_is_face", "stratum", "image_url"])
        for s in rows:
            w.writerow([s.sighting_id, s.facility_id, s.day, s.did, s.board_state, int(s.crop_is_face),
                        stratum(s), s.image_url])


def read_sample(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [r["sighting_id"] for r in csv.DictReader(fh)]


def person_strips(ex: Export, facility: str, per_person: int = 8, seed: int = 1) -> dict[str, list[str]]:
    """Up to ``per_person`` human-labeled crops per roster person, spread over days and over
    the brand's facilities, board face crops first (references are photos, never vectors or
    scores). Any labeled crop counts: the server detector may find a face the board did not
    flag, so a person with only non-face-flagged labels still has references and must still
    be offered to the adjudicator."""
    rng = random.Random(seed)
    org = ex.facility_org.get(facility)
    facs = set(ex.facility_of_org(org)) if org else {facility}
    roster = ex.roster(facility)
    by_key: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for s in ex.sightings:
        if s.facility_id in facs and s.human_state == "known" and s.image_url:
            k = ex.key(s.human_person_id)
            if k in roster:
                by_key[k][s.day].append((0 if s.crop_is_face else 1, s.image_url))
    out: dict[str, list[str]] = {}
    for k in roster:
        days = sorted(by_key.get(k, {}))
        for d in days:
            rng.shuffle(by_key[k][d])
            by_key[k][d].sort(key=lambda t: t[0], reverse=True)  # pop() takes face crops first
        urls: list[str] = []
        while len(urls) < per_person and any(by_key[k][d] for d in days):
            for d in days:
                if by_key[k][d] and len(urls) < per_person:
                    urls.append(by_key[k][d].pop()[1])
        out[k] = urls
    return out


_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sighting adjudication</title>
<style>
:root{--bg:#fafafa;--fg:#1c1c1c;--muted:#666;--line:#ddd;--card:#fff;--accent:#2563eb}
@media (prefers-color-scheme: dark){:root{--bg:#141414;--fg:#eee;--muted:#aaa;--line:#333;--card:#1e1e1e;--accent:#60a5fa}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header button{padding:6px 12px}
.people{display:flex;flex-direction:column;gap:6px;padding:8px 16px;border-bottom:1px solid var(--line)}
.person{display:flex;gap:6px;align-items:center;overflow-x:auto}
.person b{min-width:5em}
.person img{height:72px;border-radius:4px}
.item{display:flex;gap:16px;padding:12px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.item.done{opacity:.55}
.crops img{height:220px;border-radius:4px;margin-right:6px;background:#888}
.choices label{display:block;padding:3px 0;cursor:pointer}
.meta{color:var(--muted);font-size:12px}
</style></head><body>
<header><b>Adjudication</b><span id="progress"></span>
<button onclick="exportCsv()">Export CSV</button>
<span class="meta">Answers are kept in this browser until exported. Keys 1-9 pick a person, s = stranger, x = can't tell, n = next; keys act on the item you last clicked or answered.</span></header>
<div class="people" id="people"></div>
<div id="items"></div>
<script>
const PEOPLE = __PEOPLE__;
const ITEMS = __ITEMS__;
const KEY = "adjudication:" + __SET_ID__;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} progress(); }
function progress(){ document.getElementById("progress").textContent = Object.keys(answers).length + " / " + ITEMS.length; }
function esc(s){ const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
const pe = document.getElementById("people");
PEOPLE.forEach((p,i)=>{ const row=document.createElement("div"); row.className="person";
  row.innerHTML = "<b>"+(i+1)+". "+esc(p.name)+"</b>" + p.photos.map(u=>'<img loading="lazy" src="'+esc(u).replace(/"/g,'&quot;')+'">').join("");
  pe.appendChild(row); });
const box = document.getElementById("items");
ITEMS.forEach((it,idx)=>{ const d=document.createElement("div"); d.className="item"; d.id="it"+idx;
  const opts = PEOPLE.map((p,i)=>'<label><input type="radio" name="a'+idx+'" value="'+esc(p.key).replace(/"/g,'&quot;')+'"> '+(i+1)+'. '+esc(p.name)+'</label>').join("")
    + '<label><input type="radio" name="a'+idx+'" value="stranger"> s. 陌生人 / stranger</label>'
    + '<label><input type="radio" name="a'+idx+'" value="cant_tell"> x. 无法判断 / can\\'t tell</label>';
  d.innerHTML = '<div class="crops">'+it.crops.map(u=>'<img loading="lazy" src="'+esc(u).replace(/"/g,'&quot;')+'">').join("")+'</div>'
    + '<div><div class="choices">'+opts+'</div><div class="meta">#'+(idx+1)+' · '+esc(it.day)+' · '+esc(it.sighting_id)+'</div>'
    + '<input placeholder="note" style="margin-top:6px;width:16em" data-note="'+idx+'"></div>';
  box.appendChild(d);
  const prev = answers[it.sighting_id];
  if (prev) { const r=d.querySelector('input[value="'+prev.answer+'"]'); if (r) r.checked=true; d.classList.add("done");
    d.querySelector("[data-note]").value = prev.note || ""; }
  d.addEventListener("change", ev=>{ const sel=d.querySelector("input[type=radio]:checked"); if(!sel) return;
    answers[it.sighting_id] = {answer: sel.value, note: d.querySelector("[data-note]").value}; d.classList.add("done"); save();
    cur = idx; });
  d.addEventListener("click", ()=>{ cur = idx; });
});
let cur = 0;
document.addEventListener("keydown", ev=>{ if (ev.target.tagName === "INPUT" && ev.target.type === "text") return;
  const k = ev.key.toLowerCase(); const it = ITEMS[cur]; if (!it) return; let v = null;
  if (/^[1-9]$/.test(k) && PEOPLE[+k-1]) v = PEOPLE[+k-1].key; else if (k==="s") v="stranger"; else if (k==="x") v="cant_tell";
  else if (k==="n") { cur=Math.min(cur+1, ITEMS.length-1); document.getElementById("it"+cur).scrollIntoView({block:"center"}); return; }
  if (v) { const at=cur; const el=document.querySelector('#it'+at+' input[value="'+v+'"]'); if (el){ el.checked=true; el.dispatchEvent(new Event("change",{bubbles:true})); }
    cur=Math.min(at+1, ITEMS.length-1); document.getElementById("it"+cur).scrollIntoView({block:"center"}); } });
function exportCsv(){ const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
  const lines=["sighting_id,answer,note"].concat(ITEMS.filter(it=>answers[it.sighting_id]).map(it=>
    [it.sighting_id, answers[it.sighting_id].answer, answers[it.sighting_id].note||""].map(q).join(",")));
  const blob=new Blob([lines.join("\\n")+"\\n"],{type:"text/csv"}); const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download="adjudication_"+__SET_ID__+".csv"; a.click(); }
progress();
</script></body></html>
"""


def _script_json(obj) -> str:
    """JSON safe inside a <script> element: ``</script>`` or ``<!--`` in a value cannot end it."""
    return (json.dumps(obj, ensure_ascii=False)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def write_page(ex: Export, facility: str, sample_ids: list[str], path: Path, *, set_id: str,
               lan_root: Path | None = None) -> None:
    """Self-contained page. Crops: the CDN best crop, plus the LAN copies when ``lan_root``
    is given (referenced by absolute ``file://`` path — open the page on the GPU host)."""
    by_id = {s.sighting_id: s for s in ex.sightings}
    roster = ex.roster(facility)
    strips = person_strips(ex, facility)
    # People nobody has ever labeled have no photo strip and no references: the machine can
    # never name them, so the page does not offer them either (answer "stranger" or "can't tell").
    people = [{"key": k, "name": ex.names.get(k, k[-6:]), "photos": strips[k]}
              for k in sorted(roster) if strips.get(k)]
    items = []
    for sid in sample_ids:
        s = by_id.get(sid)
        if s is None:
            continue
        crops = []
        if lan_root is not None:
            d = None
            for day in (s.day,):
                cand = Path(lan_root) / s.did / day / sid
                if cand.is_dir():
                    d = cand
            if d is not None:
                crops = [f"file://{p.resolve()}" for p in sorted(d.glob("*.jpg"))]
        if not crops and s.image_url:
            crops = [s.image_url]
        items.append({"sighting_id": sid, "day": s.day, "crops": crops})
    page = (_PAGE.replace("__PEOPLE__", _script_json(people))
                 .replace("__ITEMS__", _script_json(items))
                 .replace("__SET_ID__", _script_json(html.escape(set_id))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


_MIXED_NOTE = re.compile(r"(2|two|二|两)\s*(people|persons|个?人)", re.IGNORECASE)


def read_answers(path: Path) -> dict[str, str]:
    """sighting_id -> answer. A note saying the crop shows two people turns the answer into
    ``mixed`` whatever was clicked: a mixed track has no single truth, so it is excluded
    from precision and used to check whether the machine flags it."""
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("answer"):
                continue
            out[r["sighting_id"]] = "mixed" if _MIXED_NOTE.search(r.get("note") or "") else r["answer"]
    return out


def _field(lab, name):
    if lab is None:
        return None
    return getattr(lab, name) if isinstance(lab, MachineLabel) else lab.get(name)


def score(ex: Export, answers: dict[str, str], labels: dict[str, MachineLabel] | dict[str, dict]) -> dict:
    """Score machine labels (a shadow labels CSV's rows, or MachineLabel objects) on the
    adjudicated sightings.

    - ``cant_tell`` and ``mixed`` answers are excluded from precision and counted.
    - Sightings that carry a human label (in the export, or marked ``human_labeled`` by the
      shadow run) are excluded: production never labels them, and their own label may have
      been a reference.
    - The sample is stratified (equal share per board state x face/no-face), so the pooled
      numbers are reported as is AND re-weighted to the population of unlabeled sightings of
      the same facility and days (``weighted_precision``, ``weighted_coverage``)."""
    by_id = {s.sighting_id: s for s in ex.sightings}
    total, promote, stranger_fa = Tally(), Tally(), Tally()
    per_day: dict[str, Tally] = {}
    by_stratum: dict[str, Tally] = {}
    cant = mixed = mixed_flagged = excluded = 0
    days: set[str] = set()
    facs: set[str] = set()
    for sid, ans in answers.items():
        s = by_id.get(sid)
        if s is None:
            continue
        lab = labels.get(sid)
        if s.human_labeled or str(_field(lab, "human_labeled") or "0") in ("1", "True", "true"):
            excluded += 1
            continue
        days.add(s.day)
        facs.add(s.facility_id)
        if ans == "cant_tell":
            cant += 1
            continue
        if ans == "mixed":
            mixed += 1
            if _field(lab, "relation") == "mixed":
                mixed_flagged += 1
            continue
        state, key = _field(lab, "state"), _field(lab, "person_key")
        if lab is None:
            o = "abstain"
        elif state != "known":
            o = "flagged"
        elif ans == "stranger":
            o = "wrong"
        else:
            o = "correct" if key == ans else "wrong"
        total.add(o, sid)
        per_day.setdefault(s.day, Tally()).add(o, sid)
        by_stratum.setdefault(stratum(s), Tally()).add(o, sid)
        if s.board_state in ("uncertain", "stranger"):
            promote.add(o, sid)
        if ans == "stranger":
            stranger_fa.add("wrong" if o == "wrong" else "abstain", sid)
    # population weights: unlabeled sightings with a crop, same facility and days as the sample
    pop: dict[str, int] = {}
    for s in ex.sightings:
        if s.facility_id in facs and s.day in days and s.human_state is None and s.image_url:
            pop[stratum(s)] = pop.get(stratum(s), 0) + 1
    wc = wl = wn = 0.0
    for st, t in by_stratum.items():
        w = pop.get(st, 0) / t.n if t.n else 0.0
        wc += w * t.correct
        wl += w * t.labeled
        wn += w * t.n
    return {
        "adjudicated": len(answers), "excluded_human_labeled": excluded, "cant_tell": cant,
        "mixed_noted": mixed, "mixed_flagged_by_machine": mixed_flagged,
        "precision": total.precision, "coverage": total.coverage, "correct": total.correct,
        "wrong": total.wrong, "flagged": total.flagged, "abstain": total.abstain,
        "precision_cp_lo": cp_lower(total.correct, total.labeled),
        "precision_lo95_days": bootstrap_precision(per_day),
        "weighted_precision": (wc / wl) if wl else None, "weighted_coverage": (wl / wn) if wn else None,
        "promote_precision": promote.precision, "promote_coverage": promote.coverage, "promote_n": promote.n,
        "promote_cp_lo": cp_lower(promote.correct, promote.labeled),
        "strangers": stranger_fa.n, "stranger_false_accepts": stranger_fa.wrong,
        "by_stratum": {k: {"n": t.n, "population": pop.get(k, 0), "precision": t.precision, "coverage": t.coverage}
                       for k, t in sorted(by_stratum.items())},
        "wrong_ids": total.wrong_ids,
    }
