"""Nightly regroup: sort the pictures inside each named person (docs/DESIGN_nightly_regroup.md).

Starting from the board's names (and any human confirmation), clean each person's group so it holds one person, then
give every picture of the night a name only when it clearly fits exactly one group. Pure numpy: no I/O, no randomness,
the same pictures in any order give the same answer, and nothing here reads an earlier night's output.
"""

from __future__ import annotations

import itertools
import math
import numbers
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Item:
    sid: str
    probe: bool
    weak: str | None  # the name it carries now that nobody confirmed
    anchor: str | None  # the person a human confirmed for this very picture: final
    face: np.ndarray | None  # unit vector of one clear face, else None
    body: np.ndarray | None  # unit vector; the caller passes it for probe-day items only
    face_quality: float | None = None  # how clear the face is (higher = clearer); None = unknown


@dataclass(frozen=True)
class Params:
    fmr: float = 0.005  # share of a group's impostors allowed above its floor
    min_core: int = 5  # members with a vector a group needs to act
    min_anchors: int = 3  # anchors that make a group of a person without one, and keep it alive
    min_impostors: int = 30  # impostor scores a group needs before it may name anything
    rounds: int = 10  # cap on cleaning rounds
    dup_share: float = 0.25  # two names look alike when each one's pictures look this much like the other's
    body_topk: int = 3
    hard_share: float = 0.25  # share of the least clear faces judged by floors of their own (F5); 0 = one class

    def __post_init__(self):
        if not 0 < self.fmr < 1:
            raise ValueError(f"fmr must be inside (0, 1), got {self.fmr}")
        if not 0 < self.dup_share <= 1:
            raise ValueError(f"dup_share must be inside (0, 1], got {self.dup_share}")
        if not 0 <= self.hard_share < 1:
            raise ValueError(f"hard_share must be inside [0, 1), got {self.hard_share}")
        if self.min_core < 2:  # a group of one cannot be scored without itself
            raise ValueError(f"min_core must be at least 2, got {self.min_core}")
        for name in ("min_anchors", "min_impostors", "rounds", "body_topk"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")


@dataclass(frozen=True)
class Name:
    sid: str
    person: str | None  # None = the machine makes no claim
    via: str | None  # "anchor" | "face" | "body" | None
    # named: that person's score; ambiguous / not_nearest: the best CLEARED group's; otherwise the picture's best
    score: float | None
    floor: float | None  # the floor that score was compared with
    reason: str  # anchor named disputed ambiguous not_nearest below_floor no_floor no_groups no_vector
    # ambiguous: every person whose floor it cleared, best first; disputed: the names; not_nearest: the cleared
    # group's names, then the nearest group's; a face that cleared no bar (below_floor / no_floor): the nearest
    # group's names (where it points; clothes answers have none)
    others: tuple[str, ...]


@dataclass(frozen=True)
class GroupReport:
    person: str  # for a disputed set: the names joined with "|"
    n_weak: int  # pictures with a face (body, for bodies) whose weak name is this person and have no anchor
    n_anchor: int  # anchors with a face (body)
    n_members: int  # members after cleaning (bodies: examples)
    anchor_seeded: bool
    active: bool
    floor: float | None  # the bar for easy pictures; None when it could not be set
    coverage: float | None  # share of its own members whose leave-one-out score clears the bar of their own class
    disputed_with: tuple[str, ...]
    floor_hard: float | None = None  # the bar for hard pictures (faces only)


@dataclass(frozen=True)
class Result:
    names: dict[str, Name]  # probe sids only
    face_groups: dict[str, GroupReport]
    body_groups: dict[str, GroupReport]
    disputes: tuple[tuple[str, str, float, float], ...]  # (a, b, look(b into a), look(a into b)), a < b
    displaced: tuple[tuple[str, str], ...]  # (weak name, the confirmed person its group belongs to)
    refuted: tuple[str, ...]  # names whose own group was dropped: their person is confirmed elsewhere
    dissolved: tuple[tuple[str, tuple[str, ...]], ...]  # (a pile's weak names joined by "|", the people who chose it)
    rounds: int  # cleaning rounds run


def _unit(x) -> np.ndarray | None:
    """The vector scaled to length 1, or None when it is not usable (one bad row must not stop the night)."""
    try:
        a = np.asarray(x)
        if a.dtype.kind not in "fiu":
            return None
        a = a.astype(float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else None


def _quality(x) -> float | None:
    """A face quality as a float, or None when it is not a finite real number (a bool is not one)."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        return None
    try:
        q = float(x)
    except (OverflowError, ValueError):
        return None
    return q if math.isfinite(q) else None


def _prepare(items: Iterable[Item]) -> list[Item]:
    """Items with clean vectors, in sid order (so no float sum or tie depends on the order given)."""
    rows = [replace(it, face=_unit(it.face), body=_unit(it.body), face_quality=_quality(it.face_quality))
            for it in items]
    for r in rows:
        if any(name is not None and "|" in name for name in (r.weak, r.anchor)):
            raise ValueError(f"a person name of {r.sid} contains '|', which joins the names of a disputed set")
    again = sorted(s for s, n in Counter(r.sid for r in rows).items() if n > 1)
    if again:
        raise ValueError(f"repeated sid: {again[0]}")
    rows.sort(key=lambda r: r.sid)
    for field in ("face", "body"):
        sizes = Counter(getattr(r, field).size for r in rows if getattr(r, field) is not None)
        if len(sizes) > 1:
            usual = min(sizes, key=lambda d: (-sizes[d], d))
            odd = next(r for r in rows if getattr(r, field) is not None and getattr(r, field).size != usual)
            raise ValueError(f"{field} of {odd.sid} has {getattr(odd, field).size} values, the others have {usual}")
    return rows


def _onehot(held: np.ndarray, n_groups: int) -> np.ndarray:
    return np.arange(n_groups)[None, :] == held[:, None]


def _scores(V: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Face score of every picture against every group's centroid; C[i, g] says picture i is in g's centroid."""
    sums = C.T.astype(float) @ V
    dots = V @ sums.T
    sq = np.einsum("gd,gd->g", sums, sums)[None, :]
    own = np.einsum("nd,nd->n", V, V)[:, None]
    loo = C  # a member is scored against its group without itself
    dot = np.where(loo, dots - own, dots)
    norm2 = np.where(loo, sq - 2 * dots + own, sq)
    out = np.full(dots.shape, -np.inf)
    ok = norm2 > 1e-12
    out[ok] = dot[ok] / np.sqrt(norm2[ok])
    return out


def _floors(S: np.ndarray, owner: np.ndarray, active: np.ndarray, p: Params,
            mask: np.ndarray | None = None) -> list[float | None]:
    """Per group: the score that only a share fmr of the other groups' members exceed (None = too few to say).

    The higher of a counted and a fitted estimate (F5). ``mask`` picks one quality class: its impostors only, or all
    impostors when the class has fewer than min_impostors of them."""
    out: list[float | None] = []
    z = statistics.NormalDist().inv_cdf(1 - p.fmr)
    for g in range(S.shape[1]):
        others = (owner >= 0) & (owner != g) & active[g]
        x = S[others & mask, g] if mask is not None else S[others, g]
        x = x[np.isfinite(x)]
        if mask is not None and len(x) < p.min_impostors:
            x = S[others, g]
            x = x[np.isfinite(x)]
        if len(x) < p.min_impostors:
            out.append(None)
            continue
        k = int(p.fmr * len(x) + 1e-9)
        counted = float(np.sort(x)[len(x) - 1 - k])  # covers a tail heavier than normal
        fitted = float(x.mean() + z * x.std())  # covers few scores, where the count is just the highest one
        out.append(max(counted, fitted))
    return out


def _coverage(scores: np.ndarray, floor: float | None) -> float | None:
    return float(np.mean(scores > floor)) if floor is not None and len(scores) else None


def _hard_mask(faces: list[Item], p: Params) -> np.ndarray:
    """Which faces are hard: strictly below the hard_share quantile of the known qualities of the work set."""
    q = np.array([r.face_quality if r.face_quality is not None else np.nan for r in faces], dtype=float)
    known = np.array([r.face_quality is not None for r in faces], dtype=bool)  # _prepare kept only finite numbers
    if p.hard_share == 0 or not known.any():
        return np.zeros(len(faces), dtype=bool)
    return known & (q < np.quantile(q[known], p.hard_share))


def _verdict(sid: str, s: np.ndarray, floors, active, labels, via: str) -> Name:
    """The answer for one picture from its scores against every group. Only a group with a floor can claim it."""
    live = [g for g in range(len(labels)) if active[g]]
    cleared = sorted((g for g in live if floors[g] is not None and s[g] > floors[g]), key=lambda g: (-s[g], g))
    top = min(live, key=lambda g: (-s[g], g))
    best = cleared[0] if cleared else top
    score = float(s[best]) if np.isfinite(s[best]) else None

    def none(reason, others=()):
        return Name(sid, None, None, score, floors[best], reason, others)

    if len(cleared) > 1:  # exactly one group may claim a picture
        return none("ambiguous", tuple(n for g in cleared for n in labels[g]))
    if cleared:
        if best != top:  # it clears this group's bar but is nearer to another group: not a name
            return none("not_nearest", tuple(labels[best]) + tuple(labels[top]))
        if len(labels[best]) > 1:  # a disputed set never gives a name
            return none("disputed", labels[best])
        return Name(sid, labels[best][0], via, score, floors[best], "named", ())
    # a face that fits nobody still points at its nearest group (B4 lets clothes name only people it points at)
    return none("no_floor" if floors[top] is None else "below_floor", tuple(labels[top]) if via == "face" else ())


def _face_step(faces: list[Item], p: Params):
    """F1-F6, group first and name second. Returns (names of every non-anchored face, group reports, disputes,
    displaced, refuted, dissolved, rounds, disputed sets)."""
    n = len(faces)
    V = np.stack([r.face for r in faces])
    hard = _hard_mask(faces, p)
    is_anchor = np.array([r.anchor is not None for r in faces])
    weak = [r.weak if r.anchor is None else None for r in faces]  # F1: an anchored picture forms no weak group
    names = sorted({w for w in weak if w is not None} | {r.anchor for r in faces if r.anchor is not None})
    wid = {q: i for i, q in enumerate(names)}
    wheld = np.array([wid[w] if w is not None else -1 for w in weak])
    wactive = np.bincount(wheld[wheld >= 0], minlength=len(names)) >= p.min_core
    held0 = np.where((wheld >= 0) & wactive[np.maximum(wheld, 0)], wheld, -1)

    # F2: look-alike weak names are joined into merged groups
    S0 = _scores(V, _onehot(held0, len(names)))
    parent = list(range(len(names)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    looks = []
    act = [g for g in range(len(names)) if wactive[g]]
    for a, b in itertools.combinations(act, 2):
        look_ba = float(np.mean(S0[held0 == b, a] >= np.median(S0[held0 == a, a])))
        look_ab = float(np.mean(S0[held0 == a, b] >= np.median(S0[held0 == b, b])))
        if look_ba >= p.dup_share and look_ab >= p.dup_share:
            looks.append((names[a], names[b], look_ba, look_ab))
            parent[max(root(a), root(b))] = min(root(a), root(b))
    sets: dict[int, list[str]] = {}
    for g in act:
        sets.setdefault(root(g), []).append(names[g])
    parts = sorted(tuple(v) for v in sets.values())  # the weak groups, look-alikes merged
    weak_group = {q: g for g, lab in enumerate(parts) for q in lab}
    gheld = np.array([weak_group.get(names[h], -1) if h >= 0 else -1 for h in held0])

    # F3: every anchor votes for the weak group it fits best, if it looks like a typical picture of it
    persons = sorted({r.anchor for r in faces if r.anchor is not None})
    home: dict[str, int] = {}
    if parts:
        Sg = _scores(V, _onehot(gheld, len(parts)))
        own = np.array([np.median(Sg[gheld == g, g]) for g in range(len(parts))])
        for person in persons:
            mine = [i for i in range(n) if faces[i].anchor == person]
            votes = Counter()
            for i in mine:
                best = int(np.argmax(Sg[i]))
                if np.isfinite(Sg[i, best]) and Sg[i, best] >= own[best]:  # the "typical picture" condition
                    votes[best] += 1
            top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
            need = max(2, math.ceil(p.dup_share * len(mine)))  # a single vote is never a home
            if top and top[0][1] >= need and (len(top) == 1 or top[1][1] < top[0][1]):
                home[person] = top[0][0]
    labels: list[tuple[str, ...]] = []
    made_from: list[int | None] = []  # the weak group each label came from
    displaced, refuted, dissolved = set(), set(), set()
    for g, lab in enumerate(parts):
        homers = [q for q in persons if home.get(q) == g]
        if len(homers) == 1:  # the confirmed person's group; the other weak names here are displaced
            labels.append((homers[0],))
            displaced |= {(q, homers[0]) for q in lab if q != homers[0]}
        elif len(homers) >= 2:  # a mixed pile, several people's confirmed pictures match its blend: dissolved
            dissolved.add(("|".join(lab), tuple(homers)))
            continue
        elif len(lab) > 1 or lab[0] not in home:  # nobody's home: a merged set stays disputed, a single name stays
            labels.append(lab)
        else:  # this name's person was confirmed in ANOTHER group: the name is refuted here
            refuted.add(lab[0])
            continue
        made_from.append(g)
    cand = np.full(n, -1)
    label_of_weak = {g: li for li, g in enumerate(made_from)}
    for i in range(n):
        if gheld[i] in label_of_weak and not is_anchor[i]:
            cand[i] = label_of_weak[gheld[i]]
    own_group = set()
    for person in persons:  # a confirmed person nobody's group is labelled for gets a group of their own
        if any(lab == (person,) for lab in labels):  # a disputed set containing the name does not count
            continue
        n_a = sum(1 for r in faces if r.anchor == person)
        lone = (wheld == wid[person]) & ~is_anchor & ~wactive[wid[person]]  # their own weak pictures, of a dead group
        if n_a >= p.min_anchors or n_a + int(lone.sum()) >= p.min_core:
            labels.append((person,))
            own_group.add(len(labels) - 1)
            cand[lone] = len(labels) - 1
    G = len(labels)
    if G == 0:
        nobody = {r.sid: Name(r.sid, None, None, None, None, "no_groups", ()) for r in faces if r.anchor is None}
        return nobody, {}, (), (), (), tuple(sorted(dissolved)), 0, {}
    where = {lab[0]: li for li, lab in enumerate(labels) if len(lab) == 1}  # exactly the person's name, never a set
    for i in np.flatnonzero(is_anchor):
        cand[i] = where.get(faces[i].anchor, -1)  # an anchor is a clamped member of its person's group, if any
    n_anc = np.bincount(cand[is_anchor & (cand >= 0)], minlength=G)

    # F4: cleaning only removes
    core, active, rounds = cand.copy(), np.ones(G, dtype=bool), 0
    for rounds in range(1, p.rounds + 1):
        core = np.where((core >= 0) & active[np.maximum(core, 0)], core, -1)
        S = _scores(V, _onehot(core, G))
        S[:, ~active] = -np.inf
        fe, fh = ([_floors(S, core, active, p, ~hard), _floors(S, core, active, p, hard)] if rounds > 1
                  else ([None] * G, [None] * G))
        new = np.full(n, -1)
        for i in np.flatnonzero((cand >= 0) & active[np.maximum(cand, 0)]):
            g = cand[i]
            if is_anchor[i]:
                new[i] = g
                continue
            fl = fh if hard[i] else fe  # a picture is only compared with the bars of its own class
            if fl[g] is None:  # first round, or too few others to set a bar: stay while this group fits best
                stay = bool(np.isfinite(S[i, g]) and S[i, g] >= S[i].max())
            else:
                stay = [h for h in range(G) if active[h] and fl[h] is not None and S[i, h] > fl[h]] == [g]
            if stay:
                new[i] = g
        small = active & (n_anc < p.min_anchors) & (np.bincount(new[new >= 0], minlength=G) < p.min_core)
        if small.any():
            active = active & ~small
            new = np.where((new >= 0) & active[np.maximum(new, 0)], new, -1)
        same = bool((new == core).all())
        core = new
        if same and rounds > 1 and not small.any():
            break

    # F5, F6
    S = _scores(V, _onehot(core, G))
    S[:, ~active] = -np.inf
    fe, fh = _floors(S, core, active, p, ~hard), _floors(S, core, active, p, hard)
    out = {}
    for i, r in enumerate(faces):
        if r.anchor is None:
            out[r.sid] = (Name(r.sid, None, None, None, None, "no_groups", ()) if active.sum() < 2
                          else _verdict(r.sid, S[i], fh if hard[i] else fe, active, labels, "face"))
    reports = {}
    for g, lab in enumerate(labels):
        key = "|".join(lab)
        mine = core == g
        bar = None if fe[g] is None or fh[g] is None else np.where(hard[mine], fh[g], fe[g])  # each member's own class
        reports[key] = GroupReport(key, int(((cand == g) & ~is_anchor).sum()), int(n_anc[g]), int(mine.sum()),
                                   g in own_group, bool(active[g]), fe[g], _coverage(S[mine, g], bar),
                                   lab if len(lab) > 1 else (), fh[g])
    settled = {lab for lab in labels if len(lab) > 1}  # a look-alike pair is a dispute only while its set is disputed
    disputes = sorted(d for d in looks if any(d[0] in lab and d[1] in lab for lab in settled))
    return (out, reports, tuple(disputes), tuple(sorted(displaced)), tuple(sorted(refuted)),
            tuple(sorted(dissolved)), rounds, {"|".join(lab): lab for lab in settled})


def _topk_mean(sub: np.ndarray, k: int) -> np.ndarray:
    """Mean of each row's k highest values (fewer when fewer exist; -inf marks a value that does not exist)."""
    top = -np.sort(-sub, axis=1)[:, :k]
    real = np.isfinite(top)
    count = real.sum(axis=1)
    return np.where(count > 0, np.where(real, top, 0).sum(axis=1) / np.maximum(count, 1), -np.inf)


def _body_step(rows: list[Item], names: dict[str, Name], sets: dict[str, tuple[str, ...]], p: Params):
    """B1-B4. Examples are the probes the faces (or a human) named (a clothes name is never one). Clothes judge every
    probe with a body that has no anchor and no face lock (a ``named`` or ``disputed`` face answer)."""
    brows = [r for r in rows if r.probe and r.body is not None]
    if not brows:
        return {}, {}
    key_of = {}
    for i, r in enumerate(brows):
        nm = names.get(r.sid)
        if nm is not None and nm.person is not None:
            key_of[i] = nm.person
        elif nm is not None and nm.reason == "disputed":
            key_of[i] = "|".join(nm.others)
    keys = sorted(set(key_of.values()))
    labels = [sets.get(k, (k,)) for k in keys]
    gi = {k: g for g, k in enumerate(keys)}
    owner0 = np.array([gi.get(key_of.get(i), -1) for i in range(len(brows))])
    active = np.array([(owner0 == g).sum() >= p.min_core for g in range(len(keys))], dtype=bool)
    owner = np.where(np.isin(owner0, np.flatnonzero(active)), owner0, -1)  # an inactive group holds nothing
    B = np.stack([r.body for r in brows])
    S = np.full((len(brows), len(keys)), -np.inf)
    for g in np.flatnonzero(active):
        cols = np.flatnonzero(owner == g)
        sub = B @ B[cols].T  # only this group's examples, never the full N x N table
        sub[cols, np.arange(len(cols))] = -np.inf  # the picture itself is left out
        S[:, g] = _topk_mean(sub, p.body_topk)
    floors = _floors(S, owner, active, p)
    out = {}
    for i, r in enumerate(brows):
        face = names.get(r.sid)  # the face answer, or None when the picture has no usable face
        if r.anchor is not None or (face is not None and face.reason in ("named", "disputed")):
            continue  # an anchor, or a face lock: clothes never change it
        if int(active.sum()) < 2:
            said = Name(r.sid, None, None, None, None, "no_groups", ())
        else:
            said = _verdict(r.sid, S[i], floors, active, labels, "body")
        if said.person is not None and face is not None and face.others and said.person not in face.others:
            continue  # clothes may only choose among the people the face points at (a no_groups face points nowhere)
        if said.person is not None or face is None:  # no clothes name: a face answer stays as it was
            out[r.sid] = said
    reports = {}
    for g, k in enumerate(keys):
        lab = labels[g]
        reports[k] = GroupReport(
            k, sum(1 for r in brows if r.anchor is None and r.weak in lab), sum(1 for r in brows if r.anchor in lab),
            int((owner0 == g).sum()), False, bool(active[g]), floors[g], _coverage(S[owner0 == g, g], floors[g]),
            lab if len(lab) > 1 else ())
    return out, reports


def regroup(items: Iterable[Item], params: Params = Params()) -> Result:
    rows = _prepare(items)
    names = {r.sid: Name(r.sid, r.anchor, "anchor", None, None, "anchor", ()) for r in rows if r.anchor is not None}
    faces = [r for r in rows if r.face is not None]
    face_names, face_groups, disputes, displaced, refuted, dissolved, rounds, sets = (
        _face_step(faces, params) if faces else ({}, {}, (), (), (), (), 0, {}))
    names.update(face_names)
    body_names, body_groups = _body_step(rows, names, sets, params)
    names.update(body_names)
    probes = {r.sid: names.get(r.sid) or Name(r.sid, None, None, None, None, "no_vector", ()) for r in rows if r.probe}
    return Result(probes, face_groups, body_groups, disputes, displaced, refuted, dissolved, rounds)
