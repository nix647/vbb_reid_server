"""The machine-label decision for one facility-day. Pure numpy; no model calls.

Order of evidence for a sighting (plan §5.3):
1. Mixed track — two crops of the sighting disagree on identity  -> ``flag/mixed``.
2. Own face — aggregated face score clears ``face_sim`` and ``face_margin``
   against the closed roster's references                       -> ``known`` via face.
3. Same-day body cluster — constrained linking over the day's body embeddings;
   a cluster adopts a person only when its seeds (strong face decisions, and
   same-day human labels when allowed) agree on exactly one person; a member
   with no face decision of its own inherits it if it is directly similar to a
   seed                                                         -> ``known`` via body.
4. Otherwise abstain (no row).
Relation to the board: same person -> ``agree``; board uncertain/stranger ->
``promote``; board known someone else -> ``relabel`` (emitted as ``flag`` in v1).
The machine never emits ``stranger`` or ``uncertain``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .data import Sighting


@dataclass(frozen=True)
class Thresholds:
    # face path
    face_sim: float = 0.40
    face_margin: float = 0.10
    face_q_min: float = 0.30
    face_min_px: float = 20.0
    topk: int = 3
    min_refs: int = 3
    # mixed-track check on the crops of one sighting
    mixed_face_sim: float = 0.25  # two usable faces of one sighting below this -> different people
    mixed_body_sim: float | None = None  # off until measured
    # same-day body linking
    body_enabled: bool = True
    body_link: float = 0.80
    body_link_avg: float = 0.70  # average-linkage guard against chaining
    body_knn: int = 15  # candidate edges per sighting
    body_assign: float = 0.82  # member must be this similar to at least one seed
    seed_extra: float = 0.05  # seeds need face_sim + seed_extra
    min_seeds: int = 1
    # emission
    allow_relabel: bool = False  # v1: relabels are flags

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class MachineLabel:
    sighting_id: str
    state: str  # known | flag
    relation: str  # promote | agree | relabel | mixed | conflict
    person_key: str | None
    person_id: str | None
    via: str  # face | body | mixed | conflict
    confidence: float
    margin: float | None
    face_sim: float | None
    body_sim: float | None
    evidence: dict = field(default_factory=dict)


@dataclass
class FaceScores:
    """Per-sighting aggregated face scores against each reference person."""
    persons: list[str]  # identity keys, column order
    n_refs: np.ndarray  # (P,)
    score: dict[str, np.ndarray]  # sighting_id -> (P,) aggregated score
    crop_best: dict[str, list[tuple[int, float, int]]]  # sid -> [(crop_idx, best score, argmax)]
    crop_vecs: dict[str, list[np.ndarray]]  # sid -> usable face vectors (for the mixed check)
    best_q: dict[str, float]


def _topk_mean(sims: np.ndarray, k: int) -> np.ndarray:
    """sims: (M, R) -> (M,) mean of the top-k per row (k clipped to R)."""
    if sims.shape[1] == 0:
        return np.full(sims.shape[0], -1.0)
    k = max(1, min(k, sims.shape[1]))
    part = np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def face_scores(probe_faces: dict[str, list[tuple[int, np.ndarray, float, float]]],
                refs: dict[str, np.ndarray], th: Thresholds) -> FaceScores:
    """probe_faces: sid -> [(crop_idx, unit vec, qf, face_h)]; refs: key -> (R, D) unit vecs."""
    persons = sorted(refs)
    n_refs = np.array([refs[p].shape[0] for p in persons], dtype=int)
    score: dict[str, np.ndarray] = {}
    crop_best: dict[str, list[tuple[int, float, int]]] = {}
    crop_vecs: dict[str, list[np.ndarray]] = {}
    best_q: dict[str, float] = {}
    for sid, faces in probe_faces.items():
        usable = [(c, v, q) for c, v, q, h in faces if q >= th.face_q_min and h >= th.face_min_px]
        best_q[sid] = max((q for _, _, q, _ in faces), default=0.0)
        if not usable or not persons:
            continue
        V = np.stack([v for _, v, _ in usable])
        per = np.stack([_topk_mean(V @ refs[p].T, th.topk) for p in persons], axis=1)  # (C, P)
        w = np.array([q for _, _, q in usable])
        score[sid] = (per * w[:, None]).sum(0) / w.sum()
        crop_best[sid] = [(c, float(per[i].max()), int(per[i].argmax())) for i, (c, _, _) in enumerate(usable)]
        crop_vecs[sid] = [v for _, v, _ in usable]
    return FaceScores(persons, n_refs, score, crop_best, crop_vecs, best_q)


def _face_decision(fs: FaceScores, sid: str, th: Thresholds, extra: float = 0.0):
    s = fs.score.get(sid)
    if s is None:
        return None
    order = np.argsort(s)[::-1]
    best = int(order[0])
    top = float(s[best])
    second = float(s[order[1]]) if s.shape[0] > 1 else -1.0
    margin = top - second
    if top >= th.face_sim + extra and margin >= th.face_margin and fs.n_refs[best] >= th.min_refs:
        return fs.persons[best], top, margin
    return None


def _is_mixed(fs: FaceScores, sid: str, th: Thresholds, body_crops: list[np.ndarray] | None):
    vecs = fs.crop_vecs.get(sid, [])
    reasons = []
    if len(vecs) >= 2:
        sims = [float(vecs[i] @ vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        if min(sims) < th.mixed_face_sim:
            reasons.append(f"face_pair_sim={min(sims):.3f}")
        decided = {b[2] for b in fs.crop_best.get(sid, []) if b[1] >= th.face_sim}
        if len(decided) > 1:
            reasons.append("crops_name_different_people")
    if th.mixed_body_sim is not None and body_crops and len(body_crops) >= 2:
        sims = [float(body_crops[i] @ body_crops[j]) for i in range(len(body_crops)) for j in range(i + 1, len(body_crops))]
        if min(sims) < th.mixed_body_sim:
            reasons.append(f"body_pair_sim={min(sims):.3f}")
    return reasons


def _cannot_link(sightings: list[Sighting]) -> np.ndarray:
    """Two sightings tracked on the same board at overlapping times are two people."""
    n = len(sightings)
    cl = np.zeros((n, n), dtype=bool)
    by_did: dict[str, list[int]] = {}
    for i, s in enumerate(sightings):
        by_did.setdefault(s.did, []).append(i)
    for idxs in by_did.values():
        iv = sorted(idxs, key=lambda i: sightings[i].interval_ms[0])
        for a_pos, a in enumerate(iv):
            a0, a1 = sightings[a].interval_ms
            for b in iv[a_pos + 1:]:
                b0, _ = sightings[b].interval_ms
                if b0 > a1:
                    break
                cl[a, b] = cl[b, a] = True
    return cl


def body_clusters(sightings: list[Sighting], body: dict[str, np.ndarray], seeds: dict[str, str],
                  th: Thresholds) -> tuple[dict[str, int], dict[int, str | None], np.ndarray, list[str]]:
    """Constrained average-guarded linking. Returns (sid -> cluster id, cluster id -> person or
    None, similarity matrix, node order). ``seeds``: sid -> identity key that must not mix."""
    nodes = [s for s in sightings if s.sighting_id in body]
    sids = [s.sighting_id for s in nodes]
    n = len(nodes)
    if n == 0:
        return {}, {}, np.zeros((0, 0)), []
    B = np.stack([body[s] for s in sids])
    S = B @ B.T
    cl = _cannot_link(nodes)
    parent = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    mask: dict[int, np.ndarray] = {}
    forbid: dict[int, np.ndarray] = {}
    person: dict[int, str | None] = {i: seeds.get(sids[i]) for i in range(n)}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def m_of(r: int) -> np.ndarray:
        if r not in mask:
            v = np.zeros(n, dtype=bool)
            v[members[r]] = True
            mask[r] = v
        return mask[r]

    def f_of(r: int) -> np.ndarray:
        if r not in forbid:
            forbid[r] = cl[members[r]].any(axis=0)
        return forbid[r]

    # Candidate edges: each node's k nearest neighbours above body_link (bounds the
    # edge count at n*k; appearance encoders put many unrelated pairs above any
    # fixed threshold).
    k_nn = max(1, min(th.body_knn, n - 1)) if n > 1 else 0
    pairs: set[tuple[int, int]] = set()
    if k_nn:
        Sm = S.copy()
        np.fill_diagonal(Sm, -np.inf)
        nn = np.argpartition(-Sm, k_nn - 1, axis=1)[:, :k_nn]
        for i in range(n):
            for j in nn[i]:
                j = int(j)
                if Sm[i, j] >= th.body_link:
                    pairs.add((min(i, j), max(i, j)))
    if pairs:
        pa_ = np.array(sorted(pairs))
        ei, ej = pa_[:, 0], pa_[:, 1]
        ev = S[ei, ej]
    else:
        ei = ej = np.zeros(0, dtype=int)
        ev = np.zeros(0)
    rejected: set[tuple[int, int]] = set()
    for k in np.argsort(-ev, kind="stable"):
        ra, rb = find(int(ei[k])), find(int(ej[k]))
        if ra == rb:
            continue
        key = (min(ra, rb), max(ra, rb))
        if key in rejected:
            continue
        pa, pb = person[ra], person[rb]
        if pa is not None and pb is not None and pa != pb:
            rejected.add(key)
            continue
        if (f_of(ra) & m_of(rb)).any():
            rejected.add(key)
            continue
        if S[np.ix_(members[ra], members[rb])].mean() < th.body_link_avg:
            rejected.add(key)
            continue
        # merge rb into ra (keep the larger as root)
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        ma, mb, fa, fb = m_of(ra), m_of(rb), f_of(ra), f_of(rb)
        parent[rb] = ra
        members[ra].extend(members.pop(rb))
        mask[ra] = ma | mb
        forbid[ra] = fa | fb
        mask.pop(rb, None)
        forbid.pop(rb, None)
        if person[ra] is None:
            person[ra] = person[rb]
        person.pop(rb, None)
        rejected = {r for r in rejected if rb not in r}
    cluster_of = {sids[i]: find(i) for i in range(n)}
    cluster_person = {r: person.get(r) for r in set(cluster_of.values())}
    return cluster_of, cluster_person, S, sids


def decide_day(sightings: list[Sighting], fs: FaceScores, body: dict[str, np.ndarray],
               body_crops: dict[str, list[np.ndarray]], roster: dict[str, str],
               key_of: Callable[[str | None], str | None], th: Thresholds,
               human_seeds: dict[str, str] | None = None) -> list[MachineLabel]:
    """One facility-day. ``roster``: identity key -> this facility's person_id (closed roster).
    ``human_seeds``: same-day human labels (sid -> key) usable as body seeds."""
    out: list[MachineLabel] = []
    face_dec: dict[str, tuple[str, float, float]] = {}
    mixed: dict[str, list[str]] = {}
    for s in sightings:
        reasons = _is_mixed(fs, s.sighting_id, th, body_crops.get(s.sighting_id))
        if reasons:
            mixed[s.sighting_id] = reasons
            continue
        d = _face_decision(fs, s.sighting_id, th)
        if d is not None and d[0] in roster:
            face_dec[s.sighting_id] = d

    # body linking, seeded by strong face decisions (+ same-day human labels)
    cluster_of: dict[str, int] = {}
    cluster_person: dict[int, str | None] = {}
    S = np.zeros((0, 0))
    order: list[str] = []
    if th.body_enabled and body:
        seeds: dict[str, str] = {}
        for sid, (p, top, margin) in face_dec.items():
            if top >= th.face_sim + th.seed_extra:
                seeds[sid] = p
        for sid, p in (human_seeds or {}).items():
            if p in roster and sid not in mixed:
                seeds[sid] = p
        linkable = [s for s in sightings if s.sighting_id not in mixed]
        cluster_of, cluster_person, S, order = body_clusters(linkable, body, seeds, th)
        idx = {sid: i for i, sid in enumerate(order)}
        seed_members: dict[int, list[int]] = {}
        for sid, c in cluster_of.items():
            if sid in seeds:
                seed_members.setdefault(c, []).append(idx[sid])
    else:
        seeds, idx, seed_members = {}, {}, {}

    for s in sightings:
        sid = s.sighting_id
        board_key = key_of(s.board_person_id) if s.board_state == "known" else None
        if sid in mixed:
            out.append(MachineLabel(sid, "flag", "mixed", None, None, "mixed", 0.0, None, None, None,
                                    {"reasons": mixed[sid]}))
            continue
        person, via, conf, margin, fsim, bsim, ev = None, None, 0.0, None, None, None, {}
        if sid in face_dec:
            person, top, margin = face_dec[sid]
            via, conf, fsim = "face", top, top
            ev["face_q"] = round(fs.best_q.get(sid, 0.0), 4)
            c = cluster_of.get(sid)
            cp = cluster_person.get(c) if c is not None else None
            if cp is not None and cp != person:
                out.append(MachineLabel(sid, "flag", "conflict", person, roster.get(person), "conflict",
                                        top, margin, top, None, {"cluster_person": cp}))
                continue
        elif sid in cluster_of:
            c = cluster_of[sid]
            cp = cluster_person.get(c)
            sm = seed_members.get(c, [])
            if cp is not None and len(sm) >= th.min_seeds and cp in roster:
                best = float(S[idx[sid], sm].max())
                if best >= th.body_assign:
                    person, via, conf, bsim = cp, "body", best, best
                    ev["cluster_seeds"] = len(sm)
                    ev["cluster_size"] = sum(1 for v in cluster_of.values() if v == c)
        if person is None:
            continue
        if board_key == person:
            relation, state = "agree", "known"
        elif s.board_state == "known":
            relation = "relabel"
            state = "known" if th.allow_relabel else "flag"
        else:
            relation, state = "promote", "known"
        ev["board_state"] = s.board_state
        out.append(MachineLabel(sid, state, relation, person, roster[person], via, round(conf, 5),
                                None if margin is None else round(margin, 5),
                                None if fsim is None else round(fsim, 5),
                                None if bsim is None else round(bsim, 5), ev))
    return out
