"""Look at individual sightings the way the decision sees them: every stored measure per
crop, the qf the policy gives it, whether it may be a reference face, the face scores against
each person, and the decision for its day (with the same reference mode as ``eval``)."""

from __future__ import annotations

from collections import defaultdict

from .decide import BodyIndex, Thresholds, decide_day, face_scores
from .evaluate import run_cutoff_ms
from .pipeline import RefPolicy, Workspace, org_facilities, ref_usable


def inspect(ws: Workspace, facility: str, sids: list[str], th: Thresholds, policy: RefPolicy, *,
            refs_mode: str = "temporal", log=print) -> dict:
    ex = ws.export
    by_day: dict[str, list[str]] = defaultdict(list)
    for sid in sids:
        s = ws.sighting(sid)
        if s is None or s.facility_id != facility:
            log(f"{sid}: not a facility {facility} sighting in this export")
            continue
        by_day[s.day].append(sid)
    out = {}
    facs = org_facilities(ex, facility)
    tz = ex.facility_tz.get(facility, "Asia/Shanghai")
    for day, day_sids in sorted(by_day.items()):
        if refs_mode == "temporal":
            refs = ws.references(facs, day, label_cutoff_ms=run_cutoff_ms(day, tz), policy=policy)
        else:
            refs = ws.references(facs, day, label_cutoff_ms=None, policy=policy, exclude_days={day}, allow_after=True)
        sightings = ex.by_facility_day(facility, day)
        probe_faces, body, body_crops = ws.probe_inputs(facility, day)
        fs = face_scores(probe_faces, refs, th)
        labels = {l.sighting_id: l for l in decide_day(sightings, fs, body, body_crops, ex.roster(facility), ex.key, th,
                                                       body_index=BodyIndex.build(sightings, body) if body else None)}
        ft = ws.face_table(facility, day)
        log(f"=== {facility} {day} (refs {refs_mode}: " +
            ", ".join(f"{k[-6:]}:{refs.n_sightings(k)}" for k in sorted(refs)) + ")")
        for sid in day_sids:
            s = ws.sighting(sid)
            human = ex.key(s.human_person_id) if s.human_state == "known" else s.human_state
            log(f"\n{sid}  board={s.board_state}:{(ex.key(s.board_person_id) or '-')}  human={human or '-'}  "
                f"crop_is_face={int(s.crop_is_face)}  image={s.image_url}")
            rows = [i for i in range(len(ft)) if str(ft.sighting_id[i]) == sid]
            if not rows:
                log("  no detected face in any crop")
            for i in rows:
                log(f"  crop {ft.source[i]}:{int(ft.crop_idx[i])}  qf={ft.qf[i]:.3f}  det={ft.det[i]:.3f}  "
                    f"face_h={ft.face_h[i]:.0f}  blur={ft.blur[i]:.0f}  residual={ft.residual[i]:.2f}  "
                    f"asym={ft.asym[i]:.2f}  yaw={ft.yaw[i]:.2f}  norm={ft.norm[i]:.2f}  "
                    f"faces={int(ft.n_faces[i])}  second={ft.second_ratio[i]:.2f}@{ft.second_det[i]:.2f}  "
                    f"reference-usable={'yes' if ref_usable(ft, i, policy) else 'no'}")
            sc = fs.score.get(sid)
            if sc is not None:
                order = sc.argsort()[::-1]
                log("  face scores: " + ", ".join(f"{fs.persons[j][-6:]}={sc[j]:.3f}(n={fs.n_refs[j]})" for j in order[:4]))
            elif rows:
                log("  face scores: none (no crop passes face_q_min / face_min_px / min_face_norm)")
            lab = labels.get(sid)
            log(f"  decision: {'abstain' if lab is None else f'{lab.state}/{lab.relation} via {lab.via} -> {lab.person_key}'}"
                + (f"  {lab.evidence}" if lab is not None and lab.evidence else ""))
            out[sid] = {"label": lab, "scores": None if sc is None else dict(zip(fs.persons, map(float, sc)))}
    return out
