"""Export loading, references, evaluation and the adjudication page, end to end on synthetic data."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from helpers import FACE_ID, basis, near, write_export, write_face_cache
from vbb_reid_server.adjudicate import read_answers, sample, score, write_page
from vbb_reid_server.align import ARCFACE_DST, norm_crop, umeyama_similarity
from vbb_reid_server.data import load_export
from vbb_reid_server.decide import Thresholds
from vbb_reid_server.evaluate import curve_rows, evaluate_facility, operating_point, run_cutoff_ms
from vbb_reid_server.pipeline import Workspace

RNG = np.random.default_rng(1)
TZ = ZoneInfo("Asia/Shanghai")


def ms(day: str, hour: float) -> int:
    return int((datetime.fromisoformat(day).replace(tzinfo=TZ).timestamp() + hour * 3600) * 1000)


PEOPLE = [("f2A", "2", "visitor"), ("f2B", "2", "visitor"), ("f3A", "3", "visitor"), ("f3B", "3", "visitor")]


def build(tmp_path, *, extra_rows=()):
    """Facility 2 labels A and B on 10-10; facility 3 has targets on 10-20 (labeled 10-22)."""
    rows, face_rows = [], {}
    def add(sid, fac, day, hour, *, state="stranger", pid="", cstate="", cpid="", cts=None, vec=None, did=None):
        rows.append({"sighting_id": sid, "facility_id": fac, "did": did or ("43" if fac == "2" else "45"),
                     "capture_ts": ms(day, hour), "track_started_ts": ms(day, hour) - 2000,
                     "track_ended_ts": ms(day, hour), "identity_state": state, "person_id": pid,
                     "corrected_state": cstate, "corrected_person_id": cpid,
                     "corrected_at": "x" if cstate else "", "corrected_ts": cts or "",
                     "image_url": f"https://cdn.example/{sid}.jpg"})
        if vec is not None:
            face_rows.setdefault((fac, day), []).append((sid, 0, vec, 0.9, 60.0))
    for i in range(6):
        add(f"refA{i}", "2", "2026-10-10", 9 + i * 0.1, cstate="known", cpid="f2A", cts=ms("2026-10-10", 20),
            vec=near(basis(0), 0.1, RNG))
        add(f"refB{i}", "2", "2026-10-10", 10 + i * 0.1, cstate="known", cpid="f2B", cts=ms("2026-10-10", 20),
            vec=near(basis(1), 0.1, RNG))
    for i in range(5):
        add(f"tA{i}", "3", "2026-10-20", 9 + i * 0.1, cstate="known", cpid="f3A", cts=ms("2026-10-22", 10),
            vec=near(basis(0), 0.1, RNG))
    add("tB0", "3", "2026-10-20", 11, cstate="known", cpid="f3B", cts=ms("2026-10-22", 10), vec=near(basis(1), 0.1, RNG))
    add("unl0", "3", "2026-10-20", 12, state="uncertain", vec=near(basis(0), 0.1, RNG))
    for r in extra_rows:
        rows.append(r)
    exp = tmp_path / "export"
    write_export(exp, rows, PEOPLE)
    (tmp_path / "aliases.csv").write_text("person_id,human,name\nf2A,H1,</script><b>甲</b>\nf3A,H1,</script><b>甲</b>\nf2B,H2,乙\nf3B,H2,乙\n",
                                          encoding="utf-8")
    emb = tmp_path / "emb"
    for (fac, day), fr in face_rows.items():
        write_face_cache(emb, fac, day, fr)
    return exp, emb


def test_export_roster_includes_visitors_and_aliases_join_facilities(tmp_path):
    exp, _ = build(tmp_path)
    ex = load_export(exp, tmp_path / "aliases.csv")
    assert ex.roster("3") == {"H1": "f3A", "H2": "f3B"}
    assert ex.key("f2A") == ex.key("f3A") == "H1"
    s = next(s for s in ex.sightings if s.sighting_id == "tA0")
    assert s.day == "2026-10-20" and s.human_state == "known"


def test_temporal_references_respect_label_time_and_day(tmp_path):
    exp, emb = build(tmp_path)
    ex = load_export(exp, tmp_path / "aliases.csv")
    ws = Workspace(ex, emb, FACE_ID, None)
    r = ws.references(["2", "3"], "2026-10-20", label_cutoff_ms=run_cutoff_ms("2026-10-20", "Asia/Shanghai"))
    # facility 2's labels (made 10-10) count; facility 3's own labels (made 10-22) do not exist yet.
    assert r["H1"].shape[0] == 6 and r["H2"].shape[0] == 6
    r_none = ws.references(["2", "3"], "2026-10-10", label_cutoff_ms=run_cutoff_ms("2026-10-10", "Asia/Shanghai"))
    assert r_none == {}  # same-day labels are never temporal references


def test_evaluate_scores_targets_and_counts_leave_one_out_false_accepts(tmp_path):
    exp, emb = build(tmp_path)
    ex = load_export(exp, tmp_path / "aliases.csv")
    ws = Workspace(ex, emb, FACE_ID, None)
    grid = {"face_sim": [0.5], "face_margin": [0.1], "body_enabled": [False]}
    res = evaluate_facility(ws, "3", refs_mode="temporal", grid=grid, base=Thresholds(min_refs=3),
                            days=["2026-10-20"], log=lambda *_: None)
    r = res[0]
    assert r.total.n == 6 and r.total.correct == 6 and r.total.wrong == 0
    assert r.unlabeled_promotions == 1
    # With H1 removed, H1's 5 targets must not be named H2: zero false accepts on well-separated people.
    assert r.loo.n == 6 and r.loo.wrong == 0
    rows = curve_rows(res, "synthetic", "temporal", "3")
    assert operating_point(rows, 0.0, 0.01) is not None


def test_leave_one_out_catches_a_lookalike(tmp_path):
    # Probes of H1 that also look like H2 must be counted as false accepts once H1 is hidden.
    exp, emb = build(tmp_path)
    ex = load_export(exp, tmp_path / "aliases.csv")
    ws = Workspace(ex, emb, FACE_ID, None)
    grid = {"face_sim": [0.0], "face_margin": [0.0], "body_enabled": [False]}
    res = evaluate_facility(ws, "3", refs_mode="temporal", grid=grid, base=Thresholds(min_refs=1),
                            days=["2026-10-20"], log=lambda *_: None)
    assert res[0].loo.wrong > 0  # a zero floor names everyone: LOO must see it


def test_adjudication_sample_and_page_are_blind(tmp_path):
    exp, _ = build(tmp_path)
    ex = load_export(exp, tmp_path / "aliases.csv")
    picked = sample(ex, "3", 10)
    assert [s.sighting_id for s in picked] == ["unl0"]  # labeled sightings are never sampled
    page = tmp_path / "page.html"
    write_page(ex, "3", ["unl0"], page, set_id="t1")
    text = page.read_text(encoding="utf-8")
    assert "unl0" in text and "confidence" not in text and "face_sim" not in text
    # A name cannot close the <script> element or inject markup; it is rendered via textContent.
    assert text.count("</script>") == 1 and "<b>甲</b>" not in text


def test_score_counts_stranger_false_accepts_and_skips_cant_tell(tmp_path):
    exp, _ = build(tmp_path)
    ex = load_export(exp, tmp_path / "aliases.csv")
    labels = {"unl0": {"state": "known", "person_key": "H1"}, "tA0": {"state": "known", "person_key": "H1"}}
    res = score(ex, {"unl0": "stranger", "tA0": "H1", "tA1": "cant_tell"}, labels)
    assert res["wrong"] == 1 and res["correct"] == 1 and res["cant_tell"] == 1
    assert res["stranger_false_accepts"] == 1


def test_umeyama_recovers_a_similarity_transform():
    theta, s, t = 0.3, 1.7, np.array([5.0, -3.0])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = ARCFACE_DST.astype(np.float64)
    dst = (s * (R @ src.T)).T + t
    m = umeyama_similarity(src, dst)
    assert np.allclose(m[:, :2], s * R, atol=1e-6) and np.allclose(m[:, 2], t, atol=1e-6)


def test_norm_crop_of_an_aligned_face_is_identity():
    img = (RNG.random((112, 112, 3)) * 255).astype(np.uint8)
    out, residual = norm_crop(img, ARCFACE_DST.copy())
    assert residual < 1e-4 and np.abs(out.astype(int) - img.astype(int)).mean() < 1.0


def test_two_people_note_makes_an_answer_mixed(tmp_path):
    p = tmp_path / "answers.csv"
    p.write_text('sighting_id,answer,note\n"a","H2","2 people, 昌爷 and 史迪仔"\n"b","cant_tell","2 people "\n'
                 '"c","H1",""\n"d","H1","两个人"\n', encoding="utf-8")
    assert read_answers(p) == {"a": "mixed", "b": "mixed", "c": "H1", "d": "mixed"}
