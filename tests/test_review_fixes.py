"""One test per finding of the 2026-11-04 code review (independent reviewer + own pass).
Each fails on 0.1.1."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from helpers import FACE_ID, basis, near, sighting, unit, write_export, write_face_cache
from test_embed import COLORS, StubDetector, StubEmbed, jpg
from vbb_reid_server.adjudicate import score
from vbb_reid_server.crops import CropStore
from vbb_reid_server.data import load_export
from vbb_reid_server.decide import Thresholds, decide_day, face_scores
from vbb_reid_server.embed import embed_day, load_body, load_face
from vbb_reid_server.evaluate import curve_rows, gate_reasons
from vbb_reid_server.metrics import Tally, cp_lower
from vbb_reid_server.models.registry import ModelSet
from vbb_reid_server.pipeline import Refs, Workspace
from vbb_reid_server.shadow import shadow_run

TZ = ZoneInfo("Asia/Shanghai")
RNG = np.random.default_rng(11)


def ms(day, hour):
    return int((datetime.fromisoformat(day).replace(tzinfo=TZ).timestamp() + hour * 3600) * 1000)


# --- 1. preprocessing is part of the model id -------------------------------------------

def test_two_colour_orders_of_one_weights_file_are_two_models(tmp_path):
    import onnx  # noqa: F401  (skip cleanly when onnx is absent)
    from onnx import TensorProto, helper
    from vbb_reid_server.models.onnx_models import OnnxBodyEmbedder, OnnxFaceEmbedder
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3, 112, 112])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 4])
    node = helper.make_node("ReduceMean", ["x"], ["m"], axes=[2, 3], keepdims=0)
    pad = helper.make_node("Pad", ["m", "pads"], ["y"])
    g = helper.make_graph([node, pad], "g", [x], [y],
                          initializer=[helper.make_tensor("pads", TensorProto.INT64, [4], [0, 0, 0, 1])])
    p = tmp_path / "m.onnx"
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # loadable by any onnxruntime >= 1.12
    onnx.save(model, str(p))
    a = OnnxFaceEmbedder(p, color="BGR").model_id
    b = OnnxFaceEmbedder(p, color="RGB").model_id
    c = OnnxFaceEmbedder(p, color="RGB", mean=0.0, std=255.0).model_id
    assert len({a, b, c}) == 3
    assert OnnxBodyEmbedder(p, height=256).model_id != OnnxBodyEmbedder(p, height=384).model_id


# --- 2. shadow sees only labels that existed at the nightly run ---------------------------

def test_shadow_does_not_use_labels_made_after_the_run(tmp_path):
    day = "2026-10-20"
    rows, face_rows = [], []
    for i in range(4):  # same-day sightings of A, labeled a week later
        rows.append({"sighting_id": f"t{i}", "facility_id": "3", "capture_ts": ms(day, 9 + i),
                     "corrected_state": "known", "corrected_at": "x", "corrected_person_id": "pA",
                     "corrected_ts": ms("2026-10-27", 10)})
        face_rows.append((f"t{i}", 0, near(basis(0), 0.05, RNG), 0.9, 60.0))
    exp = tmp_path / "export"
    write_export(exp, rows, [("pA", "3", "visitor"), ("pB", "3", "visitor")], facilities=(("3", "2"),))
    ex = load_export(exp)
    write_face_cache(tmp_path / "emb", "3", day, face_rows)
    ws = Workspace(ex, tmp_path / "emb", FACE_ID, None)
    run = shadow_run(ws, "3", day, Thresholds(min_refs=1, body_enabled=False), tmp_path / "out", log=lambda *_: None)
    assert run["references"] == {} and run["same_day_seeds"] == 0
    assert run["counts"] == {"sightings": 4}  # nothing named: A had no references that night


def test_adjudication_score_ignores_sightings_that_carry_a_human_label(tmp_path):
    rows = [{"sighting_id": "lab", "facility_id": "3", "capture_ts": ms("2026-10-20", 9), "corrected_state": "known",
             "corrected_at": "x", "corrected_person_id": "pA", "corrected_ts": ms("2026-10-20", 10)},
            {"sighting_id": "free", "facility_id": "3", "capture_ts": ms("2026-10-20", 9)}]
    write_export(tmp_path / "e", rows, [("pA", "3", "visitor")], facilities=(("3", "2"),))
    ex = load_export(tmp_path / "e")
    res = score(ex, {"lab": "pA", "free": "pA"}, {"lab": {"state": "known", "person_key": "pA", "human_labeled": "1"},
                                                    "free": {"state": "known", "person_key": "pA", "human_labeled": "0"}})
    assert res["excluded_human_labeled"] == 1 and res["correct"] == 1


# --- 3. the gate wants evidence ------------------------------------------------------------

def test_gate_refuses_a_perfect_row_with_few_labels():
    row = {"promote_labeled": 2, "promote_lo95_days": 1.0, "promote_cp_lo": cp_lower(2, 2),
           "loo_n": 0, "loo_fa_cp_hi": "", "correct": 2, "wrong": 0, "precision_lo95_days": 1.0, "precision_cp_lo": 0.16}
    reasons = gate_reasons(row, min_precision_lo=0.99, max_loo_fa=0.005, min_labeled=50)
    assert any("labels" in r for r in reasons) and any("binomial" in r for r in reasons)
    assert any("not measured" in r for r in reasons)


def test_cp_lower_is_not_one_for_a_handful_of_successes():
    assert cp_lower(20, 20) < 0.85 and cp_lower(1600, 1600) > 0.997


# --- 4. a face that points elsewhere vetoes a body label ----------------------------------

def test_body_never_names_someone_the_own_face_contradicts():
    A, C = basis(0), basis(2)
    refs = {"A": np.stack([near(A, 0.05, RNG) for _ in range(5)]), "C": np.stack([near(C, 0.05, RNG) for _ in range(5)])}
    roster = {"A": "pA"}  # C is registered only at a sister facility
    seed = sighting("seed", t0=0)
    sc = sighting("sc", t0=1_000_000)
    body = {"seed": unit(basis(10)), "sc": unit(basis(10) + 0.01 * basis(11))}
    th = Thresholds(face_sim=0.5, face_margin=0.1, body_enabled=True, body_link=0.8, body_link_avg=0.7,
                    body_assign=0.8, seed_extra=0.0)
    fs = face_scores({"seed": [(0, near(A, 0.05, RNG), 0.9, 60.0)], "sc": [(0, near(C, 0.05, RNG), 0.9, 60.0)]}, refs, th)
    out = {l.sighting_id: l for l in decide_day([seed, sc], fs, body, {}, roster, lambda p: p, th)}
    assert out["seed"].person_id == "pA"
    assert out["sc"].state == "flag" and out["sc"].relation == "conflict"


# --- 5/6. embed cache consistency ---------------------------------------------------------

def _one_day(tmp_path, n=3):
    day = "2026-10-20"
    rows = [{"sighting_id": f"s{i}", "facility_id": "3", "did": "45", "capture_ts": ms(day, 9 + i),
             "image_url": f"https://cdn.invalid/s{i}.jpg"} for i in range(n)]
    write_export(tmp_path / "export", rows, [("pA", "3", "visitor")], facilities=(("3", "2"),))
    return load_export(tmp_path / "export"), day


def test_a_rerun_without_its_done_file_does_not_duplicate_rows(tmp_path):
    ex, day = _one_day(tmp_path)
    lan = tmp_path / "lan"
    for i in range(3):
        jpg(lan / "45" / day / f"s{i}" / "0.jpg", COLORS["A"])
    face = StubEmbed("f")
    ms_ = ModelSet("stub", StubDetector(), face, StubEmbed("b"))
    store = CropStore(lan, None, fetch=False)
    emb = tmp_path / "emb"
    embed_day(ex.by_facility_day("3", day), store, ms_, emb, "3", day, log=lambda *_: None)
    for f in emb.rglob("*.done"):
        f.unlink()
    embed_day(ex.by_facility_day("3", day), store, ms_, emb, "3", day, log=lambda *_: None)
    assert len(load_face(emb, ms_.face_id, "3", day)) == 3 and face.calls == 3


def test_lan_crops_replace_the_cdn_copy_of_a_sighting(tmp_path):
    ex, day = _one_day(tmp_path, n=1)
    cdn = tmp_path / "cdn"
    jpg(cdn / "45" / day / "s0.jpg", COLORS["A"])
    ms_ = ModelSet("stub", StubDetector(), StubEmbed("f"), StubEmbed("b"))
    emb = tmp_path / "emb"
    embed_day(ex.by_facility_day("3", day), CropStore(None, cdn, fetch=False), ms_, emb, "3", day, log=lambda *_: None)
    lan = tmp_path / "lan"
    for i in range(2):
        jpg(lan / "45" / day / "s0" / f"{i}.jpg", COLORS["A"])
    embed_day(ex.by_facility_day("3", day), CropStore(lan, cdn, fetch=False), ms_, emb, "3", day, log=lambda *_: None)
    ft = load_face(emb, ms_.face_id, "3", day)
    bt = load_body(emb, ms_.body_id, "3", day)
    assert sorted(zip(ft.source, ft.crop_idx)) == [("lan", 0), ("lan", 1)]
    assert sorted(zip(bt.source, bt.crop_idx)) == [("lan", 0), ("lan", 1)]


def test_refresh_also_re_embeds_the_body_model(tmp_path):
    ex, day = _one_day(tmp_path, n=1)
    lan = tmp_path / "lan"
    jpg(lan / "45" / day / "s0" / "0.jpg", COLORS["A"])
    body = StubEmbed("b")
    ms_ = ModelSet("stub", StubDetector(), StubEmbed("f"), body)
    emb = tmp_path / "emb"
    store = CropStore(lan, None, fetch=False)
    embed_day(ex.by_facility_day("3", day), store, ms_, emb, "3", day, log=lambda *_: None)
    embed_day(ex.by_facility_day("3", day), store, ms_, emb, "3", day, refresh=True, log=lambda *_: None)
    assert body.calls == 2 and len(load_body(emb, ms_.body_id, "3", day)) == 1


# --- 9. every grid row is identifiable ------------------------------------------------------

def test_curve_rows_carry_a_combo_id_and_full_thresholds():
    from vbb_reid_server.evaluate import ComboResult
    rows = curve_rows([ComboResult(Thresholds(topk=1)), ComboResult(Thresholds(topk=5))], "s", "temporal", "3")
    assert [r["combo"] for r in rows] == [0, 1]
    assert json.loads(rows[0]["thresholds"])["topk"] == 1 and json.loads(rows[1]["thresholds"])["topk"] == 5


# --- 14. min_refs counts labeled sightings, not crops --------------------------------------

def test_one_labeled_sighting_with_three_crops_is_one_reference():
    A, B = basis(0), basis(1)
    refs = Refs({"A": np.stack([near(A, 0.02, RNG) for _ in range(3)]), "B": np.stack([near(B, 0.02, RNG) for _ in range(3)])},
                {"A": np.array([0]), "B": np.array([0, 1, 2])})
    fs = face_scores({"p": [(0, near(A, 0.02, RNG), 0.9, 60.0)]}, refs, Thresholds(min_refs=3))
    assert list(fs.n_refs) == [1, 3]
    out = decide_day([sighting("p")], fs, {}, {}, {"A": "pA", "B": "pB"}, lambda p: p,
                     Thresholds(min_refs=3, body_enabled=False))
    assert out == []


# --- 16. people ------------------------------------------------------------------------------

def test_deleted_people_stay_deleted_and_merge_chains_are_followed(tmp_path):
    root = tmp_path / "e"
    write_export(root, [], [])
    (root / "face_gallery.csv").write_text(
        "id,facility_id,person_id,person_type,photo_url,tentative,merged_into,state,created_at,updated_at\n"
        "1,3,gone,visitor,,0,,0,,\n2,3,X,visitor,,0,Y,1,,\n3,3,Y,visitor,,0,Z,1,,\n4,3,Z,visitor,,0,,1,,\n")
    (root / "person_identity_role.csv").write_text(
        "id,facility_id,person_id,role,state,created_at,updated_at\n1,3,gone,nurse,1,,\n")
    (root / "person_roster.csv").write_text(
        "id,did,facility_id,person_id,merged_into,state,created_at,updated_at\n1,45,3,gone,,1,,\n")
    ex = load_export(root)
    assert "gone" not in ex.people and ex.key("X") == "Z" and ex.roster("3") == {"Z": "Z"}


def test_tally_unchanged():
    t = Tally()
    t.add("correct")
    assert t.precision == 1.0


def test_rows_without_a_processed_entry_are_not_duplicated(tmp_path):
    from vbb_reid_server.embed import FACE_SCHEMA, _dir
    ex, day = _one_day(tmp_path, n=1)
    lan = tmp_path / "lan"
    jpg(lan / "45" / day / "s0" / "0.jpg", COLORS["A"])
    ms_ = ModelSet("stub", StubDetector(), StubEmbed("f"), None)
    emb = tmp_path / "emb"
    d = _dir(emb, "face", ms_.face_id, "3")
    one = np.ones(1)
    np.savez(d / f"{day}.npz", schema=np.int64(FACE_SCHEMA), processed=np.array([], dtype=object),
             sighting_id=np.array(["s0"], dtype=object), crop_idx=np.array([0]), source=np.array(["lan"], dtype=object),
             vec=np.ones((1, 4), np.float32) / 2, face_h=one * 60, det=one, blur=one * 500, residual=one * 0,
             asym=one * 0, yaw=one * 0, norm=one * 20, n_faces=one, second_ratio=one * 0, second_det=one * 0)  # a damaged cache: a row whose processed entry was lost
    embed_day(ex.by_facility_day("3", day), CropStore(lan, None, fetch=False), ms_, emb, "3", day, log=lambda *_: None)
    assert len(load_face(emb, ms_.face_id, "3", day)) == 1


# --- 2026-11-04 owner review: two-person crops and non-faces --------------------------------

def test_a_crop_with_a_second_sizeable_face_is_flagged_mixed():
    A, B = basis(0), basis(1)
    refs = {"A": np.stack([near(A, 0.05, RNG) for _ in range(5)]), "B": np.stack([near(B, 0.05, RNG) for _ in range(5)])}
    probe = {"p": [(0, near(B, 0.05, RNG), 0.9, 180.0, 20.0, 51 / 180, 0.84)]}  # a2edee5618ff's shape
    th_off = Thresholds(face_sim=0.5, face_margin=0.1, body_enabled=False)
    th_on = Thresholds(face_sim=0.5, face_margin=0.1, body_enabled=False, two_face_ratio=0.25)
    s = sighting("p")
    off = decide_day([s], face_scores(probe, refs, th_off), {}, {}, {"A": "pA", "B": "pB"}, lambda p: p, th_off)
    on = decide_day([s], face_scores(probe, refs, th_on), {}, {}, {"A": "pA", "B": "pB"}, lambda p: p, th_on)
    assert off[0].state == "known"
    assert on[0].state == "flag" and on[0].relation == "mixed" and "two_faces" in on[0].evidence["reasons"][0]


def test_a_low_norm_face_is_not_used():
    A, B = basis(0), basis(1)
    refs = {"A": np.stack([near(A, 0.05, RNG) for _ in range(5)]), "B": np.stack([near(B, 0.05, RNG) for _ in range(5)])}
    probe = {"p": [(0, near(A, 0.05, RNG), 0.9, 60.0, 3.0, 0.0, 0.0)]}  # e.g. hair
    th = Thresholds(face_sim=0.5, face_margin=0.1, body_enabled=False, min_face_norm=8.0)
    assert decide_day([sighting("p")], face_scores(probe, refs, th), {}, {}, {"A": "pA"}, lambda p: p, th) == []


def test_reference_filters_skip_low_norm_and_two_face_crops():
    from vbb_reid_server.embed import FaceTable
    from vbb_reid_server.pipeline import RefPolicy, ref_usable
    one = np.ones(3)
    ft = FaceTable(np.array(["a", "b", "c"], dtype=object), np.zeros(3, int), np.array(["cdn"] * 3, dtype=object),
                   np.zeros((3, 4), np.float32), one * 60, one, one * 500, one, one, one * 0,
                   np.array([20.0, 3.0, 20.0]), np.array([1, 1, 2]), np.array([0.0, 0.0, 0.5]), np.array([0, 0, 0.9]),
                   np.array([0.9, 0.9, 0.9]))
    pol = RefPolicy(ref_min_norm=8.0, ref_two_face_ratio=0.25)
    assert [ref_usable(ft, i, pol) for i in range(3)] == [True, False, False]
    assert all(ref_usable(ft, i, RefPolicy()) for i in range(3))


def test_verdict_separates_not_enough_evidence_from_failure():
    from vbb_reid_server.evaluate import needed_trials, verdict
    row = dict(promote_labeled=218, promote_precision=1.0, promote_lo95_days=1.0, promote_cp_lo=0.98322, loo_n=493,
               loo_false_accepts=0, loo_fa_cp_hi=0.00745, correct=218, wrong=0, precision_lo95_days=1.0,
               precision_cp_lo=0.98322)
    v, _, needs = verdict(row, min_precision_lo=0.99, max_loo_fa=0.005, min_labeled=50)
    assert v == "insufficient" and needs == {"labeled_promotions": 368, "loo_trials": 736}
    bad = dict(row, promote_labeled=300, promote_precision=0.95, promote_cp_lo=0.91)
    assert verdict(bad, min_precision_lo=0.99, max_loo_fa=0.005, min_labeled=50)[0] == "fail"
    assert needed_trials(285, 300, rate_bound=0.99, upper=False) is None  # 95 % never reaches 0.99
