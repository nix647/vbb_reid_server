"""Quality policy on stored measures. The first test pins the 0.1.0 regression found on
gpu-ubun 2026-11-03: a large alignment residual alone (a non-frontal face) zeroed qf."""

import json

import numpy as np
from helpers import FACE_ID, basis, near, write_export, write_face_cache
from vbb_reid_server.data import load_export
from vbb_reid_server.embed import StaleCache, _dir, load_face
from vbb_reid_server.pipeline import Workspace
from vbb_reid_server.quality import QualityPolicy, qf_from_measures
from vbb_reid_server.quality_report import report


def q(residual=0.0, asym=0.0, yaw=0.0, det=0.9, h=80.0, blur=500.0, policy=QualityPolicy()):
    return float(qf_from_measures([det], [h], [blur], [residual], [asym], [yaw], policy)[0])


def test_large_residual_alone_does_not_zero_quality():
    assert q(residual=8.0) > 0.85  # guard: the residual itself is not a penalty by default


def test_the_0_1_0_behaviour_is_still_available_and_zeroes():
    assert q(residual=8.0, policy=QualityPolicy(use_residual=True)) < 0.01


def test_asymmetry_counts_only_when_the_mean_residual_is_small():
    assert q(residual=2.0, asym=6.0) < 0.01  # well aligned but lopsided: penalised
    assert q(residual=5.0, asym=6.0) > 0.85  # above the cap: pose, not asymmetry


def test_pose_size_and_blur_still_bite():
    assert q(yaw=0.6) < 0.01
    assert q(h=10.0) < q(h=40.0)
    assert q(blur=10.0) < 0.01


def test_policy_from_file(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"use_pose": False, "size_floor_px": 20, "not_a_field": 1}))
    pol = QualityPolicy.from_file(p)
    assert pol.use_pose is False and pol.size_floor_px == 20


def test_a_0_1_0_cache_is_refused_on_read(tmp_path):
    d = _dir(tmp_path, "face", FACE_ID, "3")
    np.savez(d / "2026-10-20.npz", sighting_id=np.array(["a"], dtype=object), crop_idx=np.array([0]),
             source=np.array(["cdn"], dtype=object), vec=np.zeros((1, 4), np.float32), qf=np.array([0.1]),
             face_h=np.array([60.0]), det=np.array([0.9]), residual=np.array([7.0]))
    try:
        load_face(tmp_path, FACE_ID, "3", "2026-10-20")
    except StaleCache as e:
        assert "--refresh" in str(e)
    else:
        raise AssertionError("a schema-1 cache must not be read with a stored qf")


def test_quality_report_separates_good_from_bad_faces(tmp_path):
    rng = np.random.default_rng(3)
    rows, face_rows = [], []
    for i in range(40):
        who = "pA" if i % 2 else "pB"
        good = i < 20
        sid = f"s{i}"
        rows.append({"sighting_id": sid, "facility_id": "3", "capture_ts": 1792900000000 + i,
                     "corrected_state": "known", "corrected_at": "x", "corrected_person_id": who,
                     "corrected_ts": 1792900100000})
        vec = near(basis(0 if who == "pA" else 1), 0.05 if good else 1.5, rng)
        face_rows.append((sid, 0, vec, 0.95 if good else 0.2, 80.0))
    exp = tmp_path / "export"
    write_export(exp, rows, [("pA", "3", "visitor"), ("pB", "3", "visitor")], facilities=(("3", "2"),))
    ex = load_export(exp)
    day = ex.days("3")[0]
    write_face_cache(tmp_path / "emb", "3", day, face_rows)
    ws = Workspace(ex, tmp_path / "emb", FACE_ID, None)
    out = report(ws, "3", [day], bins=2, far=0.01, log=lambda *_: None)
    det_rows = [r for r in out if r["measure"] == "det"]
    assert det_rows[-1]["tar_at_far"] > det_rows[0]["tar_at_far"]  # good faces match, bad ones do not
