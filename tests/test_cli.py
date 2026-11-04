"""Command-line wiring: model selection, strict config files, eval -> point.json -> shadow."""

import json

import pytest
from helpers import FACE_ID
from test_data_eval import build
from vbb_reid_server.cli import main
from vbb_reid_server.embed import _dir


def test_an_ambiguous_model_name_is_an_error(tmp_path):
    exp, emb = build(tmp_path)
    _dir(emb, "face", "det.onnx|w600k_r50.onnx:sha256:aa:RGB:flip", "3")
    _dir(emb, "face", "det.onnx|w600k_r50.onnx:sha256:aa:RGB", "3")
    with pytest.raises(SystemExit) as e:
        main(["eval", "--export", str(exp), "--emb", str(emb), "--facility", "3", "--face-model", "w600k",
              "--out", str(tmp_path / "o")])
    assert "matches 2" in str(e.value)


def test_unknown_threshold_keys_are_an_error(tmp_path):
    exp, emb = build(tmp_path)
    th = tmp_path / "th.json"
    th.write_text(json.dumps({"face_simm": 0.4}))
    with pytest.raises(SystemExit) as e:
        main(["eval", "--export", str(exp), "--emb", str(emb), "--facility", "3", "--face-model", FACE_ID,
              "--thresholds", str(th), "--out", str(tmp_path / "o")])
    assert "face_simm" in str(e.value)


def test_eval_writes_a_point_that_shadow_reads_back(tmp_path, capsys):
    exp, emb = build(tmp_path)
    aliases = tmp_path / "aliases.csv"
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"face_sim": [0.5], "face_margin": [0.1], "face_q_min": [0.2], "body_enabled": [False],
                                "min_refs": [3]}))
    out = tmp_path / "o"
    main(["eval", "--export", str(exp), "--aliases", str(aliases), "--emb", str(emb), "--facility", "3",
          "--face-model", FACE_ID, "--refs-mode", "temporal", "--grid", str(grid), "--ref-q-min", "0.3",
          "--min-precision-lo", "0.0", "--max-loo-fa", "1.0", "--min-labeled", "1", "--out", str(out)])
    point = out / "point_3_temporal.json"
    pt = json.loads(point.read_text())
    assert pt["thresholds"]["face_q_min"] == 0.2 and pt["ref_policy"]["ref_q_min"] == 0.3
    main(["shadow", "--export", str(exp), "--aliases", str(aliases), "--emb", str(emb), "--facility", "3",
          "--face-model", FACE_ID, "--point", str(point), "--days", "2026-10-20", "--out", str(tmp_path / "sh")])
    run = json.loads(next((tmp_path / "sh").glob("run_*.json")).read_text())
    assert run["thresholds"] == pt["thresholds"] and run["ref_policy"]["ref_q_min"] == 0.3
    assert run["label_cutoff_ms"] is not None  # production-like by default
