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


def test_shipped_configs_load_and_unknown_grid_keys_are_refused(tmp_path):
    from pathlib import Path
    from vbb_reid_server.evaluate import load_grid
    from vbb_reid_server.quality import QualityPolicy
    root = Path(__file__).resolve().parents[1] / "configs"
    for p in (root / "quality").glob("*.json"):
        QualityPolicy.from_file(p)
    for p in (root / "grids").glob("*.json"):
        load_grid(str(p))
    bad = tmp_path / "g.json"
    bad.write_text(json.dumps({"face_simm": [0.3]}))
    with pytest.raises(ValueError):
        load_grid(str(bad))


def test_inspect_prints_measures_scores_and_decision(tmp_path, capsys):
    exp, emb = build(tmp_path)
    main(["inspect", "--export", str(exp), "--aliases", str(tmp_path / "aliases.csv"), "--emb", str(emb),
          "--facility", "3", "--face-model", FACE_ID, "--sightings", "tA0,unl0,nope"])
    out = capsys.readouterr().out
    assert "tA0" in out and "norm=" in out and "reference-usable=" in out and "face scores:" in out
    assert "decision:" in out and "nope: not a facility 3 sighting" in out


def test_targeted_adjudication_sample_by_machine_name_and_norm(tmp_path):
    import csv as csv_
    exp, emb = build(tmp_path)
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as fh:
        w = csv_.writer(fh)
        w.writerow(["sighting_id", "state", "human_labeled"])
        w.writerow(["unl0", "known", "0"])
    out = tmp_path / "s.csv"
    main(["adjudicate-sample", "--export", str(exp), "--facility", "3", "--n", "10", "--named-by", str(labels),
          "--norm-below", "25", "--emb", str(emb), "--face-model", FACE_ID, "--out", str(out)])
    assert [r["sighting_id"] for r in csv_.DictReader(out.open())] == ["unl0"]
    main(["adjudicate-sample", "--export", str(exp), "--facility", "3", "--n", "10", "--named-by", str(labels),
          "--norm-below", "15", "--emb", str(emb), "--face-model", FACE_ID, "--out", str(out)])
    assert list(csv_.DictReader(out.open())) == []  # the helper's faces all have norm 20

