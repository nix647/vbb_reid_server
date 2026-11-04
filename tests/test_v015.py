"""2026-11-05 owner findings after the low-norm adjudication set."""

import csv
import json

import pytest
from helpers import FACE_ID, sighting, write_export
from test_data_eval import build
from test_embed import COLORS, jpg
from vbb_reid_server.adjudicate import sample, write_page
from vbb_reid_server.cli import main
from vbb_reid_server.data import load_export


def _pool(tmp_path, n_known=40, n_stranger=3):
    rows = [{"sighting_id": f"k{i}", "facility_id": "3", "capture_ts": 1792900000000 + i * 1000,
             "identity_state": "known", "person_id": "pA", "image_url": f"https://cdn.invalid/k{i}.jpg"}
            for i in range(n_known)]
    rows += [{"sighting_id": f"s{i}", "facility_id": "3", "capture_ts": 1792900500000 + i * 1000,
              "identity_state": "stranger", "image_url": f"https://cdn.invalid/s{i}.jpg"} for i in range(n_stranger)]
    write_export(tmp_path / "e", rows, [("pA", "3", "visitor")], facilities=(("3", "2"),))
    return load_export(tmp_path / "e")


def test_unused_stratum_share_goes_to_the_other_strata(tmp_path):
    ex = _pool(tmp_path)
    got = sample(ex, "3", 20)  # two strata; the stranger one holds only 3
    assert len(got) == 20 and sum(s.board_state == "stranger" for s in got) == 3


def test_unstratified_sample_is_uniform_over_the_pool(tmp_path):
    ex = _pool(tmp_path)
    got = sample(ex, "3", 43, strata="none")
    assert len(got) == 43  # the whole pool, no per-stratum cap


def test_new_names_only_sample(tmp_path):
    exp, emb = build(tmp_path)
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sighting_id", "state", "relation", "human_labeled"])
        w.writerow(["unl0", "known", "promote", "0"])
    out = tmp_path / "s.csv"
    main(["adjudicate-sample", "--export", str(exp), "--facility", "3", "--n", "5", "--named-by", str(labels),
          "--relation", "promote", "--strata", "none", "--out", str(out)])
    assert [r["sighting_id"] for r in csv.DictReader(out.open())] == ["unl0"]
    main(["adjudicate-sample", "--export", str(exp), "--facility", "3", "--n", "5", "--named-by", str(labels),
          "--relation", "agree", "--out", str(out)])
    assert list(csv.DictReader(out.open())) == []


def test_page_can_carry_its_lan_crops_next_to_it(tmp_path):
    ex = _pool(tmp_path, n_known=1, n_stranger=0)
    lan = tmp_path / "lan"
    s = ex.sightings[0]
    for i in range(2):
        jpg(lan / s.did / s.day / s.sighting_id / f"{i}.jpg", COLORS["A"])
    page = tmp_path / "out" / "page.html"
    write_page(ex, "3", [s.sighting_id], page, set_id="t", lan_root=lan, copy_crops=True)
    text = page.read_text(encoding="utf-8")
    assert "file://" not in text and f"crops/{s.sighting_id}_0.jpg" in text
    assert (page.parent / "crops" / f"{s.sighting_id}_1.jpg").is_file()


def test_eval_fix_pins_settings_when_choosing_the_operating_point(tmp_path):
    exp, emb = build(tmp_path)
    grid = tmp_path / "g.json"
    grid.write_text(json.dumps({"face_sim": [0.5], "face_margin": [0.1], "two_face_ratio": [None, 0.25],
                                "body_enabled": [False], "min_refs": [3]}))
    out = tmp_path / "o"
    main(["eval", "--export", str(exp), "--aliases", str(tmp_path / "aliases.csv"), "--emb", str(emb), "--facility", "3",
          "--face-model", FACE_ID, "--refs-mode", "temporal", "--grid", str(grid), "--min-precision-lo", "0",
          "--max-loo-fa", "1", "--min-labeled", "1", "--fix", "two_face_ratio=0.25", "--out", str(out)])
    pt = json.loads((out / "point_3_temporal.json").read_text())
    assert pt["thresholds"]["two_face_ratio"] == 0.25
    with pytest.raises(SystemExit):
        main(["eval", "--export", str(exp), "--emb", str(emb), "--facility", "3", "--face-model", FACE_ID,
              "--grid", str(grid), "--fix", "two_face_ration=0.25", "--out", str(out)])


def test_helper_sighting_unused():
    assert sighting("x").sighting_id == "x"
