"""embed_day end to end with stub models: crop discovery (LAN over CDN), the face and body
caches, idempotent re-runs, and a body-enabled evaluation reading the result."""

from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from helpers import write_export
from vbb_reid_server.align import ARCFACE_DST
from vbb_reid_server.crops import CropStore
from vbb_reid_server.data import load_export
from vbb_reid_server.decide import Thresholds
from vbb_reid_server.embed import embed_day, load_body, load_face
from vbb_reid_server.evaluate import evaluate_facility
from vbb_reid_server.models.base import FaceDetection, l2n
from vbb_reid_server.models.registry import ModelSet
from vbb_reid_server.pipeline import Workspace

COLORS = {"A": (200, 30, 30), "B": (30, 200, 30)}


class StubDetector:
    model_id = "stub-det"

    def detect(self, img):
        if img[:16, :16].mean() < 10:  # a black corner block marks "no face visible"
            return []
        kps = ARCFACE_DST.copy() + np.array([20.0, 20.0], np.float32)
        return [FaceDetection(np.array([20, 20, 132, 160], np.float32), 0.95, kps)]


class StubEmbed:
    def __init__(self, mid):
        self.model_id = mid
        self.calls = 0

    def embed(self, imgs):
        self.calls += len(imgs)
        feats = [np.concatenate([img.reshape(-1, 3).mean(0), [1.0]]) for img in imgs]
        return l2n(np.stack(feats) - np.array([100, 100, 100, 0.0]))


def jpg(path, color, face=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(str(path))) % 2**32)
    img = np.full((256, 160, 3), color, np.uint8)
    img = np.clip(img.astype(int) + rng.integers(-20, 20, img.shape), 1, 255).astype(np.uint8)
    if not face:
        img[:16, :16] = 0
    cv2.imwrite(str(path), img)


def test_embed_then_evaluate_with_body(tmp_path):
    day = "2026-10-20"
    base = int(datetime(2026, 10, 20, 8, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp() * 1000)
    rows = []
    lan = tmp_path / "lan"
    def add(sid, who, h, *, labeled=False, face=True, ts_day=0):
        t = base + ts_day * 86_400_000 + int(h * 3_600_000)
        rows.append({"sighting_id": sid, "facility_id": "3", "did": "45", "capture_ts": t,
                     "track_started_ts": t - 1000, "track_ended_ts": t, "identity_state": "stranger",
                     "corrected_state": "known" if labeled else "", "corrected_at": "x" if labeled else "",
                     "corrected_person_id": f"p{who}" if labeled else "", "corrected_ts": t + 3_600_000 if labeled else "",
                     "image_url": f"https://cdn.invalid/{sid}.jpg", "crop_is_face": int(face)})
        d = "2026-10-19" if ts_day == -1 else day
        for i in range(2):
            jpg(lan / "45" / d / sid / f"{i}.jpg", COLORS[who], face=face)
    for i in range(4):
        add(f"rA{i}", "A", 1 + i * 0.1, labeled=True, ts_day=-1)
        add(f"rB{i}", "B", 2 + i * 0.1, labeled=True, ts_day=-1)
    add("tA", "A", 1, labeled=True)
    add("nfA", "A", 3, labeled=True, face=False)
    exp = tmp_path / "export"
    write_export(exp, rows, [("pA", "3", "visitor"), ("pB", "3", "visitor")], facilities=(("3", "2"),))
    ex = load_export(exp)
    face, body = StubEmbed("stub-face"), StubEmbed("stub-body")
    ms = ModelSet("stub", StubDetector(), face, body)
    store = CropStore(lan, None, fetch=False)
    emb = tmp_path / "emb"
    for d in ex.days("3"):
        embed_day(ex.by_facility_day("3", d), store, ms, emb, "3", d, log=lambda *_: None)
    ft = load_face(emb, ms.face_id, "3", day)
    bt = load_body(emb, ms.body_id, "3", day)
    assert sorted(set(ft.sighting_id)) == ["tA"]  # the face-less sighting has no face rows
    assert len(bt) == 4 and set(bt.crop_idx) == {0, 1}  # two LAN crops each, both embedded
    assert store.stats["lan"] > 0 and store.stats["missing"] == 0
    calls = face.calls
    embed_day(ex.by_facility_day("3", day), store, ms, emb, "3", day, log=lambda *_: None)
    assert face.calls == calls and len(load_body(emb, ms.body_id, "3", day)) == 4  # re-run is a no-op

    ws = Workspace(ex, emb, ms.face_id, ms.body_id)
    th = Thresholds(face_sim=0.9, face_margin=0.05, face_min_px=10, min_refs=2, seed_extra=0.0,
                    body_link=0.9, body_link_avg=0.9, body_assign=0.9)
    res = evaluate_facility(ws, "3", grid={"body_enabled": [True]}, base=th, days=[day], loo=False,
                            log=lambda *_: None)
    r = res[0]
    assert r.total.correct == 2 and r.total.wrong == 0
    assert r.by_via["body"].correct == 1  # the face-less sighting was named through its body
