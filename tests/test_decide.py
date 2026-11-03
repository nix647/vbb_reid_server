"""Decision rules. Each test names the guard it pins; removing the guard turns it red."""

import numpy as np
import pytest

from helpers import basis, near, sighting, unit
from vbb_reid_server.decide import Thresholds, decide_day, face_scores

RNG = np.random.default_rng(0)
A, B, C = basis(0), basis(1), basis(2)
ROSTER = {"A": "pA", "B": "pB"}


def key_of(pid):
    return {"pA": "A", "pB": "B", "pC": "C"}.get(pid, pid)


def refs(n=10, noise=0.2, persons=None):
    persons = persons or {"A": A, "B": B}
    return {k: np.stack([near(v, noise, RNG) for _ in range(n)]) for k, v in persons.items()}


def face(sid, vec, q=0.9, h=60.0, crop=0):
    return (crop, vec, q, h)


TH = Thresholds(face_sim=0.6, face_margin=0.15, body_enabled=False)


def run(sightings, probe_faces, th=TH, body=None, body_crops=None, roster=ROSTER, r=None, seeds=None):
    fs = face_scores(probe_faces, r if r is not None else refs(), th)
    return {l.sighting_id: l for l in decide_day(sightings, fs, body or {}, body_crops or {}, roster, key_of, th,
                                                 human_seeds=seeds)}


def test_face_promotes_a_board_stranger():
    s = sighting("s1", board_state="stranger")
    out = run([s], {"s1": [face("s1", near(A, 0.1, RNG))]})
    l = out["s1"]
    assert (l.state, l.relation, l.person_id, l.via) == ("known", "promote", "pA", "face")


def test_margin_guard_abstains_between_two_lookalikes():
    # A probe halfway between A and B clears face_sim for both but has no margin.
    s = sighting("s1")
    probe = unit(A + B)
    r = refs(noise=0.0)
    th = Thresholds(face_sim=0.5, face_margin=0.15, body_enabled=False)
    out = run([s], {"s1": [face("s1", probe)]}, th=th, r=r)
    assert "s1" not in out  # guard: margin


def test_similarity_floor_abstains():
    s = sighting("s1")
    out = run([s], {"s1": [face("s1", near(C, 0.05, RNG))]})
    assert "s1" not in out


def test_quality_floor_ignores_bad_faces():
    s = sighting("s1")
    out = run([s], {"s1": [face("s1", near(A, 0.05, RNG), q=0.1)]})
    assert "s1" not in out


def test_min_refs_guard():
    s = sighting("s1")
    r = {"A": np.stack([A]), "B": refs()["B"]}
    out = run([s], {"s1": [face("s1", near(A, 0.05, RNG))]}, r=r)
    assert "s1" not in out


def test_closed_roster_never_names_an_outsider():
    s = sighting("s1")
    r = refs(persons={"A": A, "B": B, "C": C})
    out = run([s], {"s1": [face("s1", near(C, 0.05, RNG))]}, r=r)
    assert "s1" not in out


def test_board_agreement_is_agree_not_promote():
    s = sighting("s1", board_state="known", board_pid="pA")
    out = run([s], {"s1": [face("s1", near(A, 0.05, RNG))]})
    assert (out["s1"].state, out["s1"].relation) == ("known", "agree")


def test_relabel_is_only_a_flag_in_v1():
    s = sighting("s1", board_state="known", board_pid="pB")
    out = run([s], {"s1": [face("s1", near(A, 0.05, RNG))]})
    assert (out["s1"].state, out["s1"].relation) == ("flag", "relabel")


def test_mixed_track_is_flagged_never_labeled():
    s = sighting("s1")
    out = run([s], {"s1": [face("s1", near(A, 0.05, RNG), crop=0), face("s1", near(B, 0.05, RNG), crop=1)]})
    assert (out["s1"].state, out["s1"].relation) == ("flag", "mixed")


def test_two_consistent_crops_are_not_mixed():
    s = sighting("s1")
    out = run([s], {"s1": [face("s1", near(A, 0.05, RNG), crop=0), face("s1", near(A, 0.05, RNG), crop=1)]})
    assert out["s1"].state == "known"


def test_machine_never_emits_stranger_or_uncertain():
    ss = [sighting(f"s{i}") for i in range(30)]
    probes = {s.sighting_id: [face(s.sighting_id, near(basis(i % 4), 0.3, RNG))] for i, s in enumerate(ss)}
    out = run(ss, probes)
    assert {l.state for l in out.values()} <= {"known", "flag"}


# ------------------------------------------------------------------ body path --

BTH = Thresholds(face_sim=0.6, face_margin=0.15, body_enabled=True, body_link=0.8, body_link_avg=0.7,
                 body_assign=0.85, seed_extra=0.0)


def test_body_propagates_from_a_face_seed_on_the_same_day():
    seed = sighting("seed", t0=0, t1=1000)
    faceless = sighting("nf", t0=600_000, t1=601_000, crop_is_face=False)
    body_a = basis(10)
    body = {"seed": near(body_a, 0.05, RNG), "nf": near(body_a, 0.05, RNG)}
    out = run([seed, faceless], {"seed": [face("seed", near(A, 0.05, RNG))]}, th=BTH, body=body)
    assert out["seed"].via == "face"
    assert (out["nf"].state, out["nf"].person_id, out["nf"].via) == ("known", "pA", "body")


def test_simultaneous_tracks_on_one_board_never_merge():
    # Same clothes, same time, same board: two people. The face-less one must not inherit A.
    seed = sighting("seed", t0=0, t1=10_000)
    twin = sighting("twin", t0=5_000, t1=12_000, crop_is_face=False)
    body_a = basis(10)
    body = {"seed": near(body_a, 0.02, RNG), "twin": near(body_a, 0.02, RNG)}
    out = run([seed, twin], {"seed": [face("seed", near(A, 0.05, RNG))]}, th=BTH, body=body)
    assert "twin" not in out  # guard: cannot-link


def test_cluster_with_two_named_seeds_never_merges():
    a = sighting("a", t0=0)
    b = sighting("b", t0=1_000_000)
    nf = sighting("nf", t0=2_000_000, crop_is_face=False)
    body_x = basis(10)
    body = {k: near(body_x, 0.02, RNG) for k in ("a", "b", "nf")}
    out = run([a, b, nf], {"a": [face("a", near(A, 0.05, RNG))], "b": [face("b", near(B, 0.05, RNG))]},
              th=BTH, body=body)
    # Merged into one cluster, one of the two would come back as a conflict flag.
    assert (out["a"].state, out["a"].person_id) == ("known", "pA")
    assert (out["b"].state, out["b"].person_id) == ("known", "pB")
    if "nf" in out:
        assert out["nf"].person_id in ("pA", "pB")


def test_body_assign_requires_direct_similarity_to_a_seed():
    # Chain seed -> m1 -> m2 where m2 is far from the seed: m2 must not inherit.
    seed = sighting("seed", t0=0)
    m1 = sighting("m1", t0=1_000_000, crop_is_face=False)
    m2 = sighting("m2", t0=2_000_000, crop_is_face=False)
    b0 = basis(10)
    b1 = unit(basis(10) + 0.8 * basis(11))
    b2 = unit(basis(10) + 1.6 * basis(11))
    body = {"seed": b0, "m1": b1, "m2": b2}
    th = Thresholds(face_sim=0.6, face_margin=0.15, body_enabled=True, body_link=0.75, body_link_avg=0.5,
                    body_assign=0.85, seed_extra=0.0)
    out = run([seed, m1, m2], {"seed": [face("seed", near(A, 0.05, RNG))]}, th=th, body=body)
    assert "m2" not in out


def test_same_day_human_label_seeds_the_body_cluster():
    lab = sighting("lab", t0=0, crop_is_face=False, human_state="known", human_pid="pB")
    nf = sighting("nf", t0=1_000_000, crop_is_face=False)
    bx = basis(12)
    body = {"lab": near(bx, 0.02, RNG), "nf": near(bx, 0.02, RNG)}
    out = run([lab, nf], {}, th=BTH, body=body, seeds={"lab": "B"})
    assert out["nf"].person_id == "pB" and out["nf"].via == "body"


def test_body_disabled_means_no_body_labels():
    seed = sighting("seed", t0=0)
    nf = sighting("nf", t0=600_000, crop_is_face=False)
    b = basis(10)
    out = run([seed, nf], {"seed": [face("seed", near(A, 0.05, RNG))]}, th=TH,
              body={"seed": b, "nf": b})
    assert "nf" not in out


@pytest.mark.parametrize("k", [1, 3, 50])
def test_topk_larger_than_refs_is_safe(k):
    s = sighting("s1")
    th = Thresholds(face_sim=0.6, face_margin=0.15, body_enabled=False, topk=k, min_refs=1)
    out = run([s], {"s1": [face("s1", near(A, 0.05, RNG))]}, th=th, r={"A": np.stack([A, A]), "B": np.stack([B])})
    assert out["s1"].person_id == "pA"
