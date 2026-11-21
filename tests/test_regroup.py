"""Nightly regroup (docs/DESIGN_nightly_regroup.md): sorting the pictures inside each named person.

Synthetic vectors only: a picture is ``unit(a_c * common + a_i * identity + a_n * noise)``. ``common`` and the
identities are fixed orthonormal vectors, so how alike two people look is set by ``a_c`` alone. Every test uses a fixed
seed; where a count is statistical (a fresh picture can clear a floor by chance, about one time in the number of
impostor scores) the test says so and allows for it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from vbb_reid_server.regroup import Item, Name, Params, _floors, _verdict, regroup

DIM, BODY_DIM = 128, 96
_Q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(DIM, 12)))
COMMON = _Q[:, 0]
ID = {name: _Q[:, i + 1] for i, name in enumerate("ABCDEFGHIJK")}
_QB, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(BODY_DIM, 6)))
CLOTHES = {name: _QB[:, i] for i, name in enumerate("ABCDE")}
UNIFORM = _QB[:, 5]

NORMAL = dict(a_c=0.5, a_i=0.8, a_n=0.6)
HARD = dict(a_c=0.8, a_i=0.6, a_n=0.4)  # different people share about 0.55 of raw similarity


def _unit(v):
    return v / np.linalg.norm(v)


def pics(rng, who, n, *, a_c=0.5, a_i=0.8, a_n=0.6):
    """n pictures of one identity; ``who`` is a name in ID or a ready identity vector."""
    ident = ID[who] if isinstance(who, str) else who
    return [_unit(a_c * COMMON + a_i * ident + a_n * _unit(rng.normal(size=DIM))) for _ in range(n)]


def clothes(rng, who, n, a_n=0.5):
    base = CLOTHES[who] if isinstance(who, str) else who
    return [_unit(base + a_n * _unit(rng.normal(size=BODY_DIM))) for _ in range(n)]


class Maker:
    """Builds items with sequential ids; keeps the ids of each group so tests can ask about them."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.ids = {}

    def add(self, tag, who, n, *, weak=None, anchor=None, probe=True, body=None, weights=NORMAL, quality=None):
        """``who`` None = no face. ``body`` is a clothes name (or None). ``quality``: one number for all the new
        pictures, or a list. Returns the new sids."""
        faces = pics(self.rng, who, n, **weights) if who is not None else [None] * n
        bodies = clothes(self.rng, body, n) if body is not None else [None] * n
        sids = []
        for k, (f, b) in enumerate(zip(faces, bodies)):
            sid = f"{tag}{k:03d}"
            q = quality[k] if isinstance(quality, list) else quality
            extra = () if q is None else (q,)
            self.items.append(Item(sid, probe, weak, anchor, f, b, *extra))
            sids.append(sid)
        self.ids.setdefault(tag, []).extend(sids)
        return sids

    def run(self, params=Params(), order=None):
        items = list(self.items)
        if order is not None:
            np.random.default_rng(order).shuffle(items)
        return regroup(items, params)


def named(res, sids):
    return [res.names[s].person for s in sids]


def reasons(res, sids):
    return [res.names[s].reason for s in sids]


def three_people(seed=1, weights=NORMAL):
    """A, B, C: 60 board-named pictures each, plus a few unnamed pictures of each (weak name None)."""
    m = Maker(seed)
    for p in "ABC":
        m.add(p, p, 60, weak=p, weights=weights)
        m.add(f"u{p}", p, 6, weights=weights)
    return m


# 1. clean names ---------------------------------------------------------------------------------------------------


def test_clean_names_keep_their_name_and_unnamed_pictures_find_theirs():
    m = three_people()
    m.add("dD", "D", 8, weights=dict(a_c=0.1, a_i=0.9, a_n=0.4))  # a 4th person who looks unlike the others
    res = m.run()
    for p in "ABC":
        assert named(res, m.ids[p]) == [p] * 60  # a member never beats the impostor that set its own floor
        assert {res.names[s].via for s in m.ids[p]} == {"face"}
        assert {res.names[s].reason for s in m.ids[p]} == {"named"}
        assert named(res, m.ids[f"u{p}"]) == [p] * 6
    assert named(res, m.ids["dD"]) == [None] * 8
    assert reasons(res, m.ids["dD"]) == ["below_floor"] * 8
    assert res.disputes == ()


def test_a_fourth_person_at_ordinary_distance_is_mostly_left_unnamed():
    m = three_people()
    m.add("fD", "D", 20)  # statistical: each picture clears a floor by chance about 1 time in 120
    res = m.run()
    assert sum(p is not None for p in named(res, m.ids["fD"])) <= 2


# 2. mixed group ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("weights", [NORMAL, HARD], ids=["ordinary", "hard-site"])
def test_pictures_of_b_under_the_name_a_come_out_b(weights):
    m = Maker(2)
    m.add("a", "A", 42, weak="A", weights=weights)
    m.add("x", "B", 18, weak="A", weights=weights)  # 30% of "A" is B
    m.add("b", "B", 60, weak="B", weights=weights)
    m.add("c", "C", 60, weak="C", weights=weights)
    res = m.run()
    assert res.disputes == ()
    assert named(res, m.ids["x"]) == ["B"] * 18
    assert named(res, m.ids["a"]) == ["A"] * 42
    assert named(res, m.ids["b"]) == ["B"] * 60
    # cleaning only removes: B's pictures left A and were named B, but they are not members of B's group
    assert res.face_groups["A"].n_members == 42 and res.face_groups["B"].n_members == 60


def test_pictures_under_a_name_that_fit_nobody_leave_every_group():
    m = Maker(40)
    m.add("a", "A", 42, weak="A")
    # 30% of the pictures under "A" show a person nobody is named after
    m.add("j", "D", 18, weak="A", weights=dict(a_c=0.1, a_i=0.9, a_n=0.4))
    m.add("b", "B", 60, weak="B")
    m.add("c", "C", 60, weak="C")
    res = m.run()
    assert named(res, m.ids["j"]) == [None] * 18 and reasons(res, m.ids["j"]) == ["below_floor"] * 18
    assert [res.face_groups[p].n_members for p in "ABC"] == [42, 60, 60]  # they sit in no group, A's or anyone's


def test_hard_site_floors_adapt_without_new_parameters():
    easy, hard = three_people(5, NORMAL).run(), three_people(5, HARD).run()
    # the floor sits just above what the other people reach, wherever that is
    assert hard.face_groups["A"].floor > easy.face_groups["A"].floor + 0.2
    assert hard.face_groups["A"].floor > 0.5
    m = three_people(5, HARD)
    res = m.run()
    for p in "ABC":
        assert named(res, m.ids[p]) == [p] * 60


def test_a_larger_fmr_lowers_the_floor():
    m = three_people(5)
    strict = m.run().face_groups["A"].floor  # 120 impostors, fmr 0.005: nothing may be above it
    loose = m.run(Params(fmr=0.05)).face_groups["A"].floor  # 6 of them may be above it
    assert loose < strict


# 3. majority wrong / 4. anchors fix it -----------------------------------------------------------------------------


def majority_wrong(anchors=0, seed=3):
    m = Maker(seed)
    m.add("a", "B", 48, weak="A")  # "A" is 80% B's face
    m.add("aa", "A", 12, weak="A")
    m.add("b", "B", 60, weak="B")
    m.add("c", "C", 60, weak="C")
    m.add("uB", "B", 6)
    m.add("uA", "A", 6)
    m.add("uC", "C", 6)
    if anchors:
        m.add("anA", "A", anchors, weak="A", anchor="A")
        m.add("anB", "B", anchors, weak="B", anchor="B")
    return m


def test_when_the_majority_of_a_name_is_another_person_both_names_are_disputed():
    m = majority_wrong()
    res = m.run()
    assert [(a, b) for a, b, _, _ in res.disputes] == [("A", "B")]
    _, _, ba, ab = res.disputes[0]
    assert ba >= 0.25 and ab >= 0.25
    probes = m.ids["a"] + m.ids["aa"] + m.ids["b"] + m.ids["uB"] + m.ids["uA"]
    assert all(res.names[s].person not in ("A", "B") for s in probes)
    for s in m.ids["b"] + m.ids["uB"]:  # B's pictures land in the merged set, which gives no name
        assert (res.names[s].reason, res.names[s].others) == ("disputed", ("A", "B"))
    assert named(res, m.ids["c"] + m.ids["uC"]) == ["C"] * 66
    g = res.face_groups["A|B"]
    assert (g.disputed_with, g.anchor_seeded, g.active) == (("A", "B"), False, True)
    assert "A" not in res.face_groups and "B" not in res.face_groups


def test_enough_confirmed_pictures_settle_the_dispute():
    m = majority_wrong(anchors=20)
    res = m.run()
    assert res.disputes == () and res.refuted == () and res.displaced == (("A", "B"),)
    # B's confirmed pictures home in the merged set: it is B's group, and what stays in it is named B
    assert named(res, m.ids["a"] + m.ids["b"] + m.ids["uB"]) == ["B"] * 114
    # A's confirmed pictures find no home, so A gets a group of its own from them; true A pictures are named A
    assert named(res, m.ids["uA"] + m.ids["aa"]) == ["A"] * 18
    assert named(res, m.ids["c"] + m.ids["uC"]) == ["C"] * 66
    a, b = res.face_groups["A"], res.face_groups["B"]
    assert (a.anchor_seeded, a.n_anchor, a.n_weak) == (True, 20, 0)
    assert (b.anchor_seeded, b.n_anchor, b.n_weak) == (False, 20, 120)


def test_a_name_whose_person_is_confirmed_in_another_group_is_refuted_and_its_pictures_leave():
    m = Maker(41)
    m.add("an", "A", 20, anchor="A")  # A's confirmed pictures show A's face ...
    m.add("b", "A", 60, weak="B")  # ... which is what the name "B" holds
    m.add("a", "D", 60, weak="A", weights=dict(a_c=0.1, a_i=0.9, a_n=0.4))  # and the name "A" holds a third person
    m.add("c", "C", 60, weak="C")
    res = m.run()
    assert res.displaced == (("B", "A"),) and res.refuted == ("A",)
    assert named(res, m.ids["b"]) == ["A"] * 60  # B's group is A's
    assert named(res, m.ids["a"]) == [None] * 60 and reasons(res, m.ids["a"]) == ["below_floor"] * 60
    assert "B" not in {n.person for n in res.names.values()}
    assert res.face_groups["A"].n_weak == 60 and res.face_groups["A"].n_members == 80


def test_a_confirmed_person_with_a_group_of_their_own_displaces_nobody():
    m = Maker(32)
    m.add("an", "A", 12, weak="A", anchor="A")
    m.add("wa", "A", 30, weak="A")
    m.add("b", "B", 60, weak="B")
    m.add("c", "C", 60, weak="C")
    res = m.run()
    assert res.displaced == () and res.refuted == () and res.disputes == ()
    assert named(res, m.ids["b"]) == ["B"] * 60
    assert res.face_groups["A"].anchor_seeded is False  # A's weak group was theirs: no group made from anchors


def test_hard_anchors_do_not_spoil_the_average():
    m = Maker(43)
    m.add("a", "A", 60, weak="A")
    m.add("an", "A", 12, weak="A", anchor="A", weights=dict(a_c=0.5, a_i=0.8, a_n=2.0))  # very noisy confirmed pictures
    m.add("b", "B", 60, weak="B")
    m.add("c", "C", 60, weak="C")
    res = m.run()
    g = res.face_groups["A"]
    assert named(res, m.ids["a"]) == ["A"] * 60
    assert (g.n_weak, g.n_anchor, g.anchor_seeded) == (60, 12, False) and g.n_members - g.n_anchor >= 54
    assert res.displaced == () and res.refuted == ()


def with_anchors(*groups, seed=44):
    """A, B, C with 60 weak pictures each, plus anchors of A: (face of, count, noise) per entry; "junk" = random."""
    m = Maker(seed)
    for p in "ABC":
        m.add(p, p, 60, weak=p)
    for k, (who, n, a_n) in enumerate(groups):
        if who == "junk":
            for j in range(n):
                v = _unit(m.rng.normal(size=DIM))
                m.items.append(Item(f"junk{k}_{j}", True, None, "A", v, None))
        else:
            m.add(f"an{k}", who, n, anchor="A", weights=dict(a_c=0.5, a_i=0.8, a_n=a_n))
    return m


@pytest.mark.parametrize("anchors", [
    [("B", 1, 0.05)],  # a single vote is not a home
    [("B", 4, 0.05), ("junk", 16, 0)],  # 4 votes of 20 anchors: below ceil(0.25 * 20) = 5
    [("B", 6, 0.05), ("C", 6, 0.05)],  # a tie between two groups is not a home
    [("B", 20, 2.0)],  # pictures no typical picture of any group: no vote at all
], ids=["one-vote", "below-share", "tie", "not-typical"])
def test_without_a_home_confirmed_pictures_join_their_persons_own_weak_group(anchors):
    m = with_anchors(*anchors)
    res = m.run()
    assert res.displaced == () and res.refuted == () and res.disputes == ()
    a = res.face_groups["A"]
    assert (a.anchor_seeded, a.n_weak, a.n_anchor) == (False, 60, sum(n for _, n, _ in anchors))


def test_enough_votes_make_a_home():
    m = with_anchors(("B", 5, 0.05), ("junk", 15, 0))  # 5 of 20 = ceil(0.25 * 20): a home
    res = m.run()
    assert res.displaced == (("B", "A"),) and res.refuted == ("A",)


def test_a_confirmed_person_without_a_group_gets_one_from_enough_anchors():
    m = three_people(45)
    m.add("e", "E", 3, anchor="E")  # min_anchors = 3
    m.add("ue", "E", 6)
    m.add("f", "F", 2, anchor="F")  # 2 anchors and nothing else: not enough
    m.add("uf", "F", 6)
    res = m.run()
    e = res.face_groups["E"]
    assert (e.anchor_seeded, e.n_anchor, e.n_members) == (True, 3, 3)
    assert named(res, m.ids["ue"]) == ["E"] * 6
    assert "F" not in res.face_groups and named(res, m.ids["uf"]) == [None] * 6
    assert named(res, m.ids["f"]) == ["F"] * 2  # an anchor is named however little there is to go on


def test_anchors_and_the_weak_pictures_of_a_dead_weak_group_can_make_a_group_together():
    m = three_people(46)
    m.add("e", "E", 2, anchor="E")
    m.add("we", "E", 3, weak="E")  # 3 weak pictures < min_core, but 2 + 3 = 5
    res = m.run()
    e = res.face_groups["E"]
    assert (e.anchor_seeded, e.n_weak, e.n_anchor, e.n_members) == (True, 3, 2, 5)


def test_a_look_alike_in_one_direction_only_is_not_a_dispute():
    m = Maker(4)
    m.add("a", "D", 60, weak="A", weights=dict(a_c=0.5, a_i=0.8, a_n=1.5))  # very loose group
    m.add("b", "D", 60, weak="B", weights=dict(a_c=0.5, a_i=0.8, a_n=0.15))  # very tight group, same person
    res = m.run()
    assert res.disputes == ()
    assert set(res.face_groups) == {"A", "B"}


# 5. anchors are final ---------------------------------------------------------------------------------------------


def test_an_anchor_is_final_and_its_weak_name_is_ignored():
    m = Maker(5)
    m.add("a", "A", 60, weak="A")
    m.add("b", "B", 60, weak="B")
    m.add("c", "C", 60, weak="C")
    odd = m.add("odd", "B", 1, weak="B", anchor="A")  # a human said A, the face is B's, the board said B
    res = m.run()
    n = res.names[odd[0]]
    assert (n.person, n.via, n.reason) == ("A", "anchor", "anchor")
    # it sits in A only: the weak name B was not used (B's own members are not asserted: this picture is B's face
    # inside A's group, so it is the highest impostor score B has and sets B's floor)
    assert (res.face_groups["A"].n_anchor, res.face_groups["A"].n_members) == (1, 61)
    assert res.face_groups["B"].n_weak == 60


def test_an_anchor_without_any_usable_vector_is_still_named():
    m = Maker(5)
    m.add("a", "A", 60, weak="A")
    m.add("b", "B", 60, weak="B")
    bare = m.add("bare", None, 1, anchor="C")
    assert named(m.run(), bare) == ["C"]


# 6. exactly one ---------------------------------------------------------------------------------------------------


def test_a_picture_halfway_between_two_people_is_ambiguous_best_first():
    m = three_people(6)
    mid = _unit(0.6 * ID["A"] + 0.4 * ID["B"])
    halves = m.add("h", mid, 10)
    res = m.run()
    for s in halves:
        n = res.names[s]
        assert (n.person, n.reason, n.others) == (None, "ambiguous", ("A", "B"))
    assert named(res, m.ids["A"]) == ["A"] * 60


# 7. too little to judge --------------------------------------------------------------------------------------------


def test_one_active_group_names_nothing():
    m = Maker(7)
    m.add("a", "A", 60, weak="A")
    m.add("b", "B", 3, weak="B")  # min_core = 5: B is not a group
    m.add("u", "A", 5)
    res = m.run()
    assert set(reasons(res, [i.sid for i in m.items])) == {"no_groups"}
    assert "B" not in res.face_groups  # 3 weak pictures are no group: no report either


def test_a_group_with_few_impostor_scores_has_no_bar_and_cannot_make_a_picture_ambiguous():
    m = Maker(8)
    m.add("a", "A", 60, weak="A")
    m.add("b", "B", 20, weak="B")  # A's impostors: 20 scores, below min_impostors = 30
    pa = m.add("pa", "A", 6)
    pb = m.add("pb", "B", 6)
    mid = m.add("mid", _unit(0.3 * ID["A"] + 0.7 * ID["B"]), 6)  # nearer to B, but A would clear it too
    res = m.run()
    assert res.face_groups["A"].floor is None and res.face_groups["B"].floor is not None
    assert named(res, pa + m.ids["a"]) == [None] * 66
    assert set(reasons(res, pa)) == {"no_floor"}
    assert named(res, pb + m.ids["b"]) == ["B"] * 26
    assert named(res, mid) == ["B"] * 6  # A cannot claim a picture, so it cannot make it ambiguous either


def test_a_picture_nearer_to_a_group_without_a_bar_than_to_the_group_it_cleared_is_not_named():
    m = Maker(8)
    m.add("a", "A", 60, weak="A")  # no bar: its impostors are B's 20 members
    m.add("b", "B", 20, weak="B")
    lean = m.add("lean", _unit(0.7 * ID["A"] + 0.3 * ID["B"]), 6)  # clears B's bar (about 0.48), scores 0.8 against A
    res = m.run()
    assert res.face_groups["A"].floor is None and res.face_groups["B"].floor is not None
    assert "B" not in named(res, lean)  # a group too thin for a bar can stop a wrong name, never give one
    assert sum(res.names[s].reason == "not_nearest" and res.names[s].others == ("B", "A") for s in lean) >= 5


# 8. bad input -----------------------------------------------------------------------------------------------------


def test_unusable_vectors_count_as_absent_without_raising():
    m = Maker(9)
    m.add("a", "A", 40, weak="A")
    m.add("b", "B", 40, weak="B")
    base = [i.sid for i in m.items]
    nan = np.full(DIM, np.nan)
    inf = np.ones(DIM)
    inf[3] = np.inf
    bad = {"nan": nan, "inf": inf, "empty": np.array([]), "twod": np.ones((4, 4)), "zero": np.zeros(DIM)}
    for k, v in bad.items():
        m.items.append(Item(f"bad_{k}", True, None, None, v, None))
        m.items.append(Item(f"badbody_{k}", True, None, None, None, v))
    m.items.append(Item("bad_both", True, None, None, nan, inf))
    res = m.run()
    assert set(reasons(res, [s for s in res.names if s.startswith("bad")])) == {"no_vector"}
    assert named(res, base) == ["A"] * 40 + ["B"] * 40  # the bad rows did not disturb anyone


def test_a_bad_face_row_does_not_hide_a_good_body():
    m = three_people(9)
    odd = Item("odd", True, None, None, np.full(DIM, np.nan), clothes(m.rng, "A", 1)[0])
    m.items.append(odd)
    assert m.run().names["odd"].reason == "no_groups"  # nobody has body examples: judged by body, not by face


def test_a_face_or_body_of_another_size_raises_and_names_the_sid():
    m = three_people(10)
    m.items.append(Item("short_face", True, None, None, np.ones(64), None))
    with pytest.raises(ValueError, match="short_face"):
        m.run()
    m = three_people(10)
    m.items.append(Item("body1", True, None, None, None, np.ones(10)))
    m.items.append(Item("body2", True, None, None, None, np.ones(10)))
    m.items.append(Item("body_odd", True, None, None, None, np.ones(11)))
    with pytest.raises(ValueError, match="body_odd"):
        m.run()


def test_a_repeated_sid_raises():
    m = three_people(11)
    m.items.append(m.items[0])
    with pytest.raises(ValueError, match=m.items[0].sid):
        m.run()


@pytest.mark.parametrize("field", ["weak", "anchor"])
def test_a_name_with_a_bar_raises_and_names_the_sid(field):
    m = three_people(11)
    m.items.append(Item("barred", True, "A|B" if field == "weak" else None, "A|B" if field == "anchor" else None,
                        m.items[0].face, None))
    with pytest.raises(ValueError, match="barred"):
        m.run()


def _floor_of(scores, fmr=0.005):
    x = np.asarray(scores, float)
    return _floors(x[:, None], np.ones(len(x), int), np.array([True]), Params(fmr=fmr, min_impostors=5))[0]


def test_the_floor_is_the_higher_of_the_counted_and_the_fitted_estimate():
    # few, tame scores: the highest one is 1.0 but mean + z * std (population std 0.4, z = 2.5758) is higher
    assert _floor_of([0, 0, 0, 0, 1]) == pytest.approx(0.2 + 2.5758293035489004 * 0.4, abs=1e-9)
    # a heavy tail: 10% of the scores sit at 1.0, so the counted value (the maximum) beats mean + z * std = 0.873
    assert _floor_of([0.0] * 90 + [1.0] * 10) == 1.0
    # fmr 0.05 of 200 scores lets 10 through: the counted value is 0.0 (5 above it), so the fitted one decides
    tail = [0.0] * 195 + [0.5, 0.6, 0.7, 0.8, 0.9]
    assert _floor_of(tail, fmr=0.05) == pytest.approx(np.mean(tail) + 1.6448536269514722 * np.std(tail))
    # enough scores for the count to allow some above it: 200 scores, fmr 0.01 lets 2 through
    assert _floor_of(list(np.linspace(0, 1, 200)), fmr=0.01) == pytest.approx(
        max(np.linspace(0, 1, 200)[197], 0.5 + 2.3263478740408408 * np.std(np.linspace(0, 1, 200))))


@pytest.mark.parametrize("bad", [dict(fmr=0), dict(fmr=1), dict(fmr=-0.1), dict(fmr=float("nan")), dict(min_core=0),
                                 dict(min_core=1), dict(dup_share=0), dict(dup_share=1.5), dict(dup_share=-0.1),
                                 dict(min_anchors=0), dict(min_impostors=0), dict(rounds=0), dict(body_topk=0),
                                 dict(hard_share=1.0), dict(hard_share=-0.1), dict(hard_share=float("nan"))])
def test_bad_params_raise(bad):
    with pytest.raises(ValueError):
        Params(**bad)


# 9. bodies --------------------------------------------------------------------------------------------------------


def with_clothes(seed=12, same_clothes=False):
    """A, B, C: 80 probes each that have a face and clothes, and 8 more that have clothes only."""
    m = Maker(seed)
    for p in "ABC":
        c = UNIFORM if same_clothes else p
        m.add(p, p, 80, weak=p, body=c)
        m.add(f"nf{p}", None, 8, body=c)
    return m


def test_a_picture_without_a_face_is_named_by_the_clothes_of_the_day():
    m = with_clothes()
    res = m.run()
    for p in "ABC":
        got = named(res, m.ids[f"nf{p}"])
        assert got.count(p) >= 7 and all(g in (p, None) for g in got)  # statistical: a fluke makes it ambiguous
        assert {res.names[s].via for s in m.ids[f"nf{p}"] if res.names[s].person} == {"body"}
        assert named(res, m.ids[p]) == [p] * 80 and {res.names[s].via for s in m.ids[p]} == {"face"}
    assert res.body_groups["A"].n_members == 80 and res.body_groups["A"].active


def test_when_everyone_wears_the_same_uniform_the_clothes_name_nobody():
    m = with_clothes(same_clothes=True)
    res = m.run()
    flat = [s for p in "ABC" for s in m.ids[f"nf{p}"]]
    assert sum(p is not None for p in named(res, flat)) <= 3  # statistical, 24 pictures
    # the faces still sort their own pictures
    assert named(res, m.ids["A"]) == ["A"] * 80


STRANGER = dict(a_c=0.1, a_i=0.9, a_n=0.4)  # a face unlike everybody's: no face bar is cleared


def test_a_face_that_fits_nobody_points_at_its_nearest_group_and_clothes_may_agree_with_it():
    m = with_clothes()
    # a stranger's face with a little of A in it: below every bar, but nearest to A
    face = _unit(0.25 * ID["A"] + 0.9 * ID["D"])
    odd = m.add("odd", face, 8, body="A", weights=STRANGER)
    res = m.run()
    plain = Maker_without_body(m, odd)
    assert sum(plain.names[s].reason == "below_floor" and plain.names[s].others == ("A",) for s in odd) >= 7
    got = named(res, odd)
    assert got.count("A") >= 7 and all(g in ("A", None) for g in got)  # statistical: a fresh picture may clear two bars
    assert {res.names[s].via for s in odd if res.names[s].person} == {"body"}
    assert [res.body_groups[p].n_members for p in "ABC"] == [80, 80, 80] and "D" not in res.body_groups  # no chaining


def test_a_face_that_fits_nobody_is_not_named_by_clothes_that_point_away_from_its_nearest_group():
    m = with_clothes()
    face = _unit(0.25 * ID["A"] + 0.9 * ID["D"])
    odd = m.add("odd", face, 8, body="B", weights=STRANGER)  # nearest to A, wearing B's clothes
    res = m.run()
    plain = Maker_without_body(m, odd)
    assert sum(res.names[s] == plain.names[s] for s in odd) == 8 and "B" not in named(res, odd)
    assert sum(plain.names[s].reason == "below_floor" and plain.names[s].others == ("A",) for s in odd) >= 7


def test_a_face_without_a_bar_points_at_its_nearest_group_and_only_agreeing_clothes_name_it():
    m = Maker(60)
    m.add("a", "A", 60, weak="A")  # A's impostors: the 10 members of B and C, below min_impostors: no face bar
    m.add("anA", None, 40, anchor="A", body="A")  # confirmed pictures with no face: clothes examples for A
    for p in "BC":
        m.add(p, p, 5, weak=p, body=p)
        m.add(f"pool{p}", p, 40, body=p)  # unnamed pictures the face bars of B and C name: clothes examples
    agree = m.add("t", "A", 8, body="A")  # A's face, A's clothes
    away = m.add("u", "A", 8, body="B")  # A's face, B's clothes
    res = m.run()
    plain = Maker_without_body(m, agree + away)
    assert all(plain.names[s].reason == "no_floor" and plain.names[s].others == ("A",) for s in agree + away)
    got = named(res, agree)
    assert got.count("A") >= 7 and all(g in ("A", None) for g in got)
    assert {res.names[s].via for s in agree if res.names[s].person} == {"body"}
    assert all(res.names[s] == plain.names[s] for s in away)  # B is not where the face points


def Maker_without_body(m, sids):
    """The result of the same items with those pictures' clothes taken away (their faces are unchanged)."""
    items = [replace(i, body=None) if i.sid in set(sids) else i for i in m.items]
    return regroup(items, Params())


def test_a_face_with_fewer_than_two_groups_can_be_named_by_clothes():
    m = Maker(61)
    m.add("a", "A", 60, weak="A")  # the only face group
    for p in "ABC":
        m.add(f"an{p}", None, 40, anchor=p, body=p)  # confirmed pictures with no face: clothes examples
    t = m.add("t", "D", 8, body="B", weights=STRANGER)
    res = m.run()
    assert reasons(Maker_without_body(m, t), t) == ["no_groups"] * 8
    got = named(res, t)
    assert got.count("B") >= 7 and all(g in ("B", None) for g in got)
    assert {res.names[s].via for s in t if res.names[s].person} == {"body"}


def test_an_ambiguous_face_may_only_be_named_by_clothes_among_its_own_choices():
    m = with_clothes()
    mid = _unit(0.6 * ID["A"] + 0.4 * ID["B"])  # between A and B
    inside = m.add("in", mid, 8, body="A")  # clothes say A, one of the face's choices
    outside = m.add("out", mid, 8, body="C")  # clothes say C, which the face did not offer
    res = m.run()
    plain = Maker_without_body(m, inside + outside)
    assert all(plain.names[s].reason == "ambiguous" and plain.names[s].others == ("A", "B") for s in inside + outside)
    got = named(res, inside)
    assert got.count("A") >= 7 and all(g in ("A", None) for g in got)
    assert {res.names[s].via for s in inside if res.names[s].person} == {"body"}
    for s in outside:  # outside the choices is no answer: the face answer stays as it was
        assert res.names[s] == plain.names[s] and res.names[s].reason == "ambiguous"


def test_a_face_that_names_a_person_is_a_lock_clothes_never_change():
    m = Maker(62)
    for p in "BC":
        m.add(p, p, 80, weak=p, body=p)
    m.add("a", "A", 60, weak="A")  # A's faces carry no clothes
    # A's face (named by face), C's clothes. Four examples are too few for a clothes group of A, so these pictures
    # do not help set C's bar, and only the lock keeps clothes from naming them C.
    t = m.add("t", "A", 4, body="C")
    res = m.run()
    plain = Maker_without_body(m, t)
    assert sum(res.names[s].via == "face" and res.names[s].person == "A" for s in t) >= 3
    assert "C" not in named(res, t) and all(res.names[s] == plain.names[s] for s in t)


def test_a_disputed_face_is_a_lock_too():
    m = with_clothes()
    for name in "XY":  # two names holding the same stranger's face: a disputed set
        m.add(f"p{name}", "D", 60, weak=name)
    t = m.add("t", "D", 4, body="A")  # 4 < min_core: as examples they do not make a clothes group of the set
    res = m.run()
    assert [(a, b) for a, b, _, _ in res.disputes] == [("X", "Y")]
    assert named(res, t) == [None] * 4  # clothes name nobody here
    assert sum(res.names[s].reason == "disputed" and res.names[s].others == ("X", "Y") for s in t) >= 3  # statistical


def test_a_disputed_face_is_not_named_by_clothes_even_for_a_person_the_set_names():
    m = Maker(63)
    for name in "AB":  # the names A and B both hold a stranger's face: the disputed set (A, B)
        m.add(f"p{name}", "D", 60, weak=name)
        m.add(f"an{name}", None, 20, anchor=name, body=name)  # clothes examples: confirmed, no face
    m.add("c", "C", 60, weak="C", body="C")
    t = m.add("t", "D", 4, body="A")  # a disputed face wearing A's clothes: A is one of the set's names
    res = m.run()
    assert [(a, b) for a, b, _, _ in res.disputes] == [("A", "B")]
    assert named(res, t) == [None] * 4  # a lock, not just the pointer: the set's own names never come from clothes
    assert sum(res.names[s].reason == "disputed" for s in t) >= 3


def test_an_anchor_is_never_touched_by_clothes():
    m = with_clothes()
    # confirmed Z (one picture: too few for a clothes group of Z), wearing A's clothes
    anc = m.add("anc", "D", 1, anchor="Z", body="A", weights=STRANGER)
    res = m.run()
    assert all(res.names[s] == Name(s, "Z", "anchor", None, None, "anchor", ()) for s in anc)


def test_a_face_clothes_leave_unnamed_keeps_its_own_reason_score_and_bar():
    m = with_clothes()
    t = m.add("t", "D", 8, body="E", weights=STRANGER)  # nobody wears E's clothes
    res = m.run()
    plain = Maker_without_body(m, t)
    assert sum(res.names[s] == plain.names[s] for s in t) >= 7  # statistical: a fresh picture may clear a bar
    assert all(plain.names[s].reason == "below_floor" and plain.names[s].score is not None for s in t)


def test_the_same_items_in_any_order_give_the_same_result_with_clothes_on_unlocked_faces():
    m = with_clothes()
    m.add("in", _unit(0.6 * ID["A"] + 0.4 * ID["B"]), 6, body="A")
    m.add("t", "D", 6, body="B", weights=STRANGER)
    base = m.run()
    for seed in (1, 2):
        assert m.run(order=seed) == base


def test_clothes_examples_come_from_probes_only():
    m = Maker(13)
    m.add("a", "A", 3, weak="A", body="A")  # 3 probes: not enough examples (min_core = 5)
    m.add("old", "A", 40, weak="A", probe=False, body="A")  # pictures of other days do not count
    for p in "BC":
        m.add(p, p, 30, weak=p, body=p)
    nf = m.add("nf", None, 8, body="A")
    res = m.run()
    assert res.body_groups["A"].n_members == 3 and res.body_groups["A"].active is False
    assert "A" not in named(res, nf)  # A has too few examples; a fresh picture may still clear B or C by chance
    assert set(res.names) == {i.sid for i in m.items if i.probe}


def test_clothes_with_one_active_group_name_nothing():
    m = Maker(14)
    m.add("a", "A", 30, weak="A", body="A")
    m.add("b", "B", 27, weak="B")
    m.add("bb", "B", 3, weak="B", body="B")  # B has only 3 clothes examples: not a group (min_core = 5)
    nf = m.add("nf", None, 4, body="B")
    res = m.run()
    assert res.body_groups["A"].active and not res.body_groups["B"].active
    assert reasons(res, nf) == ["no_groups"] * 4


# 11, 12. order and probes ------------------------------------------------------------------------------------------


def test_the_same_items_in_any_order_give_the_same_result():
    m = majority_wrong(anchors=3)
    m.add("nf", None, 6, body="A")
    base = m.run()
    for seed in (1, 2, 3):
        assert m.run(order=seed) == base


def test_only_probes_get_an_answer_but_other_pictures_still_shape_the_groups():
    m = Maker(15)
    only = m.add("p", "A", 3, weak="A")  # 3 probes alone are below min_core
    m.add("other", "A", 57, weak="A", probe=False)
    m.add("b", "B", 60, weak="B", probe=False)
    res = m.run()
    assert set(res.names) == set(only)
    assert named(res, only) == ["A"] * 3
    assert res.face_groups["A"].n_members == 60


# 13. group reports -------------------------------------------------------------------------------------------------


def test_group_reports_on_a_small_case_counted_by_hand():
    m = Maker(16)
    m.add("an", "A", 3, weak="A", anchor="A")  # 3 anchors: A starts from them alone
    m.add("a", "A", 4, weak="A")
    m.add("a_nf", None, 1, weak="A")  # no face: not counted anywhere
    m.add("an_nf", None, 1, weak="A", anchor="A")  # an anchor without a face: not counted either
    m.add("b", "B", 6, weak="B")
    res = m.run(Params(min_impostors=5))
    a, b = res.face_groups["A"], res.face_groups["B"]
    assert (a.n_weak, a.n_anchor, a.n_members, a.anchor_seeded, a.active) == (4, 3, 7, True, True)
    assert (b.n_weak, b.n_anchor, b.n_members, b.anchor_seeded, b.active) == (6, 0, 6, False, True)
    assert a.floor is not None and b.floor is not None and a.floor < 0.6 and b.floor < 0.6
    assert (a.coverage, b.coverage) == (1.0, 1.0)
    assert a.disputed_with == () == b.disputed_with
    assert res.disputes == ()
    assert res.names["a_nf000"].reason == "no_vector" and res.names["an_nf000"].reason == "anchor"
    assert res.body_groups == {}


def test_a_group_below_min_core_is_inactive_and_its_pictures_stay_unnamed():
    m = three_people(17)
    d = m.add("d", "D", 2, weak="D", weights=dict(a_c=0.1, a_i=0.9, a_n=0.4))  # two pictures: below min_core = 5
    res = m.run()
    assert "D" not in res.face_groups  # not a group, so nothing to report
    assert [res.face_groups[p].n_members for p in "ABC"] == [60, 60, 60]  # and its pictures sit in no group
    assert named(res, d) == [None, None] and reasons(res, d) == ["below_floor"] * 2


def test_a_handful_under_another_persons_name_is_no_group_and_no_dispute():
    m = Maker(22)
    m.add("b", "B", 60, weak="B")
    m.add("c", "C", 60, weak="C")
    m.add("a", "A", 60, weak="A")
    small = m.add("x", "B", 19, weak="X")  # 19 < min_core = 20: X is no group, so no dispute; pool
    res = m.run(Params(min_core=20))
    assert res.disputes == () and "X" not in res.face_groups
    assert named(res, small) == ["B"] * 19


def test_a_few_pictures_under_a_tiny_name_stay_in_the_pool_yet_get_named():
    m = three_people(23)
    tiny = m.add("t", "B", 3, weak="T")  # 3 < min_core: "T" is not a group
    res = m.run()
    assert [res.face_groups[p].n_members for p in "ABC"] == [60, 60, 60]  # the pool shapes no group
    assert "T" not in res.face_groups
    assert named(res, tiny) == ["B"] * 3  # judged in F5 like any picture: they clear exactly B's floor


def test_a_dropped_group_stays_dropped_and_its_pictures_are_pool():
    m = three_people(24)
    dd = dict(a_c=0.1, a_i=0.9, a_n=0.4)
    x = m.add("x", "D", 3, weak="X", weights=dd)
    m.add("xc", "C", 2, weak="X")  # X starts with 5 = min_core, but the two C pictures are removed in round 1
    wa = m.add("wa", "D", 1, weak="A", weights=dd)  # a stranger under A's name is removed from A as well
    res = m.run()
    g = res.face_groups["X"]
    assert (g.n_weak, g.n_members, g.active) == (5, 0, False)  # a group that fell below min_core never comes back
    assert [res.face_groups[p].n_members for p in "ABC"] == [60, 60, 60]  # nothing moved: the pool is no group
    assert named(res, x + wa) == [None] * 4
    assert named(res, ["xc000", "xc001"]) == ["C", "C"]  # in the pool, named in F6 like any picture


def test_a_group_that_falls_below_min_core_while_cleaning_is_dropped():
    m = three_people(18)
    x = m.add("x", "D", 4, weak="X", weights=dict(a_c=0.1, a_i=0.9, a_n=0.4))
    m.add("xc", "C", 1, weak="X")  # 5 = min_core at the start; its C picture leaves in round 1
    res = m.run()
    g = res.face_groups["X"]
    assert (g.n_weak, g.n_members, g.active) == (5, 0, False)
    assert named(res, x) == [None] * 4
    assert named(res, ["xc000"]) == ["C"]


def _exact_vectors(p_b, p_c):
    """a, b, c, p with a.b = a.c = 0.3, b.c = 0.8, p.a = 0.5, p.b = p_b, p.c = p_c (unit vectors, no noise)."""
    b1 = np.sqrt(1 - 0.09)
    cy = (0.8 - 0.09) / b1
    cz = np.sqrt(1 - 0.09 - cy**2)
    py = (p_b - 0.15) / b1
    pz = (p_c - 0.15 - cy * py) / cz
    pw = np.sqrt(1 - 0.25 - py**2 - pz**2)
    vec = lambda *v: np.concatenate([v, np.zeros(DIM - len(v))])  # noqa: E731
    return vec(1, 0, 0, 0), vec(0.3, b1, 0, 0), vec(0.3, cy, cz, 0), vec(0.5, py, pz, pw)


def _exact_case(p_b, p_c):
    """Exact vectors, no noise, so the rounds can be followed by hand. Cosines: a.b = a.c = 0.3, b.c = 0.8, and the
    picture p, weak name A, has p.a = 0.5, p.b = p_b, p.c = p_c. Six copies each of a, b, c; fmr 0.3 makes the
    floors A 0.3, B 0.8, C 0.8."""
    a, b, c, p = _exact_vectors(p_b, p_c)
    items = [Item(f"{n}{k}", True, n.upper(), None, v, None) for n, v in (("a", a), ("b", b), ("c", c))
             for k in range(6)]
    items.append(Item("p", True, "A", None, p, None))
    return regroup(items, Params(fmr=0.3, min_impostors=10))


def test_cleaning_judges_every_candidate_every_round_so_a_removed_picture_can_return():
    # round 1 removes p from A (B fits it better, 0.6 against 0.5); in round 2 it clears A's floor (0.3) and nobody
    # else's (B and C: 0.8), so it is a member again
    res = _exact_case(0.6, 0.6)
    assert (res.face_groups["A"].n_weak, res.face_groups["A"].n_members) == (7, 7)
    # it clears only A's bar, but B is nearer (0.6 against 0.5): no name. Cleaning is not touched by that guard.
    assert res.disputes == () and (res.names["p"].person, res.names["p"].reason) == (None, "not_nearest")


def test_a_removed_picture_that_clears_another_groups_floor_too_stays_out():
    # p.b = 0.9 now clears B's floor (0.8) as well as A's: two floors, so no membership
    res = _exact_case(0.9, 0.75)
    assert (res.face_groups["A"].n_weak, res.face_groups["A"].n_members) == (7, 6)
    assert res.names["p"].reason == "ambiguous" and res.names["p"].others == ("B", "A")


def test_a_picture_is_never_named_by_a_group_because_of_itself():
    m = three_people(19)
    dd = dict(a_c=0.1, a_i=0.9, a_n=0.4)
    two = m.add("t", "D", 1, weak="X", weights=dd) + m.add("t2", "E", 1, weak="X", weights=dd)  # two strangers
    res = m.run(Params(min_core=2))
    assert "X" not in named(res, two)  # each one is scored against the other only, which is nothing like it


def test_cleaning_stops_and_reports_how_many_rounds_it_took():
    res = three_people(20).run()
    assert 2 <= res.rounds <= 4  # round 1 has no floors; it stops on a later round that changes nothing
    assert three_people(20).run(Params(rounds=1)).rounds == 1



# 14. quality classes for face floors -----------------------------------------------------------------------------


def _hard_items(q_h=1.0):
    """Exact vectors, no noise, so every floor can be worked out by hand. A: 12 copies of a (quality 10), B: 6 copies
    of b (quality 1, hard), C: 12 copies of c (quality 10), and one probe h with no name, quality ``q_h``.
    Scores against A's group: b 0.6, c 0.3, h 0.5. So A's impostors are 12 easy at 0.3 and 6 hard at 0.6."""
    vec = lambda *v: np.concatenate([v, np.zeros(DIM - len(v))])  # noqa: E731
    a, b = vec(1), vec(0.6, 0.8)
    c, h = vec(0.3, 0, np.sqrt(0.91)), vec(0.5, 0, 0, np.sqrt(0.75))
    items = [Item(f"a{k:02d}", True, "A", None, a, None, 10.0) for k in range(12)]
    items += [Item(f"b{k:02d}", True, "B", None, b, None, 1.0) for k in range(6)]
    items += [Item(f"c{k:02d}", True, "C", None, c, None, 10.0) for k in range(12)]
    items.append(Item("h", True, None, None, h, None, q_h))
    return items


HARDP = Params(fmr=0.34, min_impostors=5)  # 0.34 of 18 scores = 6 may be above the single-class floor


def test_a_hard_impostor_that_clears_the_single_class_floor_does_not_clear_its_own_class_floor():
    res = regroup(_hard_items(), HARDP)
    a = res.face_groups["A"]
    assert (a.floor, a.floor_hard) == (pytest.approx(0.3), pytest.approx(0.6))  # easy and hard impostors apart
    h = res.names["h"]
    assert (h.person, h.reason, h.floor) == (None, "below_floor", pytest.approx(0.6))  # h.a = 0.5 < 0.6
    single = regroup(_hard_items(), replace(HARDP, hard_share=0))
    assert single.names["h"].person == "A"  # one class: the mixed bar (0.46, or 0.3 once B is gone) lets it through


def test_a_class_with_too_few_impostors_falls_back_to_both_classes():
    res = regroup(_hard_items(), Params(min_impostors=7))  # A has 6 hard impostors, 12 easy ones
    a = res.face_groups["A"]
    both = 0.4 + 2.5758293035489004 * np.sqrt(0.02)  # mean + z * std of all 18 scores (12 at 0.3, 6 at 0.6)
    assert (a.floor, a.floor_hard) == (pytest.approx(0.3), pytest.approx(both))  # the hard bar from all of them
    assert (res.names["h"].reason, res.names["h"].floor) == ("below_floor", pytest.approx(both))


def test_with_too_few_impostors_even_after_the_fallback_a_group_has_no_floor_at_all():
    res = regroup(_hard_items(), Params(min_impostors=19))
    a = res.face_groups["A"]
    assert (a.floor, a.floor_hard) == (None, None) and res.names["h"].reason == "no_floor"


def test_hard_share_zero_reproduces_the_single_class_floors_exactly():
    plain = regroup([replace(i, face_quality=None) for i in _hard_items()], HARDP)
    zero = regroup(_hard_items(), replace(HARDP, hard_share=0))
    assert zero == plain
    m = three_people(26)
    rated = [replace(i, face_quality=float(m.rng.uniform(1, 20))) for i in m.items]
    assert regroup(rated, Params(hard_share=0)) == regroup(m.items, Params(hard_share=0))


def test_a_face_of_unknown_quality_is_easy():
    items = [i if i.sid != "h" else replace(i, face_quality=None) for i in _hard_items()]
    res = regroup(items, HARDP)
    assert res.face_groups["A"].floor_hard == pytest.approx(0.6)  # B is still hard (6 of 30 known)
    assert res.names["h"].person == "A"  # h is judged by the easy bar 0.3


@pytest.mark.parametrize("q", [True, np.True_, float("nan"), float("inf"), "low", [1.0]],
                         ids=["bool", "np-bool", "nan", "inf", "str", "list"])
def test_a_quality_that_is_not_a_finite_real_number_counts_as_unknown(q):
    res = regroup(_hard_items(q), HARDP)
    # unknown = easy: h is judged by 0.3; B alone is the hard class (a bool would be 1.0, nan would blind the cut)
    assert res.names["h"].person == "A" and res.face_groups["A"].floor_hard == pytest.approx(0.6)


def test_a_quality_on_a_picture_without_a_usable_face_takes_no_part_in_the_cut():
    odd = replace(_hard_items()[-1], face=np.full(DIM, np.nan), face_quality=0.001)  # tiny quality, but no face
    res = regroup(_hard_items() + [replace(odd, sid="z")], HARDP)
    assert res.face_groups["A"].floor_hard == pytest.approx(0.6) and res.names["h"].reason == "below_floor"


def test_the_same_items_in_any_order_give_the_same_result_with_qualities():
    items = _hard_items()
    base = regroup(items, HARDP)
    for seed in (1, 2, 3):
        shuffled = list(items)
        np.random.default_rng(seed).shuffle(shuffled)
        assert regroup(shuffled, HARDP) == base


# 15. a pile that two people's confirmed pictures choose is dissolved ----------------------------------------------


def two_peoples_pile(seed=50):
    m = Maker(seed)
    for name in "XY":  # two board names that each hold the same mix of A's and B's faces
        m.add(f"{name}a", "A", 30, weak=name)
        m.add(f"{name}b", "B", 30, weak=name)
    m.add("an", "A", 20, anchor="A")
    m.add("bn", "B", 20, anchor="B")
    m.add("c", "C", 60, weak="C")
    return m


def test_a_pile_chosen_by_two_peoples_confirmed_pictures_is_dissolved():
    m = two_peoples_pile()
    res = m.run()
    assert res.disputes == ()  # no "homed" dispute any more
    assert "X|Y" not in res.face_groups and "A|B" not in res.face_groups  # no group is made from the pile
    assert res.dissolved == (("X|Y", ("A", "B")),)
    for person in "AB":  # each person is served by a group of their own, made from their anchors
        g = res.face_groups[person]
        assert (g.anchor_seeded, g.n_anchor, g.n_members, g.n_weak) == (True, 20, 20, 0)
    # the pile's pictures are in the pool and are named by those two groups: right, and none of them wrong
    mine, theirs = m.ids["Xa"] + m.ids["Ya"], m.ids["Xb"] + m.ids["Yb"]
    got_a, got_b = named(res, mine), named(res, theirs)
    assert got_a.count("A") >= 55 and got_b.count("B") >= 55  # statistical: a fresh picture may clear two floors
    assert "B" not in got_a and "A" not in got_b
    assert res.refuted == () and res.displaced == ()


def test_the_dissolved_result_does_not_depend_on_the_order_of_the_items():
    m = two_peoples_pile()
    base = m.run()
    for seed in (1, 2):
        assert m.run(order=seed) == base


def test_a_person_inside_a_disputed_set_gets_a_group_of_their_own_and_leaves_the_set():
    m = Maker(51)
    m.add("a", "D", 60, weak="A")  # the names A and B both hold D's face: a disputed set nobody settles
    m.add("b", "D", 60, weak="B")
    m.add("an", "E", 20, anchor="A")  # A's confirmed pictures show someone else's face than the pile
    m.add("uE", "E", 6)
    m.add("c", "C", 60, weak="C")
    res = m.run()
    assert res.face_groups["A|B"].n_anchor == 0  # the anchors are never members of a disputed set
    own = res.face_groups["A"]  # a disputed set that contains the name does not count as their group
    assert (own.anchor_seeded, own.n_anchor, own.n_members) == (True, 20, 20)
    assert named(res, m.ids["uE"]) == ["A"] * 6
    assert res.dissolved == () and [(a, b) for a, b, _, _ in res.disputes] == [("A", "B")]
    assert set(reasons(res, m.ids["a"] + m.ids["b"])) == {"disputed"}


def test_a_person_without_enough_anchors_inside_a_disputed_set_has_anchors_in_no_group():
    m = Maker(52)
    m.add("a", "D", 60, weak="A")
    m.add("b", "D", 60, weak="B")
    one = m.add("an", "E", 2, anchor="A")  # 2 < min_anchors and no weak pictures to add: no group of their own
    m.add("c", "C", 60, weak="C")
    res = m.run()
    assert res.face_groups["A|B"].n_anchor == 0 and "A" not in res.face_groups
    assert named(res, one) == ["A", "A"]  # an anchor still keeps its own name


# 16. the nearest-group guard (F6, B4) ------------------------------------------------------------------------------


def verdict(s, floors, labels, active=None):
    return _verdict("x", np.array(s, dtype=float), floors, active or [True] * len(s), labels, "face")


AB = [("A",), ("B",)]


def test_a_picture_that_clears_only_one_bar_but_is_nearer_to_another_group_is_not_named():
    n = verdict([0.5, 0.6], [0.3, 0.8], AB)  # clears A's bar, scores higher against B, whose bar it misses
    assert (n.person, n.via, n.reason, n.others) == (None, None, "not_nearest", ("A", "B"))
    assert (n.score, n.floor) == (0.5, 0.3)  # those of the cleared group


def test_a_picture_whose_nearest_group_also_clears_is_named_as_before():
    n = verdict([0.5, 0.4], [0.3, 0.8], AB)
    assert (n.person, n.via, n.reason, n.others, n.score, n.floor) == ("A", "face", "named", (), 0.5, 0.3)


def test_a_disputed_set_is_disputed_when_nearest_and_not_nearest_when_it_is_not():
    sets = [("X", "Y"), ("B",)]
    near = verdict([0.9, 0.2], [0.3, 0.8], sets)
    assert (near.person, near.reason, near.others) == (None, "disputed", ("X", "Y"))
    far = verdict([0.5, 0.6], [0.3, 0.8], sets)
    assert (far.reason, far.others) == ("not_nearest", ("X", "Y", "B"))


def test_when_scores_tie_the_lower_group_index_is_the_nearest():
    assert verdict([0.5, 0.5], [0.3, 0.9], AB).reason == "named"  # the cleared group is index 0: it is the nearest
    n = verdict([0.5, 0.5], [0.9, 0.3], AB)  # the cleared group is index 1; index 0 ties and wins
    assert (n.reason, n.others) == ("not_nearest", ("B", "A"))


def test_only_active_groups_count_as_nearest():
    n = verdict([0.5, 0.9, 0.4], [0.3, None, 0.9], [("A",), ("B",), ("C",)], active=[True, False, True])
    assert (n.person, n.reason) == ("A", "named")  # B scores higher but is not a group tonight


def test_ambiguous_and_below_floor_are_unchanged_by_the_guard():
    two = verdict([0.5, 0.6], [0.3, 0.4], AB)
    assert (two.reason, two.others) == ("ambiguous", ("B", "A"))
    none = verdict([0.5, 0.6], [0.7, 0.8], AB)
    assert (none.reason, none.person) == ("below_floor", None)
    nobar = verdict([0.5, 0.6], [0.7, None], AB)
    assert nobar.reason == "no_floor"


def test_the_guard_changes_the_name_but_not_who_is_a_member():
    res = _exact_case(0.6, 0.55)
    p = res.names["p"]
    assert (p.person, p.reason, p.others) == (None, "not_nearest", ("A", "B"))
    assert p.score == pytest.approx(0.5) and p.floor < 0.5  # the cleared group's: p clears A's bar
    # cleaning still keeps p in A (it clears A's bar and nobody else's in round 2): 6 copies + p
    assert [res.face_groups[k].n_members for k in "ABC"] == [7, 6, 6]


def test_a_picture_that_nobody_else_is_nearer_to_is_named_in_the_same_world():
    res = _exact_case(0.35, 0.2)  # p.a = 0.5 is its highest score
    assert (res.names["p"].person, res.names["p"].reason) == ("A", "named")
    assert [res.face_groups[k].n_members for k in "ABC"] == [7, 6, 6]


def _exact_with_clothes(p_b, p_c, clothes_of_p):
    """The exact face case, every copy also with a clothes vector (e0, e1, e2 of an 8-dim space) and p with
    ``clothes_of_p`` (an index): clothes bars are all 0, so p's clothes name whoever it wears."""
    a, b, c, p = _exact_vectors(p_b, p_c)
    eye = np.eye(8)
    items = [Item(f"{n}{k}", True, n.upper(), None, v, eye[i])
             for i, (n, v) in enumerate((("a", a), ("b", b), ("c", c))) for k in range(6)]
    items.append(Item("p", True, "A", None, p, eye[clothes_of_p]))
    return regroup(items, Params(fmr=0.3, min_impostors=10))


@pytest.mark.parametrize("wears, expect", [(0, "A"), (1, "B"), (2, None)], ids=["A", "B", "C-not-a-choice"])
def test_a_not_nearest_face_may_be_named_by_clothes_only_among_its_own_choices(wears, expect):
    res = _exact_with_clothes(0.6, 0.55, wears)
    p = res.names["p"]
    if expect is None:  # C is not one of the face's candidates (A, B): the face answer stays as it was
        assert (p.person, p.reason, p.others, p.via) == (None, "not_nearest", ("A", "B"), None)
    else:
        assert (p.person, p.via, p.reason) == (expect, "body", "named")


def test_a_faceless_picture_whose_clothes_answer_is_not_nearest_records_that_answer():
    a, b, c, p = _exact_vectors(0.6, 0.55)
    items = [Item(f"{n}{k}", True, None, n.upper(), None, v) for n, v in (("a", a), ("b", b), ("c", c))
             for k in range(6)]  # confirmed pictures with no face: the clothes examples
    items.append(Item("q", True, None, None, None, p))
    q = regroup(items, Params(fmr=0.3, min_impostors=10)).names["q"]
    assert (q.person, q.via, q.reason, q.others) == (None, None, "not_nearest", ("A", "B"))
    assert q.score == pytest.approx(0.5) and q.floor == pytest.approx(0.3)


def test_the_same_items_in_any_order_give_the_same_result_with_the_guard():
    a, b, c, p = _exact_vectors(0.6, 0.55)
    items = [Item(f"{n}{k}", True, n.upper(), None, v, np.eye(8)[i])
             for i, (n, v) in enumerate((("a", a), ("b", b), ("c", c))) for k in range(6)]
    items.append(Item("p", True, "A", None, p, np.eye(8)[1]))
    base = regroup(items, Params(fmr=0.3, min_impostors=10))
    for seed in (1, 2, 3):
        shuffled = list(items)
        np.random.default_rng(seed).shuffle(shuffled)
        assert regroup(shuffled, Params(fmr=0.3, min_impostors=10)) == base


def test_a_face_that_cleared_no_bar_points_at_its_nearest_group_and_clothes_point_nowhere():
    below = verdict([0.5, 0.6], [0.7, 0.8], AB)
    assert (below.reason, below.others, below.score) == ("below_floor", ("B",), 0.6)  # the picture's best score
    nobar = verdict([0.5, 0.6], [0.7, None], AB)
    assert (nobar.reason, nobar.others) == ("no_floor", ("B",))
    assert verdict([0.5, 0.5], [0.9, 0.9], AB).others == ("A",)  # a tie goes to the lower index
    assert verdict([0.5, 0.6], [0.9, 0.9], [("X", "Y"), ("B",)]).others == ("B",)
    by_clothes = _verdict("x", np.array([0.5, 0.6]), [0.7, 0.8], [True, True], AB, "body")
    assert (by_clothes.reason, by_clothes.others) == ("below_floor", ())  # clothes keep no pointer
