# Nightly regroup: sorting the pictures inside each named person

Status: PROPOSED, prototype stage (owner 2026-11-20). Nothing here is wired into `run` yet.

## Why

Three layers name a picture, each more accurate than the one below:

1. **The board's name**, given live with small models. A board-named person can hold pictures of several people.
2. **The nightly machine name**, given by this job with larger face and body models and the whole day at hand.
3. **A human confirmation**, final and ground truth.

Until 0.2.6 the job learned who is who only from human names (plan §5.3: "never machine labels"), so it could not
work where nobody had confirmed every person. Owner, 2026-11-20: the board's names are the starting point, with or
without human confirmation; the job's task is to **sort the pictures inside each named person**. If A holds
pictures of A, B and C, A must end up holding only A, and the others must move to B and C.

Owner's steps:

- A person **with** human-confirmed pictures: start from those, keep what matches, take out what does not.
- A person **without**: majority vote. The biggest group of look-alike pictures is that person; the rest goes to
  a pool.
- At the end, every pooled picture is assigned to the cleaned person it fits.

Four safeguards the owner agreed to:

1. **The majority can be the wrong person.** If two names end up showing the same face, the machine names
   neither and reports the pair as a question for a human.
2. **The pool has a "nobody" exit.** A picture is named only when it clearly fits exactly one person.
3. **Pictures the board left unnamed go into the pool too.**
4. **"Fits" is set per site from that site's own pictures**, never carried over from another site.

And one rule of hygiene: the sort is redone from scratch every night from the board's names, the human
confirmations and, for people with too few pictures tonight, the face names the nightly machine gave earlier
nights' pictures ("Earlier nights"; owner 2026-11-21: "a picture belongs to someone, and it was decided already").
Once the sorter is the nightly job (plan decision 38), those are its own earlier names. They enter only as weak
names: re-judged every night, never above a human, and a person whose machine names disagree with human answers
is not topped up (E5).

## Terms

- **Item**: one picture (sighting) with what the sort needs.
- **Weak name**: the name a picture carries now and nobody confirmed: a staff name without the reviewed mark,
  else the board's name. `None` when it has neither.
- **Anchor**: the person a human confirmed for this very picture (the reviewed mark, or a check-page / correction
  list answer). Final.
- **Group**: the pictures currently held under one person.
- **Centroid**: the average of a group's face vectors, scaled back to length 1.
- **Score**: the similarity of a picture to a group. Face: dot product with the group's centroid. Body: mean of
  the picture's top-k dot products with the group's body examples of that day.
- **Leave-one-out (LOO)**: when a picture is itself a member of the group it is scored against, the group is
  taken without that picture.
- **Impostors of a group**: pictures held by the *other* groups, and other people's anchors.
- **Floor of a group**: the score that only a share `fmr` of its impostors exceed. Per group, per site, per night,
  and per quality class (F5).
- **Face quality**: how clear a face is, by the face model's own measure (the length of its feature vector before
  scaling). Low quality = a blurred, turned or small face.
- **Hard picture**: a face among the least clear share `hard_share` of the night's work set. Every other face is
  an **easy picture**. Faces of unknown quality are easy.
- **Probe**: a picture of the night being sorted. Only probes get an output row; other items only help.

## Module

`src/vbb_reid_server/regroup.py`, pure functions: numpy only, no file or network access, no randomness, no
import from the rest of the package. `tests/test_regroup.py` builds its own synthetic vectors.

```python
@dataclass(frozen=True)
class Item:
    sid: str
    probe: bool
    weak: str | None
    anchor: str | None
    face: np.ndarray | None   # unit vector of one clear face, else None
    body: np.ndarray | None   # unit vector; the caller passes it for probe-day items only
    face_quality: float | None = None   # the face's quality (see Terms); None = unknown

@dataclass(frozen=True)
class Params:
    fmr: float = 0.005        # share of a group's impostors allowed above its floor
    min_core: int = 5         # members with a vector a group needs to act
    min_anchors: int = 3      # with this many anchors holding a face, the group starts from them alone
    min_impostors: int = 30   # impostor scores a group needs before it may name anything
    rounds: int = 10          # cap on cleaning rounds
    dup_share: float = 0.25   # see "Disputes"
    body_topk: int = 3
    hard_share: float = 0.25  # share of the least clear faces judged by bars of their own (F5); 0 = one class

@dataclass(frozen=True)
class Name:
    sid: str
    person: str | None        # None = the machine makes no claim
    via: str | None           # "anchor" | "face" | "body" | None
    score: float | None       # named: that person's score; ambiguous / not_nearest: the best CLEARED group's score;
                              # otherwise the picture's best score
    floor: float | None       # the floor that score was compared with
    reason: str               # see "Reasons"
    others: tuple[str, ...]   # ambiguous: every group whose floor it cleared; disputed: the names in dispute;
                              # not_nearest: the cleared group's names, then the nearest group's; a face that
                              # cleared no bar (below_floor / no_floor): the nearest group's names

@dataclass(frozen=True)
class GroupReport:
    person: str               # for a disputed set: the names joined with "|"
    n_weak: int               # the group's weak pictures (its candidates without the anchors)
    n_anchor: int             # anchors with a face
    n_members: int            # members after cleaning
    anchor_seeded: bool       # made from a person's anchors because no weak group was theirs
    active: bool
    floor: float | None       # the bar for easy pictures; None when it could not be set
    coverage: float | None    # share of its own members whose LOO score clears the bar of their own class
    disputed_with: tuple[str, ...]
    floor_hard: float | None = None   # the bar for hard pictures (faces only; bodies leave it None)

@dataclass(frozen=True)
class Result:
    names: dict[str, Name]            # probe sids only
    face_groups: dict[str, GroupReport]
    body_groups: dict[str, GroupReport]
    disputes: tuple[tuple[str, str, float, float], ...]   # (a, b, look(b into a), look(a into b)), a < b
    displaced: tuple[tuple[str, str], ...]                # (weak name, the confirmed person its group belongs to)
    refuted: tuple[str, ...]                              # names whose own group was dropped: confirmed elsewhere
    dissolved: tuple[tuple[str, tuple[str, ...]], ...]    # (the pile's weak names joined by "|", the people who
                                                          #  claimed it, sorted): see F3
    rounds: int

def regroup(items: Iterable[Item], params: Params = Params()) -> Result: ...
```

`Params.__post_init__` rejects `fmr` outside (0, 1), `dup_share` outside (0, 1], `hard_share` outside [0, 1),
`min_core` below 2 (a group of one cannot be scored without itself) and the other counts below 1. A person string
containing `|` raises `ValueError`: `|` joins the names of a disputed set.

## Faces

Work set: items whose `face` is usable (see "Input rules"). The order is **group first, name second**: groups come
from the names the pictures carry, confirmed pictures then say whose group each one is, and only then is each
group cleaned. (The first version let a person's confirmed pictures replace the group's average face and moved
pictures between names while cleaning. Checked on saved data, 2026-11-20: confirmed pictures are mostly
corrections of the board's mistakes, so the hardest pictures, and made loose groups; and pictures that fit nobody
drifted from group to group.)

**F1. Weak groups.** For each name `g`, `W_g` = the items with `weak == g` and no anchor. The weak group is active
when it has at least `min_core` pictures. Anchored items take no part in forming weak groups: an anchored item's
weak name is ignored everywhere.

**F2. Look-alike names.** Judged once, on the weak groups.

- `own_a` = median LOO score of `a`'s pictures against `a`.
- `look(b into a)` = share of `b`'s pictures whose score against `a` is at least `own_a`.

Two active weak groups **look alike** when both `look(b into a)` and `look(a into b)` are at least `dup_share`.
This fires when about half or more of one name's pictures show the other's face, the case a majority vote cannot
settle. Look-alike pairs are joined transitively into sets, and each set becomes one merged group (its pictures:
the union).

**F3. Confirmed pictures choose their group.** Every anchor is scored against every group from F2 (weak pictures
only). An anchor **votes** for the group it scores highest against, provided that score is at least that group's
`own` (it looks like a typical picture of the group); otherwise it does not vote. A person's **home** is the group
with the most of their votes, when those votes number at least `max(2, ceil(dup_share * anchors of that person))`
and strictly more than any other group got. Then every group gets its label:

- **Exactly one person's home**: the group is that person's. Each of its weak names that is not that person is
  listed in `Result.displaced` as `(weak name, person)`.
- **Home of two or more people**: the pile is mixed. Its average is a blend of several faces, which is why
  several people's confirmed pictures match it, and a blend cannot tell them apart. The pile is **dissolved**: no
  group is made from it, its weak pictures go to the pool, and `Result.dissolved` gets `(its weak names joined by
  "|", the people)`. Each of those people is then served by the own-group rule below.
- **Nobody's home, a merged set**: a **disputed set** of its names: it competes, but gives no name. Nobody knows
  which name owns the face.
- **Nobody's home, a single name**: it keeps that name, unless that name's person has a home elsewhere (a
  dissolved pile counts). Then the name is **refuted** here: the group is dropped, its pictures go to the pool, the
  name goes into `Result.refuted`.

  A known limit (adversarial review, 2026-11-21): a pile under a person's name keeps that name even when it shows
  someone else, as long as that person's confirmed pictures choose no pile at all; their confirmed pictures are then
  blended into it. The obvious fix, refuting a pile that not one of its person's confirmed pictures chose, was built
  and measured: 2 more right names on the six office-linaro nights, but 10 fewer on the simulated quiet nights with
  the top-up (114 → 104), because a topped-up pile is tight and its person's hard confirmed pictures never look
  typical of it. Reverted. Refuting whenever the votes fell short of a home was worse still: 32 fewer right names
  and 6 more wrong. Judging each confirmed picture's vote within its quality class (hard against the pile's hard
  pictures, as the bars do in F5) was tried too, 2026-11-21: alone it changed no home and no score at either site;
  with the refuting rule the quiet nights still fell (114 → 109). Not adopted.

**Own groups.** A person with anchors and no group labelled with exactly their name (a disputed set that contains
their name does not count) gets a group of their own when they have at least `min_anchors` anchors, or when their
anchors plus their own weak pictures (those of an inactive weak group) number at least `min_core`. That group's
members to start with are those anchors and those weak pictures; its `GroupReport.anchor_seeded` is true.

Anchors are members of the group labelled with exactly their person's name whenever that group exists, and are
never removed from it. They are never members of a disputed set. A person with no such group (too few anchors for
an own group) has anchors in no group; each still keeps its own name (F6).

(Checked on saved DHW data, 2026-11-20: the 6 and 22 confirmed pictures of two people both chose the pile of two
look-alike board names. Keeping that pile as a disputed set of the two people named 2 of 41 held-out confirmed
pictures right; dissolving it and giving each person their own group named 11 right, 0 wrong.)

**F4. Cleaning only removes.** A group's candidates are its weak pictures plus its anchors. Nothing ever moves to
another group here; that is F6's job. Repeat at most `rounds` times:

1. Centroid of every active group from its current members (first round: all its candidates).
2. First round: a non-anchor candidate is a member while its own group is the active group it scores highest
   against (LOO; a tie counts as its own).
3. Later rounds: set the floors (F5) from the current members. A non-anchor candidate is a member when its score
   clears its own group's floor and no other group's, all floors of the candidate's own class (easy or hard). If
   its own group has no floor, rule 2 applies to it.
4. A group left with fewer than `min_core` members and fewer than `min_anchors` anchors becomes inactive for good.
5. Stop when a later round changes no membership and drops no group.

Membership is judged over all candidates every round, so a picture removed early can come back once the group has
sharpened. The **pool** is every item that is no member at the end: no weak name, a weak name whose group is
inactive, refuted or dropped, or a candidate the cleaning removed. Pool items never shape a centroid or a floor.

**F5. Floors.** Fewer than two active groups: nothing is named (`reason="no_groups"`).

Quality classes: with `hard_share > 0`, take the face qualities known in the work set; a face whose quality is
strictly below their `hard_share` quantile (`numpy.quantile`, default method) is hard, every other face is easy.
With `hard_share == 0`, or no quality known, every face is easy. Each active group gets two floors, one per class:
a picture is only ever compared with the floors of its own class. (Why: hard pictures of different people look
more alike than easy ones, so a bar set mostly by easy pictures lets hard impostors through. Checked on saved DHW
data, 2026-11-20: with half the confirmations as anchors, wrong names fell from 7 to 3, right names 12 to 13;
office-linaro unchanged.)

For each active group `g` and class `c`, its impostor scores are the scores against `g` of every member of class
`c` of every other active group (anchors included). With fewer than `min_impostors` of them, the class falls back
to the impostors of both classes. `floor_g` of class `c` is the higher of two estimates of the score only a share
`fmr` of those impostors exceed:

- counted: the smallest impostor score such that at most a share `fmr` of the impostor scores are strictly above
  it. With fewer than `1 / fmr` scores this is simply the highest one, which a fresh impostor beats far more
  often than `fmr`;
- fitted: `mean + z * std` of the impostor scores (population std), with `z` the normal quantile of `1 - fmr`
  (`statistics.NormalDist().inv_cdf`). It covers the small-sample case; the counted one covers a tail heavier
  than normal.

With fewer than `min_impostors` impostor scores even after the fallback, the group has no floor in either class:
it names nothing (`reason="no_floor"` when it would have been the best group), and it cannot make a picture
ambiguous. It still counts as a picture's nearest group (F6), so a picture nearer to it than to the one group whose
bar it cleared is `not_nearest`: a group too thin for a bar can stop a wrong name, never give one.
`GroupReport.floor` is the easy floor, `floor_hard` the hard one, and `Name.floor` the floor of the picture's own
class.

A known limit (adversarial review, 2026-11-21): a confirmed picture is a member of its person's group for good,
so it is an impostor of every other group for good. Confirmed pictures are mostly corrections of the board, so
they tend to look like the group they were confused with, and below `1 / fmr` impostor scores the counted floor is
the single highest one. One such correction can therefore raise a thin group's bar for the night. It can only make
that group quieter, never give a wrong name; the quality classes soften it, since corrections are mostly hard
pictures and set only the hard bar.

**F6. Naming.** For every item in the work set:

- It has an anchor: `person=anchor`, `via="anchor"`, `reason="anchor"`. Always, whatever the scores say.
- Otherwise score it against every active group (LOO where it is a member) and collect the groups with
  `score > floor`, each floor of the picture's own class.
  - exactly one, and it is also the picture's **nearest** group (the active group it scores highest against; ties
    go to the lower group index):
    - a real person: named, `via="face"`, `reason="named"`;
    - a disputed set: no name, `reason="disputed"`, `others` = the set's names.
  - exactly one, but another active group scores higher (the picture is nearer to a group whose bar it did not
    clear): no name, `reason="not_nearest"`, `others` = the cleared group's names, then the nearest group's.
  - two or more: no name, `reason="ambiguous"`, `others` = those groups, best first.
  - none: no name, `reason="below_floor"` (or `no_floor`, above), `others` = the nearest group's names: where the
    face points, though it cleared no bar (B4 uses it).

  (The nearest-group condition is the usual open-set rule: accept the best match only when it clears that match's
  bar. Without it a small group with a low bar could claim a picture that looks more like someone else. Checked on
  saved data, 2026-11-21: it removed office-linaro's one wrong name and one wrong name at DHW in each setup tried,
  and cost one right name in one DHW setup.)

There is no further round: named pool pictures do not flow back into the centroids.

## Bodies

Clothes identify a person only within one day, so bodies use probe items only, and only after the faces.

**B1.** The body examples of person `g` are the body vectors of probe items named `g` by face or anchor. A
disputed set gets examples too (probes with `reason="disputed"` won by that set) and again never gives a name.
A group needs `min_core` examples to be active.

**B2.** Body score of a picture against `g`: mean of its `body_topk` highest dot products with `g`'s examples,
the picture itself left out. With fewer examples than `body_topk`, the mean of what there is.

**B3.** Floors as F5, from the other groups' examples, in one class (bodies have no quality classes;
`floor_hard` stays None).

**B4.** Named by body (owner, 2026-11-20: "if there's a face-lock, clothes shall not affect it. Clothes bar
facilitates pictures where there's no clear face lock"). Clothes judge every probe item with a body vector, no
anchor and **no face lock**:

- no usable face; or
- a usable face whose face answer is `below_floor`, `no_floor`, `no_groups`, `ambiguous` or `not_nearest`.

A face lock is a face answer of `named` (a person) or `disputed` (the face clearly showed one of a disputed set's
people); clothes never change either, and never an anchor. The rule is the same exactly-one rule against the clothes
bars and the same nearest-group condition, `via="body"` (a clothes answer that fails it is `not_nearest`), with
one limit: clothes may only choose one of the face's own candidates, the people in the face answer's `others`: for
an `ambiguous` or `not_nearest` face the groups it named there, for a `below_floor` or `no_floor` face the people of
its nearest group. A `no_groups` face points nowhere and is not limited. A clothes answer outside the candidates is
no answer, so clothes never name someone the face points away from. (Adversarial review, 2026-11-21: without the
limit for faces that fit nobody, a visitor with someone's clothes could get that person's name. Checked: no change
on the 117 office-linaro answers; 11 fewer clothes names over six nights, the ones pointing away from the face.)
When clothes give no name, a picture with a usable face keeps its face answer (its reason, score and floor); a
picture without one gets the clothes answer's reason.

(Checked on office-linaro 11-13..18, 2026-11-21: 132 pictures had a usable face but no face lock. Clothes named 44
of them; 2 of 2 among your answers were right, and the 33 the nightly job had also named were named alike. Every
ambiguous one that clothes named, they named among the face's own choices. The same check found the clothes bars
sound on the pictures they judge: faceless pictures of other people cleared a bar 2 times in 936, judged by where
the nightly job and the board agree, and 1 time in 136 among the answered ones, near the `fmr` of 0.005. At DHW
every clothes bar is above 1, so clothes name nobody there: staff look alike in clothes.)

## Reasons

`anchor`, `named`, `disputed`, `ambiguous`, `not_nearest`, `below_floor`, `no_floor`, `no_groups`, `no_vector` (a
probe with neither a usable face nor a usable body).

## Input rules

- A vector is usable when it is a 1-D float array, every value finite, and its length is above 0. It is scaled to
  length 1 on the way in. Anything else counts as absent: one bad row must not stop the night.
- All usable faces must share one dimension, all usable bodies another; a different size raises `ValueError`
  naming the sid.
- A face quality is usable when it is a finite real number (not a bool); anything else counts as unknown. It is
  read only for items whose face is usable.
- A repeated `sid` raises `ValueError`.
- The result does not depend on the order of `items`.

## Earlier nights (item 122)

**Why.** A quiet night leaves the sorter silent. A bar needs at least `min_impostors` pictures of *other* people,
and on office-linaro 2026-11-15 (62 pictures) one person was nearly alone: their bar could not be set, the faces
named nobody, and so the clothes had no examples either. 0 of the night's 17 answered pictures got a name; the
nightly job named all 17 right.

**Source (owner, 2026-11-20).** The earlier nights' names of the nightly job, which are more accurate than the
board's. Measured on office-linaro 11-13..18, every night added to the sorter: the job's names did as well as the
board's or better on every night (six-night totals: looking back 3 nights 89 vs 83 right, 7 nights 80 vs 76).

**Only to top up.** Adding every earlier picture made busy nights quieter (seven nights: 88 → 80 right). On
11-14 the bars dropped (one person's from 0.30 to 0.16), pictures clearing two bars rose from 0 to 15, and more
pictures also fell below every bar, which points to each person's average face drifting toward other days. So
earlier pictures go only to people who are thin tonight. Measured, 2026-11-21:
top-up to 30 gave 92 right / 1 wrong / 24 no name (today: 88 / 1 / 28), including 6 right on 11-15, and on 12
simulated quiet nights (busy nights cut to 62 pictures) 114 right / 0 wrong of 240 (today: 102 / 0). A top-up to
60 did worse on both.

Module `src/vbb_reid_server/regroup_inputs.py` (file reading only; no network). The caller passes everything it
pulled from the backend.

```python
@dataclass(frozen=True)
class EarlierPolicy:
    nights: int = 7                       # look back this many nights before the probe night; 0 = off
    top_up: int = 30                      # a person with fewer pictures than this tonight is topped up to it
    namers: tuple[str, ...] = ("decide", "regroup")  # whose label files may be read (E6)
    test_runs: bool = False               # E1: also read dry runs (DHW's test line, W5)
    min_checked: int = 5                  # E5: a person's earlier names need this many human-checked pictures...
    max_disagree: float = 0.2             # ...and are not used when more than this share of them disagree

@dataclass(frozen=True)
class NightReport:
    day: str
    status: str            # "used" | "no_run" | "no_labels" | "namer" | "no_faces" | "no_overlays"
    face_names: int        # face names read from the night's labels file (after E1-E3)
    added: int             # pictures this night contributed

@dataclass(frozen=True)
class EarlierReport:
    nights: tuple[NightReport, ...]       # newest first
    added: dict[str, int]                 # person -> pictures added
    checked: dict[str, tuple[int, int]]   # person -> (human-checked pictures, of which disagree), E5
    dropped: tuple[str, ...]              # people E5 kept out

def usable_faces(ft, ref_policy, vet) -> dict[str, tuple[np.ndarray, float]]
def night_run(runs_dir, facility, day) -> tuple[dict, Path] | None
def earlier_items(*, facility, day, runs_dir, face_table, key, roster, tonight, confirmed, touched,
                  capture_ts=None, skip_days=(), policy=EarlierPolicy(), ref_policy=RefPolicy(), vet=VetPolicy())
                  -> tuple[list[Item], EarlierReport]
```

- `usable_faces`: per sighting, the **first row of the table** (table order) that is a clear crop (`vetting.clear_crop`)
  and usable as a reference (`pipeline.ref_usable`): its vector and its raw feature norm (the face quality). The same
  rule makes every face the sorter sees, tonight's included.
- `night_run(runs_dir, facility, day, namers=None, test_runs=False)`: the night's real run:
  `run_<facility>_<day>_*.json` with `dry_run` exactly `false` (with `test_runs`, `true` counts too), and, when
  `namers` is given, a `namer` (absent = `"decide"`) among them; the latest `started_at` among those, and its
  labels file `labels_<same tag>.csv`. A record that cannot be read is skipped;
  none left = `None`. (The namer is checked before the latest run is chosen, so a later sorter run of a night never
  hides that night's decide run.)
- `earlier_items` inputs: `face_table(facility, day)` returns that night's face table (`Workspace.face_table`);
  `key(person_id)` folds an id with tonight's merge chain; `roster` holds tonight's roster keys; `tonight` = every
  item the caller already built (tonight's probes and the confirmed pictures of any day); `confirmed` = sid ->
  person key, or `None` for "not on the list", for every human-checked picture the caller knows (reviewed staff
  labels and check-page answers); `touched` = earlier pictures carrying any human overlay now (known, stranger or
  reset); `capture_ts` = sid -> capture time, for the newest-first order.

**Rules.**

- E1. **Nights**: the `nights` calendar days before `day`, newest first. Never `day` itself or later. A night
  without a real run, without its labels file, or whose face table is empty or cannot be read (missing, unreadable,
  or stored in an older cache format: `embed.StaleCache`) contributes nothing; its `NightReport.status` says why.
  A night in `skip_days` (the caller could not pull its current overlays) is not read at all: `"no_overlays"`.
  Keep `nights` well inside the face cache's retention (14 days, decision 14).
- E2. **Face names only**: rows with `state == "known"`, `via == "face"` and a person id. A clothes name is
  same-day evidence and says nothing about the face.
- E3. **Weak, never final**: the person is `key(person_id)` (tonight's merge chain), kept only when it is in
  `roster`. The picture is skipped when it is in `tonight`, in `confirmed` (a human checked it: confirmed pictures
  are anchors already, "not on the list" excludes it) or in `touched`, or has no usable face. Each kept picture
  becomes `Item(sid, probe=False, weak=person, anchor=None, face, body=None, face_quality)`, and the cleaning
  re-judges it every night like any weak picture.
- E4. **Top-up**: a person's pictures tonight = the items of `tonight` with a face that are either anchored to
  them (any day) or probes whose weak name is theirs. A person below `top_up` gets earlier pictures until they reach
  it: nights newest first, within a night by `capture_ts` newest first (sid order when missing), ties by sid. A
  person at or above `top_up` gets none.
- E5. **Drift check**: per person, over the face names of the nights read, `checked` = those pictures that are in
  `confirmed`, `disagree` = those whose confirmed person is not this person (`None` included). A person with
  `checked >= min_checked` and `disagree / checked > max_disagree` gets no earlier pictures and is listed in
  `dropped`. (office-linaro, 11-13..18: 59 checked, 0 disagree.)
- E6. **Provenance**: a labels file is read only when its run record's `namer` (absent = `"decide"`, every run so
  far) is in `namers`. Among a night's real runs, only those with an accepted namer are candidates, and the latest of
  them is used; a night whose real runs all have other namers has the status `"namer"`. Since plan decision 38 the
  sorter's own runs (`"namer": "regroup"`, W3) are read by default; another namer is refused.
- E7. **Deterministic**: the same files and inputs give the same items in the same order. A labels row that cannot
  be read is skipped, not fatal.

**Obligations for the caller (#119, when this is wired into `run`).** The module trusts what it is given, so:

- `touched` must come from a fresh pull of each earlier night (`BackendClient.pull_sightings`), because a labels
  file only knows the overlays as of its own night. **Fail closed**: a night whose pull failed goes in `skip_days`,
  never read with an empty `touched`, which would let pictures staff rejected come back as weak names.
- `confirmed` must hold every human check the run knows: reviewed staff labels and every check-page answer,
  "not on the list" included, and the correction list's verdicts too (a picture someone marked wrong must never
  come back under that name). Those verdicts are chosen for disagreeing, so they make the drift check (E5) stricter
  than a random sample would: it can only stop a top-up, never cause a wrong name. A picture answered "can't tell"
  is a human check as well: pass it in `touched`.
- `capture_ts` comes from the same pulls.
- Runs whose names come from the sorter write `"namer": "regroup"` into their run record (E6, W3).
- A face-model upgrade or a face-cache format change leaves the earlier nights without usable cached faces: the
  top-up then stops (`no_faces`) until those nights are embedded again in full (`run.embed_missing` re-embeds only
  the days holding references, and only those pictures). Check the night reports after an upgrade.

## Wiring into the nightly run (#119) and DHW test runs (#99)

Plan decision 38 (owner, 2026-11-21): the sorter replaces the old decision path in the nightly run ("we don't need 2
systems"), corrections of the board's names included; DHW gets nightly test runs "for me to judge". The backend
already accepts a machine name that corrects the board (`MachineLabelHelper.php:905`, relation `relabel`) and shows
it over the board's name (`IdentityResolver.php:49`), so no backend change is needed; rows stay inactive until
activated, as now.

- **W1. One namer.** `run` names with the sorter only; it no longer calls `decide_night`. `decide.py` stays for the
  offline tools (`eval`, `shadow`, the adjudication pages), which measure the old path.
- **W2. `src/vbb_reid_server/regroup_night.py`**, the inputs and the rows for one facility-night:

  ```python
  def regroup_night(ws, facility, day, *, label_cutoff_ms, ref_policy, vet, trusted, runs_dir,
                    overlays: Mapping[str, list[dict] | None], earlier=EarlierPolicy(), params=Params()) -> dict
      # -> {"labels": [MachineLabel], "sightings": [probe Sighting], "result": Result, "earlier": EarlierReport,
      #     "summary": dict}
  ```

  - Probes: every sighting of the facility-night. Face: `usable_faces` of that night's table (vector and norm as
    the quality). Body: the night's body vector (`Workspace.probe_inputs`).
  - **Anchor** of a picture, the human check that existed before `label_cutoff_ms`: a check-page or
    correction-list answer (`trusted`, saved before the cutoff per `Trusted.info`) naming a person on the roster;
    otherwise a staff `known` label with the reviewed mark, `corrected_ts` before the cutoff, person on the roster.
    An answer of "not on the list" gives no anchor, and that picture gets no machine row.
  - **Weak name** of a probe without an anchor: a staff `known` label without the reviewed mark (person on the
    roster), else the board's `known` name (on the roster), else none.
  - **Earlier confirmed pictures**: sightings of the facility before `day`, within `ref_policy.window_days`, with an
    anchor by the same rule (the export's references, and answers found through `Trusted.info`); face from their own
    night's table; not probes, no body.
  - **Earlier nights**: `earlier_items` with `confirmed` = every human check (all answers, "not on the list"
    included, and reviewed staff labels), `touched` = pictures with any overlay in `overlays[night]` plus pictures
    answered "can't tell", `skip_days` = nights whose `overlays` entry is `None`, `capture_ts` from `overlays`.
  - **Rows**: one per probe whose answer names a person (via face or body), never one for an unnamed probe:
    `MachineLabel(state="known", relation=` `"agree"` when the board's `known` name is that person, `"relabel"` when
    it is someone else, else `"promote"` `, person_key=key, person_id=` the facility's id for that key `, via=` the
    answer's via `, confidence=score, margin=None, face_sim=` score when via face `, body_sim=` score when via body
    `, evidence={"reason", "floor", "others"})`. A probe with a human overlay gets its row in the file like any other;
    the run never pushes it (unchanged rule of `run.py`).
- **W3. `run_night`.** Pulls each earlier night of the look-back (`BackendClient.pull_sightings`; a failed pull is
  `None`), calls `regroup_night`, and pushes relabels as names: the forcing of `allow_relabel` off goes. It writes
  the same files as now (labels CSV with the same columns, `label_check` empty; staff CSV; run JSON). The run JSON
  gets `"namer": "regroup"` and a `"regroup"` summary: counts by reason and via; per face group its members, floors
  and whether active; disputes, dissolved, displaced, refuted; the earlier-nights report. Guards kept: rejected rows,
  promotion share, crop failures, empty run. The thin-references guard goes (it measured the old path's references).
- **W4. Check page.** `verify-build` draws from every `known` row whose relation is `promote` or `relabel` (the
  machine's names that differ from the board) with no human label, so it measures the corrections too. With
  `--test-runs` it reads the newest dry run of the night (DHW).
- **W5. DHW test line.** `scripts/nightly_run.sh` with `VBB_RUN_TEST=1` passes `--dry-run --earlier-test-runs` to
  `run`, then still draws the check page (`verify-build --test-runs`) and skips the correction list. DHW settings:
  facility 5, root `/vmdata/vbb_eval/vbb_run_f5`, token `~/.config/vbb_reid_server/worker_f5.token`, the same point
  and models files as office-linaro; 01:30 Taiwan; `refs.json` built once with `verify-refs` for facility 5; its page
  server on port 8771, loopback only. Installing the line and the page server is the owner's call.
- **W6. `run` options.** `--earlier-nights N` (default 7, 0 = off) and `--earlier-test-runs`.
- **W7. Reference pictures for a check page from the backend** (DHW has no offline export and no one-at-a-time
  answers). `verify-refs --from-backend` (with `--backend`, `--token-file`, `--facility`, `--day`, `--emb`,
  `--face-model`, `--per-person`, `--out`) builds `refs.json` from `export_from_backend` over the reference window:
  every staff `known` label with the reviewed mark counts as an answer for its person, with the same checks and the
  same per-person draw as `pick_references` (one face in the uploaded picture, an uploaded image, a roster person,
  one picture per day in turn). `--answers` is then not needed. The backend sends no names, so the page shows each
  person's id with their reference pictures. (Checked 2026-11-21: the existing builder reads only the offline SQL
  export and answer files, so DHW's page could not be built.)
- **W8. Board-named references for people nobody confirmed** (owner, 2026-11-21: "for other people who don't have
  confirmed pics, can we just pick high quality sightings as reference? Probably mark them as unconfirmed, say, a
  colored box"). With `--from-backend`, a roster person who gets no confirmed reference picture gets up to
  `per_person` pictures under the **board's** name instead, never the sorter's (the page measures the sorter, and its
  own names as references would let it grade itself):
  - candidates: the facility's sightings of the last `--board-days` days (default 14, the face cache's retention),
    pulled with `pull_sightings`, whose board name is that person, with no human overlay of any kind, an uploaded
    picture, and exactly one face (the same one-face check);
  - ranked by face quality (the raw feature norm of the cached face), best first; one picture per day in turn, as
    for confirmed references;
  - each such picture carries `"confirmed": false` in `refs.json` (confirmed pictures carry `true`; a picture without
    the field counts as confirmed, as every existing file does);
  - the check page and the correction list's batch view show them in a colored frame with the note "board's name,
    not confirmed".
  - **Never an answer**: `vetting.load_trusted` reads only confirmed pictures of `refs.json` as answers. A picture
    with `"confirmed": false` must never become a human check, an anchor or ground truth.

## What this does not do yet

- Suggesting a new person from a tight group of unnamed pictures.
- Using "two people in one frame are different people".
- Earlier nights for clothes (clothes change from day to day).

## Evaluation before wiring

Read-only, on vectors already cached on gpu-ubun. Counts only; no names, picture ids or links in any output.

1. **DHW, 2026-11-17 + 2026-11-19, no anchors**: the owner's design with zero human input. Scored on the 81
   pictures staff corrected on 11-19 (all board mistakes): right / wrong / no name, faces alone and with bodies,
   for `fmr` 0.01, 0.005 and 0.002.
2. **DHW with anchors**: the staff-confirmed pictures split in two halves by time; one half as anchors, scored on
   the other, then swapped.
3. **office-linaro, nights 11-13 to 11-18**: each night sorted with that night's check-page answers held out,
   scored on those answers, next to what the production run named on the same pictures.

The design is wired into `run` only if (1) names more right than wrong by a wide margin, (2) is sharper than (1),
and (3) is not worse than production on office-linaro.
