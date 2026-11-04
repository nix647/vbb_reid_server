# vbb-reid-server

Server-side re-verification of VBB person sightings ("machine labels"), Phase A:
offline evaluation, shadow runs and adjudication on a GPU host. Nothing here
writes to any backend. Plan: `PLAN_nightly_machine_labels.md` (owner-approved 2026-11-03).

Separate from the board package `vbb-reid` on purpose: that one ships to boards
with `--no-deps`, an explicit package list and RKNN extractors, and boards rsync
its whole tree. Nothing here imports `ReID.*`.

## What it does

```
export CSVs ──┐
CDN crops ────┼─► embed ─► embedding cache ─► eval ─────► curve.csv, report.md
LAN crops ────┘   (GPU)    (per model)        shadow ───► labels_*.csv, run_*.json
                                              adjudicate ► sample.csv → page.html → answers.csv → score
```

- **References** are human-labeled `known` crops only (no self-training).
- **Decision** (`decide.py`): mixed-track check over a sighting's crops → own face
  (similarity floor + margin over the second person + quality floor + min references,
  closed roster) → same-day body clustering seeded by strong face decisions, with
  cannot-link for simultaneous tracks on one board and no cluster ever holding two
  named people → abstain. The machine never says `stranger`/`uncertain`; a disagreement
  with a board `known` is a `flag/relabel` in v1.
- **Evaluation** (`evaluate.py`): per facility and day; `temporal` references (labels that
  existed at the nightly run) and `lodo` (leave one day out, for comparing models);
  leave-one-person-out false accepts. The gate is on promotions (what v1 shows): a minimum
  number of labels, precision above the target on BOTH a day bootstrap (labels come in bulk
  sessions) and an exact binomial lower bound, and a measured false-accept upper bound. The
  chosen point is scored on the data that chose it (optimistic); the independent check is
  the adjudication set.
- **Adjudication** (`adjudicate.py`): stratified sample (board state × face/no-face, spread
  over days, never already-labeled sightings) and a self-contained HTML page with a photo
  strip per person, no scores, answers kept in the browser and exported as CSV.

## Runbook on gpu-ubun

First time? Follow [`SETUP_gpu-ubun.md`](SETUP_gpu-ubun.md) step by step; this section is the short form.

```bash
# 0. install (Python >= 3.10), into a venv
python3 -m venv ~/venvs/vbbrs && . ~/venvs/vbbrs/bin/activate
pip install -e '.[onnx,hf,dev]'
pytest -q

W=/vmdata/vbb_eval            # 77 GB free there; / has 54 GB
mkdir -p $W/{export,crops,cdn,emb,models,out,adj}

# 1. the export: unpack machine_eval_<date>.tgz into $W/export/
#    and write $W/aliases.csv (person_id,human,name) — stays on this host, never in git.

# 2. second crops from the boards (read-only, nice/ionice on the board); then daily by cron
scripts/rsync_board_crops.sh $W/crops 45=linaro@<board-45-ip>

# 3. embed one candidate set (repeat per candidate config)
vbb-reid-server embed --export $W/export --aliases $W/aliases.csv --emb $W/emb \
  --models configs/<candidate>.json --facility 2 3 --lan-crops $W/crops --cdn-cache $W/cdn

# 4. evaluate (CPU only; reads the cache)
vbb-reid-server eval --export $W/export --aliases $W/aliases.csv --emb $W/emb \
  --face-model w600k --body-model siglip2 --facility 3 2 --out $W/out/<candidate>

# 5. adjudication set (owner labels it in a browser on this host)
vbb-reid-server adjudicate-sample --export $W/export --aliases $W/aliases.csv --facility 3 --n 300 --out $W/adj/sample_f3.csv
vbb-reid-server adjudicate-page   --export $W/export --aliases $W/aliases.csv --facility 3 \
  --sample $W/adj/sample_f3.csv --lan-crops $W/crops --out $W/adj/page_f3.html

# 6. shadow labels for the chosen operating point (eval wrote point_<facility>_<mode>.json; shadow
#    uses only labels that existed at each night's run), then score against the answers
vbb-reid-server shadow --export $W/export --aliases $W/aliases.csv --emb $W/emb --facility 3 \
  --face-model w600k --body-model siglip2 --point $W/out/<candidate>/point_3_temporal.json --out $W/out/shadow
vbb-reid-server adjudicate-score --export $W/export --aliases $W/aliases.csv \
  --answers ~/Downloads/adjudication_sample_f3.csv --labels $W/out/shadow/labels_3_*.csv

# 7. retention (decision 14): crops cached from the CDN and embeddings, 14 days
vbb-reid-server prune --emb $W/emb --cdn-cache $W/cdn --days 14
```

## Candidate models

| Role | Ready with this code | Needs an export step first |
|---|---|---|
| Face detect | SCRFD-10G `det_10g.onnx` (InsightFace `buffalo_l`) | — |
| Face embed | board AdaFace IR-50 ONNX twin (baseline; layout/colour: probe), ArcFace `w600k_r50.onnx` (`buffalo_l`), `glintr100.onnx` (`antelopev2`) | AdaFace IR-101 WebFace12M, AdaFace ViT + KP-RPE (CVLFace): PyTorch → ONNX |
| Body embed | board ViT-S/16+ICS ONNX twin (baseline), SigLIP2 and DINOv3 via `hf_vision` | CLIP-ReID, SOLIDER: PyTorch → ONNX |

Weights licences: InsightFace and WebFace-trained weights are non-commercial research
only (plan T18). Evaluate; do not ship before the licence question is settled.

## Not verified yet

- The SCRFD decoder and the ONNX adapters have not run against real weights (no GPU or
  weights in the authoring environment). First run on gpu-ubun: `embed` one day and check
  `faces` > 0 and plausible `qf` values before a full run.
- `qf`'s pose term is approximated from landmarks; its thresholds are re-measured here.
