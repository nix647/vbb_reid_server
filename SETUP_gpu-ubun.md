# First setup and first run on gpu-ubun

Step by step, each with a check. Stop at the first check that fails and send its output.
Machine facts (2026-11-03): RTX 4070 Ti SUPER 16 GB, driver 580.178 / CUDA 13.0, Ubuntu,
user `dosomi`. Board 45 (facility 3) = `linaro@192.168.0.115`.

Layout used below:

| What | Where | Why |
|---|---|---|
| Code + Python environment | `~/vbb_reid_server`, `~/venvs/vbbrs` (on `/`) | small; safe on any filesystem |
| Data, crops, caches, weights, HF models | `/vmdata/vbb_eval/` | grows to tens of GB |

## 1. Check /vmdata's filesystem

```bash
df -T /vmdata
```

- `ext4` or `xfs`: fine for everything.
- `ntfs`, `fuseblk` or `exfat`: still fine for the data folder used here (plain files only); keep the
  Python environment on `/` as below.

Leave room for the Windows VMs: stop adding data when `/vmdata` is above ~85 % used.

## 2. Code and Python environment

```bash
sudo apt-get install -y python3-venv git
```

```bash
cd ~ && git clone -b claude/wonderful-hawking-vpm3va https://github.com/nix647/vbb_reid_server
```

```bash
python3 -m venv ~/venvs/vbbrs && . ~/venvs/vbbrs/bin/activate && pip install -U pip
```

```bash
cd ~/vbb_reid_server && pip install -e '.[onnx,hf,dev]'
```

```bash
echo 'export HF_HOME=/vmdata/vbb_eval/hf' >> ~/.bashrc && export HF_HOME=/vmdata/vbb_eval/hf
```

**Check:**

```bash
cd ~/vbb_reid_server && pytest -q && vbb-reid-server doctor
```

Expected: `74 passed` (0.1.6); `onnxruntime ... providers [...'CUDAExecutionProvider'...]`;
`torch ... cuda available=True, device NVIDIA GeForce RTX 4070 Ti SUPER`.
If `CUDAExecutionProvider` is missing, the CPU build may have been installed next to the GPU build
(0.1.0/0.1.1's `dev` extra pulled it in; both provide the same module). Fix and check again:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu && pip install 'onnxruntime-gpu[cuda,cudnn]' && vbb-reid-server doctor
```

### Proxy: keep the CDN direct

If the box has an HTTP(S) proxy configured (e.g. Clash), add the crop CDN to `no_proxy` before
any `embed`: otherwise downloads are much slower and your customers' face crops pass through
the proxy.

```bash
echo 'export no_proxy="$no_proxy,upvbbv.yuexingren.cn" NO_PROXY="$NO_PROXY,upvbbv.yuexingren.cn"' >> ~/.bashrc && . ~/.bashrc && env | grep -i no_proxy
```

## 3. Data folders, the export, the alias file

```bash
mkdir -p /vmdata/vbb_eval/{export,crops,cdn,emb,models,out,adj,hf}
```

Copy `machine_eval_20261103.tgz` to gpu-ubun, then:

```bash
tar -xzf machine_eval_20261103.tgz -C /vmdata/vbb_eval/export --strip-components=1
```

The alias file maps each person's two ids (facility 2 and facility 3) to one identity. It never goes
into git. Add a third column `name` if you want names on the labeling page (as in the chat of 2026-11-03).

```bash
cat > /vmdata/vbb_eval/aliases.csv <<'EOF'
person_id,human
p_19f9758fa4ef733bab1358f,H1
p_1a14e526b5c11ae4f4ed9f4,H1
p_19fb1f99744ae667034b8a3,H2
p_1a14e52c1d158847402b6c7,H2
p_19f975b21641607c5dd976f,H3
p_1a15dc5379a3b68740e31b6,H3
p_19f89915663867d398239d3,H4
p_1a14e4bc2a5f507ecd08d3d,H4
EOF
```

**Check:**

```bash
vbb-reid-server summary --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv
```

Expected: facility 2 with 9673 sightings, facility 3 with 17390, roster `H1`–`H4` plus the test ids.

## 4. Second crops from board 45 (read-only)

```bash
cd ~/vbb_reid_server && scripts/rsync_board_crops.sh /vmdata/vbb_eval/crops 45=linaro@192.168.0.115
```

It needs key-based SSH (`ssh linaro@192.168.0.115 true` must work without a password prompt).
Then make it daily (boards keep 7 days):

```bash
(crontab -l 2>/dev/null; echo "10 3 * * * $HOME/vbb_reid_server/scripts/rsync_board_crops.sh /vmdata/vbb_eval/crops 45=linaro@192.168.0.115 >> /vmdata/vbb_eval/rsync.log 2>&1") | crontab -
```

**Check:** the last line printed says `done did=45: N crops on disk` with N in the thousands.

## 5. The first candidate's weights (InsightFace buffalo_l: SCRFD-10G + ArcFace r50)

Research-only licence: evaluate, do not ship.

```bash
cd /vmdata/vbb_eval/models && wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && unzip -o buffalo_l.zip -d buffalo_l && ls -la buffalo_l
```

Expected: `det_10g.onnx` and `w600k_r50.onnx` among the files (the zip may unpack into
`buffalo_l/` or `buffalo_l/buffalo_l/`; fix the paths in the next step if so).

Write the model config (face only first; the body model comes after the face path is proven):

```bash
cat > /vmdata/vbb_eval/models/buffalo_l_face.json <<'EOF'
{
  "name": "buffalo_l face",
  "face_detector": {"type": "scrfd", "path": "/vmdata/vbb_eval/models/buffalo_l/det_10g.onnx", "input_size": 640, "score_thresh": 0.5},
  "face": {"type": "onnx_face", "path": "/vmdata/vbb_eval/models/buffalo_l/w600k_r50.onnx", "color": "RGB", "flip": true}
}
EOF
```

**Check:**

```bash
vbb-reid-server doctor --models /vmdata/vbb_eval/models/buffalo_l_face.json
```

Expected: `face_detector: det_10g.onnx:sha256:... [CUDAExecutionProvider]` and the same for `face`.

## 6. Embed ONE day first, and send me the output

```bash
vbb-reid-server embed --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
  --emb /vmdata/vbb_eval/emb --models /vmdata/vbb_eval/models/buffalo_l_face.json \
  --facility 3 --days 2026-10-16 --lan-crops /vmdata/vbb_eval/crops --cdn-cache /vmdata/vbb_eval/cdn
```

**Send me** every line starting with `[embed]`. What I look for: `faces` well above zero (facility 3
has ~58 % face crops), few `unreadable`, `cdn_failed` near zero, and the time per day.

Do not run the full embed until I have checked that output: if the detector decoding or the colour
order is wrong, every embedding would have to be recomputed.

## 6b. Settle face quality before the full run (0.1.2)

0.1.0 penalised the alignment residual itself and zeroed most non-frontal faces (found on
2026-10-16: 54 % of faces had residual >= 6). 0.1.1 stores the raw measurements in the cache
and computes `qf` when it is read, so the quality policy can change without re-embedding.

```bash
cd ~/vbb_reid_server && git pull && pip install -e '.[onnx,hf,dev]' && pytest -q
```

Re-embed the test day once (crops are cached, so only the GPU work repeats). The cache format
changed twice since your first run (raw quality measures; preprocessing settings now part of the
model id), so this creates a new cache folder; the old one can be deleted:

```bash
vbb-reid-server embed --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
  --emb /vmdata/vbb_eval/emb --models /vmdata/vbb_eval/models/buffalo_l_face.json \
  --facility 3 --days 2026-10-16 --lan-crops /vmdata/vbb_eval/crops --cdn-cache /vmdata/vbb_eval/cdn --refresh
```

Check that batching does not change embeddings (the `Expected shape {1,512}` warning):

```bash
vbb-reid-server doctor --models /vmdata/vbb_eval/models/buffalo_l_face.json
```

Expected: `face batched vs one-by-one: max |diff|` below 1e-3 and `min cosine` at least 0.9999. The
GPU picks different kernels per batch size, so batched and one-by-one differ slightly but each is
exactly repeatable (measured 2026-11-04 on 128 real faces: max 1.7e-4, pair similarity shift at
most 2.7e-4 against a 0.25 threshold). The pipeline always embeds in batches.

Measure which quality measures matter, on your labels:

```bash
vbb-reid-server quality-report --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
  --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5 --facility 3 --days 2026-10-16 \
  --out /vmdata/vbb_eval/out/quality_2026-10-16.csv
```

**Send me** the printed table (and the `[embed]` and `doctor` lines). For each measure it shows,
per band, the genuine and impostor median similarity and TAR (share of same-person pairs above
the 1-in-1000 impostor score). From it we set the quality policy (a small JSON passed as
`--quality`) and the cutoffs the evaluation sweeps. Then the full embed.

## 6c. Full embed, then choose the quality policy from data (0.1.2)

The embed does NOT depend on the quality policy (qf is computed when the cache is read), so it
runs first. Keep the CDN off the proxy (step 2); the 10-16 test day showed that almost all of the
time is downloading (388 s with downloads, 8.4 s from the cache).

```bash
vbb-reid-server embed --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
  --emb /vmdata/vbb_eval/emb --models /vmdata/vbb_eval/models/buffalo_l_face.json \
  --facility 3 2 --lan-crops /vmdata/vbb_eval/crops --cdn-cache /vmdata/vbb_eval/cdn 2>&1 | tee /vmdata/vbb_eval/out/embed_full.log
```

Then compare the two policies in `configs/quality/` on ALL labeled days (more people, less H1):

```bash
for P in A_default B_tuned_2026-11-04; do for F in 3 2; do
  vbb-reid-server quality-report --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
    --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5 --facility $F \
    --quality configs/quality/$P.json --out /vmdata/vbb_eval/out/quality_${P}_f$F.csv | tee /vmdata/vbb_eval/out/quality_${P}_f$F.txt
done; done
```

And the face-only evaluation under each policy (CPU only; reads the cache):

```bash
for P in A_default B_tuned_2026-11-04; do
  vbb-reid-server eval --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
    --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5 --facility 3 2 --grid configs/grids/face_v1.json \
    --quality configs/quality/$P.json --setup "buffalo_l face, policy $P" --out /vmdata/vbb_eval/out/eval_$P 2>&1 | tee /vmdata/vbb_eval/out/eval_$P.log
done
```

**Send me**: the `[embed]` summary lines of the full run (first and last few days, and any with
`unreadable` or `cdn_failed` above 0); the `qf` section of each quality-report; and the `[eval]`
summary blocks (both facilities, both reference modes, both policies). The `curve.csv` files stay
on the box; I will ask for specific rows if needed.

## 6d. Non-faces, two-person crops, and the independent check (0.1.3)

0.1.3 stores three more measures per face (raw feature norm, number of faces in the crop, size
and score of the second face). The cache schema changed, so `embed` re-embeds every day once —
GPU only, the crops are on disk:

```bash
cd ~/vbb_reid_server && git pull && pip install -e '.[onnx,hf,dev]' --no-deps && pytest -q
vbb-reid-server embed --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
  --emb /vmdata/vbb_eval/emb --models /vmdata/vbb_eval/models/buffalo_l_face.json \
  --facility 3 2 --lan-crops /vmdata/vbb_eval/crops --cdn-cache /vmdata/vbb_eval/cdn --no-fetch 2>&1 | tail -3
```

1. Look at the reviewed faces the way the decision sees them (`inspect`; use your `point_3_temporal.json`):

```bash
vbb-reid-server inspect --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
  --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5 --facility 3 --sightings a2edee5618ff \
  --point /vmdata/vbb_eval/out/eval_A_default/point_3_temporal.json
vbb-reid-server inspect --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
  --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5 --facility 2 --sightings 81e8e70dc82f,83c78b81f9ed,07e911a53a04 \
  --point /vmdata/vbb_eval/out/eval_A_default/point_3_temporal.json --refs-mode lodo
```

2. Where do the new measures put non-faces and second faces (policy A is enough here):

```bash
vbb-reid-server quality-report --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
  --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5 --facility 3 --quality configs/quality/A_default.json \
  --out /vmdata/vbb_eval/out/quality_A_f3_v013.csv | sed -n '/norm/,$p'
```

3. Evaluate with the two-faces sweep (the gate now prints PASS / FAIL / NOT ENOUGH EVIDENCE YET):

```bash
for P in A_default B_tuned_2026-11-04; do
  vbb-reid-server eval --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
    --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5 --facility 3 2 --grid configs/grids/face_v2.json \
    --quality configs/quality/$P.json --setup "buffalo_l face v2, policy $P" --out /vmdata/vbb_eval/out/eval_v2_$P 2>&1 \
    | grep -A12 '^\[eval\] facility'
done
```

4. The independent check: shadow facility 3 at each policy's temporal operating point, then score
it against your 300 adjudication answers (copy the CSV to `/vmdata/vbb_eval/adj/` first):

```bash
for P in A_default B_tuned_2026-11-04; do
  vbb-reid-server shadow --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
    --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5 --facility 3 \
    --point /vmdata/vbb_eval/out/eval_v2_$P/point_3_temporal.json --out /vmdata/vbb_eval/out/shadow_$P
  vbb-reid-server adjudicate-score --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
    --answers /vmdata/vbb_eval/adj/adjudication_adjudication_f3_set1_sample.csv \
    --labels /vmdata/vbb_eval/out/shadow_$P/labels_3_*.csv | tee /vmdata/vbb_eval/out/adjscore_$P.json
done
```

**Send me**: the `inspect` output; the norm / second_ratio / n_faces sections; the `[eval]` blocks with
their VERDICT lines; and both `adjudicate-score` results. Keep using the as-exported export for
these runs (your reviewed copies were a good diagnostic; the guards above are meant to make them
unnecessary).

## 6e. The corrected sweep, the margin question, and a targeted check of low-norm names (0.1.4)

0.1.3's `eval` cached face scores under a key that missed `two_face_ratio` and `min_face_norm`, so
sweeps of those two settings reused unguarded scores (found by the owner on 2026-11-05). 0.1.4
makes face scoring read only a settings object that is itself the cache key. No re-embed needed.
Policy A from here on (A and B did not differ on any check; A is the closer port of the board).

```bash
cd ~/vbb_reid_server && git pull && pip install -e '.[onnx,hf,dev]' --no-deps && pytest -q
O=/vmdata/vbb_eval/out/v014; mkdir -p $O
COMMON="--export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5"
```

1. Probe norm floor x two-faces rule, with and without a reference norm floor:

```bash
for R in 0 14; do
  vbb-reid-server eval $COMMON --facility 3 2 --grid configs/grids/face_v3.json \
    --quality configs/quality/A_default.json --ref-min-norm $R --setup "A v3 refnorm $R" --out $O/eval_refnorm$R 2>&1 \
    | grep -A12 '^\[eval\] facility'
done
python3 - $O <<'PY'
import csv, glob, sys
for f in sorted(glob.glob(sys.argv[1] + "/eval_refnorm*/curve.csv")):
    for r in csv.DictReader(open(f)):
        if r["facility"] == "3" and r["face_sim"] == "0.4" and r["face_margin"] == "0.05":
            print(f.split("/")[-2], r["refs_mode"], "two_face", r["two_face_ratio"] or "off", "norm", r["min_face_norm"],
                  "| labeled", r["promote_labeled"], "wrong", r["wrong"], "flagged", r["flagged"], "cp_lo", r["promote_cp_lo"],
                  "cov", r["promote_coverage"], "fa_hi", r["loo_fa_cp_hi"], "unlabeled", r["unlabeled_promotions"])
PY
```

2. Is B's reference sensitivity really B, or the margin it picked (0.15)? Policy A at B's lodo setting
on your two reviewed copies of the export:

```bash
echo '{"face_sim":[0.35],"face_margin":[0.15],"face_q_min":[0.0],"body_enabled":[false]}' > $O/one.json
for X in fix fix_bad; do
  vbb-reid-server eval --export <your $X export copy> --aliases /vmdata/vbb_eval/aliases.csv --emb /vmdata/vbb_eval/emb \
    --face-model RGB:m127.5 --facility 3 --refs-mode lodo --grid $O/one.json --quality configs/quality/A_default.json \
    --out $O/margin_$X >/dev/null
  python3 -c "import csv,sys; r=list(csv.DictReader(open(sys.argv[1])))[-1]; print(sys.argv[2], 'labeled', r['promote_labeled'], 'wrong', r['wrong'], 'unlabeled_promotions', r['unlabeled_promotions'])" $O/margin_$X/curve.csv $X
done
```

3. Look at the five two-person answers and the unreviewed lodo error:

```bash
vbb-reid-server inspect $COMMON --facility 3 --point $O/eval_refnorm0/point_3_temporal.json \
  --sightings cdcc40d83bda,9354311a2579,885436f94353,620f2f2bbf74,4ff683ec05a8,36aed033c5b4
```

4. Shadow at the corrected temporal point, score it, and draw a TARGETED set: 100 sightings the
machine named from faces with norm below 16 (the band where your hair and back-of-head boxes sit).
Label it like the first set; its precision says whether low-norm names can be trusted.

```bash
vbb-reid-server shadow $COMMON --facility 3 --point $O/eval_refnorm0/point_3_temporal.json --out $O/shadow
vbb-reid-server adjudicate-score --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
  --answers /vmdata/vbb_eval/adj/adjudication_adjudication_f3_set1_sample.csv --labels $O/shadow/labels_3_*.csv | tee $O/adjscore.json
vbb-reid-server adjudicate-sample $COMMON --facility 3 --n 100 --named-by $O/shadow/labels_3_*.csv --norm-below 16 \
  --exclude /vmdata/vbb_eval/adj/adjudication_adjudication_f3_set1_sample.csv --out /vmdata/vbb_eval/adj/sample_f3_lownorm.csv
vbb-reid-server adjudicate-page --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv --facility 3 \
  --sample /vmdata/vbb_eval/adj/sample_f3_lownorm.csv --lan-crops /vmdata/vbb_eval/crops --out /vmdata/vbb_eval/adj/page_f3_lownorm.html
```

**Send me**: the VERDICT blocks and the printed rows of step 1, the two lines of step 2, the
`inspect` output, `adjscore.json`, and — once labeled — the low-norm answers CSV.

## 6f. Why H1 attracts junk, the pinned guards, and new names only (0.1.5)

The low-norm set (2026-11-05) showed new names below norm 14 are wrong about half the time, 18 of
19 wrong names are H1 (300 references; everyone else <= 55), and the gate's PASS does not cover
this band because the labeled set is mostly clean faces. Hypothesis: for a junk face, the best of N
chance similarities grows with N, so a person with many references wins on noise; top-3 keeps that
bias. A reference cap per person tests it directly. No re-embed needed.

```bash
cd ~/vbb_reid_server && git pull && pip install -e '.[onnx,hf,dev]' --no-deps && pytest -q
O=/vmdata/vbb_eval/out/v015; mkdir -p $O; A=/vmdata/vbb_eval/adj
COMMON="--export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv --emb /vmdata/vbb_eval/emb --face-model RGB:m127.5"
```

1. The H1 test, on answers you already have: shadow WITHOUT the norm floor at three reference caps,
scored on the low-norm set and the first 300. If the cap is the cause, the 19 wrong names should
drop sharply at 50 / 25 without a floor.

```bash
echo '{"face_sim":0.4,"face_margin":0.05,"face_q_min":0.0,"body_enabled":false}' > $O/diag.json
for R in 300 50 25; do
  vbb-reid-server shadow $COMMON --facility 3 --thresholds $O/diag.json --max-refs $R --out $O/diag_refs$R >/dev/null
  for SET in adjudication_sample_f3_lownorm adjudication_adjudication_f3_set1_sample; do
    echo "max-refs $R, $SET:"; vbb-reid-server adjudicate-score --export /vmdata/vbb_eval/export \
      --aliases /vmdata/vbb_eval/aliases.csv --answers $A/$SET.csv --labels $O/diag_refs$R/labels_3_*.csv \
      | python3 -c "import json,sys; d=json.load(sys.stdin); print({k: d[k] for k in ('correct','wrong','abstain','stranger_false_accepts','precision_cp_lo')}, 'wrong:', d['wrong_ids'])"
  done
done
```

2. Guards pinned (probe norm floor 14, two-faces 0.25), similarity 0.40-0.50, per reference cap
(pick R from step 1; run 300 too as the baseline):

```bash
for R in 300 <R>; do
  vbb-reid-server eval $COMMON --facility 3 --grid configs/grids/face_v4.json --quality configs/quality/A_default.json \
    --max-refs $R --fix min_face_norm=14 --fix two_face_ratio=0.25 --setup "A v4 refs $R" --out $O/eval_v4_refs$R 2>&1 \
    | grep -A6 '^\[eval\] facility'
done
```

3. Shadow at the chosen point, score both answer sets, then the set that actually measures what the
machine would show: 200 NEW names (promote), uniform over all norms, none from earlier sets. The page
carries both LAN crops next to it, so the folder can be copied to your Mac.

```bash
vbb-reid-server shadow $COMMON --facility 3 --point $O/eval_v4_refs<R>/point_3_temporal.json --out $O/shadow
for SET in adjudication_sample_f3_lownorm adjudication_adjudication_f3_set1_sample; do
  vbb-reid-server adjudicate-score --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv \
    --answers $A/$SET.csv --labels $O/shadow/labels_3_*.csv > $O/score_$SET.json; done
vbb-reid-server adjudicate-sample --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv --facility 3 \
  --n 200 --named-by $O/shadow/labels_3_*.csv --relation promote --strata none \
  --exclude $A/adjudication_adjudication_f3_set1_sample.csv $A/sample_f3_lownorm.csv --out $A/sample_f3_newnames.csv
vbb-reid-server adjudicate-page --export /vmdata/vbb_eval/export --aliases /vmdata/vbb_eval/aliases.csv --facility 3 \
  --sample $A/sample_f3_newnames.csv --lan-crops /vmdata/vbb_eval/crops --copy-crops --out $A/newnames/page.html
```

Copy the whole `$A/newnames/` folder to the Mac. **Send me** the step-1 lines, the step-2 VERDICT
blocks, the two score files, and later the new-names answers. Reading the new-names result: 0 wrong
in 200 gives a lower bound of 0.982; proving 0.99 at zero errors takes 368. Two hundred first is a
cheap stop if errors appear; extend to 368+ if it is clean.

## 7. After that (I will send the exact commands with the results of step 6b)

- full embed of facilities 2 and 3 (face), ~2 h at the measured rate (estimate);
- `eval` → curves and operating point, sweeping the quality cutoff too;
- body candidate (SigLIP2) embed and `eval` with body linking;
- `shadow --point point_3_temporal.json` on facility 3 (uses only labels that existed at each
  night's run) and `adjudicate-score` against `adjudication_f3_set1.csv` (excludes sightings that
  carry a human label; reports pooled and population-weighted precision).

`--face-model` / `--body-model` take a cache slug or a substring of the model id that matches
exactly one cached model; ambiguous names are refused. `vbb-reid-server doctor --models <cfg>`
prints the exact ids.
