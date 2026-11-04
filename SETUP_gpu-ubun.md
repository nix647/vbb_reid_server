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

Expected: `57 passed` (0.1.2 + configs); `onnxruntime ... providers [...'CUDAExecutionProvider'...]`;
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
