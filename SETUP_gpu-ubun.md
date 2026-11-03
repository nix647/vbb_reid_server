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

Expected: `30 passed`; `onnxruntime ... providers [...'CUDAExecutionProvider'...]`;
`torch ... cuda available=True, device NVIDIA GeForce RTX 4070 Ti SUPER`.
If `CUDAExecutionProvider` is missing, run `pip install 'onnxruntime-gpu[cuda,cudnn]'` and check again.

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

## 7. After that (I will send the exact commands with the results of step 6)

- full embed of facilities 2 and 3 (face);
- `eval` → curves and operating point;
- body candidate (SigLIP2) embed and `eval` with body linking;
- `shadow` on facility 3 and `adjudicate-score` against `adjudication_f3_set1.csv`.
