# Deploying and running machine labels

How the pieces were deployed on 2026-11-05, and how to repeat or operate them. Never write a token into this file, a chat or a log.

## What runs where

| Piece | Runs on | How it gets there |
|---|---|---|
| `vbb_reid_server` (the nightly GPU job) | gpu-ubun only, never on a production host | `git pull` on the box |
| `vbb_device` (`/worker/*`, `worker:token`) | Production, `/web/vbb_device_backend` | Normal deploy |
| `vbb_app` (review feed, `nursery:machine-run`) | Production, `/web/vbb_app_backend` | Normal deploy |
| Three tables and one column | Nursery database | By hand, before the code that needs them |

## Order for a new environment or a new database

Apply the database change before deploying code that uses it: the review feed reads `v_person_machine_runs` on every load, so without the tables the review page breaks.

### 1. Database change (applied in production 2026-11-05)

Run this only if the tables are missing. Check first: it prints `false` when they are.

```bash
cd /web/vbb_app_backend && php artisan tinker --execute='var_dump(Schema::connection("nursery")->hasTable("person_machine_runs"));'
```

The file is in `vendor/vbb/nursery/database/migrations/person_machine_labels.sql` once `vbb_app` holds nursery `v0.0.111` or later. If code must not be deployed yet, take the file from git without touching the working tree: `git fetch origin <branch> && git show FETCH_HEAD:vendor/vbb/nursery/database/migrations/person_machine_labels.sql > /tmp/person_machine_labels.sql`. Then apply it:

```bash
cd /web/vbb_app_backend && timeout 120 php artisan tinker --execute='DB::connection("nursery")->statement("SET SESSION lock_wait_timeout = 10"); DB::connection("nursery")->unprepared(file_get_contents("/tmp/person_machine_labels.sql")); DB::connection("nursery")->reconnect(); var_dump(Schema::connection("nursery")->hasColumn("person_sighting_corrections", "machine_run_id"));'
```

`unprepared` hides errors, so empty output proves nothing. Check the result by reading state: the line above must print `bool(true)`, and step 2's schema check must pass. Don't re-run the file once it has worked.

### 2. Deploy the code, then check

Deploy `vbb_app` and `vbb_device`, then:

```bash
cd /web/vbb_app_backend && php artisan nursery:check-schema
```

It must print `OK`, with no 缺表 or 缺列 (missing tables or columns). On 2026-11-05 it printed `OK —— 48 张表全部就位`. This check is only meaningful after the deploy, because it compares the database with the deployed code.

From gpu-ubun, confirm the route is live. No token is needed and nothing changes:

```bash
curl -sS -m 20 -X POST -H 'Content-Type: application/json' -d '{}' https://vbb.yuexingren.cn/worker/runs/register
```

It must return `{"code":401,"message":"worker token required",...}`. An HTML 404 means the route isn't deployed.

### 3. Worker token (one per GPU host, done once)

Issue it on the production host. It is shown only this once; the database keeps only a fingerprint.

```bash
cd /web/vbb_device_backend && php artisan worker:token issue --org=2 --label=gpu-ubun
```

`--org` is the brand of the facilities the host works on (facility 3 was brand 2 in the 2026-11-03 export). A token for the wrong brand gets a 403 on the first run.

Save it on gpu-ubun as `dosomi`, in your own terminal. Paste this one line, then paste the token at the prompt:

```bash
mkdir -p ~/.config/vbb_reid_server && ( umask 077; read -rs -p 'token: ' T; echo; printf '%s\n' "$T" > ~/.config/vbb_reid_server/worker.token ); ls -l ~/.config/vbb_reid_server/worker.token; wc -c < ~/.config/vbb_reid_server/worker.token
```

It must show `-rw-------` and `52`. Keep it in the home folder, because `/vmdata` may ignore file permissions.

List or cancel tokens on the production host:

```bash
cd /web/vbb_device_backend && php artisan worker:token list
cd /web/vbb_device_backend && php artisan worker:token revoke --id=<number>
```

The worker token cannot activate or revert a run. Activation stays on the app host.

### 4. The job on gpu-ubun

The box uses `/home/dosomi/dev/vbb_reid_server` and the shared venv `/home/dosomi/dev/ReID/.venv`. To update the code:

```bash
cd /home/dosomi/dev/vbb_reid_server && git pull --ff-only && /home/dosomi/dev/ReID/.venv/bin/python -m pytest -q
```

Reinstall only if dependencies changed. Test with a run that pushes nothing:

```bash
/home/dosomi/dev/vbb_reid_server/scripts/nightly_run.sh --day <YYYY-MM-DD> --dry-run
```

`scripts/nightly_run.sh` is the nightly job, at 01:00 Shanghai time, which is facility 3's time. It keeps the backend and the CDN off the proxy that login shells configure, and refuses to start if another run holds the lock. Output goes to `/vmdata/vbb_eval/vbb_run/` (`logs/`, `runs/`, `emb/`, `cdn/`).

**Schedule:** installed on the owner's word on 2026-11-05, in `dosomi`'s crontab on gpu-ubun. The first run is 2026-11-06 01:00 Shanghai, for 11-05. The box clock is UTC, so 01:00 Shanghai is written as 17:00. If the box's time zone ever changes, this line must change with it:

```text
# Machine labels, nightly: 01:00 Shanghai. The box clock is UTC, so it is written as 17:00.
# Face only, until 2026-11-06 (owner: "Turn on the body part for our correction job"):
# 0 17 * * * /home/dosomi/dev/vbb_reid_server/scripts/nightly_run.sh
# Face + CLIP-ReID body (vbb-reid-server 0.2.2, "resize": "pil"), from 2026-11-07 01:00 Shanghai:
0 17 * * * VBB_RUN_POINT=/vmdata/vbb_eval/models/point_3_clipreid_body.json VBB_RUN_MODELS=/vmdata/vbb_eval/models/buffalo_l_face_clipreid_body.json /home/dosomi/dev/vbb_reid_server/scripts/nightly_run.sh
```

Body was switched on 2026-11-06 14:45 Shanghai (plan decision 32). The two files it names are read-only. Before the switch, a replay at the frozen line matched E20 on 12,289 of 12,290 labels: none changed person and no E20 picture changed. The 4 that differed are right at the line and come from onnxruntime's CUDA TF32 default, which is not pinned. A dry run of 11-05 also matched. To go back to face only, `crontab /vmdata/vbb_eval/out/crontab_before_body_20261106-064513.txt`. To stop the job, remove those lines with `crontab -e`. Labels already pushed stay inactive unless someone activates them.

gpu-ubun could not reach GitHub on 2026-11-06 (the proxy failed upstream, and a direct connection timed out), so 0.2.2 arrived as a git bundle made on the Mac and fast-forwarded. Check `git -C /home/dosomi/dev/vbb_reid_server log -1` against GitHub main before trusting a pull.

A run stops after 3 hours. Exit 73 means the output folder could not be created, recorded in `~/vbb_run_FAILED`; exit 75 means another run holds the lock; exit 124 means the time limit.

## Operating it

On gpu-ubun, look for failures. These files only exist if something failed, and nothing alerts you:

```bash
cat /vmdata/vbb_eval/vbb_run/FAILED ~/vbb_run_FAILED
```

On the app host, see the runs and act on them:

```bash
cd /web/vbb_app_backend
php artisan nursery:machine-run list --facility=3
php artisan nursery:machine-run show --run=<run_id>
php artisan nursery:machine-run activate --run=<run_id>
php artisan nursery:machine-run revert --run=<run_id> --reason='<why>'
php artisan nursery:machine-run off --facility=3 --reason='<why>'
php artisan nursery:machine-run on --facility=3
```

- `activate` shows that night's labels; nothing shows until then. No facility is switched on before the licence (T18) is settled, the 0.98 bar is closed, and the app shows the machine mark.
- `revert` undoes one bad night. `off` undoes all of a facility's nights, and `on` brings back the ones `off` removed.
- Changes show on the next read of the review page, with no restart.
- `nursery:machine-run expire` runs hourly by itself on the app host's scheduler.

## Checking each night's names (0.2.3)

After each successful nightly run, `nightly_run.sh` draws 20 of that night's new names (pushed `known` promotions) into `/vmdata/vbb_eval/vbb_run/verify/<facility>_<day>/frozen.json`, read-only. It does this only when `verify/refs.json` exists, and only once per night. A failed draw adds a `verify-build failed` line to `FAILED` without failing the run.

To review, run this on your own computer and leave it running. It starts the page on gpu-ubun for as long as the ssh session lasts:

```bash
ssh -L 8770:127.0.0.1:8770 dosomi@192.168.0.107 \
  /home/dosomi/dev/ReID/.venv/bin/vbb-reid-server verify-serve --root /vmdata/vbb_eval/vbb_run/verify --port 8770
```

Then open `http://localhost:8770/`. If ssh exits with an error, or the page doesn't load, nothing is served. The page lists the nights. A night shows one picture at a time beside reference pictures of each person, and asks who it is. The machine's name, scores and picture links never reach the page. After "I'm done", the list shows that night's result, totals by face and body and by person, with the 97.5 % lower bound, and the wrong names with their pictures.

Reference pictures (`refs.json`) are built once, and again when the roster changes. They come only from answers the owner gave one picture at a time and blind, never from bulk labels. Each must show exactly one face, agree with its human label if it has one, and not be on the exclude list:

```bash
cd /home/dosomi/dev/vbb_reid_server
/home/dosomi/dev/ReID/.venv/bin/vbb-reid-server verify-refs --export <phase A export> --aliases <aliases.csv> \
  --emb <phase A emb root> --face-model buffalo --facility 3 --answers <answer files…> \
  --exclude a2edee5618ff 36aed033c5b4 07e911a53a04 81e8e70dc82f 83c78b81f9ed 371e6e5c751c \
  --out /vmdata/vbb_eval/vbb_run/verify/refs.json
```

A night that was drawn before a change to `refs.json` keeps the references it was drawn with. Answers live in `marks.json`, with one line per save in `marks.log`, next to each night's `frozen.json`.

## Learning only from reviewed pictures (0.2.4, off until the owner's word)

`--label-check` on `run` or `shadow` makes the job learn only from staff labels on pictures someone looked at (plan decision 34, `vetting.py`). In the nightly job it is switched on with `VBB_RUN_LABEL_CHECK=1` on the crontab line (`scripts/nightly_run.sh` then adds `--label-check --verify-root $ROOT/verify`). Don't pass the options as extra arguments: an extra-argument run skips the night's check page, where the disagreements are reviewed. A picture counts if it carries the backend's `reviewed` mark, or if the owner answered it on the check page. `--verify-root` points at the check page's folder: `refs.json` and each finished night's answers settle the pictures they cover. A reviewed picture teaches the face only if it shows one clear face. Otherwise it only seeds that day's body match. Each run's JSON reports `label_check` counts (`reviewed`, `answered`, `not_reviewed`, `answer_disagrees`, and `no_face` / `two_faces` / `poor_face` for reviewed pictures without a clear face) and lists `disagreements`. `verify-build` adds all of them to the night's page (owner 2026-11-07: "show ALL disagreements, at least for now"); `--n-check` caps them.

Nothing is marked until the backend release with `reviewed_ids` is deployed and staff use an app build that sends it. Renames from before count as not reviewed. So the replay waits until staff at facility 3 have reviewed with the new app, on an export taken after that. The `reviewed` count in a run with the check on shows how many there are. If that count is missing or zero, the export carries no marks (too early, or its `extras` column is missing): stop, the comparison would show nothing. Use a cache built from the uploaded pictures, as the nightly job's own `emb` is. In a cache built with `--lan-crops` the board's crops replace the uploaded picture, which is the one staff looked at.

Replay before turning it on: the same shadow twice, once without and once with the check. Compare `references` per person in the two run JSONs, and score both labels files with `adjudicate-score` against the owner's answers:

```bash
cd /home/dosomi/dev/vbb_reid_server
for MODE in off on; do
  /home/dosomi/dev/ReID/.venv/bin/vbb-reid-server shadow --export <phase A export> --aliases <aliases.csv> \
    --emb <phase A emb root> --facility 3 --days <days> --point /vmdata/vbb_eval/models/point_3_clipreid_body.json \
    --face-model buffalo --body-model <CLIP-ReID cache slug> --roster-only-refs \
    $([ $MODE = on ] && echo --label-check --verify-root /vmdata/vbb_eval/vbb_run/verify) --out <out>/$MODE
done
```

It fails if either of these holds with the check on:
- precision on the answered pictures is lower;
- the labels file names someone who has fewer than `min_refs` (3) face references in the run JSON's `references`.

Either result goes to the owner before the nightly changes. People with no reviewed pictures yet simply aren't named; that is expected and not a failure. Scoring against answers that also settle labels is fair only for pictures that weren't themselves staff labels, and `adjudicate-score` counts only those.

## Known gaps

- `vbb_contracts` is not merged: `machine_labels.md` is on its `claude/wonderful-hawking-vpm3va` branch, and it belongs to someone else.
- The app's composer lock still names an older nursery, because the vendored copy is updated by hand. A `composer install` on the host would silently revert it.
