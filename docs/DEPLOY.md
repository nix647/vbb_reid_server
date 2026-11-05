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

`scripts/nightly_run.sh` is the nightly job, for 00:05 facility time. It keeps the backend and the CDN off the proxy that login shells configure, and refuses to start if another run holds the lock. Output goes to `/vmdata/vbb_eval/vbb_run/` (`logs/`, `runs/`, `emb/`, `cdn/`).

**Schedule:** in `dosomi`'s crontab on gpu-ubun. The box clock is UTC, so 00:05 Shanghai is written as 16:05. If the box's time zone ever changes, this line must change with it:

```text
5 16 * * * /home/dosomi/dev/vbb_reid_server/scripts/nightly_run.sh
```

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

## Known gaps

- `vbb_contracts` is not merged: `machine_labels.md` is on its `claude/wonderful-hawking-vpm3va` branch, and it belongs to someone else.
- The app's composer lock still names an older nursery, because the vendored copy is updated by hand. A `composer install` on the host would silently revert it.
