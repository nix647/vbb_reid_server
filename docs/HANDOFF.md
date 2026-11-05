# Handoff — nightly machine labels (server-side re-verification), as of 2026-11-05

Read this first, then `docs/PLAN_nightly_machine_labels.md` (the approved plan; decision register §2, Phase A findings §3c E1–E10) and `docs/PLAN_phase_B.md` (Phase B work packages, review outcome, build status). The interface is `nix647/vbb_contracts/machine_labels.md` (PROPOSED).

## What this is
A nightly GPU worker re-identifies each facility-day's person sightings with a stronger face model, using only human-labeled crops as references, and proposes "machine labels" for sightings the board left unsure. Labels are stored inactive; a person switches a night on in the app host; the staff app shows them with a 「机器复核」 mark and no score. Human labels always win.

## Where things stand
- **Phase A passed** on facility 3 (2026-11-05): blind uniform sample of 200 machine new names, 199 right / 0 wrong / 1 can't tell, lower bound 0.9816 ≥ the owner's 0.98 bar. Operating point v1 in plan E8 (buffalo_l, face_sim 0.40, margin 0.05, norm floor 14, two-faces 0.25, ≤ 50 refs/person, body off).
- **Phase B built on branch `claude/wonderful-hawking-vpm3va`, NOT deployed, DDL not applied:**
  - `alonsolee/vbb_nursery@917d13f` — DDL `database/migrations/person_machine_labels.sql`, `MachineLabelHelper`, `IdentityResolver`, readers.
  - `alonsolee/vbb@1724850` — feed `source`, `nursery:machine-run {list|show|activate|revert|off|on|expire}`, tests (`tests/Nursery/MachineLabel*Test.php`).
  - `nix647/vbb_device_backend@82114bd` — `/worker/*`, bearer auth, `worker:token`.
  - `nix647/vbb_reid_server` (this repo, 0.2.1) — `vbb-reid-server run`.
  - `nix647/vbb_contracts@85dc13b` — contract with all review fixes and decisions.
- An adversarial review (3 independent reviewers) was done and its findings fixed; see PLAN_phase_B "Adversarial review".

## Owner decisions in force (2026-11-05)
- Phase A bar 0.98; label budget ~800, ~600 used. Any stage needing labels states how many and why first.
- Worker tokens in the nursery DB; worker never activates (activation on the app host only).
- **No switch** (option B): a machine label shows iff its run is activated; facility off = `nursery:machine-run off` (reverts with `facility_off`), on = `on`. Feed freshness via runs' `changed_ts`.
- **Card actions confirm machine members** (D1 option B); audit `machine_run_id` records which run was overwritten; app shows a warning only.
- No facility switched on before: licence T18 settled, step 6g re-measured, app mark shipped.
- Monitoring once live: ~20 blind labels/week early on, then label-free.
- Name list / nurse pickers no longer carry `photo_url` (only the merge card draws a photo).

## Open items (who)
1. **Owner:** run SETUP step 6g on gpu-ubun (`SETUP_gpu-ubun.md`) and report the 3 score lines + 4 counts — it checks the production reference rule (roster-only, no alias file) against the Phase A answers.
2. **Owner/legal:** T18 — buffalo_l is non-commercial research only; InsightFace commercial licence enquiry.
3. **App developer (`alonsolee/vbb_app`, read-only to us):** render `source: "machine"` mark; card-action warning with the count.
4. **Other team:** review the alonsolee branches, apply the DDL, tag nursery, deploy device backend + app, issue gpu-ubun's token (`php artisan worker:token issue --org=<id> --label=gpu-ubun`).
5. **Then (B.5):** nightly `vbb-reid-server run` with labels inactive for a week; activate facility 3 via artisan; revert drill; Gate G_B.
6. **Later:** scheduled auto-activation with guards (after G_B); Phase C (teaching boards, `nix647/vbb` board changes via contract); body model; facility 1 cloud host.

## Working rules carried over
- alonsolee repos belong to another team: change only on the review branch, never deploy. `nix647/vbb` (board) is specified via contract entries only.
- Never commit names, aliases.csv, crops or embeddings; customer crops stay on gpu-ubun; don't write real names from answer files.
- No model identifiers in commits; no PRs unless asked; no unmeasured numbers.
- Tests: worker `pytest -q` (106); app `vendor/bin/phpunit --testsuite Nursery` (1 pre-existing error: ParentReportsTest fixture path); device backend `vendor/bin/phpunit` (4 pre-existing contract-fixture failures); contracts `python3 scripts/validate_schemas.py && python3 scripts/gen_status.py --check`.
- The owner prefers plain-language explanations with concrete consequences over abstract design terms.
