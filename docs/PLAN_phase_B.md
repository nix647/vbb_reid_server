# Phase B plan — store, resolve and display machine labels (no boards)

Companion to `PLAN_nightly_machine_labels.md` §5.1–5.5 and §7 Phase B, and to `vbb_contracts/machine_labels.md` §1–§4 (PROPOSED, `78a3c69`). Written 2026-11-05 after Phase A passed on facility 3 (plan §3c E7, operating point E8). Basis re-read today: `alonsolee/vbb_nursery@58f3edc`, `alonsolee/vbb@ea5184b`, `nix647/vbb_device_backend@d31afec`, `nix647/vbb_reid_server@4c403f1`.

## 0. What Phase B is, and the two gates in front of it

Phase B makes machine labels a first-class, reversible layer in the backend:
worker (gpu-ubun) → `/worker/*` on the device backend → two nursery tables → one identity resolver → review feed with a provenance mark. Boards are untouched (Phase C). No owner labels are spent: Gate G_B is judged on organic corrections and human reports.

Two things gate **switching it on**, not building it:
- **T18 licence.** Building, nightly runs and inactive rows in the DB are evaluation. Showing a `buffalo_l` label to staff, even in test facility 3, is use. Needs the InsightFace answer or a legal OK for test-facility use.
- **App provenance mark.** Decision 3 allows a machine label in the feed *with its provenance*. Until the staff app renders `source: "machine"`, a machine name looks like a board name. §7 Q5 asks how you want to handle the gap.

Build order is chosen so that every deploy is safe with the switch off (default): DDL → nursery tag → device backend → worker token → app → nightly runs (inactive) → one week shadow-in-DB → activate facility 3.

## Progress (2026-11-05)

- **B.0 done.** `vbb_contracts@54ce6c5` (C1-C8 and decisions 21-25), `@6afc53e` (C9 below).
- **B.4 client side done.** `vbb_reid_server@f11d9a7` (0.2.0): `/worker` client, backend export, `run` command, 89 tests; tested against an in-memory backend only.
- **Side change (owner 2026-11-05), done on branch:** the name list and visiting-staff rows no longer carry a photo, because only the merge-review card draws one (`vbb_app nix_dev@93270cf`). Commits: `vbb_nursery@20f2cc0`, `vbb@f72233d`, `vbb_contracts@702d400`.
- **C9, found while writing the client.** In production, references are only the labels of the facility's own roster (person ids folded through merges). Phase A unified facility 2 and 3 ids with the owner's alias file, which production does not have. Without this rule, the same person under two ids competes with themselves, the margin fails, and the machine abstains. Owner step 6g measures the rule offline on existing answers.
- **Next:** B.1 nursery (DDL, models, `MachineLabelHelper`, resolver, readers), then B.3 device backend, then B.2 app.

## Build status (2026-11-05, end of day)

Built on branch `claude/wonderful-hawking-vpm3va`, nothing deployed:
- **B.1 nursery** `alonsolee/vbb_nursery@917d13f`: DDL (3 tables + `corrections.machine_run_id`), `MachineLabelHelper`, `IdentityResolver`, feed + class expansion + merge card + attribution readers. Tests in `alonsolee/vbb`.
- **B.2 app** `alonsolee/vbb@1724850`: feed `source` + no score on machine members, `nursery:machine-run`, hourly expire. Nursery suite 2,091 tests (1 pre-existing error), 6 pre-existing unit failures unrelated.
- **B.3 device backend** `nix647/vbb_device_backend@82114bd`: `/worker/*`, bearer auth, `worker:token`. 4 pre-existing failures unrelated.
- **B.4 worker** `nix647/vbb_reid_server@8f693c8` (0.2.1).
- **Decisions applied:** D1 option B (card actions confirm machine members; audit keeps the run id), switch option B (no switch; off = facility_off reverts).
- **Not built:** the staff app's mark and card warning (app developer; contract items), Phase C, the scheduled auto-activation (after G_B).
- **Merged 2026-11-05 (owner: we merge our own repos), not deployed:** `person_machine_labels.sql` applied in production first; nursery `v0.0.111` (`6f9a653`), `alonsolee/vbb` branch `vbb_app` `ddafc6f2c`, `nix647/vbb_device_backend` main `05b974965`, vendored copies byte-identical to the tag. `vbb_contracts` is not merged: it belongs to someone else.
- **Next:** deploy device backend + app (owner), `nursery:check-schema` on the app host, issue the gpu-ubun token, then nightly `run` with labels inactive (B.5).

## Adversarial review (2026-11-05): what changed in this plan

Three independent reviews (worker 0.2.0, the photo change, the contract). Fixed and pushed: worker 0.2.1 (`vbb_reid_server@85ae1ea`), photo follow-ups (`vbb_nursery@1c7dcb2`, `vbb@f19c261`, `vbb_contracts@0d60f70`), contract (this commit series). Plan changes that follow:
- **The switch moves from `.env` to the nursery DB** (`v_machine_label_facilities`, read by the package, with a stamp that rebuilds stale feeds on read). An `.env` list needs `config:cache` and a restart of Swoole and the queue workers, and queue jobs rebuild the feed outside any request, so the per-request hand-off in B.2 would not be instant or even consistent. **Owner sign-off needed** (it breaks the package's "caller passes the switch" convention for a reason). B.2 loses the `config/monitors.php` key and the middleware; it gains `nursery:machine-labels {enable|disable|status}`. B.5 step 2 is one artisan command, no `config:cache`.
- **No cross-host cache flush**: activate/revert/switch bump the stamp; `flush-feed` is dropped.
- **DDL**: the label table's `state ENUM` clashed with the soft-delete `state TINYINT` (MySQL error 1060); it is `label_state`. Runs gain `machine_worker_id`, a unique `lease_key`, `revert_reason`; labels gain `INDEX (facility_id, person_id, active)`.
- **The worker never activates**; activation stays on the app host (artisan now, a scheduled app command with guards after G_B). Plan Q6 answer stands; only where it runs changed.
- **Status machine**: `reverted → activated` (undo an accidental revert) and `superseded → reverted` (retract what a superseded run taught) added. B.5 step 3 (revert, then re-activate) now works.
- **Idempotent register / finish / activate / revert**, run-scoped routes checked against the token's org and the registering worker, an hourly `expire` command for hard-killed runs, push validates the relation against the stored board prediction.
- **Person-keyed readers** (merge card photo, merge statistics, nurse-seen attribution) must read machine rows by person id, not only by sighting id.
- **Q12 decided 2026-11-05: option B** (card actions cover machine-named members; app shows a warning; the audit records the overwritten run_id). Original question: a whole-card human action (merge into, demote, rename) currently sends every sighting of the card, so it would turn machine labels no one looked at into human labels, boards' training data and shared faces, none of it undone by revert. Recommended: human writes skip machine-sourced sightings unless the app names them as individually confirmed.
- **Switch-on preconditions** added to the contract: T18 settled, step 6g re-measured (including names the Phase A sample never covered), the app badge shipped.

## 1. Work packages by repository

Ownership as in plan §7: **ours** = `vbb_contracts`, `vbb_device_backend`, `vbb_reid_server`; **ours on branch for their review** = `vbb_nursery`, `vbb` (decision 17). Branch everywhere: `claude/wonderful-hawking-vpm3va`.

### B.0 Contracts (`nix647/vbb_contracts`) — amend `machine_labels.md` before code

Findings while planning against the code; each becomes a contract edit (still PROPOSED, so no consumer is affected):

| # | Change | Why |
|---|---|---|
| C1 | `v_machine_workers` moves into the nursery migration (same file as the two label tables), model and auth lookup in the nursery helper; the device backend keeps the middleware | The device backend has no DDL home of its own (no `database/*.sql`; its tables come from vendored packages). The nursery connection already holds `v_orgs`, which the token's `org_id` scope references, and `nursery:check-schema` then guards all three tables. **Q1** |
| C2 | Run status enum gains `superseded` | Activating a newer run for the same `(facility, day)` must leave the older one distinguishable from a deliberate revert (Phase C retracts only reverted runs) |
| C3 | Push rejection reasons enumerated: `human_labeled`, `not_in_roster`, `not_in_run_scope`, `run_not_running`, `bad_row` | The worker logs and counts them; a schema with an open string invites drift |
| C4 | `references/pull` stated explicitly as brand-wide (every facility with the same `org_id`), `since_ts` on `capture_ts` | The schema already carries `facility_id` per reference; the text should say it |
| C5 | Activation invariant: clears the person-review feed cache of every room the run touched; revert and switch changes likewise | The feed is cached per room (`personReviewFeedKey`, TTL 7200 s); without this a label appears or disappears up to two hours late |
| C6 | Appearance `source` rule: `human` if any member is human-corrected, else `machine` if any member is machine-labeled, else `board`; sighting `source` is exact | Members of one appearance can have different sources |
| C7 | Worker controller bypasses the fleet request validation (`timestamp`/`sign`/basic params) and runs the bearer check instead | `AuthLogicApiController::isValidRequest` applies the device signature to every Feature controller; a worker holding `APP_SECRET_KEY` could impersonate boards (the reason the contract rejects it) |
| C8 | `EventEscalator` roster-photo back-fill stays human-only in v1 (remove it from the resolver's reader list) | It writes `face_gallery.photo_url` permanently; a wrong machine label would put a stranger's face on a person's card and the kill switch would not undo it. **Q3** |

Also: PROPOSED → APPLIED flips happen per surface as each repo ships, with commit hashes; `ddl/vbb_nursery.sql` mirrored after the migration is applied in production.

### B.1 Nursery package (`alonsolee/vbb_nursery`, branch for review)

Files:
- `database/migrations/person_machine_labels.sql` — `v_person_machine_runs`, `v_person_sighting_machine_labels`, `v_machine_workers` (C1), idempotent guards, `-- 施加于生产：` header; same tables appended to `database/create_vbb_nursery.sql`.
- `src/Model/Machine/VPersonMachineRunModel.php`, `VPersonSightingMachineLabelModel.php`, `VMachineWorkerModel.php` (casts: `thresholds`, `counts`, `evidence` → array).
- `src/Enums/MachineRunStatus.php`, `MachineLabelState.php`, `MachineLabelRelation.php`, `MachineLabelVia.php`.
- `src/Module/NurseryMonitor/Helpers/IdentityResolver.php` — pure static, no DB: `effective(array $row, ?array $machine, bool $machineEnabled): [state, person_id, name, source]`. `AppearanceGrouper::effectiveIdentity($row)` becomes a one-line wrapper reading `$row['_machine']` and `$row['_machine_enabled']`, so every existing caller that passes arrays keeps working and gains `source`.
- `src/Module/NurseryMonitor/Helpers/MachineLabelHelper.php` — trait on `NurseryManager`:
  - lifecycle: `registerMachineRun`, `finishMachineRun`, `activateMachineRun`, `revertMachineRun`, `getMachineRun`, `listMachineRuns`;
  - data: `pullMachineSightings`, `pullMachineReferences`, `pushMachineLabels`;
  - auth: `authenticateMachineWorker(['token_hash'])`, `issueMachineWorker`, `revokeMachineWorker`;
  - readers: `machineLabelsFor(array $sightingIds): array` (active rows, keyed by sighting_id, **empty for facilities not in the enabled list**: the switch is applied here, in one place, not in each reader);
  - switch: `setMachineLabelFacilities(array $ids)` / `machineEnabledFor($facilityId)` on the manager, per request, default `[]`.
- Reader switches (plan §5.1 list minus C8): `getPersonReviewFeed` (prefetch inside `$compute`, so the cache holds resolved identities), `PersonReviewHelper::_identityClassKey`, `FaceHelper::_creditedPersonId`, `PersonMergeHelper::_personSightingStats`, `AttributionHelper::_trackIdentity` and the nurse-seen loop (`:455-483`). Each gets one prefetch call; none queries per row. `PersonFaceVectorHelper::_faceSightingOwners` untouched (decision 10).
- Names for machine identities: the row has no `corrected_name` snapshot; the prefetch joins `v_face_gallery.name` by `person_id` (current name, like the outbox's "read live" rule).

Behaviour pinned down (also mirrored in the contract):
- **Day bounds:** `[start, end)` in epoch ms from `v_facilities.time_zone` for `day`; `capture_ts` compared, not `captured_at`.
- **Lease:** `lease_until = now + 60 min`; `register` refuses (`409`-style error in the envelope) while another `running` run holds an unexpired lease for the pair; an expired `running` run is marked `failed` with `error = lease_expired` when a new one registers.
- **Status machine:** `running → done | failed` (finish); `done → activated` (activate); `activated → reverted` (revert) or `→ superseded` (another run of the same pair activated). Anything else is refused with the current status in the message. Activate and revert are one UPDATE each on the labels (`active = 1|0 WHERE run_id = ?`) inside a transaction with the run row.
- **Push validation:** per row — `sighting_id` belongs to the run's facility-day; `state ∈ {known, flag}`; `known` needs `person_id` in the facility's closed roster (non-tentative, non-merged registered ids, visitors included); a sighting with `corrected_at` set rejects `known` with `human_labeled`; upsert on `(sighting_id, run_id)`; one transaction per request; response `{accepted, rejected[]}`.
- **Pull paging:** cursor = last `id` (opaque base64), `limit ≤ 500`, ordered by `id`; first page of `references/pull` carries `roster` and `time_zone`.
- **Revert reason** stored in `counts.revert_reason` and `reverted_at`; Phase C will read the reverted run's sent ids from here.

Tests (in `alonsolee/vbb/tests/Nursery/`, sqlite harness; the package has no test dir of its own):
- resolver precedence human > machine > board; reset-to-pending stays `uncertain` even with an active machine row;
- machine hidden when: switch off for the facility, run `done` but not activated, run reverted, run superseded;
- machine `person_id` folded through merges like board ids;
- `_faceSightingOwners` output byte-identical with and without machine rows;
- push: idempotent re-push, `human_labeled` rejection, `not_in_roster` rejection, scope rejection;
- activate supersedes the previous activated run of the pair; feed cache keys of the run's rooms are forgotten (SpyArrayStore);
- `NurserySchema` fixture gains the three tables; `SchemaFidelityTest` stays green;
- **mutations:** remove the `active = 1` filter → red; apply the switch in the resolver instead of the prefetch → the "switch off" test must still be red when a reader forgets the flag (i.e. the test asserts through a reader, not through the resolver).

### B.2 App backend (`alonsolee/vbb`, branch for review)

- `config/monitors.php`: `'machine_labels' => ['facilities' => <comma list from MONITORS_MACHINE_LABELS_FACILITIES>]`, default empty, documented in the file's header in the same voice as `suite_facilities`.
- Hand-off once per request: a global middleware in the `api` group (and the console kernel's bootstrap) calls `Nursery::setMachineLabelFacilities(config('monitors.machine_labels.facilities'))`. Per request, not per process: the switch must not leak between requests under a long-lived worker.
- `PersonReviewFormatter`: `source` on each appearance and each sighting when the grouper provides it (absent when the facility is off → old behaviour).
- Artisan `nursery:machine-run {list|show|activate|revert} {--run=} {--facility=} {--day=} {--status=} {--reason=}` and `nursery:machine-labels flush-feed --facility=` (clears every room's feed cache; needed after flipping the switch).
- Re-vendor `vendor/vbb/nursery` byte-identical to the nursery branch; bump the `composer.json` pin to the new tag; `nursery:check-schema` will report the three missing tables until the DDL is applied (that is the guard working).
- Tests: formatter emits `source`, `corrected` stays human-only, middleware sets and clears the list per request.

### B.3 Device backend (`nix647/vbb_device_backend`)

- `app/Features/Worker/feature_routes.php` (prefix `worker`): the six POST routes of contract §3.
- `WorkerController extends PublicApiLogicController`, overriding `isValidRequest()` to run `WorkerBearerInterceptor` (C7): `Authorization: Bearer <token>` → `sha256` → `Nursery::authenticateMachineWorker` → worker context `{id, org_id, scope}` on the params; missing/invalid → `401`; `activate`/`revert` need `scope = admin` → else `403`; `last_seen_at` touched at most once a minute.
- Logic classes, one per route, thin: shape checks (`run_id` pattern, `limit ≤ 500`, enums, `rows ≤ 500`), org scope (`v_facilities.org_id` of the requested facility = token org, else `403`), then the facade. Facility lookup through `Nursery::getFacilityInfo` (cached).
- Artisan `worker:token {issue|revoke|list} --org= --label= --scope=worker|admin`: prints the token once, stores only the hash.
- No long polls; every handler is one or two queries; the 60 s nginx limit is not approached at `limit = 500`.
- Rate limiting: the framework throttle is commented out in this app's kernel; v1 relies on the token and page caps. Noted as a risk, not built.
- Re-vendor the nursery package (byte-identical) and bump its pin (today `^0.0.89`, vendored 58f3edc).
- Tests (`tests/Unit`, Mockery on the facade like `DeviceOpsLogicTest`): no header → 401; bad token → 401; worker token on activate → 403; facility of another org → 403; `limit = 501` refused; push over 500 rows refused; register/finish/activate call the facade with the exact arguments. **Mutation:** drop the org check → the cross-org test goes red.

### B.4 Worker (`nix647/vbb_reid_server`, 0.2.0)

- `api.py`: `BackendClient` (bearer, JSON, 30 s timeout, 5 retries with backoff on 5xx/connection errors, never on 4xx), page iterator, envelope unwrapping (`code` ≠ success → `BackendError` with message and action).
- `data.from_backend(client, facility, day, window_days) -> Export`: the day's sightings become probes (`human_state` from `human_labeled`), brand-wide references since `day − window_days` become `known` human-labeled sightings with `corrected_ts`, roster → `Person`, `time_zone` and `org_id` → the two facility maps. Everything downstream (`Workspace`, `embed_day`, `face_scores`, `decide_day`) is unchanged, so the Phase A code path is the production code path.
- Crops: `CropStore` with the CDN cache (decision 18) and the optional LAN root for test boards; embedding cache keyed as today, pruned at 14 days.
- New command `run`: `vbb-reid-server run --facility 3 --day auto|YYYY-MM-DD --point point_3.json --models models.json --backend URL --token-file PATH [--lease-minutes 60] [--dry-run] [--activate]`. Steps: register (worker-minted `run_id`) → pull → embed → references → decide → push in batches of 500 (idempotent; a batch is retried whole) → finish `done` with counts (including `rejected` by reason) → optionally activate (admin token). Any exception → finish `failed` with the error string; the local CSV/JSON of the run are always written (`/vmdata/vbb_eval/runs/`). `--day auto` = yesterday in the facility's time zone; `--days A..B` for catch-up.
- `model_id` = `"<face_detector_id>|<face_id>|<body_id or none>|vbb-reid-server <version>"`, the ids including preprocessing as today; the contract's `model_id` description is loosened to match.
- Scheduling on gpu-ubun: `scripts/nightly_run.sh` at 00:05 Asia/Shanghai (facility 3's zone; owner 2026-11-05, was 02:30); a failure appends to `/vmdata/vbb_run/FAILED`; the app-side `nursery:machine-run list --status failed` and a watchdog line in the morning check (owner's existing routine) cover alerting. No email from gpu-ubun.
- Tests: client paging/backoff with a fake server; `from_backend` builds the same `Export` as the CSV loader for a fixture day (same ids, same labels, same tz day); `run --dry-run` on the fixture produces the same labels as `shadow` for that day (the B.4 verify step from plan §7, done offline first, then once against the real backend).

### B.5 Shadow-in-DB, activation, drill (owner runs; we read)

1. **Week 1, rows inactive.** Nightly `run` for facility 3. Each morning: `nursery:machine-run list --facility 3`; counts per relation; `agree` share of board-known sightings (should sit near 1.0 as in Phase A); push rejections (expect only `human_labeled`). No labels.
2. **Activation day.** Switch on for facility 3 (`MONITORS_MACHINE_LABELS_FACILITIES=3`, `config:cache`), activate the runs of the last 7 days (the feed window) so the feed is consistent, `nursery:machine-labels flush-feed --facility 3`.
3. **Revert drill, same day.** Revert one run; confirm every one of its labels is gone from the feed on the next refresh; re-activate.
4. **Two weeks on.** G_B as in plan §7: manual corrections per day on facility 3 fall against the week-1 baseline; no human report of a wrong machine label; drill done. Optional, pending Q9: 20 uniformly drawn machine-named sightings per week labeled blind, any wrong one pauses activation.

## 2. Deploy order and rollback

| Step | Where | Safe because |
|---|---|---|
| 1 | Apply `person_machine_labels.sql` on the nursery DB; `nursery:check-schema` green | Additive tables; nothing reads them yet |
| 2 | Tag nursery; re-vendor into device backend and app | Resolver sees no active rows and an empty facility list → identical output |
| 3 | Deploy device backend; issue the gpu-ubun token | Routes answer 401 to everyone else |
| 4 | Deploy app with the switch empty | `source` absent, feed unchanged |
| 5 | Nightly runs | Rows inactive |
| 6 | Activate + switch on facility 3 | One env var and one UPDATE to undo |

Rollback at any point: switch empty + `flush-feed` hides every label within one refresh; `revert` per run; the tables stay for audit.

## 3. Deviations from the approved plan, for your sign-off

- **C1** token table in the nursery DDL instead of the device backend (plan §5.4 wording).
- **C8** `EventEscalator` photo back-fill not switched (plan §5.1 lists it).
- **Admin activate/revert** exist on the API as specified, but Phase B uses the artisan path; the worker's `--activate` stays unused until Q6 is decided.
- Weekly fresh adjudication sample in Phase B (plan §7 B.5 does not have one; §8 proposes labels): made optional under the label budget (decision 20), Q9.

## 4. Questions for the owner (answers go into the decision register)

Answered 2026-11-05 (decision register 21-25): Q1 yes; Q5 (a) wait for the app badge, facility 3 included; Q6 manual through the trial, automatic with guards after G_B; Q8 reachable; Q9 20 a week early on, then label-free. Q3: owner recalled the photo as "best so far". Verified 2026-11-05: the stored `face_gallery.photo_url` is written once (at registration, or back-filled when empty: `EventEscalator.php:725-729`, `NurseRosterHelper.php:413-414`) and never replaced by a better crop; only when it is empty does the roster show the best-quality sighting so far, computed on read (`FaceHelper::_resolvePersonPhotos`). C8 kept as the default (machine labels never write the stored photo) unless the owner overrides. Q2, Q4, Q7, Q10, Q11 on the recommended defaults.

| # | Question | My recommendation |
|---|---|---|
| Q1 | `v_machine_workers` in the nursery migration (one connection, one schema guard, `org_id` next to `v_orgs`)? | Yes; amend the contract |
| Q2 | Tokens per GPU host, scoped to one brand (`org_id`); admin scope as a second token kept on the app host only? | Yes. Need facility 3's `org_id` |
| Q3 | Keep the roster-photo back-fill human-only (C8)? | Yes |
| Q4 | Roster cover crop and merge statistics through the resolver (decisions 8–9) even though a wrong label could briefly show the wrong face until revert + cache flush? | Yes, as decided; the flush is part of activate/revert |
| Q5 | Switch on facility 3 before the staff app renders the 「机器复核」 mark? Options: (a) wait for the app; (b) switch on anyway in the test facility, where the reviewers are the four of you | (a) for any real facility; (b) acceptable for facility 3 only if you say so |
| Q6 | After G_B: nightly auto-activation with guards (every named person has references, promotions ≤ a share of the day you set, zero `not_in_roster`/`not_in_run_scope` rejections), or manual forever? | Auto with guards after G_B; manual through B.5 |
| Q7 | On switch-on, activate the last 7 days' runs or only forward? | Last 7 days (the feed window) |
| Q8 | Is the device backend's HTTPS URL reachable from gpu-ubun (the proxy quirk from step 6)? `curl -sS https://<host>/worker/runs/register` should return the envelope's 401 | Please check |
| Q9 | Weekly monitoring labels: 20 blind per week (~10 min) or label-free metrics only? | 20 per week while the rollout is young |
| Q10 | Legal: may a research-licensed model's labels be shown to staff in test facility 3 before T18 is settled? | Ask; build is unaffected |
| Q11 | Who reviews and deploys the alonsolee branches, and when? Steps 1, 2 and 4 of §2 are theirs | Line them up now; B.1 is the long pole |

## 5. Estimates (build; calendar after that is shadow time)

B.0 half a day · B.1 two to three days (resolver, helper, five readers, tests) · B.2 half a day · B.3 one and a half days · B.4 one and a half days · B.5 one week shadow + two weeks gate. Order: B.0 → B.1 and B.4 in parallel → B.3 → B.2 → deploy.

## 6. Risks specific to Phase B

| Risk | Mitigation |
|---|---|
| Feed cache serves stale identities after activate/revert/switch | C5: cache clear is inside activate/revert; `flush-feed` command for the switch; TTL as the backstop |
| The nursery vendoring drifts (two copies, `composer install` reverts) | Same discipline as today: tag, byte-identical copies, `SchemaFidelityTest` |
| Device-backend tests need a deploy-side DB config | Logic tests mock the facade; helper logic is tested in the app's sqlite harness |
| A facility's roster changes between pull and push (merge/delete) | Push validates against the roster at push time; a rejected row is counted, not retried blindly |
| Worker names a sighting a human corrected minutes earlier | Push refuses (`human_labeled`); the resolver puts the human first anyway |
| Two workers for one pair | Lease on `(facility, day)`; expired leases fail the old run |
| Time-zone edge: run before local midnight | `--day auto` uses the facility's zone; the job fires at 00:05 local |
