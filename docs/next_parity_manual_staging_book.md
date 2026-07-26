# Consolidated Next parity staging book

This is the single operator record for the manual acceptance session that follows
the automated 45-page parity stack. Until this book is completed against the exact
candidate deployment, every covered matrix row remains `Partial` and Streamlit
remains the operational fallback.

The acceptance details for each page are defined in
`docs/next_parity_closure_program.md`. This book records the candidate identity,
fixture/recovery proof, operator result, and evidence ID without duplicating or
weakening those contracts.

## Candidate identity

| Field | Recorded value |
|---|---|
| Application candidate Git SHA | — |
| Final stacked PR | — |
| Vercel preview URL | — |
| Vercel deployment ID | — |
| Vercel immutable deployment origin | — |
| Fly staging image ref | — |
| Deployment identity preflight artifact | — |
| Staging Supabase project ref | `sijpxjxvdtrehmqvirfi` |
| Schema inventory / migration head evidence | — |
| Streamlit fallback URL / build | — |
| Staging role accounts exercised | — |
| Session start / end | — |
| Primary operator | — |
| Witness / reviewer | — |

Keep the formal table above blank until the current `staging` head after this
documentation reconciliation is deployed to both Vercel and Fly and its exact
identity is recorded. Historical checkpoints are preserved separately so they
cannot be mistaken for the final candidate.

### Preserved 2026-07-24 baseline

| Baseline field | Preserved value |
|---|---|
| Runtime candidate Git SHA | `eab384545c493f145af383c8e26d8bf97686ab21` |
| Evidence PR at reconciliation | `#1022`; evidence-only head `daffe213324ae4e5c59f565452a800b89f6004f8` |
| Vercel alias | `https://jupr-git-staging-pickleballattrespalapas1.vercel.app` |
| Vercel deployment / immutable origin | `dpl_FTmDVFSdftMmAj4sMwAHrvQpVYLE` / `https://jupr-bifdisfdg-pickleballattrespalapas1.vercel.app` |
| Fly image | `registry.fly.io/juprleagues-api-staging:deployment-01KY5SVX80SHEZMN7GX37NASRP` |
| Final none / smoke artifacts | Fly restore run `29957218074`; all controlled writes false; canonical Staging Smoke run `29957623653` passed 56 strict browser checks |
| Supabase staging | Project `sijpxjxvdtrehmqvirfi`; `ACTIVE_HEALTHY`; 37 applied migration-ledger entries; head `20260720123402_baseline_worker_run_log` |
| Streamlit fallback | `https://juprtrespalapas.streamlit.app` |
| Checkpoint reconciliation | Read-only deployment, health, migration, log, and advisor review on `2026-07-24` |

### Superseded 2026-07-25 read checkpoint

Candidate `7cedb81ca251023806b0953db996a6e7b80c381a` completed exact-identity
public and authenticated read evidence. A later fixture/recovery audit found
three candidate defects: rated singles could not be deterministically restored,
League Live could not use its required round-robin preview under its own write
waves, and emitted recovery context was not applied by Match Log. The checkpoint
is diagnostic evidence only.

| Checkpoint field | Preserved value |
|---|---|
| Runtime candidate Git SHA | `7cedb81ca251023806b0953db996a6e7b80c381a` |
| Vercel deployment / immutable origin | `dpl_3SJQdBYoUtPyeMmdEFCgmNMPmcX3` / `https://jupr-b79zliik8-pickleballattrespalapas1.vercel.app` |
| Fly none deployment | Run `30166837522`; image `registry.fly.io/juprleagues-api-staging:deployment-01KYD3NVQR5KF9GGMKG34JQ2EF` |
| `public-read` | Run `30167694296`; `56/56`; artifact `8622060773`; digest `sha256:0d8704007ff0a2f41db307ac1abd16a871ce19e25eea9eac0a7a0abdcfd18d1b` |
| `admin-read-export` | Run `30167701340`; `16/16`; artifact `8622073617`; digest `sha256:1ae4f3c4cac382d60e43e56891b2237754a024a96abf9f6a669fb02329ef9e48`; refresh session ended and exported credentials cleared |
| Supabase staging | Project `sijpxjxvdtrehmqvirfi`; 37 migration-ledger entries through `20260720123402_baseline_worker_run_log` |
| Disposition | Superseded by the singles replay, League Live preview-gate, and recovery-context repair; no result may be copied into the formal table |

PR `#1037` landed the authenticated Operations Cockpit, shared Tournament Admin
navigation, richer verified Challenge results, and atomic Challenge publication
and recovery; PR `#1038` landed the canonical-first Moneyball status query.
Populate the formal table only after the resulting current `staging` head is
deployed, using its SHA, PR set, Vercel deployment ID/origin, Fly image, exact
connector migration head, and preflight artifact as one unit. Do not append
final-candidate evidence to either preserved checkpoint table. PR `#1022`
remains the historical candidate-evidence PR; the formal `Final stacked PR`
field must identify the actual freeze/evidence PR.

### Current staging schema checkpoint

Staging Supabase project `sijpxjxvdtrehmqvirfi` has 39 applied
migration-ledger entries through connector-assigned head
`20260726011915_challenge_ladder_public_results`. Post-apply readback verified
the nullable `public_result_json` column and validated constraint, valid
active-claim index, five mutation guards, and six invoker RPCs with fixed
`pg_catalog` search paths and execution granted only to `service_role`; there
were zero active Challenge operations. This schema checkpoint is not
candidate-bound application acceptance and does not fill the formal identity
table.

### Preserved pre-fix diagnostic runs

Candidate `ccf2e469b6ef76cffbbd5525c5b1ff1f5ff503bc` reached exact-identity
public-read execution before the authenticated-session prerequisite was
hardened. These runs are retained for diagnosis only. Later changes altered the
candidate SHA, so none of these results may be copied into the blank formal
candidate table or used to mark a parity row `Pass`.

| Workflow mode | GitHub run | Exact result | Artifact / steady state |
|---|---|---|---|
| `public-read` | `30160267159` | Passed `56/56` expected browser tests; `0` skipped, `0` flaky, and `0` unexpected | Artifact `8620058712`; digest `sha256:6391d4e163e186b04e5069fb33ea7f099dcd79bbc1af4b3d259537163385140a`; candidate-bound diagnostic only |
| `admin-read-export` | `30161862524` | Failed safely during session preparation because the former `STAGING_ADMIN_EMAIL` / `STAGING_ADMIN_PASSWORD` inputs were absent; `0` browser tests and `0` writes | No artifact was uploaded; Fly remained `write_wave=none` |

The failed authenticated run did not exercise login, admin reads, exports, or
fixtures. Its failure is not acceptance evidence, and the successful public-read
artifact remains evidence only for its recorded pre-fix SHA.

If any candidate identity changes during the session, stop and start a new book.
Evidence from different SHAs, Fly releases, Vercel deployments, or Supabase projects
must not be combined into one acceptance result.
Record the identity artifact in the exact form
`candidate=<sha>; vercel=<deployment-id>; fly=<image-ref>; artifact=<run-or-url>`.

## Fail-closed preflight

- [ ] `/api/environment` reports `environment=staging`, `vercel_environment=preview`, the candidate SHA, exact Vercel deployment ID, staging Fly API origin, and staging Supabase origin.
- [ ] Fly `/health` reports the same candidate SHA, exact `FLY_IMAGE_REF`, staging app name, and staging Supabase project ref.
- [ ] Preview data isolation and preview auth isolation are both configured and active.
- [ ] The Vercel automation bypass is available only to the test runner.
- [ ] FastAPI and the server-only authenticated-workflow preparation step hold the staging service-role key; neither Vercel, the browser test, nor any `NEXT_PUBLIC_*` variable does.
- [ ] Production write flags remain off; staging is `write_wave=none` at rest and enables only one approved workflow while it is under test.
- [ ] No "enable all" configuration is used; a distinct final Fly release restores `none`, `business_data_write_wave_active=false`, and every controlled write flag false.
- [ ] Email mode is `dry_run` or `staging_redirect`, with the redirect inbox visibly identified.
- [ ] The reviewed 39-name migration inventory is applied, including `singles_replay_recovery` and `challenge_ladder_public_results`; the connector-assigned ledger head, private grants, and RLS are verified without assuming the repository filename is the remote version.
- [ ] Order-27 migration `20260719204700_tournament_operations_guard_surface.sql` and its Operations, official-publish, and email-handoff gates are integrated and verified.
- [ ] Disposable fixture IDs, exact resource refs, route-supported idempotency keys, and cleanup/recovery owners are recorded before any write.
- [ ] Match Log, Replay History, audit log, provider log, and Streamlit fallback links are open before high-risk writes.
- [ ] The already-applied singles replay migration is formally accepted against the exact candidate; League Live waves expose only the exact Match Uploader preview dependency; context recovery links filter Match Log exactly; and Replay is visibly global.
- [ ] `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES=0` and `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=0` are attested in every current wave. Direct uploader singles, duplicate cleanup, bulk exclusion, and every manual match-producing case that depends on destructive exclusion remain outside the active operator session.
- [ ] Public, auth, admin-read, and write waves pass their automated route-specific suites against this exact candidate.

## Stop conditions

Stop the affected wave immediately on wrong-project/auth origin, missing service-role
preflight, stale-state overwrite, missing audit intent, response-loss ambiguity,
unexpected live email, wrong-club visibility, private-field exposure, non-idempotent
retry, or an unavailable recovery path. Record `Blocked` or `Fail`; do not work around
the guard in SQL or the browser.

## Acceptance session order

1. Non-mutating public/admin read acceptance: public discovery and policy routes,
   authenticated automatic loading, empty/error/retry behavior, diagnostics,
   print/export views, guided Tournament Setup drafting without save, and
   desktop/mobile layout.
2. Bounded write/recovery batches: public intake/auth; registration/pairing;
   recaps and communications; league configuration/live/awards; match/player;
   ladder and one-off live sessions; tournament administration/operations/live;
   and social moderation. Each sub-batch uses exactly one named write wave and
   its own fixture, authoritative readback, recovery, and witness where the
   manual row requires one. Tournament Live score remains the sole executable
   automated mutation; the other writes remain manual-only.
3. Exact-candidate restoration: reconcile every affected resource, deploy a
   same-candidate `write_wave=none` release, attest the all-false controlled-write
   projection and `JUPR_EMAIL_MODE=dry_run`, then run canonical Staging Smoke.
   The final-candidate run must pass the 69-test strict public-read manifest and
   the separate five-test guided Tournament Setup manifest without skips or flakes.

For deferred manual recovery, use the authoritative route-specific GET readback
for the exact changed resource, record the JSON-bearing 2xx status and positive
state projection, inspect audit/completion rows, and prove Match Log/Replay
handoffs. Do not invent universal `operation_key` or `evidence_id` fields that a
route does not return.

Every mutating wave uses the state transition `none -> one named wave -> none`.
Never move directly from one active write wave to another. Record each Fly release
before the first request, perform only the packet's named action, and restore
`none` before any canonical public-read smoke or final evidence run.

### Preserved 2026-07-20/21 intake smoke evidence

The prior exploratory support-intake pass is retained as useful staging evidence,
but it does not by itself mark any formal parity ledger row `Pass` because the
candidate-bound artifact bundle was incomplete.

- Public `general_support` intake created exactly one disposable fixture:
  `req_2baca74d135646e6be38`, subject
  `STAGING SMOKE - support intake 2026-07-20`, requester
  `PCS Staging Smoke 2026-07-20`, email
  `pcs-staging-smoke-20260720@example.invalid`.
- An exact repeat returned the safe deduplication state: “This request was already
  received and remains in the staff review queue.” No second queue row appeared.
- The `support-requests` admin wave loaded one `new` request under an authenticated
  staging-admin session. The operator selected it, used confirmation
  `SAVE REQUEST STATUS`, and dismissed it with note
  `staging smoke only. no customer action. retained as test evidence.`
- Readback showed status `dismissed`, reviewer attribution, and one
  `update_public_support_request_admin` audit event for entity
  `public_support_request` with before `new` and after `dismissed`.
- GitHub run `29795882496` (job `88527006272`) restored the deployed baseline to
  `write_wave=none` with the all-false controlled-write projection.

The operator requested shorter, repeatable fixture values for subsequent intake
and deduplication tests. The next approval packet therefore uses minimal values and
authorizes only a data-correction create/exact retry followed by dismissal.

### Fly write-wave release ledger

This ledger is separate from the formal all-page parity results. A missing old
release reference is recorded as a gap, not reconstructed or silently combined
with the final `none` image.

| Sequence | Candidate | Fly workflow run / image | Wave | Action/readback | Next state |
|---|---|---|---|---|---|
| Prior 1 | `e695365ce508e03a094f528ff9c1179c7f7947de` | Historical wave release ref was not retained in the session record. | `public-intake-auth` | Created `req_2baca74d135646e6be38`; exact retry deduplicated to the same queue item. | `support-requests` only after a separate dispatch |
| Prior 2 | `e695365ce508e03a094f528ff9c1179c7f7947de` | Historical wave release ref was not retained in the session record. | `support-requests` | Dismissed the retained fixture; reviewer and one admin-audit update observed. | `none` |
| Prior 3 | `e695365ce508e03a094f528ff9c1179c7f7947de` | GitHub run `29795882496`; `registry.fly.io/juprleagues-api-staging:deployment-01KY180GGE2V9HC13E8N9VFE3Y`; machine `01KY1821CAK5RC7GE2BWMBQYYM` | `none` | Successful identity readback; business-data write false; all controlled write flags false; email `dry_run`. | Canonical public-read smoke or a separately approved new wave |

For the next candidate, create one row per dispatch with exact run URL, image ref,
machine/release ID, selected wave, approved action, readback/audit IDs, and required
next state. The last row must always be the final same-SHA `none` release.

The manually dispatched `Parity Final Evidence` workflow is the executable suite
manifest. Executable waves require the canonical `origin/staging` SHA, refuse every
non-allowlisted web, API, or auth origin, and verify live Vercel/Fly identity
attestations before any browser request can mutate staging. The workflow requires
the candidate's immutable Vercel deployment origin, verifies that the endpoint
attests the same origin and deployment ID, pins the browser to it, and re-attests
both deployments after the suite. It requires exact mutation confirmation only for the
Tournament Live write wave, writes one JSON report per invocation, and fails on
any missing real spec, skipped, flaky, unexpected, or zero-test result. Generic
JSON mutation/recovery plans are deliberately not executable in Order 29.
The `complete-book` dispatch independently re-attests both live deployments in
identity-only mode by querying the recorded immutable candidate origin directly,
never by discovering the candidate through the mutable staging alias. Its deployed
candidate may trail `origin/staging` only when the sole difference is this completed
evidence book.
Normal pull-request CI runs only the Pending-safe structural checks and never needs
staging credentials.

GitHub lists the workflow from the repository default branch
(`rollback-feb8`). Its checkout and provenance contract still execute the exact
canonical `origin/staging` candidate. A dispatch from either `staging` or
`rollback-feb8` must match that branch's current event SHA; every executable wave
then requires `candidate_sha` to equal `origin/staging`. For authenticated read
and Tournament Live modes, a server-only preparation step uses the staging
service role to require exactly one eligible existing admin assignment bound to
an Auth user. It verifies that identity, calls Supabase Admin `generate_link`,
exchanges the returned token hash directly without sending email, validates the
short-lived session against club-scoped FastAPI capabilities, and discovers only
the allowlisted fixtures. The browser step receives short-lived session material,
never the service-role key, and a mandatory cleanup step fails the job unless the
refreshable session is ended or already inactive. The access JWT must be bound to
the exact issuer, user, session, and authenticated audience with a maximum
one-hour lifetime. Supabase logout ends refreshability but may not invalidate
that JWT before its `exp` claim, so cleanup clears every exported credential and
records that remaining lifetime explicitly. Retained Playwright output redacts
the JWT, operator email, and Vercel bypass value. The workflow does not create or
delete an Auth user, change an admin role assignment, or mutate application
business data. `generate_link`, token verification, and logout intentionally
create and consume bounded staging Auth link-token, session, and audit metadata.
A run can supersede an unused staging magic/recovery link for that same account,
so do not run an automated authenticated wave concurrently with manual
password-recovery or inbox acceptance. No admin email, password, bearer token,
user ID, or fixture ID needs to be stored as a long-lived GitHub setting.

`write_wave=none` attests that every controlled FastAPI business-data write flag
is off; it does not mean Supabase Auth creates no session or audit records during
an authenticated evidence run.

This exact-candidate browser path proves a live, capability-checked Supabase
session and local-scope sign-out. It does not prove operator password entry,
password recovery, or email-inbox delivery; password-form behavior remains under
component/browser contract coverage, while password entry and recovery/inbox
acceptance remain manual.

Configure only the route-specific staging variables named by the workflow.
Tournament Live is the sole automated mutation opt-in. Keep every other mutation
flag off and execute the deferred cases from the page and fixture ledgers below,
recording the exact route, method, JSON-bearing 2xx status, positive response/readback
projection, resource ID, restoration action, post-restoration state, operator, and
artifact ID. Keep fixture tokens and payloads in environment secrets, never in this
committed book.

## Automated wave evidence ledger

| Wave | Exact command / workflow mode | Required input contract | Result | Run URL / artifact IDs | Operator |
|---|---|---|---|---|---|
| `preflight` | `make check-parity-final-evidence` plus production Next build and full focused Python suite | Candidate SHA and clean integrated Order-28 tree | `Pending` | — | — |
| `public-read` | `Parity Final Evidence` workflow mode `public-read`; local equivalent: `python scripts/run_parity_staging_wave.py public-read --candidate-sha <sha> --vercel-deployment-id <id> --vercel-deployment-origin <immutable-origin> --fly-image-ref <ref>` | Exact preview/API/Auth origins, deployment identities, and Vercel bypass | `Pending` | — | — |
| `public-intake-auth` | Workflow mode `public-intake-auth`; local equivalent uses the same runner | Server-only exact-one bound-admin lookup, no-email token-hash exchange, capability validation, and read-only intake/registration/partner-board readiness; no mutation confirmation | `Pending` | — | — |
| `admin-read-export` | Workflow mode `admin-read-export`; local equivalent uses the same runner | Server-only exact-one bound-admin lookup, no-email token-hash exchange, capability validation, exact unpublished recap plus tournament/draw fixture validation, confirmed refresh-session termination, and a maximum one-hour access-JWT lifetime | `Pending` | — | — |
| `reversible-admin-writes` | Manual-only deferred procedure; no workflow mode | Exact per-page route/resource plan, captured pre-state, truthful inverse, positive write/readback/restore projections, named operator, and a separate human witness | `Pending` | — | — |
| `match-rating-writes` | Workflow mode `match-rating-writes` for Tournament Live only; all other cases manual-only | Tournament Live disposable game/version fixture, dynamic fingerprints, distinct idempotency keys, exact mutation confirmation, and `finally` restore/re-read | `Pending` | — | — |
| `recovery` | Manual-only route-specific reconciliation; no workflow mode | Exact affected resource IDs, authoritative GET routes, JSON-bearing 2xx statuses, positive state/audit projections, Match Log/Replay handoff evidence, all mutation flags off | `Pending` | — | — |

## Deferred manual mutation ledger

Each row below is manual-only in Order 29. Before marking a row `Pass`, replace
all three evidence cells and the operator cell with these exact parseable records:

- Route/pre-state: `Verified: method=<POST|PUT|PATCH|DELETE>; path=<canonical-path>; resource=<id-or-natural-key>; prestate=<captured-baseline>`
- Write/readback: `Verified: status=<JSON-2xx>; projection=<field=value[,field=value]>; artifact=<id-or-url>`
- Inverse/readback: `Verified: method=<POST|PUT|PATCH|DELETE|RETAIN>; path=<canonical-path|N/A>; status=<JSON-2xx|N/A>; projection=<field=value[,field=value]>; artifact=<id-or-url>`
- Manual-only staging sign-off: `operator=<identity>; witness=<different-identity>`

The evidence runner records the candidate SHA, deployment identifiers, approved
scope, authoritative readback, audit attribution, and restoration artifacts. The
operator and witness do not transcribe those identifiers. Because the checker has
no candidate-bound automation artifact for an individual manual-only row, it does
not accept `review=automated` for that row. Automated-only wave rows may continue
to use their verified workflow artifact without a manual-row witness.

Use a canonical root-relative route with no query, fragment, encoded characters,
or dot segments. `JSON-2xx` excludes no-content statuses 204 and 205 because the
record must include a positive authoritative JSON projection. If no truthful inverse
exists, use a disposable fixture and `RETAIN` with `path=N/A; status=N/A`, then
record a positive retained-state projection and artifact. Never describe a forward
finalizer as cleanup.

| Manual surface | Exact route / resource / captured pre-state | Write status + positive authoritative readback | Truthful inverse + post-restore evidence | Result | Operator / witness |
|---|---|---|---|---|---|
| `manual:support-intake` | — | — | — | `Pending` | — |
| `manual:data-corrections` | — | — | — | `Pending` | — |
| `manual:email-preferences` | — | — | — | `Pending` | — |
| `manual:profile-privacy` | — | — | — | `Pending` | — |
| `manual:verified-updates` | — | — | — | `Pending` | — |
| `manual:tournament-registration` | — | — | — | `Pending` | — |
| `manual:tournament-partner-pairing` | — | — | — | `Pending` | — |
| `manual:weekly-recap` | — | — | — | `Pending` | — |
| `manual:player-updates` | — | — | — | `Pending` | — |
| `manual:subscription-outbox` | — | — | — | `Pending` | — |
| `manual:league-manager` | — | — | — | `Pending` | — |
| `manual:league-awards` | — | — | — | `Pending` | — |
| `manual:tournament-admin` | — | — | — | `Pending` | — |
| `manual:social-moderation` | — | — | — | `Pending` | — |
| `manual:score-entry` | — | — | — | `Blocked` | — |
| `manual:match-uploader` | — | — | — | `Blocked` | — |
| `manual:match-log` | — | — | — | `Blocked` | — |
| `manual:player-editor` | — | — | — | `Pending` | — |
| `manual:league-live` | — | — | — | `Blocked` | — |
| `manual:challenge-ladder` | — | — | — | `Blocked` | — |
| `manual:moneyball` | — | — | — | `Blocked` | — |
| `manual:jupr-live` | — | — | — | `Blocked` | — |
| `manual:public-live` | — | — | — | `Pending` | — |
| `manual:tournament-operations` | — | — | — | `Blocked` | — |
| `manual:tournament-live-non-score` | — | — | — | `Pending` | — |

## Exact migration and rollback ledger

Apply or verify the files in migration-version order, not PR order. Record the
remote migration-ledger row or schema-contract query for each item. Existing
prerequisites are verification-only unless review proves they are absent. Never
apply a legacy `migrations/` file blindly; record the reviewed canonical equivalent.
For completion, the “Applied or verified” cell must be exactly `Applied` or
`Verified`; prose such as “not applied” or “failed” is never a passing state.

Order 27 uses the uniquely versioned `20260719204700` Tournament Ops guard surface;
`20260719204500` belongs only to Admin Diagnostics. Do not execute Tournament Ops
or Tournament Live writes unless both the Order-27 guard surface and the Order-28
`20260719205000` live-operation migration are present in the remote ledger.

| Ledger key | Exact required file / resolved prerequisite | Applied or verified | Evidence ID / remote version / SQL hash | Disable-first or rollback action | Owner |
|---|---|---|---|---|---|
| `migration:baseline-verified-updates` | `supabase/migrations/20260420000000_verified_player_updates_foundation.sql` | — | — | Close player-update gates; restore candidate release | — |
| `migration:baseline-match-soft-delete` | `supabase/migrations/20260424_matches_soft_delete.sql` | — | — | Close Match Log apply/replay gates; restore candidate release | — |
| `migration:baseline-unsubscribe` | `supabase/migrations/20260428090000_add_unsubscribe_token_to_player_profile_update_subscriptions.sql` | — | — | Close player-update gates; restore candidate release | — |
| `migration:baseline-admin-roles` | `supabase/migrations/20260428100000_admin_role_assignments.sql` | — | — | Close admin shell/write pilot; restore candidate release | — |
| `migration:baseline-admin-audit` | `supabase/migrations/20260428101000_admin_activity_log.sql` | — | — | Stop writes if durable audit is unavailable | — |
| `migration:baseline-replay-jobs` | `supabase/migrations/20260502120000_replay_jobs.sql` | — | — | Close replay/apply gates; preserve unresolved jobs | — |
| `migration:baseline-club-config` | `supabase/migrations/20260502121500_clubs_config.sql` | — | — | Close affected feature gate; restore candidate release | — |
| `migration:baseline-leaderboards` | `supabase/migrations/20260502133000_public_leaderboards_view.sql` | — | — | Restore candidate release; no destructive schema rollback | — |
| `migration:baseline-role-scope` | `supabase/migrations/20260511120000_admin_role_assignments_club_scope.sql` | — | — | Close admin shell/write pilot on scope mismatch | — |
| `migration:baseline-club-onboarding` | `supabase/migrations/20260511143000_clubs_saas_onboarding_fields.sql` | — | — | Close affected feature gate; restore candidate release | — |
| `migration:baseline-confirmations` | `supabase/migrations/20260624000000_confirm_tournament_registrations.sql` | — | — | Close registration intake; preserve saved registrations | — |
| `migration:baseline-live-sessions` | `supabase/migrations/20260702080000_live_sessions.sql` | — | — | Close live write gates; preserve sessions | — |
| `migration:baseline-live-contract` | `supabase/migrations/20260702170000_live_sessions_schema_contract.sql` | — | — | Close live write gates; preserve sessions | — |
| `migration:baseline-match-log-resolution` | `supabase/migrations/20260715165000_admin_match_log_duplicate_resolutions.sql` | — | — | Close Match Log apply; preserve resolutions | — |
| `migration:baseline-selection-guards` | `supabase/migrations/20260717141224_selection_update_transaction_guards.sql` | — | — | Close registration edits; preserve selections | — |
| `migration:baseline-selection-locks` | `supabase/migrations/20260717142402_selection_relationship_update_lock_scope.sql` | — | — | Close registration edits; preserve selections | — |
| `migration:baseline-badge-claims` | `supabase/migrations/20260718141016_badge_eval_queue_atomic_club_claim.sql` | — | — | Stop badge worker/recompute; preserve claims | — |
| `migration:order-02-lockdown` | `supabase/migrations/20260719155515_server_only_data_api_lockdown.sql` | — | — | Close every Next/admin write gate; restore API release | — |
| `migration:order-02-canonicalize` | `supabase/migrations/20260719155737_canonicalize_server_only_tables.sql` | — | — | Close every Next/admin write gate; restore API release | — |
| `migration:order-03-registration-edit` | `supabase/migrations/20260719160821_public_registration_edit_transaction.sql` | — | — | Close registration edits; preserve selections | — |
| `migration:order-13-support-intake` | `supabase/migrations/20260719171000_public_support_intake_guardrails.sql` | — | — | Close support intake/review; preserve requests | — |
| `migration:order-16-replay-idempotency` | `supabase/migrations/20260719172000_replay_job_idempotency.sql` | — | — | Close replay/apply gates; reconcile jobs first | — |
| `migration:order-22-communications` | `supabase/migrations/20260719182606_communications_outbox_stale_guards.sql` | — | — | Stop workers/sends; reconcile `sending` rows first | — |
| `migration:order-20-league-live-domain` | `supabase/migrations/20260719182921_league_live_domain_contract.sql` | — | — | Close League Live domain/submit gates | — |
| `migration:order-21-league-live-submit` | `supabase/migrations/20260719190954_league_live_publish_reconciliation.sql` | — | — | Close submit gate; reconcile rounds before rollback | — |
| `migration:order-17-player-merge` | `supabase/migrations/20260719193000_admin_player_merge_transactions.sql` | — | — | Close Player Editor; recover through Replay History | — |
| `migration:order-15-partner-pairing` | `supabase/migrations/20260719194500_public_partner_pairing_lifecycle.sql` | — | — | Close pairing intake; preserve request history | — |
| `migration:order-24-live-ladder` | `supabase/migrations/20260719201500_live_ladder_admin_operations.sql` | — | — | Close three staging write gates; reconcile ledger | — |
| `migration:order-26-tournament-admin` | `supabase/migrations/20260719203000_tournament_admin_operations.sql` | — | — | Close tournament mutation gates; reconcile ledger | — |
| `migration:order-23-admin-diagnostics` | `supabase/migrations/20260719204500_admin_diagnostics_guarded_operations.sql` | — | — | Close diagnostics apply/tool gates; reconcile ledger | — |
| `migration:order-27-tournament-ops` | `supabase/migrations/20260719204700_tournament_operations_guard_surface.sql` | — | — | Close Tournament Operations mutations; reconcile ledger | — |
| `migration:order-28-tournament-live` | `supabase/migrations/20260719205000_tournament_live_operations.sql` after resolved Order 27 | — | — | Close Tournament Live write gate; reconcile ledger | — |
| `migration:order-25-public-live` | `supabase/migrations/20260719220000_public_live_durability.sql` | — | — | Close public-live writes; preserve recovery records | — |
| `migration:legacy-top-performer-seed` | `supabase/migrations/20260720014744_seed_top_performer_badges.sql`; verify `4/4` IDs, any present `_v2` columns aligned, and existing customized rows preserved | — | — | Close Awards write; do not remove definitions or minted evidence | — |
| `migration:baseline-worker-log` | `supabase/migrations/20260720123402_baseline_worker_run_log.sql` (canonical forward repair for the historical `20260511170000` prerequisite) | — | — | Stop email workers and close player-update gates | — |
| `migration:singles-replay-recovery` | `supabase/migrations/20260725181500_singles_replay_recovery.sql`; verify player baselines, replay-managed match marker, service-role-only bulk update RPC, and atomic Tournament Operations CAS publish preservation | — | — | Keep direct uploader singles and destructive Match Log gates closed; preserve baselines and managed history for a reviewed forward repair | — |
| `migration:challenge-ladder-public-results` | `supabase/migrations/20260725231000_challenge_ladder_public_results.sql`; verify nullable public-result relations, exact-operation receipts, atomic two-match publish, service-role-only grants, RLS, and mutation guards | Applied | Connector ledger `20260726011915_challenge_ladder_public_results`; schema/index/constraint, five-trigger, six-RPC grant/search-path, and zero-active-operation probes verified | Keep Challenge Ladder writes closed; preserve the exact receipt and linked result matches for idempotent recovery | Staging preparation |
| `migration:baseline-registration-player` | `supabase/migrations/20261020000000_tournament_registrations_player_id_postgrest_reload.sql` | — | — | Close tournament registration/admin writes | — |
| `migration:legacy-league-awards-schema` | Reviewed canonical equivalent of `migrations/20260701_league_manager_end_wizard_columns.sql` | — | — | Close League Manager/Awards writes; restore candidate | — |

## Exact feature-flag and disable ledger

Open only the row needed by the active wave. Every production write gate stays
off. Evidence must identify the Fly/Vercel configuration revision and the operator
who closed the gate after the wave. A completed row's evidence cell must be exactly
`Verified disabled: <config evidence>`; production-enabled prose is rejected.
The rows describe available isolated wave projections; they are not additive.
Never open two rows together and never synthesize an "enable all" deployment.

| Ledger key | Exact staging setting during its wave | Production invariant | Disable-first action | Evidence ID / config revision | Owner |
|---|---|---|---|---|---|
| `flag:global` | At rest: `JUPR_ENV=staging`; `JUPR_STAGING_WRITE_WAVE=none`; `JUPR_REQUIRE_API_AUDIT_LOG=1`; `JUPR_REQUIRE_WORKER_RUN_LOG=1`; `JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT=0`; `JUPR_ENABLE_NEXT_ADMIN_SHELL=1`. A selected admin wave may set the pilot to `1` only for that release. The candidate all-false projection contains 32 controlled flags. | Production write pilot remains `0` | Dispatch a distinct `none` release and verify the 32-flag all-false projection | — | — |
| `flag:public-intake-auth` | Only this wave sets `JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES=1`; `JUPR_REGISTRATION_EDIT_SECRET` and `JUPR_REGISTRATION_CONFIRMATION_SECRET` remain server-side | Secrets absent from Vercel/browser; no production test writes | Set public-intake writes `0`, restore `write_wave=none`, and close registration/support intake at API routing layer | — | — |
| `flag:admin-read` | `JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS=1`; `JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT=1`; `JUPR_ENABLE_NEXT_ADMIN_TOOLS=1`; `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1`; `JUPR_ENABLE_NEXT_ADMIN_REPLAY=1` | Apply/write gates remain `0` | Close the affected visibility gate | — | — |
| `flag:communications` | At rest, the read surfaces remain available with `JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP=1`, `JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES=1`, `JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS=0`, and `write_wave=none`. Only the isolated `communications` wave may set `JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS=1`; `JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS=1` is separate and only for the redirected-email wave. | Production read and mutation flags remain `0`; live-email gate remains `0` | Set communications mutations and auto-email to `0`; restore `write_wave=none`; stop worker; reconcile outbox | — | — |
| `flag:match-player` | Only this wave opens the admin pilot plus `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1`, `JUPR_ENABLE_NEXT_ADMIN_REPLAY=1`, `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1`, `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER=1`, `JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR=1`, and `JUPR_ENABLE_STAGING_NEXT_ADMIN_MATCH_CANONICAL_NORMALIZE_WRITES=1`. Both dormant high-risk gates remain `0`; this wave does not authorize direct singles, duplicate cleanup, or bulk exclusion. | All production match-write gates remain `0` | Close uploader/apply/replay/editor/normalize gates; reconcile only independently recoverable actions | — | — |
| `flag:direct-singles` | `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES=0` in `none` and every named staging wave, including `match-player` | Production value remains `0` | Keep forced off until the direct writer is atomic and the migration plus future rated/unrated protocol are accepted | — | — |
| `flag:match-log-destructive` | `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=0` in `none` and every named staging wave, including `match-player` | Production value remains `0` | Keep forced off until duplicate cleanup and bulk exclusion have atomic idempotent recovery and candidate-bound evidence | — | — |
| `flag:league` | Projections are isolated, never additive: `league-manager` opens only the admin pilot plus `JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES`; `league-awards` opens only the pilot plus `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE`; `league-live-domain` opens only the pilot, `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW`, and `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN`; `league-live-submit` additionally opens `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT`. Full `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER` remains off in both Live waves, and only the exact round-robin preview uploader route is allowlisted. | Every production League/Awards/Live/Match Uploader write gate remains `0` | Restore `write_wave=none`; verify every listed gate false; reconcile rounds before closing recovery | — | — |
| `flag:live-ladder-admin` | Visibility gates `JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER=1`, `JUPR_ENABLE_NEXT_ADMIN_MONEYBALL=1`, `JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE=1`; open only matching `JUPR_ENABLE_STAGING_NEXT_ADMIN_CHALLENGE_LADDER_WRITES=1`, `JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES=1`, or `JUPR_ENABLE_STAGING_NEXT_ADMIN_JUPR_LIVE_WRITES=1` | All three staging-only write flags remain `0` | Close affected staging write flag; reconcile operation ledger | — | — |
| `flag:public-live` | `JUPR_ENABLE_PUBLIC_LIVE_WRITES=1` only for disposable staging sessions | `JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION=0` | Set public-live writes `0`; preserve recovery rows | — | — |
| `flag:tournament-admin` | Keep the read surface enabled; individually open only the named `tournament-mutations`, `tournament-setup`, or `tournament-registration` wave and its matching mutation flag. `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF` is reserved and forced off because the current handoff is GET-only. | All tournament mutation flags remain `0`; import handoff remains `0` | Restore `write_wave=none`; close registration, setup, and mutation gates; reconcile ledger | — | — |
| `flag:tournament-ops` | `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS=1`; separately open `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH=1` and `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_EMAIL_HANDOFF=1` only for their isolated sub-waves. The official-singles CAS path is implemented, but destructive Match Log exclusion remains `0`. | All three Tournament Operations gates and both dormant high-risk gates remain `0` | Close email handoff, official publish, then Operations mutations; defer manual publish recovery until destructive exclusion is cleared | — | — |
| `flag:tournament-live` | `tournament-live` opens only the admin pilot plus `JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES`; terminal publish uses the separate `tournament-live-official-publish` projection with the Operations and official-publish dependencies. Move through `none` between them. | Tournament Live, Operations, and official-publish gates remain `0` | Restore `write_wave=none`; close Tournament Live and official-publish gates; reconcile operation and Match Log/Replay ledgers | — | — |
| `flag:email-safety` | `JUPR_EMAIL_MODE=dry_run` or `staging_redirect` with `JUPR_STAGING_EMAIL_REDIRECT_TO`; `JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=0` | No unrestricted live delivery | Set auto-send/live-email gates `0`; stop worker; inspect provider log | — | — |

## Dormant high-risk gate ledger

These are formal completion blockers, not optional follow-up notes. A normal
candidate must attest both gates disabled. The complete-book path must continue
to fail until each blocker is replaced by reviewed enablement evidence from a
later candidate. Do not add a separate direct-singles manual action or copy a
future protocol into Joe's active packet while either row is `Blocked`.

| Blocker key | Forced-off gate | Current status | Future enablement / acceptance evidence |
|---|---|---|---|
| `blocker:direct-uploader-singles` | `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES=0` in every staging wave and production | `Blocked` | Atomic direct writer, formally accepted staging migration, rated and unrated payload/readback equivalence, exact managed IDs, and full-replay restoration are required |
| `blocker:match-log-destructive` | `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=0` in every staging wave and production | `Blocked` | Atomic idempotent duplicate-cleanup and bulk-exclusion recovery, retry/unknown-outcome tests, exact readback, and restored-baseline evidence are required |

Tournament Operations official singles publishing remains automated-ready
through its atomic CAS RPC; it is not the direct-uploader path. Its manual
publish→Match Log→exclude→full-Replay protocol is nevertheless deferred because
the required destructive exclusion gate is still blocked.

## Disposable fixture and recovery ledger

Every fixed fixture scope must be complete before its first write. Put baseline
export hashes, row IDs, route-supported idempotency/operation keys, audit intent/completion IDs,
outbox/provider IDs, and final authoritative readback in the evidence cell.
For completion, every cleanup cell must use `Verified: <evidence>`.

| Fixture scope | IDs / namespace / deterministic keys | Creation and recovery owner | Cleanup / compensation / retained evidence | Result |
|---|---|---|---|---|
| `fixture:support-intake` | No current request ID; use only the short `Test` / `test@x.invalid` / `smoke` / `staging only` payload after the 24-hour fingerprint and hourly-cap precheck | Owner and exact request IDs pending | General/privacy dismissal and no-op correction resolution are retained forward finalizers; audit IDs pending | `Pending` |
| `fixture:registration-pairing` | No approved open event, registration IDs, confirmation/edit tokens, or pairing-board keys recorded | Staging-redirect inbox and owner pending | Registrations/team link are retained; full notification/private-field evidence pending | `Pending` |
| `fixture:league-awards-live` | Readable `Summer Social` / `Spring League` records are not approved write fixtures | Exact disposable league, roster, versions, baseline export, and owner pending | Configuration restore plus Live exclusion/replay/compensation; Awards terminal rows retained | `Pending` |
| `fixture:match-player-replay` | Synthetic player IDs `990001`–`990008` identified; current doubles/singles baselines and versions must be reread | Exact per-row player allocation and recovery owner pending | Direct singles and every exact-ID exclusion/full-Replay case remain future protocols while the dormant gates are off; legacy singles must fail closed; merge compensation pending | `Blocked` |
| `fixture:ladder-moneyball-live` | Synthetic ladder roster using player IDs `990001`–`990008` identified; current ranks/flags/ratings not yet frozen | Per-surface allocation, state versions, operation keys, and owner pending | Context-filtered Match Log, exact-ID exclusion/Replay, full-tier preview/restore, and retained operations pending | `Pending` |
| `fixture:tournament-admin-ops-live` | Read fixtures: tournaments `93000000-0000-4000-8000-000000000001` / `93000000-0000-4000-8000-000000000002`, draw `94000000-0000-4000-8000-000000000001`, game `96000000-0000-4000-8000-000000000001`; none approved for writes; no empty Live draw prepared | Disposable DRAFT/registration/draw fixtures, versions, operation keys, and owner pending | Score inverse, official-publish Match Log/Replay, or explicit retained terminal evidence pending | `Pending` |
| `fixture:recap-subscription-outbox` | Recap week `2099-01-05` exists and must be reloaded for its fresh row version; no isolated active subscription or outbox row is prepared | Redirect inbox, subscription/outbox fixture, and owner pending | Recap unpublish; pending outbox delete; delivery/subscription history retained | `Pending` |
| `fixture:auth-role-recovery` | Automated flow resolves one eligible bound admin without storing identity; manual role/recovery fixtures not recorded | Allowed/denied/wrong-club accounts, recovery inbox, and owner pending | End sessions, clear exported credentials, retain bounded Auth audit metadata | `Pending` |

## Page evidence ledger

Allowed results are `Pending`, `Pass`, `Fail`, or `Blocked`. A `Pass` requires every
automated and manual acceptance item in the closure program, plus verified cleanup
for mutating fixtures. Evidence IDs should point to screenshots, audit IDs, operation
keys, delivery/provider IDs, exported artifacts, or test-run URLs as appropriate.
Recovery cells for mutating pages must use `Verified: <evidence>`; non-mutating
pages must use either that form or exactly `N/A`.

| Streamlit key | Result | Evidence / notes | Recovery verified | Operator |
|---|---|---|---|---|
| `leaderboards` | `Pending` | — | N/A | — |
| `rating_rules` | `Pending` | — | N/A | — |
| `league_results` | `Pending` | — | N/A | — |
| `league_printout` | `Pending` | — | N/A | — |
| `match_explorer` | `Pending` | — | N/A | — |
| `players` | `Pending` | — | N/A | — |
| `badge_codex` | `Pending` | — | N/A | — |
| `badge_debug` | `Pending` | — | N/A | — |
| `badge_audit` | `Pending` | — | N/A | — |
| `match_canonical_audit` | `Pending` | — | — | — |
| `challenge_ladder` | `Pending` | — | N/A | — |
| `faqs` | `Pending` | — | N/A | — |
| `privacy_policy` | `Pending` | — | N/A | — |
| `terms_of_use` | `Pending` | — | N/A | — |
| `contact_support` | `Pending` | — | N/A | — |
| `data_corrections` | `Pending` | — | — | — |
| `email_preferences` | `Pending` | — | — | — |
| `profile_privacy` | `Pending` | — | — | — |
| `league_manager` | `Pending` | — | — | — |
| `match_uploader` | `Blocked` | Dormant direct-singles and destructive-recovery gates prevent full-row acceptance. | — | — |
| `match_log` | `Blocked` | Duplicate cleanup and bulk exclusion await atomic idempotent recovery. | — | — |
| `player_editor` | `Pending` | — | — | — |
| `admin_tools` | `Pending` | — | — | — |
| `admin_guide` | `Pending` | — | N/A | — |
| `challenge_ladder_admin` | `Pending` | — | — | — |
| `moneyball` | `Pending` | — | — | — |
| `jupr_live` | `Pending` | — | — | — |
| `jupr_live_admin` | `Pending` | — | — | — |
| `tournaments` | `Pending` | — | — | — |
| `tournament_manager` | `Pending` | — | — | — |
| `tournament_ops` | `Blocked` | Official singles CAS is automated-ready; manual publish recovery awaits destructive exclusion. | — | — |
| `tournament_live` | `Pending` | — | — | — |
| `tournament_registration` | `Pending` | — | — | — |
| `tournament_registration_admin` | `Pending` | — | — | — |
| `tournament_registration_confirmation` | `Pending` | — | N/A | — |
| `tournament_registration_edit` | `Pending` | — | — | — |
| `tournament_roster` | `Pending` | — | N/A | — |
| `tournament_partner_board` | `Pending` | — | — | — |
| `weekly_recap` | `Pending` | — | N/A | — |
| `top_players_printable` | `Pending` | — | N/A | — |
| `weekly_recap_admin` | `Pending` | — | — | — |
| `player_updates_admin` | `Pending` | — | — | — |
| `admin_login` | `Pending` | — | N/A | — |
| `reset_password` | `Pending` | — | — | — |
| `verified_updates_request` | `Pending` | — | — | — |

## Final reconciliation

This section governs the full administrative replacement and Streamlit
retirement. The earlier Tres Palapas public launch has a separate public-site
checklist and may retain guarded admin rows as `Partial` while Streamlit remains
the fallback. Deferred billing, self-serve onboarding, and a second-club pilot
are later finish lines and do not belong in this completion gate.

- [ ] All 45 ledger rows are `Pass` for the same candidate identity.
- [ ] All disposable writes are cleaned up or intentionally retained with owner and audit IDs.
- [ ] No unresolved uncertain operation, email delivery, replay, or compensation remains.
- [ ] Supabase security/performance advisors are reviewed after the final migration set.
- [ ] GitHub checks and Vercel preview remain green after evidence-only edits.
- [ ] Production flags, migration order, rollout order, rollback aliases, and on-call owner are recorded.
- [ ] The final Fly ledger row is a same-candidate `write_wave=none` release and its `/health` evidence proves `business_data_write_wave_active=false` plus every controlled write flag false.
- [ ] Both dormant high-risk blocker rows are `Cleared` by candidate-bound atomicity, idempotency, migration, recovery, and exact-readback evidence; neither is bypassed by an enabled broader wave.
- [ ] Canonical `Staging Smoke` ran only after that final `none` release and passed
      all 69 public-read plus five guided Tournament Setup checks without skips or
      flakes; `public-web-smoke` evidence is labeled diagnostic/noncanonical.
- [ ] The manually dispatched `complete-book` job passes the complete-book checker with the exact candidate SHA, Vercel deployment ID, immutable Vercel deployment origin, and Fly image ref, then passes identity-only live re-attestation.
- [ ] Only then does the final evidence PR reconcile eligible matrix rows from `Partial` to `Done`.
