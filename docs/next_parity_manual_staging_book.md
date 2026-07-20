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
- [ ] FastAPI holds the staging service-role key; neither Vercel nor any `NEXT_PUBLIC_*` variable does.
- [ ] Production write flags remain off; staging enables only the workflow currently under test.
- [ ] Email mode is `dry_run` or `staging_redirect`, with the redirect inbox visibly identified.
- [ ] Required migrations are applied in documented order and their private grants/RLS are verified.
- [ ] Order-27 migration `20260719204700_tournament_operations_guard_surface.sql` and its Operations, official-publish, and email-handoff gates are integrated and verified.
- [ ] Disposable fixture IDs, exact resource refs, route-supported idempotency keys, and cleanup/recovery owners are recorded before any write.
- [ ] Match Log, Replay History, audit log, provider log, and Streamlit fallback links are open before high-risk writes.
- [ ] Public, auth, admin-read, and write waves pass their automated route-specific suites against this exact candidate.

## Stop conditions

Stop the affected wave immediately on wrong-project/auth origin, missing service-role
preflight, stale-state overwrite, missing audit intent, response-loss ambiguity,
unexpected live email, wrong-club visibility, private-field exposure, non-idempotent
retry, or an unavailable recovery path. Record `Blocked` or `Fail`; do not work around
the guard in SQL or the browser.

## Execution waves

1. Public read-only: leaderboards, league results, players, gamification, explorer, recap, rules, FAQ, legal, and roster projections.
2. Public intake/auth: support, corrections/privacy, preferences, verified updates, registration/edit/confirmation, and partner pairing.
3. Admin read/export: printouts, diagnostics/audits, Admin Guide, reports, previews, and CSV/print output.
4. Deferred manual reversible writes: recaps, subscriptions/outbox, league configuration, ladder/session state, tournament setup/registration, and social moderation. These are not generic browser-plan automation.
5. Match/rating writes: the executable workflow automates only the route-specific Tournament Live score command, including dynamic version/fingerprint checks, a distinct idempotency key for each command, reconciliation, and `finally` restoration/readback. Uploader, Match Log, player merge, League Live, Moneyball, ladder, JUPR Live, public live, and Tournament Ops remain manual-only.
6. Deferred manual recovery: perform authoritative route-specific GET readbacks against the exact resource IDs created or changed, record exact 2xx status and positive state projections, inspect audit/completion rows, and prove Match Log/Replay handoffs. Do not invent universal `operation_key` or `evidence_id` response fields that a route does not return.

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
| `public-intake-auth` | Workflow mode `public-intake-auth`; local equivalent uses the same runner | Real staging auth account; read-only intake, registration, and partner-board readiness; no mutation confirmation | `Pending` | — | — |
| `admin-read-export` | Workflow mode `admin-read-export`; local equivalent uses the same runner | Admin tokens, role account, and unpublished recap fixture | `Pending` | — | — |
| `reversible-admin-writes` | Manual-only deferred procedure; no workflow mode | Exact per-page route/resource plan, captured pre-state, truthful inverse, positive write/readback/restore projections, operator and witness | `Pending` | — | — |
| `match-rating-writes` | Workflow mode `match-rating-writes` for Tournament Live only; all other cases manual-only | Tournament Live disposable game/version fixture, dynamic fingerprints, distinct idempotency keys, exact mutation confirmation, and `finally` restore/re-read | `Pending` | — | — |
| `recovery` | Manual-only route-specific reconciliation; no workflow mode | Exact affected resource IDs, authoritative GET routes, JSON-bearing 2xx statuses, positive state/audit projections, Match Log/Replay handoff evidence, all mutation flags off | `Pending` | — | — |

## Deferred manual mutation ledger

Each row below is manual-only in Order 29. Before marking a row `Pass`, replace
all three evidence cells and the operator cell with these exact parseable records:

- Route/pre-state: `Verified: method=<POST|PUT|PATCH|DELETE>; path=<canonical-path>; resource=<id-or-natural-key>; prestate=<captured-baseline>`
- Write/readback: `Verified: status=<JSON-2xx>; projection=<field=value[,field=value]>; artifact=<id-or-url>`
- Inverse/readback: `Verified: method=<POST|PUT|PATCH|DELETE|RETAIN>; path=<canonical-path|N/A>; status=<JSON-2xx|N/A>; projection=<field=value[,field=value]>; artifact=<id-or-url>`
- Sign-off: `operator=<identity>; witness=<different-identity>`

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
| `manual:score-entry` | — | — | — | `Pending` | — |
| `manual:match-uploader` | — | — | — | `Pending` | — |
| `manual:match-log` | — | — | — | `Pending` | — |
| `manual:player-editor` | — | — | — | `Pending` | — |
| `manual:league-live` | — | — | — | `Pending` | — |
| `manual:challenge-ladder` | — | — | — | `Pending` | — |
| `manual:moneyball` | — | — | — | `Pending` | — |
| `manual:jupr-live` | — | — | — | `Pending` | — |
| `manual:public-live` | — | — | — | `Pending` | — |
| `manual:tournament-operations` | — | — | — | `Pending` | — |
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
| `migration:baseline-worker-log` | `supabase/migrations/20260511170000_worker_run_log.sql` | — | — | Stop email workers and close player-update gates | — |
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
| `migration:baseline-registration-player` | `supabase/migrations/20261020000000_tournament_registrations_player_id_postgrest_reload.sql` | — | — | Close tournament registration/admin writes | — |
| `migration:legacy-league-awards-schema` | Reviewed canonical equivalent of `migrations/20260701_league_manager_end_wizard_columns.sql` | — | — | Close League Manager/Awards writes; restore candidate | — |

## Exact feature-flag and disable ledger

Open only the row needed by the active wave. Every production write gate stays
off. Evidence must identify the Fly/Vercel configuration revision and the operator
who closed the gate after the wave. A completed row's evidence cell must be exactly
`Verified disabled: <config evidence>`; production-enabled prose is rejected.

| Ledger key | Exact staging setting during its wave | Production invariant | Disable-first action | Evidence ID / config revision | Owner |
|---|---|---|---|---|---|
| `flag:global` | `JUPR_ENV=staging`; `JUPR_REQUIRE_API_AUDIT_LOG=1`; `JUPR_REQUIRE_WORKER_RUN_LOG=1`; `JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT=1`; `JUPR_ENABLE_NEXT_ADMIN_SHELL=1` | Production write pilot remains `0` | Set write pilot `0`; restore prior Fly release | — | — |
| `flag:public-intake-auth` | `JUPR_REGISTRATION_EDIT_SECRET` and `JUPR_REGISTRATION_CONFIRMATION_SECRET` present only server-side | Secrets absent from Vercel/browser; no production test writes | Close registration/support intake at API routing layer | — | — |
| `flag:admin-read` | `JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS=1`; `JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT=1`; `JUPR_ENABLE_NEXT_ADMIN_TOOLS=1`; `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1`; `JUPR_ENABLE_NEXT_ADMIN_REPLAY=1` | Apply/write gates remain `0` | Close the affected visibility gate | — | — |
| `flag:communications` | `JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP=1`; `JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES=1`; `JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS=1` only for the redirected-email wave | Live-email gate remains `0` | Set auto-email `0`; stop worker; reconcile outbox | — | — |
| `flag:match-player` | `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER=1`; `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1`; `JUPR_ENABLE_NEXT_ADMIN_REPLAY=1`; `JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR=1`; supporting `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1` | All production match-write gates remain `0` | Close uploader/apply/replay/editor gates; reconcile Match Log | — | — |
| `flag:league` | `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1`; `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE=1`; `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN=1`; `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT=1` only in their waves | Awards/Live write gates remain `0` | Close submit, domain, awards, then manager; reconcile rounds | — | — |
| `flag:live-ladder-admin` | Visibility gates `JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER=1`, `JUPR_ENABLE_NEXT_ADMIN_MONEYBALL=1`, `JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE=1`; open only matching `JUPR_ENABLE_STAGING_NEXT_ADMIN_CHALLENGE_LADDER_WRITES=1`, `JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES=1`, or `JUPR_ENABLE_STAGING_NEXT_ADMIN_JUPR_LIVE_WRITES=1` | All three staging-only write flags remain `0` | Close affected staging write flag; reconcile operation ledger | — | — |
| `flag:public-live` | `JUPR_ENABLE_PUBLIC_LIVE_WRITES=1` only for disposable staging sessions | `JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION=0` | Set public-live writes `0`; preserve recovery rows | — | — |
| `flag:tournament-admin` | `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS=1`; individually open `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS=1`, `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS=1`, `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS=1`, `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF=1` | All tournament mutation flags remain `0` | Close import, registration, setup, mutation gates; reconcile ledger | — | — |
| `flag:tournament-ops` | `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS=1`; separately open `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH=1` and `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_EMAIL_HANDOFF=1` only for their isolated sub-waves | All three Tournament Operations gates remain `0` | Close email handoff, official publish, then Operations mutations; reconcile operation ledger | — | — |
| `flag:tournament-live` | `JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES=1` only after Order 27 is resolved | Tournament Live write gate remains `0` | Close Tournament Live gate; reconcile operation ledger | — | — |
| `flag:email-safety` | `JUPR_EMAIL_MODE=dry_run` or `staging_redirect` with `JUPR_STAGING_EMAIL_REDIRECT_TO`; `JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=0` | No unrestricted live delivery | Set auto-send/live-email gates `0`; stop worker; inspect provider log | — | — |

## Disposable fixture and recovery ledger

Every fixed fixture scope must be complete before its first write. Put baseline
export hashes, row IDs, route-supported idempotency/operation keys, audit intent/completion IDs,
outbox/provider IDs, and final authoritative readback in the evidence cell.
For completion, every cleanup cell must use `Verified: <evidence>`.

| Fixture scope | IDs / namespace / deterministic keys | Creation and recovery owner | Cleanup / compensation / retained evidence | Result |
|---|---|---|---|---|
| `fixture:support-intake` | — | — | — | `Pending` |
| `fixture:registration-pairing` | — | — | — | `Pending` |
| `fixture:league-awards-live` | — | — | — | `Pending` |
| `fixture:match-player-replay` | — | — | — | `Pending` |
| `fixture:ladder-moneyball-live` | — | — | — | `Pending` |
| `fixture:tournament-admin-ops-live` | — | — | — | `Pending` |
| `fixture:recap-subscription-outbox` | — | — | — | `Pending` |
| `fixture:auth-role-recovery` | — | — | — | `Pending` |

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
| `match_uploader` | `Pending` | — | — | — |
| `match_log` | `Pending` | — | — | — |
| `player_editor` | `Pending` | — | — | — |
| `admin_tools` | `Pending` | — | — | — |
| `admin_guide` | `Pending` | — | N/A | — |
| `challenge_ladder_admin` | `Pending` | — | — | — |
| `moneyball` | `Pending` | — | — | — |
| `jupr_live` | `Pending` | — | — | — |
| `jupr_live_admin` | `Pending` | — | — | — |
| `tournaments` | `Pending` | — | — | — |
| `tournament_manager` | `Pending` | — | — | — |
| `tournament_ops` | `Pending` | — | — | — |
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

- [ ] All 45 ledger rows are `Pass` for the same candidate identity.
- [ ] All disposable writes are cleaned up or intentionally retained with owner and audit IDs.
- [ ] No unresolved uncertain operation, email delivery, replay, or compensation remains.
- [ ] Supabase security/performance advisors are reviewed after the final migration set.
- [ ] GitHub checks and Vercel preview remain green after evidence-only edits.
- [ ] Production flags, migration order, rollout order, rollback aliases, and on-call owner are recorded.
- [ ] The manually dispatched `complete-book` job passes the complete-book checker with the exact candidate SHA, Vercel deployment ID, immutable Vercel deployment origin, and Fly image ref, then passes identity-only live re-attestation.
- [ ] Only then does the final evidence PR reconcile eligible matrix rows from `Partial` to `Done`.
