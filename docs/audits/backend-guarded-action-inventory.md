# Backend guarded-action inventory

Audit date: 2026-08-24

Audited revision: `dcf08791` baseline, with the guarded-action remediation and forward trigger-function migration working-tree update described below.

Scope: every FastAPI `POST`, `PUT`, `PATCH`, and `DELETE` decorator in `services/api`, its request contract, service-level guard/durability path, Supabase mutation boundary, staging write-wave classification, and likely Next.js consumer.

This document is an evidence inventory, not a claim that every unsafe-method route changes data. Preview, plan, quote, and resolve endpoints deliberately remain in the inventory because the staging middleware classifies by HTTP method, and several “preview” flows persist a reviewed snapshot or audit evidence.

## Executive result

| Measure | Count | Result |
|---|---:|---|
| Unsafe-method FastAPI contracts | **198** | Complete AST inventory |
| `POST` / `PATCH` / `PUT` / `DELETE` | **159 / 30 / 8 / 1** | All enumerated below |
| Admin-prefixed / public contracts | **167 / 31** | Admin routes resolve authenticated club-scoped access; public routes use task-specific tokens, secrets, identity, and/or anti-abuse controls |
| Contracts in `OPEN_WRITE_ROUTES` | **198** | Exact match to the AST inventory |
| Missing from staging write-wave manifest | **0** | Pass |
| Stale/extra manifest entries | **0** | Pass |
| Additional named-wave assignments beyond each route's primary assignment | **12** | Intentional overlap across 10 route contracts is listed below |
| Request contracts carrying a typed-confirmation field | **136** | Post-remediation count; one is a preview request where the field is not required; apply-mode dry runs bypass phrases intentionally |
| Request contracts carrying expected-state/CAS input | **99** | Post-remediation count; nested retained requests are included, and some additional guarded families derive a server-side fingerprint |
| Request contracts carrying an idempotency/operation key | **105** | Post-remediation count; nested retained requests are included, and some tournament operations derive a deterministic server-side operation key |
| Contracts carrying neither direct CAS nor a client replay key | **59** | Post-remediation count; includes server-retained recovery requests, proof-only reconciliation requests, read-only previews/resolvers, and action-level gaps; it does **not** mean 59 unprotected writes |

### Verdict

There is **no externally exposed route that bypasses the global environment/write-wave boundary** in the audited tree. `StagingWriteWaveMiddleware` is fail-closed for every unsafe method, production writes remain disabled unless explicitly enabled, and the checked-in open-wave manifest covers all 198 contracts exactly (`services/api/middleware.py`, `scripts/staging_write_waves.py`, `tests/test_staging_write_wave_guards.py`, `tests/test_permanent_staging_write_mode.py`).

The strongest mutation families already meet most of the interaction standard: live-ladder operations, tournament guarded operations, direct match entry, replay/exclusion recovery, team competition, and public live writes use combinations of compare-and-swap, stable operation identity, durable intent/completion rows, atomic RPCs, audit evidence, and explicit reconciliation.

The remaining release work is concentrated rather than systemic. There are no **P0 boundary bypasses**. The baseline **P1 durability or consequence-confirmation gaps** and **P2 consistency gaps** are called out after the inventory with explicit post-remediation and deferred status.

### No-migration remediation status

The following changes are implemented in the working tree using the existing private `admin_guarded_operations` ledger and existing transactional commerce/generator machinery. No Supabase migration or remote write was performed. “Implemented” below means the safe application-layer guarantee is present; it does **not** claim that a direct table mutation and its later completion audit now share one SQL transaction.

| IDs | Status | Implemented guarantee | Explicit residual / deferred work |
|---|---|---|---|
| `BE-030` / `BE-198` | **Implemented** | Round publish persists its exact request and idempotency key before Match Uploader runs. A zero-evidence `intent`, `publishing`, or `retryable` operation can be retried after reload only through authenticated `RETRY LEAGUE ROUND`; the server reuses the retained request/key and verifies deterministic match contexts before any repeat write. Reconcile remains limited to states with official-match evidence. | No new schema is required. Match publication remains governed by the existing atomic direct-match RPC and League Live publish ledger. |
| `BE-026` / `BE-196` | **Implemented with residual** | Requires a stable idempotency key; records intent before session creation; exact completed retry replays the original result; changed payload/key reuse is rejected; uncertain failure is retained as `recovery_required`; authenticated inspection and proof-only `RECONCILE LIVE SESSION` are available. | Session, court rows, service audit, and ledger completion are not one SQL transaction. A create-session RPC is still the preferred final atomic form and requires a reviewed migration. |
| `BE-050` / `BE-195` | **Implemented with residual** | Requires `SAVE SOCIAL MATCH`, field-level browser-loaded `expected_current`, and an idempotency key; applies CAS filters; exact replay is safe; stale and uncertain outcomes return structured `409` envelopes; authenticated inspection and proof-only `RECONCILE SOCIAL MATCH` are available. | Domain update and completion audit are not one SQL transaction. Required-audit failure still uses CAS rollback plus the durable recovery ledger. A dedicated RPC is deferred. |
| `BE-059` / `BE-193` | **Implemented with residual** | Requires `CREATE PLAYERS`, a reviewed canonical player-list fingerprint, and an idempotency key; missing players are inserted in one multi-row PostgreSQL statement; readback/counts are verified; exact retry replays; uncertain readback can be resolved with proof-only `RECONCILE PLAYER BATCH`. | The player insert statement is atomic, but intent, player insert, domain audit, and ledger completion are separate transactions. A batch RPC with integrated receipt/audit remains deferred. |
| `BE-073` / `BE-074` | **Implemented** | Backend validates `COMPLETE SESSION` / `PUBLISH MATCHES` before operation claim, and the phrase participates in the existing durable request fingerprint. | No schema work remains for this finding. Routine generator mutations `BE-067`–`BE-072` remain documented phrase exceptions unless product classifies them as guarded consequences. |
| `BE-081` / `BE-082` / `BE-083` / `BE-194` | **Implemented with residual** | Create now requires idempotency. Player and league-rating edits return a canonical state fingerprint, require the reviewed fingerprint plus idempotency, use CAS filters, replay an exact completed request before current-state rejection, and expose authenticated inspection plus proof-only `RECONCILE PLAYER OPERATION`. `BE-083` retains `SAVE LEAGUE RATING`. | Domain row, completion audit, and ledger completion are separate transactions. Required-audit failure becomes recovery-required; a player-editor RPC is deferred. |
| `BE-111` / `BE-113` | **Implemented** | Backend now validates `SAVE PAYMENT STATUS` / `SAVE FULFILLMENT STATUS`, and the phrase is included in the existing commerce RPC request fingerprint. | Existing commerce RPC atomicity/idempotency remains authoritative; no migration is needed. |
| `BE-118` / `BE-197` | **Implemented with residual** | Registration import now requires a browser UUID, retains the exact reviewed request in tab-scoped storage, returns structured uncertainty, and exposes authenticated proof-only `RECONCILE REGISTRATION IMPORT`. An exact-request tombstone uses the existing unique client-key constraint to prevent a late original request from mutating after an absent lookup; a competing original `intent` remains locked until it settles. | The atomic team-write RPC and the private guarded-operation/audit ledger are separate transactions. Reconciliation never reruns the import and refuses to close a normal empty `recovery_required` row from a momentary readback; that state needs future commit-fence evidence or manual recovery. |
| `BE-184` | **Partially implemented; abuse control deferred** | Request contract carries a stable key; recipient/tournament/registration is deduplicated in a privacy-safe 15-minute server bucket; a durable delivery intent/receipt suppresses repeat delivery and provider-uncertain resend within that bucket; the public response remains non-enumerating. | Client-IP rate limiting, longer-lived exact-key replay, and provider receipt reconciliation need an approved rate-limit store/schema/provider contract. Provider acceptance and ledger completion are necessarily separate. |

## Audit method and stable IDs

The inventory was generated from Python AST traversal of all route modules beneath `services/api`, then statically traced through request models and imported `jupr_app/services` functions. The original filename/line traversal IDs `BE-001` through `BE-192` are preserved unchanged. Five registered reconciliation contracts are appended as `BE-193` through `BE-197`, followed by the retained League Live publish retry as `BE-198`; future routes must continue appending rather than renumbering existing rows.

The following are deliberately separate concepts:

- **Boundary guard**: global environment and staging-wave admission in `services/api/middleware.py`.
- **Authorization guard**: authenticated admin/club permission, or a public task token/secret/identity control.
- **Consequence confirmation**: server-validated phrase for a high-consequence action. A client dialog is not a security boundary.
- **Concurrency guard**: expected version, timestamp, state, reviewed fingerprint, or equivalent compare-and-swap input.
- **Replay guard**: client idempotency key, deterministic server operation key, or a business-identity uniqueness constraint that makes an exact retry safe.
- **Recovery contract**: durable pending/completed/failed/uncertain state and an exact reconcile/inspect path.
- **Atomicity**: all domain changes and the receipt/audit evidence commit in one SQL transaction/RPC, or an operation ledger makes partial progress recoverable.

The table reports fields visible in the HTTP request contract. A dash in the CAS or idempotency column does not by itself prove that a service is unsafe: the route may be read-only, naturally set-idempotent, or protected by a server-derived operation key. Conversely, merely accepting a key does not prove atomicity; the durable-family analysis below establishes that separately.

## Global write boundary and staging coverage

`services/api/middleware.py:29-84` applies the following policy to all unsafe methods:

1. an explicit runtime environment is required;
2. production writes are rejected unless `JUPR_PRODUCTION_WRITE_POLICY=enabled`;
3. staging writes are rejected unless the exact `(method, route-template)` belongs to the active wave;
4. route templates, not caller-supplied concrete URLs, are compared.

`scripts/staging_write_waves.py` is the authoritative classification. Its `open` manifest contains every audited contract, but automatic and at-rest staging use `none`. `tests/test_staging_write_wave_guards.py` checks the manifest against registered unsafe routes and rejects both omissions and stale entries. `tests/test_permanent_staging_write_mode.py` enforces the fail-closed staging posture. `fly.staging.toml` configures that posture, while the following high-risk capabilities stay independently disabled unless explicitly enabled:

- `JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL`
- `JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION`
- dormant future handoff: `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF`

Ten contracts have more than one named-wave assignment: `BE-025` through `BE-029`, `BE-058`, `BE-114`, `BE-115`, `BE-126`, and `BE-196`. Counting each assignment beyond a contract's first produces the frozen **12 multi-wave overlaps**: `BE-058` and `BE-126` each add two, while the other eight contracts add one each. These overlaps connect league-live domain/submit, match-player preview, Tournament Live official publication, and tournament email handoff. They are not duplicate HTTP routes.

## Guard and durability families

| Family / action IDs | Backend and database evidence | CAS and replay behavior | Audit, atomicity, and recovery | Assessment |
|---|---|---|---|---|
| Generic guarded diagnostics/tools: `BE-001`–`BE-003`, selected `BE-101`–`BE-106` | `jupr_app/services/admin_guarded_write_service.py`; `supabase/migrations/20260719204500_admin_diagnostics_guarded_operations.sql` | Request fingerprint plus unique operation key; exact completed request can replay | Intent is persisted before mutation; ambiguous completion becomes recovery-required; guarded operation RPC is transactional | Strong |
| Live ladder/session operations: `BE-004`–`BE-024`, `BE-064`–`BE-065`, and guarded admin generator operations | `jupr_app/services/admin_live_ladder_operation_service.py`; `supabase/migrations/20260719201500_live_ladder_admin_operations.sql` | Stable idempotency key, expected-version lease, deterministic SHA operation key; unique `(club, surface, operation, entity, idempotency)` | Intent/completion ledger; exact replay and reconcile; recovery directs administrators to Match Log/Replay History when necessary | Strong |
| League Live submit/recovery: `BE-030`–`BE-033`, `BE-196`, `BE-198` | `jupr_app/services/admin_league_live_submit_service.py`, `admin_league_live_service.py`; live-operation tables/RPCs in Supabase migrations | Expected session state plus idempotency for publish/create; explicit operation key on round writes and session-create reconciliation; zero-evidence retry reuses the server-retained request/key | Submit, retained round retry, round reconcile, verified compensation, and proof-only session-create reconcile are distinct guarded actions | Strong; older snapshot writes have lighter replay guarantees |
| League awards/lifecycle: `BE-034`–`BE-047` | `jupr_app/services/admin_league_awards_service.py`, `admin_league_manager_lifecycle_service.py`, `admin_league_manager_update_service.py`, roster services | Awards uses config version, preview/final fingerprints, and idempotency on irreversible stages; basic league/roster updates are phrase-only | Awards workflow records guarded operation evidence; legacy settings/roster paths audit but are not uniformly CAS/replay safe | Mixed |
| Match canonical/log/replay: `BE-048`–`BE-057`, `BE-092`, `BE-195` | `jupr_app/services/admin_match_canonical_audit_service.py`, `admin_match_log_service.py`, exclusion/recovery services; `supabase/migrations/20260719172000_replay_job_idempotency.sql` | Replay and exclusion/recovery use stable keys/expected state; remediated `BE-050` uses reviewed field values, CAS, phrase, and stable operation key | Replay jobs and match-edit operations have uniqueness/RLS; exclusion recovery is explicit; `BE-195` proves the reviewed before or intended after state without rerunning the Club Social edit, but domain/audit SQL atomicity remains deferred | Stronger; `BE-050`/`BE-195` application guard and exact recovery complete with atomic-RPC residual |
| Direct match entry/upload: `BE-058`–`BE-061`, `BE-166`, `BE-193` | `jupr_app/services/direct_match_write_service.py`, `admin_match_uploader_service.py`; `supabase/migrations/20260726222339_atomic_direct_match_entry.sql` | Match submits carry an idempotency key and expected source context; remediated player creation carries reviewed fingerprint and idempotency | One RPC commits match, aggregate updates, receipt, and audit atomically; `BE-059` uses a one-statement player insert plus durable application ledger, and `BE-193` reconciles only from frozen preflight plus authoritative readback | Strong for match writes; `BE-059`/`BE-193` retain an atomic-RPC residual |
| Admin play generators: `BE-066`–`BE-074` | `jupr_app/services/admin_play_generator_service.py` and live-ladder operation ledger | Expected version + idempotency on mutations; stable recovery identity; complete/publish phrases participate in request fingerprints | Durable ledger/reconciliation exists | Strong for `BE-073`/`BE-074`; routine `BE-067`–`BE-072` are documented confirmation exceptions |
| Player editor/social/rating: `BE-075`–`BE-083`, `BE-194` | `jupr_app/services/admin_player_merge_service.py`, `admin_player_editor_service.py`, `admin_player_social_identity_service.py`, `admin_player_league_rating_service.py` | Merge/social link paths are guarded; remediated create/edit/rating paths add idempotency and reviewed-state CAS | Merge has compensation and replay-evidence recovery; `BE-194` resolves uncertain basic player/rating operations only from frozen evidence and authoritative state, while integrated SQL audit atomicity remains deferred | Stronger; `BE-081`–`BE-083`/`BE-194` application guards and exact recovery complete with atomic-RPC residual |
| Communications: `BE-084`–`BE-091`, `BE-158`–`BE-160`, `BE-184`, `BE-192` | `jupr_app/services/admin_communications_service.py`, `admin_weekly_recap_service.py`, public email/edit-link/verification services | Admin batches use operation/fingerprint/expected row state; remediated `BE-184` adds a caller key plus recipient-scoped delivery dedupe | Admin send/retry distinguishes uncertain delivery; `BE-184` records delivery intent/receipt and suppresses uncertain resend within the 15-minute business bucket; IP throttling, longer replay, and provider reconciliation remain deferred | Mixed; `BE-184` duplicate delivery mitigated, abuse-control residual remains |
| Team leagues/competition: `BE-094`–`BE-100`, `BE-144`–`BE-156`, `BE-177`–`BE-178`, `BE-188`–`BE-191` | `jupr_app/services/team_league_service.py`, tournament team competition services and SQL RPCs | Expected versions/fingerprints and idempotency on consequential writes; public invitation/team paths use signed tokens and keys | Result/recovery actions are explicit and operation-ledgered; team competition commits use RPCs | Strong overall |
| Tournament guarded operations: `BE-107`–`BE-157`, `BE-197` (except semantically read-only previews) | `jupr_app/services/admin_tournament_guarded_operation.py`; `jupr_app/services/admin_tournament_registration_import_recovery_service.py`; `supabase/migrations/20260719204700_tournament_operations_guard_surface.sql`; domain-specific tournament services/RPC migrations | Reviewed expected state is checked before and after lock; active scope lock; request fingerprint; client or deterministic server idempotency; `BE-197` reserves an exact-request tombstone before treating an absent operation as not applied | Intent/completion/failure operation rows; preflight; audit; reconcile/recovery; surface-scoped operation table supports `tournament`, `setup`, `registration`, `import_handoff`, `tournament_live`, `operations` | Strong core; completed/stored-result and tombstone recovery never repeat the write; an empty normal recovery remains safely locked pending commit-fence or manual evidence |
| Public Live and public generators: `BE-161`–`BE-165`, `BE-168`–`BE-175` | `jupr_app/services/public_live_write_service.py`, `public_play_generator_service.py` | Edit token + expected version + idempotency on writes; request scope/rate controls | Uses guarded operation paths rather than browser-supplied Supabase authority | Strong for anonymous/public mutation |
| Public intake and tournament registration: `BE-167`, `BE-176`, `BE-179`–`BE-187` | public route modules and registration/pairing/support services | Edit/confirmation tokens and expected selection versions exist where updating; several create/email routes expose no client replay key | Business-key dedupe exists in some services, but the HTTP contract does not consistently preserve an exact request identity | Mixed; retry-sensitive routes require follow-up |

Private operation-ledger tables revoke `public`, `anon`, and `authenticated` access and grant the service role only; RLS/forced RLS provides defense in depth. The service-role credential remains backend-only. This matches the Supabase security model and must not be weakened to make a browser interaction easier.

## Likely frontend ownership

This mapping is based on API client calls and route/component names. It is a starting point for the UI audit; a single backend action can be invoked by more than one panel.

| Action IDs | Likely page/surface | Primary frontend evidence |
|---|---|---|
| `BE-001`–`BE-003` | Admin Badge diagnostics | `apps/web/app/admin/badges/BadgeDiagnosticsPanel.tsx` |
| `BE-004`–`BE-018` | Admin Challenge Ladder | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx` |
| `BE-019`–`BE-024` | Legacy JUPR Live admin | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx` |
| `BE-025`–`BE-033`, `BE-196`, `BE-198` | League Manager live rounds, retained publish retry, and session-create recovery | `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx` |
| `BE-034`–`BE-047` | League Manager settings, roster, lifecycle, awards | `apps/web/app/admin/league-manager/LeagueManagerPanel.tsx`, `apps/web/lib/adminLeagueManagerApi.ts` |
| `BE-048`–`BE-049` | Match canonical audit | `apps/web/app/admin/match-canonical-audit/MatchCanonicalAuditPanel.tsx` |
| `BE-050`–`BE-057`, `BE-195` | Match Log social, apply, exclusion, recovery | `apps/web/app/admin/match-log/MatchLogWorkspace.tsx` and `MatchLog*Panel.tsx` siblings |
| `BE-058`–`BE-061`, `BE-166`, `BE-193` | Match uploader / direct score entry | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx`, `apps/web/lib/adminMatchUploaderApi.ts` |
| `BE-062`–`BE-065` | Moneyball | `apps/web/app/admin/moneyball/MoneyballPanel.tsx` |
| `BE-066`–`BE-074` | Admin Round Robin/Ladder generators | `apps/web/app/admin/play-generators/GeneratorWorkspace.tsx`, `GeneratorRoundRunner.tsx` |
| `BE-075`–`BE-083`, `BE-194` | Player editor, merge, social identity, rating, and recovery | `apps/web/app/admin/players/PlayerEditorPanel.tsx`, `apps/web/lib/adminPlayerEditorApi.ts` |
| `BE-084`–`BE-091`, `BE-157` | Player updates and verified requests | `apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx`, `apps/web/lib/adminPlayerUpdatesApi.ts` |
| `BE-092` | Replay History | `apps/web/app/admin/replay-history/ReplayHistoryForm.tsx`, `apps/web/lib/adminReplayApi.ts` |
| `BE-093` | Support request administration | `apps/web/app/admin/support-requests/SupportRequestsPanel.tsx` |
| `BE-094`–`BE-100` | Team League administration | `apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx` |
| `BE-101`–`BE-106` | Admin Tools | `apps/web/app/admin/tools` |
| `BE-107`–`BE-113` | Tournament Commerce | `apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx` |
| `BE-114`–`BE-115` | Tournament Live | `apps/web/app/admin/tournament-live/TournamentLivePanel.tsx` |
| `BE-116`–`BE-137`, `BE-197` | Tournament draw/results/publish/status, registration editor/bulk, and import recovery | `apps/web/app/admin/tournaments`, including `ops/TournamentOpsPanel.tsx`, `bulk/BulkRegistrationPanel.tsx`, `editor/TournamentRegistrationEditorPanel.tsx`, and `registrations/RegistrationManagementPanel.tsx` |
| `BE-138`–`BE-143` | Tournament Setup wizard | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` and setup dialogs/cards |
| `BE-144`–`BE-156` | Team Tournament Competition | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx` |
| `BE-158`–`BE-160` | Weekly Recap | `apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx` |
| `BE-161`–`BE-165` | Public club live session | `apps/web/app/clubs/[clubSlug]/live` |
| `BE-167` | Email preferences | `apps/web/app/email-preferences/EmailPreferencesPanel.tsx` |
| `BE-168`–`BE-175` | Public Round Robin/Ladder generators | `apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx`, `PublicGeneratorRoundRunner.tsx` |
| `BE-176` | Public support intake | `apps/web/app/support/SupportRequestForm.tsx` |
| `BE-177`–`BE-178` | Public Team League registration/partner confirmation | `apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/TeamLeagueRegistrationForm.tsx` |
| `BE-179`, `BE-184`–`BE-187` | Public tournament registration/quote/edit-link/edit | `apps/web/app/clubs/[clubSlug]/tournament-registration/TournamentRegistrationForm.tsx`, `edit/EditTournamentRegistrationForm.tsx`, `TournamentCommerceChooser.tsx` |
| `BE-180`–`BE-183` | Public tournament Partner Board | `apps/web/app/clubs/[clubSlug]/tournament-partner-board/PairingInterestPanel.tsx` |
| `BE-188`–`BE-191` | Public four-player team registration/invitations | `apps/web/components/tournaments/FourPlayerTeamRegistrationCard.tsx` and tournament registration pages |
| `BE-192` | Verified Updates request | `apps/web/app/verified-updates/VerifiedUpdatesRequestForm.tsx` |

## Complete endpoint inventory

The confirmation column is the phrase enforced by the backend, not the wording of the frontend button. “—” means the request model has no backend phrase requirement. CAS and idempotency columns describe direct request fields; guarded services can add server-derived fingerprints and operation keys as described above.


#### `services/api/admin_badge_diagnostics_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-001 | `PATCH /admin/clubs/{club_id}/badges/{badge_id}/state` — `patch_admin_badge_definition_state` | `UPDATE BADGE STATE` | expected_state | operation_key | badge-diagnostics |
| BE-002 | `POST /admin/clubs/{club_id}/badges/recompute` — `post_admin_badge_recompute` | `RECOMPUTE BADGES` (apply only) | — | operation_key | badge-diagnostics |
| BE-003 | `PATCH /admin/clubs/{club_id}/badges/revoke` — `patch_admin_badge_revoke` | `REVOKE BADGE` | — | operation_key | badge-diagnostics |

#### `services/api/admin_challenge_ladder_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-004 | `POST /admin/clubs/{club_id}/challenge-ladder/challenges` — `post_admin_challenge_ladder_challenge` | `CREATE LADDER CHALLENGE` | expected_version | idempotency_key | challenge-ladder |
| BE-005 | `POST /admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/start-clock` — `post_admin_challenge_ladder_start_clock` | `START LADDER CLOCK` | expected_version | idempotency_key | challenge-ladder |
| BE-006 | `POST /admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/accept` — `post_admin_challenge_ladder_accept` | `ACCEPT LADDER CHALLENGE` | expected_version | idempotency_key | challenge-ladder |
| BE-007 | `POST /admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/forfeit` — `post_admin_challenge_ladder_forfeit` | `RECORD LADDER FORFEIT` | expected_version | idempotency_key | challenge-ladder |
| BE-008 | `POST /admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/pass` — `post_admin_challenge_ladder_pass` | `RECORD LADDER PASS` | expected_version | idempotency_key | challenge-ladder |
| BE-009 | `POST /admin/clubs/{club_id}/challenge-ladder/roster` — `post_admin_challenge_ladder_roster` | `ADD LADDER PLAYER` | expected_version | idempotency_key | challenge-ladder |
| BE-010 | `POST /admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/move` — `post_admin_challenge_ladder_roster_move` | `MOVE LADDER PLAYER` | expected_version | idempotency_key | challenge-ladder |
| BE-011 | `POST /admin/clubs/{club_id}/challenge-ladder/roster/replace-tier/preview` — `post_admin_challenge_ladder_roster_replace_preview` | — | — | — | challenge-ladder |
| BE-012 | `POST /admin/clubs/{club_id}/challenge-ladder/roster/replace-tier` — `post_admin_challenge_ladder_roster_replace` | `REPLACE LADDER TIER` | expected_version | idempotency_key | challenge-ladder |
| BE-013 | `PUT /admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/overrides` — `put_admin_challenge_ladder_player_overrides` | `SAVE LADDER OVERRIDES` | expected_version | idempotency_key | challenge-ladder |
| BE-014 | `PATCH /admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/overrides` — `put_admin_challenge_ladder_player_overrides` | `SAVE LADDER OVERRIDES` | expected_version | idempotency_key | challenge-ladder |
| BE-015 | `POST /admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result` — `post_admin_challenge_ladder_result` | `PUBLISH LADDER RESULT` | expected_version | idempotency_key | challenge-ladder |
| BE-016 | `POST /admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result/preview` — `post_admin_challenge_ladder_result_preview` | — | — | — | challenge-ladder |
| BE-017 | `PATCH /admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}` — `patch_admin_challenge_ladder_challenge` | `SAVE LADDER` | expected_version | idempotency_key | challenge-ladder |
| BE-018 | `POST /admin/clubs/{club_id}/challenge-ladder/operations/{operation_key}/reconcile` — `post_admin_challenge_ladder_reconcile` | `RECONCILE LADDER OPERATION` | — | — | challenge-ladder |

#### `services/api/admin_jupr_live_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-019 | `POST /admin/clubs/{club_id}/jupr-live/sessions` — `post_admin_jupr_live_session` | `CREATE LIVE SESSION` | expected_version | idempotency_key | jupr-live |
| BE-020 | `PATCH /admin/clubs/{club_id}/jupr-live/sessions/{session_key}` — `patch_admin_jupr_live_session` | `SAVE LIVE SESSION` | expected_version | idempotency_key | jupr-live |
| BE-021 | `PATCH /admin/clubs/{club_id}/jupr-live/sessions/{session_key}/scores` — `patch_admin_jupr_live_scores` | `SAVE LIVE SCORES` | expected_version | idempotency_key | jupr-live |
| BE-022 | `POST /admin/clubs/{club_id}/jupr-live/sessions/{session_key}/advance` — `post_admin_jupr_live_advance` | `ADVANCE LIVE ROUND` | expected_version | idempotency_key | jupr-live |
| BE-023 | `POST /admin/clubs/{club_id}/jupr-live/sessions/{session_key}/publish` — `post_admin_jupr_live_publish` | `PUBLISH LIVE MATCHES` | expected_version | idempotency_key | jupr-live |
| BE-024 | `POST /admin/clubs/{club_id}/jupr-live/operations/{operation_key}/reconcile` — `post_admin_jupr_live_reconcile` | `RECONCILE LIVE OPERATION` | — | — | jupr-live |

#### `services/api/admin_league_manager_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-025 | `POST /admin/clubs/{club_id}/league-manager/live/roster-suggestion` — `post_admin_league_live_roster_suggestion` | — | — | — | league-live-domain,league-live-submit |
| BE-026 | `POST /admin/clubs/{club_id}/league-manager/live-sessions` — `post_admin_league_live_session` | `CREATE LIVE SESSION` | — | idempotency_key | league-live-domain,league-live-submit |
| BE-027 | `PATCH /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/snapshot` — `patch_admin_league_live_session_snapshot` | `SAVE SESSION` | expected_updated_at | — | league-live-domain,league-live-submit |
| BE-028 | `POST /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/plan` — `post_admin_league_live_round_plan` | — | expected_updated_at | — | league-live-domain,league-live-submit |
| BE-029 | `POST /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}` — `post_admin_league_live_round` | `SAVE ROUND` | expected_updated_at,expected_operation_key | — | league-live-domain,league-live-submit |
| BE-030 | `POST /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/submit` — `post_admin_league_live_round_publish` | `SUBMIT LEAGUE ROUND` | expected_match_count,expected_updated_at,expected_operation_key | idempotency_key | league-live-submit |
| BE-031 | `POST /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/reconcile` — `post_admin_league_live_round_reconcile` | `RECONCILE LEAGUE ROUND` | — | — | league-live-submit |
| BE-032 | `POST /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/compensate` — `post_admin_league_live_round_compensate` | `VERIFY LEAGUE COMPENSATION` | — | — | league-live-submit |
| BE-033 | `POST /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/guests` — `post_admin_league_live_guest` | `CREATE LIVE GUEST` | expected_updated_at | idempotency_key | league-live-submit |
| BE-034 | `POST /admin/clubs/{club_id}/league-manager/leagues` — `post_admin_league_manager_league` | `CREATE LEAGUE` | — | — | league-manager |
| BE-035 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/duplicate` — `post_admin_league_manager_league_duplicate` | `DUPLICATE LEAGUE` | — | — | league-manager |
| BE-036 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/lifecycle` — `post_admin_league_manager_league_lifecycle` | action-specific `START/PAUSE/RESUME/END/ARCHIVE LEAGUE` | — | — | league-manager |
| BE-037 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/schedule/preview` — `post_admin_league_manager_schedule_preview` | — | — | — | league-manager |
| BE-038 | `PUT /admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/config` — `put_admin_league_awards_config` | — | expected_config_version | — | league-awards |
| BE-039 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/freeze` — `post_admin_league_awards_freeze` | `FREEZE LEAGUE AWARDS` | — | idempotency_key | league-awards |
| BE-040 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/preview` — `post_admin_league_awards_preview` | not required by preview | — | idempotency_key | league-awards |
| BE-041 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/overrides` — `post_admin_league_awards_overrides` | — | — | idempotency_key | league-awards |
| BE-042 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/mint` — `post_admin_league_awards_mint` | `MINT AWARDS` | — | idempotency_key | league-awards |
| BE-043 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/archive` — `post_admin_league_awards_archive` | `ARCHIVE LEAGUE` | — | idempotency_key | league-awards |
| BE-044 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/close` — `post_admin_league_awards_close` | `CLOSE LEAGUE` | — | idempotency_key | league-awards |
| BE-045 | `PATCH /admin/clubs/{club_id}/league-manager/leagues/{league_name}` — `patch_admin_league_manager_settings` | `SAVE LEAGUE` | — | — | league-manager |
| BE-046 | `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/roster/batch` — `post_admin_league_manager_roster_batch` | `SAVE LEAGUE ROSTER BATCH` | — | idempotency_key | league-manager |
| BE-047 | `PATCH /admin/clubs/{club_id}/league-manager/leagues/{league_name}/roster/{player_id}` — `patch_admin_league_manager_roster_membership` | `SAVE ROSTER` | — | — | league-manager |

#### `services/api/admin_match_canonical_audit_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-048 | `POST /admin/clubs/{club_id}/match-canonical-audit/run` — `post_admin_match_canonical_audit` | — | — | — | match-player |
| BE-049 | `POST /admin/clubs/{club_id}/match-canonical-audit/normalize` — `post_admin_match_canonical_normalize` | `APPLY NORMALIZE` (apply only) | — | operation_key | match-player |

#### `services/api/admin_match_log_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-050 | `PATCH /admin/clubs/{club_id}/match-log/social/{social_match_id}` — `patch_admin_match_log_social_row` | `SAVE SOCIAL MATCH` | expected_current | idempotency_key | match-player |
| BE-051 | `POST /admin/clubs/{club_id}/match-log/social/delete` — `post_admin_match_log_social_delete` | `DELETE` | — | — | match-player |
| BE-052 | `PATCH /admin/clubs/{club_id}/match-log/edits` — `patch_admin_match_log_edits` | `APPLY` | — | idempotency_key | match-player |
| BE-053 | `POST /admin/clubs/{club_id}/match-log/edits/{operation_id}/recover` — `post_admin_match_log_edit_recovery` | `RECOVER` | — | — | match-player |
| BE-054 | `POST /admin/clubs/{club_id}/match-log/duplicates/cleanup` — `post_admin_match_log_duplicate_cleanup` | `DELETE` | — | idempotency_key | match-exclusion-recovery |
| BE-055 | `POST /admin/clubs/{club_id}/match-log/exclude` — `post_admin_match_log_exclude_matches` | `DELETE` | — | idempotency_key | match-exclusion-recovery |
| BE-056 | `POST /admin/clubs/{club_id}/match-log/exclusions/{operation_id}/recover` — `post_admin_match_log_exclusion_recovery` | `RECOVER` | — | — | match-exclusion-recovery |
| BE-057 | `POST /admin/clubs/{club_id}/match-log/duplicates/resolve` — `post_admin_match_log_duplicate_resolution` | `NO ISSUE` | — | — | match-player |

#### `services/api/admin_match_uploader_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-058 | `POST /admin/clubs/{club_id}/match-uploader/round-robin/preview` — `post_admin_match_uploader_round_robin_preview` | — | — | — | match-player,league-live-domain,league-live-submit |
| BE-059 | `POST /admin/clubs/{club_id}/match-uploader/players` — `post_admin_match_uploader_players` | `CREATE PLAYERS` | reviewed_fingerprint | idempotency_key | match-player |
| BE-060 | `POST /admin/clubs/{club_id}/match-uploader/singles` — `post_admin_match_uploader_singles` | — | — | idempotency_key | match-player |
| BE-061 | `POST /admin/clubs/{club_id}/match-uploader/batch` — `post_admin_match_uploader_batch` | — | — | idempotency_key | match-player |

#### `services/api/admin_moneyball_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-062 | `POST /admin/clubs/{club_id}/moneyball/preview` — `post_admin_moneyball_preview` | — | — | — | moneyball |
| BE-063 | `POST /admin/clubs/{club_id}/moneyball/settlement` — `post_admin_moneyball_settlement` | — | — | — | moneyball |
| BE-064 | `POST /admin/clubs/{club_id}/moneyball/submit` — `post_admin_moneyball_submit` | `SAVE MONEYBALL` | expected_version | idempotency_key | moneyball |
| BE-065 | `POST /admin/clubs/{club_id}/moneyball/operations/{operation_key}/reconcile` — `post_admin_moneyball_reconcile` | `RECONCILE MONEYBALL` | — | — | moneyball |

#### `services/api/admin_play_generator_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-066 | `POST /admin/clubs/{club_id}/play-generators/preview` — `post_generator_preview` | — | — | — | jupr-live |
| BE-067 | `POST /admin/clubs/{club_id}/play-generators/sessions` — `post_generator_session` | — | expected_version | idempotency_key | jupr-live |
| BE-068 | `PATCH /admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/scores` — `patch_generator_round_scores` | — | expected_version | idempotency_key | jupr-live |
| BE-069 | `POST /admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/played` — `post_generator_round_played` | — | expected_version | idempotency_key | jupr-live |
| BE-070 | `POST /admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/skip` — `post_generator_round_skip` | — | expected_version | idempotency_key | jupr-live |
| BE-071 | `POST /admin/clubs/{club_id}/play-generators/sessions/{session_key}/advance` — `post_generator_advance` | — | expected_version | idempotency_key | jupr-live |
| BE-072 | `POST /admin/clubs/{club_id}/play-generators/sessions/{session_key}/roster` — `post_generator_roster` | — | expected_version | idempotency_key | jupr-live |
| BE-073 | `POST /admin/clubs/{club_id}/play-generators/sessions/{session_key}/complete` — `post_generator_complete` | `COMPLETE SESSION` | expected_version | idempotency_key | jupr-live |
| BE-074 | `POST /admin/clubs/{club_id}/play-generators/sessions/{session_key}/publish` — `post_generator_publish` | `PUBLISH MATCHES` | expected_version | idempotency_key | jupr-live |

#### `services/api/admin_player_editor_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-075 | `POST /admin/clubs/{club_id}/players/editor/merge/preview` — `post_admin_player_merge_preview` | — | — | — | match-player |
| BE-076 | `POST /admin/clubs/{club_id}/players/editor/merge` — `post_admin_player_merge_execute` | `MERGE` | — | — | match-player |
| BE-077 | `POST /admin/clubs/{club_id}/players/editor/merge/{operation_id}/compensate` — `post_admin_player_merge_compensation` | `COMPENSATE MERGE` | — | — | match-player |
| BE-078 | `POST /admin/clubs/{club_id}/players/editor/merge/{operation_id}/replay-evidence` — `post_admin_player_merge_replay_evidence` | `CONFIRM REPLAY RECOVERY` | — | — | match-player |
| BE-079 | `POST /admin/clubs/{club_id}/players/editor/social-identities/auto-link` — `post_admin_player_social_auto_link` | `LINK SOCIAL` | — | — | match-player |
| BE-080 | `PATCH /admin/clubs/{club_id}/players/editor/social-identities/{club_person_id}` — `patch_admin_player_social_identity` | `LINK SOCIAL` | — | — | match-player |
| BE-081 | `POST /admin/clubs/{club_id}/players/editor/players` — `post_admin_player_editor_player` | — (routine Create) | — | idempotency_key | match-player |
| BE-082 | `PATCH /admin/clubs/{club_id}/players/editor/players/{player_id}` — `patch_admin_player_editor_player` | — (routine Edit) | expected_state_fingerprint | idempotency_key | match-player |
| BE-083 | `PATCH /admin/clubs/{club_id}/players/editor/players/{player_id}/league-ratings/{league_rating_id}` — `patch_admin_player_editor_league_rating` | `SAVE LEAGUE RATING` | expected_state_fingerprint | idempotency_key | match-player |

#### `services/api/admin_player_updates_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-084 | `POST /admin/clubs/{club_id}/player-updates/send-range` — `post_admin_player_updates_send_range` | `SEND PLAYER UPDATES` | — | — | communications |
| BE-085 | `POST /admin/clubs/{club_id}/player-updates/digests/preview` — `post_admin_player_digest_preview` | — | — | — | communications |
| BE-086 | `POST /admin/clubs/{club_id}/player-updates/digests/queue` — `post_admin_player_digest_queue` | `QUEUE PLAYER UPDATES` | — | operation_key | communications |
| BE-087 | `POST /admin/clubs/{club_id}/player-updates/outbox/send` — `post_admin_player_updates_outbox_send` | `SEND PLAYER UPDATES` | — | operation_key | communications |
| BE-088 | `POST /admin/clubs/{club_id}/player-updates/outbox/retry` — `post_admin_player_updates_outbox_retry` | `RETRY PLAYER UPDATES` / `RETRY UNCERTAIN EMAILS` | — | operation_key | communications |
| BE-089 | `POST /admin/clubs/{club_id}/player-updates/outbox/delete` — `post_admin_player_updates_outbox_delete` | `DELETE QUEUED UPDATES` | — | operation_key | communications |
| BE-090 | `POST /admin/clubs/{club_id}/player-updates/subscriptions/{subscription_id}/replace` — `post_admin_player_updates_subscription_replace` | `REPLACE VERIFIED SUBSCRIBER` | expected_row_version | operation_key | communications |
| BE-091 | `POST /admin/clubs/{club_id}/player-updates/subscriptions/{subscription_id}/deactivate` — `post_admin_player_updates_subscription_deactivate` | `UNSUBSCRIBE VERIFIED SUBSCRIBER` | expected_row_version | — | communications |

#### `services/api/admin_replay_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-092 | `POST /admin/clubs/{club_id}/replay-history` — `post_admin_replay_history` | `REPLAY` | — | idempotency_key | match-player |

#### `services/api/admin_support_requests_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-093 | `PATCH /admin/clubs/{club_id}/support-requests/{request_id}` — `patch_admin_support_request` | `SAVE REQUEST STATUS` | expected_updated_at | — | support-requests |

#### `services/api/admin_team_league_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-094 | `PUT /admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/settings` — `put_admin_team_league_settings` | `SAVE TEAM LEAGUE` | expected_settings_version | idempotency_key | league-manager |
| BE-095 | `POST /admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/schedule-preview/{phase}` — `post_admin_team_league_schedule_preview` | — | — | — | league-manager |
| BE-096 | `POST /admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/schedule` — `post_admin_team_league_schedule` | `PUBLISH TEAM LEAGUE SCHEDULE` / `... PLAYOFFS` | expected_schedule_version,expected_standings_version,expected_roster_version | idempotency_key | league-manager |
| BE-097 | `POST /admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/waitlist-actions` — `post_admin_team_league_waitlist_action` | `PAIR WAITLIST PLAYERS` / `WITHDRAW WAITLIST PLAYERS` | — | idempotency_key | league-manager |
| BE-098 | `POST /admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/fixtures/{fixture_id}/score` — `post_admin_team_league_fixture_score` | `SAVE TEAM LEAGUE RESULT` / `... FORFEIT` | — | idempotency_key | league-manager |
| BE-099 | `POST /admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/fixtures/{fixture_id}/reconcile` — `post_admin_team_league_fixture_reconcile` | `RECONCILE TEAM LEAGUE RESULT` | — | idempotency_key | league-manager |
| BE-100 | `POST /admin/clubs/{club_id}/league-manager/team-leagues/operations/{operation_id}/resolve` — `post_admin_team_league_operation_resolution` | `FINALIZE TEAM LEAGUE RECOVERY` / `COMPENSATE ...` | — | — | league-manager |

#### `services/api/admin_tools_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-101 | `POST /admin/clubs/{club_id}/tools/social-submissions/{event_id}/moderate` — `post_admin_tools_social_submission_moderation` | `APPROVE SOCIAL SUBMISSION` / `REJECT ...` | expected_status | operation_key | admin-tools |
| BE-102 | `POST /admin/clubs/{club_id}/tools/backfills/tournament-matches/apply` — `post_admin_tools_tournament_match_backfill_apply` | `BACKFILL TOURNAMENT MATCHES` | — | operation_key | admin-tools |
| BE-103 | `PATCH /admin/clubs/{club_id}/tools/roles` — `patch_admin_tools_role_assignment` | `SAVE ROLE` / `REVOKE ROLE` | — | operation_key | admin-tools |
| BE-104 | `POST /admin/clubs/{club_id}/tools/workers/badge-queue` — `post_admin_tools_badge_queue_worker` | `PROCESS BADGE QUEUE` / `DRAIN BADGE QUEUE` | — | operation_key | admin-tools |
| BE-105 | `POST /admin/clubs/{club_id}/tools/workers/badge-recompute` — `post_admin_tools_badge_recompute` | `RUN BADGE RECOMPUTE` (apply only) | — | operation_key | admin-tools |
| BE-106 | `POST /admin/clubs/{club_id}/tools/backfills/tournament-matches/operations/{operation_key}/recover` — `post_admin_tools_tournament_match_backfill_recovery` | `RECOVER TOURNAMENT BACKFILL` | — | — | admin-tools |

#### `services/api/admin_tournament_commerce_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-107 | `POST /admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/catalog/preview` — `preview_tournament_commerce_catalog` | — | — | — | tournament-commerce-admin |
| BE-108 | `PUT /admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/catalog` — `put_tournament_commerce_catalog` | `SAVE` | expected_catalog_fingerprint | idempotency_key | tournament-commerce-admin |
| BE-109 | `POST /admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/orders/{registration_id}/quote` — `post_tournament_commerce_order_quote` | — | — | — | tournament-commerce-admin |
| BE-110 | `PUT /admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/orders/{registration_id}` — `put_tournament_commerce_order` | `SAVE EXTRAS` | expected_quote_fingerprint,expected_order_updated_at | idempotency_key | tournament-commerce-admin |
| BE-111 | `PATCH /admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/orders/{registration_id}/payment` — `patch_tournament_commerce_payment` | `SAVE PAYMENT STATUS` | expected_order_updated_at | idempotency_key | tournament-commerce-admin |
| BE-112 | `POST /admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/orders/{registration_id}/cancel` — `post_tournament_commerce_order_cancel` | `CANCEL` | expected_order_updated_at | idempotency_key | tournament-commerce-admin |
| BE-113 | `PATCH /admin/clubs/{club_id}/tournaments/commerce/tournaments/{tournament_id}/fulfillment/{fulfillment_id}` — `patch_tournament_commerce_fulfillment` | `SAVE FULFILLMENT STATUS` | expected_updated_at | idempotency_key | tournament-commerce-admin |

#### `services/api/admin_tournament_live_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-114 | `POST /admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/commands` — `post_admin_tournament_live_command` | command-specific `SAVE SCORE`, `GENERATE GAMES/PLAYOFFS/PODIUM`, `AWARD PODIUM`, `PUBLISH MATCHES` | expected_state_fingerprint,expected_draw_updated_at,expected_game_updated_at,expected_team_versions,expected_source_game_versions | idempotency_key | tournament-live,tournament-live-official-publish |
| BE-115 | `POST /admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/operations/{operation_key}/reconcile` — `post_admin_tournament_live_reconcile` | `RECONCILE TOURNAMENT LIVE` | — | — | tournament-live,tournament-live-official-publish |

#### `services/api/admin_tournament_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-116 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws` — `post_admin_tournament_draw` | `CREATE DRAW` | expected_state_fingerprint | — | tournament-operations |
| BE-117 | `PUT /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams` — `put_admin_tournament_draw_teams` | `SAVE TEAMS` | expected_state_fingerprint,expected_draw_updated_at | — | tournament-operations |
| BE-118 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-registrations` — `post_admin_tournament_registration_team_import` | `IMPORT REGISTRATIONS` | expected_state_fingerprint,expected_draw_updated_at | idempotency_key | tournament-operations |
| BE-119 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-bulk` — `post_admin_tournament_bulk_team_import` | `IMPORT TEAMS` | expected_state_fingerprint,expected_draw_updated_at | — | tournament-operations |
| BE-120 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/round-robin` — `post_admin_tournament_round_robin_games` | `GENERATE GAMES` | expected_state_fingerprint,expected_draw_updated_at,expected_team_versions | — | tournament-operations |
| BE-121 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/results-import/preview` — `post_admin_tournament_results_import_preview` | — | — | — | tournament-operations |
| BE-122 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/results-import/commit` — `post_admin_tournament_results_import_commit` | `REPLACE RESULTS` / `IMPORT RESULTS` | expected_review_fingerprint,expected_state_fingerprint,expected_draw_updated_at | — | tournament-operations |
| BE-123 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/playoffs` — `post_admin_tournament_playoff_games` | `GENERATE PLAYOFFS` | expected_state_fingerprint,expected_draw_updated_at,expected_team_versions,expected_source_game_versions | — | tournament-operations |
| BE-124 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/podium` — `post_admin_tournament_draw_podium` | `GENERATE PODIUM` | expected_state_fingerprint,expected_draw_updated_at,expected_team_versions,expected_source_game_versions | — | tournament-operations |
| BE-125 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/podium/awards` — `post_admin_tournament_draw_podium_awards` | `AWARD PODIUM` | expected_state_fingerprint | — | tournament-operations |
| BE-126 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/matches/publish` — `post_admin_tournament_draw_matches_publish` | `PUBLISH MATCHES` | expected_state_fingerprint | — | tournament-official-publish,tournament-email-handoff,tournament-live-official-publish |
| BE-127 | `PATCH /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/games/{game_id}/score` — `patch_admin_tournament_game_score` | `SAVE SCORE` | expected_state_fingerprint,expected_game_updated_at,expected_draw_updated_at | — | tournament-operations |
| BE-128 | `PATCH /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/status-action` — `patch_admin_tournament_status_action` | `ARCHIVE` / `UNARCHIVE` | expected_updated_at | — | tournament-mutations |
| BE-129 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/delete-draft` — `post_admin_tournament_delete_draft` | `DELETE DRAFT` | expected_updated_at | — | tournament-mutations |
| BE-130 | `PATCH /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}` — `patch_admin_tournament` | `SAVE TOURNAMENT` | expected_updated_at | — | tournament-mutations |
| BE-131 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/broadcast-preview` — `post_admin_tournament_broadcast_preview` | — | — | — | tournament-registration |
| BE-132 | `PATCH /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/bulk` — `patch_admin_tournament_registrations_bulk` | `BULK UPDATE REGISTRATIONS` | expected_state_fingerprint,expected_versions | — | tournament-registration |
| BE-133 | `PATCH /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/{registration_id}` — `patch_admin_tournament_registration` | `SAVE REGISTRATION` | expected_updated_at | — | tournament-registration |
| BE-134 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/{registration_id}/selections` — `post_admin_tournament_selection` | `SAVE SELECTION` | expected_state_fingerprint | — | tournament-registration |
| BE-135 | `PATCH /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}` — `patch_admin_tournament_selection` | `SAVE SELECTION` | expected_updated_at | — | tournament-registration |
| BE-136 | `DELETE /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}` — `delete_admin_tournament_selection_route` | `REMOVE SELECTION` | expected_updated_at | — | tournament-registration |
| BE-137 | `PUT /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}/partner` — `put_admin_tournament_selection_partner` | `SAVE PARTNER` | expected_updated_at | — | tournament-registration |

#### `services/api/admin_tournament_setup_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-138 | `POST /admin/clubs/{club_id}/tournaments/setup/tournaments` — `post_create_tournament_shell` | `CREATE TOURNAMENT` | — | idempotency_key | tournament-setup |
| BE-139 | `POST /admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/age-split-preview` — `post_age_split_preview` | — | — | — | tournament-setup |
| BE-140 | `POST /admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/impact` — `post_setup_impact` | — | expected_state_fingerprint | — | tournament-setup |
| BE-141 | `PATCH /admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/settings` — `patch_setup_settings` | `SAVE SETUP` | expected_state_fingerprint | — | tournament-setup |
| BE-142 | `PUT /admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/draft` — `put_setup_draft` | `SAVE SETUP DRAFT` | expected_state_fingerprint | — | tournament-setup |
| BE-143 | `POST /admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/publish` — `post_setup_publish` | `PUBLISH SETUP` | expected_state_fingerprint | — | tournament-setup |

#### `services/api/admin_tournament_team_competition_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-144 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/events/{event_option_id}/config` — `config` | `SAVE COMPETITION` | expected_updated_at | idempotency_key | tournament-setup |
| BE-145 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/rating-verifications` — `verification` | `VERIFY RATING` | expected_version | idempotency_key | tournament-registration |
| BE-146 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/rating-reviews` — `review` | `SAVE RATING REVIEW` | expected_selection_updated_at | idempotency_key | tournament-registration |
| BE-147 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/rating-reviews/close` — `close_reviews` | `CLOSE RATING REVIEW` | — | idempotency_key | tournament-registration |
| BE-148 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/teams` — `create_team` | `CREATE TEAM` | — | idempotency_key | tournament-registration |
| BE-149 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/teams/{team_id}/invitations/reissue` — `reissue` | `REISSUE INVITATION` | expected_invitation_version | idempotency_key | tournament-registration |
| BE-150 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/teams/{team_id}/roster` — `roster` | `WITHDRAW TEAM` / `REPLACE ROSTER` | expected_team_version | idempotency_key | tournament-registration |
| BE-151 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/draws/{draw_id}/round-robin` — `round_robin` | `BUILD TEAM SCHEDULE` | expected_draw_updated_at | idempotency_key | tournament-operations |
| BE-152 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/draws/{draw_id}/playoffs` — `playoffs` | `BUILD TEAM PLAYOFFS` | expected_draw_updated_at | idempotency_key | tournament-operations |
| BE-153 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/matchups/{matchup_id}/lineups` — `lineup` | `LOCK TEAM LINEUP` | expected_matchup_version,expected_lineup_version | idempotency_key | tournament-operations |
| BE-154 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/games/{match_game_id}/score` — `score` | `SAVE TEAM SCORE` | expected_game_version,expected_matchup_version | idempotency_key | tournament-operations |
| BE-155 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/games/{match_game_id}/reconcile` — `reconcile` | `RECONCILE TEAM SCORE` | expected_official_row_version,expected_game_version,expected_matchup_version | idempotency_key | tournament-operations |
| BE-156 | `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/team-competition/draws/{draw_id}/podium` — `podium` | `PUBLISH TEAM PODIUM` / `SAVE TEAM PODIUM` | expected_draw_updated_at | idempotency_key | tournament-operations |

#### `services/api/admin_verified_updates_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-157 | `PATCH /admin/clubs/{club_id}/verified-updates/requests/{subscription_id}` — `patch_admin_verified_update_request` | `SAVE VERIFIED REQUEST` | — | — | communications |

#### `services/api/admin_weekly_recap_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-158 | `POST /admin/clubs/{club_id}/weekly-recap/generate` — `post_weekly_recap_generate` | `GENERATE RECAP` | expected_row_version | — | communications |
| BE-159 | `PATCH /admin/clubs/{club_id}/weekly-recap/recaps/{week_start}` — `patch_weekly_recap_save` | `SAVE RECAP` | expected_row_version | — | communications |
| BE-160 | `POST /admin/clubs/{club_id}/weekly-recap/recaps/{week_start}/publish` — `post_weekly_recap_publish` | `PUBLISH RECAP` / `UNPUBLISH RECAP` | expected_row_version | — | communications |

#### `services/api/main.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-161 | `POST /clubs/{club_slug}/live-sessions` — `create_club_live_session` | — | — | idempotency_key | public-live |
| BE-162 | `PATCH /clubs/{club_slug}/live-sessions/{session_key}/scores` — `update_club_live_session_scores` | — | expected_version | idempotency_key | public-live |
| BE-163 | `POST /clubs/{club_slug}/live-sessions/{session_key}/advance` — `advance_club_live_session` | — | expected_version | idempotency_key | public-live |
| BE-164 | `POST /clubs/{club_slug}/live-sessions/{session_key}/substitutions` — `substitute_club_live_session` | — | expected_version | idempotency_key | public-live |
| BE-165 | `POST /clubs/{club_slug}/live-sessions/{session_key}/complete` — `complete_club_live_session` | — | expected_version | idempotency_key | public-live |
| BE-166 | `POST /admin/clubs/{club_id}/matches/batch` — `submit_admin_match_batch` | — | — | idempotency_key | match-player |

#### `services/api/public_email_preferences_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-167 | `POST /email-preferences/unsubscribe` — `post_public_email_unsubscribe` | — | — | — | public-intake-auth |

#### `services/api/public_play_generator_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-168 | `POST /clubs/{club_slug}/play-generators/preview` — `post_public_generator_preview` | — | — | — | public-live |
| BE-169 | `POST /clubs/{club_slug}/play-generators/sessions` — `post_public_generator_session` | — | — | idempotency_key | public-live |
| BE-170 | `PATCH /clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/scores` — `patch_public_generator_round_scores` | — | expected_version | idempotency_key | public-live |
| BE-171 | `POST /clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/played` — `post_public_generator_round_played` | — | expected_version | idempotency_key | public-live |
| BE-172 | `POST /clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/skip` — `post_public_generator_round_skip` | — | expected_version | idempotency_key | public-live |
| BE-173 | `POST /clubs/{club_slug}/play-generators/sessions/{session_key}/advance` — `post_public_generator_advance` | — | expected_version | idempotency_key | public-live |
| BE-174 | `POST /clubs/{club_slug}/play-generators/sessions/{session_key}/roster` — `post_public_generator_roster` | — | expected_version | idempotency_key | public-live |
| BE-175 | `POST /clubs/{club_slug}/play-generators/sessions/{session_key}/complete` — `post_public_generator_complete` | — | expected_version | idempotency_key | public-live |

#### `services/api/public_support_intake_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-176 | `POST /clubs/{club_slug}/support/intake` — `post_club_public_support_intake` | — | — | — | public-intake-auth |

#### `services/api/public_team_league_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-177 | `POST /clubs/{club_slug}/team-leagues/{league_name}/registrations` — `post_public_team_league_registration` | `REGISTER TEAM` / `JOIN PARTNER WAITLIST` | — | idempotency_key | public-intake-auth |
| BE-178 | `POST /clubs/{club_slug}/team-leagues/partner-confirmations` — `post_public_team_league_partner_confirmation` | — | — | idempotency_key | public-intake-auth |

#### `services/api/public_tournament_commerce_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-179 | `POST /clubs/{club_slug}/tournament-commerce/quote` — `post_tournament_commerce_quote` | — | — | — | public-intake-auth |

#### `services/api/public_tournament_pairing_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-180 | `POST /clubs/{club_slug}/tournament-registration/pairing-requests/{partner_request_id}/accept` — `accept_club_tournament_pairing_request` | — | — | — | public-intake-auth |
| BE-181 | `POST /clubs/{club_slug}/tournament-registration/pairing-requests/{partner_request_id}/decline` — `decline_club_tournament_pairing_request` | — | — | — | public-intake-auth |
| BE-182 | `POST /clubs/{club_slug}/tournament-registration/pairing-requests/{partner_request_id}/cancel` — `cancel_club_tournament_pairing_request` | — | — | — | public-intake-auth |
| BE-183 | `POST /clubs/{club_slug}/tournament-registration/pairing-interest` — `create_club_tournament_pairing_interest` | — | — | — | public-intake-auth |

#### `services/api/public_tournament_registration_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-184 | `POST /clubs/{club_slug}/tournament-registration/edit-link/request` — `request_club_tournament_registration_edit_link` | — (public non-enumerating request) | recipient/tournament delivery bucket (server-derived) | idempotency_key | public-intake-auth |
| BE-185 | `POST /clubs/{club_slug}/tournament-registration/profile-resolution` — `resolve_club_tournament_registration_profile` | — | — | — | public-intake-auth |
| BE-186 | `POST /clubs/{club_slug}/tournament-registration/edit` — `submit_club_tournament_registration_edit` | — | expected_updated_at,expected_selection_versions | — | public-intake-auth |
| BE-187 | `POST /clubs/{club_slug}/tournament-registration` — `submit_club_tournament_registration` | — | — | — | public-intake-auth |

#### `services/api/public_tournament_team_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-188 | `POST /clubs/{club_slug}/tournament-registration/four-player-team` — `create_team` | — | — | idempotency_key | public-intake-auth |
| BE-189 | `POST /clubs/{club_slug}/tournament-registration/four-player-team/recover` — `recover_team_setup` | — | — | — | public-intake-auth |
| BE-190 | `POST /clubs/{club_slug}/tournament-team-invitation/resolve` — `resolve_invitation` | — | — | — | public-intake-auth |
| BE-191 | `POST /clubs/{club_slug}/tournament-team-invitation/respond` — `respond_invitation` | — | — | idempotency_key | public-intake-auth |

#### `services/api/public_verified_updates_routes.py`

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-192 | `POST /clubs/{club_slug}/verified-updates/request` — `post_verified_updates_request` | — | — | — | public-intake-auth |

#### Appended authenticated recovery contracts

These IDs are deliberately appended after the original `BE-001`–`BE-192` inventory. `BE-193`–`BE-197` inspect frozen operation evidence and authoritative stored state without rerunning the original mutation. `BE-198` is a distinct zero-evidence retry: it replays only the exact server-retained League Live request and idempotency key after verifying that the durable state is retryable.

| ID | Route / surface | Backend confirmation phrase | CAS / reviewed state | Idempotency | Staging wave |
|---|---|---|---|---|---|
| BE-193 | Match Uploader — `POST /admin/clubs/{club_id}/match-uploader/player-operations/{operation_key}/reconcile` — `post_admin_match_uploader_player_batch_reconcile` | `RECONCILE PLAYER BATCH` | — | — | match-player |
| BE-194 | Player Editor — `POST /admin/clubs/{club_id}/players/editor/operations/{operation_key}/reconcile` — `post_admin_player_editor_operation_reconcile` | `RECONCILE PLAYER OPERATION` | — | — | match-player |
| BE-195 | Club Social — `POST /admin/clubs/{club_id}/match-log/social/operations/{operation_key}/reconcile` — `post_admin_match_log_social_operation_reconcile` | `RECONCILE SOCIAL MATCH` | — | — | match-player |
| BE-196 | League Live — `POST /admin/clubs/{club_id}/league-manager/live-operations/{operation_key}/reconcile` — `post_admin_league_live_create_reconcile` | `RECONCILE LIVE SESSION` | — | — | league-live-domain,league-live-submit |
| BE-197 | Tournament Ops — `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-registrations/operations/{operation_reference}/reconcile` — `post_admin_tournament_registration_team_import_reconcile` | `RECONCILE REGISTRATION IMPORT` | retained_request.expected_state_fingerprint,retained_request.expected_draw_updated_at | retained_request.idempotency_key | tournament-operations |
| BE-198 | League Live — `POST /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/retry` — `post_admin_league_live_round_retry` | `RETRY LEAGUE ROUND` | server-retained expected_updated_at,expected_operation_key | server-retained idempotency_key | league-live-submit |

## Action-level findings and required remediation

### P0 — boundary bypasses

**None found.** Every unsafe FastAPI contract is in the staging write-wave manifest, unsafe production traffic is fail-closed by default, and admin mutation routes pass through authentication/club-role resolution. This conclusion is limited to the registered FastAPI tree at the audited revision; the manifest-drift tests must remain release-blocking.

### P1 — post-remediation status

| IDs | Baseline finding / risk | Status and implemented backend change | Residual / disposition |
|---|---|---|---|
| `BE-050` / `BE-195` | Club Social Match Log edit could overwrite browser-stale state and a lost response could not be distinguished safely. | **Application guard implemented.** Request now carries every patched field's `expected_current`, a stable key, and `SAVE SOCIAL MATCH`; the update uses CAS; intent/result are ledgered; exact replay and structured stale/uncertain `409` responses are present. `BE-195` requires `RECONCILE SOCIAL MATCH` and proves the reviewed-before or intended-after values from authoritative state without rerunning the patch. | **Deferred migration:** domain update + audit/receipt are not one SQL transaction. Required-audit failure uses CAS rollback and recovery state. Do not claim full database atomicity. |
| `BE-059` / `BE-193` | Per-player insert loop allowed partial bulk completion and unsafe retry. | **Application guard implemented.** The reviewed, normalized/deduplicated list is fingerprinted; `CREATE PLAYERS` and idempotency are mandatory; new players use one multi-row insert statement; result reports created/unchanged/readback counts; exact replay is safe. `BE-193` requires `RECONCILE PLAYER BATCH` and resolves only from frozen preflight evidence plus authoritative player readback. | **Deferred migration:** the insert statement is atomic, but intent/audit/completion are separate. A batch RPC is required for full transaction-level conformance. |
| `BE-081` / `BE-082` / `BE-194` | Player create had no replay key; player update had no browser-loaded CAS or replay identity. | **Application guard implemented.** Create requires idempotency. Detail exposes `state_fingerprint`; update requires it plus idempotency, uses field CAS, exact completion replay, and durable recovery state. `BE-194` requires `RECONCILE PLAYER OPERATION` and proves create/edit results from frozen evidence plus authoritative state. | **Deferred migration:** player row + audit/receipt are not one SQL transaction. Routine Create/Edit intentionally has no typed phrase. |
| `BE-083` / `BE-194` | Rating edit had a phrase but no reviewed state or retry identity. | **Application guard implemented.** League-rating detail exposes `state_fingerprint`; update requires it plus idempotency and `SAVE LEAGUE RATING`, applies CAS, and supports exact replay. `BE-194` also reconciles this workflow from the planned rating evidence and authoritative row. | **Deferred migration:** rating row + audit/receipt are not one SQL transaction. |
| `BE-074` | Official match publication lacked backend proof of consequence confirmation. | **Complete.** `PUBLISH MATCHES` is validated before claim and included in the existing durable request fingerprint. | No schema residual. |
| `BE-118` / `BE-197` | A registration-import response loss left the browser without a safe retry or authoritative way to distinguish completion from non-application. | **Application guard implemented.** The browser creates and tab-persists one UUID plus the exact reviewed import request before send. Structured uncertainty retains that request. `BE-197` requires `RECONCILE REGISTRATION IMPORT`, authenticates Tournament Admin authority, and never reruns the import. An absent operation is first serialized against any late original by inserting the exact same-key/fingerprint tombstone; an original `intent` that wins remains uncertain until settled. | Team rows are atomically written, but that RPC and the generic operation/audit ledger are separate transactions. Stored result can prove completion and a winning tombstone proves non-start; a normal empty `recovery_required` row stays locked because fingerprint equality cannot fence a late RPC commit. |
| `BE-026` / `BE-196` | A lost session-create response could create a second active session; compensation could fail. | **Application guard implemented.** Create requires a stable key; intent precedes mutation; exact completion replay and changed-payload rejection are enforced; ambiguous failure is recovery-required. `BE-196` requires `RECONCILE LIVE SESSION` and proves the exact planned session and court state from authoritative storage without rerunning creation. | **Deferred migration:** session + court rows + service audit + operation completion are not one transaction. Existing compensation remains; create-session RPC is preferred. |
| `BE-184` | Retries could send duplicate edit-link email and the public request was an abuse surface. | **Duplicate-delivery mitigation implemented.** The contract carries a stable key; recipient/tournament/registration gets a privacy-safe 15-minute dedupe receipt; repeat and provider-uncertain requests within that bucket do not resend; the response stays generic. | **Partially deferred:** client-IP rate limiting, longer-lived exact-key replay, and provider reconciliation require approved infrastructure/provider work. |

### P2 — standardization and concurrency follow-up

| IDs | Gap | Existing mitigation | Required alignment |
|---|---|---|---|
| `BE-034` / `BE-035` | League create/duplicate has no client replay key; audit is written after insert. | Unique league name prevents a second identical row, so retry usually becomes a conflict rather than duplication. | Add a creation idempotency receipt and atomic audit so a lost success can replay as success. |
| `BE-038`, `BE-041` | Awards config/overrides do not require backend confirmation. | Config version or reviewed preview fingerprint protects concurrency; awards irreversible stages are strongly guarded. | Treat as routine Edit only if the UI review is sufficient; otherwise add action-specific phrase consistently. |
| `BE-045`, `BE-047` | League settings/roster requests do not carry the version the user loaded. | Services re-read immediately and use current status/`updated_at` or set-state checks during the write. | Send browser-loaded expected version/timestamp so a stale form cannot be silently applied to a newer record; add a stable replay key. |
| `BE-067`–`BE-073` | Baseline generator session/round/complete mutations had no backend phrase. | Expected version, idempotency, durable operation ledger, audit, and reconcile are strong. `BE-073` now requires `COMPLETE SESSION` and includes it in the request fingerprint. | **Partially complete.** `BE-067`–`BE-072` remain documented routine Create/Edit/progression exceptions; revisit only if product reclassifies one as a guarded consequence. |
| `BE-111`, `BE-113` | Baseline tournament payment and fulfillment status updates had no backend consequence phrase. | Both carry expected state and idempotency and use tournament commerce guarded RPC persistence. Backend now requires `SAVE PAYMENT STATUS` / `SAVE FULFILLMENT STATUS` and fingerprints the phrase. | **Complete.** No migration required. |
| `BE-157` | Verified-request administration is phrase-only, without client CAS/replay identity. | Permission and audit checks exist. | Add expected request version plus idempotency/receipt; return 409 on stale review. |
| `BE-158`–`BE-160` | Weekly recap mutations have confirmation and expected state but no caller idempotency key. | Service uses authoritative state/business identity and publication checks. | Add stable request keys for generate/save/publish, especially because publish/unpublish affects public communication. |
| `BE-180`–`BE-183` | Public pairing requests do not expose a client idempotency field. | This is **not an unguarded write**: edit tokens bind the actor; `jupr_app/services/public_tournament_partner_request_service.py` calls atomic create/transition RPCs; migration `supabase/migrations/20260719194500_public_partner_pairing_lifecycle.sql` enforces uniqueness, locking, idempotent transitions, and prevents duplicate notification. | Preserve the current database guarantees; optionally surface the returned idempotent state as a standard action result. Do not add a redundant typed phrase to a public token action. |
| `BE-186`, `BE-187` | Registration edit has CAS but no top-level request key; create uses nested commerce idempotency only when commerce applies. | Registration identity/unique rules prevent duplicate active registration, edit uses token plus expected registration/selection versions, and confirmation email is suppressed on an idempotent repository replay. | Add a top-level registration idempotency key/receipt so a lost success replays the original confirmation instead of returning a duplicate-registration conflict. |
| `BE-176`, `BE-192` | No caller replay key appears in the route contract. | These are **not duplicate-write gaps**: support intake derives a daily request fingerprint and unique dedupe key with rate limiting (`jupr_app/services/public_support_intake_service.py:158-314`); verified updates uses existing-request checks, a race-safe unique open-request index, honeypot, and email rate limit (`jupr_app/services/public_verified_updates_service.py:149-217`). | Document the business-identity replay semantics in the API response contract and keep the unique indexes/rate-limit tests. A client key is optional. |

### P3 — conservative unsafe-method classification

Preview, plan, quote, resolve, or evidence endpoints such as `BE-011`, `BE-016`, `BE-025`, `BE-028`, `BE-037`, `BE-048`, `BE-058`, `BE-062`, `BE-063`, `BE-066`, `BE-075`, `BE-085`, `BE-095`, `BE-107`, `BE-109`, `BE-121`, `BE-131`, `BE-139`, `BE-140`, `BE-168`, `BE-179`, `BE-185`, `BE-189`, and `BE-190` are intentionally covered by the unsafe-method middleware. Do not remove them from the manifest merely because the current implementation is read-only: some previews persist a reviewed snapshot or audit record, and future implementation changes should fail closed.

## Audit-log requirements and atomicity

Admin route helpers authenticate a Supabase bearer token, resolve the club-scoped role/permission, and generally write an activity record for denied and successful actions (`services/api/auth.py` plus each `admin_*_routes.py` helper). The environment can make successful-action audit persistence mandatory. That is useful, but **audit-after-write plus compensating rollback is not equivalent to atomic audit**:

- strong families persist operation intent before mutation and completion afterward, with exact recovery for an uncertain result;
- atomic RPC families commit domain rows, idempotency receipt, and audit in one transaction;
- legacy direct-table services may mutate first and only then discover that required audit persistence is unavailable.

The P1 rows above identify the clearest legacy cases. Any implementation wave should prefer the existing guarded runners/RPC patterns rather than inventing another ledger shape.

Database privilege posture is correct in the inspected migrations: operation ledgers are service-role-only, browser roles are revoked, and RLS/forced RLS is retained. Backend remediation must continue to use the configured server-side `SUPABASE_SERVICE_ROLE_KEY`; it must never pass that credential to Next.js client code.

## Error and status mapping

The dominant mapping is sensible but not uniform:

| HTTP status | Current meaning | Evidence and issue |
|---:|---|---|
| `400` | validation or incorrect confirmation phrase | Widely used by route `_handle` helpers; adequate when the client receives a typed error kind |
| `401` | missing/invalid bearer | `services/api/auth.py`; standard |
| `403` | feature/write gate or insufficient permission | Admin/public write-gate helpers; the client must distinguish disabled capability from permission denial |
| `404` | missing resource where explicitly separated | Some route families map `ValueError` missing cases; others collapse them into `400` |
| `409` | stale/CAS conflict, idempotency mismatch, active lock, or recovery-required/uncertain | Correct status class, but most families return a plain string. Match Log is the stronger example: `services/api/admin_match_log_routes.py:303-376` returns structured codes and operation/recovery fields |
| `429` | public rate limit | Support/verified public intake; preserve `Retry-After` consistently |
| `503` | write capability/schema/service-role unavailable, durable persistence unavailable | Correct for operational prerequisites; direct match recovery-required currently also uses `503`, whereas other uncertain families use `409` |
| `500` | unexpected runtime or critical rollback failure | Some legacy services expose only a string; critical partial-write states need a stable recovery code/reference |

Adopt one machine-readable error envelope across all consequential endpoints:

```json
{
  "code": "STALE_VERSION",
  "kind": "conflict",
  "message": "The record changed after it was loaded.",
  "operation_key": "...",
  "recovery_required": false,
  "request_id": "...",
  "details": {}
}
```

`kind` should be one of `validation`, `conflict`, `forbidden`, `failed`, or `uncertain` as defined by `docs/interaction-standard.md`. Do not make clients infer uncertainty by regex-searching a message. A durable action that cannot prove completion should return `kind: "uncertain"`, the same operation key, and a reconcile/inspect URL or action—not generic `500` and not advice to submit a new key.

## Backend conformance contract for implementation

Every new or remediated consequential write should meet all applicable checks below:

1. **Boundary:** exact method/template is present in a named staging wave and covered by manifest-drift tests.
2. **Authority:** admin bearer + club permission, or a narrowly scoped public token/secret/business identity; service-role credentials stay server-side.
3. **Preflight:** validate the complete request and readable consequences before any domain write.
4. **Confirmation:** validate an action-specific phrase for destructive, public, financial, rating, bulk, communication, recovery, and publication actions. Routine reversible Create/Edit can be a documented exception.
5. **CAS:** compare the version/fingerprint that the user actually reviewed, not only a value re-read immediately before update.
6. **Replay:** accept or derive a stable request identity, reject key/payload mismatches, and replay the exact completed result.
7. **Atomicity:** commit all domain rows plus receipt/audit atomically; if that is impossible, persist intent first and make every partial stage recoverable.
8. **Audit:** record actor, role, source, entity, before/after, request fingerprint, operation key, and completion/recovery state.
9. **Readback:** return authoritative resulting state and meaningful counts; `200` alone is not proof for the UI.
10. **Uncertainty:** retain the operation identity and provide exact inspect/reconcile/compensate behavior; never require a blind fresh mutation.
11. **Errors:** return a stable code/kind envelope and consistent `400/401/403/404/409/429/503/500` semantics.
12. **Tests:** cover duplicate submit, same-key/different-payload rejection, stale CAS, timeout-after-commit replay, audit failure, partial RPC failure, authorization denial, write-wave denial, and reconciliation.

## Verification commands

The audit reran a dependency-free AST/manifest check and obtained:

```text
unsafe_routes=198 manifest=198 missing=0 stale=0 overlaps=12 open_flags=32
```

The repository's boundary-equivalent tests are:

```bash
# Registered unsafe-method contracts (AST-based in the audit script): 198
# OPEN_WRITE_ROUTES entries: 198
# Set comparison: 0 missing, 0 stale

pytest -q tests/test_staging_write_wave_guards.py \
  tests/test_permanent_staging_write_mode.py
```

Result for this exact boundary command: **20 passed**, with one framework deprecation warning.

The targeted application-guard remediation, separate from the forward trigger-function migration, was additionally verified with:

```bash
PYTHONPATH=/tmp/jupr-pytest:$PYTHONPATH python -m pytest -q \
  tests/test_api_contract_play_generators.py \
  tests/test_admin_player_editor_service.py \
  tests/test_api_contract_admin_player_editor.py \
  tests/test_admin_match_uploader_service.py \
  tests/test_api_contract_admin_match_uploader.py \
  tests/test_api_contract_admin_match_log.py \
  tests/test_api_contract_admin_league_live.py \
  tests/test_api_contract_tournament_registration_edit.py \
  tests/test_public_tournament_registration_edit_service.py \
  tests/test_admin_tournament_commerce_safeguards.py
```

Final result for this exact backend-remediation command: **146 passed**, with three pre-existing framework deprecation warnings. It covers the remediated phrase contracts, reviewed-state rejection, exact completed replay, changed-request/key rejection, duplicate email suppression, durable-intent failure, authorization, proof-only reconciliation behavior, Unicode fingerprint parity, and timeout-after-commit ambiguity. It does not prove cross-table SQL atomicity for the explicitly deferred RPC work above.

The final frozen candidate broad run completed with **2835 passed, 9 failed, and 40 warnings**. The exact `dcf08791` baseline completed with **2756 passed, 10 failed, and 40 warnings**. Each of the candidate's nine failures belongs to the baseline's ten-failure set; one baseline failure now passes and no new interaction-standard regression remains. These counts are regression evidence rather than a claim that the repository-wide suite is globally green.

Re-run the AST/manifest comparison whenever a FastAPI route decorator changes. The stable-ID table should be updated in the same pull request; new unsafe routes receive IDs after `BE-198`. Four authenticated operation-inspection routes remain safe-method `GET` contracts; five separately appended proof-only reconciliation routes `BE-193`–`BE-197` and the server-retained League Live retry `BE-198` are `POST` contracts, raising the unsafe-route and staging-manifest totals from 192 to 198.
