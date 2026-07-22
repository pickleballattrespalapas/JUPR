# Tournament Operations parity evidence (Order 27)

Status: automated implementation and focused evidence are present. Staging acceptance has not been executed, so `tournament_ops`, `tournament_live`, and related Tournament Admin matrix rows remain `Partial`.

## Scope and authority

Python/FastAPI is authoritative for draw creation, registration/manual/bulk team import, reviewed DUPR results import, round-robin generation and scoring, playoff progression, podium generation, awards, official singles/doubles Match Log publication, and player-update handoff. Next.js supplies route-specific operator workflows at:

- `/admin/tournaments/ops/draws`
- `/admin/tournaments/ops/import`
- `/admin/tournaments/ops/results`
- `/admin/tournaments/ops/publish`

The browser receives public-safe status and state. `SUPABASE_SERVICE_ROLE_KEY` stays on FastAPI. The canonical private schema seam is `supabase/migrations/20260719204700_tournament_operations_guard_surface.sql`.

## Staging gates

All mutations require `JUPR_ENV=staging`, FastAPI's service role, authenticated club membership, the action's role permission, an explicit Yes/No confirmation dialog that supplies the action-specific internal confirmation value, current reviewed state, and required audit intent/completion.

- `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS=1`: Tournament Admin visibility.
- `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS=1`: Order-27 operations mutations.
- `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH=1`: separate official Match Log/rating boundary.
- `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_EMAIL_HANDOFF=1`: separate automatic player-update handoff.
- `JUPR_EMAIL_MODE=dry_run` or `staging_redirect`: mandatory for staging official publish; live delivery is refused.
- `JUPR_REQUIRE_API_AUDIT_LOG=1`: required during acceptance.

Production refuses these staging-only mutation gates. Do not copy service-role credentials to Vercel or a browser environment.

Role boundaries are explicit: structural operations require `manage_tournaments`, scoring requires `enter_scores`, official publish requires both `manage_tournaments` and `manage_matches`, and an import that creates players also requires `manage_players`.

## Mutation and recovery contract

Every route submits the state fingerprint displayed with the same captured snapshot. The server re-reads the full club/tournament state before durable intent and again after acquiring the tournament operation lock. Exact retries share a deterministic request fingerprint and operation key; a conflicting fingerprint is refused.

Atomic SQL functions take deterministic child-row locks before the draw lock, compare the reviewed versions under those locks, validate club ownership and lifecycle constraints, and either commit the reviewed set or roll back it. Direct service-role/Streamlit writes to draw teams, games, podium, or draw-scoped podium badges advance the draw version through `BEFORE` triggers. Structural team changes are blocked after scheduling while seed-only updates remain supported; round-robin outcomes cannot change after playoffs, and game outcomes cannot change after podium or official publication. Results import creates new players and draw teams/games/podium in one transaction. `create_new` refuses an existing normalized name, APPEND refuses an already-assigned player, imported non-round-robin stages normalize to `PLAYOFF`, downstream playoff rows are locked before score propagation, and negative scores are rejected in Python and SQL.

### Required expected-version payloads (Order-28 handoff)

Order-28 Tournament Live callers must use versions from one authoritative Tournament Ops snapshot and must not synthesize timestamps. Every mutation still sends `expected_state_fingerprint`; the database-level contract adds:

- team replace, registration-team import, bulk-team import, and reviewed results import: `expected_draw_updated_at`;
- round-robin generation: `expected_draw_updated_at` plus `expected_team_versions`, the exact complete draw-team set as `{id, updated_at}` rows;
- playoff generation and podium generation: `expected_draw_updated_at`, the exact complete `expected_team_versions`, and the exact complete draw-game set in `expected_source_game_versions`;
- score save: `expected_draw_updated_at` plus `expected_game_updated_at` for the selected game. FastAPI derives every downstream dependency update from the same loaded draw-game set, includes each dependency's `expected_updated_at` in the private RPC, and SQL verifies all primary/dependency versions under one sorted child lock set before the draw CAS.

Missing or incomplete expected versions are refused in the guarded staging runtime. Any stale draw, team, primary game, source-game, or dependency-game comparison returns 409 without a domain write; reload the entire authoritative snapshot before forming a new request. The old team-write, score, and results-import RPC overloads without these expected versions are explicitly dropped and receive no execute grant.

Official publish uses a narrower response-loss rule because legacy rating processors write the official matches. The durable request stores the complete deterministic `tournament_game_id` plan. A same-key `recovery_required` replay invokes a read-only callback scoped by `club_id`, `tournament_id`, and every expected game id:

- exact one-to-one complete match set: reconstruct the result, persist the completion audit, and mark the operation completed without invoking a match processor;
- zero, partial, duplicate, changed-game-set, read failure, or cross-club evidence: remain `recovery_required`;
- no recovery branch blindly resubmits the publisher.

Preserve the operation key printed in a 409 response. If exact reconciliation cannot be proven, stop. Use the Streamlit Tournament Operations fallback and the Match Log/Replay History correction path; do not generate a new request in an attempt to clear the lock.

## Staging acceptance book

Use a disposable tournament, separate draws for singles and doubles, and staging-only players/email destinations.

1. Apply migrations through `20260719204700`, deploy FastAPI and Next from the same commit, enable only the gates above, and confirm the status panel reports service role, strict audit, operations gate, official gate, and safe email mode.
2. With an authorized administrator, create a draw; import registrations and bulk teams; verify cross-club players and duplicate roster assignments are refused.
3. Generate round robin, enter valid scores, prove stale draw/game/dependency versions return 409, generate playoffs, score through dependencies, then generate and award the podium. In two database sessions, hold an RPC dependency-team lock while inserting/updating a game FK or inserting a podium row directly; verify there is no `40P01`, the direct-first write advances the draw version, and the waiting RPC fails stale without a partial write.
4. Preview a DUPR CSV and verify `dry_run=true`, `write_count=0`, a 64-character review fingerprint, explicit mapping decisions, and no player/team/game/podium writes. Commit the exact reviewed fingerprint on a fresh disposable draw; verify player creation and results are all present or all absent.
5. Exercise singles and doubles official publish separately. Confirm finalized games create exactly one official match per game id, rating/replay records are consistent, and email remains dry-run/redirected.
6. Replay the identical successful request and verify `idempotent_replay=true` with no extra match. Inject/fixture a response-lost operation: exact complete evidence must reconcile; zero/partial/foreign-club evidence must stay 409 recovery-required with no second processor call.
7. Verify `admin_activity_log` has intent, completion/failure, actor, entity, operation key, and request fingerprint for each mutation.
8. Compare labels, empty/error states, mobile layout, exports, singles behavior, and fallback links with Streamlit. Record screenshots and operation keys.

Stop immediately on any cross-club row, unscoped official match, live email, missing audit, fingerprint mismatch accepted as a retry, partial write outside recovery state, duplicate `tournament_game_id`, or a recovery flow that invokes the mutation again. Disable the three Order-27 gates and preserve fixture/evidence rows for investigation.

## Automated evidence

Focused commands:

```bash
/tmp/jupr-followup-venv/bin/pytest -q \
  tests/test_tournament_admin_guarded_operations.py \
  tests/test_tournament_operations_order27.py \
  tests/test_api_contract_admin_tournament_draw_create.py \
  tests/test_api_contract_admin_tournament_team_editor.py \
  tests/test_api_contract_admin_tournament_registration_import.py \
  tests/test_api_contract_admin_tournament_bulk_team_import.py \
  tests/test_api_contract_admin_tournament_game_generation.py \
  tests/test_api_contract_admin_tournament_game_scoring.py \
  tests/test_api_contract_admin_tournament_playoff_generation.py \
  tests/test_api_contract_admin_tournament_podium.py \
  tests/test_api_contract_admin_tournament_awards.py \
  tests/test_api_contract_admin_tournament_match_publish.py \
  tests/test_api_contract_admin_tournament_singles_publish.py
cd apps/web && npx tsc --noEmit
```

Optional staging browser evidence:

```bash
cd apps/web
npx playwright test e2e/tournament-operations.staging.spec.ts
```

Authenticated fixtures use `STAGING_ADMIN_BEARER_TOKEN`, `JUPR_TOURNAMENT_OPS_TOURNAMENT_ID`, `JUPR_TOURNAMENT_OPS_DRAW_ID`, and `JUPR_TOURNAMENT_OPS_GAME_ID`. Mutating fixture evidence remains skipped unless `JUPR_TOURNAMENT_OPS_ALLOW_MUTATION_E2E=1` is explicitly set.

## Independent migration review disposition

The independent read-only review cleared service-role-only RLS/function grants and the earlier lifecycle, snapshot/fingerprint coupling, pagination, and cross-club team validation changes. It originally blocked on non-atomic player creation, unlocked dependency propagation, fail-open/NULL/case draw uniqueness, duplicate APPEND rosters, imported stage mismatch, negative scores, non-reconcilable official publish, and a final child-write/CAS interleaving gap. This order addresses each in the service/migration and adversarial contracts above, including exact draw/team/game versions, child-to-draw lock ordering, direct-writer version triggers, and derived-state guards.

The reviewer also called out that the legacy official publisher updates matches/ratings row-by-row. Order 27 does not pretend that processor is transactional: it preserves availability through the deterministic exact-set reconciliation rule, and routes any non-exact outcome to Match Log/Replay recovery. Final approval still requires the staging fault-injection exercise and cumulative Order-29 evidence.
