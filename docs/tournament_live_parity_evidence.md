# Tournament Live parity evidence (order 28)

Order 28 makes `/admin/tournament-live` automated-ready and manual-ready as a
draw-scoped in-play tournament runner. It does not move the parity matrix row to
`Done`; Order 29 still owns real staging acceptance and matrix reconciliation.

## Product and authority boundary

- Tournament Live operates one prepared `tournament_event_draw` during play:
  score entry, round-robin generation, playoff generation/progression, podium,
  awards, and terminal official-match publishing.
- Tournament Ops remains the setup/import/correction surface. Order 28 calls its
  existing Python services and domain bracket functions; it does not implement a
  second scoring or bracket engine.
- JUPR Live remains the one-off Round Robin, League/Ladder, and Club Social
  product. Tournament Live never creates or edits a JUPR Live session.
- The browser renders FastAPI state, retains an idempotency UUID for an exact
  response-loss retry, and submits commands. FastAPI authenticates the JWT,
  permits scorekeepers to read/save scores, requires `manage_tournaments` for
  bracket/podium commands, and requires both `manage_tournaments` and
  `manage_matches` for rated official publish. It computes readiness/state
  fingerprints and performs every domain write with the server-only Supabase credential.

## Staging-only gates

Apply `supabase/migrations/20260719203000_tournament_admin_operations.sql`, the
Order-27 `20260719204700_tournament_operations_guard_surface.sql` extension, and then
`supabase/migrations/20260719205000_tournament_live_operations.sql` through the
reviewed migration process. The final surface constraint preserves both
`operations` and `tournament_live`. Keep the live write flag closed until all
status checks are green.

```text
JUPR_ENV=staging
JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS=1
JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES=1
JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH=1  # only during the terminal publish exercise
JUPR_REQUIRE_API_AUDIT_LOG=1
SUPABASE_SERVICE_ROLE_KEY=<FastAPI-only staging secret>
JUPR_STREAMLIT_FALLBACK_URL=<reviewed Tournament Live fallback>
```

Production and every non-staging environment refuse Tournament Live commands,
even if the write flag is accidentally present. Reads, operation history, and
the Streamlit fallback remain available when writes are closed. Never expose the
service role or any write gate as a `NEXT_PUBLIC_*` value.
If automatic player updates are enabled for the terminal publish, also use the
Order-27 email-handoff gate and `dry_run`/`staging_redirect` delivery mode.

## Stale, idempotency, lock, and audit contract

The draw snapshot fingerprints the tournament/draw, teams, game assignments and
scores, podium, verified podium badges, and official `tournament_game_id` links.
Operation rows and player display names are excluded so audit/history refreshes
do not manufacture stale state.

1. Each command supplies the reviewed 64-character draw fingerprint, exact draw
   `updated_at`, the command-specific game/team/source-game versions, and a
   browser-retained UUID. FastAPI rejects incomplete version sets and UUID reuse
   with a changed state, command, or payload.
2. A partial unique lock serializes active operations by
   `tournament:{tournament_id}:draw:{draw_id}`. FastAPI checks state before and
   after acquiring that durable lock.
3. Exact deterministic preflight runs before the intent audit and domain write.
   The operation request durably stores the reviewed structural result: every
   round/slot/team pairing, every playoff code/round/team/source reference,
   every podium placement/team, every award recipient/context, or the official
   publish game-ID set, row versions, and per-match content fingerprints. Score
   requests store the primary score result plus the full expected downstream
   playoff assignment/source projection.
4. Required `*_intent` audit precedes the shared Tournament Ops service call.
   After the durable lock is acquired, Live captures the fingerprint-matching
   snapshot once and passes its exact draw/team/game versions to the Order-27
   SQL compare-and-swap RPCs with `atomic=True`; the mutation callback does not
   take a fresh unbound Live snapshot. Podium awards validate and insert the
   exact reviewed recipient/context set in one child-before-draw transaction.
   Official publish calculates the complete match/player/league-rating write
   plan in Python, then a single service-role-only CAS RPC locks and compares
   tournament/event-option/draw/team/game/player/league inputs before applying
   that core atomically. The durable result precedes
   required `*_completion` audit. A completed exact retry returns the stored
   result without a second domain write.
5. Any post-intent exception becomes `recovery_required`. The entire draw stays
   locked; the UI preserves the exact UUID/request across reloads.
6. `RECONCILE TOURNAMENT LIVE` is readback-only. It compares authoritative rows
   with the exact projection stored before mutation; counts, current-standings
   recomputation, and broad placement checks are never proof. It may close the
   lock as `completed` on an exact set or as `not_applied` when the complete draw
   fingerprint is unchanged. Official-publish completion additionally requires
   exact match-content fingerprints and one operation-key/request-fingerprint
   bound domain audit receipt (plus client UUID for Live) written only
   after match, rating, player, and automatic-update processors return. Match
   rows alone never prove processor completion. Reconciliation never repeats
   the mutation.
7. Partial, duplicate, changed, or unproven generated rows, score dependencies,
   official publishing, and awards remain locked. Use
   Match Log and Replay History plus the Streamlit fallback; do not infer success
   from a timeout or manually edit the operation row.

The official-publish database claim also rejects legacy processors at their
first match insert and rejects direct player, league-rating, and league-metadata
writes while the claim is active. Thus a legacy writer cannot insert a match and
then fail only at its later rating update. This is an active-claim boundary; it
does not claim that legacy writers are globally serialized when no official
publish owns the club rating domain. A lost CAS response or missing bound
post-processor receipt remains recovery-required and must never trigger a
republish.

## Exact confirmation phrases

| Command | Phrase |
|---|---|
| Save a game score | `SAVE SCORE` |
| Generate round robin | `GENERATE GAMES` |
| Generate playoffs | `GENERATE PLAYOFFS` |
| Generate podium | `GENERATE PODIUM` |
| Award podium | `AWARD PODIUM` |
| Publish official rating matches | `PUBLISH MATCHES` |
| Verify interrupted operation | `RECONCILE TOURNAMENT LIVE` |

## Automated evidence

```bash
/tmp/jupr-followup-venv/bin/python -m pytest -q \
  tests/test_admin_tournament_live_runner.py \
  tests/test_tournament_official_atomic_publish.py \
  tests/test_tournament_live_static_contract.py \
  tests/test_tournament_admin_guarded_operations.py \
  tests/test_api_contract_admin_tournament_game_scoring.py \
  tests/test_api_contract_admin_tournament_game_generation.py \
  tests/test_api_contract_admin_tournament_playoff_generation.py \
  tests/test_api_contract_admin_tournament_podium.py \
  tests/test_api_contract_admin_tournament_awards.py \
  tests/test_api_contract_admin_tournament_match_publish.py \
  tests/test_tournament_operations_order27.py

make check-next-parity-matrix check-parity-closure-program
cd apps/web && npx tsc --noEmit && npm run build
cd apps/web && npx playwright test --list e2e/tournament-live.staging.spec.ts
```

The focused suite covers the separate product boundary, stable draw snapshot,
closed non-staging writes, exact confirmation and reviewed versions, stale
no-write behavior, durable intent/completion evidence, exact replay,
idempotency collision, exact score/dependency and generated-row recovery,
exact award recipients, atomic award drift refusal, and refusal to unlock
partial/duplicate/changed official publishing or match rows without the exact
post-processor completion receipt.

The browser suite is opt-in. In the candidate-bound workflow, a server-only
preparation step creates a uniquely owned DRAFT tournament, draw, four
null-player teams, and one unpublished round-robin game, then exports only its
IDs and score values to the browser step. No fixture IDs are stored as repository
variables. The mutation case requires
`JUPR_TOURNAMENT_LIVE_ALLOW_MUTATION_E2E=1`; its `finally` block restores the
original result and fails loudly if that readback cannot be verified. An
always-run server-side cleanup refuses to proceed if any official match links to
the fixture, deletes only the manifest-owned core rows, and proves they are gone.
Operation and audit evidence is intentionally retained. This automated case does
not publish an official match or exercise rating writes.

## Manual staging book

Use one disposable, unpublished draw with four linked-player teams, complete
order-27 setup evidence, no official matches, and a recorded baseline export.
Capture request IDs, both idempotency keys, audit row IDs, operation rows, game
IDs, badge contexts, official match IDs, and before/after screenshots.

1. Keep the live write flag off. On desktop and a 390 px mobile viewport, load
   the draw, inspect games/readiness/history, follow the Streamlit fallback, and
   confirm every command remains disabled.
2. Open only the live flag. Score one round-robin game. In a second tab, submit
   the old fingerprint and verify 409 with no operation/audit/domain write.
3. Retry the first request with the same UUID/body and verify one game mutation,
   one operation key, `idempotent_replay=true`, and intent/completion evidence.
4. Finish round robin, generate playoffs, score every dependency in order, and
   verify Python-assigned semifinal/final/bronze teams after each reload.
5. Generate the draw podium, mint awards, and verify all expected
   `(player_id,badge_id,context_id)` rows before publish becomes ready.
6. Publish official matches once. Verify every tournament game has exactly one
   content-identical official link, rating/player-update behavior matches Order
   27, the domain audit contains the reviewed publish-plan fingerprint, and
   exact retry returns the stored result without another match.
7. Simulate response loss after a stored result and reconcile it. Separately use
   controlled partial-publish and post-match/pre-receipt fixtures; prove both
   remain locked. Also alter one stored date/participant/score/bonus field and
   prove the content fingerprint refuses completion until Match Log/Replay
   recovery establishes authoritative evidence.
8. Confirm the games table is readable on desktop, game cards replace it on
   mobile, controls have 44 px targets, keyboard focus is visible, and the page
   has no horizontal viewport overflow.
9. Restore all reversible score fixtures, retain immutable audit/operation
   evidence, verify official fixture cleanup through the approved Order 27
   recovery procedure, and close the live write flag.

Stop immediately on wrong-club data, a missing migration/service role/audit, a
non-409 stale write, duplicate official links, a browser-computed bracket, an
unlocked uncertain operation, failed cleanup, or any request to use production.

## Rollback

Close `JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES` first. The nullable
idempotency column and private indexes can remain safely deployed while the UI
is read-only. Do not delete audit/operation evidence. Recover scored or published
data only through Tournament Ops, Match Log, and Replay History under their
existing reviewed procedures. This implementation did not apply migrations,
deploy code, enable flags, or mutate staging/live data.
