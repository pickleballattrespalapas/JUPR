# Live/Ladder parity staging runbook

Order 24 makes Challenge Ladder Admin, Moneyball, and one-off JUPR Live Admin automated-ready and manual-ready. It does not mark any parity-matrix row `Done`. Streamlit remains the write fallback until this book is executed and signed off against isolated staging fixtures.

No staging or production mutation was performed while implementing this order. The mutation browser suite is opt-in and refuses a non-staging API origin.

## Authority and boundaries

- Next.js renders controls and sends reviewed inputs. It does not calculate official schedules, winners, rank effects, settlement, ratings, or public live projections.
- FastAPI authenticates the Supabase bearer token, resolves the club role, verifies the staging-only workflow flag, and calls Python service/domain authority.
- FastAPI requires `SUPABASE_SERVICE_ROLE_KEY` for every mutation and recovery route. Never expose that key to Next.js or a browser.
- Challenge Ladder writes require `manage_matches`. Moneyball and JUPR Live writes require `enter_scores`.
- Every write carries the current Python version/fingerprint and a stable idempotency key. A private service-role-only ledger serializes a club/surface version, persists intent before mutation, snapshots the result, and persists completion or failure audit evidence.
- Official match contexts are unique for each durable operation. Match Log is the correction surface and Replay History is the verification surface.
- JUPR Live Admin is only for one-off Round Robin and League/Ladder sessions. Tournament brackets/draws are rejected and belong to Tournament Live/Ops.

Order 23's generic `admin_guarded_write_operations` ledger protects diagnostics and maintenance actions. Order 24's `live_ladder_admin_operations` ledger is intentionally separate: it stores domain request fingerprints, authoritative versions, official-match recovery contexts, and replayable result snapshots for these three rating-adjacent surfaces. Do not merge or substitute the tables during stacking.

## Install and enable

Apply `supabase/migrations/20260719201500_live_ladder_admin_operations.sql` to the staging project, then reload the PostgREST schema. The migration creates the private ledger, its active-version lease, and the partial unique index for deterministic UUIDv5 official-match contexts. UUIDv5 matches the canonical `matches.context_id` type while isolating new operation contexts from legacy rows.

Set only on FastAPI staging:

```text
JUPR_ENV=staging
SUPABASE_SERVICE_ROLE_KEY=<staging service role>
JUPR_REQUIRE_API_AUDIT_LOG=1

JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER=1
JUPR_ENABLE_STAGING_NEXT_ADMIN_CHALLENGE_LADDER_WRITES=1

JUPR_ENABLE_NEXT_ADMIN_MONEYBALL=1
JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES=1

JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE=1
JUPR_ENABLE_STAGING_NEXT_ADMIN_JUPR_LIVE_WRITES=1
```

The broad visibility flags may remain enabled while their staging write flag is off. In that state, Next is read-only and explicitly directs staff to the corresponding Streamlit fallback. Production must keep all three `JUPR_ENABLE_STAGING_*_WRITES` flags off.

Before staff use, confirm each status response reports `authority: python_fastapi` where applicable and `writes_enabled: true`. Confirm an account without the required permission receives `403`, and a mutation without a bearer token receives `401` before any table access.

## Durable operation states

`intent` means the required audit intent and operation row exist but mutation is not complete. `running` owns the authoritative version lease. `mutated` has a result snapshot awaiting completion audit. `completed` is safe to replay and returns the stored result without repeating domain work. `recovery_required` means the outcome cannot be inferred from the HTTP response; stop and reconcile. `failed` means the second version check rejected the operation before domain mutation.

An HTTP timeout is never proof of failure. Keep the operation key and original idempotency key. Do not edit the payload, refresh into a new operation, or click publish again until reconciliation is complete.

## Exact confirmation phrases

Challenge Ladder uses `CREATE LADDER CHALLENGE`, `START LADDER CLOCK`, `ACCEPT LADDER CHALLENGE`, `SAVE LADDER`, `RECORD LADDER FORFEIT`, `RECORD LADDER PASS`, `ADD LADDER PLAYER`, `MOVE LADDER PLAYER`, `REPLACE LADDER TIER`, `SAVE LADDER OVERRIDES`, and `PUBLISH LADDER RESULT`. Recovery uses `RECONCILE LADDER OPERATION`.

Moneyball official publish uses `SAVE MONEYBALL`. Recovery uses `RECONCILE MONEYBALL`.

JUPR Live uses `CREATE LIVE SESSION`, `SAVE LIVE SESSION`, `SAVE LIVE SCORES`, `ADVANCE LIVE ROUND`, and `PUBLISH LIVE MATCHES`. Recovery uses `RECONCILE LIVE OPERATION`.

## Manual Challenge Ladder lifecycle

1. Create isolated ladder players/roster rows and record their starting ranks. Do not use active club-night data.
2. Load the Python dashboard and record its `state_version`.
3. Create a challenge, copy the generated notice, start the clock, accept it, and confirm public ladder state.
4. Preview two played partner matches. Change one score and confirm the reviewed preview becomes invalid. Preview again.
5. Publish with `PUBLISH LADDER RESULT`. Confirm exactly two `challenge_ladder` match rows, the expected winner/rank result, completed challenge state, intent/completion audits, and one completed durable operation.
6. Repeat the exact HTTP request/idempotency key or use reconciliation. Confirm the stored response returns with `idempotent_replay: true` and no additional match/rank/audit-domain mutation occurs.
7. Exercise forfeit, monthly pass, roster add/move, whole-tier preview/apply, vacation/reinstate, and tier-movement review with fresh dashboard versions. Confirm the tier and result previews perform no inserts/updates.
8. Correct the disposable official matches through Match Log, run Replay History, verify ratings/ranks, and restore the fixture roster.

## Manual Moneyball lifecycle

1. Select exactly eight disposable players and deliberately order them P1 through P8. Verify changing the order changes the Python preview fingerprint and schedule.
2. Compare every scheduled team, round/court, expected win percentage, and expected score with Streamlit.
3. Enter non-tied scores. Review the Python settlement table, including player names, games, net amount, owes/receives direction, zero-sum total, excluded ties, and `settlement_fingerprint`.
4. Change a score after review and confirm official publish is blocked until a new settlement is reviewed.
5. Publish once with `SAVE MONEYBALL`. Verify submitted count, exact `moneyball` contexts, audit intent/completion, and operation result snapshot.
6. Submit the identical request again and reconcile with `RECONCILE MONEYBALL`; both must return the stored result without a second match.
7. Correct/delete the disposable rows only in Match Log, then run and verify Replay History. Never submit the night again as a correction.

## Manual one-off JUPR Live lifecycle

1. Create a four-player linked Round Robin with `CREATE LIVE SESSION`. Confirm a tournament event type is rejected with a Tournament Live/Ops handoff.
2. Open the public route using the resolved club slug, not the opaque club ID. Confirm the created title, teams, and empty scores match admin state.
3. Save a score with `SAVE LIVE SCORES`; refresh admin and public views and confirm the same sanitized score appears publicly. Confirm listing/refresh performs no cleanup write.
4. Create a League/Ladder fixture, save the current round, and advance it with `ADVANCE LIVE ROUND`. Compare the next Python-generated court/round state with Streamlit.
5. Publish one scored doubles match with `PUBLISH LIVE MATCHES`. FastAPI first claims the session version with a visible pending operation, then submits official matches and clears the pending marker on verified completion.
6. Repeat the exact publish request and reconcile with `RECONCILE LIVE OPERATION`. Confirm one official context and a stored response, not a duplicate.
7. Confirm the admin-created/scored session remains visible through `/clubs/{club_slug}/live-sessions/{session_key}`. Complete/archive the disposable session explicitly.
8. Correct the official row through Match Log and verify Replay History. If a publish remains uncertain, preserve its pending marker and operation row until evidence resolves it.

Expired-session cleanup must always receive an explicit `club_id`. A read/list route must never opportunistically abandon sessions.

## Response-loss and uncertainty drill

1. Capture the operation key shown before the request.
2. Simulate a lost response after mutation or force completion-audit failure in the automated fault test.
3. Reload `GET .../operations/{operation_key}`. If a result snapshot exists, use the surface's exact reconcile phrase; FastAPI writes the missing completion audit and returns the snapshot.
4. If no snapshot exists, the API truthfully returns `outcome: uncertain`. Stop. Open the concrete Match Log context(s), inspect whether official rows exist, correct as necessary, and run Replay History.
5. Do not create a new idempotency key to bypass `recovery_required`. The active-version lease intentionally blocks another owner of that snapshot.

For JUPR Live, inspect the private admin session's `state.official_publish.pending_operation_key` and `pending_live_match_ids` (the public projection intentionally omits them). Their presence is evidence of an interrupted publish reservation, not proof that match insertion failed.

## Automated evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache-parity-live-ladder uv run --with pytest \
  --with-requirements requirements.txt \
  --with-requirements services/api/requirements.txt \
  python -m pytest -q \
  tests/test_admin_live_ladder_operation_service.py \
  tests/test_api_contract_admin_moneyball.py \
  tests/test_api_contract_admin_jupr_live.py \
  tests/test_api_contract_admin_jupr_live_full.py \
  tests/test_live_session_state.py \
  tests/test_api_contract_admin_challenge_ladder_full.py

cd apps/web
npm ci --cache /tmp/npm-cache-parity-live-ladder
npx tsc --noEmit
npm run build
npx playwright test e2e/live-ladder.staging.spec.ts
```

The default browser test is non-mutating. To run the disposable lifecycle, set `JUPR_RUN_LIVE_LADDER_MUTATION_E2E=1`, `STAGING_ADMIN_BEARER_TOKEN`, `JUPR_LIVE_LADDER_E2E_RUN_ID`, and the fixture variables named in `live-ladder.staging.spec.ts`. Use a unique run ID. The suite asserts the configured API host contains `staging`; nevertheless, review the origin before enabling the opt-in.

## Rollback and sign-off

To stop immediately, turn off the affected `JUPR_ENABLE_STAGING_NEXT_ADMIN_*_WRITES` flag. This does not erase evidence and leaves Streamlit available. Do not drop the ledger or unique context index while any row is `intent`, `running`, `mutated`, or `recovery_required`. Resolve those rows first.

Record the staging API origin, web deployment, migration head, fixture IDs, operation keys, Match Log contexts, replay job IDs, tester, date, and outcome for each surface. Promote no matrix row to `Done` until all three manual lifecycles, the response-loss drill, mobile/keyboard review, correction/replay cleanup, and a second-person audit review are signed off.
