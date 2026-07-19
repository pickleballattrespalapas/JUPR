# League Awards parity evidence

## Status

Implementation is complete for the order-19 League Awards slice. **Parity remains Partial** until the combined manual staging gate is run with the rest of the remaining pages. Streamlit remains the operator fallback throughout that gate.

## Delivered contract

- `GET .../awards` recovers the latest persisted wizard revision from `leagues_metadata.end_awards`.
- `POST .../awards/freeze` ends the active/paused league and persists the freeze actor/time.
- `POST .../awards/preview` runs the existing Python top-performer calculation and persists the exact award rows plus a SHA-256 fingerprint.
- `POST .../awards/overrides` rejects a stale preview, restricts winners to the league roster, and requires a persisted reason whenever the computed winner changes.
- `POST .../awards/mint` mints through the existing Python badge domain, then reads back every expected `player_badges` key. Missing or unreadable rows leave the wizard in `mint_failed`; the API never returns a false-success mint result.
- Every awards response reports required badge-definition readiness. The mint endpoint fails before audit intent, attempt accounting, or database mutation when any of the four Python-authoritative definitions is missing or unreadable; the browser disables mint at the same boundary.
- `POST .../awards/archive` remains blocked until mint verification succeeds, then persists archive attribution and the league lifecycle state.
- Each mutation carries a bounded idempotency key. A failed/running mint is retryable, while verified mint/archive calls are safe idempotent replays.
- No browser code talks to Supabase tables. The write gate additionally requires `SUPABASE_SERVICE_ROLE_KEY` on FastAPI; the secret is never a Vercel variable.

The workflow is stored in the existing `leagues_metadata.end_awards` JSONB field (schema version 2). This avoids a new exposed table and preserves the legacy `top_performers` projection used by Streamlit and public trophy displays.

## Automated evidence

Run from the repository root:

```bash
python -m pytest -q \
  tests/test_api_contract_admin_league_awards.py \
  tests/test_league_awards_static_contract.py \
  tests/test_end_league_top_performers.py

python -m compileall -q jupr_app services tests

cd apps/web
npx tsc --noEmit
npm run build
JUPR_RUN_LEAGUE_AWARDS_UI_E2E=1 npm run test:e2e:staging -- league-awards.staging.spec.ts
```

The focused browser test intercepts FastAPI responses and verifies browser state recovery and all six controls without writing staging data. The real-write test is intentionally deferred to the manual gate below.

## Runtime prerequisites

Apply/verify these existing contracts in isolated staging before opening the write gate:

1. `migrations/20260701_league_manager_end_wizard_columns.sql` (especially `leagues_metadata.status`, `ended_at`, `ended_by`, `end_awards`, and `awards_config`). Schema-changing migrations should ultimately be mirrored into the canonical `supabase/migrations/` history before production rollout.
2. `migrations/20260215_end_league_top_performers.sql` for all four top-performer badge definitions and the `player_badges` unique context contract. This is a hard, fail-closed mint prerequisite: runtime readiness must report `badge_definitions_ready=true` and `4/4` before smoke testing.
3. `supabase/migrations/20260428101000_admin_activity_log.sql` for required staging audit evidence.
4. Staging FastAPI secrets/config: `SUPABASE_SERVICE_ROLE_KEY`, Supabase JWT verification settings, `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1`, `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE=1`, and `JUPR_REQUIRE_API_AUDIT_LOG=1`.
5. Keep `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE` absent/false in production until the manual gate is signed off.

No migration or external environment mutation is performed by this PR.

Read-only Supabase inspection on 2026-07-19 confirmed that JUPR Staging has `end_awards jsonb`, the four-column `player_badges` unique-context index, `admin_activity_log`, and RLS enabled on both workflow tables. It also found **0 of 4 required top-performer badge definitions**. Mint smoke is therefore intentionally blocked until `migrations/20260215_end_league_top_performers.sql` (or its reviewed canonical Supabase equivalent) seeds `top_performer_highest_rating`, `top_performer_most_improved`, `top_performer_best_win_pct`, and `top_performer_most_wins` in staging. This PR does not seed or otherwise mutate live staging.

## Deferred manual staging gate

Use a disposable, isolated staging league with at least three players and enough league-rating rows to satisfy the configured minimum games.

1. Confirm the page identifies staging, the staging API/Auth origins are isolated, the signed-in role has `manage_matches`, and recovered awards state reports all four badge definitions ready. Stop if the red mint-readiness warning appears.
2. Recover the league at `not started`, refresh the browser, and confirm the same revision/state returns.
3. Type `FREEZE LEAGUE AWARDS`; confirm the league becomes ended and Streamlit shows the same lifecycle state.
4. Persist the preview; record its fingerprint and compare every category/rank/player/metric with the Streamlit/Python preview.
5. Change one winner with a meaningful reason. Confirm a missing/short reason fails, a stale fingerprint fails, and refresh recovers the saved winner and reason.
6. In a disposable copy, deliberately make `player_badges` unavailable to the staging API or use a controlled test double. Confirm mint returns an error, the page recovers `mint failed`, and no success copy appears.
7. Restore the dependency and retry with the same operation key. Confirm expected equals verified, each expected context exists exactly once, and a second retry creates no duplicate.
8. Confirm archive is blocked before verified mint, succeeds after verified mint, and a repeated archive request is an idempotent replay.
9. Inspect `admin_activity_log` for freeze, preview, override, verified mint, and archive attribution with no access token, secret, or unrelated player data.
10. Confirm the Streamlit League Manager remains usable as fallback and that production still has the League Awards write flag off.

Save screenshots/API payload excerpts with secrets redacted. Only after this manual staging gate and the final combined page test book pass should the page be marked Done.

## Recovery / false-success rules

- `minting` after a process interruption is retryable from the persisted final awards.
- `mint_failed` includes a bounded admin-visible error and retained attempt history; it is not success.
- `minted` means every expected `(player_id, badge_id, context_id)` was read back.
- `archived` requires `minted` with a verified result (or the legacy explicit no-mint compatibility path, which the new UI does not expose).
- Preview/override changes lock after the first mint attempt to prevent winner changes from creating competing badge contexts.
