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
2. `supabase/migrations/20260720014744_seed_top_performer_badges.sql` for all four top-performer badge definitions. The earlier `migrations/20260215_end_league_top_performers.sql` remains the source of the canonical copy, while the new Supabase migration is the deployable, insert-missing-only seed. The `player_badges` unique-context contract must already be present. Badge readiness is a hard, fail-closed mint prerequisite: runtime readiness must report `badge_definitions_ready=true` and `4/4` before smoke testing.
3. `supabase/migrations/20260428101000_admin_activity_log.sql` for required staging audit evidence.
4. Staging FastAPI secrets/config: `SUPABASE_SERVICE_ROLE_KEY`, Supabase JWT verification settings, `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1`, `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE=1`, and `JUPR_REQUIRE_API_AUDIT_LOG=1`.
5. Keep `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE` absent/false in production until the manual gate is signed off.

No migration or external environment mutation is performed by this PR.

Read-only Supabase inspection on 2026-07-19 confirmed that JUPR Staging has `end_awards jsonb`, the four-column `player_badges` unique-context index, `admin_activity_log`, and RLS enabled on both workflow tables. It also found **0 of 4 required top-performer badge definitions**. Mint smoke is therefore intentionally blocked until `supabase/migrations/20260720014744_seed_top_performer_badges.sql` seeds `top_performer_highest_rating`, `top_performer_most_improved`, `top_performer_best_win_pct`, and `top_performer_most_wins` in staging. The migration inserts only missing IDs, preserves any customized/frozen existing row, aligns any present legacy `_v2` compatibility columns on newly inserted rows, and aborts if the complete set is not present afterward. This PR adds the migration but does not apply it or otherwise mutate live staging.

### Badge-seed preflight and verification

Before applying the migration, keep `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE=0` and capture the current definitions:

```sql
select badge_id, name, prestige, category, is_stackable, is_active,
       rarity, tier, icon_key, lore, hint, scope, state
from public.badges
where badge_id in (
  'top_performer_highest_rating',
  'top_performer_most_improved',
  'top_performer_best_win_pct',
  'top_performer_most_wins'
)
order by badge_id;
```

Review `supabase db push --dry-run` against the isolated staging project, then apply migrations through the normal single-operator release path. Do not copy the legacy SQL into the Dashboard. Existing matching IDs are intentionally not overwritten.

Some older badge schemas carry compatibility columns without a synchronization trigger. Inventory them before applying; absent columns are valid and require no repair:

```sql
select column_name, data_type, is_nullable, column_default
from information_schema.columns
where table_schema = 'public'
  and table_name = 'badges'
  and column_name in (
    'name_v2', 'prestige_v2', 'category_v2', 'is_active_v2',
    'is_stackable_v2', 'lore_v2', 'hint_v2', 'scope_v2'
  )
order by column_name;
```

After the migration, record the remote migration-ledger row and run the same projection above. It must return four distinct IDs. Also verify the deployable version directly:

```sql
select version
from supabase_migrations.schema_migrations
where version = '20260720014744';

select count(distinct badge_id) as ready_count
from public.badges
where badge_id in (
  'top_performer_highest_rating',
  'top_performer_most_improved',
  'top_performer_best_win_pct',
  'top_performer_most_wins'
);
```

On schemas with compatibility columns, this schema-tolerant query must return zero mismatches. It uses `to_jsonb` so absent `_v2` columns do not cause an error:

```sql
with target_rows as (
  select badge.badge_id, to_jsonb(badge) as row_json
  from public.badges as badge
  where badge.badge_id in (
    'top_performer_highest_rating',
    'top_performer_most_improved',
    'top_performer_best_win_pct',
    'top_performer_most_wins'
  )
), column_pairs(compatibility_column, canonical_column) as (
  values
    ('name_v2', 'name'),
    ('prestige_v2', 'prestige'),
    ('category_v2', 'category'),
    ('is_active_v2', 'is_active'),
    ('is_stackable_v2', 'is_stackable'),
    ('lore_v2', 'lore'),
    ('hint_v2', 'hint'),
    ('scope_v2', 'scope')
)
select badge_id, compatibility_column,
       row_json -> canonical_column as canonical_value,
       row_json -> compatibility_column as compatibility_value
from target_rows
cross join column_pairs
where row_json ? compatibility_column
  and row_json -> compatibility_column is distinct from row_json -> canonical_column;
```

For the audited staging state, where all four IDs were absent before this migration, require exactly one migration-ledger row, `ready_count = 4`, and zero compatibility mismatches. On a replay against an environment with pre-existing target IDs, require those rows to match the captured preflight snapshot instead; the migration intentionally does not rewrite them. Then recover the Awards page and require its readiness payload to report `badge_definitions_ready=true`, `badge_definition_count=4`, and no missing IDs before enabling the staging write gate. On any mismatch, keep the gate closed. Rollback is disable-first: do not delete definitions that may already be referenced by `player_badges`.

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
