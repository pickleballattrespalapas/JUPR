# Migration source of truth

## Canonical location

`supabase/migrations/` is the **canonical location** for schema migrations that must exist in Supabase.

- If a migration changes Supabase schema (tables, columns, constraints, indexes, functions, grants, triggers, policies), the SQL must be captured in `supabase/migrations/`.
- Do **not** rely on `migrations/` alone for Supabase schema changes.

## Current inventory (as of 2026-04-28)

Inventory below reflects files currently present in this repository.

### `supabase/migrations/` (canonical)

- `20260420000000_verified_player_updates_foundation.sql`
- `20260424_matches_soft_delete.sql`
- `20260428_add_unsubscribe_token_to_player_profile_update_subscriptions.sql`
- `2026XX_backport_tournament_engine.sql`

Note (2026-04-28):

- Verified player update schema migrations originally lived only in `migrations/` (`20260420_verified_player_updates.sql`, `20260421_verified_player_updates_date_window_uniqueness.sql`, and `20260421_player_update_outbox_allow_repeat_queue_rows.sql`).
- Supabase migration runs only read `supabase/migrations/`, so we backfilled their schema intent into `supabase/migrations/20260420000000_verified_player_updates_foundation.sql` to guarantee the base tables exist before later unsubscribe/update migrations run.

### `migrations/` (legacy/archive)

The repository still contains a historical `migrations/` folder with many SQL files.
Treat this folder as **legacy/archive** unless a file is explicitly documented for non-Supabase usage.

Current files in `migrations/`:

- `20250220_badges_v1.sql`
- `20250301_player_inactivity.sql`
- `20250320_gamification_v2.sql`
- `20250325_player_badges_dedupe.sql`
- `20250410_badges_copy_pack.sql`
- `20250415_gamification_story_badges.sql`
- `20250420_badge_definitions_seed.sql`
- `20250425_tournaments.sql`
- `20250505_tournament_podium.sql`
- `20260130_unique_leagues_metadata.sql`
- `20260207_weekly_recaps.sql`
- `20260215_end_league_top_performers.sql`
- `20260301_bulk_replay_rpcs.sql`
- `20260301_premium_league_editor.sql`
- `20260309_tournament_registration.sql`
- `20260310_tournament_event_draws.sql`
- `20260329_live_social_events.sql`
- `20260330_live_social_moderation_columns.sql`
- `20260417_live_events_submitted_by_name_canonical.sql`
- `20260417_live_events_submitted_by_name_canonical_rollback.sql`
- `20260417_tournament_builder_draft_snapshot.sql`
- `20260420_verified_player_updates.sql`
- `20260421_player_update_outbox_allow_repeat_queue_rows.sql`
- `20260421_verified_player_updates_date_window_uniqueness.sql`
- `20260615_player_badges_unique_contract.sql`
- `20260620_badge_state.sql`
- `20260625_badge_recompute_runs.sql`
- `20260630_player_badges_revocation.sql`
- `20260705_badge_eval_queue.sql`
- `20260712_badge_eval_triggers.sql`
- `20260715_perf_indexes.sql`
- `20260720_badge_eval_queue_backfill.sql`
- `20260720_player_badges_provenance_grants.sql`
- `20260725_events.sql`
- `20260801_badge_queue_and_facts_grants.sql`
- `20260805_player_badges_awarded_by_text.sql`
- `20260810_player_badges_awarded_by_text.sql`
- `20261010_tournament_builder_refactor.sql`
- `20261011_tournament_dates.sql`
- `20261018_tournament_registration_schema_contract.sql`

Important:

- New Supabase schema changes must be added to `supabase/migrations/`.
- If a SQL file is added under `migrations/`, it must be treated as legacy/special-case and documented in `docs/migrations_root_explanations.md`.

## Handling duplicates or overlaps

We are intentionally **not** deleting or moving old SQL files aggressively in this change.

Potential overlap area currently observed:

- `supabase/migrations/2026XX_backport_tournament_engine.sql`
- Multiple legacy tournament-related files in `migrations/` (for example `20250425_tournaments.sql`, `20260309_tournament_registration.sql`, `20261010_tournament_builder_refactor.sql`).
- Legacy root migrations that appear to target the same badge-award text change pattern:
  - `migrations/20260805_player_badges_awarded_by_text.sql`
  - `migrations/20260810_player_badges_awarded_by_text.sql`

If you discover clear duplicates, document them first in this file (or PR notes) before any cleanup.

## Manual SQL apply checklist (production Supabase)

There is currently no staging Supabase. Do not auto-apply SQL from this repo.

For schema changes:

1. Review SQL in the PR.
2. Paste and run SQL in **Supabase SQL editor** (production project).
3. Confirm expected table/column/constraint exists.
4. Deploy the `Test` branch application.
5. Run smoke tests on key flows.

## Guardrail for future changes

Use the guard script:

```bash
python scripts/check_migration_sources.py
python scripts/check_player_update_migration_foundation.py
```

Behavior:

- Detects newly-added `migrations/*.sql` files in the current branch (compared to `Test` by default).
- Fails unless each new root migration is documented in `docs/migrations_root_explanations.md`.
- This keeps contributors from accidentally treating `migrations/` as the primary Supabase source.
