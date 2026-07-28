# Migration source of truth

## Canonical location

`supabase/migrations/` is the **canonical location** for schema migrations that must exist in Supabase.
Migration prefixes in this folder must be unique across files.
Use `YYYYMMDDHHMMSS_name.sql` to avoid collisions; avoid date-only prefixes like `YYYYMMDD`.

- If a migration changes Supabase schema (tables, columns, constraints, indexes, functions, grants, triggers, policies), the SQL must be captured in `supabase/migrations/`.
- Do **not** rely on `migrations/` alone for Supabase schema changes.

The Partner Board lifecycle transaction is captured in
`supabase/migrations/20260719194500_public_partner_pairing_lifecycle.sql`. It
builds on the partner-link tables provisioned by
`migrations/20261019_tournament_registration_partner_links.sql` and the canonical
selection lock protocol already captured under `supabase/migrations/`.

## Current inventory (as of 2026-07-28)

Inventory below reflects files currently present in this repository.

### `supabase/migrations/` (canonical)

This directory contains 53 SQL files: 52 deployable migrations with numeric
versions plus one deliberately non-deployable `2026XX` schema placeholder.

- `20260420000000_verified_player_updates_foundation.sql`
- `20260424_matches_soft_delete.sql`
- `20260428090000_add_unsubscribe_token_to_player_profile_update_subscriptions.sql`
- `20260428100000_admin_role_assignments.sql`
- `20260428101000_admin_activity_log.sql`
- `20260502120000_replay_jobs.sql`
- `20260502121500_clubs_config.sql`
- `20260502133000_public_leaderboards_view.sql`
- `20260511120000_admin_role_assignments_club_scope.sql`
- `20260511143000_clubs_saas_onboarding_fields.sql`
- `20260511170000_worker_run_log.sql`
- `20260624000000_confirm_tournament_registrations.sql`
- `20260702080000_live_sessions.sql`
- `20260702170000_live_sessions_schema_contract.sql`
- `20260715165000_admin_match_log_duplicate_resolutions.sql`
- `20260717141224_selection_update_transaction_guards.sql`
- `20260717142402_selection_relationship_update_lock_scope.sql`
- `20260718141016_badge_eval_queue_atomic_club_claim.sql`
- `20260719155515_server_only_data_api_lockdown.sql`
- `20260719155737_canonicalize_server_only_tables.sql`
- `20260719160821_public_registration_edit_transaction.sql`
- `20260719171000_public_support_intake_guardrails.sql`
- `20260719172000_replay_job_idempotency.sql`
- `20260719182606_communications_outbox_stale_guards.sql`
- `20260719182921_league_live_domain_contract.sql`
- `20260719190954_league_live_publish_reconciliation.sql`
- `20260719193000_admin_player_merge_transactions.sql`
- `20260719194500_public_partner_pairing_lifecycle.sql`
- `20260719201500_live_ladder_admin_operations.sql`
- `20260719203000_tournament_admin_operations.sql`
- `20260719204500_admin_diagnostics_guarded_operations.sql`
- `20260719204700_tournament_operations_guard_surface.sql`
- `20260719205000_tournament_live_operations.sql`
- `20260719220000_public_live_durability.sql`
- `20260720014744_seed_top_performer_badges.sql`
- `20260720123402_baseline_worker_run_log.sql`
- `20260725181500_singles_replay_recovery.sql`
- `20260725231000_challenge_ladder_public_results.sql`
- `20260726143742_match_exclusion_atomic_recovery.sql`
- `20260726183000_seed_live_badge_catalog.sql`
- `20260726204954_match_exclusion_nonretryable_conflicts.sql`
- `20260726212447_replay_projection_nonretryable_conflicts.sql`
- `20260726222339_atomic_direct_match_entry.sql`
- `20260727183000_atomic_league_roster_batch.sql`
- `20260727210000_match_log_club_recovery_mutex.sql`
- `20260727211500_direct_match_active_league_guard.sql`
- `20260728010000_tournament_commerce.sql`
- `20260728020000_combined_rating_team_tournaments.sql`
- `20260728030000_team_leagues_and_awards.sql`
- `20260728040000_team_league_awards_hardening.sql`
- `20260728041000_team_league_registration_identity_recovery.sql`
- `20261020000000_tournament_registrations_player_id_postgrest_reload.sql`
- `2026XX_backport_tournament_engine.sql`

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
- `20261019_tournament_registration_partner_links.sql`
- `20261020_tournament_registrations_player_id_postgrest_reload.sql`
- `20261028_public_support_intake_guardrails.sql`

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

## Manual SQL apply checklist (staging then production)

Staging Supabase exists and is required for staging-first validation. Do not auto-apply SQL blindly; review and apply intentionally per environment.

For schema changes:

1. Review SQL in the PR.
2. Paste and run SQL in **Supabase SQL editor** (staging project first).
3. Confirm expected table/column/constraint exists in staging.
4. Deploy the `staging` branch application and run staging smoke checks.
5. Promote to production only after staging verification; then run reviewed SQL in production.

The communications migration must precede the corresponding API/web release. Verify the four `row_version` columns, the outbox `sending` state, and `replace_verified_update_subscription`; confirm `anon` and `authenticated` cannot access the four server-only tables or execute the replacement RPC.

## Guardrail for future changes

Use the guard scripts:

```bash
python scripts/check_migration_sources.py
python scripts/check_supabase_migration_versions.py
```

Behavior:

- `check_migration_sources.py` detects newly-added `migrations/*.sql` files in the current branch (compared to `origin/staging` by default).
- `check_migration_sources.py` fails unless each new root migration is documented in `docs/migrations_root_explanations.md`.
- `check_supabase_migration_versions.py` fails if two `supabase/migrations/*.sql` files share the same version prefix (text before the first underscore).
- This keeps contributors from accidentally treating `migrations/` as the primary Supabase source and prevents Supabase `schema_migrations` version collisions.
