# Supabase staging advisor review — 2026-07-24

This is a read-only review of Supabase project `sijpxjxvdtrehmqvirfi` (`JUPR
Staging`). It records preparation work for the staging candidate; it does not
authorize schema changes and it does not describe production state.

## Snapshot

- Project status: `ACTIVE_HEALTHY`
- PostgreSQL: `17.6.1`
- Applied migration ledger: 37 entries
- Current migration head: `20260720123402_baseline_worker_run_log`
- Security advisor: 72 findings
  - 71 `INFO` — RLS enabled with no policy
  - 1 `WARN` — leaked-password protection disabled
- Performance advisor: 135 findings
  - 46 `INFO` — unindexed foreign keys
  - 80 `INFO` — unused indexes
  - 8 `WARN` — duplicate indexes
  - 1 `INFO` — Auth database connection strategy

No advisor-driven DDL was applied during this review. Index scan counters are a
point-in-time operational hint, not proof that an index is safe to remove.

## Security disposition

The 71 RLS-with-no-policy findings were reviewed as a category and are consistent
with the application's service-role-only pattern for private administrative
tables: RLS with no browser policy is fail-closed for Data API clients. This is an
architectural inference, not an individual grant attestation for all 71 tables.
Before changing any finding, verify that table's grants, API exposure, and
FastAPI service-role access together. Do not add permissive policies merely to
clear the advisor.

Leaked-password protection remains an owner-controlled Supabase Auth setting.
Enable it only after Joe confirms the project plan and authentication behavior,
then verify sign-up, password change, and recovery against staging. This is a
public-launch prerequisite, not an engineering change to make silently during the
candidate freeze. See [Supabase password security](https://supabase.com/docs/guides/auth/password-security).

## Exact duplicate-index groups

The following groups have the same indexed relation, uniqueness, key/operator
definition, expression, and predicate in the staging catalog:

| Table | Equivalent indexes | Review disposition |
| --- | --- | --- |
| `ladder_player_flags` | `ladder_player_flags_pkey`; `ladder_player_flags_unique_key` | Keep the primary-key constraint. Review the secondary name and migration provenance before considering removal. |
| `ladder_roster` | `ladder_roster_tier_rank_uidx`; `ladder_roster_unique_rank_per_tier` | Both are partial unique indexes for active rows. Select one canonical migration-owned name only after query-plan regression tests. |
| `league_ratings` | `league_ratings_unique_idx`; `league_ratings_unique_key`; `uq_league_ratings_club_player_league` | Three equivalent unique indexes. Preserve one canonical index after checking rollback scripts and all historical migrations. |
| `leagues_metadata` | `leagues_metadata_club_id_league_name_key`; `uq_leagues_metadata_key` | Keep the unique-constraint index. Review the standalone duplicate separately. |
| `matches` | `matches_club_id_idempotency_key_key`; `matches_club_id_idempotency_key_uq`; `uq_matches_club_idempotency` | Three equivalent partial unique idempotency indexes. Preserve enforcement continuously and validate write/reconciliation plans before cleanup. |
| `player_badges` | `player_badges_unique_key`; `uq_player_badges_award_key` | Keep one only after badge award/recompute plans are tested. |
| `player_profile_update_outbox` | `player_profile_update_outbox_subscription_week_window_idx`; `player_profile_update_outbox_subscription_window_idx` | Equivalent non-unique indexes. Check worker and stale-guard query plans before cleanup. |
| `players` | `players_club_normalized_name_uidx`; `players_unique_normalized_idx` | Equivalent unique name indexes. Preserve one continuously; verify player create, rename, link, and merge paths. |

Recommended follow-up is a dedicated, reversible staging migration after formal
candidate acceptance—not during it. The migration should identify the retained
name for each group, refuse to remove constraint-backed indexes, capture
before/after query plans for affected writes and lookups, run API and replay
regression tests, and include a rollback that recreates the exact definition.

## Other performance findings

- Prioritize unindexed foreign keys by delete/update frequency and table size.
  Adding every suggested index can increase write cost without improving real
  traffic.
- Treat unused-index findings as observation candidates. Counters can reset and
  staging traffic is not representative enough to establish production safety.
- Review the Auth database connection recommendation with the Supabase project
  plan and expected concurrency; it is not an application migration.

## Release disposition

These advisor findings do not invalidate the current staging health result, but
they remain explicit release-preparation items. The public launch requires an
owner decision on leaked-password protection. Duplicate-index cleanup and general
SaaS scaling work are later-phase maintenance and do not block the initial Tres
Palapas launch unless a measured correctness or performance problem is found.
