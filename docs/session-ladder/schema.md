# Session Ladder Engine MVP Schema

This document summarizes the MVP persistence model introduced for Session Ladder Engine.

## Existing entities inspected before design
The current database already includes (from live schema snapshot / migrations):
- `clubs`
- `players`
- `league_ratings`
- `leagues_metadata`
- `events`

There is **no canonical `seasons` table** or `rating_history` table currently present in the checked-in schema snapshots/migrations, so `season_id` is stored as nullable text for forward compatibility.

## New tables

## 1) `session_ladder_sessions`
Represents a single run-night/session.

Key columns:
- `id` (uuid pk)
- `club_id` (FK to `clubs.id`)
- `league_id` (text, **league key** used for rating partition continuity)
- `season_id` (nullable text)
- `session_starts_at`, `session_ends_at`
- `courts_available`
- `players_per_court` (check: 4 or 5)
- `state` (`draft/roster_open/seeded_locked/round_1_active/round_1_closed/round_2_active/round_2_closed/round_3_active/round_3_closed/completed/published`)
- audit: `created_by`, `updated_by`, `created_at`, `updated_at`

Indexes:
- `(club_id, session_starts_at desc)`
- `(club_id, state, session_starts_at desc)`

## 2) `session_ladder_roster_entries`
One row per player expected/checked in for a session.

Key columns:
- `session_id` FK to `session_ladder_sessions`
- `player_id` FK to `players`
- `status` (`EXPECTED/CHECKED_IN/NO_SHOW/WALK_IN`)
- `rating_snapshot` (numeric)
- `seed_order` (nullable int)
- audit: `created_by`, `updated_by`, `created_at`, `updated_at`

Constraints/indexes:
- unique `(session_id, player_id)`
- index `(session_id, status, seed_order)`
- index `(club_id, player_id)`

## 3) `session_ladder_court_pods`
Represents one court assignment for one round.

Key columns:
- `session_id` FK
- `round_number` (> 0)
- `court_number` (> 0)
- `state` (`planned/in_progress/complete/void`)
- audit fields

Constraints/indexes:
- unique `(session_id, round_number, court_number)`
- indexes for `(session_id, round_number, court_number)` and `(club_id, round_number, court_number)`

## 4) `session_ladder_court_pod_players`
Players assigned to each court pod with display order/label.

Key columns:
- `court_pod_id` FK
- `session_id` FK
- `player_id` FK
- `player_label`, `player_order`

Constraints/indexes:
- unique `(court_pod_id, player_order)`
- unique `(court_pod_id, player_id)`
- indexes for session and player lookup

## 5) `session_ladder_games`
First-class game records for score capture.

Key columns:
- `court_pod_id` FK
- `session_id` FK
- `game_number`
- `team_a_player_ids bigint[2]`
- `team_b_player_ids bigint[2]`
- `score_a`, `score_b`
- `edited_by`, timestamps

Constraints/indexes:
- unique `(court_pod_id, game_number)`
- checks for non-negative scores and 2 players per team
- indexes for session/court and club/time queries

## 6) `session_ladder_round_stats_cache` (optional cache)
Derived per-round stats cache (recomputable from games).

Key columns:
- `session_id`, `round_number`, `player_id`
- `wins`, `losses`, `points_for`, `points_against`, `point_differential`
- `source_hash`, `computed_at`, `updated_at`

Constraint:
- unique `(session_id, round_number, player_id)`

## Integrity + audit behavior
- Child-table guard triggers auto-fill/validate `club_id` (and `session_id` for court-pod children) from parent records.
- `updated_at` triggers are attached to mutable tables.

## RLS model
RLS is enabled for all Session Ladder tables:
- **Read**: club-scoped rows only (`club_id = jwt_club_id`).
- **Write**: allowed when `session_ladder_can_write(club_id)` returns true.
  - grants write for JWT role `admin`/`manager`, or
  - `club_user_roles` entries for `admin`/`coordinator`/`score_entry`.

This follows existing tenant-club policy patterns while adding manager/admin write semantics requested for Session Ladder operations.

## Completion & publish additions

- `session_ladder_sessions` now includes:
  - `rounds_planned` (`2` or `3`)
  - `ratings_applied_at`
  - `published_at`
  - `recap_json`
  - `leaderboard_json`
- `session_ladder_rating_history` stores per-player session rating deltas (idempotent key: `session_id + player_id`).
- `session_ladder_attendance` stores per-session attendance facts.
- `session_ladder_awards_attendance` stores cumulative attendance counts by `(club_id, league_id, season_id, player_id)` for awards eligibility checks.

## League key and rating continuity contract

- Session Ladder stores the **rating partition key** in `session_ladder_sessions.league_id`.
- This key must match `league_ratings.league_name` exactly for ratings to continue in the same league partition.
- Session creation resolves/validates this value against `leagues_metadata.league_name` (case-insensitive lookup, canonical stored casing) before the session row is created.
