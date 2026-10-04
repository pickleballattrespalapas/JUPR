# Player Inactivity (14-day global rule)

## Overview
Players are marked **inactive** if they have not logged any recorded match in **14 days** across all leagues. Inactive players are hidden by default only on overall and league leaderboards (including printable rankings). They remain searchable in the Players directory and available for registration, roster selection, check-in, live play, match history, and awards. Ladder membership, league membership, withdrawals, and merged profiles have their own separate rules. A player with no recorded games uses `created_at` as the baseline for inactivity.

## Data model
The `players` table stores:

- `last_game_at`: the most recent match timestamp across all leagues.
- `active`: manual leaderboard visibility flag.
- `inactive_at`: set when a player crosses the 14-day inactivity threshold. A player is visible on active leaderboards when `inactive_at IS NULL` and `active` is not false.

## Daily job
The canonical migration
`supabase/migrations/20261004032453_restore_daily_player_inactivity.sql`
installs the daily `pg_cron` job and runs the overdue-player sweep immediately.
The old `migrations/20250301_player_inactivity.sql` is archival and is not part
of the current deployment path.

- **Job name**: `mark-inactive-players-daily`
- **Schedule**: `0 3 * * *` (03:00 UTC)
- **Action**: runs `public.mark_inactive_players()` (idempotent).
- **Owner**: `postgres`; application roles cannot execute this maintenance function.
- **Scope**: only sets a missing `inactive_at`. It preserves manual `active` flags,
  existing inactivity timestamps, ratings, and recorded match timestamps.

To run the job manually as the database owner:

```sql
SELECT public.mark_inactive_players();
```

Check the enabled schedule and actual execution history:

```sql
SELECT jobid, jobname, schedule, command, username, active
FROM cron.job
WHERE jobname = 'mark-inactive-players-daily';

SELECT r.status, r.start_time, r.end_time, r.return_message
FROM cron.job_run_details r
JOIN cron.job j ON j.jobid = r.jobid
WHERE j.jobname = 'mark-inactive-players-daily'
ORDER BY r.start_time DESC
LIMIT 10;
```

Run `tests/sql/player_inactivity_daily_job.sql` against staging for transactional
coverage of the cutoff boundary, never-played players, manual visibility,
idempotency, permissions, and schedule. All test fixtures roll back.

## Backfill
The legacy migration backfilled `last_game_at` from `matches`. Match processing
maintains it thereafter. The restored job uses those existing timestamps and
marks inactive players by comparing `COALESCE(last_game_at, created_at)` to
`NOW() - INTERVAL '14 days'`.

## Match deletions/voids
If matches are deleted/voided and you need to refresh activity timestamps for affected players,
use the admin match log flow. It recomputes `last_game_at` for impacted player IDs after deletions.
