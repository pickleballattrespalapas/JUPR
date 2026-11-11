# Player Inactivity (14-day global rule)

## Overview
Players are marked **inactive** if they have not logged any recorded match in **14 days** across all leagues. Inactive players are hidden by default only on overall and league leaderboards (including printable rankings). They remain searchable in the Players directory and available for registration, roster selection, check-in, live play, match history, and awards. Ladder membership, league membership, withdrawals, and merged profiles have their own separate rules. A player with no recorded games starts inactive; registration or profile creation alone does not establish activity.

## Data model
The `players` table stores:

- `last_game_at`: the most recent match timestamp across all leagues.
- `active`: manual leaderboard visibility flag.
- `inactive_at`: set when a player crosses the 14-day inactivity threshold. A player is visible on active leaderboards when `inactive_at IS NULL` and `active` is not false.

## Immediate reactivation

Saving recent scored games updates activity synchronously for both singles and
doubles, including unrated play. Atomic saves include activity in the match
transaction; legacy writers refresh it before completing. The activity trigger
clears automatic `inactive_at` markers as soon as `last_game_at` is recent and
the stored manual `active` flag is not false. Results do not wait for a worker.

Date-only games may be stored at midnight, before the profile was created later
that day. A newer inactivity timestamp alone must not prevent reactivation.
Explicit manual `active=false` exclusions after the game and merged profiles
remain excluded. Older automatic markers on inactive profiles still clear when
a newer recent game is recorded.

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
- **Scope**: marks missing inactivity timestamps for stale or never-played
  profiles and repairs automatic markers for recent players. It preserves
  explicit later manual exclusions, merged profiles, ratings, and game dates.

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

Run `npm run test:player-reactivation` in `tests/sql` for the activity trigger,
same-day profile creation, repair, manual/merged exclusions, boundary cases,
idempotency, rollback, and permission checks.

## Backfill
The legacy migration backfilled `last_game_at` from `matches`. Match processing
maintains it thereafter. The current job uses `last_game_at` alone to require
a recorded game within the preceding 14 days. The migration
`supabase/migrations/20261111200814_immediate_player_game_reactivation.sql`
repairs existing automatic markers without changing ratings or results.

## Match deletions/voids
If matches are deleted/voided and you need to refresh activity timestamps for affected players,
use the admin match log flow. It recomputes `last_game_at` for impacted player IDs after deletions.
