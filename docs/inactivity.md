# Player Inactivity (14-day global rule)

## Overview
Players are marked **inactive** if they have not logged any recorded match in **14 days** across all leagues. Inactive players are hidden by default across leaderboards, standings, and ladders. A player with no recorded games uses `created_at` as the baseline for inactivity.

## Data model
The `players` table stores:

- `last_game_at`: the most recent match timestamp across all leagues.
- `inactive_at`: set when a player crosses the 14-day inactivity threshold. A player is active if `inactive_at IS NULL`.

## Daily job
The migration installs a daily `pg_cron` job:

- **Job name**: `mark-inactive-players-daily`
- **Schedule**: `0 3 * * *` (03:00 UTC)
- **Action**: runs `public.mark_inactive_players()` (idempotent).

To run the job manually:

```sql
SELECT public.mark_inactive_players();
```

## Backfill
The migration backfills `last_game_at` from `matches` and marks inactive players by comparing
`COALESCE(last_game_at, created_at)` to `NOW() - INTERVAL '14 days'`.

## Match deletions/voids
If matches are deleted/voided and you need to refresh activity timestamps for affected players,
use the admin match log flow. It recomputes `last_game_at` for impacted player IDs after deletions.
