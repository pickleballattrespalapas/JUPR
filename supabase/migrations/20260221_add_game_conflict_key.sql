-- Add conflict key column if missing
alter table public.tournament_games
add column if not exists game_conflict_key text;

-- Backfill safely (adjust if your structure differs)
update public.tournament_games
set game_conflict_key =
    coalesce(game_conflict_key,
        tournament_id || '-' || coalesce(round::text, '') || '-' ||
        coalesce(team_a_id::text, '') || '-' ||
        coalesce(team_b_id::text, '')
    );

-- Create tenant-safe uniqueness index
create unique index if not exists
tournament_games_conflict_uidx
on public.tournament_games (club_id, game_conflict_key);
