-- Tournament Games Uniqueness Enforcement (Tenant Safe)
-- Forward-only, non-destructive migration.

-- Round Robin uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS tournament_games_rr_unique
ON public.tournament_games (
    club_id,
    tournament_id,
    stage,
    rr_round_number,
    rr_slot_number
)
WHERE stage = 'ROUND_ROBIN';

-- Playoff uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS tournament_games_playoff_unique
ON public.tournament_games (
    club_id,
    tournament_id,
    stage,
    playoff_game_code,
    series_game_number
)
WHERE stage = 'PLAYOFF';
