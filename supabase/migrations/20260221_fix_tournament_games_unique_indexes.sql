-- Ensure correct tenant-scoped uniqueness for tournament_games
-- Safe forward-only migration

DROP INDEX IF EXISTS tournament_games_rr_unique;
DROP INDEX IF EXISTS tournament_games_playoff_unique;

CREATE UNIQUE INDEX tournament_games_rr_unique
ON public.tournament_games (
    club_id,
    tournament_id,
    stage,
    rr_round_number,
    rr_slot_number
)
WHERE stage = 'ROUND_ROBIN';

CREATE UNIQUE INDEX tournament_games_playoff_unique
ON public.tournament_games (
    club_id,
    tournament_id,
    stage,
    playoff_game_code,
    series_game_number
)
WHERE stage = 'PLAYOFF';
