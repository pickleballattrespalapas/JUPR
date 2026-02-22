-- Canonicalize tournament_games uniqueness
-- This removes legacy/global uniqueness definitions and installs tenant-scoped invariants.
-- Safe because duplicate check has already been confirmed clean.

-- Drop legacy/global RR uniqueness variants
DROP INDEX IF EXISTS tournament_rr_unique_slot;
DROP INDEX IF EXISTS tournament_games_rr_unique;
DROP INDEX IF EXISTS tournament_games_rr_unique_slot;

-- Drop legacy/global Playoff uniqueness variants
DROP INDEX IF EXISTS tournament_playoff_unique;
DROP INDEX IF EXISTS tournament_playoff_unique_slot;
DROP INDEX IF EXISTS tournament_games_playoff_unique;

-- Install canonical tenant-scoped RR uniqueness
CREATE UNIQUE INDEX tournament_games_rr_unique
ON public.tournament_games (
    club_id,
    tournament_id,
    stage,
    rr_round_number,
    rr_slot_number
)
WHERE stage = 'ROUND_ROBIN';

-- Install canonical tenant-scoped Playoff uniqueness
CREATE UNIQUE INDEX tournament_games_playoff_unique
ON public.tournament_games (
    club_id,
    tournament_id,
    stage,
    playoff_game_code,
    series_game_number
)
WHERE stage = 'PLAYOFF';
