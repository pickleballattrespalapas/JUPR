-- Prevent duplicate round robin slots
CREATE UNIQUE INDEX IF NOT EXISTS
tournament_rr_unique_slot
ON tournament_games (
    tournament_id,
    stage,
    rr_round_number,
    rr_slot_number
)
WHERE stage = 'ROUND_ROBIN';

-- Prevent duplicate playoff games
CREATE UNIQUE INDEX IF NOT EXISTS
tournament_playoff_unique_slot
ON tournament_games (
    tournament_id,
    stage,
    playoff_game_code,
    COALESCE(series_game_number, 1)
)
WHERE stage = 'PLAYOFF';
