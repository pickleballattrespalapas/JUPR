-- =========================================
-- MATCH IDEMPOTENCY (GLOBAL)
-- =========================================

CREATE UNIQUE INDEX IF NOT EXISTS
matches_idempotency_unique
ON matches (
    club_id,
    idempotency_key
)
WHERE idempotency_key IS NOT NULL;

-- =========================================
-- TOURNAMENT ROUND ROBIN
-- =========================================

CREATE UNIQUE INDEX IF NOT EXISTS
tournament_rr_unique_slot
ON tournament_games (
    tournament_id,
    stage,
    rr_round_number,
    rr_slot_number
)
WHERE stage = 'ROUND_ROBIN';

-- =========================================
-- TOURNAMENT PLAYOFF
-- =========================================

CREATE UNIQUE INDEX IF NOT EXISTS
tournament_playoff_unique_slot
ON tournament_games (
    tournament_id,
    stage,
    playoff_game_code,
    COALESCE(series_game_number, 1)
)
WHERE stage = 'PLAYOFF';

-- =========================================
-- TOURNAMENT GAME → MATCH LINK
-- =========================================

CREATE UNIQUE INDEX IF NOT EXISTS
matches_unique_tournament_game
ON matches (
    club_id,
    tournament_game_id
)
WHERE tournament_game_id IS NOT NULL;
