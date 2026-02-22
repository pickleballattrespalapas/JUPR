-- ========================================
-- TOURNAMENT ROUND ROBIN UNIQUENESS
-- ========================================

CREATE UNIQUE INDEX IF NOT EXISTS
tournament_rr_unique_slot
ON tournament_games (
    tournament_id,
    stage,
    rr_round_number,
    rr_slot_number
)
WHERE stage = 'ROUND_ROBIN';


-- ========================================
-- TOURNAMENT PLAYOFF UNIQUENESS
-- ========================================

CREATE UNIQUE INDEX IF NOT EXISTS
tournament_playoff_unique_slot
ON tournament_games (
    tournament_id,
    stage,
    playoff_game_code,
    COALESCE(series_game_number, 1)
)
WHERE stage = 'PLAYOFF';


-- ========================================
-- LEAGUE MATCH UNIQUENESS
-- (Prevent duplicate league match entries)
-- ========================================

CREATE UNIQUE INDEX IF NOT EXISTS
league_match_unique
ON matches (
    club_id,
    context_type,
    context_id,
    tournament_game_id
)
WHERE context_type = 'TOURNAMENT';


-- ========================================
-- GENERIC MATCH IDEMPOTENCY
-- (Prevents double-submit of same match payload)
-- ========================================

CREATE UNIQUE INDEX IF NOT EXISTS
matches_idempotency_unique
ON matches (
    club_id,
    idempotency_key
)
WHERE idempotency_key IS NOT NULL;
