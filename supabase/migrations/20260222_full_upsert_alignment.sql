-- =========================================
-- MATCHES TABLE
-- =========================================

-- Required for replay_history upsert
CREATE UNIQUE INDEX IF NOT EXISTS
matches_unique_club_id_id
ON matches (club_id, id);

-- Required for tournament sync upsert
CREATE UNIQUE INDEX IF NOT EXISTS
matches_unique_tournament_game
ON matches (club_id, tournament_game_id)
WHERE tournament_game_id IS NOT NULL;

-- Required for idempotency enforcement
CREATE UNIQUE INDEX IF NOT EXISTS
matches_unique_idempotency
ON matches (club_id, idempotency_key)
WHERE idempotency_key IS NOT NULL;


-- =========================================
-- TOURNAMENT TABLES
-- =========================================

CREATE UNIQUE INDEX IF NOT EXISTS
tournament_teams_unique
ON tournament_teams (tournament_id, team_number);

CREATE UNIQUE INDEX IF NOT EXISTS
tournament_podium_unique
ON tournament_podium (tournament_id, placement);

CREATE UNIQUE INDEX IF NOT EXISTS
tournament_rr_unique
ON tournament_games (
    tournament_id,
    stage,
    rr_round_number,
    rr_slot_number
)
WHERE stage = 'ROUND_ROBIN';

CREATE UNIQUE INDEX IF NOT EXISTS
tournament_playoff_unique
ON tournament_games (
    tournament_id,
    stage,
    playoff_game_code,
    COALESCE(series_game_number, 1)
)
WHERE stage = 'PLAYOFF';


-- =========================================
-- BADGES
-- =========================================

CREATE UNIQUE INDEX IF NOT EXISTS
badges_unique_badge_id
ON badges (badge_id);

CREATE UNIQUE INDEX IF NOT EXISTS
player_badges_unique_context
ON player_badges (club_id, player_id, badge_id, context_id);

CREATE UNIQUE INDEX IF NOT EXISTS
badge_queue_unique_match
ON badge_eval_queue (event_type, match_id)
WHERE match_id IS NOT NULL;


-- =========================================
-- WEEKLY RECAPS
-- =========================================

CREATE UNIQUE INDEX IF NOT EXISTS
weekly_recaps_unique
ON weekly_recaps (club_id, week_start);
