CREATE UNIQUE INDEX IF NOT EXISTS
matches_unique_tournament_game
ON matches (
    club_id,
    tournament_game_id
)
WHERE tournament_game_id IS NOT NULL;
