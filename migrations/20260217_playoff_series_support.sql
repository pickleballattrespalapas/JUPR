-- Add playoff format setting at tournament level
ALTER TABLE public.tournaments
ADD COLUMN IF NOT EXISTS playoff_best_of integer NOT NULL DEFAULT 1;

-- Add series game tracking to tournament games
ALTER TABLE public.tournament_games
ADD COLUMN IF NOT EXISTS series_game_number integer NULL;

-- Optional helpful index
CREATE INDEX IF NOT EXISTS idx_tournament_games_series
ON public.tournament_games (tournament_id, playoff_game_code, series_game_number);
