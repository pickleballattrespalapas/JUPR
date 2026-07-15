-- Guard official tournament-game publishing so each tournament game maps to at most one official match.

CREATE UNIQUE INDEX IF NOT EXISTS idx_matches_unique_tournament_game_id
  ON public.matches (tournament_game_id)
  WHERE tournament_game_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_matches_tournament_game_id
  ON public.matches (tournament_game_id)
  WHERE tournament_game_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_matches_tournament_draw_context
  ON public.matches (tournament_id, context_id)
  WHERE tournament_id IS NOT NULL;
