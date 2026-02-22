-- PostgREST-compatible ON CONFLICT target for tournament_games.
-- Replaces partial-unique-index inference (which requires ON CONFLICT ... WHERE, not supported by PostgREST).

ALTER TABLE public.tournament_games
ADD COLUMN IF NOT EXISTS game_conflict_key text;

-- Backfill for existing rows (safe, deterministic)
UPDATE public.tournament_games
SET game_conflict_key =
  CASE
    WHEN stage = 'ROUND_ROBIN'
      THEN 'RR:' || COALESCE(rr_round_number::text,'') || ':' || COALESCE(rr_slot_number::text,'')
    WHEN stage = 'PLAYOFF'
      THEN 'PO:' || COALESCE(playoff_game_code,'') || ':' || COALESCE(series_game_number::text,'')
    ELSE
      'OTHER:' || COALESCE(id::text,'')
  END
WHERE game_conflict_key IS NULL;

-- Make it generated (stored) going forward.
-- NOTE: Postgres requires DROP/ADD to convert to GENERATED; do it safely.
ALTER TABLE public.tournament_games
DROP COLUMN IF EXISTS game_conflict_key;

ALTER TABLE public.tournament_games
ADD COLUMN game_conflict_key text GENERATED ALWAYS AS (
  CASE
    WHEN stage = 'ROUND_ROBIN'
      THEN 'RR:' || COALESCE(rr_round_number::text,'') || ':' || COALESCE(rr_slot_number::text,'')
    WHEN stage = 'PLAYOFF'
      THEN 'PO:' || COALESCE(playoff_game_code,'') || ':' || COALESCE(series_game_number::text,'')
    ELSE
      'OTHER:' || COALESCE(id::text,'')
  END
) STORED;

-- Drop prior uniqueness variants that may conflict or cause drift (safe)
DROP INDEX IF EXISTS tournament_games_rr_unique;
DROP INDEX IF EXISTS tournament_games_playoff_unique;
DROP INDEX IF EXISTS tournament_rr_unique_slot;
DROP INDEX IF EXISTS tournament_playoff_unique;
DROP INDEX IF EXISTS tournament_playoff_unique_slot;

-- Install canonical PostgREST-compatible uniqueness constraint
CREATE UNIQUE INDEX IF NOT EXISTS tournament_games_conflict_unique
ON public.tournament_games (club_id, tournament_id, stage, game_conflict_key);
