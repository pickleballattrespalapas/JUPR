ALTER TABLE public.players
  ADD COLUMN IF NOT EXISTS last_game_at timestamptz NULL,
  ADD COLUMN IF NOT EXISTS inactive_at timestamptz NULL;

-- Backfill last_game_at from recorded matches (global across leagues).
WITH player_matches AS (
  SELECT club_id, t1_p1 AS player_id, "date" AS game_at, score_t1, score_t2 FROM public.matches
  UNION ALL
  SELECT club_id, t1_p2 AS player_id, "date" AS game_at, score_t1, score_t2 FROM public.matches
  UNION ALL
  SELECT club_id, t2_p1 AS player_id, "date" AS game_at, score_t1, score_t2 FROM public.matches
  UNION ALL
  SELECT club_id, t2_p2 AS player_id, "date" AS game_at, score_t1, score_t2 FROM public.matches
),
player_max AS (
  SELECT
    club_id,
    player_id,
    MAX(game_at) AS max_game_at
  FROM player_matches
  WHERE COALESCE(score_t1, 0) + COALESCE(score_t2, 0) > 0
  GROUP BY club_id, player_id
)
UPDATE public.players p
SET last_game_at = pm.max_game_at
FROM player_max pm
WHERE p.club_id = pm.club_id
  AND p.id = pm.player_id;

-- Never-played players use created_at for inactivity baseline.
UPDATE public.players
SET inactive_at = NOW()
WHERE inactive_at IS NULL
  AND COALESCE(last_game_at, created_at) <= (NOW() - INTERVAL '14 days');

CREATE INDEX IF NOT EXISTS idx_players_inactive_at ON public.players (inactive_at);
CREATE INDEX IF NOT EXISTS idx_players_last_game_at ON public.players (last_game_at);

CREATE OR REPLACE FUNCTION public.mark_inactive_players() RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE public.players
  SET inactive_at = NOW()
  WHERE inactive_at IS NULL
    AND COALESCE(last_game_at, created_at) <= (NOW() - INTERVAL '14 days');
END;
$$;

CREATE EXTENSION IF NOT EXISTS pg_cron WITH SCHEMA extensions;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM cron.job WHERE jobname = 'mark-inactive-players-daily'
  ) THEN
    PERFORM cron.schedule(
      'mark-inactive-players-daily',
      '0 3 * * *',
      $$SELECT public.mark_inactive_players();$$
    );
  END IF;
END
$$;
