
CREATE OR REPLACE FUNCTION public.replace_league_ratings(
  p_club_id TEXT,
  p_rows JSONB,
  p_reset BOOLEAN DEFAULT FALSE
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  IF p_reset THEN
    -- Only reset the leagues present in the payload (supports partial resets)
    DELETE FROM public.league_ratings lr
    WHERE lr.club_id = p_club_id
      AND lr.league_name IN (
        SELECT DISTINCT x.league_name
        FROM jsonb_to_recordset(p_rows) AS x(league_name TEXT)
        WHERE x.league_name IS NOT NULL
      );
  END IF;

  INSERT INTO public.league_ratings (
    club_id,
    league_name,
    player_id,
    rating,
    starting_rating,
    wins,
    losses,
    matches_played,
    is_active
  )
  SELECT
    p_club_id,
    x.league_name,
    x.player_id,
    x.rating,
    COALESCE(x.starting_rating, x.rating),
    COALESCE(x.wins, 0),
    COALESCE(x.losses, 0),
    COALESCE(x.matches_played, 0),
    COALESCE(x.is_active, TRUE)
  FROM jsonb_to_recordset(p_rows) AS x(
    club_id TEXT,
    league_name TEXT,
    player_id BIGINT,
    rating DOUBLE PRECISION,
    starting_rating DOUBLE PRECISION,
    wins INTEGER,
    losses INTEGER,
    matches_played INTEGER,
    is_active BOOLEAN
  )
  WHERE x.league_name IS NOT NULL AND x.player_id IS NOT NULL
  ON CONFLICT (club_id, player_id, league_name) DO UPDATE SET
    rating = EXCLUDED.rating,
    starting_rating = EXCLUDED.starting_rating,
    wins = EXCLUDED.wins,
    losses = EXCLUDED.losses,
    matches_played = EXCLUDED.matches_played,
    is_active = EXCLUDED.is_active;
END;
$$;

-- Optional: grant for non-service usage
GRANT EXECUTE ON FUNCTION public.replace_league_ratings(TEXT, JSONB, BOOLEAN) TO anon, authenticated, service_role;
