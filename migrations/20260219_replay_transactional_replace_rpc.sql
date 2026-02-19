CREATE OR REPLACE FUNCTION public.replay_transactional_replace(
  p_club_id TEXT,
  p_league_rows JSONB,
  p_match_snapshot_rows JSONB,
  p_match_batch_size INTEGER DEFAULT 500
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  -- Replace league ratings for this club
  DELETE FROM public.league_ratings
  WHERE club_id = p_club_id;

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
  FROM jsonb_to_recordset(p_league_rows) AS x(
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
  WHERE x.league_name IS NOT NULL AND x.player_id IS NOT NULL;

  -- Update match snapshot columns in-place
  UPDATE public.matches m
  SET
    elo_delta = s.elo_delta,
    t1_p1_r = s.t1_p1_r,
    t1_p2_r = s.t1_p2_r,
    t2_p1_r = s.t2_p1_r,
    t2_p2_r = s.t2_p2_r,
    delta_t1_p1_r = s.delta_t1_p1_r,
    delta_t1_p2_r = s.delta_t1_p2_r,
    delta_t2_p1_r = s.delta_t2_p1_r,
    delta_t2_p2_r = s.delta_t2_p2_r
  FROM jsonb_to_recordset(p_match_snapshot_rows) AS s(
    id BIGINT,
    elo_delta DOUBLE PRECISION,
    t1_p1_r DOUBLE PRECISION,
    t1_p2_r DOUBLE PRECISION,
    t2_p1_r DOUBLE PRECISION,
    t2_p2_r DOUBLE PRECISION,
    delta_t1_p1_r DOUBLE PRECISION,
    delta_t1_p2_r DOUBLE PRECISION,
    delta_t2_p1_r DOUBLE PRECISION,
    delta_t2_p2_r DOUBLE PRECISION
  )
  WHERE m.club_id = p_club_id AND m.id = s.id;
END;
$$;

-- Back-compat wrapper for code that tries replay_transactional_replace_v1
CREATE OR REPLACE FUNCTION public.replay_transactional_replace_v1(
  p_club_id TEXT,
  p_league_rows JSONB,
  p_match_snapshot_rows JSONB,
  p_match_batch_size INTEGER DEFAULT 500
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  PERFORM public.replay_transactional_replace(p_club_id, p_league_rows, p_match_snapshot_rows, p_match_batch_size);
END;
$$;

GRANT EXECUTE ON FUNCTION public.replay_transactional_replace(TEXT, JSONB, JSONB, INTEGER) TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.replay_transactional_replace_v1(TEXT, JSONB, JSONB, INTEGER) TO anon, authenticated, service_role;
