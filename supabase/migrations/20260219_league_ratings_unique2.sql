-- Required for PostgREST upserts on league_ratings
CREATE UNIQUE INDEX IF NOT EXISTS league_ratings_club_player_league_uidx
  ON public.league_ratings (club_id, player_id, league_name);
