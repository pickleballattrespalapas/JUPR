-- Required for sb_upsert(... conflict="club_id,player_id,league_name") in jupr_app/domain/match_processing.py.
CREATE UNIQUE INDEX IF NOT EXISTS league_ratings_club_player_league_uidx
ON public.league_ratings (club_id, player_id, league_name)
WHERE club_id IS NOT NULL AND player_id IS NOT NULL;
