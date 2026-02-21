-- Tournament RLS policy pack keyed by JWT user_metadata.club_id

ALTER TABLE public.tournaments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tournament_teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tournament_games ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tournament_podium ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tournaments_select_by_club ON public.tournaments;
CREATE POLICY tournaments_select_by_club
ON public.tournaments
FOR SELECT
USING (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

DROP POLICY IF EXISTS tournaments_insert_by_club ON public.tournaments;
CREATE POLICY tournaments_insert_by_club
ON public.tournaments
FOR INSERT
WITH CHECK (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

DROP POLICY IF EXISTS tournaments_update_by_club ON public.tournaments;
CREATE POLICY tournaments_update_by_club
ON public.tournaments
FOR UPDATE
USING (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
)
WITH CHECK (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

DROP POLICY IF EXISTS tournaments_delete_by_club ON public.tournaments;
CREATE POLICY tournaments_delete_by_club
ON public.tournaments
FOR DELETE
USING (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

DROP POLICY IF EXISTS tournament_teams_all_by_club ON public.tournament_teams;
CREATE POLICY tournament_teams_all_by_club
ON public.tournament_teams
FOR ALL
USING (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
)
WITH CHECK (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

DROP POLICY IF EXISTS tournament_games_all_by_club ON public.tournament_games;
CREATE POLICY tournament_games_all_by_club
ON public.tournament_games
FOR ALL
USING (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
)
WITH CHECK (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

DROP POLICY IF EXISTS tournament_podium_all_by_club ON public.tournament_podium;
CREATE POLICY tournament_podium_all_by_club
ON public.tournament_podium
FOR ALL
USING (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
)
WITH CHECK (
  club_id = (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);
