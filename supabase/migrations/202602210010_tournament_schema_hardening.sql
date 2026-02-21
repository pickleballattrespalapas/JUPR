-- Tournament tenant hardening (safe for existing production data)

ALTER TABLE public.tournament_teams
  ADD COLUMN IF NOT EXISTS club_id text;

ALTER TABLE public.tournament_games
  ADD COLUMN IF NOT EXISTS club_id text;

ALTER TABLE public.tournament_podium
  ADD COLUMN IF NOT EXISTS club_id text;

UPDATE public.tournament_teams tt
SET club_id = t.club_id
FROM public.tournaments t
WHERE tt.tournament_id = t.id
  AND tt.club_id IS NULL;

UPDATE public.tournament_games tg
SET club_id = t.club_id
FROM public.tournaments t
WHERE tg.tournament_id = t.id
  AND tg.club_id IS NULL;

UPDATE public.tournament_podium tp
SET club_id = t.club_id
FROM public.tournaments t
WHERE tp.tournament_id = t.id
  AND tp.club_id IS NULL;

DO $$
DECLARE
  teams_missing bigint;
  games_missing bigint;
  podium_missing bigint;
BEGIN
  SELECT count(*) INTO teams_missing FROM public.tournament_teams WHERE club_id IS NULL;
  SELECT count(*) INTO games_missing FROM public.tournament_games WHERE club_id IS NULL;
  SELECT count(*) INTO podium_missing FROM public.tournament_podium WHERE club_id IS NULL;

  IF teams_missing > 0 OR games_missing > 0 OR podium_missing > 0 THEN
    RAISE EXCEPTION
      'Tournament tenant hardening failed: null club_id remains (teams=%, games=%, podium=%)',
      teams_missing, games_missing, podium_missing;
  END IF;
END $$;

ALTER TABLE public.tournament_teams
  ALTER COLUMN club_id SET NOT NULL;

ALTER TABLE public.tournament_games
  ALTER COLUMN club_id SET NOT NULL;

ALTER TABLE public.tournament_podium
  ALTER COLUMN club_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tournament_teams_club_id ON public.tournament_teams (club_id);
CREATE INDEX IF NOT EXISTS idx_tournament_games_club_id ON public.tournament_games (club_id);
CREATE INDEX IF NOT EXISTS idx_tournament_podium_club_id ON public.tournament_podium (club_id);

ALTER TABLE public.tournament_teams
  DROP CONSTRAINT IF EXISTS uq_tournament_team_number;
ALTER TABLE public.tournament_teams
  DROP CONSTRAINT IF EXISTS tournament_teams_club_tournament_team_number_uq;
ALTER TABLE public.tournament_teams
  ADD CONSTRAINT tournament_teams_club_tournament_team_number_uq
  UNIQUE (club_id, tournament_id, team_number);

ALTER TABLE public.tournament_podium
  DROP CONSTRAINT IF EXISTS tournament_podium_unique_placement;
ALTER TABLE public.tournament_podium
  DROP CONSTRAINT IF EXISTS tournament_podium_club_tournament_placement_uq;
ALTER TABLE public.tournament_podium
  ADD CONSTRAINT tournament_podium_club_tournament_placement_uq
  UNIQUE (club_id, tournament_id, placement);

ALTER TABLE public.tournament_podium
  DROP CONSTRAINT IF EXISTS tournament_podium_unique_team;
ALTER TABLE public.tournament_podium
  DROP CONSTRAINT IF EXISTS tournament_podium_club_tournament_team_uq;
ALTER TABLE public.tournament_podium
  ADD CONSTRAINT tournament_podium_club_tournament_team_uq
  UNIQUE (club_id, tournament_id, team_id);
