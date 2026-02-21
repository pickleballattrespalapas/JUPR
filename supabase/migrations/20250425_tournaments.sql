-- Tournament tables and match context columns

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name = 'matches'
  ) THEN
    ALTER TABLE public.matches
      ADD COLUMN IF NOT EXISTS context_type text NULL,
      ADD COLUMN IF NOT EXISTS context_id uuid NULL,
      ADD COLUMN IF NOT EXISTS tournament_id uuid NULL,
      ADD COLUMN IF NOT EXISTS tournament_game_id uuid NULL;

    CREATE INDEX IF NOT EXISTS idx_matches_tournament_id ON public.matches (tournament_id);
    CREATE INDEX IF NOT EXISTS idx_matches_context_type ON public.matches (context_type);
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS public.tournaments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  club_id text NOT NULL,
  name text NOT NULL,
  status text NOT NULL DEFAULT 'DRAFT',
  team_count integer NOT NULL,
  playoff_advance_count integer NULL,
  created_by_admin_id text NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tournaments_club_id ON public.tournaments (club_id);

CREATE TABLE IF NOT EXISTS public.tournament_teams (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tournament_id uuid NOT NULL REFERENCES public.tournaments(id) ON DELETE CASCADE,
  team_number integer NOT NULL,
  player1_id integer NULL,
  player2_id integer NULL,
  seed integer NULL,
  CONSTRAINT uq_tournament_team_number UNIQUE (tournament_id, team_number)
);

CREATE INDEX IF NOT EXISTS idx_tournament_teams_tournament_id ON public.tournament_teams (tournament_id);

CREATE TABLE IF NOT EXISTS public.tournament_games (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tournament_id uuid NOT NULL REFERENCES public.tournaments(id) ON DELETE CASCADE,
  stage text NOT NULL,
  rr_round_number integer NULL,
  rr_slot_number integer NULL,
  playoff_game_code text NULL,
  playoff_round text NULL,
  team_a_id uuid NULL REFERENCES public.tournament_teams(id),
  team_b_id uuid NULL REFERENCES public.tournament_teams(id),
  team_a_source jsonb NULL,
  team_b_source jsonb NULL,
  score_a integer NULL,
  score_b integer NULL,
  winner_team_id uuid NULL REFERENCES public.tournament_teams(id),
  loser_team_id uuid NULL REFERENCES public.tournament_teams(id),
  finalized_at timestamptz NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tournament_games_tournament_id ON public.tournament_games (tournament_id);
CREATE INDEX IF NOT EXISTS idx_tournament_games_stage ON public.tournament_games (stage);
