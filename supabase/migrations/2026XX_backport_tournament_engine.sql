-- Backport tournament engine schema updates.
-- Scoped to tournament tables only.

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

CREATE INDEX IF NOT EXISTS idx_tournaments_club_id
  ON public.tournaments (club_id);

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

CREATE INDEX IF NOT EXISTS idx_tournament_games_tournament_id
  ON public.tournament_games (tournament_id);

CREATE INDEX IF NOT EXISTS idx_tournament_games_stage
  ON public.tournament_games (stage);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'tournament_games_rr_unique'
      AND conrelid = 'public.tournament_games'::regclass
  ) THEN
    ALTER TABLE public.tournament_games
      ADD CONSTRAINT tournament_games_rr_unique
      UNIQUE (tournament_id, rr_round_number, rr_slot_number);
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'tournament_games_playoff_unique'
      AND conrelid = 'public.tournament_games'::regclass
  ) THEN
    ALTER TABLE public.tournament_games
      ADD CONSTRAINT tournament_games_playoff_unique
      UNIQUE (tournament_id, playoff_game_code);
  END IF;
END
$$;

CREATE TABLE IF NOT EXISTS public.tournament_podium (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tournament_id uuid NOT NULL REFERENCES public.tournaments(id) ON DELETE CASCADE,
  placement integer NOT NULL,
  team_id uuid NOT NULL REFERENCES public.tournament_teams(id) ON DELETE RESTRICT,
  source text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT tournament_podium_unique_placement UNIQUE (tournament_id, placement),
  CONSTRAINT tournament_podium_unique_team UNIQUE (tournament_id, team_id),
  CONSTRAINT tournament_podium_valid_placement CHECK (placement BETWEEN 1 AND 3)
);

CREATE INDEX IF NOT EXISTS idx_tournament_podium_tournament_id
  ON public.tournament_podium (tournament_id);
