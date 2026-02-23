-- Multi-division tournament support
-- Additive, club-scoped tables for tournament divisions, entries, and matches.

CREATE TABLE IF NOT EXISTS public.tournament_divisions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  club_id text NOT NULL,
  tournament_id uuid NOT NULL REFERENCES public.tournaments(id) ON DELETE CASCADE,
  title text NOT NULL,
  format text NOT NULL,
  max_teams integer,
  status text NOT NULL DEFAULT 'draft',
  created_at timestamptz DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS tournament_divisions_club_tournament_title_key
  ON public.tournament_divisions (club_id, tournament_id, title);

CREATE TABLE IF NOT EXISTS public.division_entries (
  id uuid PRIMARY KEY,
  club_id text NOT NULL,
  division_id uuid NOT NULL REFERENCES public.tournament_divisions(id) ON DELETE CASCADE,
  team_id uuid NOT NULL REFERENCES public.teams(id),
  seed integer,
  created_at timestamptz DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS division_entries_club_division_team_key
  ON public.division_entries (club_id, division_id, team_id);

CREATE TABLE IF NOT EXISTS public.division_matches (
  id uuid PRIMARY KEY,
  club_id text NOT NULL,
  division_id uuid NOT NULL REFERENCES public.tournament_divisions(id) ON DELETE CASCADE,
  round_number integer NOT NULL,
  bracket_position integer NOT NULL,
  team_a_id uuid NOT NULL REFERENCES public.teams(id),
  team_b_id uuid NOT NULL REFERENCES public.teams(id),
  winner_team_id uuid REFERENCES public.teams(id),
  score_json jsonb,
  status text NOT NULL DEFAULT 'scheduled',
  created_at timestamptz DEFAULT now()
);
