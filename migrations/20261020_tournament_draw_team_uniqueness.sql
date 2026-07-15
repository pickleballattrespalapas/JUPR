-- Allow multiple division draws in one tournament to each use team numbers 1..N.
-- The original tournament-wide unique constraint is too strict after draw-scoped ops were added.

alter table public.tournament_teams
  drop constraint if exists uq_tournament_team_number;

create unique index if not exists uq_tournament_teams_legacy_team
  on public.tournament_teams (tournament_id, team_number)
  where draw_id is null;

create unique index if not exists uq_tournament_teams_draw_team
  on public.tournament_teams (tournament_id, draw_id, team_number)
  where draw_id is not null;
