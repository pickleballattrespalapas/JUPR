-- Forward-compatible additive tournament schema alignment for legacy production.
-- Do not drop columns or constraints here.

alter table if exists public.tournaments
  add column if not exists team_count integer,
  add column if not exists playoff_advance_count integer,
  add column if not exists created_by_admin_id text,
  add column if not exists updated_at timestamptz;

alter table if exists public.tournament_games
  add column if not exists stage text,
  add column if not exists rr_round_number integer,
  add column if not exists rr_slot_number integer,
  add column if not exists playoff_game_code text,
  add column if not exists playoff_round text,
  add column if not exists team_a_id uuid,
  add column if not exists team_b_id uuid,
  add column if not exists team_a_source jsonb,
  add column if not exists team_b_source jsonb,
  add column if not exists score_a integer,
  add column if not exists score_b integer,
  add column if not exists winner_team_id uuid,
  add column if not exists loser_team_id uuid,
  add column if not exists finalized_at timestamptz,
  add column if not exists updated_at timestamptz;

alter table if exists public.tournament_teams
  add column if not exists seed integer;

alter table if exists public.tournament_podium
  add column if not exists source text;

create index if not exists idx_tournament_games_club_playoff_code
  on public.tournament_games (club_id, playoff_game_code);
