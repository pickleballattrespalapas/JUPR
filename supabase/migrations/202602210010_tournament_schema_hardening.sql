-- ============================================
-- TOURNAMENT SCHEMA HARDENING
-- ============================================

-- 1) tournament_teams
alter table public.tournament_teams
add column if not exists club_id text;

update public.tournament_teams tt
set club_id = t.club_id
from public.tournaments t
where tt.tournament_id = t.id
  and tt.club_id is null;

alter table public.tournament_teams
alter column club_id set not null;

create index if not exists idx_tournament_teams_club_id
on public.tournament_teams (club_id);

alter table public.tournament_teams
drop constraint if exists uq_tournament_team_number;

alter table public.tournament_teams
add constraint uq_tournament_team_number
unique (club_id, tournament_id, team_number);


-- 2) tournament_games
alter table public.tournament_games
add column if not exists club_id text;

update public.tournament_games tg
set club_id = t.club_id
from public.tournaments t
where tg.tournament_id = t.id
  and tg.club_id is null;

alter table public.tournament_games
alter column club_id set not null;

create index if not exists idx_tournament_games_club_id
on public.tournament_games (club_id);


-- 3) tournament_podium
alter table public.tournament_podium
add column if not exists club_id text;

update public.tournament_podium tp
set club_id = t.club_id
from public.tournaments t
where tp.tournament_id = t.id
  and tp.club_id is null;

alter table public.tournament_podium
alter column club_id set not null;

create index if not exists idx_tournament_podium_club_id
on public.tournament_podium (club_id);

alter table public.tournament_podium
drop constraint if exists tournament_podium_unique_placement;

alter table public.tournament_podium
add constraint tournament_podium_unique_placement
unique (club_id, tournament_id, placement);

alter table public.tournament_podium
drop constraint if exists tournament_podium_unique_team;

alter table public.tournament_podium
add constraint tournament_podium_unique_team
unique (club_id, tournament_id, team_id);
