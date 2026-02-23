-- 20260221_tournament_tenant_hardening.sql
-- Forward-only tenant hardening for tournament child tables
-- Non-destructive: adds columns, backfills, then enforces NOT NULL + uniqueness.

begin;

-- Avoid surprising long locks; fail fast if something is blocking.
set local lock_timeout = '3s';
set local statement_timeout = '60s';

-- 1) Add club_id columns (nullable first)
alter table public.tournament_teams
  add column if not exists club_id text;

alter table public.tournament_games
  add column if not exists club_id text;

alter table public.tournament_podium
  add column if not exists club_id text;

-- 2) Backfill club_id from tournaments
update public.tournament_teams tt
set club_id = t.club_id
from public.tournaments t
where t.id = tt.tournament_id
  and (tt.club_id is null or tt.club_id = '');

update public.tournament_games tg
set club_id = t.club_id
from public.tournaments t
where t.id = tg.tournament_id
  and (tg.club_id is null or tg.club_id = '');

update public.tournament_podium tp
set club_id = t.club_id
from public.tournaments t
where t.id = tp.tournament_id
  and (tp.club_id is null or tp.club_id = '');

-- 3) Enforce NOT NULL (after backfill)
alter table public.tournament_teams
  alter column club_id set not null;

alter table public.tournament_games
  alter column club_id set not null;

alter table public.tournament_podium
  alter column club_id set not null;

-- 4) Tenant-safe uniqueness
-- (Keep existing constraints; add tenant-aware versions.)
alter table public.tournament_teams
  drop constraint if exists uq_tournament_team_number,
  add constraint uq_tournament_team_number unique (club_id, tournament_id, team_number);

alter table public.tournament_podium
  drop constraint if exists tournament_podium_unique_placement,
  add constraint tournament_podium_unique_placement unique (club_id, tournament_id, placement);

alter table public.tournament_podium
  drop constraint if exists tournament_podium_unique_team,
  add constraint tournament_podium_unique_team unique (club_id, tournament_id, team_id);

-- 5) Helpful indexes (tenant + parent lookup)
create index if not exists idx_tournament_teams_club_tournament on public.tournament_teams (club_id, tournament_id);
create index if not exists idx_tournament_games_club_tournament on public.tournament_games (club_id, tournament_id);
create index if not exists idx_tournament_podium_club_tournament on public.tournament_podium (club_id, tournament_id);

-- 6) Guardrails: auto-fill / validate club_id on writes
-- (Prevents future bugs where app forgets to send club_id.)
create or replace function public.tournament_child_set_club_id()
returns trigger
language plpgsql
as $$
declare
  parent_club text;
begin
  select club_id into parent_club
  from public.tournaments
  where id = new.tournament_id;

  if parent_club is null then
    raise exception 'Invalid tournament_id % (no parent tournament)', new.tournament_id;
  end if;

  if new.club_id is null or new.club_id = '' then
    new.club_id := parent_club;
  end if;

  if new.club_id <> parent_club then
    raise exception 'club_id mismatch: got %, expected % for tournament_id %', new.club_id, parent_club, new.tournament_id;
  end if;

  return new;
end;
$$;

drop trigger if exists trg_tournament_teams_set_club_id on public.tournament_teams;
create trigger trg_tournament_teams_set_club_id
before insert or update of tournament_id, club_id on public.tournament_teams
for each row execute function public.tournament_child_set_club_id();

drop trigger if exists trg_tournament_games_set_club_id on public.tournament_games;
create trigger trg_tournament_games_set_club_id
before insert or update of tournament_id, club_id on public.tournament_games
for each row execute function public.tournament_child_set_club_id();

drop trigger if exists trg_tournament_podium_set_club_id on public.tournament_podium;
create trigger trg_tournament_podium_set_club_id
before insert or update of tournament_id, club_id on public.tournament_podium
for each row execute function public.tournament_child_set_club_id();

commit;
