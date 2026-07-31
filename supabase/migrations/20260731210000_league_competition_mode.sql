-- Structural Individual / Team league mode for staging acceptance.
update public.leagues_metadata lm
set league_type = case
  when exists (
    select 1 from public.team_league_settings tls
    where tls.club_id = lm.club_id and tls.league_name = lm.league_name
  ) then 'Team'
  else 'Individual'
end
where coalesce(lm.league_type, '') not in ('Individual', 'Team');

alter table public.leagues_metadata
  drop constraint if exists leagues_metadata_league_type_check;
alter table public.leagues_metadata
  add constraint leagues_metadata_league_type_check
  check (league_type in ('Individual', 'Team'));

create or replace function public.prevent_league_competition_mode_change()
returns trigger
language plpgsql
as $$
begin
  if old.league_type is distinct from new.league_type then
    raise exception 'league competition mode is immutable after creation';
  end if;
  return new;
end;
$$;

drop trigger if exists trg_prevent_league_competition_mode_change on public.leagues_metadata;
create trigger trg_prevent_league_competition_mode_change
before update of league_type on public.leagues_metadata
for each row execute function public.prevent_league_competition_mode_change();

create or replace function public.require_team_league_mode()
returns trigger
language plpgsql
as $$
begin
  if not exists (
    select 1 from public.leagues_metadata lm
    where lm.club_id = new.club_id
      and lm.league_name = new.league_name
      and lm.league_type = 'Team'
  ) then
    raise exception 'team-league setup requires a Team league created in League Manager';
  end if;
  return new;
end;
$$;

drop trigger if exists trg_require_team_league_mode on public.team_league_settings;
create trigger trg_require_team_league_mode
before insert or update of club_id, league_name on public.team_league_settings
for each row execute function public.require_team_league_mode();
