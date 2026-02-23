begin;

drop trigger if exists trg_teams_club_guard on public.tournament_teams;
drop trigger if exists trg_games_club_guard on public.tournament_games;
drop trigger if exists trg_podium_club_guard on public.tournament_podium;

drop function if exists public.tournament_child_enforce_club();



create or replace function public.tournament_child_enforce_club()
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
    raise exception 'Invalid tournament_id %', new.tournament_id;
  end if;

  if new.club_id <> parent_club then
    raise exception 'club_id mismatch for tournament child row';
  end if;

  return new;
end;
$$;

create trigger trg_teams_club_guard
before insert or update on public.tournament_teams
for each row execute function public.tournament_child_enforce_club();

create trigger trg_games_club_guard
before insert or update on public.tournament_games
for each row execute function public.tournament_child_enforce_club();

create trigger trg_podium_club_guard
before insert or update on public.tournament_podium
for each row execute function public.tournament_child_enforce_club();

-- recreate function + triggers here

commit;
