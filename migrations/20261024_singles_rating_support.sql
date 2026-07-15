alter table if exists public.players
  add column if not exists singles_rating double precision,
  add column if not exists singles_wins integer not null default 0,
  add column if not exists singles_losses integer not null default 0,
  add column if not exists singles_matches_played integer not null default 0,
  add column if not exists singles_last_game_at timestamptz;

update public.players
set singles_rating = coalesce(singles_rating, rating, 1200)
where singles_rating is null;

alter table if exists public.players
  alter column singles_rating set default 1200;

alter table if exists public.matches
  add column if not exists match_format text not null default 'doubles';

alter table if exists public.matches
  alter column t1_p2 drop not null,
  alter column t2_p2 drop not null,
  alter column t1_p2_r drop not null,
  alter column t2_p2_r drop not null,
  alter column t1_p2_r_end drop not null,
  alter column t2_p2_r_end drop not null;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'matches_match_format_check'
  ) then
    alter table public.matches
      add constraint matches_match_format_check check (match_format in ('doubles', 'singles'));
  end if;
end $$;

create index if not exists idx_matches_club_format_date on public.matches (club_id, match_format, date desc);
