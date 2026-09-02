-- Repair legacy production-shaped tables so the Next/FastAPI staging surface
-- can exercise its real read/write contracts. All changes are additive and
-- idempotent so this migration can be promoted after staging verification.

alter table if exists public.clubs
  add column if not exists slug text,
  add column if not exists tagline text,
  add column if not exists support_email text,
  add column if not exists public_base_url text,
  add column if not exists is_active boolean not null default true,
  add column if not exists updated_at timestamptz,
  add column if not exists plan_status text not null default 'pilot',
  add column if not exists features_json jsonb not null default '{}'::jsonb,
  add column if not exists created_by_email text,
  add column if not exists onboarding_status text not null default 'draft';

update public.clubs
set
  slug = coalesce(
    nullif(trim(slug), ''),
    case
      when id = 'tres_palapas' then 'tres-palapas'
      else trim(both '-' from regexp_replace(lower(id), '[^a-z0-9]+', '-', 'g'))
    end
  ),
  is_active = case
    when lower(coalesce(status, 'active')) = 'active' then true
    else coalesce(is_active, false)
  end,
  updated_at = coalesce(updated_at, created_at, now());

do $$
begin
  if exists (
    select 1
    from public.clubs
    where slug is null or trim(slug) = ''
  ) then
    raise exception 'clubs.slug backfill produced a blank slug';
  end if;

  if exists (
    select 1
    from public.clubs
    group by slug
    having count(*) > 1
  ) then
    raise exception 'clubs.slug backfill produced duplicate slugs';
  end if;
end
$$;

alter table if exists public.clubs
  alter column slug set not null,
  alter column updated_at set default now(),
  alter column updated_at set not null;

create unique index if not exists clubs_slug_key on public.clubs (slug);

alter table if exists public.players
  add column if not exists singles_rating double precision,
  add column if not exists singles_wins integer not null default 0,
  add column if not exists singles_losses integer not null default 0,
  add column if not exists singles_matches_played integer not null default 0,
  add column if not exists singles_last_game_at timestamptz;

update public.players
set singles_rating = coalesce(singles_rating, rating::double precision, 1200.0)
where singles_rating is null;

alter table if exists public.players
  alter column singles_rating set default 1200.0;

alter table if exists public.matches
  add column if not exists is_active boolean not null default true,
  add column if not exists created_at timestamptz,
  add column if not exists match_format text not null default 'doubles',
  add column if not exists rating_scope text;

update public.matches
set created_at = coalesce(created_at, date, now())
where created_at is null;

alter table if exists public.matches
  alter column created_at set default now(),
  alter column created_at set not null;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'matches_match_format_check'
      and conrelid = 'public.matches'::regclass
  ) then
    alter table public.matches
      add constraint matches_match_format_check
      check (match_format in ('doubles', 'singles'));
  end if;

  if not exists (
    select 1
    from pg_constraint
    where conname = 'matches_rating_scope_check'
      and conrelid = 'public.matches'::regclass
  ) then
    alter table public.matches
      add constraint matches_rating_scope_check
      check (
        rating_scope is null
        or rating_scope = ''
        or rating_scope = 'overall_only'
        or rating_scope = 'unrated'
      );
  end if;
end
$$;

create index if not exists idx_matches_club_format_date
  on public.matches (club_id, match_format, date desc);

alter table if exists public.leagues_metadata
  add column if not exists updated_at timestamptz,
  add column if not exists event_tags jsonb not null default '{}'::jsonb;

update public.leagues_metadata
set updated_at = coalesce(updated_at, created_at, now())
where updated_at is null;

alter table if exists public.leagues_metadata
  alter column updated_at set default now(),
  alter column updated_at set not null;

alter table if exists public.tournaments
  add column if not exists start_date date,
  add column if not exists end_date date,
  add column if not exists event_tags jsonb not null default '{}'::jsonb;

alter table if exists public.tournament_registration_selections
  add column if not exists updated_at timestamptz;

update public.tournament_registration_selections
set updated_at = coalesce(updated_at, created_at, now())
where updated_at is null;

alter table if exists public.tournament_registration_selections
  alter column updated_at set default now(),
  alter column updated_at set not null;
