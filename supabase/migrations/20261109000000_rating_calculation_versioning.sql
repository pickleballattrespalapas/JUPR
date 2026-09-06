-- Preserve the exact rating policy attached to every match calculation.
-- This is audit metadata only; it does not add player-facing confidence states.

create table if not exists public.rating_calculation_versions (
  algorithm_version text not null,
  parameter_version text not null,
  parameters jsonb not null,
  description text not null,
  created_at timestamptz not null default now(),
  primary key (algorithm_version, parameter_version),
  constraint rating_calculation_versions_parameters_object_check
    check (jsonb_typeof(parameters) = 'object'),
  constraint rating_calculation_versions_algorithm_nonempty_check
    check (btrim(algorithm_version) <> ''),
  constraint rating_calculation_versions_parameter_nonempty_check
    check (btrim(parameter_version) <> '')
);

alter table public.rating_calculation_versions enable row level security;
revoke all on table public.rating_calculation_versions from anon, authenticated;
grant select on table public.rating_calculation_versions to service_role;

insert into public.rating_calculation_versions (
  algorithm_version,
  parameter_version,
  parameters,
  description
)
values (
  'jupr-hybrid-score-share-v1',
  'flat-k32-floor1-loser-cap16-v1',
  jsonb_build_object(
    'overall_k_factor', 32.0,
    'min_win_delta_elo', 1.0,
    'cap_loser_gain_elo', 16.0,
    'elo_per_jupr', 400.0,
    'winner_must_gain', true,
    'loser_may_gain_for_outperformance', true
  ),
  'Original flat-K JUPR score-share policy: winners gain and losers may gain for score outperformance.'
)
on conflict (algorithm_version, parameter_version) do nothing;

alter table public.matches
  add column if not exists rating_algorithm_version text,
  add column if not exists rating_parameter_version text,
  add column if not exists rating_parameters jsonb;

update public.matches
set
  rating_algorithm_version = coalesce(
    nullif(btrim(rating_algorithm_version), ''),
    'jupr-hybrid-score-share-v1'
  ),
  rating_parameter_version = coalesce(
    nullif(btrim(rating_parameter_version), ''),
    'flat-k32-floor1-loser-cap16-v1'
  ),
  rating_parameters = coalesce(
    rating_parameters,
    jsonb_build_object(
      'overall_k_factor', 32.0,
      'min_win_delta_elo', 1.0,
      'cap_loser_gain_elo', 16.0,
      'elo_per_jupr', 400.0,
      'winner_must_gain', true,
      'loser_may_gain_for_outperformance', true
    )
  );

update public.matches as match_row
set rating_parameters = match_row.rating_parameters || jsonb_build_object(
  'league_k_factor', league.k_factor::double precision
)
from public.leagues_metadata as league
where league.club_id = match_row.club_id
  and lower(btrim(league.league_name)) = lower(btrim(match_row.league))
  and coalesce(league.match_format, 'doubles') = coalesce(match_row.match_format, 'doubles')
  and lower(btrim(coalesce(match_row.match_type, ''))) <> 'popup'
  and lower(btrim(coalesce(match_row.rating_scope, ''))) not in ('overall_only', 'unrated');

alter table public.matches
  alter column rating_algorithm_version set default 'jupr-hybrid-score-share-v1',
  alter column rating_algorithm_version set not null,
  alter column rating_parameter_version set default 'flat-k32-floor1-loser-cap16-v1',
  alter column rating_parameter_version set not null,
  alter column rating_parameters set default jsonb_build_object(
    'overall_k_factor', 32.0,
    'min_win_delta_elo', 1.0,
    'cap_loser_gain_elo', 16.0,
    'elo_per_jupr', 400.0,
    'winner_must_gain', true,
    'loser_may_gain_for_outperformance', true
  ),
  alter column rating_parameters set not null;

do $block$
begin
  if not exists (
    select 1
    from pg_constraint
    where conrelid = 'public.matches'::regclass
      and conname = 'matches_rating_parameters_object_check'
  ) then
    alter table public.matches
      add constraint matches_rating_parameters_object_check
      check (jsonb_typeof(rating_parameters) = 'object');
  end if;

  if not exists (
    select 1
    from pg_constraint
    where conrelid = 'public.matches'::regclass
      and conname = 'matches_rating_calculation_version_fk'
  ) then
    alter table public.matches
      add constraint matches_rating_calculation_version_fk
      foreign key (rating_algorithm_version, rating_parameter_version)
      references public.rating_calculation_versions (
        algorithm_version,
        parameter_version
      );
  end if;
end
$block$;

comment on column public.matches.rating_algorithm_version is
  'Immutable identifier for the mathematical rating update used by this match.';
comment on column public.matches.rating_parameter_version is
  'Immutable identifier for the default policy parameter set used by this match.';
comment on column public.matches.rating_parameters is
  'Exact JSON parameter snapshot used for this match, including any league K override.';

create or replace function public.stamp_match_rating_calculation_version()
returns trigger
language plpgsql
set search_path = ''
as $function$
declare
  v_league_k_factor integer;
begin
  if tg_op = 'INSERT' then
    new.rating_algorithm_version := coalesce(
      nullif(pg_catalog.btrim(new.rating_algorithm_version), ''),
      'jupr-hybrid-score-share-v1'
    );
    new.rating_parameter_version := coalesce(
      nullif(pg_catalog.btrim(new.rating_parameter_version), ''),
      'flat-k32-floor1-loser-cap16-v1'
    );
  elsif new.rating_algorithm_version is not distinct from old.rating_algorithm_version
        and new.rating_parameter_version is not distinct from old.rating_parameter_version
        and new.rating_parameters is not distinct from old.rating_parameters then
    new.rating_algorithm_version := 'jupr-hybrid-score-share-v1';
    new.rating_parameter_version := 'flat-k32-floor1-loser-cap16-v1';
  end if;

  if new.rating_algorithm_version = 'jupr-hybrid-score-share-v1'
     and new.rating_parameter_version = 'flat-k32-floor1-loser-cap16-v1' then
    new.rating_parameters := jsonb_build_object(
      'overall_k_factor', 32.0,
      'min_win_delta_elo', 1.0,
      'cap_loser_gain_elo', 16.0,
      'elo_per_jupr', 400.0,
      'winner_must_gain', true,
      'loser_may_gain_for_outperformance', true
    );

    if lower(pg_catalog.btrim(coalesce(new.match_type, ''))) <> 'popup'
       and lower(pg_catalog.btrim(coalesce(new.rating_scope, ''))) not in ('overall_only', 'unrated') then
      select league.k_factor
        into v_league_k_factor
        from public.leagues_metadata as league
       where league.club_id = new.club_id
         and lower(pg_catalog.btrim(league.league_name)) = lower(pg_catalog.btrim(new.league))
         and coalesce(league.match_format, 'doubles') = coalesce(new.match_format, 'doubles')
       limit 1;
      if v_league_k_factor is not null then
        new.rating_parameters := new.rating_parameters || jsonb_build_object(
          'league_k_factor', v_league_k_factor::double precision
        );
      end if;
    end if;
  end if;
  return new;
end
$function$;

revoke all on function public.stamp_match_rating_calculation_version() from public;

drop trigger if exists stamp_inserted_match_rating_calculation_version on public.matches;
create trigger stamp_inserted_match_rating_calculation_version
before insert
on public.matches
for each row
execute function public.stamp_match_rating_calculation_version();

drop trigger if exists stamp_replayed_match_rating_calculation_version on public.matches;
create trigger stamp_replayed_match_rating_calculation_version
before update of
  elo_delta,
  t1_p1_r,
  t1_p2_r,
  t2_p1_r,
  t2_p2_r,
  t1_p1_r_end,
  t1_p2_r_end,
  t2_p1_r_end,
  t2_p2_r_end
on public.matches
for each row
execute function public.stamp_match_rating_calculation_version();
