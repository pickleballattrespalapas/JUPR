-- Tournament venue inventory and per-day court availability.
-- Keeps stable court identities separate from optional public titles and
-- stores venue address/directions for public and administrative use.

alter table if exists public.tournament_registration_settings
  add column if not exists venue_address text,
  add column if not exists venue_directions text,
  add column if not exists venue_courts_json jsonb not null default '[]'::jsonb;

alter table if exists public.tournament_registration_settings
  drop constraint if exists tournament_registration_settings_venue_courts_chk,
  add constraint tournament_registration_settings_venue_courts_chk
    check (jsonb_typeof(venue_courts_json) = 'array');

with first_day as (
  select distinct on (tournament_id)
    tournament_id,
    greatest(coalesce(court_count, 10), 1) as court_count,
    coalesce(court_labels, '[]'::jsonb) as court_labels
  from public.tournament_registration_days
  order by tournament_id, sort_order, event_date, id
), generated as (
  select
    day.tournament_id,
    jsonb_agg(
      jsonb_build_object(
        'id', 'venue-court-' || series.position,
        'title', coalesce(day.court_labels ->> (series.position - 1), '')
      )
      order by series.position
    ) as courts
  from first_day day
  cross join lateral generate_series(1, least(day.court_count, 100)) as series(position)
  group by day.tournament_id
)
update public.tournament_registration_settings settings
set venue_courts_json = generated.courts
from generated
where settings.tournament_id = generated.tournament_id
  and coalesce(jsonb_array_length(settings.venue_courts_json), 0) = 0;

alter table if exists public.tournament_registration_days
  add column if not exists available_court_ids jsonb not null default '[]'::jsonb;

alter table if exists public.tournament_registration_days
  drop constraint if exists tournament_registration_days_available_courts_chk,
  add constraint tournament_registration_days_available_courts_chk
    check (jsonb_typeof(available_court_ids) = 'array');

update public.tournament_registration_days day
set available_court_ids = (
  select jsonb_agg('venue-court-' || series.position order by series.position)
  from generate_series(1, least(greatest(coalesce(day.court_count, 10), 1), 100)) as series(position)
)
where coalesce(jsonb_array_length(day.available_court_ids), 0) = 0;

comment on column public.tournament_registration_settings.venue_address is
  'Public venue address used for display and map links.';
comment on column public.tournament_registration_settings.venue_directions is
  'Optional parking, gate, building, or arrival instructions.';
comment on column public.tournament_registration_settings.venue_courts_json is
  'Ordered venue court inventory. Each row has a stable id and optional title.';
comment on column public.tournament_registration_days.available_court_ids is
  'Ordered stable venue court IDs available on this tournament day.';
