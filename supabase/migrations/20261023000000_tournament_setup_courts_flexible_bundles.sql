-- Tournament setup court configuration and flexible event bundle rules.
-- Adds publishable per-day court metadata and permits bundles such as
-- "any 1 of these events" or "any 2 of these events".

alter table if exists public.tournament_registration_days
  add column if not exists court_count integer not null default 10,
  add column if not exists court_labels jsonb not null default '[]'::jsonb,
  add column if not exists court_open_time time,
  add column if not exists court_close_time time,
  add column if not exists court_notes text;

update public.tournament_registration_days
set
  court_count = coalesce(court_count, 10),
  court_labels = coalesce(court_labels, '[]'::jsonb)
where court_count is null or court_labels is null;

alter table if exists public.tournament_registration_days
  drop constraint if exists tournament_registration_days_court_count_chk,
  add constraint tournament_registration_days_court_count_chk
    check (court_count between 1 and 100),
  drop constraint if exists tournament_registration_days_court_labels_chk,
  add constraint tournament_registration_days_court_labels_chk
    check (jsonb_typeof(court_labels) = 'array'),
  drop constraint if exists tournament_registration_days_court_window_chk,
  add constraint tournament_registration_days_court_window_chk
    check (
      court_open_time is null
      or court_close_time is null
      or court_close_time > court_open_time
    );

comment on column public.tournament_registration_days.court_count is
  'Number of facility courts available for this tournament day.';
comment on column public.tournament_registration_days.court_labels is
  'Ordered JSON array of court labels available for this tournament day.';
comment on column public.tournament_registration_days.court_open_time is
  'Optional first playable court time for this tournament day.';
comment on column public.tournament_registration_days.court_close_time is
  'Optional final playable court time for this tournament day.';
comment on column public.tournament_registration_days.court_notes is
  'Organizer notes about daily court availability or restrictions.';

alter table if exists public.tournament_commerce_bundle_components
  drop constraint if exists tournament_commerce_component_type_chk,
  add constraint tournament_commerce_component_type_chk
    check (component_type in ('EVENT_OPTION', 'EVENT_CHOICE', 'ITEM_VARIANT')),
  drop constraint if exists tournament_commerce_component_target_chk,
  add constraint tournament_commerce_component_target_chk check (
    (
      component_type in ('EVENT_OPTION', 'EVENT_CHOICE')
      and event_option_id is not null
      and item_id is null
      and variant_id is null
    )
    or (
      component_type = 'ITEM_VARIANT'
      and event_option_id is null
      and item_id is not null
      and variant_id is not null
    )
  );

comment on column public.tournament_commerce_bundle_components.component_type is
  'EVENT_OPTION requires one fixed event; EVENT_CHOICE contributes an eligible event to an any-N pool; ITEM_VARIANT requires one fixed extra option.';
comment on column public.tournament_commerce_bundle_components.quantity is
  'For EVENT_CHOICE rows, stores the number of eligible events required; otherwise stores quantity per bundle.';
