alter table public.tournament_registration_settings
  add column if not exists weather_policy_markdown text;

alter table public.tournament_event_options
  add column if not exists scheduled_day_ids jsonb not null default '[]'::jsonb;

update public.tournament_event_options
set scheduled_day_ids = jsonb_build_array(registration_day_id)
where coalesce(jsonb_array_length(scheduled_day_ids), 0) = 0
  and registration_day_id is not null
  and trim(registration_day_id) <> '';

comment on column public.tournament_registration_settings.weather_policy_markdown is
  'Public tournament weather, delay, reschedule, and cancellation policy.';

comment on column public.tournament_event_options.scheduled_day_ids is
  'Ordered tournament day IDs on which this registered division may be scheduled. registration_day_id remains the primary grouping day.';
