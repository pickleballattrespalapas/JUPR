-- Structured Tournament Setup basics metadata.
-- Staging acceptance requires a venue/location, an IANA timezone, and editable sponsors.

alter table public.tournament_registration_settings
  add column if not exists location_name text,
  add column if not exists timezone text,
  add column if not exists sponsors_json jsonb not null default '[]'::jsonb;

comment on column public.tournament_registration_settings.location_name is
  'Operator-entered tournament venue or location shown in setup and public summaries.';

comment on column public.tournament_registration_settings.timezone is
  'IANA timezone used for tournament registration and schedule timestamps.';

comment on column public.tournament_registration_settings.sponsors_json is
  'Ordered sponsor records managed by Tournament Setup.';
