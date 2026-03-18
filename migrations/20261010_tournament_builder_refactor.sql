alter table if exists tournament_registration_settings
  drop column if exists max_events_per_day;

alter table if exists tournament_registration_days
  add column if not exists enabled boolean not null default true;

alter table if exists tournament_event_options
  add column if not exists event_family_label text,
  add column if not exists division_name text,
  add column if not exists event_format_default text,
  add column if not exists scoring_default text,
  add column if not exists event_format_override text,
  add column if not exists scoring_override text,
  add column if not exists skill_mode text,
  add column if not exists age_mode text,
  add column if not exists age_rules text,
  add column if not exists waitlist_enabled boolean not null default true,
  add column if not exists partner_board_enabled boolean not null default true,
  add column if not exists status text,
  add column if not exists enabled boolean not null default true;
