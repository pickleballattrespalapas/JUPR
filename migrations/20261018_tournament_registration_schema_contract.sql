-- Tournament registration schema contract hardening (builder-era model)
-- Apply after: migrations/20261010_tournament_builder_refactor.sql

alter table if exists tournament_registration_settings
  add column if not exists builder_draft_json jsonb,
  add column if not exists builder_draft_updated_at timestamptz;

alter table if exists tournament_registration_days
  add column if not exists enabled boolean not null default true;

update tournament_registration_days
set enabled = true
where enabled is null;

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

update tournament_event_options
set
  waitlist_enabled = coalesce(waitlist_enabled, true),
  partner_board_enabled = coalesce(partner_board_enabled, coalesce(public_partner_board, true)),
  enabled = coalesce(enabled, true),
  status = coalesce(nullif(lower(trim(status)), ''), 'draft'),
  division_name = coalesce(nullif(division_name, ''), nullif(label, '')),
  event_family_label = coalesce(nullif(event_family_label, ''), nullif(label, '')),
  event_format_default = coalesce(nullif(event_format_default, ''), 'ROUND_ROBIN_PLUS_PLAYOFF'),
  scoring_default = coalesce(nullif(scoring_default, ''), 'GAME_TO_15'),
  age_mode = coalesce(nullif(age_mode, ''), 'ALL_AGES'),
  skill_mode = coalesce(nullif(skill_mode, ''), case when lower(coalesce(skill_label, 'open')) = 'open' then 'OPEN' else 'SKILL_BRACKET' end)
where true;

-- Optional rollback (destructive: removes builder-era columns)
-- alter table if exists tournament_event_options
--   drop column if exists event_family_label,
--   drop column if exists division_name,
--   drop column if exists event_format_default,
--   drop column if exists scoring_default,
--   drop column if exists event_format_override,
--   drop column if exists scoring_override,
--   drop column if exists skill_mode,
--   drop column if exists age_mode,
--   drop column if exists age_rules,
--   drop column if exists waitlist_enabled,
--   drop column if exists partner_board_enabled,
--   drop column if exists status,
--   drop column if exists enabled;
--
-- alter table if exists tournament_registration_days
--   drop column if exists enabled;
--
-- alter table if exists tournament_registration_settings
--   drop column if exists builder_draft_json,
--   drop column if exists builder_draft_updated_at;
