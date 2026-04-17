alter table if exists tournament_registration_settings
  add column if not exists builder_draft_json jsonb,
  add column if not exists builder_draft_updated_at timestamptz;
