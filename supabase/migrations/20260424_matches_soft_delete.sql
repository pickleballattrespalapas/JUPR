-- Soft-delete and correction metadata for rated match history integrity.
alter table if exists public.matches
  add column if not exists deleted_at timestamptz,
  add column if not exists deleted_by text,
  add column if not exists deleted_source text,
  add column if not exists delete_note text,
  add column if not exists updated_at timestamptz,
  add column if not exists updated_by text,
  add column if not exists correction_note text;
