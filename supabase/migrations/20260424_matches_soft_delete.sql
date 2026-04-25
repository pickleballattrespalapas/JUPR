-- Soft-delete and correction metadata for rated match history integrity.
alter table if exists public.matches
  add column if not exists deleted_at timestamptz,
  add column if not exists deleted_by text,
  add column if not exists deleted_source text,
  add column if not exists delete_note text,
  add column if not exists rating_scope text,
  add column if not exists updated_at timestamptz,
  add column if not exists updated_by text,
  add column if not exists correction_note text;

alter table if exists public.matches
  drop constraint if exists matches_rating_scope_check;

alter table if exists public.matches
  add constraint matches_rating_scope_check
  check (
    rating_scope is null
    or rating_scope = ''
    or rating_scope = 'overall_only'
    or rating_scope = 'unrated'
  );
