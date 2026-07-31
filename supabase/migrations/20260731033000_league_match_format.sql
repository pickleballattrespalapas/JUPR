-- Separate official leagues by match format so singles matches cannot be
-- entered into doubles leagues, and vice versa.

alter table public.leagues_metadata
  add column if not exists match_format text;

update public.leagues_metadata
   set match_format = 'doubles'
 where match_format is null
    or pg_catalog.lower(pg_catalog.btrim(match_format)) not in ('doubles', 'singles');

alter table public.leagues_metadata
  alter column match_format set default 'doubles';

alter table public.leagues_metadata
  alter column match_format set not null;

alter table public.leagues_metadata
  drop constraint if exists leagues_metadata_match_format_check;

alter table public.leagues_metadata
  add constraint leagues_metadata_match_format_check
  check (match_format in ('doubles', 'singles'));

create index if not exists leagues_metadata_active_match_format_idx
  on public.leagues_metadata (club_id, match_format, is_active, status)
  where ended_at is null;

comment on column public.leagues_metadata.match_format is
  'Official league match format. Doubles and singles leagues are intentionally separate.';

notify pgrst, 'reload schema';
