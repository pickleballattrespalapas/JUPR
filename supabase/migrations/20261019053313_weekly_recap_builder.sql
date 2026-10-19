-- Preserve the published edition while administrators work on a draft revision.
alter table public.weekly_recaps add column if not exists published_json jsonb;
update public.weekly_recaps set published_json = final_json
where status = 'published' and published_json is null;

create table if not exists public.club_recap_preferences (
  club_id text primary key,
  revision bigint not null default 1 check (revision > 0),
  settings jsonb not null default '{}'::jsonb check (jsonb_typeof(settings) = 'object'),
  updated_at timestamptz not null default now()
);
alter table public.club_recap_preferences enable row level security;
revoke all on public.club_recap_preferences from public, anon, authenticated;
grant select, insert, update, delete on public.club_recap_preferences to service_role;
comment on table public.club_recap_preferences is 'Server-only recap layouts; FastAPI verifies club administration before access.';
