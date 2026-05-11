alter table if exists public.clubs
  add column if not exists plan_status text not null default 'pilot',
  add column if not exists features_json jsonb not null default '{}'::jsonb,
  add column if not exists created_by_email text,
  add column if not exists onboarding_status text not null default 'draft';

create index if not exists clubs_slug_idx on public.clubs (slug);
create index if not exists clubs_active_idx on public.clubs (is_active);
