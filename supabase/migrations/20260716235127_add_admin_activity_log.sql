create table if not exists public.admin_activity_log (
  id bigint generated always as identity primary key,
  club_id text not null,
  actor_email text not null,
  actor_role text not null,
  action_type text not null,
  entity_type text not null,
  entity_id text not null,
  before_json jsonb,
  after_json jsonb,
  note text,
  source_page text,
  flagged_for_review boolean not null default false,
  created_at timestamptz not null default now()
);
create index if not exists admin_activity_log_club_created_idx on public.admin_activity_log (club_id, created_at desc);
create index if not exists admin_activity_log_flagged_idx on public.admin_activity_log (club_id, flagged_for_review, created_at desc);
comment on table public.admin_activity_log is 'Admin activity audit log. Retain approximately 1 year for trust and operational review.';
comment on column public.admin_activity_log.note is 'Optional reason/note captured from operator workflow context.';
