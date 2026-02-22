-- Canonical creation of public.badge_eval_queue lives in:
--   supabase/migrations/20260705_badge_eval_queue.sql
-- Keep this migration for additive hardening/backfill only.

alter table if exists public.badge_eval_queue
  add column if not exists attempts integer not null default 0,
  add column if not exists last_error text,
  add column if not exists processed_at timestamptz;

create index if not exists badge_eval_queue_status_created_idx
    on public.badge_eval_queue (status, created_at);

create index if not exists badge_eval_queue_club_status_idx
    on public.badge_eval_queue (club_id, status);

create unique index if not exists badge_eval_queue_event_match_uidx
    on public.badge_eval_queue (event_type, match_id)
    where match_id is not null;
