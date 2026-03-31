alter table if exists public.live_events
    add column if not exists submitted_by_name text null,
    add column if not exists submission_mode text null,
    add column if not exists moderated_at timestamptz null,
    add column if not exists moderated_by text null,
    add column if not exists rejection_reason text null;

create index if not exists live_events_social_review_idx
    on public.live_events (club_id, result_mode, status, updated_at desc);
