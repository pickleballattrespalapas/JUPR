alter table public.weekly_recaps
    add column if not exists visibility text not null default 'public',
    add column if not exists featured_upcoming_event jsonb not null default '{}'::jsonb,
    add column if not exists featured_past_event jsonb null,
    add column if not exists content_snapshot jsonb not null default '{}'::jsonb;

alter table public.weekly_recaps
    drop constraint if exists weekly_recaps_visibility_check;

alter table public.weekly_recaps
    add constraint weekly_recaps_visibility_check
    check (visibility in ('public', 'unlisted', 'private'));
