-- Follow-up migration: make digest/outbox uniqueness respect the full selected date window.

alter table if exists public.player_weekly_profile_digests
    drop constraint if exists player_weekly_profile_digests_club_id_player_id_week_start_key;

alter table if exists public.player_weekly_profile_digests
    add constraint player_weekly_profile_digests_club_player_window_key
    unique (club_id, player_id, week_start, week_end);

alter table if exists public.player_profile_update_outbox
    drop constraint if exists player_profile_update_outbox_subscription_id_week_start_key;

alter table if exists public.player_profile_update_outbox
    add constraint player_profile_update_outbox_subscription_window_key
    unique (subscription_id, week_start, week_end);

drop index if exists public.player_profile_update_outbox_subscription_week_idx;

create index if not exists player_profile_update_outbox_subscription_week_window_idx
    on public.player_profile_update_outbox (subscription_id, week_start, week_end);
