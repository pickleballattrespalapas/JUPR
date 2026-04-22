-- Allow a new outbox row every time a digest is generated, even for the same
-- subscription and selected date window.
--
-- Keep digest uniqueness on (club_id, player_id, week_start, week_end), but
-- remove outbox uniqueness so repeated generations create fresh pending sends.

alter table if exists public.player_profile_update_outbox
    drop constraint if exists player_profile_update_outbox_subscription_window_key;

alter table if exists public.player_profile_update_outbox
    drop constraint if exists player_profile_update_outbox_subscription_id_week_start_key;

drop index if exists public.player_profile_update_outbox_subscription_week_window_idx;
drop index if exists public.player_profile_update_outbox_subscription_week_idx;

create index if not exists player_profile_update_outbox_subscription_week_window_idx
    on public.player_profile_update_outbox (subscription_id, week_start, week_end);
