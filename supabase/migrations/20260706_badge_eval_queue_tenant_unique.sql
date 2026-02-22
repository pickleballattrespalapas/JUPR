drop index if exists public.badge_eval_queue_event_match_uidx;

create unique index if not exists badge_eval_queue_club_event_match_uidx
    on public.badge_eval_queue (club_id, event_type, match_id)
    where match_id is not null;
