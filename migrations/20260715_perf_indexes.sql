create index if not exists player_badges_club_player_idx
    on public.player_badges (club_id, player_id);

create index if not exists player_badges_club_badge_idx
    on public.player_badges (club_id, badge_id);

create index if not exists player_badges_club_player_badge_active_idx
    on public.player_badges (club_id, player_id, badge_id)
    where revoked_at is null;

create index if not exists badge_eval_queue_club_status_created_idx
    on public.badge_eval_queue (club_id, status, created_at);

create index if not exists player_badge_facts_lookup_idx
    on public.player_badge_facts (club_id, player_id, context_id);
