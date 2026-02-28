-- Ensure badge async infrastructure is writable from app clients.
-- This keeps queue/facts persistence aligned with player_badges behavior.

grant select, insert, update on public.badge_eval_queue to anon, authenticated;
grant select, insert, update on public.player_badge_facts to anon, authenticated;
