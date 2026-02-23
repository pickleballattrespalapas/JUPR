begin;

alter table if exists public.session_ladder_sessions
  add column if not exists published_snapshot_json jsonb null;

update public.session_ladder_sessions
set published_snapshot_json = jsonb_build_object(
  'recap', coalesce(recap_json, '{}'::jsonb),
  'leaderboard', coalesce(leaderboard_json, '[]'::jsonb),
  'attendance', '{}'::jsonb
)
where published_snapshot_json is null
  and (published_at is not null or state = 'published')
  and (recap_json is not null or leaderboard_json is not null);

commit;
