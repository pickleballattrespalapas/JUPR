-- Repair the privileges of the team-match management RPCs on databases where
-- the original staging-only migration left PostgreSQL's default PUBLIC EXECUTE
-- grant in place. The historical canonical source is secure for fresh installs;
-- this later migration is required to harden already-applied ledgers safely.

alter function public.admin_update_team_match_competition_v1(
  text,
  text,
  integer,
  jsonb,
  text
) security definer;
alter function public.admin_update_team_match_competition_v1(
  text,
  text,
  integer,
  jsonb,
  text
) set search_path = '';
revoke all on function public.admin_update_team_match_competition_v1(
  text,
  text,
  integer,
  jsonb,
  text
) from public, anon, authenticated;
grant execute on function public.admin_update_team_match_competition_v1(
  text,
  text,
  integer,
  jsonb,
  text
) to service_role;

alter function public.admin_save_team_match_team_v1(
  text,
  text,
  text,
  text,
  text,
  jsonb,
  integer,
  integer,
  text
) security definer;
alter function public.admin_save_team_match_team_v1(
  text,
  text,
  text,
  text,
  text,
  jsonb,
  integer,
  integer,
  text
) set search_path = '';
revoke all on function public.admin_save_team_match_team_v1(
  text,
  text,
  text,
  text,
  text,
  jsonb,
  integer,
  integer,
  text
) from public, anon, authenticated;
grant execute on function public.admin_save_team_match_team_v1(
  text,
  text,
  text,
  text,
  text,
  jsonb,
  integer,
  integer,
  text
) to service_role;

alter function public.admin_replace_team_match_schedule_v1(
  text,
  text,
  integer,
  jsonb,
  text
) security definer;
alter function public.admin_replace_team_match_schedule_v1(
  text,
  text,
  integer,
  jsonb,
  text
) set search_path = '';
revoke all on function public.admin_replace_team_match_schedule_v1(
  text,
  text,
  integer,
  jsonb,
  text
) from public, anon, authenticated;
grant execute on function public.admin_replace_team_match_schedule_v1(
  text,
  text,
  integer,
  jsonb,
  text
) to service_role;

notify pgrst, 'reload schema';
