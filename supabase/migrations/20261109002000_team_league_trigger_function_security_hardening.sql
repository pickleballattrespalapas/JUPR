-- Harden the two Team League trigger functions created after the broad
-- public-function search_path repair. Trigger execution does not require
-- client Data API privileges, so keep the functions server-only as well.

alter function public.prevent_league_competition_mode_change()
  set search_path = '';
revoke all on function public.prevent_league_competition_mode_change()
  from public, anon, authenticated;
grant execute on function public.prevent_league_competition_mode_change()
  to service_role;

alter function public.require_team_league_mode()
  set search_path = '';
revoke all on function public.require_team_league_mode()
  from public, anon, authenticated;
grant execute on function public.require_team_league_mode()
  to service_role;

notify pgrst, 'reload schema';
