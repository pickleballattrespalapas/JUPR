-- Remove platform-default privileges from the private check-in surface.
--
-- The canonical creation migration already grants the service role only the
-- reads and writes used by FastAPI. Older project defaults can grant broader
-- table privileges at object creation, so rebind the ACL explicitly for
-- already-provisioned staging databases too.

do $migration_preflight$
begin
  if to_regclass('public.tournament_registration_check_ins') is null
     or to_regprocedure(
       'public.admin_upsert_tournament_registration_check_in(text,text,text,timestamptz,boolean,boolean,integer,text,text,text)'
     ) is null then
    raise exception using
      errcode = '42P01',
      message = 'Tournament registration check-in storage must exist before its privileges are hardened.';
  end if;
end
$migration_preflight$;

revoke all on table public.tournament_registration_check_ins
  from public, anon, authenticated, service_role;
grant select, insert, update on table public.tournament_registration_check_ins
  to service_role;

revoke all on function public.admin_upsert_tournament_registration_check_in(
  text, text, text, timestamptz, boolean, boolean, integer, text, text, text
) from public, anon, authenticated, service_role;
grant execute on function public.admin_upsert_tournament_registration_check_in(
  text, text, text, timestamptz, boolean, boolean, integer, text, text, text
) to service_role;
