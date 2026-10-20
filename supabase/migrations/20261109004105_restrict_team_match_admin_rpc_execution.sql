-- These legacy team-match admin RPCs trust the server to authorize the caller.
-- Their SECURITY DEFINER bodies do not authorize anonymous or signed-in users.
-- Revoke inherited PUBLIC access as well as any direct client-role grants,
-- preserving the service-role boundary without changing bodies or data.
-- Some installations do not include this legacy feature; absent signatures
-- are intentionally skipped. Other overloads and routines are untouched.
begin;

do $permissions$
declare
  routine_signature text;
  routine_oid oid;
begin
  foreach routine_signature in array array[
    'public.admin_replace_team_match_schedule_v1(text,text,integer,jsonb,text)',
    'public.admin_save_team_match_team_v1(text,text,text,text,text,jsonb,integer,integer,text)',
    'public.admin_update_team_match_competition_v1(text,text,integer,jsonb,text)'
  ]
  loop
    routine_oid := pg_catalog.to_regprocedure(routine_signature);
    if routine_oid is null then
      continue;
    end if;

    execute pg_catalog.format(
      'revoke execute on function %s from public, anon, authenticated',
      routine_signature
    );
    execute pg_catalog.format(
      'grant execute on function %s to service_role', routine_signature
    );

    if pg_catalog.has_function_privilege('anon', routine_oid, 'EXECUTE')
       or pg_catalog.has_function_privilege('authenticated', routine_oid, 'EXECUTE')
       or not pg_catalog.has_function_privilege('service_role', routine_oid, 'EXECUTE') then
      raise exception 'Team-match RPC permissions did not close the client boundary: %',
        routine_signature;
    end if;
  end loop;
end
$permissions$;

commit;
