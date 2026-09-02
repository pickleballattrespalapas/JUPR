-- Canonicalize public views for the server-only FastAPI data plane. Views are
-- SECURITY DEFINER by default in PostgreSQL, so set security_invoker even when
-- direct Data API privileges are revoked. Either legacy view may be absent on
-- a project whose prerequisite tables were unavailable when it was created.

do $server_only_view_security$
declare
  view_name text;
  relation_kind text;
begin
  foreach view_name in array array[
    'league_settings',
    'public_leaderboards'
  ]
  loop
    relation_kind := null;
    select class.relkind::text
      into relation_kind
      from pg_catalog.pg_class as class
      join pg_catalog.pg_namespace as namespace
        on namespace.oid = class.relnamespace
     where namespace.nspname = 'public'
       and class.relname = view_name;

    if relation_kind is null then
      continue;
    end if;
    if relation_kind <> 'v' then
      raise exception using
        errcode = '42809',
        message = pg_catalog.format(
          'public.%I must be a regular view before security hardening',
          view_name
        );
    end if;

    execute pg_catalog.format(
      'alter view %I.%I set (security_invoker = true)',
      'public',
      view_name
    );
    execute pg_catalog.format(
      'revoke all on table %I.%I from public, anon, authenticated, service_role',
      'public',
      view_name
    );
    execute pg_catalog.format(
      'grant select on table %I.%I to service_role',
      'public',
      view_name
    );
  end loop;
end
$server_only_view_security$;

notify pgrst, 'reload schema';
