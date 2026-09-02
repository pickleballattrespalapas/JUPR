-- Pin routine lookup paths so callers cannot influence object resolution.
-- pg_temp is explicit and last to prevent temporary objects shadowing trusted
-- objects used by SECURITY DEFINER routines.

begin;

do $block$
declare
    target record;
begin
    for target in
        select
            n.nspname as schema_name,
            p.proname as routine_name,
            pg_get_function_identity_arguments(p.oid) as identity_arguments,
            case p.prokind when 'p' then 'procedure' else 'function' end as routine_kind
        from pg_proc p
        join pg_namespace n on n.oid = p.pronamespace
        where n.nspname = 'public'
          and p.prokind in ('f', 'p')
    loop
        execute format(
            'alter %s %I.%I(%s) set search_path to pg_catalog, public, extensions, pg_temp',
            target.routine_kind,
            target.schema_name,
            target.routine_name,
            target.identity_arguments
        );
    end loop;
end
$block$;

commit;
