-- Close the inherited PUBLIC-role gap left by role-specific Data API revokes.
begin;

revoke all privileges on schema public from public, anon, authenticated;
grant usage on schema public to service_role;

do $block$
declare
    target record;
begin
    for target in
        select n.nspname as schema_name, c.relname as relation_name
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = 'public'
          and c.relkind in ('r', 'p', 'v', 'm', 'f')
    loop
        execute format(
            'revoke all privileges on table %I.%I from public, anon, authenticated',
            target.schema_name,
            target.relation_name
        );
    end loop;
end
$block$;

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
            'revoke all privileges on %s %I.%I(%s) from public, anon, authenticated',
            target.routine_kind,
            target.schema_name,
            target.routine_name,
            target.identity_arguments
        );
        execute format(
            'grant execute on %s %I.%I(%s) to service_role',
            target.routine_kind,
            target.schema_name,
            target.routine_name,
            target.identity_arguments
        );
    end loop;
end
$block$;

revoke all privileges on all sequences in schema public from public, anon, authenticated;
grant all privileges on all sequences in schema public to service_role;

alter default privileges in schema public
    revoke all privileges on tables from public, anon, authenticated;
alter default privileges in schema public
    revoke all privileges on sequences from public, anon, authenticated;
alter default privileges in schema public
    revoke execute on functions from public, anon, authenticated;

commit;
