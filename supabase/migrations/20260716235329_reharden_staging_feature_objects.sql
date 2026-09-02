grant usage on schema public to service_role;

do $block$
declare target record;
begin
  for target in
    select n.nspname as schema_name, c.relname as relation_name
    from pg_class c join pg_namespace n on n.oid = c.relnamespace
    where n.nspname = 'public' and c.relkind in ('r', 'p')
  loop
    execute format('alter table %I.%I enable row level security', target.schema_name, target.relation_name);
    execute format('revoke all privileges on table %I.%I from anon, authenticated', target.schema_name, target.relation_name);
    execute format('grant all privileges on table %I.%I to service_role', target.schema_name, target.relation_name);
  end loop;
end
$block$;

do $block$
declare target record;
begin
  for target in
    select schemaname, tablename, policyname from pg_policies where schemaname = 'public'
  loop
    execute format('drop policy %I on %I.%I', target.policyname, target.schemaname, target.tablename);
  end loop;
end
$block$;

do $block$
declare target record;
begin
  for target in select schemaname, viewname from pg_views where schemaname = 'public'
  loop
    execute format('alter view %I.%I set (security_invoker = true)', target.schemaname, target.viewname);
    execute format('revoke all privileges on table %I.%I from anon, authenticated', target.schemaname, target.viewname);
    execute format('grant select on table %I.%I to service_role', target.schemaname, target.viewname);
  end loop;
end
$block$;

do $block$
declare target record;
begin
  for target in
    select n.nspname as schema_name, p.proname as routine_name,
           pg_get_function_identity_arguments(p.oid) as identity_arguments,
           case p.prokind when 'p' then 'procedure' else 'function' end as routine_kind
    from pg_proc p join pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public' and p.prokind in ('f', 'p')
  loop
    execute format(
      'revoke all privileges on %s %I.%I(%s) from public, anon, authenticated',
      target.routine_kind, target.schema_name, target.routine_name, target.identity_arguments
    );
    execute format(
      'grant execute on %s %I.%I(%s) to service_role',
      target.routine_kind, target.schema_name, target.routine_name, target.identity_arguments
    );
    execute format(
      'alter %s %I.%I(%s) set search_path to pg_catalog, public, extensions, pg_temp',
      target.routine_kind, target.schema_name, target.routine_name, target.identity_arguments
    );
  end loop;
end
$block$;

revoke all privileges on all sequences in schema public from anon, authenticated;
grant all privileges on all sequences in schema public to service_role;

alter default privileges in schema public revoke all privileges on tables from anon, authenticated;
alter default privileges in schema public revoke execute on functions from public, anon, authenticated;
alter default privileges in schema public grant all privileges on tables to service_role;
alter default privileges in schema public grant all privileges on sequences to service_role;
alter default privileges in schema public grant execute on functions to service_role;

notify pgrst, 'reload schema';
