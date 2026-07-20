-- JUPR's Postgres data plane is server-only. Browsers use Supabase Auth, then
-- call the FastAPI service with the short-lived user JWT. FastAPI performs the
-- club/role/permission check and uses the service role for database access.
--
-- Keep this migration intentionally idempotent. It canonicalizes the staging
-- hardening that was applied during the Next migration and makes fresh projects
-- opt out of direct Data API access by default.

revoke all on schema public from public, anon, authenticated;
grant usage on schema public to service_role;

do $lockdown_tables$
declare
  relation record;
begin
  for relation in
    select namespace.nspname as schema_name,
           class.relname as relation_name,
           class.relkind as relation_kind
      from pg_catalog.pg_class as class
      join pg_catalog.pg_namespace as namespace
        on namespace.oid = class.relnamespace
     where namespace.nspname = 'public'
       and class.relkind in ('r', 'p', 'v', 'm')
  loop
    if relation.relation_kind in ('r', 'p') then
      execute format(
        'alter table %I.%I enable row level security',
        relation.schema_name,
        relation.relation_name
      );
    end if;

    execute format(
      'revoke all on table %I.%I from public, anon, authenticated',
      relation.schema_name,
      relation.relation_name
    );

    if relation.relation_kind in ('v', 'm') then
      execute format(
        'grant select on table %I.%I to service_role',
        relation.schema_name,
        relation.relation_name
      );
    else
      execute format(
        'grant all privileges on table %I.%I to service_role',
        relation.schema_name,
        relation.relation_name
      );
    end if;
  end loop;
end
$lockdown_tables$;

do $lockdown_sequences$
declare
  sequence_record record;
begin
  for sequence_record in
    select namespace.nspname as schema_name,
           class.relname as sequence_name
      from pg_catalog.pg_class as class
      join pg_catalog.pg_namespace as namespace
        on namespace.oid = class.relnamespace
     where namespace.nspname = 'public'
       and class.relkind = 'S'
  loop
    execute format(
      'revoke all on sequence %I.%I from public, anon, authenticated',
      sequence_record.schema_name,
      sequence_record.sequence_name
    );
    execute format(
      'grant usage, select, update on sequence %I.%I to service_role',
      sequence_record.schema_name,
      sequence_record.sequence_name
    );
  end loop;
end
$lockdown_sequences$;

do $lockdown_functions$
declare
  function_record record;
begin
  for function_record in
    select namespace.nspname as schema_name,
           proc.proname as function_name,
           pg_catalog.pg_get_function_identity_arguments(proc.oid) as identity_arguments
      from pg_catalog.pg_proc as proc
      join pg_catalog.pg_namespace as namespace
        on namespace.oid = proc.pronamespace
     where namespace.nspname = 'public'
  loop
    execute format(
      'revoke execute on function %I.%I(%s) from public, anon, authenticated',
      function_record.schema_name,
      function_record.function_name,
      function_record.identity_arguments
    );
    execute format(
      'grant execute on function %I.%I(%s) to service_role',
      function_record.schema_name,
      function_record.function_name,
      function_record.identity_arguments
    );
  end loop;
end
$lockdown_functions$;

-- Existing projects historically auto-granted public-schema objects to the
-- Data API roles. Make all future exposure an explicit migration decision while
-- retaining service-role access for the FastAPI data plane.
alter default privileges for role postgres in schema public
  revoke all on tables from public, anon, authenticated;
alter default privileges for role postgres in schema public
  grant all on tables to service_role;

alter default privileges for role postgres in schema public
  revoke all on sequences from public, anon, authenticated;
alter default privileges for role postgres in schema public
  grant usage, select, update on sequences to service_role;

alter default privileges for role postgres in schema public
  revoke execute on functions from public, anon, authenticated;
alter default privileges for role postgres in schema public
  grant execute on functions to service_role;

notify pgrst, 'reload schema';
