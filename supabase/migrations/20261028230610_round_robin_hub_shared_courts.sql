-- A dedicated court excludes other groups; its owner may also use shared courts.
-- Patch only that restriction, preserving the deployed transaction, identity,
-- concurrency, lease and access-control safeguards and existing hub/session data.
do $migration$
declare
  definition text;
  restriction text := $restriction$
         or (nullif(court->>'dedicated_session_key','') is null and exists (
           select 1 from jsonb_array_elements(p_state->'courts') c where c->>'dedicated_session_key'=assignment->>'session_key'))$restriction$;
begin
  select pg_get_functiondef('public.commit_round_robin_hub(text,uuid,bigint,text,text,text,text,jsonb,jsonb,text)'::regprocedure)
    into definition;
  if (length(definition) - length(replace(definition, restriction, ''))) / length(restriction) <> 1 then
    raise exception 'Expected exactly one legacy shared-court restriction in commit_round_robin_hub; review before migrating.';
  end if;
  execute replace(definition, restriction, '');
end
$migration$;
