begin;

-- An immutable request with stale application evidence cannot succeed by
-- retrying its transaction. PostgREST treats 40001 as a serialization failure
-- and can loop forever. Only these explicit badge business-conflict raises
-- change to HTTP 409; genuine PostgreSQL serialization failures are untouched.
-- Preserve the installed function bodies, signatures, locks, ACLs and security
-- settings instead of replacing them from an older environment's snapshot.
do $migration$
declare
  routine record;
  routine_oid regprocedure;
  definition text;
  old_conflicts integer;
  fixed_conflicts integer;
begin
  for routine in
    select * from (values
      ('public.apply_program_badges_v1(text,bigint,jsonb,jsonb,jsonb)', 1),
      ('public.admin_resolve_round_robin_v1(text,text,uuid,text,uuid,jsonb)', 3),
      ('public.admin_manage_badges_v1(text,text,uuid,text,uuid,text,jsonb)', 1),
      ('public.validate_badge_season_award_v1()', 1)
    ) as expected(signature, conflict_count)
  loop
    routine_oid := to_regprocedure(routine.signature);
    if routine_oid is null then
      raise exception 'Badge conflict prerequisite is missing: %', routine.signature;
    end if;
    definition := pg_get_functiondef(routine_oid);
    select count(*) into old_conflicts
      from regexp_matches(definition, 'errcode\s*=\s*''40001''', 'g');
    select count(*) into fixed_conflicts
      from regexp_matches(definition, 'errcode\s*=\s*''PT409''', 'g');
    if old_conflicts + fixed_conflicts <> routine.conflict_count then
      raise exception 'Badge conflict definition changed for %; expected % reviewed raises, found %',
        routine.signature, routine.conflict_count, old_conflicts + fixed_conflicts;
    end if;
    if old_conflicts > 0 then
      execute regexp_replace(definition, 'errcode\s*=\s*''40001''', 'errcode=''PT409''', 'g');
    end if;
  end loop;
end;
$migration$;

notify pgrst, 'reload schema';
commit;
