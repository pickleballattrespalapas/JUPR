-- Player inactivity controls leaderboard visibility, not identity availability.
-- Preserve each routine's authentication, club boundaries, membership, CAS,
-- transaction, and rating checks. Retained merge-source identities stay blocked.
-- Existing function ACLs, security mode, search_path, and signatures are preserved.
-- Scope this production patch to interclub participation only.
begin;

do $activity$
declare
  patch record;
  routine_count integer;
  routine_oid oid;
  definition text;
  occurrences integer;
begin
  for patch in
    select * from (values
      ('pcs_interclub_pool_search_players', $old$p.active is true$old$, $new$pg_catalog.strpos(pg_catalog.lower(coalesce(p.name, '')), '(merged into ') = 0$new$, 1, false),
      ('pcs_reconcile_interclub_meet_signups', $old$p.active is not true$old$, $new$p.id is null or pg_catalog.strpos(pg_catalog.lower(coalesce(p.name, '')), '(merged into ') > 0$new$, 1, false),
      ('pcs_save_interclub_meet_roster_before_signups', $old$player.active is not true$old$, $new$player.id is null or pg_catalog.strpos(pg_catalog.lower(coalesce(player.name, '')), '(merged into ') > 0$new$, 1, false),
      ('pcs_save_interclub_meet_roster_before_signups', $old$Choose active players from this club$old$, $new$Choose current player profiles from this club$new$, 1, false)
    ) as changes(function_name, old_text, new_text, expected_count, required)
  loop
    select count(*), min(p.oid::bigint)::oid
      into routine_count, routine_oid
      from pg_catalog.pg_proc p
      join pg_catalog.pg_namespace n on n.oid = p.pronamespace
     where n.nspname = 'public' and p.proname = patch.function_name
       and p.prokind = 'f';
    if routine_count = 0 and not patch.required then
      continue;
    end if;
    if routine_count <> 1 then
      raise exception 'Expected one existing routine for %, found %',
        patch.function_name, routine_count;
    end if;
    definition := pg_catalog.pg_get_functiondef(routine_oid);
    occurrences := (length(definition) - length(replace(definition, patch.old_text, '')))
      / length(patch.old_text);
    if occurrences = 0 and strpos(definition, patch.new_text) > 0 then
      continue; -- Safe to verify or reapply an already installed change.
    end if;
    if occurrences <> patch.expected_count then
      raise exception 'Player activity patch for % expected % occurrences, found %',
        patch.function_name, patch.expected_count, occurrences;
    end if;
    execute replace(definition, patch.old_text, patch.new_text);
  end loop;
end
$activity$;

commit;
