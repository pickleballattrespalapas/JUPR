-- Enforce combined-source eligibility where a draw becomes active or a game
-- receives a court. Existing score entry, pause, cancellation and release paths
-- deliberately remain available when a source division needs repair.
create or replace function private.assert_tournament_combined_draw_sources(
  p_draw_id uuid,
  p_registration_day_id text
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_source_ids text[];
  v_family text;
  v_count integer;
begin
  select draw.* into v_draw
    from public.tournament_event_draws as draw
   where draw.id = p_draw_id
   for share;
  if not found then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMBINED_SOURCE_INVALID: playing draw is unavailable.';
  end if;
  v_source_ids := public.tournament_draw_source_event_ids(pg_catalog.to_jsonb(v_draw));
  if coalesce(cardinality(v_source_ids), 0) < 2 then
    -- Existing single-division RPC guards retain their exact behavior.
    return;
  end if;
  if v_draw.registration_day_id is distinct from p_registration_day_id
     or p_registration_day_id is null
     or not coalesce(v_draw.event_option_id = any(v_source_ids), false)
     or cardinality(v_source_ids) <> (
       select count(distinct value) from unnest(v_source_ids) as value
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMBINED_SOURCE_INVALID: combined draw sources or day changed.';
  end if;

  -- Row locks keep a source edit from racing the final assignment write.
  perform event.id
    from public.tournament_event_options as event
   where event.id = any(v_source_ids)
   order by event.id
   for share;
  select count(*) into v_count
    from public.tournament_event_options as event
   where event.id = any(v_source_ids);
  if v_count <> cardinality(v_source_ids) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMBINED_SOURCE_INVALID: a combined registration division is missing.';
  end if;
  select private.tournament_combined_draw_family(pg_catalog.to_jsonb(event))
    into v_family
    from public.tournament_event_options as event
   where event.id = v_draw.event_option_id;
  for v_event in
    select event.* from public.tournament_event_options as event
     where event.id = any(v_source_ids)
     order by event.id
  loop
    if v_event.tournament_id::text is distinct from v_draw.tournament_id::text
       or v_event.registration_day_id is distinct from p_registration_day_id
       or v_event.enabled is not true
       or lower(coalesce(v_event.status, '')) in
         ('cancelled', 'canceled', 'inactive', 'archived', 'disabled', 'deleted', 'void', 'voided')
       or upper(coalesce(v_event.competition_format, 'STANDARD')) <> 'STANDARD'
       or upper(btrim(coalesce(pg_catalog.to_jsonb(v_event)->>'eligibility_mode', ''))) = 'COMBINED_RATING_CAP'
       or upper(btrim(coalesce(pg_catalog.to_jsonb(v_event)->>'skill_mode', ''))) = 'COMBINED_RATING_CAP'
       or private.tournament_combined_draw_family(pg_catalog.to_jsonb(v_event))
            is distinct from v_family
       or (case
         when v_event.scheduled_day_ids is null then true
         when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
           then pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 0
             or v_event.scheduled_day_ids ? p_registration_day_id
         else false
       end) is not true then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_COMBINED_SOURCE_INVALID: every combined division must be compatible, enabled and scheduled for this day.';
    end if;
  end loop;
end;
$function$;

create or replace function private.guard_tournament_combined_draw_activation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if new.state = 'ACTIVE' then
    if tg_op = 'INSERT' then
      perform private.assert_tournament_combined_draw_sources(new.draw_id, new.registration_day_id);
    elsif new.state is distinct from old.state
       or new.draw_id is distinct from old.draw_id
       or new.registration_day_id is distinct from old.registration_day_id then
      perform private.assert_tournament_combined_draw_sources(new.draw_id, new.registration_day_id);
    end if;
  end if;
  return new;
end;
$function$;

create or replace function private.guard_tournament_combined_draw_assignment()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if new.state in ('HELD', 'CALLED', 'ON_COURT', 'RESERVED')
     and new.released_at is null then
    if tg_op = 'INSERT' then
      perform private.assert_tournament_combined_draw_sources(new.draw_id, new.registration_day_id);
    elsif new.state is distinct from old.state
       or new.draw_id is distinct from old.draw_id
       or new.registration_day_id is distinct from old.registration_day_id
       or new.court_id is distinct from old.court_id
       or new.reserved_court_id is distinct from old.reserved_court_id
       or old.released_at is not null then
      perform private.assert_tournament_combined_draw_sources(new.draw_id, new.registration_day_id);
    end if;
  end if;
  return new;
end;
$function$;

drop trigger if exists trg_tournament_combined_draw_activation
  on public.tournament_day_live_draws;
create trigger trg_tournament_combined_draw_activation
before insert or update of state, draw_id, registration_day_id
on public.tournament_day_live_draws
for each row execute function private.guard_tournament_combined_draw_activation();

drop trigger if exists trg_tournament_combined_draw_assignment
  on public.tournament_day_live_queue;
create trigger trg_tournament_combined_draw_assignment
before insert or update of state, draw_id, registration_day_id, court_id, reserved_court_id, released_at
on public.tournament_day_live_queue
for each row execute function private.guard_tournament_combined_draw_assignment();

revoke all on function private.assert_tournament_combined_draw_sources(uuid,text)
  from public, anon, authenticated;
revoke all on function private.guard_tournament_combined_draw_activation()
  from public, anon, authenticated;
revoke all on function private.guard_tournament_combined_draw_assignment()
  from public, anon, authenticated;
grant execute on function private.assert_tournament_combined_draw_sources(uuid,text)
  to service_role;
grant execute on function private.guard_tournament_combined_draw_activation()
  to service_role;
grant execute on function private.guard_tournament_combined_draw_assignment()
  to service_role;

-- Releasing a scored court can promote its reservation in the same transaction.
-- An invalid reserved draw must return to WAITING through the established
-- cleanup path instead of rolling the just-completed game score back.
do $migration$
declare
  v_definition text;
  v_anchor text := E'  if not v_ready then\n';
  v_guard text := E'  begin\n'
    || E'    perform private.assert_tournament_combined_draw_sources(\n'
    || E'      v_reservation.draw_id, v_reservation.registration_day_id\n'
    || E'    );\n'
    || E'  exception when sqlstate ''22023'' then\n'
    || E'    if sqlerrm not like ''JUPR_TOURNAMENT_COMBINED_SOURCE_INVALID:%'' then\n'
    || E'      raise;\n'
    || E'    end if;\n'
    || E'    v_ready := false;\n'
    || E'  end;\n\n';
begin
  select pg_catalog.pg_get_functiondef(
    'public.promote_tournament_day_court_reservation()'::regprocedure
  ) into v_definition;
  if position('private.assert_tournament_combined_draw_sources(' in v_definition) = 0 then
    if position(v_anchor in v_definition) = 0
       or position(v_anchor in substring(v_definition from position(v_anchor in v_definition) + length(v_anchor))) > 0 then
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_COMBINED_PROMOTION_MIGRATION: reservation cleanup anchor changed.';
    end if;
    execute replace(v_definition, v_anchor, v_guard || v_anchor);
  end if;
end;
$migration$;
