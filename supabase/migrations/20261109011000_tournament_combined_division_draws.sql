-- A playing draw may cover several registration divisions. Registration
-- selections, accepted partnerships and commerce remain in their source event.
alter table public.tournament_event_draws
  add column if not exists source_event_option_ids text[] not null default '{}',
  add column if not exists combined_into_draw_id uuid null references public.tournament_event_draws(id) on delete restrict;
alter table public.tournament_teams
  add column if not exists source_event_option_id text null;

comment on column public.tournament_event_draws.source_event_option_ids is
  'All registration divisions covered by this playing draw, including its primary event. Empty means only event_option_id.';
comment on column public.tournament_event_draws.combined_into_draw_id is
  'The active playing draw that superseded this unplayed source draw.';
comment on column public.tournament_teams.source_event_option_id is
  'Original registration division, retained when a team plays in a combined draw.';

create or replace function public.tournament_draw_source_event_ids(p_draw jsonb)
returns text[] language sql immutable security invoker set search_path = ''
as $function$
  select coalesce(array_agg(value order by position),'{}'::text[]) from (
    select btrim(value) value,min(position) position from (
      select p_draw->>'event_option_id' value,0::bigint position
      union all
      select value,ordinality from pg_catalog.jsonb_array_elements_text(
        case when pg_catalog.jsonb_typeof(p_draw->'source_event_option_ids')='array'
          then p_draw->'source_event_option_ids' else '[]'::jsonb end) with ordinality
    ) raw where nullif(btrim(value),'') is not null group by btrim(value)
  ) sources;
$function$;

create or replace function private.tournament_combined_draw_family(p_event jsonb)
returns text language sql immutable security invoker set search_path = ''
as $function$
  select lower(regexp_replace(btrim(coalesce(nullif(p_event->>'event_family_key',''),
      nullif(p_event->>'event_family_label',''), '')), '\s+', ' ', 'g'))
    || '|' || upper(btrim(coalesce(p_event->>'event_type','')))
    || '|' || upper(btrim(coalesce(nullif(p_event->>'gender_restriction',''),'ANY')))
    || '|' || coalesce(p_event->>'partner_required','false')
    || '|' || upper(btrim(coalesce(nullif(p_event->>'competition_format',''),'STANDARD')));
$function$;

-- Every draw create/scope change serializes with combination. This prevents a
-- new source draw from appearing between the reviewed snapshot and the move.
create or replace function public.guard_tournament_combined_draw_scope()
returns trigger language plpgsql security invoker set search_path = ''
as $function$
begin
  if tg_op='INSERT' then
    perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(
      'jupr:tournament-draw-combination:' || new.tournament_id::text, 0));
  elsif new.tournament_id is distinct from old.tournament_id
    or new.registration_day_id is distinct from old.registration_day_id
    or new.event_option_id is distinct from old.event_option_id
    or new.source_event_option_ids is distinct from old.source_event_option_ids
    or new.combined_into_draw_id is distinct from old.combined_into_draw_id
    or ((coalesce(old.hidden_from_primary_ops,false) or upper(coalesce(old.status,'')) in ('CANCELLED','CANCELED','ARCHIVED','INACTIVE','DISABLED','DELETED','VOID','VOIDED'))
      and not coalesce(new.hidden_from_primary_ops,false) and upper(coalesce(new.status,'')) not in ('CANCELLED','CANCELED','ARCHIVED','INACTIVE','DISABLED','DELETED','VOID','VOIDED')) then
    perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(
      'jupr:tournament-draw-combination:' || new.tournament_id::text, 0));
  end if;
  if tg_op = 'UPDATE' and (
      new.source_event_option_ids is distinct from old.source_event_option_ids
      or new.combined_into_draw_id is distinct from old.combined_into_draw_id
      or (cardinality(old.source_event_option_ids)>1 and (new.event_option_id is distinct from old.event_option_id
        or new.registration_day_id is distinct from old.registration_day_id or new.tournament_id is distinct from old.tournament_id))
    ) and coalesce(pg_catalog.current_setting('jupr.tournament_draw_combination', true), '') <> 'on' then
    raise exception using errcode='42501', message='JUPR_TOURNAMENT_COMBINE_REQUIRES_RPC';
  end if;
  if tg_op = 'INSERT' and (cardinality(new.source_event_option_ids) > 0 or new.combined_into_draw_id is not null)
     and coalesce(pg_catalog.current_setting('jupr.tournament_draw_combination', true), '') <> 'on' then
    raise exception using errcode='42501', message='JUPR_TOURNAMENT_COMBINE_REQUIRES_RPC';
  end if;
  if new.combined_into_draw_id is not null then
    if new.combined_into_draw_id = new.id or not coalesce(new.hidden_from_primary_ops,false)
       or upper(coalesce(new.status,'')) not in ('CANCELLED','CANCELED') then
      raise exception using errcode='23514', message='JUPR_TOURNAMENT_COMBINE_RETIRED_DRAW_INVALID';
    end if;
  elsif not coalesce(new.hidden_from_primary_ops,false)
    and upper(coalesce(new.status,'')) not in ('CANCELLED','CANCELED','ARCHIVED','INACTIVE','DISABLED','DELETED','VOID','VOIDED')
    and exists (
      select 1 from public.tournament_event_draws other
      where other.id <> new.id and other.tournament_id = new.tournament_id
        and other.registration_day_id is not distinct from new.registration_day_id
        and other.combined_into_draw_id is null and not coalesce(other.hidden_from_primary_ops,false)
        and upper(coalesce(other.status,'')) not in ('CANCELLED','CANCELED','ARCHIVED','INACTIVE','DISABLED','DELETED','VOID','VOIDED')
        and (cardinality(other.source_event_option_ids)>1 or cardinality(new.source_event_option_ids)>1)
        and public.tournament_draw_source_event_ids(to_jsonb(other)) && public.tournament_draw_source_event_ids(to_jsonb(new))
    ) then
    raise exception using errcode='23514', message='JUPR_TOURNAMENT_COMBINE_SOURCE_ALREADY_COVERED';
  end if;
  return new;
end
$function$;
drop trigger if exists trg_guard_tournament_combined_draw_scope on public.tournament_event_draws;
create trigger trg_guard_tournament_combined_draw_scope before insert or update of
  tournament_id,registration_day_id,event_option_id,source_event_option_ids,combined_into_draw_id,status,hidden_from_primary_ops
  on public.tournament_event_draws for each row execute function public.guard_tournament_combined_draw_scope();

create or replace function public.admin_combine_tournament_draw_divisions_cas(
  p_club_id text, p_tournament_id text, p_target_draw_id text,
  p_expected_draws jsonb, p_expected_teams jsonb, p_expected_events jsonb,
  p_source_event_option_ids text[], p_name text
)
returns jsonb language plpgsql security invoker set search_path = ''
as $function$
declare
  v_target public.tournament_event_draws%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_draw_ids uuid[];
  v_source_ids text[];
  v_count integer;
  v_max_number integer;
  v_team record;
  v_saved jsonb;
  v_moved_count integer;
begin
  if coalesce(cardinality(p_source_event_option_ids),0) not between 2 and 16
     or nullif(btrim(p_name),'') is null or length(btrim(p_name)) > 180
     or pg_catalog.jsonb_typeof(p_expected_draws) is distinct from 'array'
     or pg_catalog.jsonb_typeof(p_expected_teams) is distinct from 'array'
     or pg_catalog.jsonb_typeof(p_expected_events) is distinct from 'array'
     or exists (select 1 from unnest(p_source_event_option_ids) id where nullif(btrim(id),'') is null) then
    raise exception using errcode='22023', message='JUPR_TOURNAMENT_COMBINE_INVALID';
  end if;
  select array_agg(distinct id order by id) into v_source_ids from unnest(p_source_event_option_ids) id;
  if cardinality(v_source_ids) <> cardinality(p_source_event_option_ids) then
    raise exception using errcode='22023', message='JUPR_TOURNAMENT_COMBINE_INVALID';
  end if;
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(
    'jupr:tournament-draw-combination:' || p_tournament_id, 0));
  select draw.* into v_target from public.tournament_event_draws draw
    join public.tournaments tournament on tournament.id=draw.tournament_id
    where draw.id::text=p_target_draw_id and draw.tournament_id::text=p_tournament_id
      and tournament.club_id::text=p_club_id
      and upper(coalesce(tournament.status,'')) not in ('COMPLETED','ARCHIVED','CANCELLED','CANCELED') for share of tournament;
  if not found or v_target.event_option_id is null or v_target.registration_day_id is null
     or not (v_target.event_option_id=any(v_source_ids))
     or not public.tournament_draw_source_event_ids(to_jsonb(v_target)) <@ v_source_ids then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_DRAW_STALE';
  end if;
  perform day.id from public.tournament_registration_days day
    where day.id=v_target.registration_day_id and day.tournament_id::text=p_tournament_id and day.enabled is true for share;
  if not found then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_DAY_UNAVAILABLE';
  end if;
  if exists (select 1 from public.tournament_event_draws draw where draw.tournament_id=v_target.tournament_id
    and draw.registration_day_id is distinct from v_target.registration_day_id
    and draw.combined_into_draw_id is null and not coalesce(draw.hidden_from_primary_ops,false)
    and upper(coalesce(draw.status,'')) not in ('CANCELLED','CANCELED','ARCHIVED','INACTIVE','DISABLED','DELETED','VOID','VOIDED')
    and public.tournament_draw_source_event_ids(to_jsonb(draw)) && v_source_ids) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_EVENTS_INCOMPATIBLE';
  end if;
  perform event.id from public.tournament_event_options event
    where event.tournament_id::text=p_tournament_id and event.id::text=any(v_source_ids)
    order by event.id for share;
  if (select count(*) from public.tournament_event_options event where event.tournament_id::text=p_tournament_id and event.id::text=any(v_source_ids)) <> cardinality(v_source_ids)
    or jsonb_array_length(p_expected_events)<>cardinality(v_source_ids)
    or (select count(distinct e.id) from jsonb_to_recordset(p_expected_events) e(id text))<>cardinality(v_source_ids)
    or exists (select 1 from jsonb_to_recordset(p_expected_events) e(id text,updated_at timestamptz)
      left join public.tournament_event_options actual on actual.id::text=e.id and actual.tournament_id::text=p_tournament_id
      where actual.id is null or not(e.id=any(v_source_ids)) or e.updated_at is null or actual.updated_at is distinct from e.updated_at) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_EVENTS_STALE';
  end if;
  select * into v_event from public.tournament_event_options where id::text=v_target.event_option_id and tournament_id::text=v_target.tournament_id::text;
  if exists (select 1 from public.tournament_event_options e
    where e.tournament_id::text=v_target.tournament_id::text and e.id::text=any(v_source_ids)
    and (e.enabled is distinct from true or upper(coalesce(e.status,'')) in ('CANCELLED','CANCELED','ARCHIVED','DISABLED','INACTIVE','DELETED','VOID','VOIDED')
      or upper(btrim(coalesce(nullif(e.competition_format,''),'STANDARD')))<>'STANDARD'
      or upper(coalesce(to_jsonb(e)->>'eligibility_mode',''))='COMBINED_RATING_CAP'
      or upper(coalesce(to_jsonb(e)->>'skill_mode',''))='COMBINED_RATING_CAP'
      or private.tournament_combined_draw_family(to_jsonb(e))<>private.tournament_combined_draw_family(to_jsonb(v_event))
      or e.registration_day_id is distinct from v_target.registration_day_id
      or case when coalesce(to_jsonb(e)->'scheduled_day_ids','null'::jsonb) in ('null'::jsonb,'[]'::jsonb) then false
        when jsonb_typeof(to_jsonb(e)->'scheduled_day_ids')<>'array' then true
        else not ((to_jsonb(e)->'scheduled_day_ids') ? v_target.registration_day_id) end)) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_EVENTS_INCOMPATIBLE';
  end if;
  select array_agg(draw.id order by draw.id) into v_draw_ids from public.tournament_event_draws draw
    where draw.tournament_id=v_target.tournament_id and draw.registration_day_id=v_target.registration_day_id
      and draw.combined_into_draw_id is null and not coalesce(draw.hidden_from_primary_ops,false)
      and upper(coalesce(draw.status,'')) not in ('CANCELLED','CANCELED','ARCHIVED','INACTIVE','DISABLED','DELETED','VOID','VOIDED')
      and public.tournament_draw_source_event_ids(to_jsonb(draw)) && v_source_ids;
  if not coalesce(v_target.id=any(v_draw_ids),false)
     or jsonb_array_length(p_expected_draws)<>cardinality(v_draw_ids)
     or (select count(distinct d.id) from jsonb_to_recordset(p_expected_draws) d(id text))<>cardinality(v_draw_ids) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_DRAW_STALE';
  end if;
  -- Preserve child -> draw lock order used by all existing roster writers.
  perform team.id from public.tournament_teams team where team.draw_id=any(v_draw_ids) order by team.id for no key update;
  perform game.id from public.tournament_games game where game.draw_id=any(v_draw_ids) order by game.id for update;
  perform draw.id from public.tournament_event_draws draw where draw.id=any(v_draw_ids) order by draw.id for update;
  if exists (select 1 from jsonb_to_recordset(p_expected_draws) expected(id text,updated_at timestamptz)
      left join public.tournament_event_draws actual on actual.id::text=expected.id
      where actual.id is null or not(actual.id=any(v_draw_ids)) or expected.updated_at is null or actual.updated_at is distinct from expected.updated_at)
     or exists (select 1 from public.tournament_event_draws d where d.id=any(v_draw_ids)
       and (upper(coalesce(d.status,'')) not in ('DRAFT','READY') or d.pool_config is not null
         or upper(coalesce(d.draw_kind,'STANDARD'))<>'STANDARD' or coalesce(d.hidden_from_primary_ops,false)
         or (d.id<>v_target.id and cardinality(d.source_event_option_ids)>1))) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_DRAW_STALE';
  end if;
  select count(*) into v_count from public.tournament_teams where draw_id=any(v_draw_ids);
  if v_count>64 or jsonb_array_length(p_expected_teams)<>v_count
     or (select count(distinct t.id) from jsonb_to_recordset(p_expected_teams) t(id text))<>v_count
     or exists (select 1 from jsonb_to_recordset(p_expected_teams) expected(id text,updated_at timestamptz)
       left join public.tournament_teams actual on actual.id::text=expected.id
       where actual.id is null or not(actual.draw_id=any(v_draw_ids)) or expected.updated_at is null or actual.updated_at is distinct from expected.updated_at)
     or exists (select 1 from public.tournament_teams t join public.tournament_event_draws d on d.id=t.draw_id
       where t.draw_id=any(v_draw_ids) and (t.tournament_id<>v_target.tournament_id
         or t.registration_day_id is distinct from d.registration_day_id or t.event_option_id is distinct from d.event_option_id
         or not(coalesce(t.source_event_option_id,t.event_option_id)=any(v_source_ids)))) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_TEAMS_STALE';
  end if;
  if exists (select player_id from public.tournament_teams t cross join lateral
      (values(t.player1_id),(t.player2_id)) players(player_id)
      where t.draw_id=any(v_draw_ids) and player_id is not null group by player_id having count(*)>1) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_DUPLICATE_PLAYER';
  end if;
  if exists (select 1 from public.tournament_games where draw_id=any(v_draw_ids))
     or exists (select 1 from public.tournament_podium where draw_id=any(v_draw_ids))
     or exists (select 1 from public.tournament_day_live_draws where draw_id=any(v_draw_ids))
     or exists (select 1 from public.tournament_day_live_queue where draw_id=any(v_draw_ids)) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_DRAW_IN_USE';
  end if;
  select count(*) into v_moved_count from public.tournament_teams where draw_id=any(v_draw_ids) and draw_id<>v_target.id;
  perform pg_catalog.set_config('jupr.tournament_draw_combination','on',true);
  -- Source drafts remain auditable, but no longer count as playable draws.
  update public.tournament_event_draws set combined_into_draw_id=v_target.id,
    status='CANCELLED',hidden_from_primary_ops=true,updated_at=clock_timestamp()
    where id=any(v_draw_ids) and id<>v_target.id;
  update public.tournament_event_draws set source_event_option_ids=v_source_ids,
    name=btrim(p_name),updated_at=clock_timestamp() where id=v_target.id;
  select coalesce(max(team_number),0) into v_max_number from public.tournament_teams where draw_id=v_target.id;
  update public.tournament_teams set source_event_option_id=coalesce(source_event_option_id,event_option_id),updated_at=clock_timestamp()
    where draw_id=v_target.id;
  for v_team in select * from public.tournament_teams where draw_id=any(v_draw_ids) and draw_id<>v_target.id order by draw_id,team_number,id loop
    v_max_number:=v_max_number+1;
    update public.tournament_teams set source_event_option_id=coalesce(source_event_option_id,event_option_id),
      draw_id=v_target.id,event_option_id=v_target.event_option_id,registration_day_id=v_target.registration_day_id,
      team_number=v_max_number,seed=null,updated_at=clock_timestamp() where id=v_team.id;
  end loop;
  update public.tournament_event_draws set updated_at=clock_timestamp() where id=v_target.id returning * into v_target;
  select coalesce(jsonb_agg(to_jsonb(t) order by t.team_number),'[]'::jsonb) into v_saved from public.tournament_teams t where t.draw_id=v_target.id;
  return jsonb_build_object('ok',true,'draw',to_jsonb(v_target),'teams',v_saved,
    'combined_source_draw_ids',to_jsonb(array_remove(v_draw_ids,v_target.id)),'moved_team_count',v_moved_count,'source_event_option_ids',to_jsonb(v_source_ids));
end
$function$;

revoke all on function public.tournament_draw_source_event_ids(jsonb) from public,anon,authenticated;
revoke all on function private.tournament_combined_draw_family(jsonb) from public,anon,authenticated;
revoke all on function public.guard_tournament_combined_draw_scope() from public,anon,authenticated;
revoke all on function public.admin_combine_tournament_draw_divisions_cas(text,text,text,jsonb,jsonb,jsonb,text[],text) from public,anon,authenticated;
grant execute on function public.tournament_draw_source_event_ids(jsonb) to service_role;
grant execute on function private.tournament_combined_draw_family(jsonb) to service_role;
grant execute on function public.guard_tournament_combined_draw_scope() to service_role;
grant execute on function public.admin_combine_tournament_draw_divisions_cas(text,text,text,jsonb,jsonb,jsonb,text[],text) to service_role;

create or replace function public.admin_write_tournament_draw_teams_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_replace boolean,
  p_teams jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_saved jsonb;
  v_selection_ids text[];
begin
  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if jsonb_typeof(coalesce(p_teams, '[]'::jsonb)) <> 'array' or jsonb_array_length(coalesce(p_teams, '[]'::jsonb)) = 0 then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAMS_REQUIRED';
  end if;
  if exists (
    select 1
      from jsonb_to_recordset(p_teams) as x(player1_id integer, player2_id integer)
      left join public.players p1 on p1.id = x.player1_id and p1.club_id = p_club_id
      left join public.players p2 on p2.id = x.player2_id and p2.club_id = p_club_id
     where p1.id is null or (x.player2_id is not null and p2.id is null)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_PLAYER_OUTSIDE_CLUB';
  end if;

  -- Import and public registration edits share the same parent/selection lock
  -- order. Include the other partner before reading the current confirmed link.
  select array_agg(distinct selection.id order by selection.id) into v_selection_ids
    from public.tournament_registration_selections selection
    join public.tournament_registrations registration on registration.id=selection.registration_id
    join jsonb_to_recordset(p_teams) x(source_selection_id text,source_event_option_id text,player1_id integer,player2_id integer)
      on nullif(x.source_selection_id,'') is not null
      and (selection.id=x.source_selection_id or (
        registration.player_id in (x.player1_id,x.player2_id)
        and selection.event_option_id=coalesce(nullif(x.source_event_option_id,''),v_draw.event_option_id)))
    where selection.tournament_id=v_draw.tournament_id::text;
  if cardinality(v_selection_ids)>0 then
    perform private.lock_tournament_registration_selection_scope(v_selection_ids);
  end if;

  perform team.id
    from public.tournament_teams team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
   order by team.id
   for no key update;
  perform game.id
    from public.tournament_games game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
   order by game.id
   for update;
  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id
     and d.updated_at = p_expected_draw_updated_at
     and d.combined_into_draw_id is null
     and not coalesce(d.hidden_from_primary_ops,false)
     and upper(coalesce(d.status,'')) in ('DRAFT','READY')
     and upper(coalesce(t.status,'')) not in ('COMPLETED','ARCHIVED','CANCELLED','CANCELED')
   for update of d;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if exists (
    select 1 from public.tournament_games
     where tournament_id::text = p_tournament_id and draw_id::text = p_draw_id
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_GAMES';
  end if;

  if v_draw.pool_config is not null or exists (
    select 1 from public.tournament_day_live_draws where draw_id=v_draw.id
  ) or exists (select 1 from public.tournament_day_live_queue where draw_id=v_draw.id) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_DRAW_IN_USE';
  end if;
  if exists (select 1 from jsonb_to_recordset(p_teams) x(
      registration_day_id text,event_option_id text,source_event_option_id text,source_selection_id text,player1_id integer,player2_id integer)
    where (nullif(x.registration_day_id,'') is not null and x.registration_day_id is distinct from v_draw.registration_day_id)
      or (nullif(x.event_option_id,'') is not null and x.event_option_id is distinct from v_draw.event_option_id)
      or not(coalesce(nullif(x.source_event_option_id,''),nullif(x.event_option_id,''),v_draw.event_option_id)=any(public.tournament_draw_source_event_ids(to_jsonb(v_draw))))
      or (nullif(x.source_selection_id,'') is not null and not exists (
        select 1 from public.tournament_registration_selections selection
        join public.tournament_registrations registration on registration.id=selection.registration_id
        where selection.id=x.source_selection_id and selection.tournament_id=v_draw.tournament_id::text
          and selection.registration_day_id=v_draw.registration_day_id
          and selection.event_option_id=coalesce(nullif(x.source_event_option_id,''),v_draw.event_option_id)
          and registration.player_id in (x.player1_id,x.player2_id)
      ))
      or (cardinality(v_draw.source_event_option_ids)>1 and nullif(x.source_selection_id,'') is not null and x.player2_id is not null
        and not exists (select 1 from public.tournament_registration_team_links link
          where link.tournament_id::text=v_draw.tournament_id::text
            and link.event_option_id=coalesce(nullif(x.source_event_option_id,''),v_draw.event_option_id)
            and link.status in ('CONFIRMED','ADMIN_CONFIRMED')
            and x.source_selection_id in (link.selection1_id,link.selection2_id)
            and ((link.player1_id=x.player1_id and link.player2_id=x.player2_id)
              or (link.player1_id=x.player2_id and link.player2_id=x.player1_id))
        ))) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_TEAM_SOURCE_SCOPE_INVALID';
  end if;
  if exists (select player_id from jsonb_to_recordset(p_teams) x(player1_id integer,player2_id integer)
    cross join lateral (values(x.player1_id),(x.player2_id)) p(player_id)
    where player_id is not null group by player_id having count(*)>1)
    or (not p_replace and exists (select 1 from jsonb_to_recordset(p_teams) x(player1_id integer,player2_id integer)
      join public.tournament_teams t on t.draw_id=v_draw.id
        and (t.player1_id in (x.player1_id,x.player2_id) or t.player2_id in (x.player1_id,x.player2_id)))) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_COMBINE_DUPLICATE_PLAYER';
  end if;
  if p_replace then
    delete from public.tournament_teams
     where tournament_id::text = p_tournament_id and draw_id::text = p_draw_id;
  end if;

  with inserted as (
    insert into public.tournament_teams (
      id, tournament_id, draw_id, registration_day_id, event_option_id,
      team_number, player1_id, player2_id, seed, source, notes, source_selection_id, source_event_option_id, created_at, updated_at
    )
    select
      coalesce(nullif(x.id, '')::uuid, gen_random_uuid()),
      v_draw.tournament_id,
      v_draw.id,
      coalesce(nullif(x.registration_day_id, ''), v_draw.registration_day_id),
      coalesce(nullif(x.event_option_id, ''), v_draw.event_option_id),
      x.team_number,
      x.player1_id,
      x.player2_id,
      x.seed,
      coalesce(nullif(x.source, ''), 'MANUAL'),
      nullif(x.notes, ''),
      nullif(x.source_selection_id, ''),
      coalesce(nullif(x.source_event_option_id,''),v_draw.event_option_id),
      coalesce(x.created_at, timezone('utc', now())),
      timezone('utc', now())
    from jsonb_to_recordset(p_teams) as x(
      id text, registration_day_id text, event_option_id text, team_number integer,
      player1_id integer, player2_id integer, seed integer, source text, notes text,
      source_selection_id text, source_event_option_id text,
      created_at timestamptz
    )
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted) order by inserted.team_number), '[]'::jsonb)
    into v_saved from inserted;

  update public.tournament_event_draws
     set updated_at = clock_timestamp()
   where id = v_draw.id;
  return jsonb_build_object('ok', true, 'teams', v_saved);
end;
$$;

revoke all on function public.admin_write_tournament_draw_teams_cas(text,text,text,timestamptz,boolean,jsonb) from public,anon,authenticated;
grant execute on function public.admin_write_tournament_draw_teams_cas(text,text,text,timestamptz,boolean,jsonb) to service_role;

-- Preserve the currently deployed registration transaction bodies while fixing
-- their event lookup to follow immutable team provenance. Fail closed if the
-- expected guard is absent, rather than replacing an unknown function body.
do $migration$
declare
  routine_oid regprocedure;
  definition text;
  patched text;
  signature text;
begin
  signature := 'private.public_edit_before_division_transfer(text,text,timestamp with time zone,jsonb,jsonb,jsonb)';
  routine_oid := to_regprocedure(signature);
  if routine_oid is null then raise exception 'Required registration edit guard is missing: %',signature; end if;
  definition := pg_get_functiondef(routine_oid);
  if position('team.source_event_option_id::text' in definition)=0 then
    patched := replace(definition,
      E'and coalesce(\n        draw.event_option_id::text,\n        team.event_option_id::text\n      ) = selection.event_option_id',
      E'and coalesce(\n        team.source_event_option_id::text,\n        draw.event_option_id::text,\n        team.event_option_id::text\n      ) = selection.event_option_id');
    if patched=definition then raise exception 'Registration edit source guard changed; review required.'; end if;
    execute patched;
  end if;
  signature := 'private.transfer_tournament_registration_team_division(text,text,text,timestamp with time zone,jsonb)';
  routine_oid := to_regprocedure(signature);
  if routine_oid is null then raise exception 'Required division transfer guard is missing: %',signature; end if;
  definition := pg_get_functiondef(routine_oid);
  if position('t.source_event_option_id::text' in definition)=0 then
    patched := replace(definition,'coalesce(d.event_option_id::text,t.event_option_id::text)',
      'coalesce(t.source_event_option_id::text,d.event_option_id::text,t.event_option_id::text)');
    patched := replace(patched,'d.event_option_id::text in (v_source.id,v_target.id)',
      'public.tournament_draw_source_event_ids(to_jsonb(d)) && array[v_source.id,v_target.id]');
    if patched=definition then raise exception 'Division transfer source guard changed; review required.'; end if;
    execute patched;
  end if;
  signature := 'public.server_cancel_tournament_registrations(text,jsonb,text)';
  routine_oid := to_regprocedure(signature);
  if routine_oid is null then raise exception 'Required registration cancellation guard is missing: %',signature; end if;
  definition := pg_get_functiondef(routine_oid);
  if position('t.source_event_option_id::text' in definition)=0 then
    patched := replace(definition,'coalesce(d.event_option_id::text,t.event_option_id::text)',
      'coalesce(t.source_event_option_id::text,d.event_option_id::text,t.event_option_id::text)');
    if patched=definition then raise exception 'Registration cancellation source guard changed; review required.'; end if;
    execute patched;
  end if;
end
$migration$;

notify pgrst, 'reload schema';
