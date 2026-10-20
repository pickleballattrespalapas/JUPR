-- Deliberately locale-independent: ASCII/Latin-1 case and explicit whitespace
-- match the Python name-review function exactly; other Unicode letters stay intact.
create or replace function public.round_robin_hub_name_key(p_name text)
returns text language sql immutable security invoker set search_path=public,pg_temp as $$
  select btrim(regexp_replace(translate(translate(coalesce(p_name,''),
    'ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ','abcdefghijklmnopqrstuvwxyzàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ'),
    chr(9) || chr(10) || chr(11) || chr(12) || chr(13) || chr(28) || chr(29) || chr(30) || chr(31) || chr(133) || chr(160) || chr(5760) || chr(8192) || chr(8193) || chr(8194) || chr(8195) || chr(8196) || chr(8197) || chr(8198) || chr(8199) || chr(8200) || chr(8201) || chr(8202) || chr(8232) || chr(8233) || chr(8239) || chr(8287) || chr(12288), repeat(' ',28)), ' +', ' ', 'g'));
$$;
revoke all on function public.round_robin_hub_name_key(text) from public, anon, authenticated;
grant execute on function public.round_robin_hub_name_key(text) to service_role;

-- Version 2 identities are scoped to a session participant, unless a club profile
-- or an explicit same-person decision joins them. Names only request review.
-- Old API workers may continue to commit version 1 states during deployment.
create or replace function public.round_robin_hub_identity_snapshot(p_state jsonb, p_events jsonb)
returns jsonb language plpgsql immutable security invoker set search_path = public, pg_temp as $$
declare
  nodes jsonb;
  resolved jsonb;
  decisions jsonb := coalesce(p_state->'identity_decisions', '[]'::jsonb);
  mapping jsonb := coalesce(p_state->'participant_identities', '{}'::jsonb);
  expected_mapping jsonb;
  pending jsonb;
begin
  if p_state->>'identity_version' is distinct from '2'
     or jsonb_typeof(decisions) is distinct from 'array'
     or jsonb_typeof(mapping) is distinct from 'object' then
    raise exception using errcode='22023', message='Invalid hub participant identity schema.';
  end if;
  select coalesce(jsonb_agg(jsonb_build_object(
      'ref', 'guest:' || e.key || ':' || (p->>'id'),
      'session_key', e.key, 'participant_id', p->>'id',
      'name_key', public.round_robin_hub_name_key(p->>'name'),
      'profile_id', case when nullif(p->>'player_id','') is not null then (p->>'player_id')::bigint::text end
    )), '[]'::jsonb) into nodes
    from jsonb_each(p_events) e,
      lateral jsonb_array_elements(coalesce(e.value->'participants','[]'::jsonb)) p;
  if exists (select 1 from jsonb_array_elements(nodes) n
      where nullif(n->>'participant_id','') is null or nullif(n->>'name_key','') is null)
     or (select count(*) from jsonb_array_elements(nodes)) <>
        (select count(distinct n->>'ref') from jsonb_array_elements(nodes) n) then
    raise exception using errcode='22023', message='Every hub participant needs a unique session-scoped ID and name.';
  end if;
  if exists (select 1 from jsonb_array_elements(decisions) d
      where coalesce(d->>'resolution','') not in ('same','different')
        or not exists (select 1 from jsonb_array_elements(nodes) n
          where n->>'session_key'=d#>>'{left,session_key}' and n->>'participant_id'=d#>>'{left,participant_id}')
        or not exists (select 1 from jsonb_array_elements(nodes) n
          where n->>'session_key'=d#>>'{right,session_key}' and n->>'participant_id'=d#>>'{right,participant_id}')) then
    raise exception using errcode='22023', message='An identity decision references an unavailable participant.';
  end if;
  with recursive node as (
    select n->>'ref' ref, n->>'profile_id' profile from jsonb_array_elements(nodes) n
  ), edge as (
    select 'guest:' || (d#>>'{left,session_key}') || ':' || (d#>>'{left,participant_id}') a,
           'guest:' || (d#>>'{right,session_key}') || ':' || (d#>>'{right,participant_id}') b
      from jsonb_array_elements(decisions) d where d->>'resolution'='same'
    union
    select a.ref,b.ref from node a join node b on a.profile=b.profile where a.profile is not null
  ), linked as (
    select a,b from edge union select b,a from edge
  ), reach(a,b) as (
    select ref,ref from node
    union
    select r.a,e.b from reach r join linked e on e.a=r.b
  ), component as (
    select r.a ref, min(r.b collate "C") representative,
      min(n.profile) profile, count(distinct n.profile) profiles
      from reach r join node n on n.ref=r.b group by r.a
  ) select coalesce(jsonb_agg(n || jsonb_build_object(
      'identity', case when c.profile is not null then 'player:' || c.profile else c.representative end,
      'canonical_profile_id', c.profile, 'profile_count', c.profiles
    )), '[]'::jsonb) into resolved
    from jsonb_array_elements(nodes) n join component c on c.ref=n->>'ref';
  if exists(select 1 from jsonb_array_elements(resolved) r where (r->>'profile_count')::integer>1)
     or exists(select 1 from jsonb_array_elements(decisions) d
       join jsonb_array_elements(resolved) l on l->>'session_key'=d#>>'{left,session_key}' and l->>'participant_id'=d#>>'{left,participant_id}'
       join jsonb_array_elements(resolved) r on r->>'session_key'=d#>>'{right,session_key}' and r->>'participant_id'=d#>>'{right,participant_id}'
       where d->>'resolution'='different' and l->>'identity'=r->>'identity') then
    raise exception using errcode='22023', message='Identity decisions conflict with an authoritative club profile or each other.';
  end if;
  select coalesce(jsonb_object_agg(session_key, participants),'{}'::jsonb) into expected_mapping from (
    select r->>'session_key' session_key, jsonb_object_agg(r->>'participant_id',r->'identity') participants
      from jsonb_array_elements(resolved) r group by r->>'session_key'
  ) m;
  if mapping is distinct from expected_mapping then
    raise exception using errcode='22023', message='Hub participant identities do not match the authoritative profiles and decisions.';
  end if;
  with different as (
    select l->>'identity' a,r->>'identity' b from jsonb_array_elements(decisions) d
      join jsonb_array_elements(resolved) l on l->>'session_key'=d#>>'{left,session_key}' and l->>'participant_id'=d#>>'{left,participant_id}'
      join jsonb_array_elements(resolved) r on r->>'session_key'=d#>>'{right,session_key}' and r->>'participant_id'=d#>>'{right,participant_id}'
      where d->>'resolution'='different'
  ) select coalesce(jsonb_agg(distinct l->'ref'), '[]'::jsonb) into pending
    from jsonb_array_elements(resolved) l join jsonb_array_elements(resolved) r
      on l->>'name_key'=r->>'name_key' and l->>'identity'<>r->>'identity'
    where (l->>'profile_id' is null or r->>'profile_id' is null)
      and not exists (select 1 from different d
        where (d.a=l->>'identity' and d.b=r->>'identity') or (d.b=l->>'identity' and d.a=r->>'identity'));
  -- An unresolved occurrence blocks every occurrence of its canonical person.
  select coalesce(jsonb_agg(distinct n->'ref'),'[]'::jsonb) into pending
    from jsonb_array_elements(resolved) n
    where n->>'identity' in (select r->>'identity' from jsonb_array_elements(resolved) r
      where pending ? (r->>'ref'));
  return jsonb_build_object('participants',resolved,'pending',pending);
end;
$$;
revoke all on function public.round_robin_hub_identity_snapshot(jsonb,jsonb) from public, anon, authenticated;
grant execute on function public.round_robin_hub_identity_snapshot(jsonb,jsonb) to service_role;

create or replace function public.commit_round_robin_hub(
  p_club_id text,
  p_hub_id uuid,
  p_expected_version bigint,
  p_operation_key text,
  p_fingerprint text,
  p_title text,
  p_status text,
  p_state jsonb,
  p_sessions jsonb,
  p_actor_email text
) returns jsonb
language plpgsql security invoker set search_path = public, pg_temp as $$
declare
  hub public.round_robin_hubs%rowtype;
  prior public.round_robin_hub_operations%rowtype;
  live public.live_sessions%rowtype;
  item jsonb;
  patch jsonb;
  court jsonb;
  assignment jsonb;
  round_row jsonb;
  match_row jsonb;
  participant jsonb;
  member text;
  participant_id text;
  identity_key text;
  player_name text;
  player_profile text;
  group_keys text[];
  read_keys text[];
  old_keys text[];
  group_count integer;
  court_count integer;
  now_at timestamptz := clock_timestamp();
  context_before text := coalesce(current_setting('jupr.round_robin_hub_transaction', true), '');
  result_sessions jsonb;
  normalized_label text;
  identity_snapshot jsonb;
  identity_participant jsonb;
  identity_events jsonb;
  assignment_slot text;
  previous_hub_state jsonb;
  previous_busy_refs jsonb;
begin
  if nullif(btrim(p_club_id), '') is null or p_hub_id is null
     or p_expected_version is null or p_expected_version < 0
     or length(coalesce(p_operation_key, '')) not between 8 and 160
     or length(coalesce(p_fingerprint, '')) not between 16 and 128
     or length(btrim(coalesce(p_title, ''))) not between 1 and 160
     or p_status not in ('active', 'completed') or p_status is null
     or jsonb_typeof(p_state) is distinct from 'object'
     or jsonb_typeof(p_state->'courts') is distinct from 'array'
     or jsonb_typeof(p_state->'groups') is distinct from 'array'
     or jsonb_typeof(p_sessions) is distinct from 'array' then
    raise exception using errcode = '22023', message = 'Invalid Round Robin Hub command.';
  end if;
  -- Serializes attachment, court leases and player leases across this club's hubs.
  perform pg_advisory_xact_lock(hashtextextended('round-robin-hub:' || p_club_id, 0));
  select * into hub from public.round_robin_hubs where club_id=p_club_id and id=p_hub_id for update;
  previous_hub_state := coalesce(hub.state,'{}'::jsonb);
  select * into prior from public.round_robin_hub_operations
    where club_id=p_club_id and hub_id=p_hub_id and operation_key=p_operation_key;
  if found then
    if prior.request_fingerprint <> p_fingerprint then
      raise exception using errcode = '40001', message = 'This operation key was already used for a different hub command.';
    end if;
    select coalesce(jsonb_agg(to_jsonb(s) order by s.session_key), '[]'::jsonb) into result_sessions
      from public.live_sessions s where s.club_id=p_club_id
      and s.session_key in (select jsonb_array_elements_text(prior.session_keys));
    return jsonb_build_object('hub',to_jsonb(hub),'sessions',result_sessions,'replayed',true);
  end if;
  if (hub.state->>'identity_version'='2' and p_state->>'identity_version' is distinct from '2')
     or (p_state ? 'identity_version' and p_state->>'identity_version' is distinct from '2') then
    raise exception using errcode='22023', message='A hub participant identity schema cannot be downgraded.';
  end if;
  if hub.id is null then
    if p_expected_version <> 0 then
      raise exception using errcode = '40001', message = 'The Round Robin Hub no longer exists. Reload before saving.';
    end if;
    insert into public.round_robin_hubs (id,club_id,title,status,state,created_by_email)
      values (p_hub_id,p_club_id,btrim(p_title),p_status,p_state,nullif(btrim(p_actor_email),'')) returning * into hub;
  elsif hub.version <> p_expected_version then
    raise exception using errcode = '40001', message = 'The Round Robin Hub changed. Reload before saving.';
  end if;

  select coalesce(array_agg(value->>'session_key' order by value->>'session_key'),'{}')
    into group_keys from jsonb_array_elements(p_state->'groups');
  select coalesce(array_agg(value->>'session_key' order by value->>'session_key'),'{}')
    into read_keys from jsonb_array_elements(p_sessions);
  select coalesce(array_agg(session_key order by session_key),'{}') into old_keys
    from public.round_robin_hub_sessions where club_id=p_club_id and hub_id=p_hub_id;
  group_count := cardinality(group_keys);
  if group_count > 100 or cardinality(read_keys) > 200
     or group_count <> (select count(distinct x) from unnest(group_keys) x)
     or cardinality(read_keys) <> (select count(distinct x) from unnest(read_keys) x)
     or exists (select 1 from unnest(group_keys || read_keys) x where nullif(btrim(x),'') is null)
     or not group_keys <@ read_keys or not old_keys <@ read_keys then
    raise exception using errcode = '22023', message = 'Every hub session must be supplied once for concurrency verification.';
  end if;
  if exists (select 1 from jsonb_array_elements(coalesce(previous_hub_state->'identity_decisions','[]'::jsonb)) d
    where d->>'resolution'='same' and d#>>'{left,session_key}'<>d#>>'{right,session_key}'
      and ((d#>>'{left,session_key}'=any(group_keys)) <> (d#>>'{right,session_key}'=any(group_keys)))) then
    raise exception using errcode='22023', message='Resolve linked player identities before removing this group from the hub.';
  end if;
  if exists (select 1 from public.round_robin_hub_sessions
    where club_id=p_club_id and session_key=any(group_keys) and hub_id<>p_hub_id) then
    raise exception using errcode = '40001', message = 'A selected round robin already belongs to another hub.';
  end if;

  -- Lock the complete read set in a stable order before applying any session changes.
  for member in select unnest(read_keys) order by 1 loop
    select * into live from public.live_sessions where club_id=p_club_id and session_key=member for update;
    select value into item from jsonb_array_elements(p_sessions) where value->>'session_key'=member;
    if live.id is null
       or nullif(item->>'expected_version','') is null
       or live.updated_at is distinct from (item->>'expected_version')::timestamptz
       or (item ? 'expected_numeric_version' and live.version is distinct from (item->>'expected_numeric_version')::bigint) then
      raise exception using errcode = '40001', message = 'A round robin changed. Reload the hub before saving.';
    end if;
    if nullif(live.pending_operation_key,'') is not null then
      raise exception using errcode = '55000', message = 'A round robin has an unresolved operation. Resolve it before changing the hub.';
    end if;
    patch := coalesce(item->'patch','{}'::jsonb);
    if jsonb_typeof(patch) is distinct from 'object' or exists (
      select 1 from jsonb_object_keys(patch) key where key not in
      ('state','status','title','version','updated_at','last_seen_at','expires_at','completed_at','last_operation_key','last_request_fingerprint')
    ) then
      raise exception using errcode = '22023', message = 'Unsupported session field in hub command.';
    end if;
    if patch ? 'version' and (patch->>'version')::bigint <> live.version+1 then
      raise exception using errcode = '40001', message = 'The supplied session version is stale.';
    end if;
    if member=any(group_keys) then
      if coalesce(patch->'state',live.state)->>'round_robin_hub_id' is distinct from p_hub_id::text
        or coalesce(patch->'state',live.state) #>> '{page_state,event,generatorKind}' is distinct from 'round_robin'
        or coalesce(coalesce(patch->'state',live.state)->>'mode','') not in ('admin_play_generator','public_play_generator') then
        raise exception using errcode = '22023', message = 'Only linked round-robin generator sessions can join this hub.';
      end if;
    elsif member=any(old_keys) then
      if nullif(coalesce(patch->'state',live.state)->>'round_robin_hub_id','') is not null then
        raise exception using errcode = '22023', message = 'A detached session must release its hub association.';
      end if;
    else
      raise exception using errcode = '22023', message = 'The command contains a session outside this hub.';
    end if;
  end loop;

  court_count := jsonb_array_length(p_state->'courts');
  if court_count > 40
     or court_count <> (select count(distinct value->>'id') from jsonb_array_elements(p_state->'courts')) then
    raise exception using errcode = '22023', message = 'Each hub court needs a unique ID.';
  end if;
  for court in select value from jsonb_array_elements(p_state->'courts') loop
    if length(btrim(coalesce(court->>'id',''))) not between 1 and 160
       or length(btrim(coalesce(court->>'label',''))) not between 1 and 80
       or (nullif(court->>'dedicated_session_key','') is not null and not (court->>'dedicated_session_key'=any(group_keys))) then
      raise exception using errcode = '22023', message = 'Each court needs a name and a valid dedicated group.';
    end if;
    for assignment in select value from jsonb_array_elements(jsonb_build_array(court->'current',court->'next')) where value <> 'null'::jsonb loop
      if not (assignment->>'session_key'=any(group_keys)) or nullif(assignment->>'match_id','') is null
         or (nullif(court->>'dedicated_session_key','') is not null and court->>'dedicated_session_key'<>assignment->>'session_key')
         or (nullif(court->>'dedicated_session_key','') is null and exists (
           select 1 from jsonb_array_elements(p_state->'courts') c where c->>'dedicated_session_key'=assignment->>'session_key')) then
        raise exception using errcode = '22023', message = 'A match cannot use a court outside its group allocation.';
      end if;
    end loop;
    if p_status='completed' and (court->'current' <> 'null'::jsonb or court->'next' <> 'null'::jsonb) then
      raise exception using errcode = '22023', message = 'Release every court and reservation before ending the hub.';
    end if;
  end loop;
  if exists (
    select 1 from jsonb_array_elements(p_state->'courts') c,
      lateral jsonb_array_elements(jsonb_build_array(c->'current',c->'next')) a
    where a <> 'null'::jsonb group by a->>'session_key',a->>'match_id' having count(*)>1
  ) then
    raise exception using errcode = '22023', message = 'A match cannot occupy or reserve two courts.';
  end if;

  select coalesce(jsonb_agg(distinct to_jsonb('guest:' || (a->>'session_key') || ':' || pid)), '[]'::jsonb)
    into previous_busy_refs
    from jsonb_array_elements(coalesce(previous_hub_state->'courts','[]'::jsonb)) c,
      lateral jsonb_array_elements(jsonb_build_array(c->'current',c->'next')) a
      join public.live_sessions s on s.club_id=p_club_id and s.session_key=a->>'session_key',
      lateral jsonb_array_elements(s.state#>'{page_state,event,rounds}') r,
      lateral (
        select m from jsonb_array_elements(coalesce(r->'matches','[]'::jsonb)) m
        union all select m from jsonb_array_elements(coalesce(r->'courts','[]'::jsonb)) rc,
          lateral jsonb_array_elements(coalesce(rc->'matches','[]'::jsonb)) m
      ) match,
      lateral jsonb_array_elements_text(coalesce(match.m->'sideA',match.m->'teamA','[]'::jsonb) || coalesce(match.m->'sideB',match.m->'teamB','[]'::jsonb)) pid
    where r->>'number'=a->>'round_number' and match.m->>'id'=a->>'match_id';
  perform set_config('jupr.round_robin_hub_transaction',p_hub_id::text,true);
  for item in select value from jsonb_array_elements(p_sessions) order by value->>'session_key' loop
    patch := coalesce(item->'patch','{}'::jsonb);
    if patch='{}'::jsonb then continue; end if;
    update public.live_sessions s set
      state=case when patch ? 'state' then patch->'state' else s.state end,
      status=case when patch ? 'status' then patch->>'status' else s.status end,
      title=case when patch ? 'title' then patch->>'title' else s.title end,
      version=s.version+1,
      updated_at=now_at,
      last_seen_at=case when patch ? 'last_seen_at' then (patch->>'last_seen_at')::timestamptz else now_at end,
      expires_at=case when patch ? 'expires_at' then (patch->>'expires_at')::timestamptz else s.expires_at end,
      completed_at=case when patch ? 'completed_at' then (patch->>'completed_at')::timestamptz else s.completed_at end,
      last_operation_key=case when patch ? 'last_operation_key' then patch->>'last_operation_key' else s.last_operation_key end,
      last_request_fingerprint=case when patch ? 'last_request_fingerprint' then patch->>'last_request_fingerprint' else s.last_request_fingerprint end
      where s.club_id=p_club_id and s.session_key=item->>'session_key';
  end loop;
  perform set_config('jupr.round_robin_hub_transaction',context_before,true);
  delete from public.round_robin_hub_sessions where club_id=p_club_id and hub_id=p_hub_id and not session_key=any(group_keys);
  insert into public.round_robin_hub_sessions(club_id,hub_id,session_key)
    select p_club_id,p_hub_id,unnest(group_keys) on conflict (club_id,session_key) do nothing;

  if p_state->>'identity_version'='2' then
    select coalesce(jsonb_object_agg(session_key,state#>'{page_state,event}'),'{}'::jsonb)
      into identity_events from public.live_sessions
      where club_id=p_club_id and session_key=any(group_keys);
    identity_snapshot := public.round_robin_hub_identity_snapshot(p_state,identity_events);
    if exists (
      select 1 from jsonb_array_elements(coalesce(p_state->'identity_decisions','[]'::jsonb)) d
      join jsonb_array_elements(identity_snapshot->'participants') endpoint
        on endpoint->>'session_key'=d#>>'{left,session_key}' and endpoint->>'participant_id'=d#>>'{left,participant_id}'
      join jsonb_array_elements(identity_snapshot->'participants') endpoint_right
        on endpoint_right->>'session_key'=d#>>'{right,session_key}' and endpoint_right->>'participant_id'=d#>>'{right,participant_id}'
      join jsonb_array_elements(identity_snapshot->'participants') occupied
        on occupied->>'identity'=endpoint->>'identity'
      where d->>'resolution'='same' and previous_busy_refs ? (occupied->>'ref')
        and coalesce(previous_hub_state #>> array['participant_identities', d#>>'{left,session_key}', d#>>'{left,participant_id}'],
          case when endpoint->>'profile_id' is null then endpoint->>'ref' else 'player:' || (endpoint->>'profile_id') end)
          is distinct from
          coalesce(previous_hub_state #>> array['participant_identities', d#>>'{right,session_key}', d#>>'{right,participant_id}'],
          case when endpoint_right->>'profile_id' is null then endpoint_right->>'ref' else 'player:' || (endpoint_right->>'profile_id') end)
        and not exists (select 1 from jsonb_array_elements(coalesce(previous_hub_state->'identity_decisions','[]'::jsonb)) old_decision
          where old_decision->>'resolution'='same' and (
            (old_decision->'left'=d->'left' and old_decision->'right'=d->'right') or
            (old_decision->'left'=d->'right' and old_decision->'right'=d->'left')))
    ) then
      raise exception using errcode='22023', message='Release the affected current games and reservations before merging player identities.';
    end if;
  end if;
  delete from public.round_robin_hub_court_leases where club_id=p_club_id and hub_id=p_hub_id;
  delete from public.round_robin_hub_player_leases where club_id=p_club_id and hub_id=p_hub_id;
  if p_status='active' then
    for court in select value from jsonb_array_elements(p_state->'courts') loop
      normalized_label := lower(regexp_replace(btrim(court->>'label'),'\s+',' ','g'));
      insert into public.round_robin_hub_court_leases(club_id,court_key,hub_id,court_id)
        values (p_club_id,normalized_label,p_hub_id,court->>'id');
      for assignment_slot in select unnest(array['current','next']) loop
      assignment := court->assignment_slot;
      if assignment is null or assignment='null'::jsonb then continue; end if;
      select * into live from public.live_sessions where club_id=p_club_id and session_key=assignment->>'session_key';
      select r into round_row from jsonb_array_elements(live.state #> '{page_state,event,rounds}') r
        where r->>'number'=assignment->>'round_number';
      select m into match_row from (
        select value m from jsonb_array_elements(coalesce(round_row->'matches','[]'::jsonb))
        union all
        select m from jsonb_array_elements(coalesce(round_row->'courts','[]'::jsonb)) c,
          lateral jsonb_array_elements(coalesce(c->'matches','[]'::jsonb)) m
      ) matches where m->>'id'=assignment->>'match_id' limit 1;
      if match_row is null then
        raise exception using errcode = '22023', message = 'The assigned match is no longer present in its round robin.';
      end if;
      for participant_id in select jsonb_array_elements_text(
        coalesce(match_row->'sideA',match_row->'teamA','[]'::jsonb) || coalesce(match_row->'sideB',match_row->'teamB','[]'::jsonb)
      ) loop
        select p into participant from jsonb_array_elements(live.state #> '{page_state,event,participants}') p where p->>'id'=participant_id;
        if participant is null then
          raise exception using errcode = '22023', message = 'An assigned player is missing from its round robin.';
        end if;
        player_name := public.round_robin_hub_name_key(participant->>'name');
        player_profile := nullif(participant->>'player_id','');
        if identity_snapshot is not null then
          select r into identity_participant from jsonb_array_elements(identity_snapshot->'participants') r
            where r->>'session_key'=assignment->>'session_key' and r->>'participant_id'=participant_id;
          if identity_participant is null then
            raise exception using errcode='22023', message='An assigned player is missing its hub identity.';
          end if;
          identity_key := identity_participant->>'identity';
          player_profile := identity_participant->>'canonical_profile_id';
          -- Pending reviews only block new affected assignments. Existing games
          -- and reservations can still be scored, released, or moved intact.
          if (identity_snapshot->'pending') ? (identity_participant->>'ref') and not exists (
            select 1 from jsonb_array_elements(coalesce(previous_hub_state->'courts','[]'::jsonb)) previous
            where previous_busy_refs ? (identity_participant->>'ref')
              and previous->assignment_slot->>'session_key'=assignment->>'session_key'
              and previous->assignment_slot->>'match_id'=assignment->>'match_id'
              and previous->assignment_slot->>'round_number'=assignment->>'round_number'
          ) then
            raise exception using errcode='22023', message='Resolve this participant identity review before assigning or reserving the match.';
          end if;
        else
          -- Exact legacy identity semantics remain available to old API workers.
          player_name := lower(regexp_replace(btrim(participant->>'name'),'\s+',' ','g'));
          identity_key := case when player_profile is not null then 'player:' || player_profile else 'name:' || player_name end;
        end if;
        if exists (select 1 from public.round_robin_hub_player_leases l
          where l.club_id=p_club_id and l.player_name_key=player_name
            and (l.profile_id is null or player_profile is null)
            and identity_snapshot is null) then
          raise exception using errcode = '40001', message = 'A player with this name is already assigned or reserved in another hub. Use the same club profile in every group, or resolve the other hub before assigning.';
        end if;
        insert into public.round_robin_hub_player_leases(club_id,player_key,player_name_key,profile_id,hub_id,court_id,session_key,match_id)
          values(p_club_id,identity_key,player_name,player_profile,p_hub_id,court->>'id',assignment->>'session_key',assignment->>'match_id');
      end loop;
      end loop;
    end loop;
  end if;
  update public.round_robin_hubs set title=btrim(p_title),status=p_status,state=p_state,
    version=case when p_expected_version=0 then 1 else version+1 end,updated_at=now_at
    where club_id=p_club_id and id=p_hub_id returning * into hub;
  insert into public.round_robin_hub_operations(club_id,hub_id,operation_key,request_fingerprint,committed_version,session_keys,actor_email)
    values(p_club_id,p_hub_id,p_operation_key,p_fingerprint,hub.version,to_jsonb(read_keys),nullif(btrim(p_actor_email),''));
  select coalesce(jsonb_agg(to_jsonb(s) order by s.session_key),'[]'::jsonb) into result_sessions
    from public.live_sessions s where s.club_id=p_club_id and s.session_key=any(read_keys);
  return jsonb_build_object('hub',to_jsonb(hub),'sessions',result_sessions,'replayed',false);
end;
$$;
revoke all on function public.commit_round_robin_hub(text,uuid,bigint,text,text,text,text,jsonb,jsonb,text) from public, anon, authenticated;
grant execute on function public.commit_round_robin_hub(text,uuid,bigint,text,text,text,text,jsonb,jsonb,text) to service_role;
comment on function public.commit_round_robin_hub(text,uuid,bigint,text,text,text,text,jsonb,jsonb,text) is
  'Service-only atomic hub/session CAS, idempotency, court and player leases. The API authenticates club staff and validates domain commands before calling.';

notify pgrst, 'reload schema';
