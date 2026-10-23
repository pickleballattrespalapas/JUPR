-- A stale business snapshot is terminal (PT409), not a retryable serialization
-- failure. Durable mutation receipts are written in the same transaction as play.
drop index if exists public.live_ladder_admin_operations_active_version_unique;
create unique index live_ladder_admin_operations_active_version_unique
  on public.live_ladder_admin_operations(club_id,surface,expected_version)
  where surface <> 'play_generator' and status in ('intent','running','mutated','recovery_required');
create unique index live_ladder_admin_operations_generator_version_unique
  on public.live_ladder_admin_operations(club_id,surface,entity_id,expected_version)
  where surface = 'play_generator' and status in ('intent','running','mutated','recovery_required');

create table public.play_generator_mutation_receipts (
  operation_key text primary key,
  club_id text not null,
  session_key text not null,
  request_fingerprint text not null,
  committed_version timestamptz not null,
  created_at timestamptz not null default clock_timestamp()
);
alter table public.play_generator_mutation_receipts enable row level security;
alter table public.play_generator_mutation_receipts force row level security;
revoke all on public.play_generator_mutation_receipts from public,anon,authenticated;
grant select,insert on public.play_generator_mutation_receipts to service_role;

create function public.record_play_generator_mutation_receipt(
  p_club_id text,p_session_key text,p_operation_key text,p_request_fingerprint text,p_committed_version timestamptz
) returns void language plpgsql security invoker set search_path=public,pg_temp as $$
declare prior public.play_generator_mutation_receipts%rowtype;
begin
  if nullif(p_operation_key,'') is null or nullif(p_request_fingerprint,'') is null
     or nullif(p_session_key,'') is null or nullif(p_club_id,'') is null or p_committed_version is null then
    raise exception using errcode='22023',message='A generator receipt requires its exact operation identity.';
  end if;
  insert into public.play_generator_mutation_receipts(operation_key,club_id,session_key,request_fingerprint,committed_version)
    values(p_operation_key,p_club_id,p_session_key,p_request_fingerprint,p_committed_version)
    on conflict(operation_key) do nothing;
  select * into prior from public.play_generator_mutation_receipts where operation_key=p_operation_key;
  if prior.club_id<>p_club_id or prior.session_key<>p_session_key or prior.request_fingerprint<>p_request_fingerprint then
    raise exception using errcode='PT409',message='This generator operation key belongs to a different request.';
  end if;
end;
$$;
revoke all on function public.record_play_generator_mutation_receipt(text,text,text,text,timestamptz) from public,anon,authenticated;
grant execute on function public.record_play_generator_mutation_receipt(text,text,text,text,timestamptz) to service_role;

create function public.commit_admin_play_generator_session(
  p_club_id text,p_session_key text,p_expected_version timestamptz,p_expected_numeric_version bigint,
  p_patch jsonb,p_operation_key text,p_request_fingerprint text
) returns jsonb language plpgsql security invoker set search_path=public,pg_temp as $$
declare
  op public.live_ladder_admin_operations%rowtype;
  receipt public.play_generator_mutation_receipts%rowtype;
  live public.live_sessions%rowtype;
  now_at timestamptz := clock_timestamp();
begin
  select * into op from public.live_ladder_admin_operations where operation_key=p_operation_key for update;
  if op.operation_key is null or op.club_id is distinct from p_club_id or op.surface<>'play_generator'
     or op.request_fingerprint is distinct from p_request_fingerprint
     or (op.entity_id <> p_session_key and not (op.entity_id='new' and p_expected_version is null)) then
    raise exception using errcode='22023',message='The generator write does not match its durable operation.';
  end if;
  select * into receipt from public.play_generator_mutation_receipts where operation_key=p_operation_key;
  if found then
    if receipt.club_id<>p_club_id or receipt.request_fingerprint<>p_request_fingerprint
      or (op.entity_id<>'new' and receipt.session_key<>p_session_key) then
      raise exception using errcode='PT409',message='This generator operation key belongs to a different request.';
    end if;
    select * into live from public.live_sessions where club_id=p_club_id and session_key=receipt.session_key;
    return to_jsonb(live);
  end if;
  if op.status not in ('running','recovery_required') then
    raise exception using errcode='55000',message='This generator operation is not available for mutation.';
  end if;
  if jsonb_typeof(p_patch) is distinct from 'object' or nullif(p_session_key,'') is null then
    raise exception using errcode='22023',message='Invalid generator mutation.';
  end if;
  select * into live from public.live_sessions where club_id=p_club_id and session_key=p_session_key for update;
  if p_expected_version is null then
    if live.id is not null or op.entity_id<>'new' or p_patch->>'club_id' is distinct from p_club_id
      or p_patch->>'session_key' is distinct from p_session_key or p_patch#>>'{state,mode}' is distinct from 'admin_play_generator'
      or nullif(p_patch#>>'{state,round_robin_hub_id}','') is not null then
      raise exception using errcode='22023',message='Invalid new generator session.';
    end if;
    insert into public.live_sessions(club_id,session_key,title,status,state,source,created_by_email,created_at,updated_at,
      last_seen_at,expires_at,version,last_operation_key,last_request_fingerprint)
      values(p_club_id,p_session_key,p_patch->>'title','active',p_patch->'state','play_generator',p_patch->>'created_by_email',
      now_at,now_at,now_at,(p_patch->>'expires_at')::timestamptz,1,p_operation_key,p_request_fingerprint) returning * into live;
  else
    if live.id is null or live.updated_at is distinct from p_expected_version
       or live.version is distinct from p_expected_numeric_version then
      raise exception using errcode='PT409',message='This generator session changed. Reload it before saving.';
    end if;
    if live.state->>'mode' is distinct from 'admin_play_generator'
      or nullif(live.state->>'round_robin_hub_id','') is not null or nullif(live.pending_operation_key,'') is not null
      or live.state->'generator_submission' is not null and live.state->'generator_submission'<>'null'::jsonb
      or nullif(live.state#>>'{official_publish,published_at}','') is not null
      or jsonb_array_length(coalesce(live.state#>'{official_publish,published_match_ids}','[]'::jsonb))>0 then
      raise exception using errcode='55000',message='This generator requires its authorized hub or results-review workflow.';
    end if;
    if exists(select 1 from jsonb_object_keys(p_patch) k where k not in
        ('state','status','updated_at','last_seen_at','completed_at','last_operation_key','last_request_fingerprint'))
      or p_patch#>>'{state,mode}' is distinct from 'admin_play_generator'
      or nullif(p_patch#>>'{state,round_robin_hub_id}','') is not null then
      raise exception using errcode='22023',message='Unsupported generator mutation fields.';
    end if;
    update public.live_sessions s set state=p_patch->'state',
      status=case when p_patch ? 'status' then p_patch->>'status' else s.status end,
      completed_at=case when p_patch ? 'completed_at' then (p_patch->>'completed_at')::timestamptz else s.completed_at end,
      version=s.version+1,updated_at=now_at,last_seen_at=now_at,last_operation_key=p_operation_key,
      last_request_fingerprint=p_request_fingerprint
      where club_id=p_club_id and session_key=p_session_key returning * into live;
  end if;
  perform public.record_play_generator_mutation_receipt(p_club_id,live.session_key,p_operation_key,p_request_fingerprint,live.updated_at);
  return to_jsonb(live);
end;
$$;
revoke all on function public.commit_admin_play_generator_session(text,text,timestamptz,bigint,jsonb,text,text) from public,anon,authenticated;
grant execute on function public.commit_admin_play_generator_session(text,text,timestamptz,bigint,jsonb,text,text) to service_role;

-- Public organizer and shared-runner authentication is checked by the API.
-- The private operation ledger binds this RPC to that authorized request.
create function public.commit_public_play_generator_session(
  p_club_id text,p_session_key text,p_expected_version timestamptz,p_expected_numeric_version bigint,
  p_patch jsonb,p_operation_key text,p_request_fingerprint text
) returns jsonb language plpgsql security invoker set search_path=public,pg_temp as $$
declare op public.public_live_operations%rowtype; receipt public.play_generator_mutation_receipts%rowtype;
  live public.live_sessions%rowtype; now_at timestamptz := clock_timestamp();
begin
  select * into op from public.public_live_operations where operation_key=p_operation_key for update;
  if op.operation_key is null or op.club_id is distinct from p_club_id or op.session_key is distinct from p_session_key
     or op.request_fingerprint is distinct from p_request_fingerprint then
    raise exception using errcode='22023',message='The generator write does not match its authorized operation.';
  end if;
  select * into receipt from public.play_generator_mutation_receipts where operation_key=p_operation_key;
  if found then
    if receipt.club_id<>p_club_id or receipt.session_key<>p_session_key or receipt.request_fingerprint<>p_request_fingerprint then
      raise exception using errcode='PT409',message='This generator operation key belongs to a different request.';
    end if;
    select * into live from public.live_sessions where club_id=p_club_id and session_key=p_session_key;
    return to_jsonb(live);
  end if;
  if op.status not in ('intent','running','recovery_required') or jsonb_typeof(p_patch) is distinct from 'object' then
    raise exception using errcode='55000',message='This generator operation is not available for mutation.';
  end if;
  select * into live from public.live_sessions where club_id=p_club_id and session_key=p_session_key for update;
  if live.id is null or live.updated_at is distinct from p_expected_version or live.version is distinct from p_expected_numeric_version then
    raise exception using errcode='PT409',message='This generator session changed. Refresh before continuing.';
  end if;
  if coalesce(live.state->>'mode','') not in ('admin_play_generator','public_play_generator')
    or nullif(live.state->>'round_robin_hub_id','') is not null or nullif(live.pending_operation_key,'') is not null
    or live.state->'generator_submission' is not null and live.state->'generator_submission'<>'null'::jsonb
    or nullif(live.state#>>'{official_publish,published_at}','') is not null
    or jsonb_array_length(coalesce(live.state#>'{official_publish,published_match_ids}','[]'::jsonb))>0 then
    raise exception using errcode='55000',message='This generator requires its authorized hub or results-review workflow.';
  end if;
  if exists(select 1 from jsonb_object_keys(p_patch) k where k not in
      ('state','status','version','updated_at','last_seen_at','expires_at','completed_at','last_operation_key','last_request_fingerprint'))
    or p_patch#>>'{state,mode}' is distinct from live.state->>'mode'
    or nullif(p_patch#>>'{state,round_robin_hub_id}','') is not null then
    raise exception using errcode='22023',message='Unsupported generator mutation fields.';
  end if;
  update public.live_sessions s set state=p_patch->'state',
    status=case when p_patch ? 'status' then p_patch->>'status' else s.status end,
    completed_at=case when p_patch ? 'completed_at' then (p_patch->>'completed_at')::timestamptz else s.completed_at end,
    expires_at=case when p_patch ? 'expires_at' then (p_patch->>'expires_at')::timestamptz else s.expires_at end,
    version=s.version+1,updated_at=now_at,last_seen_at=now_at,last_operation_key=p_operation_key,last_request_fingerprint=p_request_fingerprint
    where club_id=p_club_id and session_key=p_session_key returning * into live;
  perform public.record_play_generator_mutation_receipt(p_club_id,p_session_key,p_operation_key,p_request_fingerprint,live.updated_at);
  return to_jsonb(live);
end;
$$;
revoke all on function public.commit_public_play_generator_session(text,text,timestamptz,bigint,jsonb,text,text) from public,anon,authenticated;
grant execute on function public.commit_public_play_generator_session(text,text,timestamptz,bigint,jsonb,text,text) to service_role;

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
      raise exception using errcode = 'PT409', message = 'This operation key was already used for a different hub command.';
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
      raise exception using errcode = 'PT409', message = 'The Round Robin Hub no longer exists. Reload before saving.';
    end if;
    insert into public.round_robin_hubs (id,club_id,title,status,state,created_by_email)
      values (p_hub_id,p_club_id,btrim(p_title),p_status,p_state,nullif(btrim(p_actor_email),'')) returning * into hub;
  elsif hub.version <> p_expected_version then
    raise exception using errcode = 'PT409', message = 'The Round Robin Hub changed. Reload before saving.';
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
    raise exception using errcode = 'PT409', message = 'A selected round robin already belongs to another hub.';
  end if;

  -- Lock the complete read set in a stable order before applying any session changes.
  for member in select unnest(read_keys) order by 1 loop
    select * into live from public.live_sessions where club_id=p_club_id and session_key=member for update;
    select value into item from jsonb_array_elements(p_sessions) where value->>'session_key'=member;
    if live.id is null
       or nullif(item->>'expected_version','') is null
       or live.updated_at is distinct from (item->>'expected_version')::timestamptz
       or (item ? 'expected_numeric_version' and live.version is distinct from (item->>'expected_numeric_version')::bigint) then
      raise exception using errcode = 'PT409', message = 'A round robin changed. Reload the hub before saving.';
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
      raise exception using errcode = 'PT409', message = 'The supplied session version is stale.';
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

  if p_expected_version>0 and hub.state=p_state and hub.title=btrim(p_title) and hub.status=p_status
     and not exists(select 1 from jsonb_array_elements(p_sessions) e where coalesce(e->'patch','{}'::jsonb)<>'{}'::jsonb) then
    insert into public.round_robin_hub_operations(club_id,hub_id,operation_key,request_fingerprint,committed_version,session_keys,actor_email)
      values(p_club_id,p_hub_id,p_operation_key,p_fingerprint,hub.version,to_jsonb(read_keys),nullif(btrim(p_actor_email),''));
    select coalesce(jsonb_agg(to_jsonb(s) order by s.session_key),'[]'::jsonb) into result_sessions
      from public.live_sessions s where s.club_id=p_club_id and s.session_key=any(read_keys);
    return jsonb_build_object('hub',to_jsonb(hub),'sessions',result_sessions,'replayed',false);
  end if;

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
    if nullif(patch->>'last_operation_key','') is not null then
      select * into live from public.live_sessions where club_id=p_club_id and session_key=item->>'session_key';
      perform public.record_play_generator_mutation_receipt(p_club_id,live.session_key,
        patch->>'last_operation_key',patch->>'last_request_fingerprint',live.updated_at);
    end if;
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
    -- A merge may retain one existing court commitment while linking an idle
    -- duplicate. It cannot combine two previously distinct committed people,
    -- including through a transitive identity component.
    if exists (
      select occupied->>'identity'
      from jsonb_array_elements(identity_snapshot->'participants') occupied
      where previous_busy_refs ? (occupied->>'ref')
      group by occupied->>'identity'
      having count(distinct coalesce(
        previous_hub_state #>> array['participant_identities',occupied->>'session_key',occupied->>'participant_id'],
        case when occupied->>'profile_id' is null then occupied->>'ref' else 'player:' || (occupied->>'profile_id') end
      )) > 1
    ) then
      raise exception using errcode='22023', message='Resolve the affected current games and reservations before merging two committed player identities.';
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
          raise exception using errcode = 'PT409', message = 'A player with this name is already assigned or reserved in another hub. Use the same club profile in every group, or resolve the other hub before assigning.';
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
