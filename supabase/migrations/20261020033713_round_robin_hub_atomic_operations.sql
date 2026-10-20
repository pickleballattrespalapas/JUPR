-- A hub is the transactional court authority for its existing generator sessions.
-- Private session JSON (including organizer secrets) stays service-role-only.
create table if not exists public.round_robin_hubs (
  id uuid primary key,
  club_id text not null,
  title text not null check (length(btrim(title)) between 1 and 160),
  status text not null default 'active' check (status in ('active', 'completed')),
  state jsonb not null default '{}'::jsonb check (jsonb_typeof(state) = 'object'),
  version bigint not null default 1 check (version > 0),
  created_at timestamptz not null default clock_timestamp(),
  updated_at timestamptz not null default clock_timestamp(),
  created_by_email text,
  unique (club_id, id)
);
create index if not exists round_robin_hubs_club_status_updated_idx
  on public.round_robin_hubs (club_id, status, updated_at desc);

create table if not exists public.round_robin_hub_sessions (
  club_id text not null,
  hub_id uuid not null,
  session_key text not null,
  primary key (club_id, session_key),
  foreign key (club_id, hub_id) references public.round_robin_hubs (club_id, id) on delete cascade,
  foreign key (club_id, session_key) references public.live_sessions (club_id, session_key) on delete restrict
);
create index if not exists round_robin_hub_sessions_hub_idx
  on public.round_robin_hub_sessions (club_id, hub_id);

create table if not exists public.round_robin_hub_court_leases (
  club_id text not null,
  court_key text not null,
  hub_id uuid not null,
  court_id text not null,
  primary key (club_id, court_key),
  unique (club_id, hub_id, court_id),
  foreign key (club_id, hub_id) references public.round_robin_hubs (club_id, id) on delete cascade
);
create index if not exists round_robin_hub_court_leases_hub_idx
  on public.round_robin_hub_court_leases (club_id, hub_id);

create table if not exists public.round_robin_hub_player_leases (
  club_id text not null,
  player_key text not null,
  player_name_key text not null,
  profile_id text,
  hub_id uuid not null,
  court_id text not null,
  session_key text not null,
  match_id text not null,
  primary key (club_id, player_key),
  foreign key (club_id, hub_id) references public.round_robin_hubs (club_id, id) on delete cascade
);
create index if not exists round_robin_hub_player_leases_hub_idx
  on public.round_robin_hub_player_leases (club_id, hub_id);
create index if not exists round_robin_hub_player_leases_name_idx
  on public.round_robin_hub_player_leases (club_id, player_name_key);

create table if not exists public.round_robin_hub_operations (
  club_id text not null,
  hub_id uuid not null,
  operation_key text not null,
  request_fingerprint text not null,
  committed_version bigint not null,
  session_keys jsonb not null default '[]'::jsonb,
  actor_email text,
  created_at timestamptz not null default clock_timestamp(),
  primary key (club_id, hub_id, operation_key),
  foreign key (club_id, hub_id) references public.round_robin_hubs (club_id, id) on delete cascade
);

alter table public.round_robin_hubs enable row level security;
alter table public.round_robin_hub_sessions enable row level security;
alter table public.round_robin_hub_court_leases enable row level security;
alter table public.round_robin_hub_player_leases enable row level security;
alter table public.round_robin_hub_operations enable row level security;
revoke all on public.round_robin_hubs, public.round_robin_hub_sessions,
  public.round_robin_hub_court_leases, public.round_robin_hub_player_leases,
  public.round_robin_hub_operations from public, anon, authenticated;
grant select, insert, update, delete on public.round_robin_hubs, public.round_robin_hub_sessions,
  public.round_robin_hub_court_leases, public.round_robin_hub_player_leases,
  public.round_robin_hub_operations to service_role;

create or replace function public.guard_round_robin_hub_session()
returns trigger language plpgsql security invoker set search_path = public, pg_temp as $$
declare
  old_hub text := nullif(old.state->>'round_robin_hub_id', '');
  new_hub text;
  context_hub text := nullif(current_setting('jupr.round_robin_hub_transaction', true), '');
begin
  if tg_op = 'DELETE' then
    if old_hub is not null or exists (
      select 1 from public.round_robin_hub_sessions
      where club_id = old.club_id and session_key = old.session_key
    ) then
      raise exception using errcode = '55000', message = 'Remove this session from its Round Robin Hub before deleting it.';
    end if;
    return old;
  end if;
  new_hub := nullif(new.state->>'round_robin_hub_id', '');
  if old_hub is not null or new_hub is not null then
    if old.club_id is distinct from new.club_id or old.session_key is distinct from new.session_key then
      raise exception using errcode = '55000', message = 'A hub-managed session cannot change its identity.';
    end if;
    if old_hub is distinct from new_hub
       or old.state #> '{page_state,event}' is distinct from new.state #> '{page_state,event}'
       or old.status is distinct from new.status
       or old.completed_at is distinct from new.completed_at then
      if context_hub is null
         or (old_hub is not null and context_hub <> old_hub)
         or (new_hub is not null and context_hub <> new_hub) then
        raise exception using errcode = '55000', message = 'This session is managed by a Round Robin Hub. Reload and save through the shared court authority.';
      end if;
    end if;
  end if;
  return new;
end;
$$;
revoke all on function public.guard_round_robin_hub_session() from public, anon, authenticated;
grant execute on function public.guard_round_robin_hub_session() to service_role;
drop trigger if exists guard_round_robin_hub_session_before_write on public.live_sessions;
create trigger guard_round_robin_hub_session_before_write before update or delete on public.live_sessions
for each row execute function public.guard_round_robin_hub_session();

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

  delete from public.round_robin_hub_court_leases where club_id=p_club_id and hub_id=p_hub_id;
  delete from public.round_robin_hub_player_leases where club_id=p_club_id and hub_id=p_hub_id;
  if p_status='active' then
    for court in select value from jsonb_array_elements(p_state->'courts') loop
      normalized_label := lower(regexp_replace(btrim(court->>'label'),'\s+',' ','g'));
      insert into public.round_robin_hub_court_leases(club_id,court_key,hub_id,court_id)
        values (p_club_id,normalized_label,p_hub_id,court->>'id');
      for assignment in select value from jsonb_array_elements(jsonb_build_array(court->'current',court->'next')) where value <> 'null'::jsonb loop
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
        player_name := lower(regexp_replace(btrim(participant->>'name'),'\s+',' ','g'));
        player_profile := nullif(participant->>'player_id','');
        identity_key := case when player_profile is not null then 'player:' || player_profile else 'name:' || player_name end;
        if exists (select 1 from public.round_robin_hub_player_leases l
          where l.club_id=p_club_id and l.player_name_key=player_name
            and (l.profile_id is null or player_profile is null)) then
          raise exception using errcode = '40001', message = 'A player with this name is already assigned or reserved. Use the same club profile in every group, or distinguish different players with the same name.';
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
