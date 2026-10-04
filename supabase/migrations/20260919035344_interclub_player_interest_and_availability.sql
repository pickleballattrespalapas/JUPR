begin;

-- An interest pool is not a roster. It deliberately does not write rating entries,
-- team membership, player profiles, or another club's information.
create table public.pcs_interclub_pool_settings (
 season_id uuid not null, club_id text not null, share_id uuid not null default gen_random_uuid() unique,
 open boolean not null default false, revision integer not null default 1 check(revision>0),
 updated_at timestamptz not null default now(), primary key(season_id,club_id),
 foreign key(season_id,club_id) references public.pcs_interclub_participations(season_id,club_id)
);
create table public.pcs_interclub_pool_members (
 id uuid primary key default gen_random_uuid(), season_id uuid not null, club_id text not null,
 name text not null check(length(trim(name)) between 1 and 120),
 email text not null check(length(email) between 3 and 254),
 divisions jsonb not null default '[]' check(jsonb_typeof(divisions)='array'),
 notes text not null default '' check(length(notes)<=1000),
 status text not null default 'active' check(status in ('active','withdrawn')),
 player_id bigint, revision integer not null default 1 check(revision>0),
 token_nonce uuid not null default gen_random_uuid(), request_id uuid not null,
 request_fingerprint text not null, consent_at timestamptz not null default now(),
 created_at timestamptz not null default now(),updated_at timestamptz not null default now(),
 unique(id,season_id,club_id), unique(season_id,club_id,request_id),
 foreign key(season_id,club_id) references public.pcs_interclub_pool_settings(season_id,club_id),
 foreign key(club_id,player_id) references public.players(club_id,id)
);
create unique index pcs_interclub_pool_member_identity_idx on public.pcs_interclub_pool_members
 (season_id,club_id,lower(regexp_replace(trim(name),'\s+',' ','g')),lower(trim(email)));
create unique index pcs_interclub_pool_member_player_idx on public.pcs_interclub_pool_members
 (season_id,club_id,player_id) where player_id is not null and status='active';
create index pcs_interclub_pool_member_club_idx on public.pcs_interclub_pool_members(club_id,season_id,status);
create table public.pcs_interclub_availability_settings (
 season_id uuid not null, club_id text not null,meet_id uuid not null,
 open boolean not null default false,deadline timestamptz not null,
 revision integer not null default 1 check(revision>0),updated_at timestamptz not null default now(),
 primary key(season_id,club_id,meet_id),
 foreign key(season_id,club_id) references public.pcs_interclub_participations(season_id,club_id),
 foreign key(meet_id,season_id) references public.pcs_interclub_meets(id,season_id)
);
create table public.pcs_interclub_availability_responses (
 id uuid primary key default gen_random_uuid(),season_id uuid not null,club_id text not null,meet_id uuid not null,member_id uuid not null,
 status text not null default 'invited' check(status in ('invited','available','maybe','unavailable')),
 revision integer not null default 1 check(revision>0),token_nonce uuid not null default gen_random_uuid(),
 invited_at timestamptz not null default now(),responded_at timestamptz,
 unique(season_id,club_id,meet_id,member_id),
 foreign key(season_id,club_id,meet_id) references public.pcs_interclub_availability_settings(season_id,club_id,meet_id),
 foreign key(member_id,season_id,club_id) references public.pcs_interclub_pool_members(id,season_id,club_id)
);
create index pcs_interclub_availability_member_idx on public.pcs_interclub_availability_responses(member_id,season_id,club_id);
create table public.pcs_interclub_pool_rate_buckets (
 scope text not null,bucket timestamptz not null,requests integer not null check(requests>0),primary key(scope,bucket)
);

alter table public.pcs_interclub_pool_settings enable row level security;
alter table public.pcs_interclub_pool_members enable row level security;
alter table public.pcs_interclub_availability_settings enable row level security;
alter table public.pcs_interclub_availability_responses enable row level security;
alter table public.pcs_interclub_pool_rate_buckets enable row level security;
revoke all on public.pcs_interclub_pool_settings,public.pcs_interclub_pool_members,public.pcs_interclub_availability_settings,
 public.pcs_interclub_availability_responses,public.pcs_interclub_pool_rate_buckets from public,anon,authenticated;
grant all on public.pcs_interclub_pool_settings,public.pcs_interclub_pool_members,public.pcs_interclub_availability_settings,
 public.pcs_interclub_availability_responses,public.pcs_interclub_pool_rate_buckets to service_role;

create function public.pcs_interclub_pool_action(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,p_action text,p_payload jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; settings public.pcs_interclub_pool_settings; member public.pcs_interclub_pool_members;
 meet public.pcs_interclub_meets; availability public.pcs_interclub_availability_settings; result jsonb;
 selected_count integer; wanted_count integer; member_ids uuid[];
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_club_id and status='accepted') then
  raise exception 'Accept your club invitation first' using errcode='42501'; end if;
 if p_action='settings' then
  select * into settings from public.pcs_interclub_pool_settings where season_id=p_season_id and club_id=p_club_id for update;
  if coalesce(settings.revision,0) is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Settings changed' using errcode='40001'; end if;
  if (p_payload->>'open')::boolean and (((season.details->>'end_date')::date+1)::timestamp at time zone (season.details->>'timezone'))<=now() then raise exception 'Season ended' using errcode='40001'; end if;
  insert into public.pcs_interclub_pool_settings(season_id,club_id,open)
  values(p_season_id,p_club_id,(p_payload->>'open')::boolean)
  on conflict(season_id,club_id) do update set open=excluded.open,revision=pcs_interclub_pool_settings.revision+1,
   share_id=case when coalesce((p_payload->>'rotate_link')::boolean,false) then gen_random_uuid() else pcs_interclub_pool_settings.share_id end,updated_at=now()
  returning * into settings;
  result:=to_jsonb(settings);
 elsif p_action='member' then
  select * into member from public.pcs_interclub_pool_members where id=(p_payload->>'member_id')::uuid and season_id=p_season_id and club_id=p_club_id for update;
  if not found then raise exception 'Member unavailable' using errcode='P0002'; end if;
  if member.revision is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Member changed' using errcode='40001'; end if;
  if p_payload->>'status' not in ('active','withdrawn') then raise exception 'Invalid member status' using errcode='22023'; end if;
  if p_payload->>'player_id' is not null and not exists(select 1 from public.players where club_id=p_club_id and id=(p_payload->>'player_id')::bigint and active is true) then
   raise exception 'Choose an active player from this club' using errcode='22023'; end if;
  update public.pcs_interclub_pool_members set player_id=(p_payload->>'player_id')::bigint,status=p_payload->>'status',revision=revision+1,updated_at=now()
  where id=member.id returning * into member;
  result:=to_jsonb(member);
 elsif p_action in ('availability','invite') then
  select * into meet from public.pcs_interclub_meets where id=(p_payload->>'meet_id')::uuid and season_id=p_season_id for share;
  if not found then raise exception 'Meet unavailable' using errcode='P0002'; end if;
  if not(meet.club_ids ? p_club_id) then raise exception 'Club not scheduled for meet' using errcode='42501'; end if;
  if meet.starts_at<=now() then raise exception 'Meet has started' using errcode='40001'; end if;
  select * into availability from public.pcs_interclub_availability_settings where season_id=p_season_id and club_id=p_club_id and meet_id=meet.id for update;
  if p_action='availability' then
   if coalesce(availability.revision,0) is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Availability settings changed' using errcode='40001'; end if;
   if (p_payload->>'deadline')::timestamptz>meet.starts_at or ((p_payload->>'open')::boolean and (p_payload->>'deadline')::timestamptz<=now()) then
    raise exception 'Choose a future deadline before the meet starts' using errcode='22023'; end if;
   insert into public.pcs_interclub_availability_settings(season_id,club_id,meet_id,open,deadline)
   values(p_season_id,p_club_id,meet.id,(p_payload->>'open')::boolean,(p_payload->>'deadline')::timestamptz)
   on conflict(season_id,club_id,meet_id) do update set open=excluded.open,deadline=excluded.deadline,revision=pcs_interclub_availability_settings.revision+1,updated_at=now()
   returning * into availability;
   result:=to_jsonb(availability);
  else
   if availability.open is not true or availability.deadline<=now() then raise exception 'Open meet signups first' using errcode='40001'; end if;
   select array_agg(distinct value::uuid) into member_ids from jsonb_array_elements_text(p_payload->'member_ids');
   wanted_count:=coalesce(cardinality(member_ids),0);
   if wanted_count not between 1 and 200 then raise exception 'Choose one to 200 players' using errcode='22023'; end if;
   select count(*) into selected_count from public.pcs_interclub_pool_members where id=any(member_ids) and season_id=p_season_id and club_id=p_club_id and status='active';
   if selected_count<>wanted_count then raise exception 'Choose active players from your season pool' using errcode='22023'; end if;
   insert into public.pcs_interclub_availability_responses(season_id,club_id,meet_id,member_id)
   select p_season_id,p_club_id,meet.id,unnest(member_ids) on conflict(season_id,club_id,meet_id,member_id) do nothing;
   select coalesce(jsonb_agg(to_jsonb(r)),'[]') into result from public.pcs_interclub_availability_responses r
    where r.season_id=p_season_id and r.club_id=p_club_id and r.meet_id=meet.id and r.member_id=any(member_ids);
   result:=jsonb_build_object('responses',result,'meet',to_jsonb(meet),'settings',to_jsonb(availability));
  end if;
 else raise exception 'Unknown action' using errcode='22023'; end if;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'player_pool_'||p_action,jsonb_build_object('member_id',member.id,'meet_id',meet.id));
 return result;
end $$;

create function public.pcs_interclub_pool_public_action(p_action text,p_payload jsonb,p_requester_hash text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; settings public.pcs_interclub_pool_settings; member public.pcs_interclub_pool_members;
 response public.pcs_interclub_availability_responses; meet public.pcs_interclub_meets; availability public.pcs_interclub_availability_settings;
 sid uuid; cid text; clean_name text; clean_email text; clean_divisions jsonb; n integer; bucket_time timestamptz:=date_trunc('hour',now());
begin
 if p_action='signup' then
  select season_id,club_id into sid,cid from public.pcs_interclub_pool_settings where share_id=(p_payload->>'share_id')::uuid;
 else sid:=(p_payload->>'season_id')::uuid;cid:=p_payload->>'club_id'; end if;
 if sid is null or cid is null then raise exception 'Signup unavailable' using errcode='P0002'; end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||sid::text,0));
 select * into season from public.pcs_interclub_seasons where id=sid;
 select * into settings from public.pcs_interclub_pool_settings where season_id=sid and club_id=cid for share;
 if not exists(select 1 from public.pcs_interclub_participations where season_id=sid and club_id=cid and status='accepted') then
  raise exception 'Signup unavailable' using errcode='P0002'; end if;
 if (((season.details->>'end_date')::date+1)::timestamp at time zone (season.details->>'timezone'))<=now() then raise exception 'Season signup has ended' using errcode='40001'; end if;
 if p_requester_hash is null or length(p_requester_hash)<>64 then raise exception 'Request scope unavailable' using errcode='22023'; end if;
 -- Database counters protect all API workers. Both per-visitor and club ceilings
 -- apply; forged forwarding metadata cannot evade the club ceiling.
 insert into public.pcs_interclub_pool_rate_buckets(scope,bucket,requests) values('visitor:'||p_requester_hash,bucket_time,1)
 on conflict(scope,bucket) do update set requests=pcs_interclub_pool_rate_buckets.requests+1 returning requests into n;
 if n>60 then raise exception 'Try again later' using errcode='54000'; end if;
 insert into public.pcs_interclub_pool_rate_buckets(scope,bucket,requests) values('club:'||cid,bucket_time,1)
 on conflict(scope,bucket) do update set requests=pcs_interclub_pool_rate_buckets.requests+1 returning requests into n;
 if n>1000 then raise exception 'Try again later' using errcode='54000'; end if;
 delete from public.pcs_interclub_pool_rate_buckets where bucket<now()-interval '2 days';
 if p_action in ('signup','update_season') then
  clean_name:=trim(regexp_replace(p_payload->>'name','\s+',' ','g'));clean_email:=lower(trim(p_payload->>'email'));
  clean_divisions:=p_payload->'divisions';
  if clean_name is null or length(clean_name) not between 1 and 120 or clean_email is null or length(clean_email)>254
   or clean_email !~ '^[^[:space:]@,<>]+@[^[:space:]@,<>]+\.[^[:space:]@,<>]+$'
   or jsonb_typeof(clean_divisions) is distinct from 'array' or jsonb_array_length(clean_divisions)>8
   or not (season.details->'divisions' @> clean_divisions) or length(coalesce(p_payload->>'notes',''))>1000 then
   raise exception 'Check your name, email, divisions and note' using errcode='22023'; end if;
 end if;
 if p_action='signup' then
  if settings.open is not true or settings.share_id<>(p_payload->>'share_id')::uuid then raise exception 'Season signup is closed' using errcode='40001'; end if;
  if (p_payload->>'email_consent')::boolean is not true then raise exception 'Agree to season invitations' using errcode='22023'; end if;
  select * into member from public.pcs_interclub_pool_members where season_id=sid and club_id=cid and request_id=(p_payload->>'request_id')::uuid;
  if found then
   if member.request_fingerprint<>p_payload->>'request_fingerprint' then raise exception 'This request changed' using errcode='40001'; end if;
   return jsonb_build_object('status','registered','member',to_jsonb(member));
  end if;
  if exists(select 1 from public.pcs_interclub_pool_members where season_id=sid and club_id=cid and lower(name)=lower(clean_name) and email=clean_email) then
   return jsonb_build_object('status','already_registered'); end if;
  if (select count(*) from public.pcs_interclub_pool_members where season_id=sid and club_id=cid)>=1000 then raise exception 'Contact the club to join' using errcode='54000'; end if;
  insert into public.pcs_interclub_pool_members(season_id,club_id,name,email,divisions,notes,request_id,request_fingerprint)
  values(sid,cid,clean_name,clean_email,clean_divisions,coalesce(p_payload->>'notes',''),(p_payload->>'request_id')::uuid,p_payload->>'request_fingerprint') returning * into member;
  return jsonb_build_object('status','registered','member',to_jsonb(member));
 elsif p_action='update_season' then
  select * into member from public.pcs_interclub_pool_members where id=(p_payload->>'id')::uuid and season_id=sid and club_id=cid for update;
  if not found or member.token_nonce is distinct from (p_payload->>'nonce')::uuid then raise exception 'Link unavailable' using errcode='P0002'; end if;
  if member.revision is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Your response changed' using errcode='40001'; end if;
  if p_payload->>'status' not in ('active','withdrawn') then raise exception 'Invalid status' using errcode='22023'; end if;
  if settings.open is not true and p_payload->>'status'<>'withdrawn' then raise exception 'Season signup is closed' using errcode='40001'; end if;
  update public.pcs_interclub_pool_members set name=clean_name,email=clean_email,divisions=clean_divisions,notes=coalesce(p_payload->>'notes',''),
   status=p_payload->>'status',revision=revision+1,updated_at=now() where id=member.id returning * into member;
  return to_jsonb(member);
 elsif p_action='respond_meet' then
  select * into response from public.pcs_interclub_availability_responses where id=(p_payload->>'id')::uuid and season_id=sid and club_id=cid for update;
  if not found or response.token_nonce is distinct from (p_payload->>'nonce')::uuid then raise exception 'Link unavailable' using errcode='P0002'; end if;
  select * into member from public.pcs_interclub_pool_members where id=response.member_id and season_id=sid and club_id=cid;
  select * into meet from public.pcs_interclub_meets where id=response.meet_id and season_id=sid;
  select * into availability from public.pcs_interclub_availability_settings where season_id=sid and club_id=cid and meet_id=meet.id;
  if member.status<>'active' or availability.open is not true or availability.deadline<=now() or meet.starts_at<=now() or not(meet.club_ids ? cid) then
   raise exception 'This meet is no longer accepting responses' using errcode='40001'; end if;
  if response.revision is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Your response changed' using errcode='40001'; end if;
  if p_payload->>'status' not in ('available','maybe','unavailable') then raise exception 'Choose your availability' using errcode='22023'; end if;
  update public.pcs_interclub_availability_responses set status=p_payload->>'status',revision=revision+1,responded_at=now() where id=response.id returning * into response;
  return to_jsonb(response);
 else raise exception 'Unknown action' using errcode='22023'; end if;
end $$;

revoke all on function public.pcs_interclub_pool_action(uuid,text,text,uuid,text,jsonb) from public,anon,authenticated;
revoke all on function public.pcs_interclub_pool_public_action(text,jsonb,text) from public,anon,authenticated;
grant execute on function public.pcs_interclub_pool_action(uuid,text,text,uuid,text,jsonb) to service_role;
grant execute on function public.pcs_interclub_pool_public_action(text,jsonb,text) to service_role;
notify pgrst,'reload schema';
commit;
