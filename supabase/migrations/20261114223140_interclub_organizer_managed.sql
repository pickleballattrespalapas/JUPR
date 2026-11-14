begin;

-- A participating club is a label in details, not a clubs/account record.
create table public.pcs_managed_interclub_seasons (
 id uuid primary key,
 organizer_club_id text not null references public.clubs(id),
 revision integer not null default 1 check(revision>0),
 details jsonb not null check(jsonb_typeof(details)='object'),
 published boolean not null default false,
 source_season_id uuid unique,
 created_at timestamptz not null default now(),
 updated_at timestamptz not null default now()
);
create index pcs_managed_interclub_seasons_owner on public.pcs_managed_interclub_seasons(organizer_club_id,updated_at desc);
create table public.pcs_managed_interclub_meets (
 id uuid primary key,
 season_id uuid not null references public.pcs_managed_interclub_seasons(id),
 revision integer not null default 1 check(revision>0),
 details jsonb not null check(jsonb_typeof(details)='object'),
 state text not null default 'planning' check(state in ('planning','draft','submitted','approved')),
 document jsonb,
 approved_document jsonb,
 host_token_hash text unique check(host_token_hash ~ '^[a-f0-9]{64}$'),
 host_expires_at timestamptz,
 updated_at timestamptz not null default now()
);
create index pcs_managed_interclub_meets_season on public.pcs_managed_interclub_meets(season_id);
create table public.pcs_managed_interclub_audit (
 id bigint generated always as identity primary key,
 season_id uuid not null references public.pcs_managed_interclub_seasons(id),
 meet_id uuid references public.pcs_managed_interclub_meets(id),
 actor_id uuid,
 actor_kind text not null check(actor_kind in ('organizer','host')),
 action text not null,
 revision integer not null,
 before_data jsonb,
 after_data jsonb,
 created_at timestamptz not null default now()
);
create index pcs_managed_interclub_audit_season on public.pcs_managed_interclub_audit(season_id,created_at);
create index pcs_managed_interclub_audit_meet on public.pcs_managed_interclub_audit(meet_id,created_at);
alter table public.pcs_managed_interclub_seasons enable row level security;
alter table public.pcs_managed_interclub_meets enable row level security;
alter table public.pcs_managed_interclub_audit enable row level security;
revoke all on public.pcs_managed_interclub_seasons,public.pcs_managed_interclub_meets,public.pcs_managed_interclub_audit from public,anon,authenticated;
grant select,insert,update on public.pcs_managed_interclub_seasons,public.pcs_managed_interclub_meets,public.pcs_managed_interclub_audit to service_role;
grant usage,select on sequence public.pcs_managed_interclub_audit_id_seq to service_role;

-- Only the API service can call this function. Host credentials are checked
-- again while holding the row lock, including concurrent revoke/replace cases.
create function public.pcs_managed_interclub_write(
 p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,p_meet_id uuid,
 p_action text,p_revision integer,p_payload jsonb,p_token_hash text default null
) returns jsonb language plpgsql security invoker set search_path=public as $$
declare s public.pcs_managed_interclub_seasons; m public.pcs_managed_interclub_meets;
 before_row jsonb; after_row jsonb; imported jsonb; is_host boolean := p_token_hash is not null;
begin
 if is_host then
  if p_action not in ('scores','submit') or p_meet_id is null then
   raise exception 'Host links allow scores and submission only' using errcode='42501';
  end if;
 else
  if p_actor_id is null or not exists(select 1 from public.admin_role_assignments
    where club_id=p_club_id and email=lower(trim(p_actor_email)) and (user_id is null or user_id=p_actor_id)
    and role in ('super_admin','administrator','club_owner') and revoked_at is null and (expires_at is null or expires_at>now())) then
   raise exception 'Organizer administrator required' using errcode='42501';
  end if;
 end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-managed-interclub:'||p_season_id::text,0));
 select * into s from public.pcs_managed_interclub_seasons where id=p_season_id for update;
 if not found then
  if is_host or p_action not in ('season','import') or p_revision<>0 then
   raise exception 'Season unavailable' using errcode='P0002';
  end if;
  if jsonb_typeof(p_payload->'details') is distinct from 'object' then raise exception 'Invalid season' using errcode='22023'; end if;
  if p_action='import' and not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then
   raise exception 'Source season unavailable' using errcode='42501';
  end if;
  insert into public.pcs_managed_interclub_seasons(id,organizer_club_id,details,source_season_id)
   values(p_season_id,p_club_id,p_payload->'details',case when p_action='import' then p_season_id end) returning * into s;
  if p_action='import' then
   for imported in select jsonb_array_elements(p_payload->'meets') loop
    insert into public.pcs_managed_interclub_meets(id,season_id,details,state,document,approved_document)
     values((imported->>'id')::uuid,s.id,imported->'details',imported->>'state',nullif(imported->'document','null'::jsonb),nullif(imported->'approved_document','null'::jsonb));
   end loop;
  end if;
  after_row=to_jsonb(s);
 elsif not is_host and s.organizer_club_id<>p_club_id then
  raise exception 'Organizer mismatch' using errcode='42501';
 elsif p_action in ('season','publish','unpublish') and not is_host then
  if s.revision<>p_revision then raise exception 'Season changed; reload' using errcode='PT409'; end if;
  before_row=to_jsonb(s);
  if p_action='season' then
   if jsonb_typeof(p_payload->'details') is distinct from 'object' then raise exception 'Invalid season' using errcode='22023'; end if;
   update public.pcs_managed_interclub_seasons set details=p_payload->'details' where id=s.id;
  else
   update public.pcs_managed_interclub_seasons set published=(p_action='publish') where id=s.id;
  end if;
  update public.pcs_managed_interclub_seasons set revision=revision+1,updated_at=now() where id=s.id returning * into s;
  after_row=to_jsonb(s);
 elsif p_action='import' and not is_host then
  -- A lost-response retry never recreates or overwrites the imported season.
  if s.source_season_id is distinct from p_season_id then raise exception 'Season already exists' using errcode='PT409'; end if;
  return to_jsonb(s);
 else
  select * into m from public.pcs_managed_interclub_meets where id=p_meet_id and season_id=s.id for update;
  if is_host and (not found or m.host_token_hash is distinct from p_token_hash or m.host_expires_at is null or m.host_expires_at<=now()) then
   raise exception 'Host link unavailable' using errcode='42501';
  end if;
  if m.id is null then
   if p_action<>'plan' or p_revision<>0 or is_host then raise exception 'Meet unavailable' using errcode='P0002'; end if;
   if s.revision is distinct from (p_payload->>'season_revision')::integer then raise exception 'Season changed; reload' using errcode='PT409'; end if;
   insert into public.pcs_managed_interclub_meets(id,season_id,details) values(p_meet_id,s.id,p_payload->'details') returning * into m;
  else
   if m.revision<>p_revision then raise exception 'Meet changed; reload' using errcode='PT409'; end if;
   before_row=to_jsonb(m)-'host_token_hash';
   if p_action='plan' then
    if m.state not in ('planning','draft') or m.approved_document is not null or
     exists(select 1 from jsonb_array_elements(m.document->'encounters') e,
      jsonb_array_elements(e->'pairings') p,jsonb_array_elements(p->'games') g
      where g->>'a' is not null or g->>'b' is not null or g->>'status' not in ('pending','forfeit','double_forfeit')) then
     raise exception 'Scored meets cannot be regenerated' using errcode='22023';
    end if;
    if s.revision is distinct from (p_payload->>'season_revision')::integer then raise exception 'Season changed; reload' using errcode='PT409'; end if;
    m.details=p_payload->'details'; m.document=null; m.state='planning';
   elsif p_action='generate' then
    if m.state<>'planning' or m.document is not null then raise exception 'Schedule already generated' using errcode='PT409'; end if;
    m.document=p_payload->'document'; m.state='draft';
   elsif p_action in ('scores','submit') then
    if m.state<>'draft' or m.document is null then raise exception 'Meet is not open for scores' using errcode='PT409'; end if;
    if p_action='scores' then m.document=p_payload->'document'; else m.state='submitted'; end if;
   elsif p_action='approve' and not is_host then
    if m.state<>'submitted' then raise exception 'Submit the meet first' using errcode='PT409'; end if;
    m.approved_document=m.document; m.state='approved';
   elsif p_action='reopen' and not is_host then
    if m.state not in ('submitted','approved') or length(trim(coalesce(p_payload->>'reason','')))<3 then raise exception 'A correction reason is required' using errcode='22023'; end if;
    m.state='draft';
   elsif p_action='share' and not is_host then
    if m.document is null or m.state not in ('draft','submitted','approved') then raise exception 'Generate the schedule first' using errcode='22023'; end if;
    if (p_payload->>'token_hash' ~ '^[a-f0-9]{64}$') is not true or ((p_payload->>'expires_at')::timestamptz>now()) is not true then raise exception 'Invalid host access' using errcode='22023'; end if;
    m.host_token_hash=p_payload->>'token_hash';m.host_expires_at=(p_payload->>'expires_at')::timestamptz;
   elsif p_action='revoke' and not is_host then
    m.host_token_hash=null;m.host_expires_at=null;
   else raise exception 'Action unavailable' using errcode='42501';
   end if;
   update public.pcs_managed_interclub_meets set details=m.details,state=m.state,document=m.document,
    approved_document=m.approved_document,host_token_hash=m.host_token_hash,host_expires_at=m.host_expires_at,
    revision=revision+1,updated_at=now() where id=m.id returning * into m;
  end if;
  update public.pcs_managed_interclub_seasons set revision=revision+1,updated_at=now() where id=s.id returning * into s;
  after_row=to_jsonb(m)-'host_token_hash';
 end if;
 insert into public.pcs_managed_interclub_audit(season_id,meet_id,actor_id,actor_kind,action,revision,before_data,after_data)
  values(s.id,m.id,case when is_host then null else p_actor_id end,case when is_host then 'host' else 'organizer' end,
   p_action,coalesce(m.revision,s.revision),before_row,after_row || case when p_action='reopen' then jsonb_build_object('reason',p_payload->>'reason') else '{}'::jsonb end);
 return after_row;
end $$;
revoke all on function public.pcs_managed_interclub_write(uuid,text,text,uuid,uuid,text,integer,jsonb,text) from public,anon,authenticated;
grant execute on function public.pcs_managed_interclub_write(uuid,text,text,uuid,uuid,text,integer,jsonb,text) to service_role;
commit;
