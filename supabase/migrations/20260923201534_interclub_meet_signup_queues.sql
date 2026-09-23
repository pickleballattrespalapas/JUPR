begin;

-- A club explicitly delegates lineup maintenance while its meet signup is open.
-- Public clients never receive database grants or supply the delegated actor.
create table public.pcs_interclub_meet_signup_settings (
 season_id uuid not null, club_id text not null, meet_id uuid not null references public.pcs_interclub_meets(id),
 share_id uuid not null default gen_random_uuid() unique, open boolean not null default false,
 deadline timestamptz not null, meet_revision integer not null, revision integer not null default 1,
 actor_id uuid not null, actor_email text not null, updated_at timestamptz not null default now(),
 primary key(season_id,club_id,meet_id),
 foreign key(season_id,club_id) references public.pcs_interclub_participations(season_id,club_id)
);
create table public.pcs_interclub_meet_signups (
 id uuid primary key default gen_random_uuid(), season_id uuid not null, club_id text not null, meet_id uuid not null,
 member_id uuid not null references public.pcs_interclub_pool_members(id), player_id bigint not null,
 division text not null, name text not null, email text not null default '',
 registration_order bigint generated always as identity, registered_at timestamptz not null default clock_timestamp(),
 status text not null default 'active' check(status in ('active','withdrawn')),
 placement text not null default 'review' check(placement in ('confirmed','waitlist','review','withdrawn')),
 priority text not null default 'review' check(priority in ('in_band','play_up','review')),
 gender text not null default 'unknown', rating numeric, reason text not null default '',
 admin_promoted boolean not null default false, revision integer not null default 1,
 request_id uuid not null, fingerprint text not null, token_nonce uuid not null default gen_random_uuid(),
 unique(season_id,club_id,meet_id,player_id), unique(request_id),
 foreign key(season_id,club_id,meet_id) references public.pcs_interclub_meet_signup_settings(season_id,club_id,meet_id),
 foreign key(club_id,player_id) references public.players(club_id,id)
);
create index pcs_interclub_meet_signups_queue_idx on public.pcs_interclub_meet_signups(season_id,club_id,meet_id,division,gender,registration_order);
create index pcs_interclub_meet_signups_member_idx on public.pcs_interclub_meet_signups(member_id);
create index pcs_interclub_meet_signups_player_idx on public.pcs_interclub_meet_signups(club_id,player_id);
create table public.pcs_interclub_meet_signup_teams (
 season_id uuid not null, club_id text not null, meet_id uuid not null, division text not null,
 team_id uuid not null default gen_random_uuid(), team_revision integer not null default 0,
 signature jsonb not null default '[]', primary key(season_id,club_id,meet_id,division),
 foreign key(season_id,club_id,meet_id) references public.pcs_interclub_meet_signup_settings(season_id,club_id,meet_id)
);
create index pcs_interclub_meet_signup_teams_team_idx on public.pcs_interclub_meet_signup_teams(team_id);
do $$ declare t text; begin
 foreach t in array array['pcs_interclub_meet_signup_settings','pcs_interclub_meet_signups','pcs_interclub_meet_signup_teams'] loop
  execute format('alter table public.%I enable row level security',t);
  execute format('revoke all on public.%I from public,anon,authenticated',t);
  execute format('grant all on public.%I to service_role',t);
 end loop;
end $$;
grant usage,select on sequence public.pcs_interclub_meet_signups_registration_order_seq to service_role;

-- Playing up remains legal. Priority is separate, uses the exact league rating,
-- and never rounds 2.93 into the 3.0 band. The open division starts at 4.5.
create function public.pcs_interclub_signup_priority(p_rating numeric,p_division text)
returns text language sql immutable security invoker set search_path=public as $$
 select case when not coalesce(public.pcs_interclub_rating_in_division(p_rating,p_division),false) then 'review'
  when p_rating >= case when lower(p_division) in ('open','4.5/open') then 4.5 else p_division::numeric end then 'in_band'
  else 'play_up' end
$$;

-- Preserve every existing deadline, played-meet, pool and rating guard.
alter function public.pcs_save_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,text,text,bigint[],boolean)
 rename to pcs_save_interclub_meet_roster_before_signups;
create function public.pcs_save_interclub_meet_roster(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_meet_id uuid,p_meet_revision integer,p_team_id uuid,p_revision integer,p_name text,p_division text,p_player_ids bigint[],p_withdraw boolean default false)
returns jsonb language plpgsql security invoker set search_path=public as $$
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if exists(select 1 from public.pcs_interclub_meet_signup_settings s join public.pcs_interclub_meets m on m.id=s.meet_id
  where s.season_id=p_season_id and s.club_id=p_club_id and s.meet_id=p_meet_id and s.open
  and s.meet_revision=m.revision and now()<least(s.deadline,m.roster_deadline,m.starts_at)) then
  raise exception 'Close meet signup before editing its automatic lineup.' using errcode='PT409';
 end if;
 return public.pcs_save_interclub_meet_roster_before_signups(p_actor_id,p_actor_email,p_club_id,p_season_id,
  p_meet_id,p_meet_revision,p_team_id,p_revision,p_name,p_division,p_player_ids,p_withdraw);
end $$;

-- Caller holds the staff lock then the season lock, matching roster operations.
-- FIFO is a database sequence assigned under that lock, never a client timestamp.
create function public.pcs_reconcile_interclub_meet_signups(p_season_id uuid,p_club_id text,p_meet_id uuid)
returns void language plpgsql security invoker set search_path=public as $$
declare cfg public.pcs_interclub_meet_signup_settings; meet public.pcs_interclub_meets;
 q record; mapping public.pcs_interclub_meet_signup_teams; team public.pcs_interclub_teams;
 ids bigint[]; sig jsonb; result jsonb; team_name text;
begin
 select * into cfg from public.pcs_interclub_meet_signup_settings where season_id=p_season_id and club_id=p_club_id and meet_id=p_meet_id;
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id;
 if cfg.open is not true or cfg.meet_revision<>meet.revision or now()>=least(cfg.deadline,meet.roster_deadline,meet.starts_at) then
  raise exception 'Meet signup is closed or the schedule changed.' using errcode='PT409'; end if;
 -- Refresh source eligibility before every mutation. The roster writer independently
 -- enforces the same hard rules when four spots form a complete team.
 with source as (
  select s.id,p.name,public.pcs_interclub_gender(p.gender) as gender,
   public.pcs_interclub_rating_at(p_season_id,e.id,now()) as rating,
   case when p.active is not true or m.status<>'active' or m.approval_status<>'approved' or m.player_id is distinct from s.player_id or e.id is null
    then 'An approved season player profile is required.'
    when exists(select 1 from public.pcs_interclub_represented_players r where r.season_id=p_season_id and r.identity_key=m.identity_key and r.club_id<>p_club_id)
    then 'This player represents another club this season.'
    when meet.competition_phase='final' and (not exists(select 1 from public.pcs_interclub_appearances a where a.season_id=p_season_id and a.club_id=p_club_id and a.entry_id=e.id and a.phase='regular')
     or not exists(select 1 from public.pcs_interclub_appearances a where a.season_id=p_season_id and a.club_id=p_club_id and a.division=s.division and a.phase='regular'))
    then 'A regular-season appearance is required for the final.'
    else '' end as problem
  from public.pcs_interclub_meet_signups s join public.players p on p.id=s.player_id and p.club_id=s.club_id
  join public.pcs_interclub_pool_members m on m.id=s.member_id
  left join public.pcs_interclub_entries e on e.season_id=s.season_id and e.club_id=s.club_id and e.player_id=s.player_id and e.pool_member_id=m.id
  where s.season_id=p_season_id and s.club_id=p_club_id and s.meet_id=p_meet_id and s.status='active'
 ), classified as (
  select source.*,case when source.problem<>'' or source.gender not in ('female','male') then 'review'
   else public.pcs_interclub_signup_priority(source.rating,s.division) end as priority
  from source join public.pcs_interclub_meet_signups s using(id)
 ) update public.pcs_interclub_meet_signups s set name=c.name,gender=c.gender,rating=c.rating,priority=c.priority,
  placement=case when c.priority='review' then 'review' else 'waitlist' end,
  reason=case when c.problem<>'' then c.problem when c.gender not in ('female','male') then 'Ask your club to complete your gender on your player profile.'
   when c.priority='review' then 'Your league rating is not eligible for this division.'
   when c.priority='play_up' then 'Playing up: waitlisted behind players in this rating band. An admin can fill a vacancy.'
   else 'Waiting for a spot in registration order.' end
 from classified c where s.id=c.id;

 with ranked as (
  select id,row_number() over(partition by division,gender order by case when priority='in_band' then 0 else 1 end,registration_order) as n
  from public.pcs_interclub_meet_signups where season_id=p_season_id and club_id=p_club_id and meet_id=p_meet_id and status='active'
   and (priority='in_band' or (priority='play_up' and admin_promoted))
 ) update public.pcs_interclub_meet_signups s set placement='confirmed',reason=case when s.priority='play_up' then 'An admin approved you to fill a vacancy.' else 'Your spot is reserved.' end
 from ranked r where s.id=r.id and r.n<=2;

 insert into public.pcs_interclub_meet_signup_teams(season_id,club_id,meet_id,division)
 select distinct season_id,club_id,meet_id,division from public.pcs_interclub_meet_signups where season_id=p_season_id and club_id=p_club_id and meet_id=p_meet_id
 on conflict do nothing;
 for mapping in select * from public.pcs_interclub_meet_signup_teams where season_id=p_season_id and club_id=p_club_id and meet_id=p_meet_id order by division loop
  select * into team from public.pcs_interclub_teams where id=mapping.team_id;
  if coalesce(team.revision,0)<>mapping.team_revision then raise exception 'A lineup was edited manually. Keep signup closed and manage this meet in Lineups.' using errcode='PT409'; end if;
  select array_agg(player_id order by player_id),coalesce(jsonb_agg(jsonb_build_array(player_id,rating,gender) order by player_id),'[]') into ids,sig
  from public.pcs_interclub_meet_signups where season_id=p_season_id and club_id=p_club_id and meet_id=p_meet_id and division=mapping.division and status='active' and placement='confirmed';
  if cardinality(ids)=4 and (sig<>mapping.signature or team.withdrawn is true or team.id is null) then
   select left(name,55)||' · '||mapping.division into team_name from public.clubs where id=p_club_id;
   result:=public.pcs_save_interclub_meet_roster_before_signups(cfg.actor_id,cfg.actor_email,p_club_id,p_season_id,p_meet_id,meet.revision,mapping.team_id,mapping.team_revision,team_name,mapping.division,ids,false);
   update public.pcs_interclub_meet_signup_teams set team_revision=(result->'team'->>'revision')::integer,signature=sig where season_id=p_season_id and club_id=p_club_id and meet_id=p_meet_id and division=mapping.division;
  elsif coalesce(cardinality(ids),0)<4 and team.id is not null and not team.withdrawn then
   result:=public.pcs_save_interclub_meet_roster_before_signups(cfg.actor_id,cfg.actor_email,p_club_id,p_season_id,p_meet_id,meet.revision,mapping.team_id,mapping.team_revision,team.name,mapping.division,'{}',true);
   update public.pcs_interclub_meet_signup_teams set team_revision=(result->'team'->>'revision')::integer,signature='[]' where season_id=p_season_id and club_id=p_club_id and meet_id=p_meet_id and division=mapping.division;
  end if;
 end loop;
end $$;

create function public.pcs_interclub_meet_signup_action(p_action text,p_payload jsonb,p_actor_id uuid default null,p_actor_email text default null,p_requester_hash text default null)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare cfg public.pcs_interclub_meet_signup_settings; meet public.pcs_interclub_meets; season public.pcs_interclub_seasons;
 member public.pcs_interclub_pool_members; signup public.pcs_interclub_meet_signups; player public.players;
 sid uuid; cid text; mid uuid; actor uuid; delegated_email text; n integer; joining boolean:=p_action in ('join','add');
begin
 if p_actor_id is not null then
  sid:=(p_payload->>'season_id')::uuid;cid:=p_payload->>'club_id';mid:=(p_payload->>'meet_id')::uuid;
  actor:=p_actor_id;delegated_email:=p_actor_email;
 else
  if p_action='join' then select * into cfg from public.pcs_interclub_meet_signup_settings where share_id=(p_payload->>'share_id')::uuid;
  elsif p_action='withdraw' then
   select * into signup from public.pcs_interclub_meet_signups where id=(p_payload->>'id')::uuid and token_nonce=(p_payload->>'nonce')::uuid;
   select * into cfg from public.pcs_interclub_meet_signup_settings where season_id=signup.season_id and club_id=signup.club_id and meet_id=signup.meet_id;
  else raise exception 'Invalid public action' using errcode='42501'; end if;
  sid:=cfg.season_id;cid:=cfg.club_id;mid:=cfg.meet_id;actor:=cfg.actor_id;delegated_email:=cfg.actor_email;
 end if;
 if sid is null or cid is null or mid is null then raise exception 'Meet signup unavailable' using errcode='P0002'; end if;
 perform public.pcs_require_interclub_admin(actor,delegated_email,cid);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||sid::text,0));
 perform public.pcs_require_interclub_registration_phase(sid,'closed');
 select * into season from public.pcs_interclub_seasons where id=sid;
 select * into meet from public.pcs_interclub_meets where id=mid and season_id=sid for update;
 if not found or not(meet.club_ids ? cid) or not exists(select 1 from public.pcs_interclub_participations where season_id=sid and club_id=cid and status='accepted') then
  raise exception 'Meet signup unavailable' using errcode='P0002'; end if;
 select * into cfg from public.pcs_interclub_meet_signup_settings where season_id=sid and club_id=cid and meet_id=mid for update;
 if p_action='settings' and p_actor_id is not null then
  if coalesce(cfg.revision,0) is distinct from (p_payload->>'expected_revision')::integer or meet.revision is distinct from (p_payload->>'expected_meet_revision')::integer then
   raise exception 'Settings changed. Reload before continuing.' using errcode='PT409'; end if;
  if (p_payload->>'open')::boolean then
   if now()>=meet.roster_deadline or (p_payload->>'deadline')::timestamptz not between now()+interval '1 second' and meet.roster_deadline then
    raise exception 'Choose a future signup deadline at or before the roster deadline.' using errcode='PT422'; end if;
   if exists(select 1 from public.pcs_interclub_teams t where t.season_id=sid and t.club_id=cid and t.meet_id=mid and not t.withdrawn and not exists(select 1 from public.pcs_interclub_meet_signup_teams a where a.team_id=t.id)) then
    raise exception 'This meet already has manual lineups. Withdraw those teams before opening automatic signup, or continue in Lineups.' using errcode='PT409'; end if;
   if cfg.meet_revision is not null and cfg.meet_revision<>meet.revision then
    update public.pcs_interclub_meet_signups set status='withdrawn',placement='withdrawn',reason='The meet changed. Register again to confirm the new schedule.',admin_promoted=false,revision=revision+1 where season_id=sid and club_id=cid and meet_id=mid;
   end if;
  end if;
  insert into public.pcs_interclub_meet_signup_settings(season_id,club_id,meet_id,open,deadline,meet_revision,actor_id,actor_email)
  values(sid,cid,mid,(p_payload->>'open')::boolean,(p_payload->>'deadline')::timestamptz,meet.revision,actor,delegated_email)
  on conflict(season_id,club_id,meet_id) do update set open=excluded.open,deadline=excluded.deadline,meet_revision=excluded.meet_revision,actor_id=excluded.actor_id,actor_email=excluded.actor_email,revision=pcs_interclub_meet_signup_settings.revision+1,updated_at=now()
  returning * into cfg;
  if cfg.open then perform public.pcs_reconcile_interclub_meet_signups(sid,cid,mid); end if;
 else
  if p_actor_id is null and p_action='join' and cfg.share_id is distinct from (p_payload->>'share_id')::uuid then raise exception 'Link changed' using errcode='P0002'; end if;
  if joining then
   select * into signup from public.pcs_interclub_meet_signups where request_id=(p_payload->>'request_id')::uuid;
   if found then
    if signup.season_id<>sid or signup.club_id<>cid or signup.meet_id<>mid or signup.fingerprint is distinct from p_payload->>'fingerprint' then raise exception 'Request changed. Reload signup.' using errcode='PT409'; end if;
    return jsonb_build_object('entry',to_jsonb(signup),'replayed',true);
   end if;
  end if;
  if cfg.open is not true or cfg.meet_revision<>meet.revision or now()>=least(cfg.deadline,meet.roster_deadline,meet.starts_at) then
   raise exception 'Signup is closed or the meet changed. Contact your club for lineup changes.' using errcode='PT409'; end if;
  if p_actor_id is not null then update public.pcs_interclub_meet_signup_settings set actor_id=actor,actor_email=delegated_email where season_id=sid and club_id=cid and meet_id=mid; end if;
  if joining then
   if p_actor_id is null then
    if p_requester_hash is null or length(p_requester_hash)<>64 then raise exception 'Request unavailable' using errcode='22023'; end if;
    insert into public.pcs_interclub_pool_rate_buckets(scope,bucket,requests) values('meet-signup:'||p_requester_hash,date_trunc('hour',now()),1)
     on conflict(scope,bucket) do update set requests=pcs_interclub_pool_rate_buckets.requests+1 returning requests into n;
    if n>40 then raise exception 'Too many signups' using errcode='54000'; end if;
   end if;
   select * into player from public.players where id=(p_payload->>'player_id')::bigint and club_id=cid and active is true for share;
   select * into member from public.pcs_interclub_pool_members where season_id=sid and club_id=cid and player_id=player.id and status='active' and approval_status='approved';
   if player.id is null or member.id is null then raise exception 'Choose an approved player from your club’s season pool.' using errcode='PT422'; end if;
   if p_actor_id is null and lower(regexp_replace(trim(player.name),'\s+',' ','g'))<>lower(regexp_replace(trim(p_payload->>'name'),'\s+',' ','g')) then raise exception 'Choose your player profile again.' using errcode='PT422'; end if;
   if not (season.details->'divisions' ? (p_payload->>'division')) then raise exception 'Choose a division for this season.' using errcode='PT422'; end if;
   if not coalesce(public.pcs_interclub_rating_in_division((select public.pcs_interclub_rating_at(sid,e.id,now()) from public.pcs_interclub_entries e where e.season_id=sid and e.club_id=cid and e.player_id=player.id),p_payload->>'division'),false) then
    raise exception 'Your league rating is above this division’s limit or is unavailable. Choose an eligible division.' using errcode='PT422'; end if;
   select * into signup from public.pcs_interclub_meet_signups where season_id=sid and club_id=cid and meet_id=mid and player_id=player.id;
   if found and signup.status='active' then return jsonb_build_object('duplicate',true); end if;
   if exists(select 1 from public.pcs_interclub_team_players t where t.season_id=sid and t.club_id=cid and t.meet_id=mid and t.player_id=player.id) then
    raise exception 'This player already has a lineup for this meet.' using errcode='PT409'; end if;
   if (select count(*) from public.pcs_interclub_meet_signups where season_id=sid and club_id=cid and meet_id=mid)>=500 and signup.id is null then raise exception 'Meet signup capacity reached' using errcode='54000'; end if;
   insert into public.pcs_interclub_meet_signups(season_id,club_id,meet_id,member_id,player_id,division,name,email,request_id,fingerprint)
   values(sid,cid,mid,member.id,player.id,p_payload->>'division',player.name,coalesce(p_payload->>'email',''),(p_payload->>'request_id')::uuid,p_payload->>'fingerprint')
   on conflict(season_id,club_id,meet_id,player_id) do update set member_id=excluded.member_id,division=excluded.division,name=excluded.name,email=excluded.email,
    status='active',admin_promoted=false,registered_at=clock_timestamp(),registration_order=default,request_id=excluded.request_id,fingerprint=excluded.fingerprint,token_nonce=gen_random_uuid(),revision=pcs_interclub_meet_signups.revision+1
   returning * into signup;
  elsif p_action in ('withdraw','remove','promote') then
   select * into signup from public.pcs_interclub_meet_signups where id=(p_payload->>'id')::uuid and season_id=sid and club_id=cid and meet_id=mid for update;
   if not found or (p_actor_id is null and signup.token_nonce is distinct from (p_payload->>'nonce')::uuid) then raise exception 'Private signup unavailable' using errcode='P0002'; end if;
   if signup.revision is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Signup changed. Reload before continuing.' using errcode='PT409'; end if;
   if p_action='promote' then
    if p_actor_id is null then raise exception 'Administrator required' using errcode='42501'; end if;
    perform public.pcs_reconcile_interclub_meet_signups(sid,cid,mid);
    select * into signup from public.pcs_interclub_meet_signups where id=signup.id;
    if signup.status<>'active' or signup.priority<>'play_up' or signup.placement<>'waitlist' or
     (select count(*) from public.pcs_interclub_meet_signups where season_id=sid and club_id=cid and meet_id=mid and division=signup.division and gender=signup.gender and status='active' and placement='confirmed')>=2 then
     raise exception 'Only a waitlisted player playing up can be approved into an empty spot.' using errcode='PT422'; end if;
    update public.pcs_interclub_meet_signups set admin_promoted=true,revision=revision+1 where id=signup.id;
   else
    if p_action='remove' and p_actor_id is null then raise exception 'Administrator required' using errcode='42501'; end if;
    update public.pcs_interclub_meet_signups set status='withdrawn',placement='withdrawn',reason='Withdrawn from this meet.',admin_promoted=false,revision=revision+1 where id=signup.id;
   end if;
  elsif p_action<>'refresh' or p_actor_id is null then raise exception 'Invalid action' using errcode='42501'; end if;
  perform public.pcs_reconcile_interclub_meet_signups(sid,cid,mid);
 end if;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(sid,actor,cid,'meet_signup_'||p_action,jsonb_build_object('meet_id',mid,'signup_id',signup.id,'actor_kind',case when p_actor_id is null then 'player' else 'admin' end));
 select * into signup from public.pcs_interclub_meet_signups where id=signup.id;
 return jsonb_build_object('entry',case when signup.id is null then null else to_jsonb(signup) end);
end $$;

revoke all on function public.pcs_interclub_signup_priority(numeric,text),public.pcs_reconcile_interclub_meet_signups(uuid,text,uuid),
 public.pcs_interclub_meet_signup_action(text,jsonb,uuid,text,text),
 public.pcs_save_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,text,text,bigint[],boolean) from public,anon,authenticated;
grant execute on function public.pcs_interclub_signup_priority(numeric,text),public.pcs_reconcile_interclub_meet_signups(uuid,text,uuid),
 public.pcs_interclub_meet_signup_action(text,jsonb,uuid,text,text),
 public.pcs_save_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,text,text,bigint[],boolean) to service_role;
notify pgrst,'reload schema';
commit;
