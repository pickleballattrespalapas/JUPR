begin;

-- Admin-entered verbal/email commitments can omit email. Public signup still
-- requires a real address and consent. No contact consent is invented for admin additions.
alter table public.pcs_interclub_pool_members drop constraint pcs_interclub_pool_members_email_check;
alter table public.pcs_interclub_pool_members add constraint pcs_interclub_pool_members_email_check
 check(email='' or length(email) between 3 and 254);
alter table public.pcs_interclub_pool_members alter column consent_at drop not null;
-- Explicitly chosen different profiles may legitimately share a name/address.
drop index public.pcs_interclub_pool_member_identity_idx;
create unique index pcs_interclub_pool_member_identity_idx on public.pcs_interclub_pool_members
 (season_id,club_id,lower(regexp_replace(trim(name),'\s+',' ','g')),lower(trim(email)),coalesce(player_id,0));

-- One source for current league rating and division bands, shared with roster
-- enforcement. No meet deadline snapshot is changed by this read helper.
create function public.pcs_interclub_pool_player_details(p_season_id uuid,p_club_id text,p_player_ids bigint[])
returns jsonb language sql stable security invoker set search_path=public as $$
 with rated as (
  select p.id,p.rating/400.0 as club_rating,
   public.pcs_interclub_rating_at(p_season_id,e.id,now()) as league_rating,s.details
  from public.players p join public.pcs_interclub_seasons s on s.id=p_season_id
  left join public.pcs_interclub_entries e on e.season_id=p_season_id and e.club_id=p.club_id and e.player_id=p.id
  where p.club_id=p_club_id and p.id=any(p_player_ids)
   and exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_club_id and status='accepted')
 ) select coalesce(jsonb_agg(jsonb_build_object('player_id',id,'league_rating',league_rating,
  'eligible_divisions',(select coalesce(jsonb_agg(d.value order by d.ordinality),'[]'::jsonb)
   from jsonb_array_elements_text(details->'divisions') with ordinality as d(value,ordinality)
   where public.pcs_interclub_rating_in_division(coalesce(league_rating,club_rating),d.value)))),'[]'::jsonb) from rated
$$;
revoke all on function public.pcs_interclub_pool_player_details(uuid,text,bigint[]) from public,anon,authenticated;
grant execute on function public.pcs_interclub_pool_player_details(uuid,text,bigint[]) to service_role;

-- Normalize whitespace consistently with signup matching, and put all exact
-- matches before partial matches so a result limit cannot hide an ambiguity.
create function public.pcs_interclub_pool_search_players(p_season_id uuid,p_club_id text,p_query text)
returns jsonb language sql stable security invoker set search_path=public as $$
 with term as (select lower(regexp_replace(trim(p_query),'\s+',' ','g')) as value),
 choices as (
  select p.id,p.name,p.rating,p.gender,lower(regexp_replace(trim(p.name),'\s+',' ','g'))=term.value as exact
  from public.players p cross join term
  where p.club_id=p_club_id and p.active is true and length(term.value)>=2 and length(term.value)<=120
   and strpos(lower(regexp_replace(trim(p.name),'\s+',' ','g')),term.value)>0
   and exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_club_id and status='accepted')
  order by exact desc,p.name,p.id limit 20
 ) select coalesce(jsonb_agg(jsonb_build_object('id',id,'name',name,'rating',rating,'gender',gender) order by exact desc,name,id),'[]'::jsonb) from choices
$$;
revoke all on function public.pcs_interclub_pool_search_players(uuid,text,text) from public,anon,authenticated;
grant execute on function public.pcs_interclub_pool_search_players(uuid,text,text) to service_role;

create or replace function public.pcs_guard_interclub_pool_identity() returns trigger language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; represented text; affected_meet uuid;
begin
 select * into season from public.pcs_interclub_seasons where id=new.season_id;
 if tg_op='INSERT' then
  -- Without a supplied email there is no cross-club personal identity evidence.
  -- Do not conflate unrelated namesakes who verbally committed at two clubs.
  new.identity_key:=encode(sha256(convert_to(case when new.email='' then
   'club:'||new.club_id||':player:'||coalesce(new.player_id::text,new.id::text)
   else lower(regexp_replace(trim(new.name),'\s+',' ','g'))||':'||lower(trim(new.email)) end,'UTF8')),'hex');
  new.late_join:=new.created_at>=((season.details->>'start_date')::date::timestamp at time zone (season.details->>'timezone'));
  new.approval_status:='pending';new.approved_at:=null;new.approved_by:=null;
 else
  -- Preserve eligibility at deadlines already passed before a later withdrawal
  -- or approval edit changes the live pool row.
  for affected_meet in select id from public.pcs_interclub_meets where season_id=old.season_id and club_ids ? old.club_id and roster_deadline<=now() order by id loop
   perform public.pcs_lock_interclub_meet_eligibility(affected_meet);
  end loop;
  if new.season_id<>old.season_id or new.club_id<>old.club_id or new.identity_key<>old.identity_key then raise exception 'Pool identity is immutable' using errcode='22023'; end if;
  new.late_join:=old.late_join;
  -- Linking a different directory player needs a fresh approval after the season starts.
  if old.player_id is not null and new.player_id is distinct from old.player_id then
   if exists(select 1 from public.pcs_interclub_represented_players a where a.season_id=old.season_id and a.identity_key=old.identity_key) then raise exception 'A player who has competed cannot be reassigned' using errcode='22023'; end if;
   new.late_join:=new.late_join or now()>=((season.details->>'start_date')::date::timestamp at time zone (season.details->>'timezone'));
   new.approval_status:='pending';new.approved_at:=null;new.approved_by:=null;
  end if;
 end if;
 if new.player_id is not null and not new.late_join and new.approval_status='pending' and exists(
  select 1 from public.players where club_id=new.club_id and id=new.player_id and active is true
   and rating>0 and rating::text not in ('NaN','Infinity','-Infinity')) then
  new.approval_status:='approved';new.approved_at:=coalesce(new.approved_at,now());
 end if;
 if new.player_id is null and new.approval_status='approved' then new.approval_status:='pending';new.approved_at:=null; end if;
 select club_id into represented from public.pcs_interclub_represented_players where season_id=new.season_id and identity_key=new.identity_key;
 if new.status='active' and represented is not null and represented<>new.club_id then raise exception 'Player already represents another club this season' using errcode='22023'; end if;
 return new;
end $$;
create or replace function public.pcs_interclub_pool_public_action(p_action text,p_payload jsonb,p_requester_hash text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; settings public.pcs_interclub_pool_settings; member public.pcs_interclub_pool_members;
 response public.pcs_interclub_availability_responses; meet public.pcs_interclub_meets; availability public.pcs_interclub_availability_settings;
 sid uuid; cid text; selected_player bigint; match_count integer; clean_name text; clean_email text; clean_divisions jsonb; n integer; bucket_time timestamptz:=date_trunc('hour',now());
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
 if (((season.details->>'end_date')::date+1)::timestamp at time zone (season.details->>'timezone'))<=now() then raise exception 'Season signup has ended' using errcode='PT409'; end if;
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
   or (clean_email !~ '^[^[:space:]@,<>]+@[^[:space:]@,<>]+\.[^[:space:]@,<>]+$' and not (p_action='update_season' and p_payload->>'status'='withdrawn' and clean_email=''))
   or jsonb_typeof(clean_divisions) is distinct from 'array' or jsonb_array_length(clean_divisions)>8
   or not (season.details->'divisions' @> clean_divisions) or length(coalesce(p_payload->>'notes',''))>1000 then
   raise exception 'Check your name, email, divisions and note' using errcode='22023'; end if;
 end if;
 if p_action='signup' then
  if settings.open is not true or settings.share_id<>(p_payload->>'share_id')::uuid then raise exception 'Season signup is closed' using errcode='PT409'; end if;
  if (p_payload->>'email_consent')::boolean is not true then raise exception 'Agree to season invitations' using errcode='22023'; end if;
  select * into member from public.pcs_interclub_pool_members where season_id=sid and club_id=cid and request_id=(p_payload->>'request_id')::uuid;
  if found then
   if member.request_fingerprint<>p_payload->>'request_fingerprint' then raise exception 'This request changed' using errcode='PT409'; end if;
   return jsonb_build_object('status','registered','member',to_jsonb(member));
  end if;
  -- A directory selection is scoped to this club and must match the entered
  -- name. It is not an account claim. Explicit JSON null opts out of matching.
  if p_payload ? 'player_id' then
   selected_player:=(p_payload->>'player_id')::bigint;
   if selected_player is not null and not exists(select 1 from public.players where club_id=cid and id=selected_player and active is true
    and lower(regexp_replace(trim(name),'\s+',' ','g'))=lower(clean_name)) then
    raise exception 'Choose a matching active profile from this club' using errcode='22023'; end if;
  else
   select count(*),min(id) into match_count,selected_player from public.players where club_id=cid and active is true
    and lower(regexp_replace(trim(name),'\s+',' ','g'))=lower(clean_name);
   if match_count>1 then raise exception 'Choose the correct matching club profile' using errcode='PT422'; end if;
  end if;
  if exists(select 1 from public.pcs_interclub_pool_members where season_id=sid and club_id=cid and
   ((selected_player is not null and player_id=selected_player) or
    ((selected_player is null or player_id is null) and lower(regexp_replace(trim(name),'\s+',' ','g'))=lower(clean_name) and (email=clean_email or email='')))) then
   return jsonb_build_object('status','already_registered'); end if;
  if (select count(*) from public.pcs_interclub_pool_members where season_id=sid and club_id=cid)>=1000 then raise exception 'Contact the club to join' using errcode='54000'; end if;
  insert into public.pcs_interclub_pool_members(season_id,club_id,name,email,divisions,notes,request_id,request_fingerprint,player_id)
  values(sid,cid,clean_name,clean_email,clean_divisions,coalesce(p_payload->>'notes',''),(p_payload->>'request_id')::uuid,p_payload->>'request_fingerprint',selected_player) returning * into member;
  return jsonb_build_object('status','registered','member',to_jsonb(member));
 elsif p_action='update_season' then
  select * into member from public.pcs_interclub_pool_members where id=(p_payload->>'id')::uuid and season_id=sid and club_id=cid for update;
  if not found or member.token_nonce is distinct from (p_payload->>'nonce')::uuid then raise exception 'Link unavailable' using errcode='P0002'; end if;
  if member.revision is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Your response changed' using errcode='PT409'; end if;
  if p_payload->>'status' not in ('active','withdrawn') then raise exception 'Invalid status' using errcode='22023'; end if;
  if settings.open is not true and p_payload->>'status'<>'withdrawn' then raise exception 'Season signup is closed' using errcode='PT409'; end if;
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
   raise exception 'This meet is no longer accepting responses' using errcode='PT409'; end if;
  if response.revision is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Your response changed' using errcode='PT409'; end if;
  if p_payload->>'status' not in ('available','maybe','unavailable') then raise exception 'Choose your availability' using errcode='22023'; end if;
  update public.pcs_interclub_availability_responses set status=p_payload->>'status',revision=revision+1,responded_at=now() where id=response.id returning * into response;
  return to_jsonb(response);
 else raise exception 'Unknown action' using errcode='22023'; end if;
end $$;


-- Atomic per-club batch: preview may become stale, so all ownership, duplicate,
-- season and capacity checks are repeated under the same season lock as signup.
create function public.pcs_interclub_pool_bulk_add(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,p_members jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; item jsonb; clean_name text; clean_email text; divisions jsonb;
 selected_player bigint; added integer:=0; skipped integer:=0; created_ids uuid[]:='{}'; member_id uuid;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_club_id and status='accepted') then
  raise exception 'Accept your club invitation first' using errcode='42501'; end if;
 if (((season.details->>'end_date')::date+1)::timestamp at time zone (season.details->>'timezone'))<=now() then
  raise exception 'Season has ended' using errcode='PT409'; end if;
 if jsonb_typeof(p_members) is distinct from 'array' or jsonb_array_length(p_members) not between 1 and 200 then
  raise exception 'Choose one to 200 players' using errcode='22023'; end if;
 -- Adding players does not open the public signup link.
 insert into public.pcs_interclub_pool_settings(season_id,club_id,open) values(p_season_id,p_club_id,false)
 on conflict(season_id,club_id) do nothing;
 for item in select value from jsonb_array_elements(p_members) loop
  if jsonb_typeof(item) is distinct from 'object' then raise exception 'Invalid player' using errcode='22023'; end if;
  clean_name:=trim(regexp_replace(coalesce(item->>'name',''),'\s+',' ','g'));
  clean_email:=lower(trim(coalesce(item->>'email','')));
  selected_player:=(item->>'player_id')::bigint;
  divisions:=coalesce(item->'divisions','[]'::jsonb);
  if selected_player is not null then
   select name into clean_name from public.players where id=selected_player and club_id=p_club_id and active is true for share;
   if not found then raise exception 'Choose an active club player' using errcode='22023'; end if;
  end if;
  if clean_name is null or length(clean_name) not between 1 and 120 or length(clean_email)>254
   or (clean_email<>'' and clean_email !~ '^[^[:space:]@,<>]+@[^[:space:]@,<>]+\.[^[:space:]@,<>]+$')
   or jsonb_typeof(divisions) is distinct from 'array' or jsonb_array_length(divisions)>8
   or not (season.details->'divisions' @> divisions) or length(coalesce(item->>'notes',''))>1000 then
   raise exception 'Check name, email and season divisions' using errcode='22023'; end if;
  if exists(select 1 from public.pcs_interclub_pool_members where season_id=p_season_id and club_id=p_club_id and
   ((selected_player is not null and player_id=selected_player) or
    ((selected_player is null or player_id is null) and lower(regexp_replace(trim(name),'\s+',' ','g'))=lower(regexp_replace(trim(clean_name),'\s+',' ','g'))
     and (email=clean_email or email='' or clean_email='')))) then skipped:=skipped+1; continue; end if;
  if (select count(*) from public.pcs_interclub_pool_members where season_id=p_season_id and club_id=p_club_id)>=1000 then
   raise exception 'The season pool is full' using errcode='54000'; end if;
  insert into public.pcs_interclub_pool_members(season_id,club_id,name,email,divisions,notes,player_id,request_id,request_fingerprint,consent_at)
  values(p_season_id,p_club_id,clean_name,clean_email,divisions,coalesce(item->>'notes',''),selected_player,gen_random_uuid(),
   'admin:'||p_actor_id::text,null) returning id into member_id;
  created_ids:=array_append(created_ids,member_id); added:=added+1;
 end loop;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'player_pool_bulk_add',jsonb_build_object('member_ids',created_ids,'added',added,'skipped',skipped));
 return jsonb_build_object('added_count',added,'skipped_count',skipped);
end $$;
revoke all on function public.pcs_interclub_pool_bulk_add(uuid,text,text,uuid,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_interclub_pool_bulk_add(uuid,text,text,uuid,jsonb) to service_role;

revoke all on function public.pcs_guard_interclub_pool_identity() from public,anon,authenticated;
grant execute on function public.pcs_guard_interclub_pool_identity() to service_role;
revoke all on function public.pcs_interclub_pool_public_action(text,jsonb,text) from public,anon,authenticated;
grant execute on function public.pcs_interclub_pool_public_action(text,jsonb,text) to service_role;
notify pgrst,'reload schema';
commit;
