begin;

-- Late approval authority is the immutable request record, never its optional
-- explanation. An administrator may record contact details without granting
-- email consent, creating an account, or subscribing the player to anything.
alter table public.pcs_interclub_late_player_requests
 drop constraint pcs_interclub_late_player_requests_reason_check,
 add constraint pcs_interclub_late_player_requests_reason_check check(length(trim(reason))<=500),
 alter column reason set default '',
 add column contact_email text not null default '' check(length(contact_email)<=254 and
  (contact_email='' or contact_email ~ '^[^[:space:]@,<>]+@[^[:space:]@,<>]+\.[^[:space:]@,<>]+$'));

-- Only a successful, atomic admin creation can produce this replay receipt.
-- Requests are scoped to actor, club, season, mode and the complete submitted
-- payload; knowing a UUID never grants access to another person's private link.
create table public.pcs_interclub_player_creation_requests (
 season_id uuid not null,
 club_id text not null,
 request_id uuid not null,
 actor_id uuid not null,
 payload jsonb not null,
 member_id uuid not null,
 created_at timestamptz not null default clock_timestamp(),
 primary key(season_id,club_id,request_id),
 foreign key(member_id,season_id,club_id) references public.pcs_interclub_pool_members(id,season_id,club_id)
);
create index pcs_interclub_player_creation_member_idx
 on public.pcs_interclub_player_creation_requests(member_id,season_id,club_id);
alter table public.pcs_interclub_player_creation_requests enable row level security;
revoke all on public.pcs_interclub_player_creation_requests from public,anon,authenticated,service_role;
grant select,insert on public.pcs_interclub_player_creation_requests to service_role;

-- This internal service-only primitive deliberately never matches/claims an
-- existing player. Callers must use an explicit player ID for an existing one.
-- Profile + pool insertion share the caller's transaction, so later validation
-- failures cannot leave an orphan profile or change any existing rating.
create function public.pcs_interclub_create_directory_player(p_club_id text,p_new_player jsonb)
returns bigint language plpgsql security invoker set search_path=public as $$
declare clean_name text; clean_email text; initial_rating numeric; new_id bigint; gender_value text;
begin
 if jsonb_typeof(p_new_player) is distinct from 'object'
  or (p_new_player - array['name','starting_jupr','gender','email'])<>'{}'::jsonb
  or jsonb_typeof(p_new_player->'name') is distinct from 'string'
  or jsonb_typeof(p_new_player->'starting_jupr') is distinct from 'number'
  or (p_new_player ? 'gender' and jsonb_typeof(p_new_player->'gender') not in ('string','null'))
  or (p_new_player ? 'email' and jsonb_typeof(p_new_player->'email') not in ('string','null')) then
  raise exception 'Enter a name and starting JUPR for the new player' using errcode='22023'; end if;
 clean_name:=trim(regexp_replace(p_new_player->>'name','\s+',' ','g'));
 clean_email:=lower(trim(coalesce(p_new_player->>'email','')));
 initial_rating:=(p_new_player->>'starting_jupr')::numeric;
 gender_value:=p_new_player->>'gender';
 if length(clean_name) not between 1 and 120 or initial_rating not between 1 and 7
  or initial_rating::text in ('NaN','Infinity','-Infinity')
  or (gender_value is not null and gender_value not in ('male','female'))
  or length(clean_email)>254 or (clean_email<>'' and clean_email !~ '^[^[:space:]@,<>]+@[^[:space:]@,<>]+\.[^[:space:]@,<>]+$') then
  raise exception 'Check the new player name, starting JUPR, gender and email' using errcode='22023'; end if;
 -- Coordinate same-name creation across seasons and with tournament intake.
 perform pg_advisory_xact_lock(hashtextextended('public-primary-player-name:'||p_club_id||':'||lower(clean_name),0));
 if exists(select 1 from public.players where club_id=p_club_id
  and lower(regexp_replace(trim(name),'\s+',' ','g'))=lower(clean_name)) then
  raise exception 'A club profile already uses this name. Choose the existing player or ask the club to review it.' using errcode='P4091'; end if;
 insert into public.players(club_id,name,rating,starting_rating,gender,wins,losses,matches_played,active,last_game_at,inactive_at)
 values(p_club_id,clean_name,initial_rating*400,initial_rating*400,gender_value,0,0,0,true,null,null)
 returning id into new_id;
 return new_id;
end $$;

create function public.pcs_interclub_register_new_player(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_late boolean,p_request_id uuid,p_new_player jsonb,p_divisions jsonb,p_reason text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; member public.pcs_interclub_pool_members;
 receipt public.pcs_interclub_player_creation_requests; submitted jsonb; selected_player bigint;
 clean_name text; clean_email text; chosen_divisions jsonb; member_id uuid:=gen_random_uuid();
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_club_id and status='accepted') then
  raise exception 'Accept your club invitation first' using errcode='42501'; end if;
 if p_request_id is null or p_late is null then raise exception 'Include the request ID and registration mode' using errcode='22023'; end if;
 submitted:=jsonb_build_object('new_player',p_new_player,'divisions',p_divisions,'reason',coalesce(p_reason,''),'late',p_late);
 select * into receipt from public.pcs_interclub_player_creation_requests
  where season_id=p_season_id and club_id=p_club_id and request_id=p_request_id;
 if found then
  if receipt.actor_id<>p_actor_id or receipt.payload is distinct from submitted then
   raise exception 'This request changed. Reload before continuing.' using errcode='PT409'; end if;
  select * into member from public.pcs_interclub_pool_members where id=receipt.member_id and season_id=p_season_id and club_id=p_club_id;
  return to_jsonb(member);
 end if;
 if exists(select 1 from public.pcs_interclub_pool_members where season_id=p_season_id and club_id=p_club_id and request_id=p_request_id) then
  raise exception 'This request ID is already in use' using errcode='PT409'; end if;
 perform public.pcs_require_interclub_registration_phase(p_season_id,case when p_late then 'closed' else 'open' end);
 if (((season.details->>'end_date')::date+1)::timestamp at time zone (season.details->>'timezone'))<=clock_timestamp() then
  raise exception 'This season has ended' using errcode='PT409'; end if;
 if length(trim(coalesce(p_reason,'')))>500 or jsonb_typeof(p_divisions) is distinct from 'array'
  or jsonb_array_length(p_divisions)>8 or not (season.details->'divisions' @> p_divisions)
  or (select count(distinct value) from jsonb_array_elements_text(p_divisions))<>jsonb_array_length(p_divisions) then
  raise exception 'Choose season divisions and an optional note of at most 500 characters' using errcode='22023'; end if;
 clean_name:=trim(regexp_replace(p_new_player->>'name','\s+',' ','g'));
 clean_email:=lower(trim(coalesce(p_new_player->>'email','')));
 if exists(select 1 from public.pcs_interclub_pool_members where season_id=p_season_id and club_id=p_club_id
  and lower(regexp_replace(trim(name),'\s+',' ','g'))=lower(clean_name)) then
  raise exception 'A season signup already uses this name. Link or review that signup first.' using errcode='P4092'; end if;
 if (select count(*) from public.pcs_interclub_pool_members where season_id=p_season_id and club_id=p_club_id)>=1000 then
  raise exception 'The season pool is full' using errcode='54000'; end if;
 selected_player:=public.pcs_interclub_create_directory_player(p_club_id,p_new_player);
 chosen_divisions:=p_divisions;
 if chosen_divisions='[]'::jsonb then
  select coalesce(jsonb_agg(d.value order by d.ordinality),'[]'::jsonb) into chosen_divisions
   from jsonb_array_elements_text(season.details->'divisions') with ordinality d(value,ordinality)
   where public.pcs_interclub_rating_in_division((p_new_player->>'starting_jupr')::numeric,d.value);
 end if;
 if p_late then
  insert into public.pcs_interclub_late_player_requests(member_id,season_id,club_id,player_id,requested_by,reason,contact_email)
  values(member_id,p_season_id,p_club_id,selected_player,p_actor_id,trim(coalesce(p_reason,'')),clean_email);
 end if;
 insert into public.pcs_interclub_pool_members(id,season_id,club_id,name,email,divisions,notes,player_id,request_id,request_fingerprint,consent_at)
 values(member_id,p_season_id,p_club_id,clean_name,clean_email,chosen_divisions,case when p_late then '' else trim(coalesce(p_reason,'')) end,
  selected_player,p_request_id,'admin-create:'||p_actor_id::text,null) returning * into member;
 insert into public.pcs_interclub_player_creation_requests(season_id,club_id,request_id,actor_id,payload,member_id)
 values(p_season_id,p_club_id,p_request_id,p_actor_id,submitted,member.id);
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,case when p_late then 'late_player_requested' else 'player_created_and_registered' end,
  jsonb_build_object('member_id',member.id,'player_id',selected_player,'created_player',true,'revision',member.revision,'reason',trim(coalesce(p_reason,''))));
 return to_jsonb(member);
end $$;

-- Preserve the installed phase, identity, approval and cutoff guards. Patch
-- only the existing request-note validation and its blank-note persistence.
do $patch$
declare routine regprocedure; definition text; anchor text; replacement text;
begin
 routine:=to_regprocedure('public.pcs_request_interclub_late_player(uuid,text,text,uuid,bigint,jsonb,text)');
 if routine is null then raise exception 'Late request routine missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:='if p_reason is null or length(trim(p_reason)) not between 1 and 500';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Late request note validation changed'; end if;
 definition:=replace(definition,anchor,'if length(trim(coalesce(p_reason,'''')))>500');
 definition:=replace(definition,'trim(p_reason)','trim(coalesce(p_reason,''''))');
 execute replace(definition,'Choose season divisions and explain the late request','Choose season divisions and an optional note of at most 500 characters');
 routine:=to_regprocedure('public.pcs_guard_interclub_pool_registration()');
 if routine is null then raise exception 'Pool registration guard missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:='or new.email<>''''';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Late request contact guard changed'; end if;
 replacement:=$body$or new.email is distinct from (select r.contact_email from public.pcs_interclub_late_player_requests r where r.member_id=new.id)$body$;
 execute replace(definition,anchor,replacement);
end $patch$;

-- Public creation stays inside the established signup transaction: phase,
-- consent, rate limits and exact fingerprint replay are checked first. The
-- explicit-null namesake fallback remains an unlinked signup, as before.
do $patch$
declare routine regprocedure; definition text; anchor text; replacement text;
begin
 routine:=to_regprocedure('public.pcs_interclub_pool_public_action(text,jsonb,text)');
 if routine is null then raise exception 'Public pool signup routine missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:=$body$  if p_payload ? 'player_id' then
   selected_player:=(p_payload->>'player_id')::bigint;$body$;
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Public pool profile selection changed'; end if;
 replacement:=$body$  if p_payload ? 'new_player' and p_payload->'new_player'<>'null'::jsonb then
   if p_payload->>'player_id' is not null
    or lower(regexp_replace(trim(p_payload->'new_player'->>'name'),'\s+',' ','g')) is distinct from lower(clean_name)
    or lower(trim(p_payload->'new_player'->>'email')) is distinct from clean_email then
    raise exception 'The new profile must use the signup name and email' using errcode='22023'; end if;
   if exists(select 1 from public.pcs_interclub_pool_members where season_id=sid and club_id=cid
    and lower(regexp_replace(trim(name),'\s+',' ','g'))=lower(clean_name)) then
    raise exception 'A season signup already uses this name. Ask the club to link or review it.' using errcode='P4092'; end if;
   selected_player:=public.pcs_interclub_create_directory_player(cid,p_payload->'new_player');
  elsif p_payload ? 'player_id' then
   selected_player:=(p_payload->>'player_id')::bigint;$body$;
 execute replace(definition,anchor,replacement);
end $patch$;

revoke all on function public.pcs_interclub_create_directory_player(text,jsonb),
 public.pcs_interclub_register_new_player(uuid,text,text,uuid,boolean,uuid,jsonb,jsonb,text) from public,anon,authenticated;
grant execute on function public.pcs_interclub_create_directory_player(text,jsonb),
 public.pcs_interclub_register_new_player(uuid,text,text,uuid,boolean,uuid,jsonb,jsonb,text) to service_role;
notify pgrst,'reload schema';
commit;
