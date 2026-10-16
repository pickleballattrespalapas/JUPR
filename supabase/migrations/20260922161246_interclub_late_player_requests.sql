begin;

-- The record authorizes one exact pending pool insertion. It is created only by
-- the trusted admin RPC, never from a public signup or a client-supplied flag.
-- A deferred reference allows the request and its member to be inserted in one
-- transaction, with no orphaned authorization surviving a failed pool insert.
create table public.pcs_interclub_late_player_requests (
 member_id uuid primary key,
 season_id uuid not null,
 club_id text not null,
 player_id bigint not null,
 requested_by uuid not null,
 requested_at timestamptz not null default now(),
 reason text not null check(length(trim(reason)) between 1 and 500),
 unique(season_id,club_id,player_id),
 foreign key(season_id,club_id) references public.pcs_interclub_participations(season_id,club_id),
 foreign key(club_id,player_id) references public.players(club_id,id),
 foreign key(member_id,season_id,club_id) references public.pcs_interclub_pool_members(id,season_id,club_id)
  deferrable initially deferred
);
alter table public.pcs_interclub_late_player_requests enable row level security;
revoke all on public.pcs_interclub_late_player_requests from public,anon,authenticated,service_role;
grant select,insert on public.pcs_interclub_late_player_requests to service_role;

create function public.pcs_guard_interclub_late_request() returns trigger
language plpgsql security invoker set search_path=public as $$
begin
 if tg_op<>'INSERT' then raise exception 'A submitted late request is immutable' using errcode='42501'; end if;
 if exists(select 1 from public.pcs_interclub_pool_members where id=new.member_id) then
  raise exception 'A late request cannot be attached to an existing signup' using errcode='22023';
 end if;
 return new;
end $$;
create trigger pcs_interclub_late_request_immutable before insert or update or delete on public.pcs_interclub_late_player_requests
 for each row execute function public.pcs_guard_interclub_late_request();

-- Keep the original late request pending even when registration closes before
-- the season's first day. Subsequent ordinary member edits cannot autoapprove
-- it or swap the requested player. Commissioner review remains authoritative.
do $patch$
declare routine regprocedure; definition text; anchor text; replacement text;
begin
 routine:=to_regprocedure('public.pcs_guard_interclub_pool_identity()');
 if routine is null then raise exception 'Pool identity guard missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:='  new.late_join:=new.created_at>=((season.details->>''start_date'')::date::timestamp at time zone (season.details->>''timezone''));';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Pool late-join guard changed'; end if;
 replacement:=anchor||$body$
  if exists(select 1 from public.pcs_interclub_late_player_requests r where r.member_id=new.id
   and r.season_id=new.season_id and r.club_id=new.club_id and r.player_id=new.player_id) then new.late_join:=true; end if;$body$;
 definition:=replace(definition,anchor,replacement);
 anchor:='  new.late_join:=old.late_join;';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Pool immutable late-join guard changed'; end if;
 replacement:=anchor||$body$
  if new.player_id is distinct from old.player_id and exists(
   select 1 from public.pcs_interclub_late_player_requests where member_id=old.id) then
   raise exception 'A late request keeps its requested club player' using errcode='22023';
  end if;$body$;
 execute replace(definition,anchor,replacement);
end $patch$;

-- Approval may wait for the season lock across a roster deadline. Record the
-- actual approval instant, not the earlier transaction start, so late approval
-- cannot retrospectively enter an already-closed meet's eligibility snapshot.
do $patch$
declare routine regprocedure; definition text; anchor text;
begin
 routine:=to_regprocedure('public.pcs_review_interclub_pool_member(uuid,text,text,uuid,uuid,integer,boolean,text)');
 if routine is null then raise exception 'Pool approval routine missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:='approved_at=case when p_approve then now() end';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Pool approval timestamp changed'; end if;
 execute replace(definition,anchor,'approved_at=case when p_approve then clock_timestamp() end');
end $patch$;

create or replace function public.pcs_guard_interclub_pool_registration() returns trigger
language plpgsql security invoker set search_path=public as $$
begin
 if tg_op='INSERT' then
  if exists(select 1 from public.pcs_interclub_late_player_requests r where r.member_id=new.id
   and r.season_id=new.season_id and r.club_id=new.club_id and r.player_id=new.player_id) then
   perform public.pcs_require_interclub_registration_phase(new.season_id,'closed');
   if new.status<>'active' or new.approval_status<>'pending' or new.late_join is not true
    or new.approved_at is not null or new.approved_by is not null or new.consent_at is not null or new.email<>'' then
    raise exception 'Late player requests require commissioner approval and do not grant contact consent' using errcode='22023';
   end if;
  else
   perform public.pcs_require_interclub_registration_phase(new.season_id,'open');
  end if;
 elsif old.status='withdrawn' and new.status='active' then
  -- A prior request is not a reusable authority to restore a withdrawn player.
  perform public.pcs_require_interclub_registration_phase(new.season_id,'open');
 end if;
 return new;
end $$;

create function public.pcs_request_interclub_late_player(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_player_id bigint,p_divisions jsonb,p_reason text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; player public.players; member public.pcs_interclub_pool_members; member_id uuid:=gen_random_uuid();
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform public.pcs_require_interclub_registration_phase(p_season_id,'closed');
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_club_id and status='accepted') then
  raise exception 'Accept your club invitation first' using errcode='42501'; end if;
 if (((season.details->>'end_date')::date+1)::timestamp at time zone (season.details->>'timezone'))<=clock_timestamp() then
  raise exception 'This season has ended' using errcode='PT409'; end if;
 select * into player from public.players where id=p_player_id and club_id=p_club_id and active is true for share;
 if not found then raise exception 'Choose an active player from this club' using errcode='22023'; end if;
 if p_reason is null or length(trim(p_reason)) not between 1 and 500
  or jsonb_typeof(p_divisions) is distinct from 'array' or jsonb_array_length(p_divisions)>8
  or not (season.details->'divisions' @> p_divisions)
  or (select count(distinct value) from jsonb_array_elements_text(p_divisions))<>jsonb_array_length(p_divisions) then
  raise exception 'Choose season divisions and explain the late request' using errcode='22023'; end if;
 -- Any existing signup, including a withdrawn/rejected one, must be handled
 -- explicitly through its existing record; retries cannot create duplicates.
 if exists(select 1 from public.pcs_interclub_pool_members where season_id=p_season_id and club_id=p_club_id and
  (player_id=p_player_id or (player_id is null and lower(regexp_replace(trim(name),'\s+',' ','g'))=lower(regexp_replace(trim(player.name),'\s+',' ','g'))))) then
  raise exception 'This player already has a season signup. Reload the player pool.' using errcode='23505'; end if;
 if (select count(*) from public.pcs_interclub_pool_members where season_id=p_season_id and club_id=p_club_id)>=1000 then
  raise exception 'The season pool is full' using errcode='54000'; end if;
 insert into public.pcs_interclub_late_player_requests(member_id,season_id,club_id,player_id,requested_by,reason)
  values(member_id,p_season_id,p_club_id,p_player_id,p_actor_id,trim(p_reason));
 insert into public.pcs_interclub_pool_members(id,season_id,club_id,name,email,divisions,player_id,request_id,request_fingerprint,consent_at)
  values(member_id,p_season_id,p_club_id,player.name,'',p_divisions,p_player_id,gen_random_uuid(),'late-request:'||p_actor_id::text,null)
  returning * into member;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
  values(p_season_id,p_actor_id,p_club_id,'late_player_requested',jsonb_build_object('member_id',member.id,'player_id',p_player_id,
   'revision',member.revision,'reason',trim(p_reason)));
 return to_jsonb(member);
end $$;

revoke all on function public.pcs_guard_interclub_late_request(),
 public.pcs_request_interclub_late_player(uuid,text,text,uuid,bigint,jsonb,text) from public,anon,authenticated;
grant execute on function public.pcs_guard_interclub_late_request(),
 public.pcs_request_interclub_late_player(uuid,text,text,uuid,bigint,jsonb,text) to service_role;
notify pgrst,'reload schema';
commit;
