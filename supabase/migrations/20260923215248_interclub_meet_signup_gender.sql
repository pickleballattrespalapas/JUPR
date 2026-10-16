begin;

-- Gender declarations belong to this meet, not the shared club profile.
-- A separate admin decision assigns a lineup place without changing the declaration.
alter table public.pcs_interclub_meet_signups
 add column declared_gender text check (declared_gender in ('female','male','non_binary','prefer_not_to_say')),
 add column reviewed_gender text check (reviewed_gender in ('female','male'));

create function public.pcs_interclub_meet_gender(p_season_id uuid,p_meet_id uuid,p_club_id text,p_player_id bigint,p_profile_gender text)
returns text language sql stable security invoker set search_path=public as $$
 select coalesce((
  select case when s.declared_gender is null then coalesce(s.reviewed_gender,public.pcs_interclub_gender(p_profile_gender))
   else coalesce(s.reviewed_gender,public.pcs_interclub_gender(s.declared_gender)) end
  from public.pcs_interclub_meet_signups s
  where s.season_id=p_season_id and s.meet_id=p_meet_id and s.club_id=p_club_id and s.player_id=p_player_id and s.status='active'
 ),public.pcs_interclub_gender(p_profile_gender))
$$;
revoke all on function public.pcs_interclub_meet_gender(uuid,uuid,text,bigint,text) from public,anon,authenticated;
grant execute on function public.pcs_interclub_meet_gender(uuid,uuid,text,bigint,text) to service_role;

CREATE OR REPLACE FUNCTION public.pcs_lock_interclub_meet_eligibility(p_meet_id uuid)
 RETURNS void
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
declare meet public.pcs_interclub_meets;
begin
 select * into meet from public.pcs_interclub_meets where id=p_meet_id;
 if not found then raise exception 'Meet unavailable' using errcode='P0002'; end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||meet.season_id::text,0));
 if meet.roster_deadline>now() then return; end if;
 insert into public.pcs_interclub_meet_eligibility_snapshots(season_id,meet_id,deadline,entry_id,club_id,player_id,rating,gender)
 select e.season_id,meet.id,meet.roster_deadline,e.id,e.club_id,e.player_id,
 public.pcs_interclub_rating_at(e.season_id,e.id,meet.roster_deadline),public.pcs_interclub_meet_gender(e.season_id,meet.id,e.club_id,p.id,p.gender)
 from public.pcs_interclub_entries e join public.players p on p.id=e.player_id and p.club_id=e.club_id
 join public.pcs_interclub_pool_members m on m.id=e.pool_member_id
 where e.season_id=meet.season_id and meet.club_ids ? e.club_id and e.entered_at<=meet.roster_deadline
 and m.status='active' and m.approval_status='approved' and m.approved_at<=meet.roster_deadline
 and public.pcs_interclub_rating_at(e.season_id,e.id,meet.roster_deadline) is not null
 on conflict(meet_id,deadline,entry_id) do nothing;
end $function$;

CREATE OR REPLACE FUNCTION public.pcs_interclub_meet_player_ratings(p_season_id uuid, p_meet_id uuid, p_club_id text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
declare meet public.pcs_interclub_meets; result jsonb;
begin
 perform public.pcs_require_interclub_registration_phase(p_season_id,'closed');
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id and club_ids ? p_club_id;
 if not found then raise exception 'Meet unavailable for this club' using errcode='P0002'; end if;
 perform public.pcs_lock_interclub_meet_eligibility(p_meet_id);
 if meet.roster_deadline<=now() then
  select coalesce(jsonb_agg(jsonb_build_object('entry_id',entry_id,'player_id',player_id,'eligibility_rating',rating,'gender',gender,'rating_deadline',deadline,'rating_locked',true)),'[]') into result
   from public.pcs_interclub_meet_eligibility_snapshots where meet_id=p_meet_id and deadline=meet.roster_deadline and club_id=p_club_id;
 else
  select coalesce(jsonb_agg(jsonb_build_object('entry_id',e.id,'player_id',e.player_id,'eligibility_rating',public.pcs_interclub_rating_at(p_season_id,e.id,now()),
   'gender',public.pcs_interclub_meet_gender(p_season_id,p_meet_id,p_club_id,p.id,p.gender),'rating_deadline',meet.roster_deadline,'rating_locked',false)),'[]') into result
  from public.pcs_interclub_entries e join public.pcs_interclub_pool_members m on m.id=e.pool_member_id
  join public.players p on p.id=e.player_id and p.club_id=e.club_id
  where e.season_id=p_season_id and e.club_id=p_club_id and m.status='active' and m.approval_status='approved';
 end if;
 return result;
end $function$;

CREATE OR REPLACE FUNCTION public.pcs_save_interclub_meet_roster_before_signups(p_actor_id uuid, p_actor_email text, p_club_id text, p_season_id uuid, p_meet_id uuid, p_meet_revision integer, p_team_id uuid, p_revision integer, p_name text, p_division text, p_player_ids bigint[], p_withdraw boolean DEFAULT false)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
declare meet public.pcs_interclub_meets; season public.pcs_interclub_seasons; team public.pcs_interclub_teams; player public.players; entry public.pcs_interclub_entries;
 saved public.pcs_interclub_roster_versions; lineup jsonb:='[]'; problems jsonb:='[]'; rule jsonb; old_team jsonb;
 women integer:=0; men integer:=0; gender_value text; team_created timestamptz; member public.pcs_interclub_pool_members; eligibility_rating numeric; eligibility_deadline timestamptz;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 perform public.pcs_require_interclub_registration_phase(p_season_id,'closed');
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_club_id and status='accepted') then
  raise exception 'Accept the club invitation first' using errcode='42501'; end if;
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id for update;
 if not found then raise exception 'Meet unavailable' using errcode='P0002'; end if;
 if not (meet.club_ids ? p_club_id) then raise exception 'This club is not scheduled for this meet' using errcode='42501'; end if;
 if meet.revision<>p_meet_revision or meet.starts_at<=now() then raise exception 'Meet changed or has started' using errcode='PT409'; end if;
 select * into team from public.pcs_interclub_teams where id=p_team_id for update;
 if found then
  if team.season_id<>p_season_id or team.club_id<>p_club_id or team.meet_id is distinct from p_meet_id then raise exception 'Only the represented club can edit its team' using errcode='42501'; end if;
  if team.revision<>p_revision then raise exception 'Roster changed' using errcode='PT409'; end if;
  if not p_withdraw and team.division<>p_division then raise exception 'A saved team keeps its division' using errcode='22023'; end if;
  team_created:=team.created_at;
 elsif p_revision<>0 or p_withdraw then raise exception 'Roster changed' using errcode='PT409';
 else team_created:=now(); end if;
 old_team:=to_jsonb(team);
 eligibility_deadline:=least(meet.roster_deadline,now());
 perform public.pcs_lock_interclub_meet_eligibility(p_meet_id);
 if p_withdraw then
  if team.withdrawn then raise exception 'Team already withdrawn' using errcode='PT409'; end if;
  select * into saved from public.pcs_interclub_roster_versions where team_id=p_team_id and revision=p_revision;
  update public.pcs_interclub_teams set revision=revision+1,withdrawn=true,updated_at=now() where id=p_team_id returning * into team;
  delete from public.pcs_interclub_team_players where team_id=p_team_id;
  insert into public.pcs_interclub_roster_versions(team_id,revision,name,roster,issues,status,late_change)
  values(p_team_id,team.revision,team.name,saved.roster,saved.issues,'withdrawn',now()>meet.roster_deadline) returning * into saved;
 else
  rule:=season.rules->p_division;
  if rule is null or length(trim(p_name)) not between 1 and 80 or p_name is null
   or p_player_ids is null or cardinality(p_player_ids) not in (2,4) or (select count(distinct id) from unnest(p_player_ids) id)<>cardinality(p_player_ids) then
   raise exception 'Choose a division, team name and two or four different players' using errcode='22023'; end if;
  if cardinality(p_player_ids)=2 and meet.competition_phase<>'regular' then raise exception 'Championship and qualifying matchups require four players' using errcode='22023'; end if;
  -- Lock all selected source rows in a stable order. Cross-club and inactive
  -- players are hard errors, never eligibility exceptions.
  for player in select * from public.players where id=any(p_player_ids) order by id for share loop
   if player.club_id<>p_club_id or player.active is not true then raise exception 'Choose active players from this club' using errcode='22023'; end if;
   select * into member from public.pcs_interclub_pool_members where season_id=p_season_id and club_id=p_club_id and player_id=player.id and status='active' and approval_status='approved';
   if not found then raise exception 'Each player needs an approved season interest signup linked to this club player' using errcode='22023'; end if;
   if exists(select 1 from public.pcs_interclub_represented_players where season_id=p_season_id and identity_key=member.identity_key and club_id<>p_club_id) then raise exception 'Player represents another club this season' using errcode='22023'; end if;
   select * into entry from public.pcs_interclub_entries where season_id=p_season_id and club_id=p_club_id and player_id=player.id and pool_member_id=member.id;
   if not found then raise exception 'Season rating seed unavailable' using errcode='22023'; end if;
   if meet.roster_deadline<=now() then
    select rating into eligibility_rating from public.pcs_interclub_meet_eligibility_snapshots where meet_id=p_meet_id and deadline=meet.roster_deadline and entry_id=entry.id;
   else eligibility_rating:=public.pcs_interclub_rating_at(p_season_id,entry.id,now()); end if;
   if eligibility_rating is null then raise exception 'Player was not eligible at the roster deadline' using errcode='22023'; end if;
   if not public.pcs_interclub_rating_in_division(eligibility_rating,p_division) then raise exception 'Choose a division whose upper rating limit exceeds the current interclub rating' using errcode='22023'; end if;
   if meet.roster_deadline<=now() then
    select gender into gender_value from public.pcs_interclub_meet_eligibility_snapshots where meet_id=p_meet_id and deadline=meet.roster_deadline and entry_id=entry.id;
   else gender_value:=public.pcs_interclub_meet_gender(p_season_id,p_meet_id,p_club_id,player.id,player.gender); end if;
   if gender_value='female' then women:=women+1; elsif gender_value='male' then men:=men+1; end if;
   lineup:=lineup||jsonb_build_array(jsonb_build_object('entry_id',entry.id,'player_id',player.id::text,'name',player.name,'starting_rating',entry.starting_rating,'eligibility_rating',eligibility_rating,'rating_deadline',meet.roster_deadline,'rating_locked',meet.roster_deadline<=now(),'gender',gender_value));
  end loop;
  if jsonb_array_length(lineup)<>cardinality(p_player_ids) then raise exception 'Choose available players from this club' using errcode='22023'; end if;
  if (cardinality(p_player_ids)=4 and (women<>2 or men<>2)) or (cardinality(p_player_ids)=2 and women+men<>2) then raise exception 'A full team needs two women and two men; a partial team needs two players with known gender for its remaining pairing' using errcode='22023'; end if;
  if team_created>meet.roster_deadline then problems:=problems||jsonb_build_array(jsonb_build_object('code','late_new_team','message','This team was first entered after this meet’s roster deadline.')); end if;
  if exists(select 1 from public.pcs_interclub_team_players where meet_id=p_meet_id and season_id=p_season_id and club_id=p_club_id and player_id=any(p_player_ids) and team_id<>p_team_id) then
   raise exception 'A player already belongs to another team in this meet and division' using errcode='23505'; end if;
  insert into public.pcs_interclub_teams(id,season_id,club_id,division,name,revision,meet_id)
  values(p_team_id,p_season_id,p_club_id,p_division,trim(p_name),1,p_meet_id)
  on conflict(id) do update set name=excluded.name,revision=pcs_interclub_teams.revision+1,withdrawn=false,updated_at=now() returning * into team;
  delete from public.pcs_interclub_team_players where team_id=p_team_id;
  insert into public.pcs_interclub_team_players(team_id,season_id,club_id,division,player_id,meet_id)
  select p_team_id,p_season_id,p_club_id,p_division,unnest(p_player_ids),p_meet_id;
  insert into public.pcs_interclub_roster_versions(team_id,revision,name,roster,issues,status,late_change)
  values(p_team_id,team.revision,team.name,lineup,problems,case when jsonb_array_length(problems)=0 then 'eligible' else 'needs_exception' end,now()>meet.roster_deadline) returning * into saved;
 end if;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,case when p_withdraw then 'withdraw_team' else 'submit_roster' end,jsonb_build_object('meet_id',p_meet_id,'before',old_team,'team',to_jsonb(team),'roster',to_jsonb(saved)));
 return jsonb_build_object('team',to_jsonb(team),'roster',to_jsonb(saved));
end $function$;

CREATE OR REPLACE FUNCTION public.pcs_reconcile_interclub_meet_signups(p_season_id uuid, p_club_id text, p_meet_id uuid)
 RETURNS void
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
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
  select s.id,p.name,public.pcs_interclub_meet_gender(p_season_id,p_meet_id,p_club_id,p.id,p.gender) as gender,
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
  reason=case when c.problem<>'' then c.problem when c.gender not in ('female','male') then 'An admin will review your registration and confirm your lineup placement.'
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
end $function$;

CREATE OR REPLACE FUNCTION public.pcs_interclub_meet_signup_action(p_action text, p_payload jsonb, p_actor_id uuid DEFAULT NULL::uuid, p_actor_email text DEFAULT NULL::text, p_requester_hash text DEFAULT NULL::text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
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
   if p_payload->>'gender' is not null and p_payload->>'gender' not in ('female','male','non_binary','prefer_not_to_say') then
    raise exception 'Choose Woman, Man, Non-binary or Prefer not to say.' using errcode='PT422'; end if;
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
   insert into public.pcs_interclub_meet_signups(season_id,club_id,meet_id,member_id,player_id,division,name,email,request_id,fingerprint,declared_gender)
   values(sid,cid,mid,member.id,player.id,p_payload->>'division',player.name,coalesce(p_payload->>'email',''),(p_payload->>'request_id')::uuid,p_payload->>'fingerprint',p_payload->>'gender')
   on conflict(season_id,club_id,meet_id,player_id) do update set member_id=excluded.member_id,division=excluded.division,name=excluded.name,email=excluded.email,
    status='active',admin_promoted=false,declared_gender=excluded.declared_gender,reviewed_gender=null,registered_at=clock_timestamp(),registration_order=default,request_id=excluded.request_id,fingerprint=excluded.fingerprint,token_nonce=gen_random_uuid(),revision=pcs_interclub_meet_signups.revision+1
   returning * into signup;
  elsif p_action in ('withdraw','remove','promote','review_gender') then
   select * into signup from public.pcs_interclub_meet_signups where id=(p_payload->>'id')::uuid and season_id=sid and club_id=cid and meet_id=mid for update;
   if not found or (p_actor_id is null and signup.token_nonce is distinct from (p_payload->>'nonce')::uuid) then raise exception 'Private signup unavailable' using errcode='P0002'; end if;
   if signup.revision is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Signup changed. Reload before continuing.' using errcode='PT409'; end if;
   if p_action='review_gender' then
    if p_actor_id is null then raise exception 'Administrator required' using errcode='42501'; end if;
    if p_payload->>'lineup_gender' is null or p_payload->>'lineup_gender' not in ('female','male') then
     raise exception 'Choose a women’s or men’s lineup place.' using errcode='PT422'; end if;
    if signup.status<>'active' or signup.placement<>'review' or signup.gender<>'unknown' then
     raise exception 'This signup is not waiting for gender review. Reload before continuing.' using errcode='PT409'; end if;
    update public.pcs_interclub_meet_signups set reviewed_gender=p_payload->>'lineup_gender',revision=revision+1 where id=signup.id;
   elsif p_action='promote' then
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
 values(sid,actor,cid,'meet_signup_'||p_action,jsonb_build_object('meet_id',mid,'signup_id',signup.id,'actor_kind',case when p_actor_id is null then 'player' else 'admin' end,'declared_gender',p_payload->>'gender','lineup_gender',p_payload->>'lineup_gender'));
 select * into signup from public.pcs_interclub_meet_signups where id=signup.id;
 return jsonb_build_object('entry',case when signup.id is null then null else to_jsonb(signup) end);
end $function$;

notify pgrst,'reload schema';
commit;
