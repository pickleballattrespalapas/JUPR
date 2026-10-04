begin;

-- Clubs enroll for a season; each lineup belongs to one scheduled meet.
-- Existing season submissions remain archived with a NULL meet_id.
create table public.pcs_interclub_meets (
 id uuid primary key default gen_random_uuid(),
 season_id uuid not null references public.pcs_interclub_seasons(id),
 plan_index integer not null check(plan_index>=0),
 host_club_id text not null references public.clubs(id),
 club_ids jsonb not null check(jsonb_typeof(club_ids)='array' and jsonb_array_length(club_ids) between 2 and 4),
 starts_at timestamptz not null,
 duration_minutes integer not null check(duration_minutes between 30 and 180),
 courts integer not null check(courts between 1 and 100),
 roster_deadline timestamptz not null check(roster_deadline<=starts_at),
 revision integer not null default 1 check(revision>0),
 unique(id,season_id), unique(season_id,plan_index), check(id<>season_id)
);
create index pcs_interclub_meets_host_idx on public.pcs_interclub_meets(host_club_id);
create index pcs_interclub_meets_schedule_idx on public.pcs_interclub_meets(season_id,starts_at,id);
alter table public.pcs_interclub_meets enable row level security;
revoke all on public.pcs_interclub_meets from public,anon,authenticated;
grant all on public.pcs_interclub_meets to service_role;
alter table public.pcs_interclub_seasons alter column roster_deadline drop not null;
comment on column public.pcs_interclub_seasons.roster_deadline is 'Legacy season deadline only. Meet rosters use pcs_interclub_meets.roster_deadline.';

-- Backfill scheduled meets only, without assigning any old season lineup to one.
insert into public.pcs_interclub_meets(season_id,plan_index,host_club_id,club_ids,starts_at,duration_minutes,courts,roster_deadline)
 select s.id,(m.ordinality-1)::integer,m.value->>'host_club_id',m.value->'club_ids',(m.value->>'starts_at')::timestamptz,
 (m.value->>'duration_minutes')::integer,(m.value->>'courts')::integer,(m.value->>'starts_at')::timestamptz
 from public.pcs_interclub_seasons s cross join lateral jsonb_array_elements(s.details->'meets') with ordinality m;

alter table public.pcs_interclub_teams add column meet_id uuid,
 add column roster_scope_id uuid generated always as (coalesce(meet_id,season_id)) stored,
 add constraint pcs_interclub_teams_meet_fk foreign key(meet_id,season_id) references public.pcs_interclub_meets(id,season_id),
 add constraint pcs_interclub_teams_scope_key unique(id,roster_scope_id,season_id,club_id,division);
create index pcs_interclub_teams_meet_idx on public.pcs_interclub_teams(meet_id,season_id);
drop index public.pcs_interclub_team_name_idx;
create unique index pcs_interclub_team_name_idx on public.pcs_interclub_teams(roster_scope_id,club_id,division,lower(name)) where not withdrawn;
alter table public.pcs_interclub_team_players add column meet_id uuid,
 add column roster_scope_id uuid generated always as (coalesce(meet_id,season_id)) stored,
 drop constraint pcs_interclub_team_players_pkey,
 add primary key(roster_scope_id,club_id,division,player_id),
 add constraint pcs_interclub_team_players_scope_fk foreign key(team_id,roster_scope_id,season_id,club_id,division)
  references public.pcs_interclub_teams(id,roster_scope_id,season_id,club_id,division);
-- The original FK still protects archived NULL-meet rows; the new FK also
-- prevents a current player slot from claiming a different meet than its team.

create or replace view public.pcs_interclub_current_rosters with(security_invoker=true) as
 select t.id,t.season_id,t.club_id,t.division,t.name,t.revision,t.withdrawn,t.created_at,t.updated_at,
 r.roster,r.issues,r.status,r.late_change,r.submitted_at,r.decision_reason,r.decided_at,t.meet_id
 from public.pcs_interclub_teams t join public.pcs_interclub_roster_versions r on r.team_id=t.id and r.revision=t.revision;
create view public.pcs_interclub_meet_workspaces with(security_invoker=true) as
 select m.*,m.starts_at>now() as roster_open,
 m.starts_at>now() and not exists(select 1 from public.pcs_interclub_teams t where t.meet_id=m.id) as deadline_editable
 from public.pcs_interclub_meets m;
revoke all on public.pcs_interclub_current_rosters,public.pcs_interclub_meet_workspaces from public,anon,authenticated;
grant select on public.pcs_interclub_current_rosters,public.pcs_interclub_meet_workspaces to service_role;

create or replace function public.pcs_open_interclub_registration(p_actor_id uuid,p_actor_email text,p_club_id text,
 p_season_id uuid,p_revision integer,p_rules jsonb,p_deadline timestamptz)
returns jsonb language plpgsql security invoker set search_path=public as $$
begin
 raise exception 'Rosters now belong to individual meets. Reload and choose a meet.' using errcode='40001';
end $$;

create or replace function public.pcs_save_interclub_roster(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_team_id uuid,p_revision integer,p_name text,p_division text,p_player_ids bigint[],p_withdraw boolean default false)
returns jsonb language plpgsql security invoker set search_path=public as $$
begin
 raise exception 'Rosters now belong to individual meets. Reload and choose a meet.' using errcode='40001';
end $$;

create or replace function public.pcs_review_interclub_roster(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_team_id uuid,p_revision integer,p_approve boolean,p_reason text)
returns jsonb language plpgsql security invoker set search_path=public as $$
begin
 raise exception 'Rosters now belong to individual meets. Reload and choose a meet.' using errcode='40001';
end $$;

create function public.pcs_open_interclub_meet_registration(p_actor_id uuid,p_actor_email text,p_club_id text,
 p_season_id uuid,p_revision integer,p_rules jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare draft public.pcs_interclub_drafts; season public.pcs_interclub_seasons; division text; rule jsonb; target_club text;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if found then
  if season.organizer_club_id<>p_club_id then raise exception 'Organizer required' using errcode='42501'; end if;
  if season.source_revision=p_revision and season.rules=p_rules then return to_jsonb(season); end if;
  raise exception 'Registration already open with different settings' using errcode='40001';
 end if;
 select * into draft from public.pcs_interclub_drafts where id=p_season_id for share;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if draft.organizer_club_id<>p_club_id then raise exception 'Organizer required' using errcode='42501'; end if;
 if draft.revision<>p_revision then raise exception 'Draft changed' using errcode='40001'; end if;
 if now()>=(((draft.draft->>'end_date')::date+1)::timestamp at time zone (draft.draft->>'timezone'))
  or jsonb_array_length(draft.draft->'club_ids')<2 or p_rules is null or jsonb_typeof(p_rules)<>'object'
  or (select count(*) from jsonb_object_keys(p_rules))<>jsonb_array_length(draft.draft->'divisions') then
  raise exception 'Check clubs, rules and season dates' using errcode='22023';
 end if;
 for division in select jsonb_array_elements_text(draft.draft->'divisions') loop
  rule:=p_rules->division;
  if rule is null or jsonb_typeof(rule)<>'object'
   or (rule->>'min_rating')::numeric not between 1 and 7 or (rule->>'max_rating')::numeric not between 1 and 7
   or (rule->>'min_rating')::numeric>(rule->>'max_rating')::numeric
   or (rule->>'women_required')::integer not between 0 and 4 then
   raise exception 'Invalid division rules' using errcode='22023';
  end if;
 end loop;
 insert into public.pcs_interclub_seasons(id,organizer_club_id,source_revision,details,rules)
 values(p_season_id,p_club_id,p_revision,draft.draft,p_rules) returning * into season;
 for target_club in select jsonb_array_elements_text(draft.draft->'club_ids') loop
  insert into public.pcs_interclub_participations(season_id,club_id) values(p_season_id,target_club);
 end loop;
 insert into public.pcs_interclub_meets(season_id,plan_index,host_club_id,club_ids,starts_at,duration_minutes,courts,roster_deadline)
 select p_season_id,(m.ordinality-1)::integer,m.value->>'host_club_id',m.value->'club_ids',(m.value->>'starts_at')::timestamptz,
 (m.value->>'duration_minutes')::integer,(m.value->>'courts')::integer,(m.value->>'starts_at')::timestamptz
 from jsonb_array_elements(draft.draft->'meets') with ordinality m;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'open',to_jsonb(season));
 return to_jsonb(season);
end $$;

-- Only the organizer sets a meet's deadline. Once a roster is submitted,
-- retain that deadline so its eligibility and late status cannot change silently.
create function public.pcs_set_interclub_meet_deadline(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_meet_id uuid,p_revision integer,p_deadline timestamptz)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets; prior timestamptz;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then
  raise exception 'Only the organizer sets meet deadlines' using errcode='42501'; end if;
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id for update;
 if not found then raise exception 'Meet unavailable' using errcode='P0002'; end if;
 if meet.revision<>p_revision or meet.starts_at<=now()
  or exists(select 1 from public.pcs_interclub_teams where meet_id=p_meet_id) then
  raise exception 'Meet changed or rosters already submitted' using errcode='40001'; end if;
 if p_deadline is null or p_deadline<=now() or p_deadline>meet.starts_at then
  raise exception 'Choose a future deadline no later than this meet' using errcode='22023'; end if;
 prior:=meet.roster_deadline;
 update public.pcs_interclub_meets set roster_deadline=p_deadline,revision=revision+1 where id=p_meet_id returning * into meet;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'meet_deadline',jsonb_build_object('meet_id',p_meet_id,'before',prior,'after',p_deadline,'revision',meet.revision));
 return to_jsonb(meet);
end $$;

create function public.pcs_save_interclub_meet_roster(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_meet_id uuid,p_meet_revision integer,p_team_id uuid,p_revision integer,p_name text,p_division text,p_player_ids bigint[],p_withdraw boolean default false)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets; season public.pcs_interclub_seasons; team public.pcs_interclub_teams; player public.players; entry public.pcs_interclub_entries;
 saved public.pcs_interclub_roster_versions; lineup jsonb:='[]'; problems jsonb:='[]'; rule jsonb; old_team jsonb;
 women integer:=0; men integer:=0; gender_value text; team_created timestamptz;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_club_id and status='accepted') then
  raise exception 'Accept the club invitation first' using errcode='42501'; end if;
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id for update;
 if not found then raise exception 'Meet unavailable' using errcode='P0002'; end if;
 if not (meet.club_ids ? p_club_id) then raise exception 'This club is not scheduled for this meet' using errcode='42501'; end if;
 if meet.revision<>p_meet_revision or meet.starts_at<=now() then raise exception 'Meet changed or has started' using errcode='40001'; end if;
 select * into team from public.pcs_interclub_teams where id=p_team_id for update;
 if found then
  if team.season_id<>p_season_id or team.club_id<>p_club_id or team.meet_id is distinct from p_meet_id then raise exception 'Only the represented club can edit its team' using errcode='42501'; end if;
  if team.revision<>p_revision then raise exception 'Roster changed' using errcode='40001'; end if;
  if not p_withdraw and team.division<>p_division then raise exception 'A saved team keeps its division' using errcode='22023'; end if;
  team_created:=team.created_at;
 elsif p_revision<>0 or p_withdraw then raise exception 'Roster changed' using errcode='40001';
 else team_created:=now(); end if;
 old_team:=to_jsonb(team);
 if p_withdraw then
  if team.withdrawn then raise exception 'Team already withdrawn' using errcode='40001'; end if;
  select * into saved from public.pcs_interclub_roster_versions where team_id=p_team_id and revision=p_revision;
  update public.pcs_interclub_teams set revision=revision+1,withdrawn=true,updated_at=now() where id=p_team_id returning * into team;
  delete from public.pcs_interclub_team_players where team_id=p_team_id;
  insert into public.pcs_interclub_roster_versions(team_id,revision,name,roster,issues,status,late_change)
  values(p_team_id,team.revision,team.name,saved.roster,saved.issues,'withdrawn',now()>meet.roster_deadline) returning * into saved;
 else
  rule:=season.rules->p_division;
  if rule is null or length(trim(p_name)) not between 1 and 80 or p_name is null
   or p_player_ids is null or cardinality(p_player_ids)<>4 or (select count(distinct id) from unnest(p_player_ids) id)<>4 then
   raise exception 'Choose a division, team name and four different players' using errcode='22023'; end if;
  -- Lock all selected source rows in a stable order. Cross-club and inactive
  -- players are hard errors, never eligibility exceptions.
  for player in select * from public.players where id=any(p_player_ids) order by id for share loop
   if player.club_id<>p_club_id or player.active is not true then raise exception 'Choose active players from this club' using errcode='22023'; end if;
   select * into entry from public.pcs_interclub_entries where season_id=p_season_id and club_id=p_club_id and player_id=player.id;
   if not found then
    if player.rating is null or player.rating<=0 or player.rating::text in ('NaN','Infinity','-Infinity') then
     raise exception 'Set a starting club rating before submitting this player' using errcode='22023'; end if;
    insert into public.pcs_interclub_entries(season_id,club_id,player_id,starting_rating)
    values(p_season_id,p_club_id,player.id,player.rating/400.0) returning * into entry;
   end if;
   gender_value:=case lower(trim(coalesce(player.gender,''))) when 'female' then 'female' when 'f' then 'female' when 'woman' then 'female'
    when 'male' then 'male' when 'm' then 'male' when 'man' then 'male' else 'unknown' end;
   if gender_value='female' then women:=women+1; elsif gender_value='male' then men:=men+1; end if;
   lineup:=lineup||jsonb_build_array(jsonb_build_object('entry_id',entry.id,'player_id',player.id::text,'name',player.name,'starting_rating',entry.starting_rating,'gender',gender_value));
   if entry.starting_rating<(rule->>'min_rating')::numeric then problems:=problems||jsonb_build_array(jsonb_build_object('code','rating_below_minimum','message',player.name||' is below the minimum starting rating.')); end if;
   if entry.starting_rating>(rule->>'max_rating')::numeric then problems:=problems||jsonb_build_array(jsonb_build_object('code','rating_above_maximum','message',player.name||' is above the maximum starting rating.')); end if;
  end loop;
  if jsonb_array_length(lineup)<>4 then raise exception 'Choose four available players from this club' using errcode='22023'; end if;
  if (rule->>'women_required') is not null and (women<>(rule->>'women_required')::integer or men<>4-(rule->>'women_required')::integer) then
   problems:=problems||jsonb_build_array(jsonb_build_object('code','team_composition','message','This division requires '||(rule->>'women_required')||' women and '||(4-(rule->>'women_required')::integer)::text||' men. Check missing gender details in your club directory.')); end if;
  if team_created>meet.roster_deadline then problems:=problems||jsonb_build_array(jsonb_build_object('code','late_new_team','message','This team was first entered after this meet’s roster deadline.')); end if;
  if exists(select 1 from public.pcs_interclub_team_players where meet_id=p_meet_id and season_id=p_season_id and club_id=p_club_id and division=p_division and player_id=any(p_player_ids) and team_id<>p_team_id) then
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
end $$;

create function public.pcs_review_interclub_meet_roster(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_meet_id uuid,p_meet_revision integer,p_team_id uuid,p_revision integer,p_approve boolean,p_reason text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets; team public.pcs_interclub_teams; saved public.pcs_interclub_roster_versions;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then
  raise exception 'Only organizer administrators decide eligibility exceptions' using errcode='42501'; end if;
 if p_reason is null or length(trim(p_reason)) not between 1 and 500 or p_approve is null then raise exception 'Explain the eligibility decision' using errcode='22023'; end if;
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id for update;
 if not found then raise exception 'Meet unavailable' using errcode='P0002'; end if;
 if meet.revision<>p_meet_revision or meet.starts_at<=now() then raise exception 'Meet changed or has started' using errcode='40001'; end if;
 select * into team from public.pcs_interclub_teams where id=p_team_id and season_id=p_season_id and meet_id=p_meet_id for update;
 if not found then raise exception 'Team unavailable' using errcode='P0002'; end if;
 if team.revision<>p_revision or team.withdrawn then raise exception 'Roster changed' using errcode='40001'; end if;
 update public.pcs_interclub_roster_versions set status=case when p_approve then 'exception_approved' else 'exception_denied' end,
  decision_reason=trim(p_reason),decided_at=now() where team_id=p_team_id and revision=p_revision and status='needs_exception' returning * into saved;
 if not found then raise exception 'This roster no longer needs a decision' using errcode='40001'; end if;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'eligibility_decision',jsonb_build_object('meet_id',p_meet_id,'team_id',p_team_id,'revision',p_revision,'approve',p_approve,'reason',trim(p_reason)));
 return jsonb_build_object('team',to_jsonb(team),'roster',to_jsonb(saved));
end $$;

revoke all on function public.pcs_open_interclub_meet_registration(uuid,text,text,uuid,integer,jsonb),
 public.pcs_set_interclub_meet_deadline(uuid,text,text,uuid,uuid,integer,timestamptz),
 public.pcs_save_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,text,text,bigint[],boolean),
 public.pcs_review_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,boolean,text) from public,anon,authenticated;
grant execute on function public.pcs_open_interclub_meet_registration(uuid,text,text,uuid,integer,jsonb),
 public.pcs_set_interclub_meet_deadline(uuid,text,text,uuid,uuid,integer,timestamptz),
 public.pcs_save_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,text,text,bigint[],boolean),
 public.pcs_review_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,boolean,text) to service_role;
commit;
