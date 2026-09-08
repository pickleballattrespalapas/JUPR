begin;

-- Registration is an immutable snapshot of a planning draft. Later planning
-- edits cannot silently change the terms clubs accepted.
create table public.pcs_interclub_seasons (
 id uuid primary key references public.pcs_interclub_drafts(id),
 organizer_club_id text not null references public.clubs(id),
 source_revision integer not null check(source_revision>0),
 details jsonb not null check(jsonb_typeof(details)='object'),
 rules jsonb not null check(jsonb_typeof(rules)='object'),
 roster_deadline timestamptz not null,
 opened_at timestamptz not null default now()
);
create index pcs_interclub_seasons_organizer_idx on public.pcs_interclub_seasons(organizer_club_id,opened_at desc);
create table public.pcs_interclub_participations (
 season_id uuid not null references public.pcs_interclub_seasons(id),
 club_id text not null references public.clubs(id),
 status text not null default 'invited' check(status in ('invited','accepted','declined','cancelled')),
 revision integer not null default 1 check(revision>0),
 updated_at timestamptz not null default now(),
 primary key(season_id,club_id)
);
create index pcs_interclub_participations_club_idx on public.pcs_interclub_participations(club_id,updated_at desc);

-- The composite FK enforces represented-club ownership even below the API.
alter table public.players add constraint players_club_id_id_key unique(club_id,id);
create table public.pcs_interclub_entries (
 id uuid not null default gen_random_uuid() unique,
 season_id uuid not null,
 club_id text not null,
 player_id bigint not null,
 starting_rating numeric not null check(starting_rating>0),
 entered_at timestamptz not null default now(),
 primary key(season_id,club_id,player_id),
 foreign key(season_id,club_id) references public.pcs_interclub_participations(season_id,club_id),
 foreign key(club_id,player_id) references public.players(club_id,id)
);
create index pcs_interclub_entries_player_idx on public.pcs_interclub_entries(club_id,player_id);
create table public.pcs_interclub_teams (
 id uuid primary key,
 season_id uuid not null,
 club_id text not null,
 division text not null,
 name text not null check(length(trim(name)) between 1 and 80),
 revision integer not null check(revision>0),
 withdrawn boolean not null default false,
 created_at timestamptz not null default now(),
 updated_at timestamptz not null default now(),
 unique(id,season_id,club_id,division),
 foreign key(season_id,club_id) references public.pcs_interclub_participations(season_id,club_id)
);
create index pcs_interclub_teams_participation_idx on public.pcs_interclub_teams(season_id,club_id);
create unique index pcs_interclub_team_name_idx on public.pcs_interclub_teams(season_id,club_id,division,lower(name)) where not withdrawn;
create table public.pcs_interclub_roster_versions (
 team_id uuid not null references public.pcs_interclub_teams(id),
 revision integer not null check(revision>0),
 name text not null,
 roster jsonb not null check(jsonb_typeof(roster)='array' and jsonb_array_length(roster)=4),
 issues jsonb not null check(jsonb_typeof(issues)='array'),
 status text not null check(status in ('eligible','needs_exception','exception_approved','exception_denied','withdrawn')),
 late_change boolean not null,
 submitted_at timestamptz not null default now(),
 decision_reason text,
 decided_at timestamptz,
 primary key(team_id,revision)
);
-- Only current teams reserve a player's place within a club/division. Earlier
-- lineups remain in immutable roster versions for later meet snapshots.
create table public.pcs_interclub_team_players (
 team_id uuid not null,
 season_id uuid not null,
 club_id text not null,
 division text not null,
 player_id bigint not null,
 primary key(season_id,club_id,division,player_id),
 foreign key(team_id,season_id,club_id,division) references public.pcs_interclub_teams(id,season_id,club_id,division),
 foreign key(season_id,club_id,player_id) references public.pcs_interclub_entries(season_id,club_id,player_id)
);
create index pcs_interclub_team_players_team_idx on public.pcs_interclub_team_players(team_id);
create index pcs_interclub_team_players_entry_idx on public.pcs_interclub_team_players(season_id,club_id,player_id);
create table public.pcs_interclub_registration_audit (
 id bigint generated always as identity primary key,
 season_id uuid not null references public.pcs_interclub_seasons(id),
 actor_id uuid not null,
 actor_club_id text not null,
 action text not null,
 details jsonb not null,
 created_at timestamptz not null default now()
);
create index pcs_interclub_registration_audit_season_idx on public.pcs_interclub_registration_audit(season_id,created_at);

create view public.pcs_interclub_current_rosters with (security_invoker=true) as
 select t.*,r.roster,r.issues,r.status,r.late_change,r.submitted_at,r.decision_reason,r.decided_at
 from public.pcs_interclub_teams t join public.pcs_interclub_roster_versions r on r.team_id=t.id and r.revision=t.revision;
revoke all on public.pcs_interclub_current_rosters from public,anon,authenticated;
grant select on public.pcs_interclub_current_rosters to service_role;

alter table public.pcs_interclub_seasons enable row level security;
alter table public.pcs_interclub_participations enable row level security;
alter table public.pcs_interclub_entries enable row level security;
alter table public.pcs_interclub_teams enable row level security;
alter table public.pcs_interclub_roster_versions enable row level security;
alter table public.pcs_interclub_team_players enable row level security;
alter table public.pcs_interclub_registration_audit enable row level security;
revoke all on public.pcs_interclub_seasons,public.pcs_interclub_participations,public.pcs_interclub_entries,
 public.pcs_interclub_teams,public.pcs_interclub_roster_versions,public.pcs_interclub_team_players,public.pcs_interclub_registration_audit from public,anon,authenticated;
grant all on public.pcs_interclub_seasons,public.pcs_interclub_participations,public.pcs_interclub_entries,
 public.pcs_interclub_teams,public.pcs_interclub_roster_versions,public.pcs_interclub_team_players,public.pcs_interclub_registration_audit to service_role;
grant usage,select on sequence public.pcs_interclub_registration_audit_id_seq to service_role;

create function public.pcs_require_interclub_admin(p_actor_id uuid,p_actor_email text,p_club_id text)
returns void language plpgsql security invoker set search_path=public as $$
begin
 -- Lock order for registration writes: actor's staff lock, season lock, rows.
 perform pg_advisory_xact_lock(hashtextextended('pcs-staff:'||p_club_id,0));
 perform 1 from public.admin_role_assignments where club_id=p_club_id and email=lower(trim(p_actor_email))
  and (user_id is null or user_id=p_actor_id) and role in ('super_admin','administrator','club_owner')
  and revoked_at is null and (expires_at is null or expires_at>now()) for share;
 if not found or p_actor_id is null then raise exception 'Club administrator required' using errcode='42501'; end if;
end $$;

create function public.pcs_open_interclub_registration(p_actor_id uuid,p_actor_email text,p_club_id text,
 p_season_id uuid,p_revision integer,p_rules jsonb,p_deadline timestamptz)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare draft public.pcs_interclub_drafts; season public.pcs_interclub_seasons; division text; rule jsonb; target_club text;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if found then
  if season.organizer_club_id<>p_club_id then raise exception 'Organizer required' using errcode='42501'; end if;
  if season.source_revision=p_revision and season.rules=p_rules and season.roster_deadline=p_deadline then return to_jsonb(season); end if;
  raise exception 'Registration already open with different settings' using errcode='40001';
 end if;
 select * into draft from public.pcs_interclub_drafts where id=p_season_id for share;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if draft.organizer_club_id<>p_club_id then raise exception 'Organizer required' using errcode='42501'; end if;
 if draft.revision<>p_revision then raise exception 'Draft changed' using errcode='40001'; end if;
 if p_deadline is null or p_deadline<=now() or p_deadline>=(((draft.draft->>'end_date')::date+1)::timestamp at time zone (draft.draft->>'timezone'))
  or jsonb_array_length(draft.draft->'club_ids')<2 or p_rules is null or jsonb_typeof(p_rules)<>'object'
  or (select count(*) from jsonb_object_keys(p_rules))<>jsonb_array_length(draft.draft->'divisions') then
  raise exception 'Check clubs, rules and future roster deadline' using errcode='22023';
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
 insert into public.pcs_interclub_seasons(id,organizer_club_id,source_revision,details,rules,roster_deadline)
 values(p_season_id,p_club_id,p_revision,draft.draft,p_rules,p_deadline) returning * into season;
 for target_club in select jsonb_array_elements_text(draft.draft->'club_ids') loop
  insert into public.pcs_interclub_participations(season_id,club_id) values(p_season_id,target_club);
 end loop;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'open',to_jsonb(season));
 return to_jsonb(season);
end $$;

create function public.pcs_interclub_participation(p_actor_id uuid,p_actor_email text,p_club_id text,
 p_season_id uuid,p_target_club_id text,p_revision integer,p_action text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; participation public.pcs_interclub_participations; old_state jsonb; next_status text;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if p_action in ('accept','decline') then
  if p_target_club_id is distinct from p_club_id then raise exception 'Club must respond for itself' using errcode='42501'; end if;
 elsif p_action in ('cancel','reinvite') then
  if season.organizer_club_id<>p_club_id then raise exception 'Organizer required' using errcode='42501'; end if;
 else raise exception 'Invalid response' using errcode='22023'; end if;
 select * into participation from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_target_club_id for update;
 if not found then raise exception 'Invitation unavailable' using errcode='P0002'; end if;
 if participation.revision<>p_revision then raise exception 'Participation changed' using errcode='40001'; end if;
 if participation.status='accepted' or (p_action in ('accept','decline','cancel') and participation.status<>'invited')
  or (p_action='reinvite' and participation.status not in ('cancelled','declined')) then
  raise exception 'Invitation response no longer available' using errcode='40001'; end if;
 old_state:=to_jsonb(participation);
 next_status:=case p_action when 'accept' then 'accepted' when 'decline' then 'declined' when 'cancel' then 'cancelled' else 'invited' end;
 update public.pcs_interclub_participations set status=next_status,revision=revision+1,updated_at=now()
 where season_id=p_season_id and club_id=p_target_club_id returning * into participation;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,p_action,jsonb_build_object('before',old_state,'after',to_jsonb(participation)));
 return to_jsonb(participation);
end $$;

create function public.pcs_save_interclub_roster(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_team_id uuid,p_revision integer,p_name text,p_division text,p_player_ids bigint[],p_withdraw boolean default false)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; team public.pcs_interclub_teams; player public.players; entry public.pcs_interclub_entries;
 saved public.pcs_interclub_roster_versions; lineup jsonb:='[]'; problems jsonb:='[]'; rule jsonb; old_team jsonb;
 women integer:=0; men integer:=0; gender_value text; team_created timestamptz;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=p_club_id and status='accepted') then
  raise exception 'Accept the club invitation first' using errcode='42501'; end if;
 select * into team from public.pcs_interclub_teams where id=p_team_id for update;
 if found then
  if team.season_id<>p_season_id or team.club_id<>p_club_id then raise exception 'Only the represented club can edit its team' using errcode='42501'; end if;
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
  values(p_team_id,team.revision,team.name,saved.roster,saved.issues,'withdrawn',now()>season.roster_deadline) returning * into saved;
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
  if team_created>season.roster_deadline then problems:=problems||jsonb_build_array(jsonb_build_object('code','late_new_team','message','This team was first entered after the roster deadline.')); end if;
  if exists(select 1 from public.pcs_interclub_team_players where season_id=p_season_id and club_id=p_club_id and division=p_division and player_id=any(p_player_ids) and team_id<>p_team_id) then
   raise exception 'A player already belongs to another team in this division' using errcode='23505'; end if;
  insert into public.pcs_interclub_teams(id,season_id,club_id,division,name,revision)
  values(p_team_id,p_season_id,p_club_id,p_division,trim(p_name),1)
  on conflict(id) do update set name=excluded.name,revision=pcs_interclub_teams.revision+1,withdrawn=false,updated_at=now() returning * into team;
  delete from public.pcs_interclub_team_players where team_id=p_team_id;
  insert into public.pcs_interclub_team_players(team_id,season_id,club_id,division,player_id)
  select p_team_id,p_season_id,p_club_id,p_division,unnest(p_player_ids);
  insert into public.pcs_interclub_roster_versions(team_id,revision,name,roster,issues,status,late_change)
  values(p_team_id,team.revision,team.name,lineup,problems,case when jsonb_array_length(problems)=0 then 'eligible' else 'needs_exception' end,now()>season.roster_deadline) returning * into saved;
 end if;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,case when p_withdraw then 'withdraw_team' else 'submit_roster' end,jsonb_build_object('before',old_team,'team',to_jsonb(team),'roster',to_jsonb(saved)));
 return jsonb_build_object('team',to_jsonb(team),'roster',to_jsonb(saved));
end $$;

create function public.pcs_review_interclub_roster(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_team_id uuid,p_revision integer,p_approve boolean,p_reason text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare team public.pcs_interclub_teams; saved public.pcs_interclub_roster_versions;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then
  raise exception 'Only organizer administrators decide eligibility exceptions' using errcode='42501'; end if;
 if p_reason is null or length(trim(p_reason)) not between 1 and 500 or p_approve is null then raise exception 'Explain the eligibility decision' using errcode='22023'; end if;
 select * into team from public.pcs_interclub_teams where id=p_team_id and season_id=p_season_id for update;
 if not found then raise exception 'Team unavailable' using errcode='P0002'; end if;
 if team.revision<>p_revision or team.withdrawn then raise exception 'Roster changed' using errcode='40001'; end if;
 update public.pcs_interclub_roster_versions set status=case when p_approve then 'exception_approved' else 'exception_denied' end,
  decision_reason=trim(p_reason),decided_at=now() where team_id=p_team_id and revision=p_revision and status='needs_exception' returning * into saved;
 if not found then raise exception 'This roster no longer needs a decision' using errcode='40001'; end if;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'eligibility_decision',jsonb_build_object('team_id',p_team_id,'revision',p_revision,'approve',p_approve,'reason',trim(p_reason)));
 return jsonb_build_object('team',to_jsonb(team),'roster',to_jsonb(saved));
end $$;

revoke all on function public.pcs_require_interclub_admin(uuid,text,text),
 public.pcs_open_interclub_registration(uuid,text,text,uuid,integer,jsonb,timestamptz),
 public.pcs_interclub_participation(uuid,text,text,uuid,text,integer,text),
 public.pcs_save_interclub_roster(uuid,text,text,uuid,uuid,integer,text,text,bigint[],boolean),
 public.pcs_review_interclub_roster(uuid,text,text,uuid,uuid,integer,boolean,text) from public,anon,authenticated;
grant execute on function public.pcs_require_interclub_admin(uuid,text,text),
 public.pcs_open_interclub_registration(uuid,text,text,uuid,integer,jsonb,timestamptz),
 public.pcs_interclub_participation(uuid,text,text,uuid,text,integer,text),
 public.pcs_save_interclub_roster(uuid,text,text,uuid,uuid,integer,text,text,bigint[],boolean),
 public.pcs_review_interclub_roster(uuid,text,text,uuid,uuid,integer,boolean,text) to service_role;
commit;
