begin;

-- Pool membership, represented-club identity and meet eligibility remain private.
alter table public.pcs_interclub_pool_members
 add column approval_status text not null default 'pending' check(approval_status in ('pending','approved','rejected')),
 add column late_join boolean not null default false,
 add column approved_at timestamptz,
 add column approved_by uuid,
 add column approval_reason text,
 add column identity_key text;
update public.pcs_interclub_pool_members m set
 identity_key=encode(sha256(convert_to(lower(regexp_replace(trim(m.name),'\s+',' ','g'))||':'||lower(trim(m.email)),'UTF8')),'hex'),
 late_join=m.created_at>=((s.details->>'start_date')::date::timestamp at time zone (s.details->>'timezone')),
 approval_status=case when m.player_id is not null and m.created_at<((s.details->>'start_date')::date::timestamp at time zone (s.details->>'timezone')) then 'approved' else 'pending' end,
 approved_at=case when m.player_id is not null and m.created_at<((s.details->>'start_date')::date::timestamp at time zone (s.details->>'timezone')) then m.created_at end
 from public.pcs_interclub_seasons s where s.id=m.season_id;
alter table public.pcs_interclub_pool_members alter column identity_key set not null;
alter table public.pcs_interclub_entries add column pool_member_id uuid references public.pcs_interclub_pool_members(id);
update public.pcs_interclub_entries e set pool_member_id=m.id from public.pcs_interclub_pool_members m
 where m.season_id=e.season_id and m.club_id=e.club_id and m.player_id=e.player_id and m.status='active';

-- Immutable first representation: correcting/deleting a score never authorizes a transfer.
create table public.pcs_interclub_represented_players (
 season_id uuid not null references public.pcs_interclub_seasons(id),identity_key text not null,
 club_id text not null references public.clubs(id),first_played_at timestamptz not null default now(),
 primary key(season_id,identity_key)
);
create table public.pcs_interclub_appearances (
 season_id uuid not null references public.pcs_interclub_seasons(id),meet_id uuid not null references public.pcs_interclub_meets(id),
 entry_id uuid not null references public.pcs_interclub_entries(id),club_id text not null,player_id bigint not null,
 division text not null,batch_id uuid not null,revision integer not null check(revision>0),
 phase text not null check(phase in ('regular','final','qualifier')),game_id text not null,
 primary key(batch_id,revision,game_id,entry_id),foreign key(club_id,player_id) references public.players(club_id,id)
);
create index pcs_interclub_appearances_eligibility_idx on public.pcs_interclub_appearances(season_id,club_id,entry_id,phase,division);
create table public.pcs_interclub_meet_eligibility_snapshots (
 season_id uuid not null references public.pcs_interclub_seasons(id),meet_id uuid not null references public.pcs_interclub_meets(id),
 deadline timestamptz not null,entry_id uuid not null references public.pcs_interclub_entries(id),
 club_id text not null,player_id bigint not null,rating numeric not null check(rating>0),
 rating_source text not null default 'interclub_at_deadline',gender text not null,
 created_at timestamptz not null default now(),primary key(meet_id,deadline,entry_id),
 foreign key(club_id,player_id) references public.players(club_id,id)
);
alter table public.pcs_interclub_represented_players enable row level security;
alter table public.pcs_interclub_appearances enable row level security;
alter table public.pcs_interclub_meet_eligibility_snapshots enable row level security;
revoke all on public.pcs_interclub_represented_players,public.pcs_interclub_appearances,public.pcs_interclub_meet_eligibility_snapshots from public,anon,authenticated;
grant all on public.pcs_interclub_represented_players,public.pcs_interclub_appearances,public.pcs_interclub_meet_eligibility_snapshots to service_role;

-- Rating pipeline replaces this fallback with its immutable as-known ledger lookup.
create or replace function public.pcs_interclub_rating_at(p_season_id uuid,p_entry_id uuid,p_cutoff timestamptz)
returns numeric language sql stable security invoker set search_path=public as $$
 select starting_rating from public.pcs_interclub_entries where season_id=p_season_id and id=p_entry_id and entered_at<=p_cutoff
$$;
create function public.pcs_interclub_gender(p_gender text) returns text language sql immutable as $$
 select case lower(trim(coalesce(p_gender,''))) when 'female' then 'female' when 'f' then 'female' when 'woman' then 'female' when 'women' then 'female'
 when 'male' then 'male' when 'm' then 'male' when 'man' then 'male' when 'men' then 'male' else 'unknown' end
$$;
create function public.pcs_interclub_rating_in_division(p_rating numeric,p_division text) returns boolean language plpgsql immutable as $$
begin
 if p_rating is null then return false; end if;
 if lower(p_division) in ('open','4.5/open') then return p_rating>=4.5; end if;
 if p_division !~ '^[2-6]\.[05]$' then return false; end if;
 return p_rating>=p_division::numeric and p_rating<p_division::numeric+0.5;
end $$;

create function public.pcs_guard_interclub_pool_identity() returns trigger language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; represented text; affected_meet uuid;
begin
 select * into season from public.pcs_interclub_seasons where id=new.season_id;
 if tg_op='INSERT' then
  new.identity_key:=encode(sha256(convert_to(lower(regexp_replace(trim(new.name),'\s+',' ','g'))||':'||lower(trim(new.email)),'UTF8')),'hex');
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
 if new.player_id is not null and not new.late_join and new.approval_status='pending' then
  new.approval_status:='approved';new.approved_at:=coalesce(new.approved_at,now());
 end if;
 if new.player_id is null and new.approval_status='approved' then new.approval_status:='pending';new.approved_at:=null; end if;
 select club_id into represented from public.pcs_interclub_represented_players where season_id=new.season_id and identity_key=new.identity_key;
 if new.status='active' and represented is not null and represented<>new.club_id then raise exception 'Player already represents another club this season' using errcode='22023'; end if;
 return new;
end $$;
create trigger pcs_interclub_pool_identity before insert or update on public.pcs_interclub_pool_members for each row execute function public.pcs_guard_interclub_pool_identity();

create function public.pcs_seed_approved_interclub_entry() returns trigger language plpgsql security invoker set search_path=public as $$
declare source_rating numeric; prior_entry public.pcs_interclub_entries; prior_identity text;
begin
 if new.player_id is not null and new.status='active' and new.approval_status='approved' then
  select * into prior_entry from public.pcs_interclub_entries where season_id=new.season_id and club_id=new.club_id and player_id=new.player_id;
  if prior_entry.pool_member_id is not null and prior_entry.pool_member_id<>new.id then
   select identity_key into prior_identity from public.pcs_interclub_pool_members where id=prior_entry.pool_member_id;
   if prior_identity is distinct from new.identity_key and (exists(select 1 from public.pcs_interclub_represented_players where season_id=new.season_id and identity_key=prior_identity) or exists(select 1 from public.pcs_interclub_meet_eligibility_snapshots where entry_id=prior_entry.id)) then
    raise exception 'An established season player identity cannot be reassigned' using errcode='22023';
   end if;
  end if;
  select rating/400.0 into source_rating from public.players where id=new.player_id and club_id=new.club_id and active is true;
  if source_rating is null or source_rating<=0 or source_rating::text in ('NaN','Infinity','-Infinity') then raise exception 'Set a club rating before approving this player' using errcode='22023'; end if;
  insert into public.pcs_interclub_entries(season_id,club_id,player_id,starting_rating,pool_member_id)
  values(new.season_id,new.club_id,new.player_id,source_rating,new.id)
  on conflict(season_id,club_id,player_id) do update set pool_member_id=excluded.pool_member_id;
 end if;
 return new;
end $$;
create trigger pcs_interclub_pool_seed after insert or update on public.pcs_interclub_pool_members for each row execute function public.pcs_seed_approved_interclub_entry();
insert into public.pcs_interclub_entries(season_id,club_id,player_id,starting_rating,pool_member_id)
 select m.season_id,m.club_id,m.player_id,p.rating/400.0,m.id from public.pcs_interclub_pool_members m join public.players p on p.id=m.player_id and p.club_id=m.club_id
 where m.approval_status='approved' and m.status='active' and p.active is true and p.rating>0
 on conflict(season_id,club_id,player_id) do update set pool_member_id=excluded.pool_member_id;

create function public.pcs_review_interclub_pool_member(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,p_member_id uuid,p_revision integer,p_approve boolean,p_reason text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare member public.pcs_interclub_pool_members;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then raise exception 'Organizer approval required' using errcode='42501'; end if;
 select * into member from public.pcs_interclub_pool_members where id=p_member_id and season_id=p_season_id for update;
 if not found then raise exception 'Member unavailable' using errcode='P0002'; end if;
 if member.revision<>p_revision then raise exception 'Member changed' using errcode='40001'; end if;
 if p_approve is null or p_reason is null or length(trim(p_reason)) not between 1 and 500 or (p_approve and (member.player_id is null or member.status<>'active')) then raise exception 'Link an active player and explain approval' using errcode='22023'; end if;
 update public.pcs_interclub_pool_members set approval_status=case when p_approve then 'approved' else 'rejected' end,
 approved_at=case when p_approve then now() end,approved_by=p_actor_id,approval_reason=trim(p_reason),revision=revision+1,updated_at=now() where id=p_member_id returning * into member;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'pool_approval',jsonb_build_object('member_id',member.id,'approve',p_approve,'reason',trim(p_reason),'revision',member.revision));
 return jsonb_build_object('id',member.id,'club_id',member.club_id,'name',member.name,'player_id',member.player_id,'revision',member.revision,'approval_status',member.approval_status,'late_join',member.late_join,'approval_reason',member.approval_reason);
end $$;

create function public.pcs_lock_interclub_meet_eligibility(p_meet_id uuid) returns void language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets;
begin
 select * into meet from public.pcs_interclub_meets where id=p_meet_id;
 if not found then raise exception 'Meet unavailable' using errcode='P0002'; end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||meet.season_id::text,0));
 if meet.roster_deadline>now() then return; end if;
 insert into public.pcs_interclub_meet_eligibility_snapshots(season_id,meet_id,deadline,entry_id,club_id,player_id,rating,gender)
 select e.season_id,meet.id,meet.roster_deadline,e.id,e.club_id,e.player_id,
 public.pcs_interclub_rating_at(e.season_id,e.id,meet.roster_deadline),public.pcs_interclub_gender(p.gender)
 from public.pcs_interclub_entries e join public.players p on p.id=e.player_id and p.club_id=e.club_id
 join public.pcs_interclub_pool_members m on m.id=e.pool_member_id
 where e.season_id=meet.season_id and meet.club_ids ? e.club_id and e.entered_at<=meet.roster_deadline
 and m.status='active' and m.approval_status='approved' and m.approved_at<=meet.roster_deadline
 and public.pcs_interclub_rating_at(e.season_id,e.id,meet.roster_deadline) is not null
 on conflict(meet_id,deadline,entry_id) do nothing;
end $$;

create function public.pcs_preserve_interclub_gender_snapshots() returns trigger language plpgsql security invoker set search_path=public as $$
declare affected_meet uuid;
begin
 if new.gender is distinct from old.gender then
  for affected_meet in select distinct m.id from public.pcs_interclub_entries e join public.pcs_interclub_meets m on m.season_id=e.season_id
   where e.player_id=old.id and e.club_id=old.club_id and m.club_ids ? old.club_id and m.roster_deadline<=now() order by m.id loop
   perform public.pcs_lock_interclub_meet_eligibility(affected_meet);
  end loop;
 end if;
 return new;
end $$;
create trigger pcs_interclub_gender_snapshot before update of gender on public.players for each row execute function public.pcs_preserve_interclub_gender_snapshots();

create function public.pcs_record_interclub_representation() returns trigger language plpgsql security invoker set search_path=public as $$
declare member public.pcs_interclub_pool_members; entry public.pcs_interclub_entries; represented text;
begin
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||new.season_id::text,0));
 select * into entry from public.pcs_interclub_entries where id=new.entry_id and season_id=new.season_id and club_id=new.club_id and player_id=new.player_id;
 if not found then raise exception 'Appearance must use represented club entry' using errcode='22023'; end if;
 select * into member from public.pcs_interclub_pool_members where id=entry.pool_member_id;
 if member.id is null then raise exception 'Approved pool member required' using errcode='22023'; end if;
 insert into public.pcs_interclub_represented_players(season_id,identity_key,club_id) values(new.season_id,member.identity_key,new.club_id) on conflict do nothing;
 select club_id into represented from public.pcs_interclub_represented_players where season_id=new.season_id and identity_key=member.identity_key;
 if represented<>new.club_id then raise exception 'Player represents another club this season' using errcode='22023'; end if;
 return new;
end $$;
create trigger pcs_interclub_appearance_club before insert or update on public.pcs_interclub_appearances for each row execute function public.pcs_record_interclub_representation();

-- Always validate actual game participants, including injury replacements. UUIDs
-- are season entry IDs; no contact details are returned to the organizer.
create function public.pcs_assert_interclub_competition_eligibility(p_season_id uuid,p_meet_id uuid,p_document jsonb,p_phase text)
returns void language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets; season public.pcs_interclub_seasons; encounter jsonb; pairing jsonb; game jsonb;
 lineup jsonb; side text; cid text; eid text; cutoff timestamptz; snap public.pcs_interclub_meet_eligibility_snapshots;
 entry public.pcs_interclub_entries; member public.pcs_interclub_pool_members; represented text; women integer; men integer;
begin
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id;
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if meet.id is null or p_phase not in ('regular','final','qualifier') then raise exception 'Meet unavailable' using errcode='22023'; end if;
 if meet.roster_deadline>now() then raise exception 'The roster deadline has not passed' using errcode='40001'; end if;
 perform public.pcs_lock_interclub_meet_eligibility(p_meet_id);
 for encounter in select value from jsonb_array_elements(p_document->'encounters') loop
  if not(season.details->'divisions' ? (encounter->>'division')) then raise exception 'Choose a season skill level' using errcode='22023'; end if;
  foreach side in array array['a','b'] loop
   cid:=encounter->>('club_'||side);
   if not(meet.club_ids ? cid) or not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=cid and status='accepted') then raise exception 'Club has not accepted this season or is not scheduled' using errcode='22023'; end if;
   if p_phase='final' and not exists(select 1 from public.pcs_interclub_appearances a where a.season_id=p_season_id and a.club_id=cid and a.phase='regular' and a.division=encounter->>'division') then raise exception 'Finalist club must have played this skill level in the regular season' using errcode='22023'; end if;
   for pairing in select value from jsonb_array_elements(encounter->'pairings') loop
    cutoff:=coalesce((pairing->>'eligibility_deadline')::timestamptz,meet.roster_deadline);
    if cutoff>meet.roster_deadline then raise exception 'Invalid eligibility deadline' using errcode='22023'; end if;
    -- Validate base lineups and each game's optional actual-player overrides.
    for game in select pairing union all select value from jsonb_array_elements(pairing->'games') loop
     lineup:=game->('players_'||side);
     if lineup is null or jsonb_array_length(lineup)=0 then continue; end if;
     women:=0;men:=0;
     for eid in select jsonb_array_elements_text(lineup) loop
      select * into entry from public.pcs_interclub_entries where id=eid::uuid and season_id=p_season_id and club_id=cid;
      if not found then raise exception 'Player belongs to another club or season' using errcode='22023'; end if;
      select * into member from public.pcs_interclub_pool_members where id=entry.pool_member_id and player_id=entry.player_id and club_id=cid and season_id=p_season_id;
      select * into snap from public.pcs_interclub_meet_eligibility_snapshots where meet_id=p_meet_id and deadline=cutoff and entry_id=entry.id;
      if member.id is null or snap.entry_id is null or snap.club_id<>cid then raise exception 'Player needs an approved season signup before the meet roster deadline' using errcode='22023'; end if;
      if not public.pcs_interclub_rating_in_division(snap.rating,encounter->>'division') then raise exception 'Player must enter the skill level matching the league rating at the roster deadline' using errcode='22023'; end if;
      select club_id into represented from public.pcs_interclub_represented_players where season_id=p_season_id and identity_key=member.identity_key;
      if represented is not null and represented<>cid then raise exception 'Player represents another club this season' using errcode='22023'; end if;
      if p_phase='final' and not exists(select 1 from public.pcs_interclub_appearances a where a.season_id=p_season_id and a.club_id=cid and a.entry_id=entry.id and a.phase='regular') then raise exception 'Final player must have a regular season appearance for this club' using errcode='22023'; end if;
      if snap.gender='female' then women:=women+1; elsif snap.gender='male' then men:=men+1; end if;
     end loop;
     if jsonb_array_length(lineup)<>2 or (pairing->>'kind'='women' and women<>2) or (pairing->>'kind'='men' and men<>2)
      or (pairing->>'kind' in ('mixed_a','mixed_b') and (women<>1 or men<>1)) then raise exception 'Select the correct two players for this doubles pairing' using errcode='22023'; end if;
    end loop;
   end loop;
   if p_phase<>'regular' and encounter->'tiebreak' is not null and jsonb_typeof(encounter->'tiebreak')='object' then
    lineup:=encounter->'tiebreak'->('order_'||side);
    for eid in select jsonb_array_elements_text(coalesce(lineup,'[]')) loop
     if not exists(select 1 from jsonb_array_elements(encounter->'pairings') p
       where p->('players_'||side) ? eid or exists(select 1 from jsonb_array_elements(p->'games') g where g->('players_'||side) ? eid)) then
      raise exception 'Singles rotation must use the eligible matchup players' using errcode='22023'; end if;
    end loop;
   end if;
  end loop;
 end loop;
end $$;
create or replace function public.pcs_save_interclub_meet_roster(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_meet_id uuid,p_meet_revision integer,p_team_id uuid,p_revision integer,p_name text,p_division text,p_player_ids bigint[],p_withdraw boolean default false)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets; season public.pcs_interclub_seasons; team public.pcs_interclub_teams; player public.players; entry public.pcs_interclub_entries;
 saved public.pcs_interclub_roster_versions; lineup jsonb:='[]'; problems jsonb:='[]'; rule jsonb; old_team jsonb;
 women integer:=0; men integer:=0; gender_value text; team_created timestamptz; member public.pcs_interclub_pool_members; eligibility_rating numeric; eligibility_deadline timestamptz;
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
 eligibility_deadline:=least(meet.roster_deadline,now());
 perform public.pcs_lock_interclub_meet_eligibility(p_meet_id);
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
   select * into member from public.pcs_interclub_pool_members where season_id=p_season_id and club_id=p_club_id and player_id=player.id and status='active' and approval_status='approved';
   if not found then raise exception 'Each player needs an approved season interest signup linked to this club player' using errcode='22023'; end if;
   if exists(select 1 from public.pcs_interclub_represented_players where season_id=p_season_id and identity_key=member.identity_key and club_id<>p_club_id) then raise exception 'Player represents another club this season' using errcode='22023'; end if;
   select * into entry from public.pcs_interclub_entries where season_id=p_season_id and club_id=p_club_id and player_id=player.id and pool_member_id=member.id;
   if not found then raise exception 'Season rating seed unavailable' using errcode='22023'; end if;
   if meet.roster_deadline<=now() then
    select rating into eligibility_rating from public.pcs_interclub_meet_eligibility_snapshots where meet_id=p_meet_id and deadline=meet.roster_deadline and entry_id=entry.id;
   else eligibility_rating:=public.pcs_interclub_rating_at(p_season_id,entry.id,now()); end if;
   if eligibility_rating is null then raise exception 'Player was not eligible at the roster deadline' using errcode='22023'; end if;
   if not public.pcs_interclub_rating_in_division(eligibility_rating,p_division) then raise exception 'Choose the skill level matching the current interclub rating' using errcode='22023'; end if;
   gender_value:=case lower(trim(coalesce(player.gender,''))) when 'female' then 'female' when 'f' then 'female' when 'woman' then 'female' when 'women' then 'female'
    when 'male' then 'male' when 'm' then 'male' when 'man' then 'male' when 'men' then 'male' else 'unknown' end;
   if gender_value='female' then women:=women+1; elsif gender_value='male' then men:=men+1; end if;
   lineup:=lineup||jsonb_build_array(jsonb_build_object('entry_id',entry.id,'player_id',player.id::text,'name',player.name,'starting_rating',entry.starting_rating,'eligibility_rating',eligibility_rating,'rating_deadline',meet.roster_deadline,'rating_locked',meet.roster_deadline<=now(),'gender',gender_value));
  end loop;
  if jsonb_array_length(lineup)<>4 then raise exception 'Choose four available players from this club' using errcode='22023'; end if;
  if women<>2 or men<>2 then raise exception 'A skill-level team must have two women and two men' using errcode='22023'; end if;
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
end $$;


revoke all on function public.pcs_interclub_rating_at(uuid,uuid,timestamptz),public.pcs_interclub_gender(text),
 public.pcs_interclub_rating_in_division(numeric,text),public.pcs_guard_interclub_pool_identity(),public.pcs_seed_approved_interclub_entry(),
 public.pcs_review_interclub_pool_member(uuid,text,text,uuid,uuid,integer,boolean,text),public.pcs_lock_interclub_meet_eligibility(uuid),
 public.pcs_preserve_interclub_gender_snapshots(),public.pcs_record_interclub_representation(),public.pcs_assert_interclub_competition_eligibility(uuid,uuid,jsonb,text)
 from public,anon,authenticated;
grant execute on function public.pcs_interclub_rating_at(uuid,uuid,timestamptz),public.pcs_interclub_gender(text),
 public.pcs_interclub_rating_in_division(numeric,text),public.pcs_guard_interclub_pool_identity(),public.pcs_seed_approved_interclub_entry(),
 public.pcs_review_interclub_pool_member(uuid,text,text,uuid,uuid,integer,boolean,text),public.pcs_lock_interclub_meet_eligibility(uuid),
 public.pcs_preserve_interclub_gender_snapshots(),public.pcs_record_interclub_representation(),public.pcs_assert_interclub_competition_eligibility(uuid,uuid,jsonb,text)
 to service_role;
create or replace function public.pcs_open_interclub_meet_registration(p_actor_id uuid,p_actor_email text,p_club_id text,
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
 values(p_season_id,p_club_id,p_revision,draft.draft || jsonb_build_object('registration_rules',p_rules),p_rules) returning * into season;
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

create function public.pcs_interclub_meet_player_ratings(p_season_id uuid,p_meet_id uuid,p_club_id text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets; result jsonb;
begin
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id and club_ids ? p_club_id;
 if not found then raise exception 'Meet unavailable for this club' using errcode='P0002'; end if;
 perform public.pcs_lock_interclub_meet_eligibility(p_meet_id);
 if meet.roster_deadline<=now() then
  select coalesce(jsonb_agg(jsonb_build_object('entry_id',entry_id,'player_id',player_id,'eligibility_rating',rating,'gender',gender,'rating_deadline',deadline,'rating_locked',true)),'[]') into result
   from public.pcs_interclub_meet_eligibility_snapshots where meet_id=p_meet_id and deadline=meet.roster_deadline and club_id=p_club_id;
 else
  select coalesce(jsonb_agg(jsonb_build_object('entry_id',e.id,'player_id',e.player_id,'eligibility_rating',public.pcs_interclub_rating_at(p_season_id,e.id,now()),
   'gender',public.pcs_interclub_gender(p.gender),'rating_deadline',meet.roster_deadline,'rating_locked',false)),'[]') into result
  from public.pcs_interclub_entries e join public.pcs_interclub_pool_members m on m.id=e.pool_member_id
  join public.players p on p.id=e.player_id and p.club_id=e.club_id
  where e.season_id=p_season_id and e.club_id=p_club_id and m.status='active' and m.approval_status='approved';
 end if;
 return result;
end $$;
revoke all on function public.pcs_interclub_meet_player_ratings(uuid,uuid,text) from public,anon,authenticated;
grant execute on function public.pcs_interclub_meet_player_ratings(uuid,uuid,text) to service_role;

notify pgrst,'reload schema';
commit;
