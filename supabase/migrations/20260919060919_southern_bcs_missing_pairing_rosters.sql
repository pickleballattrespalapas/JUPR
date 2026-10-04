begin;
-- Two actual players may enter one valid doubles pairing; the missing pairing
-- forfeits. No absent players or fictitious player identities are required.
alter table public.pcs_interclub_roster_versions drop constraint pcs_interclub_roster_versions_roster_check;
alter table public.pcs_interclub_roster_versions add constraint pcs_interclub_roster_versions_roster_check
 check(jsonb_typeof(roster)='array' and jsonb_array_length(roster) in (2,4));
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
   if not public.pcs_interclub_rating_in_division(eligibility_rating,p_division) then raise exception 'Choose the skill level matching the current interclub rating' using errcode='22023'; end if;
   gender_value:=case lower(trim(coalesce(player.gender,''))) when 'female' then 'female' when 'f' then 'female' when 'woman' then 'female' when 'women' then 'female'
    when 'male' then 'male' when 'm' then 'male' when 'man' then 'male' when 'men' then 'male' else 'unknown' end;
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
end $$;


notify pgrst,'reload schema';
commit;
