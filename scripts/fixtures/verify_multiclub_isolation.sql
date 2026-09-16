-- Staging-only verification. All attempted writes are rolled back.
begin;
do $verify$
declare
 actor_id uuid; actor_email text; own record; other record; inactive_id bigint;
 lineup bigint[]; wrong_lineup bigint[]; message text; before_other text; after_other text;
 season uuid := 'a7c8e9be-333b-51b7-8e68-9abb9c2d803f';
 checked_pairs integer := 0; checked_clubs integer := 0;
begin
 if not exists(select 1 from clubs where id='tres_palapas' and features_json @> '{"staging_fixture":true,"synthetic_data_only":true}')
 or (select count(*) from clubs where features_json->>'isolation_fixture'='multiclub-isolation-20260916')<>3 then
  raise exception 'Three-club staging fixture required'; end if;
 select a.user_id,a.email into strict actor_id,actor_email from admin_role_assignments a
 join auth.users u on u.id=a.user_id and lower(u.email)=a.email
 where a.club_id='la-ribera-pickelball-club' and a.role='administrator' and a.revoked_at is null and u.email_confirmed_at is not null;
 for own in select t.*,m.revision as meet_revision from pcs_interclub_teams t join pcs_interclub_meets m on m.id=t.meet_id
  where t.season_id=season and t.division='3.5' and m.plan_index=0 loop
  select array_agg(player_id order by player_id) into lineup from pcs_interclub_team_players where team_id=own.id;
  for other in select t.* from pcs_interclub_teams t join pcs_interclub_meets m on m.id=t.meet_id
   where t.season_id=season and t.division='3.5' and m.plan_index=0 and t.club_id<>own.club_id loop
   wrong_lineup := lineup;
   select min(player_id) into inactive_id from pcs_interclub_team_players where team_id=other.id;
   wrong_lineup[4] := inactive_id;
   begin
    perform pcs_save_interclub_meet_roster(actor_id,actor_email,own.club_id,season,own.meet_id,own.meet_revision,
     own.id,own.revision,own.name,own.division,wrong_lineup);
    raise exception 'Foreign player was accepted';
   exception when invalid_parameter_value then
    get stacked diagnostics message = message_text;
    if message<>'Choose active players from this club' then raise; end if;
   end;
   checked_pairs := checked_pairs+1;
   begin
    perform pcs_save_interclub_meet_roster(actor_id,actor_email,own.club_id,season,own.meet_id,own.meet_revision,
     other.id,other.revision,other.name,other.division,lineup);
    raise exception 'Foreign team was editable';
   exception when insufficient_privilege then
    get stacked diagnostics message = message_text;
    if message<>'Only the represented club can edit its team' then raise; end if;
   end;
  end loop;
  select id into strict inactive_id from players where club_id=own.club_id and active=false;
  wrong_lineup := lineup; wrong_lineup[4] := inactive_id;
  begin
   perform pcs_save_interclub_meet_roster(actor_id,actor_email,own.club_id,season,own.meet_id,own.meet_revision,
    own.id,own.revision,own.name,own.division,wrong_lineup);
   raise exception 'Inactive player was accepted';
  exception when invalid_parameter_value then
   get stacked diagnostics message = message_text;
   if message<>'Choose active players from this club' then raise; end if;
  end;
  -- A successful substitution must leave every other club and meet unchanged.
  select md5(jsonb_agg(to_jsonb(t) order by t.id)::text) into before_other
   from pcs_interclub_current_rosters t where t.season_id=season and t.id<>own.id;
  select array_agg(p.id order by p.id) into wrong_lineup from players p
   where p.club_id=own.club_id and p.name in ('Alex Rivera [TEST]','Sam Torres [TEST]','Taylor Vega [TEST]','Casey Luna [TEST]');
  begin
   perform pcs_save_interclub_meet_roster(actor_id,actor_email,own.club_id,season,own.meet_id,own.meet_revision,
    own.id,own.revision,own.name,own.division,wrong_lineup);
   if not exists(select 1 from pcs_interclub_teams where id=own.id and revision=own.revision+1) then
    raise exception 'Expected successful substitution'; end if;
   select md5(jsonb_agg(to_jsonb(t) order by t.id)::text) into after_other
    from pcs_interclub_current_rosters t where t.season_id=season and t.id<>own.id;
   if before_other is distinct from after_other then raise exception 'Substitution changed another team or meet'; end if;
   raise exception using errcode='PT001',message='Rollback successful substitution probe';
  exception when sqlstate 'PT001' then null;
  end;
  checked_clubs := checked_clubs+1;
 end loop;
 if checked_pairs<>6 or checked_clubs<>3 then raise exception 'Incomplete fixture probe coverage'; end if;
 if exists(select 1 from pcs_interclub_teams where season_id=season and revision<>1) then
  raise exception 'Probe left a changed roster'; end if;
 if exists(select 1 from matches m cross join lateral unnest(array[m.t1_p1,m.t1_p2,m.t2_p1,m.t2_p2]) pid
   join players p on p.id=pid where m.notes='multiclub-isolation-20260916' and m.club_id<>p.club_id) then
  raise exception 'Match references another club player'; end if;
 if exists(select 1 from league_ratings l join players p on p.id=l.player_id
   where l.league_name='Isolation Test Ladder' and l.club_id<>p.club_id) then
  raise exception 'League references another club player'; end if;
 if exists(select 1 from pcs_interclub_team_players t join players p on p.id=t.player_id
   where t.season_id=season and t.club_id<>p.club_id) then
  raise exception 'Interclub team references another club player'; end if;
end $verify$;
rollback;
select 'passed' as status,6 as rejected_foreign_player_attempts,6 as rejected_foreign_team_edits,
 3 as rejected_inactive_players,3 as isolated_substitutions,0 as persisted_probe_changes;
