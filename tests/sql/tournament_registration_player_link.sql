-- Staging/isolated DB only. Exercise real RPCs, triggers and rollback behavior.
begin;
set local statement_timeout='60s';
set local role service_role;
do $test$
declare
  club text; tournament uuid; suffix text; scenario text;
  reg1 text; reg2 text; sel1 text; sel2 text; day1 text; event1 text; link1 text; request1 text;
  player1 integer; player2 integer; selected_player integer;
  version timestamptz; expected_version timestamptz;
  result jsonb; before_registration jsonb; before_partner jsonb; before_player jsonb;
  expected_code text;
begin
  assert not has_function_privilege('anon','public.admin_link_tournament_registration_player_v1(text,text,text,integer,timestamptz,text,text)','EXECUTE');
  assert not has_function_privilege('authenticated','public.admin_link_tournament_registration_player_v1(text,text,text,integer,timestamptz,text,text)','EXECUTE');
  select id::text into strict club from public.clubs order by id limit 1;
  foreach scenario in array array['success','second_member','stale','wrong_club','duplicate','existing_registration','team_conflict','finalized','imported','cancelled'] loop
    tournament:=gen_random_uuid(); suffix:=replace(tournament::text,'-','');
    reg1:='link_reg1_'||suffix; reg2:='link_reg2_'||suffix;
    sel1:='link_sel1_'||suffix; sel2:='link_sel2_'||suffix;
    day1:='link_day_'||suffix; event1:='link_event_'||suffix;
    link1:='link_team_'||suffix; request1:='link_request_'||suffix;
    insert into public.tournaments(id,club_id,name,status,team_count)
      values(tournament,club,'Disposable profile linking test','DRAFT',8);
    insert into public.tournament_registration_days(id,tournament_id,sort_order,label,enabled)
      values(day1,tournament::text,0,'Test day',true);
    insert into public.tournament_event_options(id,tournament_id,registration_day_id,sort_order,label,event_type,gender_restriction,partner_required,enabled,status,eligibility_mode)
      values(event1,tournament::text,day1,0,'Test doubles','GENDER_DOUBLES','ANY',true,true,'open','STANDARD');
    insert into public.players(club_id,name,rating,starting_rating,wins,losses,matches_played,active)
      values(club,'Link One '||suffix,1600,1600,0,0,0,true) returning id into player1;
    insert into public.players(club_id,name,rating,starting_rating,wins,losses,matches_played,active)
      values(club,'Link Two '||suffix,1650,1650,0,0,0,true) returning id into player2;
    insert into public.tournament_registrations(id,tournament_id,player_id,display_name,email,status,payment_status,doubles_skill,singles_skill)
      values(reg1,tournament::text,null,'Link One','one-'||suffix||'@example.invalid','confirmed','paid',3.5,3.1),
            (reg2,tournament::text,player2,'Link Two','two-'||suffix||'@example.invalid','confirmed','waived',4.1,3.9);
    insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode)
      values(sel1,tournament::text,reg1,day1,event1,'HAS_PARTNER'),(sel2,tournament::text,reg2,day1,event1,'HAS_PARTNER');
    insert into public.tournament_registration_partner_requests(id,tournament_id,event_option_id,requester_selection_id,requester_registration_id,target_selection_id,target_registration_id,status,source,target_player_id)
      values(request1,tournament,event1,sel1,reg1,sel2,reg2,'ACCEPTED','NEEDS_PARTNER_LIST',player2);
    insert into public.tournament_registration_team_links(id,tournament_id,event_option_id,registration1_id,registration2_id,selection1_id,selection2_id,player2_id,status,accepted_request_id)
      values(link1,tournament,event1,reg1,reg2,sel1,sel2,player2,'CONFIRMED',request1);
    insert into public.tournament_registration_team_members(id,team_link_id,tournament_id,event_option_id,selection_id,registration_id,player_id,player_order,status)
      values('member1_'||suffix,link1,tournament,event1,sel1,reg1,null,1,'ACTIVE'),
            ('member2_'||suffix,link1,tournament,event1,sel2,reg2,player2,2,'ACTIVE');
    expected_code:=null; selected_player:=player1;
    if scenario='second_member' then
      update public.tournament_registrations set player_id=player1 where id=reg1;
      update public.tournament_registration_team_members set player_id=player1 where registration_id=reg1;
      update public.tournament_registration_team_links set player1_id=player1 where id=link1;
      update public.tournament_registrations set player_id=null where id=reg2;
      update public.tournament_registration_team_members set player_id=null where registration_id=reg2;
      update public.tournament_registration_team_links set player2_id=null where id=link1;
      update public.tournament_registration_partner_requests set target_player_id=null where id=request1;
      reg1:=reg2; selected_player:=player2;
    elsif scenario='wrong_club' then
      update public.players set club_id=(select id::text from public.clubs where id::text<>club order by id limit 1) where id=player1;
      expected_code:='PLAYER';
    elsif scenario='duplicate' then
      selected_player:=player2; expected_code:='DUPLICATE';
    elsif scenario='existing_registration' then
      insert into public.tournament_registrations(id,tournament_id,player_id,display_name,email,status)
        values('existing_'||suffix,tournament::text,player1,'Already entered','existing-'||suffix||'@example.invalid','confirmed');
      expected_code:='UNIQUE_REGISTRATION';
    elsif scenario='team_conflict' then
      update public.tournament_registration_team_members set player_id=player2 where registration_id=reg1;
      expected_code:='TEAM_CONFLICT';
    elsif scenario='finalized' then
      insert into public.tournament_rating_eligibility_reviews(tournament_id,event_option_id,selection_id,registration_id,review_phase,state,finalized_at,combined_rating_cap,player_rating_source,partner_rating_source)
        values(tournament,event1,sel1,reg1,'REGISTRATION_CLOSE','REVIEW_REQUIRED',clock_timestamp(),10,'MISSING','MISSING');
      expected_code:='FINALIZED';
    elsif scenario='imported' then
      insert into public.tournament_teams(tournament_id,team_number,player1_id,source,source_selection_id)
        values(tournament,1,player2,'REGISTRATION',sel2);
      expected_code:='IMPORTED';
    elsif scenario='cancelled' then
      update public.tournament_registrations set status='cancelled' where id=reg1;
      expected_code:='STATUS';
    end if;
    select updated_at,to_jsonb(r) into version,before_registration from public.tournament_registrations r where id=reg1;
    select to_jsonb(r) into before_partner from public.tournament_registrations r where id=reg2;
    select to_jsonb(p) into before_player from public.players p where id=selected_player;
    expected_version:=version;
    if scenario='stale' then expected_version:=version-interval '1 second'; expected_code:='STALE'; end if;
    begin
      result:=public.admin_link_tournament_registration_player_v1(club,tournament::text,reg1,selected_player,expected_version,'fixture@example.invalid','club_owner');
    exception when unique_violation then
      if scenario<>'existing_registration' then raise; end if;
      result:=jsonb_build_object('ok',false,'code','UNIQUE_REGISTRATION');
    end;
    if expected_code is not null then
      assert result->>'code'=expected_code,scenario||': '||result::text;
      assert (select to_jsonb(r)=before_registration from public.tournament_registrations r where id=reg1),scenario||' changed rejected registration';
      assert not exists(select 1 from public.admin_activity_log where entity_id=reg1),scenario||' wrote an audit';
    else
      assert result->>'ok'='true',result::text;
      assert (select player_id=selected_player from public.tournament_registrations where id=reg1);
      assert not exists(select 1 from public.tournament_registration_team_members where registration_id=reg1 and player_id is distinct from selected_player);
      assert (select player1_id=player1 and player2_id=player2 from public.tournament_registration_team_links where id=link1);
      assert (select (to_jsonb(r)-'player_id'-'updated_at')=(before_registration-'player_id'-'updated_at') from public.tournament_registrations r where id=reg1),'Registration data changed';
      assert (select to_jsonb(p)=before_player from public.players p where id=selected_player),'Player rating/history changed';
      if scenario='success' then
        assert (select to_jsonb(r)=before_partner from public.tournament_registrations r where id=reg2),'Partner changed';
        assert (select requester_player_id=player1 from public.tournament_registration_partner_requests where id=request1);
      else
        assert (select target_player_id=player2 from public.tournament_registration_partner_requests where id=request1);
      end if;
      result:=public.admin_link_tournament_registration_player_v1(club,tournament::text,reg1,selected_player,version,'fixture@example.invalid','club_owner');
      assert result->>'idempotent_replay'='true','Retry was not a no-op';
      assert (select count(*)=1 from public.admin_activity_log where entity_id=reg1),'Retry added another audit';
    end if;
  end loop;
end
$test$;
rollback;
