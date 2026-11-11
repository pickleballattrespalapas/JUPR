-- Staging/isolated database only. Exercise the real public registration RPC,
-- including its player resolver, manual-partner triggers and invitation guard.
-- Every synthetic row rolls back; no API or email is invoked.
begin;
set local statement_timeout = '60s';
set local role service_role;
do $test$
declare
  club text; tournament uuid; suffix text; scenario text;
  day_id text; event1 text; event2 text;
  reg1 text; reg2 text; guest text; sel1 text; sel2 text;
  invite1 text; invite2 text; guest_sel1 text; guest_sel2 text;
  player1 integer; player2 integer;
  guest_name text; guest_email text;
  result jsonb; rejected boolean; should_reject boolean;
begin
  select id::text into strict club from public.clubs order by id limit 1;
  foreach scenario in array array[
    'legacy_same_pair', 'verified_same_pair', 'normalized_identity',
    'other_requester_email', 'other_requester_name', 'requester_other_partner',
    'expired_reservation', 'later_conflict_rolls_back'
  ] loop
    tournament := gen_random_uuid(); suffix := replace(tournament::text, '-', '');
    day_id := 'day_'||suffix; event1 := 'mixed_'||suffix; event2 := 'men_'||suffix;
    reg1 := 'partner1_'||suffix; reg2 := 'partner2_'||suffix; guest := 'guest_'||suffix;
    sel1 := 'sel1_'||suffix; sel2 := 'sel2_'||suffix;
    guest_sel1 := 'guestsel1_'||suffix; guest_sel2 := 'guestsel2_'||suffix;
    invite1 := 'inv1_'||suffix; invite2 := 'inv2_'||suffix;
    guest_name := 'Regression Guest '||suffix; guest_email := 'guest-'||suffix||'@example.invalid';
    insert into public.tournaments(id,club_id,name,status,team_count)
      values(tournament,club,'Disposable reservation regression','DRAFT',8);
    insert into public.tournament_registration_days(id,tournament_id,sort_order,label,enabled)
      values(day_id,tournament::text,0,'Regression day',true);
    insert into public.tournament_event_options(id,tournament_id,registration_day_id,sort_order,label,
      event_type,gender_restriction,partner_required,enabled,status,skill_mode,skill_label)
      values(event1,tournament::text,day_id,0,'Mixed Open','MIXED_DOUBLES','MIXED',true,true,'open','OPEN','Open'),
            (event2,tournament::text,day_id,1,'Men Open','GENDER_DOUBLES','MEN',true,true,'open','OPEN','Open');
    insert into public.players(club_id,name,rating,starting_rating,wins,losses,matches_played,active)
      values(club,'Regression Partner One '||suffix,1400,1400,0,0,0,true) returning id into player1;
    insert into public.players(club_id,name,rating,starting_rating,wins,losses,matches_played,active)
      values(club,'Regression Partner Two '||suffix,1400,1400,0,0,0,true) returning id into player2;
    insert into public.tournament_registrations(id,tournament_id,player_id,display_name,email,status,
      payment_status,age,gender,doubles_skill)
      values(reg1,tournament::text,player1,'Regression Partner One '||suffix,'one-'||suffix||'@example.invalid',
        'confirmed','unpaid',65,'Women',3.5),
        (reg2,tournament::text,player2,'Regression Partner Two '||suffix,'two-'||suffix||'@example.invalid',
        'confirmed','unpaid',75,'Men',3.5);
    insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,
      event_option_id,partner_mode,show_on_partner_board)
      values(sel1,tournament::text,reg1,day_id,event1,'NEEDS_PARTNER',true),
            (sel2,tournament::text,reg2,day_id,event2,'NEEDS_PARTNER',true);
    insert into public.tournament_partner_invitations(id,club_id,tournament_id,event_option_id,target_selection_id,
      requester_name,requester_email,message,request_key,status,accepted_at,verified_at,expires_at)
      values(invite1,club,tournament,event1,sel1,
        case when scenario='other_requester_name' then 'Different Person'
             when scenario='normalized_identity' then '  REGRESSION   GUEST '||upper(suffix)||'  '
             else guest_name end,
        case when scenario='other_requester_email' then 'someone-else-'||suffix||'@example.invalid'
             when scenario='normalized_identity' then upper(guest_email) else guest_email end,
        'Regression request',invite1,'RESERVED',now(),
        case when scenario='verified_same_pair' then now() else null end,
        case when scenario='expired_reservation' then now()-interval '1 day' else now()+interval '7 days' end),
        (invite2,club,tournament,event2,sel2,guest_name,
        case when scenario='later_conflict_rolls_back' then 'different-'||suffix||'@example.invalid' else guest_email end,
        'Regression request',invite2,'RESERVED',now(),null,now()+interval '7 days');

    if scenario='requester_other_partner' then
      -- The sender has no selection yet. Email must protect this reservation
      -- even when their new form names a different partner in the same event.
      insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,
        event_option_id,partner_mode,show_on_partner_board)
        values('alternate_'||suffix,tournament::text,reg2,day_id,event1,'NEEDS_PARTNER',true);
      update public.tournament_partner_invitations set target_selection_id='alternate_'||suffix where id=invite1;
    end if;

    rejected := false;
    should_reject := scenario in ('other_requester_email','other_requester_name',
      'requester_other_partner','later_conflict_rolls_back');
    begin
      result := public.create_tournament_registration_canonical_v1(
        jsonb_build_object('id',guest,'tournament_id',tournament::text,'submitted_at',now(),'updated_at',now(),
          'status','confirmed','payment_status','unpaid','first_name','Regression','last_name','Guest '||suffix,
          'display_name',guest_name,'email',guest_email,'doubles_skill',3.5,'age',75,'gender','Men'),
        jsonb_build_array(
          jsonb_build_object('id',guest_sel1,'registration_day_id',day_id,'event_option_id',event1,
            'partner_mode','HAS_PARTNER','partner_name','Regression Partner One '||suffix,
            'partner_email','one-'||suffix||'@example.invalid','partner_age',65,'partner_gender','Women',
            'partner_skill',3.5,'show_on_partner_board',false,'sort_order',0,'created_at',now(),'updated_at',now()),
          jsonb_build_object('id',guest_sel2,'registration_day_id',day_id,'event_option_id',event2,
            'partner_mode','HAS_PARTNER','partner_name','Regression Partner Two '||suffix,
            'partner_email','two-'||suffix||'@example.invalid','partner_age',75,'partner_gender','Men',
            'partner_skill',3.5,'show_on_partner_board',false,'sort_order',1,'created_at',now(),'updated_at',now())
        ));
    exception when others then
      if sqlerrm not like 'JUPR_INVITATION:%' then raise; end if;
      rejected := true;
    end;
    if rejected is distinct from should_reject then
      raise exception 'Scenario % rejected %, expected %',scenario,rejected,should_reject;
    end if;
    if rejected then
      if exists(select 1 from public.tournament_registrations where id=guest)
         or exists(select 1 from public.players where club_id=club and name=guest_name)
         or exists(select 1 from public.tournament_registration_team_links where tournament_id=tournament)
         or (select count(*) from public.tournament_partner_invitations
              where id in(invite1,invite2) and status='RESERVED') <> 2 then
        raise exception 'Scenario % must roll back registration, profile, both teams and invitation completion',scenario;
      end if;
    else
      if result->>'ok' <> 'true'
         or (select count(*) from public.tournament_registrations where tournament_id=tournament::text) <> 3
         or (select count(*) from public.tournament_registration_selections where tournament_id=tournament::text) <> 4
         or (select count(*) from public.tournament_registration_team_links where tournament_id=tournament
             and status='ADMIN_CONFIRMED') <> 2
         or (select count(*) from public.tournament_registration_team_members where tournament_id=tournament
             and status='ACTIVE') <> 4 then
        raise exception 'Scenario % must save one registration, two events, and two canonical teams',scenario;
      end if;
      if not exists(select 1 from public.tournament_partner_invitations i
          join public.tournament_registration_team_links l on l.accepted_request_id=i.partner_request_id
          where i.id=invite2 and i.status='COMPLETED' and i.completed_at is not null
            and i.requester_selection_id=guest_sel2 and l.tournament_id=tournament) then
        raise exception 'Scenario % lost its accepted second partnership',scenario;
      end if;
      if (select status from public.tournament_partner_invitations where id=invite1)
          <> (case when scenario='expired_reservation' then 'CANCELLED' else 'COMPLETED' end) then
        raise exception 'Scenario % left an incorrect pending reservation',scenario;
      end if;
      -- Retrying cannot duplicate the registration or consume the invitation twice.
      begin
        perform public.create_tournament_registration_canonical_v1(
          jsonb_build_object('id','retry_'||suffix,'tournament_id',tournament::text,'display_name',guest_name,
            'email',guest_email,'doubles_skill',3.5,'submitted_at',now(),'updated_at',now(),
            'status','confirmed','payment_status','unpaid'),jsonb_build_array(jsonb_build_object('id','retry_sel_'||suffix)));
        raise exception 'Expected duplicate registration rejection';
      exception when others then
        if sqlerrm not like 'JUPR_TOURNAMENT_REGISTRATION_DUPLICATE:%' then raise; end if;
      end;
    end if;
  end loop;
end;
$test$;
rollback;
