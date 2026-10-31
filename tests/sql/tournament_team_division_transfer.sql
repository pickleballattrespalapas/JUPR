-- Staging or isolated database only. Real RPCs/triggers; synthetic rows roll
-- back and no API/email is invoked. Run after the division transfer migration.
begin;
set local statement_timeout = '60s';
set local role service_role;
do $test$
declare
  club text;
  tournament uuid;
  suffix text;
  scenario text;
  reg1 text; reg2 text; sel1 text; sel2 text;
  day1 text; day2 text; source_event text; target_event text;
  request_id text; link_id text;
  player1 integer; player2 integer;
  context jsonb; result jsonb; desired jsonb; versions jsonb;
  primary_version timestamptz; selection_version timestamptz;
  before_rows jsonb; after_rows jsonb; before_orders jsonb;
  expected_reason text;
begin
  select id::text into strict club from public.clubs order by id limit 1;
  foreach scenario in array array['admin_success','admin_note_edit','admin_note_clear','public_success','stale_partner','stale_link',
    'cancelled','capacity','family','price','target_closed','duplicate','draw','public_rollback'] loop
    tournament := gen_random_uuid(); suffix := replace(tournament::text,'-','');
    reg1 := 'reg1_'||suffix; reg2 := 'reg2_'||suffix;
    sel1 := 'sel1_'||suffix; sel2 := 'sel2_'||suffix;
    day1 := 'day1_'||suffix; day2 := 'day2_'||suffix;
    source_event := 'source_'||suffix; target_event := 'target_'||suffix;
    request_id := 'request_'||suffix; link_id := 'link_'||suffix;
    insert into public.tournaments(id,club_id,name,status,team_count)
      values(tournament,club,'Disposable division transfer regression','DRAFT',8);
    insert into public.tournament_registration_days(id,tournament_id,sort_order,label,enabled)
      values(day1,tournament::text,0,'Thursday',true),(day2,tournament::text,1,'Friday',true);
    insert into public.tournament_event_options(id,tournament_id,registration_day_id,sort_order,label,event_family_label,
      event_type,gender_restriction,partner_required,enabled,status,eligibility_mode,skill_label,price_usd)
      values(source_event,tournament::text,day1,0,'Men 4.0','Men Doubles','GENDER_DOUBLES','MEN',true,true,'open','STANDARD','4.0',20),
            (target_event,tournament::text,day2,0,'Men 4.5','Men Doubles','GENDER_DOUBLES','MEN',true,true,'open','STANDARD','4.5',20);
    insert into public.players(club_id,name,rating,starting_rating,wins,losses,matches_played,active)
      values(club,'Transfer One '||suffix,1600,1600,0,0,0,true) returning id into player1;
    insert into public.players(club_id,name,rating,starting_rating,wins,losses,matches_played,active)
      values(club,'Transfer Two '||suffix,1650,1650,0,0,0,true) returning id into player2;
    insert into public.tournament_registrations(id,tournament_id,player_id,display_name,email,status,payment_status,age,gender,doubles_skill)
      values(reg1,tournament::text,player1,'Transfer One','one-'||suffix||'@example.invalid','confirmed','paid',60,'Men',4),
            (reg2,tournament::text,player2,'Transfer Two','two-'||suffix||'@example.invalid','confirmed','waived',62,'Men',4.12);
    insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode)
      values(sel1,tournament::text,reg1,day1,source_event,'HAS_PARTNER'),
            (sel2,tournament::text,reg2,day1,source_event,'HAS_PARTNER');
    insert into public.tournament_registration_partner_requests(id,tournament_id,event_option_id,requester_selection_id,
      requester_registration_id,target_selection_id,target_registration_id,status,source)
      values(request_id,tournament,source_event,sel1,reg1,sel2,reg2,'ACCEPTED','NEEDS_PARTNER_LIST');
    insert into public.tournament_registration_team_links(id,tournament_id,event_option_id,registration1_id,registration2_id,
      selection1_id,selection2_id,status,accepted_request_id)
      values(link_id,tournament,source_event,reg1,reg2,sel1,sel2,'CONFIRMED',request_id);
    insert into public.tournament_registration_team_members(id,team_link_id,tournament_id,event_option_id,selection_id,registration_id,player_order,status)
      values('member1_'||suffix,link_id,tournament,source_event,sel1,reg1,1,'ACTIVE'),
            ('member2_'||suffix,link_id,tournament,source_event,sel2,reg2,2,'ACTIVE');
    -- Exercise the canonical manual-partner trigger with real stored snapshots.
    update public.tournament_registration_selections set partner_name='Transfer Two',partner_email='two-'||suffix||'@example.invalid',
      partner_age=62,partner_gender='Men',partner_skill=4.12 where id=sel1;
    update public.tournament_registration_selections set partner_note='2.0 level' where id=sel1;
    update public.tournament_registration_selections set partner_note='Partner note retained' where id=sel2;
    -- Cancellation is exercised without an order: a real paid cancellation
    -- requires its own commerce revision evidence, outside this transfer test.
    if scenario <> 'cancelled' then
      insert into public.tournament_commerce_orders(tournament_id,club_id,registration_id,status,payment_status,current_revision,
        request_fingerprint,quote_fingerprint,list_subtotal_minor,discount_minor,total_minor)
        values(tournament,club,reg1,'ACTIVE','PAID',1,'request1','quote1',2000,0,2000),
              (tournament,club,reg2,'ACTIVE','WAIVED',1,'request2','quote2',2000,2000,0);
    end if;
    expected_reason := null;
    if scenario='cancelled' then
      update public.tournament_registrations set status='cancelled' where id=reg2;
      expected_reason:='REGISTRATION_CANCELLED';
    elsif scenario='capacity' then
      update public.tournament_event_options set capacity_teams=1 where id=target_event;
      insert into public.tournament_registrations(id,tournament_id,display_name,email)
        values('filler_'||suffix,tournament::text,'Filler','filler-'||suffix||'@example.invalid');
      insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode)
        values('filler_sel_'||suffix,tournament::text,'filler_'||suffix,day2,target_event,'NEEDS_PARTNER');
      expected_reason:='DIVISION_FULL';
    elsif scenario='family' then
      update public.tournament_event_options set event_family_label='Mixed Doubles' where id=target_event;
      expected_reason:='INVALID_DIVISION_TARGET';
    elsif scenario='price' then
      update public.tournament_event_options set price_usd=30 where id=target_event;
      expected_reason:='DIVISION_PRICE_CHANGED';
    elsif scenario='target_closed' then
      update public.tournament_event_options set enabled=false where id=target_event;
      expected_reason:='INVALID_DIVISION_TARGET';
    elsif scenario='duplicate' then
      insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode)
        values('duplicate_'||suffix,tournament::text,reg2,day2,target_event,'NEEDS_PARTNER');
      expected_reason:='DUPLICATE_EVENT_FAMILY';
    elsif scenario='draw' then
      insert into public.tournament_event_draws(tournament_id,registration_day_id,event_option_id,name,status)
        values(tournament,day2,target_event,'Started target','playing');
      expected_reason:='REGISTRATION_IMPORTED_TO_DRAW';
    end if;
    select jsonb_build_object('team_link_id',l.id,'team_link_updated_at',l.updated_at,
      'partner_selection_id',s.id,'partner_selection_updated_at',s.updated_at,
      'partner_registration_id',r.id,'partner_registration_updated_at',r.updated_at)
      into context from public.tournament_registration_team_links l
      join public.tournament_registration_selections s on s.id=sel2
      join public.tournament_registrations r on r.id=reg2 where l.id=link_id;
    if scenario in ('stale_partner','public_rollback') then
      context:=jsonb_set(context,'{partner_selection_updated_at}',to_jsonb(now()-interval '1 day'));
      expected_reason:='STALE_TEAM_VERSION';
    elsif scenario='stale_link' then
      context:=jsonb_set(context,'{team_link_updated_at}',to_jsonb(now()-interval '1 day'));
      expected_reason:='STALE_TEAM_VERSION';
    end if;
    select jsonb_build_object('registrations',(select jsonb_agg(to_jsonb(r) order by id) from public.tournament_registrations r where id in(reg1,reg2)),
      'selections',(select jsonb_agg(to_jsonb(s) order by id) from public.tournament_registration_selections s where id in(sel1,sel2)),
      'links',(select jsonb_agg(to_jsonb(l)) from public.tournament_registration_team_links l where id=link_id),
      'requests',(select jsonb_agg(to_jsonb(r)) from public.tournament_registration_partner_requests r where id=request_id),
      'members',(select jsonb_agg(to_jsonb(m) order by id) from public.tournament_registration_team_members m where team_link_id=link_id)) into before_rows;
    select jsonb_agg(to_jsonb(o) order by id) into before_orders from public.tournament_commerce_orders o where registration_id in(reg1,reg2);
    select updated_at into primary_version from public.tournament_registrations where id=reg1;
    select updated_at into selection_version from public.tournament_registration_selections where id=sel1;
    if scenario in ('public_success','public_rollback') then
      select jsonb_agg(jsonb_build_object('id',id,'updated_at',updated_at)) into versions
        from public.tournament_registration_selections where registration_id=reg1;
      select jsonb_agg((select jsonb_object_agg(key,value) from jsonb_each(to_jsonb(s)) where key=any(array[
        'id','tournament_id','registration_id','registration_day_id','event_option_id','partner_mode','partner_name',
        'partner_email','partner_phone','partner_dupr_id','partner_skill','partner_age','partner_gender','partner_note',
        'show_on_partner_board','sort_order','created_at','updated_at'])) || jsonb_build_object(
          'event_option_id',target_event,'registration_day_id',day2,'division_transfer_context',context)) into desired
        from public.tournament_registration_selections s where registration_id=reg1;
      result:=public.server_update_public_tournament_registration_edit(tournament::text,reg1,primary_version,versions,
        jsonb_build_object('display_name','Edited Name','age',60,'gender','Men','doubles_skill',4),desired);
    else
      desired:=jsonb_build_object('event_option_id',target_event,'registration_day_id',day2,'division_transfer_context',context);
      if scenario='admin_note_edit' then
        desired:=desired || jsonb_build_object('partner_note','Moving up');
      elsif scenario='admin_note_clear' then
        desired:=desired || jsonb_build_object('partner_note',null);
      end if;
      result:=public.admin_update_tournament_registration_selection(tournament::text,sel1,selection_version,
        desired);
    end if;
    if expected_reason is null then
      if (result->>'ok') is distinct from 'true' then raise exception 'Successful % failed: %',scenario,result; end if;
      if (select count(*) from public.tournament_registration_selections where id in(sel1,sel2)
          and event_option_id=target_event and registration_day_id=day2)<>2
        or (select event_option_id from public.tournament_registration_team_links where id=link_id)<>target_event
        or (select event_option_id from public.tournament_registration_partner_requests where id=request_id)<>target_event
        or (select count(*) from public.tournament_registration_team_members where team_link_id=link_id and event_option_id=target_event and status='ACTIVE')<>2
        or (select count(*) from public.tournament_registration_team_links where tournament_id=tournament)<>1 then
        raise exception 'Incomplete or duplicated team transfer for %',scenario;
      end if;
      if (select payment_status from public.tournament_registrations where id=reg1)<>'paid'
        or (select payment_status from public.tournament_registrations where id=reg2)<>'waived'
        or (select partner_email from public.tournament_registration_selections where id=sel1)<>'two-'||suffix||'@example.invalid' then
        raise exception 'Payment or partner data changed for %',scenario;
      end if;
      if (select partner_note from public.tournament_registration_selections where id=sel1)
          is distinct from (case scenario when 'admin_note_edit' then 'Moving up' when 'admin_note_clear' then null else '2.0 level' end)
        or (select partner_note from public.tournament_registration_selections where id=sel2) is distinct from 'Partner note retained' then
        raise exception 'Public notes were not preserved or updated correctly for %',scenario;
      end if;
    else
      if (result->>'ok') is distinct from 'false' or coalesce(result->>'reason',result->>'code')<>expected_reason then
        raise exception 'Wrong conflict for % expected %, got %',scenario,expected_reason,result;
      end if;
      select jsonb_build_object('registrations',(select jsonb_agg(to_jsonb(r) order by id) from public.tournament_registrations r where id in(reg1,reg2)),
        'selections',(select jsonb_agg(to_jsonb(s) order by id) from public.tournament_registration_selections s where id in(sel1,sel2)),
        'links',(select jsonb_agg(to_jsonb(l)) from public.tournament_registration_team_links l where id=link_id),
        'requests',(select jsonb_agg(to_jsonb(r)) from public.tournament_registration_partner_requests r where id=request_id),
        'members',(select jsonb_agg(to_jsonb(m) order by id) from public.tournament_registration_team_members m where team_link_id=link_id)) into after_rows;
      if before_rows is distinct from after_rows then raise exception 'Partial mutation leaked for %',scenario; end if;
    end if;
    if before_orders is distinct from (select jsonb_agg(to_jsonb(o) order by id) from public.tournament_commerce_orders o where registration_id in(reg1,reg2)) then
      raise exception 'Commerce history was changed for %',scenario;
    end if;
  end loop;
end
$test$;
select '14 atomic team transfer scenarios passed; fixtures rolled back' as result;
rollback;
