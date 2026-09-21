-- Staging-only rehearsal. Every fixture and cancellation is rolled back.
begin;
set local statement_timeout = '45s';
set local role service_role;
do $test$
#variable_conflict use_variable
declare
  club text;
  tour uuid;
  suffix text;
  scenario text;
  gone text;
  survivor text;
  third text;
  day_id text;
  men text;
  mixed text;
  gs text;
  ss text;
  ms text;
  ts text;
  link_id text;
  request_id text;
  version timestamptz;
  changes jsonb;
  result jsonb;
  before_other jsonb;
  before_survivor jsonb;
  order_id uuid;
  revision_id uuid;
  item_id uuid;
  variant_id uuid;
  line_id uuid;
  rejected boolean;
begin
  if has_function_privilege('anon','public.server_cancel_tournament_registrations(text,jsonb,text)','execute')
     or has_function_privilege('authenticated','public.server_cancel_tournament_registrations(text,jsonb,text)','execute') then
    raise exception 'Public clients can cancel registrations';
  end if;
  select id::text into strict club from public.clubs order by id limit 1;
  foreach scenario in array array['CONFIRMED','LEGACY_CANCELLED','PENDING','COMMERCE','FULFILLED','STALE_BATCH','BOTH_CANCEL'] loop
    tour := gen_random_uuid(); suffix := replace(tour::text,'-','');
    gone := 'gone_' || suffix; survivor := 'stay_' || suffix; third := 'third_' || suffix;
    day_id := 'day_' || suffix; men := 'men_' || suffix; mixed := 'mixed_' || suffix;
    gs := 'gs_' || suffix; ss := 'ss_' || suffix; ms := 'ms_' || suffix; ts := 'ts_' || suffix;
    link_id := 'link_' || suffix; request_id := 'request_' || suffix;
    insert into public.tournaments(id,club_id,name,status,team_count) values(tour,club,'Disposable cancellation rehearsal','DRAFT',8);
    insert into public.tournament_registration_days(id,tournament_id,sort_order,label) values(day_id,tour::text,0,'Rehearsal');
    insert into public.tournament_event_options(id,tournament_id,registration_day_id,sort_order,label,event_type,partner_required)
      values(men,tour::text,day_id,0,'Men Open','GENDER_DOUBLES',true),(mixed,tour::text,day_id,1,'Mixed Open','MIXED_DOUBLES',true);
    insert into public.tournament_registrations(id,tournament_id,display_name,email,status,age,gender,doubles_skill)
      values(gone,tour::text,'Testing Fixture',gone || '@example.invalid',case when scenario='LEGACY_CANCELLED' then 'cancelled' else 'confirmed' end,42,'Men',5),
            (survivor,tour::text,'Joe Fixture',survivor || '@example.invalid','confirmed',38,'Men',5),
            (third,tour::text,'Paola Fixture',third || '@example.invalid','confirmed',50,'Women',4.65);
    insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode,sort_order)
      values(gs,tour::text,gone,day_id,men,case when scenario='PENDING' then 'NEEDS_PARTNER' else 'HAS_PARTNER' end,0),
            (ss,tour::text,survivor,day_id,men,case when scenario='PENDING' then 'NEEDS_PARTNER' else 'HAS_PARTNER' end,0),
            (ms,tour::text,survivor,day_id,mixed,'HAS_PARTNER',1),(ts,tour::text,third,day_id,mixed,'HAS_PARTNER',0);
    insert into public.tournament_registration_partner_requests(id,tournament_id,event_option_id,requester_selection_id,requester_registration_id,target_selection_id,target_registration_id,status,source)
      values(request_id,tour,men,ss,survivor,gs,gone,case when scenario='PENDING' then 'PENDING' else 'ACCEPTED' end,'NEEDS_PARTNER_LIST');
    if scenario <> 'PENDING' then
      insert into public.tournament_registration_team_links(id,tournament_id,event_option_id,registration1_id,registration2_id,selection1_id,selection2_id,status,accepted_request_id)
        values(link_id,tour,men,gone,survivor,gs,ss,'CONFIRMED',request_id);
      insert into public.tournament_registration_team_members(id,team_link_id,tournament_id,event_option_id,selection_id,registration_id,player_order,status)
        values('m1_' || suffix,link_id,tour,men,gs,gone,1,'ACTIVE'),('m2_' || suffix,link_id,tour,men,ss,survivor,2,'ACTIVE');
    end if;
    insert into public.tournament_registration_team_links(id,tournament_id,event_option_id,registration1_id,registration2_id,selection1_id,selection2_id,status)
      values('mixedlink_' || suffix,tour,mixed,survivor,third,ms,ts,'ADMIN_CONFIRMED');
    select jsonb_build_object('selection',(select to_jsonb(s) from public.tournament_registration_selections s where id=ms),
      'link',(select to_jsonb(l) from public.tournament_registration_team_links l where id='mixedlink_' || suffix),
      'partner',(select to_jsonb(r) from public.tournament_registrations r where id=third)) into before_other;
    select to_jsonb(s) into before_survivor from public.tournament_registration_selections s where id=ss;
    select updated_at into version from public.tournament_registrations where id=gone;
    changes := jsonb_build_array(jsonb_build_object('id',gone,'expected_updated_at',version,'patch','{}'::jsonb));

    if scenario in ('COMMERCE','FULFILLED') then
      order_id:=gen_random_uuid(); revision_id:=gen_random_uuid(); item_id:=gen_random_uuid(); variant_id:=gen_random_uuid(); line_id:=gen_random_uuid();
      insert into public.tournament_commerce_orders(id,tournament_id,club_id,registration_id,current_revision,request_fingerprint,quote_fingerprint,list_subtotal_minor,discount_minor,total_minor)
        values(order_id,tour,club,gone,1,'request','quote',1000,0,1000);
      insert into public.tournament_commerce_order_revisions(id,order_id,tournament_id,registration_id,revision,request_fingerprint,quote_fingerprint,catalog_fingerprint,currency,list_subtotal_minor,discount_minor,total_minor,price_snapshot,actor_type,actor_label,source)
        values(revision_id,order_id,tour,gone,1,'request','quote','catalog','USD',1000,0,1000,'{}','ADMIN','rehearsal','rehearsal');
      insert into public.tournament_commerce_items(id,tournament_id,club_id,name,kind,base_price_minor) values(item_id,tour,club,'Rehearsal item','OTHER',1000);
      insert into public.tournament_commerce_item_variants(id,item_id,name) values(variant_id,item_id,'Default');
      insert into public.tournament_commerce_order_lines(id,order_id,revision_id,tournament_id,line_key,line_type,item_id,variant_id,label_snapshot,quantity,list_unit_minor,final_unit_minor,list_total_minor,final_total_minor,line_snapshot,requires_fulfillment)
        values(line_id,order_id,revision_id,tour,'item','ITEM',item_id,variant_id,'Rehearsal item',1,1000,1000,1000,1000,'{}',true);
      insert into public.tournament_commerce_fulfillment(order_id,revision_id,order_line_id,tournament_id,item_id,variant_id,label_snapshot,quantity,status)
        values(order_id,revision_id,line_id,tour,item_id,variant_id,'Rehearsal item',1,case when scenario='FULFILLED' then 'FULFILLED' else 'PENDING' end);
    end if;
    if scenario='STALE_BATCH' then
      changes := changes || jsonb_build_array(jsonb_build_object('id',third,'expected_updated_at','2000-01-01T00:00:00Z','patch','{}'::jsonb));
    elsif scenario='BOTH_CANCEL' then
      changes := changes || jsonb_build_array(jsonb_build_object('id',survivor,'expected_updated_at',(select updated_at from public.tournament_registrations where id=survivor),'patch','{}'::jsonb));
    end if;
    if scenario in ('FULFILLED','STALE_BATCH') then
      rejected:=false;
      begin
        perform public.server_cancel_tournament_registrations(tour::text,changes,'rehearsal@example.invalid');
      exception when others then
        if position(case when scenario='FULFILLED' then 'FULFILLED_REGISTRATION_CANCEL_LOCKED' else 'JUPR_CANCEL_CONFLICT' end in sqlerrm)=0 then raise; end if;
        rejected:=true;
      end;
      if not rejected or (select updated_at from public.tournament_registrations where id=gone) is distinct from version
        or not exists(select 1 from public.tournament_registration_team_links where id=link_id)
        or exists(select 1 from private.tournament_registration_cancellations where registration_id=gone) then
        raise exception 'Failed cancellation did not roll back %',scenario;
      end if;
      continue;
    end if;
    result:=public.server_cancel_tournament_registrations(tour::text,changes,'rehearsal@example.invalid');
    if result->>'ok' <> 'true' or exists(select 1 from public.tournament_registrations where id=gone)
      or exists(select 1 from public.tournament_registration_selections where id=gs)
      or exists(select 1 from public.tournament_registration_team_links where id=link_id)
      or exists(select 1 from public.tournament_registration_team_members where team_link_id=link_id)
      or exists(select 1 from public.tournament_registration_partner_requests where id=request_id) then
      raise exception 'Registration or connections retained in %',scenario;
    end if;
    if scenario='BOTH_CANCEL' then
      if exists(select 1 from public.tournament_registrations where id=survivor)
        or (select partner_mode from public.tournament_registration_selections where id=ts) <> 'NEEDS_PARTNER' then
        raise exception 'Batch cancellation failed';
      end if;
    else
      if (select partner_mode from public.tournament_registration_selections where id=ss) <> 'NEEDS_PARTNER' then
        raise exception 'Remaining partner not released in %',scenario;
      end if;
      if scenario <> 'PENDING' and (select show_on_partner_board from public.tournament_registration_selections where id=ss) is distinct from true then
        raise exception 'Remaining partner not on public board in %',scenario;
      end if;
      if (select jsonb_build_object('selection',(select to_jsonb(s) from public.tournament_registration_selections s where id=ms),
          'link',(select to_jsonb(l) from public.tournament_registration_team_links l where id='mixedlink_' || suffix),
          'partner',(select to_jsonb(r) from public.tournament_registrations r where id=third))) is distinct from before_other then
        raise exception 'Unrelated mixed partnership changed in %',scenario;
      end if;
    end if;
    if scenario='COMMERCE' and ((select status from public.tournament_commerce_orders where id=order_id) <> 'CANCELLED'
      or (select count(*) from public.tournament_commerce_order_revisions rev where rev.order_id=order_id) < 2
      or exists(select 1 from public.tournament_commerce_fulfillment f where f.order_id=order_id and status <> 'CANCELLED')) then
      raise exception 'Commerce cancellation lost history or fulfillment was not released';
    end if;
    if not exists(select 1 from private.tournament_registration_cancellations where registration_id=gone) then
      raise exception 'Cancellation receipt missing';
    end if;
    -- Same email starts a new independent registration, with no old selections.
    insert into public.tournament_registrations(id,tournament_id,display_name,email,status)
      values('fresh_' || suffix,tour::text,'Fresh fixture',gone || '@example.invalid','confirmed');
    if exists(select 1 from public.tournament_registration_selections where registration_id='fresh_' || suffix) then
      raise exception 'Fresh signup inherited cancelled entries';
    end if;
  end loop;
end $test$;
select 'Cancellation, partner release, unrelated-event preservation, stale-batch/fulfilled rollback, commerce history and fresh signup passed' as result;
rollback;
