begin;
set local statement_timeout = '25s';
do $test$
declare
  tid uuid := gen_random_uuid();
  cid text := (select club_id from public.tournaments limit 1);
  prefix text := 'partner_sql_' || replace(gen_random_uuid()::text, '-', '');
  eid text;
  did text;
  request_data jsonb;
  result jsonb;
  versions jsonb;
  created jsonb;
  i integer;
  rejected boolean := false;
begin
  eid := prefix || '_event'; did := prefix || '_day';
  insert into public.tournaments(id, club_id, name, team_count) values(tid, cid, 'Disposable partner invitation SQL test', 4);
  insert into public.tournament_registration_settings(id, tournament_id, registration_status, partner_board_enabled)
    values(prefix || '_settings', tid::text, 'open', true);
  insert into public.tournament_registration_days(id,tournament_id,sort_order,label) values(did,tid::text,0,'Day 1');
  insert into public.tournament_event_options(id,tournament_id,registration_day_id,sort_order,label,event_type,status,partner_required,partner_board_enabled)
    values(eid,tid::text,did,0,'Open doubles','DOUBLES','open',true,true);
  for i in 1..7 loop
    insert into public.tournament_registrations(id,tournament_id,display_name,email,wants_partner_board_contact)
      values(prefix || '_reg' || i,tid::text,'Fixture ' || i,'fixture' || i || '@example.invalid',true);
    insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode,show_on_partner_board)
      values(prefix || '_sel' || i,tid::text,prefix || '_reg' || i,did,eid,'NEEDS_PARTNER',true);
  end loop;
  request_data := jsonb_build_object('id',prefix || '_one','club_id',cid,'tournament_id',tid,
    'target_selection_id',prefix || '_sel2','requester_name','Fixture 1','requester_email','fixture1@example.invalid',
    'message','Want to partner?', 'request_key', prefix || '_key1','verified',false);
  created := public.create_tournament_partner_invitation(request_data);
  if created->>'status' <> 'UNVERIFIED' then raise exception 'create did not require verification'; end if;
  if not (public.create_tournament_partner_invitation(request_data)->>'idempotent')::boolean then raise exception 'create retry was not idempotent'; end if;
  begin
    perform public.transition_tournament_partner_invitation(prefix || '_one','accept');
  exception when others then rejected := sqlerrm like '%sender must confirm%'; end;
  if not rejected then raise exception 'unverified request could be accepted'; end if;
  request_data := request_data || jsonb_build_object('send_directly',true);
  created := public.create_tournament_partner_invitation(request_data);
  if created->>'status' <> 'PENDING' or created->>'verified_at' is not null then raise exception 'direct send did not preserve unverified email state'; end if;
  perform public.create_tournament_partner_invitation(request_data || jsonb_build_object('id',prefix || '_competing','request_key',prefix || '_key2','requester_name','Fixture 3','requester_email','fixture3@example.invalid','verified',true));
  select jsonb_build_object('requester_registration',r.updated_at,'target_registration',tr.updated_at,
    'requester_selection',s.updated_at,'target_selection',ts.updated_at,'event',e.updated_at) into versions
    from public.tournament_registrations r,public.tournament_registrations tr,
      public.tournament_registration_selections s,public.tournament_registration_selections ts,public.tournament_event_options e
    where r.id=prefix || '_reg1' and tr.id=prefix || '_reg2' and s.id=prefix || '_sel1' and ts.id=prefix || '_sel2' and e.id=eid;
  update public.tournament_partner_invitations set requester_name='Different Person' where id=prefix || '_one';
  rejected := false;
  begin
    perform public.transition_tournament_partner_invitation(prefix || '_one','accept',prefix || '_sel1',versions);
  exception when others then rejected := sqlerrm like '%name on this request%'; end;
  if not rejected then raise exception 'direct request paired a different registered name'; end if;
  update public.tournament_partner_invitations set requester_name='Fixture 1' where id=prefix || '_one';
  result := public.transition_tournament_partner_invitation(prefix || '_one','accept',prefix || '_sel1',versions);
  if result->>'status' <> 'COMPLETED' then raise exception 'registered players were not paired'; end if;
  if (select count(*) from public.tournament_registration_team_members where tournament_id=tid and status='ACTIVE') <> 2 then raise exception 'incorrect team membership'; end if;
  if exists(select 1 from public.tournament_registration_selections where id in (prefix || '_sel1',prefix || '_sel2') and (partner_mode<>'HAS_PARTNER' or show_on_partner_board)) then raise exception 'players still need partners'; end if;
  if (select status from public.tournament_partner_invitations where id=prefix || '_competing') <> 'CANCELLED' then raise exception 'competing request not closed'; end if;
  if not (public.transition_tournament_partner_invitation(prefix || '_one','accept',prefix || '_sel1',versions)->>'idempotent')::boolean then raise exception 'repeated accept not idempotent'; end if;

  request_data := request_data || jsonb_build_object('id',prefix || '_guest','request_key',prefix || '_key3',
    'target_selection_id',prefix || '_sel4','requester_name','Guest New','requester_email','guest@example.invalid','verified',false);
  perform public.create_tournament_partner_invitation(request_data);
  result := public.transition_tournament_partner_invitation(prefix || '_guest','accept');
  if result->>'status' <> 'RESERVED' then raise exception 'guest partnership not reserved'; end if;
  created := public.create_tournament_partner_request(prefix || '_oldrequest',tid::text,eid,prefix || '_sel5',prefix || '_sel4','Fixture 4','PUBLIC_PARTNER_BOARD');
  rejected := false;
  begin
    perform public.transition_tournament_partner_request(created->>'id',prefix || '_sel4','accept');
  exception when others then rejected := sqlerrm like '%reserved partnership%'; end;
  if not rejected then raise exception 'legacy acceptance bypassed guest reservation'; end if;
  insert into public.tournament_registrations(id,tournament_id,display_name,email,wants_partner_board_contact)
    values(prefix || '_guestreg',tid::text,'Guest New','guest@example.invalid',false);
  insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode,show_on_partner_board)
    values(prefix || '_guestsel',tid::text,prefix || '_guestreg',did,eid,'NEEDS_PARTNER',false);
  select jsonb_build_object('requester_registration',r.updated_at,'target_registration',tr.updated_at,
    'requester_selection',s.updated_at,'target_selection',ts.updated_at,'event',e.updated_at) into versions
    from public.tournament_registrations r,public.tournament_registrations tr,
      public.tournament_registration_selections s,public.tournament_registration_selections ts,public.tournament_event_options e
    where r.id=prefix || '_guestreg' and tr.id=prefix || '_reg4' and s.id=prefix || '_guestsel' and ts.id=prefix || '_sel4' and e.id=eid;
  result := public.transition_tournament_partner_invitation(prefix || '_guest','complete',prefix || '_guestsel',versions);
  if result->>'status' <> 'COMPLETED' then raise exception 'guest registration did not complete partnership'; end if;
  if (select count(*) from public.tournament_registration_team_members where tournament_id=tid and status='ACTIVE') <> 4 then raise exception 'guest completion missing members'; end if;
  if (select status from public.tournament_registration_partner_requests where id=prefix || '_oldrequest') <> 'CANCELLED' then raise exception 'legacy competitor not cancelled'; end if;

  request_data := request_data || jsonb_build_object('id',prefix || '_stale','request_key',prefix || '_key4',
    'target_selection_id',prefix || '_sel6','requester_name','Fixture 7','requester_email','fixture7@example.invalid');
  perform public.create_tournament_partner_invitation(request_data);
  rejected := false;
  begin
    perform public.transition_tournament_partner_invitation(prefix || '_stale','accept',prefix || '_sel7','{}'::jsonb);
  exception when others then rejected := sqlerrm like '%Registration changed%'; end;
  if not rejected then raise exception 'stale registration versions were accepted'; end if;
  if (select status from public.tournament_partner_invitations where id=prefix || '_stale') <> 'PENDING' then raise exception 'failed acceptance changed invitation'; end if;
  perform public.transition_tournament_partner_invitation(prefix || '_stale','decline');
  if (select status from public.tournament_partner_invitations where id=prefix || '_stale') <> 'DECLINED' then raise exception 'decline failed'; end if;
  request_data := request_data || jsonb_build_object('id',prefix || '_legacy','request_key',prefix || '_key5',
    'requester_name','Fixture 5','requester_email','fixture5@example.invalid','send_directly',false);
  perform public.create_tournament_partner_invitation(request_data);
  result := public.transition_tournament_partner_invitation(prefix || '_legacy','verify');
  if result->>'status' <> 'PENDING' or result->>'verified_at' is null then raise exception 'existing verification link stopped working'; end if;
  if not public.claim_partner_invitation_delivery(prefix || '_one','completed_target','first') then raise exception 'first delivery not claimed'; end if;
  if public.claim_partner_invitation_delivery(prefix || '_one','completed_target','second') then raise exception 'duplicate delivery was claimed'; end if;
  if has_table_privilege('anon','public.tournament_partner_invitations','select')
    or has_table_privilege('authenticated','public.tournament_partner_invitations','select')
    or has_function_privilege('anon','public.transition_tournament_partner_invitation(text,text,text,jsonb)','execute') then
      raise exception 'private invitation data exposed to browser roles';
  end if;
end;
$test$;
select 'PASS: registered acceptance, guest reservation/completion, competing requests, repeat clicks, stale versions, email deduplication and private grants' as result;
rollback;
