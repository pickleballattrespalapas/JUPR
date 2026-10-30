-- Staging only; disposable fixtures and every write roll back.
begin;
set local statement_timeout = '45s';
set local role service_role;
do $test$
declare
  club text; tour uuid := gen_random_uuid(); suffix text := replace(tour::text,'-','');
  rid text := 'withdraw_' || suffix; partner text := 'partner_' || suffix;
  sid text := 'entry_' || suffix; psid text := 'partnerentry_' || suffix;
  dayid text := 'day_' || suffix; eid text := 'event_' || suffix;
  stamp timestamptz; versions jsonb; bad_versions jsonb; result jsonb;
  email_hash text; scenario text; rejected boolean;
begin
  if has_function_privilege('anon','public.server_withdraw_public_tournament_registration(text,text,text,text,timestamptz,jsonb)','execute')
     or has_function_privilege('authenticated','public.server_withdraw_public_tournament_registration(text,text,text,text,timestamptz,jsonb)','execute') then
    raise exception 'Withdrawal is exposed to public database clients';
  end if;
  select id::text into strict club from public.clubs order by id limit 1;
  insert into public.tournaments(id,club_id,name,status,team_count) values(tour,club,'Disposable withdrawal test','DRAFT',8);
  insert into public.tournament_registration_days(id,tournament_id,sort_order,label) values(dayid,tour::text,0,'Test');
  insert into public.tournament_event_options(id,tournament_id,registration_day_id,sort_order,label,event_type,partner_required)
    values(eid,tour::text,dayid,0,'Doubles','GENDER_DOUBLES',true);
  insert into public.tournament_registrations(id,tournament_id,display_name,email,status,age,gender,doubles_skill,payment_status)
    values(rid,tour::text,'Withdrawal Fixture',rid || '@example.invalid','confirmed',42,'Men',4,'paid'),
          (partner,tour::text,'Partner Fixture',partner || '@example.invalid','confirmed',42,'Men',4,'unpaid');
  insert into public.tournament_registration_selections(id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode,sort_order)
    values(sid,tour::text,rid,dayid,eid,'HAS_PARTNER',0),(psid,tour::text,partner,dayid,eid,'HAS_PARTNER',0);
  insert into public.tournament_registration_team_links(id,tournament_id,event_option_id,registration1_id,registration2_id,selection1_id,selection2_id,status)
    values('link_' || suffix,tour,eid,rid,partner,sid,psid,'ADMIN_CONFIRMED');
  select updated_at,encode(sha256(convert_to(email,'UTF8')),'hex') into stamp,email_hash from public.tournament_registrations where id=rid;
  select jsonb_agg(jsonb_build_object('id',id,'updated_at',updated_at)) into versions from public.tournament_registration_selections where registration_id=rid;
  foreach scenario in array array['EMAIL','CLUB','REGISTRATION_VERSION','SELECTION_VERSION','MISSING_SELECTION','DUPLICATE_SELECTION'] loop
    bad_versions := case scenario when 'SELECTION_VERSION' then jsonb_build_array(jsonb_build_object('id',sid,'updated_at','2000-01-01T00:00:00Z'))
      when 'MISSING_SELECTION' then '[]'::jsonb when 'DUPLICATE_SELECTION' then versions || versions else versions end;
    rejected := false;
    begin
      perform public.server_withdraw_public_tournament_registration(case when scenario='CLUB' then 'other' else club end,tour::text,rid,
        case when scenario='EMAIL' then 'wrong' else email_hash end,
        case when scenario='REGISTRATION_VERSION' then '2000-01-01T00:00:00Z'::timestamptz else stamp end,bad_versions);
    exception when others then
      if position('JUPR_WITHDRAW_IDENTITY' in sqlerrm)=0 and position('JUPR_CANCEL_CONFLICT' in sqlerrm)=0 then raise; end if;
      rejected := true;
    end;
    if not rejected or not exists(select 1 from public.tournament_registrations where id=rid)
      or not exists(select 1 from public.tournament_registration_team_links where id='link_' || suffix) then
      raise exception 'Unsafe withdrawal in %', scenario;
    end if;
  end loop;
  result := public.server_withdraw_public_tournament_registration(club,tour::text,rid,email_hash,stamp,versions);
  if result->>'withdrawn' <> 'true' or exists(select 1 from public.tournament_registrations where id=rid)
    or (select partner_mode from public.tournament_registration_selections where id=psid) <> 'NEEDS_PARTNER'
    or not exists(select 1 from private.tournament_registration_cancellations c where c.registration_id=rid
      and c.before_json->'registration'->>'payment_status'='paid' and c.actor_email='public-withdrawal:' || rid) then
    raise exception 'Withdrawal did not retain audit/payment evidence and release partner';
  end if;
  insert into public.tournament_registrations(id,tournament_id,display_name,email,status)
    values('fresh_' || suffix,tour::text,'Fresh Fixture',rid || '@example.invalid','confirmed');
  result := public.server_withdraw_public_tournament_registration(club,tour::text,rid,email_hash,stamp,versions);
  if result->>'idempotent' <> 'true' or not exists(select 1 from public.tournament_registrations where id='fresh_' || suffix)
    or (select count(*) from private.tournament_registration_cancellations c where c.registration_id=rid) <> 1 then
    raise exception 'Lost-response retry affected a fresh registration or duplicated the audit';
  end if;
end $test$;
select 'PASS: identity, stale snapshots, partner release, paid audit, safe retry, public privileges' as result;
rollback;
