-- Staging-only transaction rehearsal. All people, clubs and seasons roll back.
begin;
do $$
declare
 home text:='late-organizer-'||gen_random_uuid()::text; away text:='late-club-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='late-request-qa@example.invalid'; sid uuid:=gen_random_uuid();
 ids bigint[]:='{}'; pid bigint:=8500000000000+(random()*1000000000)::bigint; i integer; mid uuid; next_mid uuid;
 pending uuid; rejected uuid; after_cutoff uuid; baseline uuid; fake uuid:=gen_random_uuid(); first_entry uuid;
 result jsonb; member public.pcs_interclub_pool_members; original_window jsonb; cutoff timestamptz; share uuid; count_before integer;
begin
 if has_table_privilege('anon','public.pcs_interclub_late_player_requests','SELECT')
  or has_table_privilege('authenticated','public.pcs_interclub_late_player_requests','INSERT')
  or has_table_privilege('service_role','public.pcs_interclub_late_player_requests','UPDATE')
  or has_function_privilege('authenticated','public.pcs_request_interclub_late_player(uuid,text,text,uuid,bigint,jsonb,text)','EXECUTE')
  or not(select relrowsecurity from pg_class where oid='public.pcs_interclub_late_player_requests'::regclass) then
  raise exception 'Late-request privileges are incorrect'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
  values(home,home,'Late organizer',false,'draft','draft'),(away,away,'Late requesting club',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
  values(home,actor_email,'administrator',actor),(away,actor_email,'administrator',actor);
 for i in 1..7 loop
  ids:=array_append(ids,pid+i);
  insert into public.players(id,club_id,name,normalized_name,rating,active,gender)
   values(pid+i,case when i=7 then home else away end,'Late QA '||i,'late qa '||i,1160,i<>6,'male');
 end loop;
 perform public.pcs_save_interclub_draft(actor,actor_email,home,sid,0,jsonb_build_object('name','Late Request QA','start_date',(current_date+1)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions','["3.0"]'::jsonb,'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
   jsonb_build_object('host_club_id',home,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '20 days','duration_minutes',180,'courts',4),
   jsonb_build_object('host_club_id',away,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '40 days','duration_minutes',180,'courts',4))));
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,home,sid,1,'{"3.0":{"min_rating":null,"max_rating":3.499,"women_required":2}}');
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,actor_email,away,sid,away,1,'accept');
 select id into mid from public.pcs_interclub_meets where season_id=sid and plan_index=0;
 select id into next_mid from public.pcs_interclub_meets where season_id=sid and plan_index=1;
 select share_id into share from public.pcs_interclub_pool_settings where season_id=sid and club_id=away;
 begin
  perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[1],'["3.0"]','Unconfigured');
  raise exception 'Unconfigured registration allowed late request';
 exception when sqlstate 'PT423' then null; end;
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,now()+interval '1 day',now()+interval '2 days');
 begin
  perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[1],'["3.0"]','Scheduled');
  raise exception 'Scheduled registration allowed late request';
 exception when sqlstate 'PT423' then null; end;
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,1,now()-interval '1 day',now()+interval '1 day');
 begin
  perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[1],'["3.0"]','Use ordinary signup');
  raise exception 'Open registration used late-request exception';
 exception when sqlstate 'PT423' then null; end;
 perform public.pcs_interclub_pool_bulk_add(actor,actor_email,away,sid,jsonb_build_array(jsonb_build_object('player_id',ids[4],'divisions','["3.0"]'::jsonb)));
 select id into baseline from public.pcs_interclub_pool_members where season_id=sid and player_id=ids[4];
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,2,now()-interval '1 day',now()-interval '1 hour');
 select jsonb_build_array(registration_opens_at,registration_closes_at,registration_revision) into original_window from public.pcs_interclub_seasons where id=sid;
 begin
  perform public.pcs_request_interclub_late_player(gen_random_uuid(),actor_email,away,sid,ids[1],'["3.0"]','Forged actor');
  raise exception 'Forged actor submitted request';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set revoked_at=now() where club_id=away;
 begin
  perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[1],'["3.0"]','Revoked');
  raise exception 'Revoked administrator submitted request';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set revoked_at=null where club_id=away;
 for i in 6..7 loop
  begin
   perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[i],'["3.0"]','Inactive or foreign');
   raise exception 'Inactive or foreign club player requested';
  exception when invalid_parameter_value then null; end;
 end loop;
 begin
  perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[1],'["4.0"]','Unknown division');
  raise exception 'Foreign season division accepted';
 exception when invalid_parameter_value then null; end;
 begin
  insert into public.pcs_interclub_late_player_requests(member_id,season_id,club_id,player_id,requested_by,reason)
   values(baseline,sid,away,ids[4],actor,'Forged association');
  raise exception 'Existing signup gained late-request authority';
 exception when invalid_parameter_value then null; end;
 -- The trusted admin path records a pending request even before season start.
 set local role service_role;
 result:=public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[1],'["3.0"]','Returning member after signup closed');
 reset role;
 pending:=(result->>'id')::uuid;
 if result->>'approval_status'<>'pending' or (result->>'late_join')::boolean is not true or result->>'email'<>'' or result->>'consent_at' is not null then
  raise exception 'Late request autoapproved or invented contact consent'; end if;
 if exists(select 1 from public.pcs_interclub_entries where season_id=sid and player_id=ids[1]) then raise exception 'Pending request received league rating entry'; end if;
 perform public.pcs_interclub_pool_action(actor,actor_email,away,sid,'member',jsonb_build_object('member_id',pending,'expected_revision',1,'player_id',ids[1],'status','active'));
 select * into member from public.pcs_interclub_pool_members where id=pending;
 if member.approval_status<>'pending' or not member.late_join then raise exception 'Ordinary member edit autoapproved request'; end if;
 begin
  perform public.pcs_interclub_pool_action(actor,actor_email,away,sid,'member',jsonb_build_object('member_id',pending,'expected_revision',member.revision,'player_id',ids[5],'status','active'));
  raise exception 'Requested identity reassigned';
 exception when invalid_parameter_value then null; end;
 begin
  update public.pcs_interclub_late_player_requests set reason='Changed after request' where member_id=pending;
  raise exception 'Submitted request mutated';
 exception when insufficient_privilege then null; end;
 select count(*) into count_before from public.pcs_interclub_pool_members where season_id=sid;
 begin
  perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[1],'["3.0"]','Lost response retry');
  raise exception 'Retry duplicated pending request';
 exception when unique_violation then null; end;
 if (select count(*) from public.pcs_interclub_pool_members where season_id=sid)<>count_before then raise exception 'Duplicate changed pool'; end if;
 begin
  perform public.pcs_save_interclub_meet_roster(actor,actor_email,away,sid,mid,1,gen_random_uuid(),0,'Pending pair','3.0',array[ids[1],ids[4]]);
  raise exception 'Pending request entered roster';
 exception when invalid_parameter_value then null; end;
 -- Ordinary public intake, bulk additions and raw pool inserts stay closed.
 begin
  perform public.pcs_interclub_pool_bulk_add(actor,actor_email,away,sid,jsonb_build_array(jsonb_build_object('player_id',ids[5],'late_join',true)));
  raise exception 'Bulk add bypassed closed registration';
 exception when sqlstate 'PT423' then null; end;
 begin
  perform public.pcs_interclub_pool_public_action('signup',jsonb_build_object('share_id',share,'late_request',true),repeat('q',64));
  raise exception 'Public signup bypassed closed registration';
 exception when sqlstate 'PT423' then null; end;
 begin
  insert into public.pcs_interclub_pool_members(id,season_id,club_id,name,email,player_id,request_id,request_fingerprint,consent_at,late_join)
   values(fake,sid,away,'Raw addition','',ids[5],gen_random_uuid(),'forged',null,true);
  raise exception 'Client-controlled late flag bypassed pool guard';
 exception when sqlstate 'PT423' then null; end;
 begin
  perform public.pcs_review_interclub_pool_member(actor,actor_email,away,sid,pending,member.revision,true,'Club self-approval');
  raise exception 'Requesting club approved its own late player';
 exception when insufficient_privilege then null; end;
 perform public.pcs_review_interclub_pool_member(actor,actor_email,home,sid,pending,member.revision,true,'Commissioner accepts late player');
 select id into first_entry from public.pcs_interclub_entries where season_id=sid and player_id=ids[1];
 if first_entry is null or (select starting_rating from public.pcs_interclub_entries where id=first_entry)<>2.9 then raise exception 'Approval did not preserve source rating'; end if;
 begin
  perform public.pcs_review_interclub_pool_member(actor,actor_email,home,sid,pending,member.revision,false,'Stale decision');
  raise exception 'Stale commissioner revision accepted';
 exception when sqlstate 'PT409' then null; end;
 perform public.pcs_save_interclub_meet_roster(actor,actor_email,away,sid,next_mid,1,gen_random_uuid(),0,'Approved pair','3.0',array[ids[1],ids[4]]);
 result:=public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[2],'["3.0"]','Second request'); rejected:=(result->>'id')::uuid;
 perform public.pcs_review_interclub_pool_member(actor,actor_email,home,sid,rejected,1,false,'Declined late request');
 if exists(select 1 from public.pcs_interclub_entries where season_id=sid and player_id=ids[2]) then raise exception 'Rejected player received rating seed'; end if;
 begin
  perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[2],'["3.0"]','Retry rejected');
  raise exception 'Rejected request bypassed existing signup';
 exception when unique_violation then null; end;
 perform public.pcs_interclub_pool_action(actor,actor_email,away,sid,'member',jsonb_build_object('member_id',rejected,'expected_revision',2,'player_id',ids[2],'status','withdrawn'));
 begin
  perform public.pcs_interclub_pool_action(actor,actor_email,away,sid,'member',jsonb_build_object('member_id',rejected,'expected_revision',3,'player_id',ids[2],'status','active'));
  raise exception 'Late request authorized closed restoration';
 exception when sqlstate 'PT423' then null; end;
 begin
  perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[2],'["3.0"]','Retry withdrawn');
  raise exception 'Withdrawn request duplicated';
 exception when unique_violation then null; end;
 -- Actual commissioner approval follows the transaction-start cutoff (also
 -- reproduces the effect of a transaction that waited for the season lock).
 result:=public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[3],'["3.0"]','Needs future meet only'); after_cutoff:=(result->>'id')::uuid;
 cutoff:=now();
 update public.pcs_interclub_meets set roster_deadline=cutoff where id=mid;
 perform public.pcs_review_interclub_pool_member(actor,actor_email,home,sid,after_cutoff,1,true,'Approved after this meet cutoff');
 if (select approved_at from public.pcs_interclub_pool_members where id=after_cutoff)<=cutoff then raise exception 'Approval was backdated to transaction start'; end if;
 perform public.pcs_lock_interclub_meet_eligibility(mid);
 if exists(select 1 from public.pcs_interclub_meet_eligibility_snapshots where meet_id=mid and player_id=ids[3]) then raise exception 'Late approval entered past deadline snapshot'; end if;
 begin
  perform public.pcs_save_interclub_meet_roster(actor,actor_email,away,sid,mid,1,gen_random_uuid(),0,'Too late for cutoff','3.0',array[ids[3],ids[4]]);
  raise exception 'Late approval entered closed meet roster';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_assert_interclub_competition_eligibility(sid,mid,jsonb_build_object('encounters',jsonb_build_array(
   jsonb_build_object('division','3.0','club_a',away,'club_b',home,'pairings',jsonb_build_array(
    jsonb_build_object('kind','men','players_a',jsonb_build_array(
     (select id from public.pcs_interclub_entries where season_id=sid and player_id=ids[3]),
     (select id from public.pcs_interclub_entries where season_id=sid and player_id=ids[4])),
     'players_b','[]'::jsonb,'games','[]'::jsonb))))),'regular');
  raise exception 'Late approval entered competition past roster cutoff';
 exception when invalid_parameter_value then null; end;
 if (select count(*) from public.pcs_interclub_late_player_requests where season_id=sid)<>3
  or (select count(*) from public.pcs_interclub_registration_audit where season_id=sid and action='late_player_requested')<>3
  or (select count(*) from public.pcs_interclub_registration_audit where season_id=sid and action='pool_approval')<>3 then raise exception 'Request or decision audit missing'; end if;
 if exists(select 1 from public.players where id=any(ids) and rating<>1160) then raise exception 'Late request changed club ratings'; end if;
 if (select jsonb_build_array(registration_opens_at,registration_closes_at,registration_revision) from public.pcs_interclub_seasons where id=sid) is distinct from original_window then
  raise exception 'Late request reopened registration'; end if;
 update public.pcs_interclub_seasons set details=jsonb_set(details,'{end_date}',to_jsonb((current_date-1)::text)) where id=sid;
 begin
  perform public.pcs_request_interclub_late_player(actor,actor_email,away,sid,ids[5],'["3.0"]','Season ended');
  raise exception 'Ended season accepted request';
 exception when sqlstate 'PT409' then null; end;
end $$;
rollback;
