-- Staging-only rollback rehearsal: synthetic players, pool entries and lineups.
begin;
do $$
declare
 home text:='playup-home-'||gen_random_uuid()::text; away text:='playup-away-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='playup-qa@example.invalid'; sid uuid:=gen_random_uuid(); mid uuid;
 ids bigint[]:='{}'; entries uuid[]:='{}'; pid bigint:=8300000000000+(random()*1000000000)::bigint;
 i integer; cid text; division text; bad numeric; bad_division text; team uuid; result jsonb; details jsonb; doc jsonb;
 divisions jsonb:='["3.0","3.5","4.0","4.5","Open","4.5/Open"]'; rules jsonb:='{}'; initial_window jsonb;
begin
 -- Upper caps are exact and exclusive, with no lower-rating restriction.
 foreach division in array array['2.5','3.0','3.5','4.0','4.5','5.0','Open','4.5/Open'] loop
  if not public.pcs_interclub_rating_in_division(2.9,division)
   or not public.pcs_interclub_rating_in_division(0.001,division) then raise exception 'Play up rejected for %',division; end if;
  foreach bad in array array[null::numeric,0,-1,'NaN'::numeric,'Infinity'::numeric,'-Infinity'::numeric] loop
   if public.pcs_interclub_rating_in_division(bad,division) is distinct from false then raise exception 'Invalid rating % accepted for %',bad,division; end if;
  end loop;
  if division not in ('Open','4.5/Open') then
   if not public.pcs_interclub_rating_in_division(division::numeric+0.499999,division)
    or public.pcs_interclub_rating_in_division(division::numeric+0.5,division)
    or public.pcs_interclub_rating_in_division(division::numeric+0.500001,division) then raise exception 'Upper cap incorrect for %',division; end if;
  elsif not public.pcs_interclub_rating_in_division(100,division) then raise exception 'Open has an upper cap'; end if;
 end loop;
 foreach bad_division in array array[null::text,'','bad','3.25','NaN','Infinity'] loop
  if public.pcs_interclub_rating_in_division(2.9,bad_division) is distinct from false then raise exception 'Invalid division accepted'; end if;
 end loop;
 if has_function_privilege('anon','public.pcs_interclub_rating_in_division(numeric,text)','EXECUTE')
  or has_function_privilege('authenticated','public.pcs_interclub_rating_in_division(numeric,text)','EXECUTE') then raise exception 'Private rating helper exposed'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
  values(home,home,'Play-up Home',false,'draft','draft'),(away,away,'Play-up Away',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
  values(home,actor_email,'administrator',actor),(away,actor_email,'administrator',actor);
 for division in select jsonb_array_elements_text(divisions) loop
  rules:=rules||jsonb_build_object(division,jsonb_build_object('min_rating',null,'max_rating',
   case when division in ('Open','4.5/Open') then null else division::numeric+0.499 end,'women_required',2));
 end loop;
 perform public.pcs_save_interclub_draft(actor,actor_email,home,sid,0,jsonb_build_object('name','Play-up QA','start_date',(current_date+1)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',divisions,'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
   jsonb_build_object('host_club_id',home,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '20 days','duration_minutes',180,'courts',4))));
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,home,sid,1,rules);
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,actor_email,away,sid,away,1,'accept');
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,now()-interval '3 days',now()+interval '1 day');
 select id into mid from public.pcs_interclub_meets where season_id=sid;
 for i in 1..9 loop
  cid:=case when i<=4 or i=9 then home else away end;
  ids:=array_append(ids,pid+i);
  insert into public.players(id,club_id,name,normalized_name,rating,active,gender)
   values(pid+i,cid,'Play-up QA '||i,'play-up qa '||i,case when i=9 then 1400 else 1160 end,true,case when (i-1)%4<2 then 'female' else 'male' end);
  -- Directory-only candidates already display the complete play-up choices.
  details:=public.pcs_interclub_pool_player_details(sid,cid,array[pid+i]);
  if i<>9 and (details->0->'eligible_divisions' is distinct from divisions or details->0->>'league_rating' is not null) then
   raise exception 'Unseeded 2.9 candidate lost play-up divisions'; end if;
  result:=public.pcs_interclub_pool_bulk_add(actor,actor_email,cid,sid,jsonb_build_array(
   jsonb_build_object('player_id',pid+i,'email','','divisions',case when i=9 then '["3.5"]'::jsonb else '["3.0"]'::jsonb end)));
  if result->>'added_count'<>'1' then raise exception 'Candidate not added'; end if;
  entries:=array_append(entries,(select id from public.pcs_interclub_entries where season_id=sid and player_id=pid+i));
 end loop;
 -- An established league rating, not a later source-club change, governs play up.
 update public.players set rating=2400 where id=ids[1];
 details:=public.pcs_interclub_pool_player_details(sid,home,array[ids[1]]);
 if (details->0->>'league_rating')::numeric<>2.9 or details->0->'eligible_divisions' is distinct from divisions then
  raise exception 'Club rating replaced the league rating authority'; end if;
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,1,now()-interval '3 days',now()-interval '2 days');
 select jsonb_build_array(registration_opens_at,registration_closes_at,registration_revision) into initial_window from public.pcs_interclub_seasons where id=sid;
 -- The same real 2.9 players can form an eligible 3.0 or higher lineup. A saved
 -- preference for 3.0 is preserved and does not impose an eligibility minimum.
 for division in select jsonb_array_elements_text(divisions) loop
  team:=gen_random_uuid();
  result:=public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,team,0,'Play up '||division,division,ids[1:4]);
  if result->'roster'->>'status'<>'eligible' then raise exception '2.9 roster rejected from %',division; end if;
  perform public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,team,1,'','',array[]::bigint[],true);
 end loop;
 begin
  perform public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,gen_random_uuid(),0,'Over cap','3.0',array[ids[9],ids[2],ids[3],ids[4]]);
  raise exception '3.5 player entered below exclusive upper cap';
 exception when invalid_parameter_value then null; end;
 -- At the meet deadline, actual competition pairings share the same rule and
 -- retain the frozen league rating even when the live seed changes afterwards.
 update public.pcs_interclub_meets set roster_deadline=now() where id=mid;
 perform public.pcs_lock_interclub_meet_eligibility(mid);
 doc:=jsonb_build_object('encounters',jsonb_build_array(jsonb_build_object('division','4.0','club_a',home,'club_b',away,
  'pairings',jsonb_build_array(jsonb_build_object('kind','women','players_a',to_jsonb(entries[1:2]),'players_b',to_jsonb(entries[5:6]),'games','[]'::jsonb)))));
 perform public.pcs_assert_interclub_competition_eligibility(sid,mid,doc,'regular');
 update public.pcs_interclub_entries set starting_rating=5.0 where id=entries[1];
 perform public.pcs_assert_interclub_competition_eligibility(sid,mid,doc,'regular');
 doc:=jsonb_set(jsonb_set(doc,'{encounters,0,division}','"3.0"'),'{encounters,0,pairings,0,players_a}',to_jsonb(array[entries[9],entries[2]]));
 begin
  perform public.pcs_assert_interclub_competition_eligibility(sid,mid,doc,'regular');
  raise exception 'Competition override admitted a player at the upper cap';
 exception when invalid_parameter_value then null; end;
 if exists(select 1 from public.pcs_interclub_pool_members m where m.season_id=sid and (m.divisions<>'["3.0"]'::jsonb and m.player_id<>ids[9]
  or m.email<>'' or m.consent_at is not null or m.approval_status<>'approved')) then raise exception 'Play up rewrote saved preferences or consent'; end if;
 if (select jsonb_build_array(registration_opens_at,registration_closes_at,registration_revision) from public.pcs_interclub_seasons where id=sid) is distinct from initial_window then
  raise exception 'Eligibility changed commissioner registration dates'; end if;
 if exists(select 1 from public.players where id=any(ids) and rating<>case when id=ids[1] then 2400 when id=ids[9] then 1400 else 1160 end) then
  raise exception 'Eligibility changed source player ratings'; end if;
end $$;
rollback;
