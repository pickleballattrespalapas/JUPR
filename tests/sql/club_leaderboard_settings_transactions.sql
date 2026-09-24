-- Execute as a single transaction on staging. Every test change rolls back.
begin;
do $$
declare
  club text;
  actor uuid := gen_random_uuid();
  email text := 'leaderboard-' || gen_random_uuid() || '@invalid.example';
  result jsonb;
  version integer;
begin
  select id into club from public.clubs order by id limit 1;
  insert into auth.users(id,email,email_confirmed_at,created_at,updated_at)
    values(actor,email,now(),now(),now());
  insert into public.admin_role_assignments(club_id,user_id,email,role)
    values(club,actor,email,'administrator');
  select coalesce((select revision from public.club_leaderboard_settings where club_id=club),0) into version;
  result := public.save_club_leaderboard_settings(club,actor,email,version,'save','{"cards":["most_wins"],"seasons":[]}');
  if result->'draft'->'cards' <> '["most_wins"]' then raise exception 'Draft was not saved'; end if;
  begin
    perform public.save_club_leaderboard_settings(club,actor,email,version,'publish');
    raise exception 'Stale revision was accepted';
  exception when serialization_failure then null; end;
  result := public.save_club_leaderboard_settings(club,actor,email,version+1,'publish');
  if result->'published' <> result->'draft' then raise exception 'Publish lost the stored draft'; end if;
  perform public.save_club_leaderboard_settings(club,actor,email,version+2,'save','{"cards":[]}');
  result := public.save_club_leaderboard_settings(club,actor,email,version+3,'discard');
  if result->'draft' <> result->'published' then raise exception 'Discard lost published settings'; end if;
  begin
    perform public.save_club_leaderboard_settings(club,gen_random_uuid(),email,version+4,'publish');
    raise exception 'Forged actor accepted';
  exception when insufficient_privilege then null; end;
  update public.admin_role_assignments set revoked_at=now() where user_id=actor;
  begin
    perform public.save_club_leaderboard_settings(club,actor,email,version+4,'publish');
    raise exception 'Revoked actor accepted';
  exception when insufficient_privilege then null; end;
  if (select count(*) from public.club_leaderboard_settings_audit where actor_id=actor) <> 4 then
    raise exception 'Settings audit does not match committed actions';
  end if;
  if has_function_privilege('authenticated','public.save_club_leaderboard_settings(text,uuid,text,integer,text,jsonb)','EXECUTE')
     or has_table_privilege('anon','public.club_leaderboard_settings','SELECT') then
    raise exception 'Client access exposed';
  end if;
end $$;
rollback;
select 'Leaderboard transaction checks passed; all test changes rolled back' as result;
