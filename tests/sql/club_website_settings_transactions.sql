-- Staging rehearsal. No test identities, drafts, or publications are retained.
begin;
do $$
declare
  club text;
  actor uuid := gen_random_uuid();
  email text := 'website-' || gen_random_uuid() || '@invalid.example';
  state jsonb;
  published_before jsonb;
  board_before jsonb;
  version integer;
begin
  select id into club from public.clubs where is_active order by id limit 1;
  insert into auth.users(id, email, email_confirmed_at, created_at, updated_at)
    values(actor, email, now(), now(), now());
  insert into public.admin_role_assignments(club_id, user_id, email, role)
    values(club, actor, email, 'administrator');
  select revision, published into version, published_before from public.pcs_club_sites where club_id = club;
  select to_jsonb(b) into board_before from public.club_leaderboard_settings b where club_id = club;
  state := public.save_club_website_settings(actor, email, club, version, 'save',
    '{"schema_version":1,"name":"Website rehearsal","visibility":"listed","pages":[{"slug":"home","title":"Home"}],"page_visibility":{"players":"private"}}');
  if state->'published' is distinct from published_before then raise exception 'Saving changed the live website'; end if;
  begin
    perform public.save_club_website_settings(actor, email, club, version, 'publish', null);
    raise exception 'Stale revision was accepted';
  exception when serialization_failure then null; end;
  state := public.save_club_website_settings(actor, email, club, version + 1, 'publish', null);
  if state->'published' <> state->'draft' then raise exception 'Publication did not use saved draft'; end if;
  perform public.save_club_website_settings(actor, email, club, version + 2, 'save',
    '{"name":"Changed draft","visibility":"unlisted"}');
  state := public.save_club_website_settings(actor, email, club, version + 3, 'discard', null);
  if state->'draft' <> state->'published' then raise exception 'Restore did not restore published draft'; end if;
  state := public.save_club_website_settings(actor, email, club, version + 4, 'unpublish', null);
  if state->'published' <> 'null'::jsonb or state->'draft'->>'name' <> 'Website rehearsal' then
    raise exception 'Unpublishing lost the draft';
  end if;
  state := public.save_club_website_settings(actor, email, club, version + 5, 'publish', null);
  if state->'published'->>'name' <> 'Website rehearsal' then raise exception 'Republish failed'; end if;
  if (select public_site_status from public.clubs where id = club) <> 'published' then raise exception 'Club status diverged'; end if;
  if (select to_jsonb(b) from public.club_leaderboard_settings b where club_id = club) is distinct from board_before then
    raise exception 'Website editing changed independent leaderboard settings';
  end if;
  begin
    perform public.save_club_website_settings(gen_random_uuid(), email, club, version + 6, 'publish', null);
    raise exception 'Forged actor accepted';
  exception when insufficient_privilege then null; end;
  update public.admin_role_assignments set revoked_at = now() where user_id = actor;
  begin
    perform public.save_club_website_settings(actor, email, club, version + 6, 'publish', null);
    raise exception 'Revoked actor accepted';
  exception when insufficient_privilege then null; end;
  if (select count(*) from public.club_website_settings_audit where actor_id = actor) <> 6 then
    raise exception 'Website audit is incomplete';
  end if;
  if has_function_privilege('authenticated','public.save_club_website_settings(uuid,text,text,integer,text,jsonb)','EXECUTE')
    or has_function_privilege('anon','public.save_club_website_settings(uuid,text,text,integer,text,jsonb)','EXECUTE')
    or has_table_privilege('anon','public.pcs_club_sites','SELECT')
    or has_table_privilege('authenticated','public.club_website_settings_audit','SELECT') then
    raise exception 'Client access exposed';
  end if;
end $$;
rollback;
select 'Website transaction checks passed; test changes rolled back' as result;
