-- Transactional checks: all writes roll back, including created club/roles.
begin;
do $$
declare actor uuid; actor_email text; original jsonb; saved jsonb; doc jsonb; club jsonb; protected text;
begin
 select u.id,u.email into actor,actor_email from auth.users u join public.pcs_platform_admins p on p.user_id=u.id and p.revoked_at is null
 where exists(select 1 from public.admin_role_assignments a where a.club_id='cabo-test-club' and lower(a.email)=lower(u.email) and a.revoked_at is null and (a.user_id is null or a.user_id=u.id)) limit 1;
 if actor is null then raise exception 'Existing fixture administrator required'; end if;
 select md5(string_agg(to_jsonb(s)::text,',' order by club_id)) into protected from public.pcs_club_sites s where club_id<>'cabo-test-club';
 select to_jsonb(s) into original from public.pcs_club_sites s where club_id='cabo-test-club';
 doc:=original->'draft'||'{"name":"PRIVATE DRAFT CHECK","visibility":"listed"}'::jsonb;
 saved:=public.pcs_write_club_site(actor,actor_email,'cabo-test-club',(original->>'revision')::integer,'save',doc);
 if saved->'published' is distinct from original->'published' then raise exception 'Draft leaked to published site'; end if;
 begin
  perform public.pcs_write_club_site(actor,actor_email,'cabo-test-club',(original->>'revision')::integer,'publish',null);
  raise exception 'Stale publication unexpectedly succeeded';
 exception when serialization_failure then null; end;
 begin
  perform public.pcs_write_club_site(gen_random_uuid(),'unassigned@example.invalid','cabo-test-club',(saved->>'revision')::integer,'publish',null);
  raise exception 'Foreign user unexpectedly published';
 exception when insufficient_privilege then null; end;
 saved:=public.pcs_write_club_site(actor,actor_email,'cabo-test-club',(saved->>'revision')::integer,'publish',null);
 if saved->'published'->>'name'<>'PRIVATE DRAFT CHECK' then raise exception 'Publication not atomic'; end if;
 if not exists(select 1 from public.pcs_public_club_directory where slug='cabo-test-club' and name='PRIVATE DRAFT CHECK') then raise exception 'Published listed site missing'; end if;
 doc:=doc||'{"visibility":"unlisted"}'::jsonb;
 saved:=public.pcs_write_club_site(actor,actor_email,'cabo-test-club',(saved->>'revision')::integer,'save',doc);
 if not exists(select 1 from public.pcs_public_club_directory where slug='cabo-test-club') then raise exception 'Draft visibility changed directory'; end if;
 saved:=public.pcs_write_club_site(actor,actor_email,'cabo-test-club',(saved->>'revision')::integer,'publish',null);
 if exists(select 1 from public.pcs_public_club_directory where slug='cabo-test-club') then raise exception 'Unlisted club in directory'; end if;
 saved:=public.pcs_write_club_site(actor,actor_email,'cabo-test-club',(saved->>'revision')::integer,'unpublish',null);
 if saved->'published'<>'null'::jsonb or (select public_site_status from public.clubs where id='cabo-test-club')<>'draft' then raise exception 'Unpublish failed'; end if;
 if protected is distinct from (select md5(string_agg(to_jsonb(s)::text,',' order by club_id)) from public.pcs_club_sites s where club_id<>'cabo-test-club') then raise exception 'Other club changed'; end if;
 club:=public.pcs_create_own_club(actor,actor_email,'qa-public-rollback-club','Rollback club',doc||'{"name":"Rollback club"}'::jsonb);
 if (select public_site_status from public.clubs where id=club->>'club_id')<>'draft' or not exists(select 1 from public.admin_role_assignments where club_id=club->>'club_id' and user_id=actor and role='administrator') then raise exception 'Self signup failed'; end if;
 begin
  perform public.pcs_create_own_club(actor,actor_email,'cabo-test-club','Takeover',doc);
  raise exception 'Existing club was claimed';
 exception when unique_violation then null; end;
 begin
  perform public.pcs_create_own_club(gen_random_uuid(),'unverified@example.invalid','qa-unverified-club','Unverified',doc);
  raise exception 'Unverified account created club';
 exception when insufficient_privilege then null; end;
 if has_table_privilege('anon','public.pcs_club_sites','select') or has_table_privilege('authenticated','public.pcs_club_sites','select')
 or has_function_privilege('authenticated','public.pcs_write_club_site(uuid,text,text,integer,text,jsonb)','execute')
 or has_function_privilege('anon','public.pcs_create_own_club(uuid,text,text,text,jsonb)','execute') then raise exception 'Direct public access granted'; end if;
end $$;
select 'passed: draft isolation, revision conflicts, foreign identity denial, listed/unlisted/unpublish, other-club isolation, verified signup, duplicate club protection, service-only grants' as checks;
rollback;
