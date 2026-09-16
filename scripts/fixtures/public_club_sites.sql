-- Synthetic public site fixtures. Staging only; never replaces later edits.
begin;
do $$
declare actor uuid; actor_email text; cid text; c public.clubs; doc jsonb; saved jsonb;
 season public.pcs_interclub_seasons; publication jsonb; clubs jsonb; meets jsonb; results jsonb;
begin
 perform pg_advisory_xact_lock(hashtextextended('pcs-public-site-fixture-v1',0));
 if not exists(select 1 from public.clubs where id='tres_palapas' and name='Tres Palapas — Staging Fixtures' and features_json @> '{"staging_fixture":true,"synthetic_data_only":true}')
  or (select count(*) from public.clubs where id in ('la-ribera-pickelball-club','cabo-test-club','la-paz-test-club') and features_json->>'isolation_fixture' is not null)<>3 then
  raise exception 'Staging isolation fixtures required'; end if;
 if exists(select 1 from public.pcs_platform_audit where action='public_site_fixture_v1') then return; end if;
 select u.id,u.email into actor,actor_email from auth.users u join public.pcs_platform_admins p on p.user_id=u.id and p.revoked_at is null
 where exists(select 1 from public.admin_role_assignments a where a.club_id='la-ribera-pickelball-club' and lower(a.email)=lower(u.email) and a.revoked_at is null and (a.user_id is null or a.user_id=u.id)) limit 1;
 if actor is null then raise exception 'Existing fixture administrator required'; end if;
 foreach cid in array array['la-ribera-pickelball-club','cabo-test-club','la-paz-test-club'] loop
  select * into c from public.clubs where id=cid;
  select draft into doc from public.pcs_club_sites where club_id=cid;
  doc:=doc||jsonb_build_object('name',c.name,'description','A synthetic staging club for testing players, matches and public websites. No real visitor arrangements are advertised.',
    'location',case when cid='cabo-test-club' then 'Cabo · test location' when cid='la-paz-test-club' then 'La Paz · test location' else 'La Ribera · test location' end,
    'visitor_info','Test website only. Use the links below to explore this club’s separate players, leagues and results.',
    'accent',case when cid='cabo-test-club' then '#0f766e' when cid='la-paz-test-club' then '#7e22ce' else '#1d4ed8' end,
    'visibility',case when cid='la-paz-test-club' then 'unlisted' else 'listed' end,
    'display',case when cid='cabo-test-club' then '{"singles":false,"badges":false}'::jsonb else '{}'::jsonb end,
    'pages',jsonb_build_array(jsonb_build_object('slug','home','title','Home','in_navigation',true,'blocks',jsonb_build_array(
      jsonb_build_object('id','welcome','kind','text','heading','Welcome to '||c.name,'text','This content belongs only to this test club. Change the draft to test preview and publication.','url','','alt','','span',6,'align','left','tone','soft','padding','medium'),
      jsonb_build_object('id','visit','kind','button','heading','Visit our information page','text','Visitor information','url','/clubs/'||c.slug||'/pages/visiting','alt','','span',6,'align','left','tone','plain','padding','medium'))),
      jsonb_build_object('slug','visiting','title','Visiting','in_navigation',true,'blocks',jsonb_build_array(jsonb_build_object('id','details','kind','text','heading','Visiting '||c.name,'text','A custom page for this test club. Edit its content and layout in Club website.','url','','alt','','span',12,'align','left','tone','soft','padding','large')))));
  saved:=public.pcs_write_club_site(actor,actor_email,cid,(select revision from public.pcs_club_sites where club_id=cid),'save',doc);
  perform public.pcs_write_club_site(actor,actor_email,cid,(saved->>'revision')::integer,'publish',null);
 end loop;
 select * into season from public.pcs_interclub_seasons where organizer_club_id='la-ribera-pickelball-club' and details->>'name'='Three Club Isolation Test';
 if found then
  select jsonb_agg(jsonb_build_object('id',selected_club.id,'name',selected_club.name) order by selected_club.name) into clubs
   from public.clubs selected_club join public.pcs_interclub_participations p on selected_club.id=p.club_id where p.season_id=season.id and p.status='accepted';
  select jsonb_agg(jsonb_build_object('id',m.id,'host_club_id',m.host_club_id,'club_ids',m.club_ids,'starts_at',m.starts_at,'duration_minutes',m.duration_minutes,'courts',m.courts) order by m.starts_at) into meets
   from public.pcs_interclub_meets m where m.season_id=season.id;
  -- Schedule is public; encounter results begin empty so testers can enter and preview their own scores.
  results:='{"results":[]}'::jsonb;
  saved:=public.pcs_write_interclub_publication(actor,actor_email,season.organizer_club_id,season.id,0,'save',results);
  publication:=jsonb_build_object('name',season.details->>'name','start_date',season.details->>'start_date','end_date',season.details->>'end_date',
    'timezone',season.details->>'timezone','divisions',season.details->'divisions','clubs',clubs,'meets',meets,'results','[]'::jsonb);
  perform public.pcs_write_interclub_publication(actor,actor_email,season.organizer_club_id,season.id,(saved->>'revision')::integer,'publish',publication);
 end if;
 insert into public.pcs_platform_audit(actor_id,club_id,action) values(actor,'la-ribera-pickelball-club','public_site_fixture_v1');
end $$;
commit;
