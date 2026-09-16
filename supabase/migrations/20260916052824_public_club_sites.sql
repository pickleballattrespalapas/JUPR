begin;

alter table public.clubs add column public_site_status text not null default 'draft'
 check(public_site_status in ('draft','published'));
create table public.pcs_club_sites (
 club_id text primary key references public.clubs(id),
 revision integer not null default 1 check(revision>0),
 draft jsonb not null check(jsonb_typeof(draft)='object'),
 published jsonb check(published is null or jsonb_typeof(published)='object'),
 published_at timestamptz,
 updated_at timestamptz not null default now()
);
alter table public.pcs_club_sites enable row level security;
revoke all on public.pcs_club_sites from public,anon,authenticated;
grant all on public.pcs_club_sites to service_role;

-- Preserve existing public websites without newly listing them in a directory.
-- Their administrators explicitly choose listed visibility at publication.
insert into public.pcs_club_sites(club_id,draft,published,published_at)
select id,doc,case when is_active then doc end,case when is_active then now() end
from public.clubs cross join lateral (
 select jsonb_build_object('schema_version',1,'name',name,'description',coalesce(tagline,''),
 'location','','visitor_info','','logo_url',coalesce(logo_url,''),'accent','#1d4ed8',
 'visibility','unlisted','display','{}'::jsonb,'pages',jsonb_build_array(jsonb_build_object(
 'slug','home','title','Home','in_navigation',true,'blocks','[]'::jsonb))) doc
) d;
update public.clubs set public_site_status='published' where is_active;

create view public.pcs_public_club_directory with(security_invoker=true) as
 select c.slug,s.published->>'name' as name,s.published->>'description' as description,
 s.published->>'location' as location,s.published->>'logo_url' as logo_url,s.published_at
 from public.clubs c join public.pcs_club_sites s on s.club_id=c.id
 where c.is_active and c.public_site_status='published' and s.published->>'visibility'='listed';
revoke all on public.pcs_public_club_directory from public,anon,authenticated;
grant select on public.pcs_public_club_directory to service_role;

create function public.pcs_write_club_site(p_actor_id uuid,p_actor_email text,p_club_id text,
 p_revision integer,p_action text,p_document jsonb) returns jsonb
language plpgsql security invoker set search_path=public as $$
declare site public.pcs_club_sites; club public.clubs;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 select * into club from public.clubs where id=p_club_id for update;
 if not found then raise exception 'Club unavailable' using errcode='P0002'; end if;
 select * into site from public.pcs_club_sites where club_id=p_club_id for update;
 if coalesce(site.revision,0)<>p_revision then raise exception 'Draft changed' using errcode='40001'; end if;
 if p_action='save' then
  if p_document is null or jsonb_typeof(p_document)<>'object' or octet_length(p_document::text)>2500000
   or length(trim(p_document->>'name')) not between 1 and 120
   or p_document->>'visibility' not in ('listed','unlisted') then
   raise exception 'Invalid site document' using errcode='22023'; end if;
  insert into public.pcs_club_sites(club_id,draft) values(p_club_id,p_document)
   on conflict(club_id) do update set draft=excluded.draft,revision=pcs_club_sites.revision+1,updated_at=now();
 elsif p_action='publish' then
  if site.club_id is null or not club.is_active then raise exception 'Active club and draft required' using errcode='22023'; end if;
  update public.pcs_club_sites set published=draft,published_at=now(),revision=revision+1,updated_at=now() where club_id=p_club_id;
  update public.clubs set public_site_status='published' where id=p_club_id;
 elsif p_action='unpublish' then
  update public.pcs_club_sites set published=null,published_at=null,revision=revision+1,updated_at=now() where club_id=p_club_id;
  update public.clubs set public_site_status='draft' where id=p_club_id;
 elsif p_action='discard' then
  if site.published is null then raise exception 'No published version' using errcode='22023'; end if;
  update public.pcs_club_sites set draft=published,revision=revision+1,updated_at=now() where club_id=p_club_id;
 else raise exception 'Invalid action' using errcode='22023'; end if;
 insert into public.pcs_platform_audit(actor_id,club_id,action,details)
 values(p_actor_id,p_club_id,'site_'||p_action,jsonb_build_object('prior_revision',p_revision));
 select * into site from public.pcs_club_sites where club_id=p_club_id;
 return to_jsonb(site)||jsonb_build_object('slug',club.slug,'club_active',club.is_active);
end $$;
revoke all on function public.pcs_write_club_site(uuid,text,text,integer,text,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_write_club_site(uuid,text,text,integer,text,jsonb) to service_role;

create function public.pcs_create_own_club(p_actor_id uuid,p_actor_email text,p_slug text,p_name text,p_document jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare c public.clubs;
begin
 -- Identity comes from a verified bearer token; confirmed email is checked
 -- against Auth, never supplied profile metadata or the requested club email.
 if not exists(select 1 from auth.users where id=p_actor_id and lower(email)=lower(p_actor_email)
  and email_confirmed_at is not null and coalesce(is_anonymous,false)=false) then
  raise exception 'Verified account required' using errcode='42501'; end if;
 if p_slug !~ '^[a-z0-9]+(-[a-z0-9]+)*$' or length(p_slug) not between 3 and 60 or length(trim(p_name)) not between 1 and 120 then
  raise exception 'Invalid club details' using errcode='22023'; end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-signup:'||p_actor_id::text,0));
 if (select count(*) from public.pcs_platform_audit where actor_id=p_actor_id and action='self_create_club' and created_at>now()-interval '1 day')>=5 then
  raise exception 'Creation limit reached' using errcode='54000'; end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-club:'||p_slug,0));
 if exists(select 1 from public.clubs where id=p_slug or slug=p_slug) then
  raise exception 'Club address already exists' using errcode='23505'; end if;
 insert into public.clubs(id,slug,name,is_active,status,plan_status,onboarding_status,public_site_status)
 values(p_slug,p_slug,trim(p_name),true,'active','free','ready','draft') returning * into c;
 insert into public.admin_role_assignments(club_id,user_id,email,role,scopes)
 values(c.id,p_actor_id,lower(p_actor_email),'administrator','[]');
 insert into public.pcs_club_sites(club_id,draft) values(c.id,p_document);
 insert into public.pcs_platform_audit(actor_id,club_id,action) values(p_actor_id,c.id,'self_create_club');
 return jsonb_build_object('club_id',c.id,'club_slug',c.slug,'club_name',c.name,'roles',jsonb_build_array('administrator'));
end $$;
revoke all on function public.pcs_create_own_club(uuid,text,text,text,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_create_own_club(uuid,text,text,text,jsonb) to service_role;

create table public.pcs_interclub_publications (
 season_id uuid primary key references public.pcs_interclub_seasons(id),
 revision integer not null default 1,
 draft jsonb not null default '{"results":[]}',
 published jsonb,
 published_at timestamptz,
 updated_at timestamptz not null default now()
);
alter table public.pcs_interclub_publications enable row level security;
revoke all on public.pcs_interclub_publications from public,anon,authenticated;
grant all on public.pcs_interclub_publications to service_role;
create function public.pcs_write_interclub_publication(p_actor_id uuid,p_actor_email text,p_club_id text,
 p_season_id uuid,p_revision integer,p_action text,p_document jsonb) returns jsonb
language plpgsql security invoker set search_path=public as $$
declare publication public.pcs_interclub_publications; season public.pcs_interclub_seasons;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id for share;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if season.organizer_club_id<>p_club_id then raise exception 'Organizer required' using errcode='42501'; end if;
 select * into publication from public.pcs_interclub_publications where season_id=p_season_id for update;
 if coalesce(publication.revision,0)<>p_revision then raise exception 'Draft changed' using errcode='40001'; end if;
 if p_action='save' then
  insert into public.pcs_interclub_publications(season_id,draft) values(p_season_id,p_document)
   on conflict(season_id) do update set draft=excluded.draft,revision=pcs_interclub_publications.revision+1,updated_at=now();
 elsif p_action='publish' then
  if publication.season_id is null then raise exception 'Save draft first' using errcode='22023'; end if;
  -- Snapshot carries public schedule/results only; no roster/contact records.
  update public.pcs_interclub_publications set published=p_document,published_at=now(),revision=revision+1,updated_at=now() where season_id=p_season_id;
 elsif p_action='unpublish' then
  update public.pcs_interclub_publications set published=null,published_at=null,revision=revision+1,updated_at=now() where season_id=p_season_id;
 else raise exception 'Invalid action' using errcode='22023'; end if;
 insert into public.pcs_platform_audit(actor_id,club_id,action,details)
 values(p_actor_id,p_club_id,'interclub_site_'||p_action,jsonb_build_object('season_id',p_season_id));
 select * into publication from public.pcs_interclub_publications where season_id=p_season_id;
 return to_jsonb(publication);
end $$;
revoke all on function public.pcs_write_interclub_publication(uuid,text,text,uuid,integer,text,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_write_interclub_publication(uuid,text,text,uuid,integer,text,jsonb) to service_role;
commit;
