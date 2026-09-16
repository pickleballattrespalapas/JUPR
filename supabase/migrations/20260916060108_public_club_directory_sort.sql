begin;

-- Keep pagination alphabetical regardless of the capitalization in club names.
create or replace view public.pcs_public_club_directory with(security_invoker=true) as
 select c.slug,s.published->>'name' as name,s.published->>'description' as description,
 s.published->>'location' as location,s.published->>'logo_url' as logo_url,s.published_at,
 lower(s.published->>'name') as sort_name
 from public.clubs c join public.pcs_club_sites s on s.club_id=c.id
 where c.is_active and c.public_site_status='published' and s.published->>'visibility'='listed';

revoke all on public.pcs_public_club_directory from public,anon,authenticated;
grant select on public.pcs_public_club_directory to service_role;

commit;
