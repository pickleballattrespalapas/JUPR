begin;

-- Existing club websites only. This migration adds no signup, billing,
-- directory, staff invitation, or interclub operation.
alter table public.clubs add column if not exists public_site_status text not null default 'draft'
  check (public_site_status in ('draft', 'published'));
create table if not exists public.pcs_club_sites (
  club_id text primary key references public.clubs(id),
  revision integer not null default 1 check (revision > 0),
  draft jsonb not null check (jsonb_typeof(draft) = 'object'),
  published jsonb check (published is null or jsonb_typeof(published) = 'object'),
  published_at timestamptz,
  updated_at timestamptz not null default now()
);
create table public.club_website_settings_audit (
  id bigint generated always as identity primary key,
  club_id text not null references public.clubs(id),
  actor_id uuid not null,
  action text not null check (action in ('save', 'publish', 'unpublish', 'discard')),
  before_state jsonb not null,
  after_state jsonb not null,
  created_at timestamptz not null default now()
);
create index club_website_settings_audit_club_time_idx
  on public.club_website_settings_audit(club_id, created_at desc);
alter table public.pcs_club_sites enable row level security;
alter table public.club_website_settings_audit enable row level security;
revoke all on public.pcs_club_sites, public.club_website_settings_audit from public, anon, authenticated;
grant select, insert, update on public.pcs_club_sites to service_role;
grant select, insert on public.club_website_settings_audit to service_role;
grant usage, select on sequence public.club_website_settings_audit_id_seq to service_role;

-- Preserve existing public identity, indexing, and all enabled sections.
-- Existing website drafts/publications (including an explicit unpublish) are
-- never replaced. Leaderboard settings remain in their independent table.
with seeded as (
  insert into public.pcs_club_sites(club_id, draft, published, published_at)
  select id, doc, case when is_active then doc end, case when is_active then now() end
  from public.clubs cross join lateral (
    select jsonb_build_object(
      'schema_version', 1, 'name', name, 'description', coalesce(tagline, ''),
      'location', '', 'visitor_info', case when coalesce(support_email, '') <> ''
        then 'Contact: ' || support_email else '' end,
      'logo_url', coalesce(logo_url, ''), 'accent', '#1d4ed8',
      'visibility', 'listed', 'display', '{}'::jsonb, 'page_visibility', '{}'::jsonb,
      'pages', jsonb_build_array(jsonb_build_object(
        'slug', 'home', 'title', 'Home', 'in_navigation', true, 'blocks', '[]'::jsonb))
    ) as doc
  ) as initial
  on conflict (club_id) do nothing
  returning club_id, published
)
update public.clubs c set public_site_status = 'published'
from seeded s where c.id = s.club_id and s.published is not null;

-- The API passes a verified bearer identity. Definer access is needed only to
-- recheck auth.users under the staff lock; clients cannot invoke this RPC.
create function public.save_club_website_settings(
  p_actor_id uuid, p_actor_email text, p_club_id text,
  p_revision integer, p_action text, p_document jsonb
) returns jsonb language plpgsql security definer set search_path = '' as $$
declare
  prior public.pcs_club_sites;
  saved public.pcs_club_sites;
  club public.clubs;
begin
  perform pg_advisory_xact_lock(hashtextextended('pcs-staff:' || p_club_id, 0));
  if not exists (
    select 1 from public.admin_role_assignments a join auth.users u on u.id = p_actor_id
    where a.club_id = p_club_id and a.email = lower(trim(p_actor_email))
      and lower(u.email) = lower(trim(p_actor_email)) and u.email_confirmed_at is not null
      and coalesce(u.is_anonymous, false) = false
      and (a.user_id is null or a.user_id = p_actor_id)
      and a.role in ('super_admin', 'club_owner', 'administrator')
      and a.revoked_at is null and (a.expires_at is null or a.expires_at > now())
  ) then raise exception 'Administrator access required' using errcode = '42501'; end if;
  if p_revision is null or p_revision < 0 or p_action is null
    or p_action not in ('save', 'publish', 'unpublish', 'discard') then
    raise exception 'Invalid website action' using errcode = '22023';
  end if;
  select * into club from public.clubs where id = p_club_id for update;
  if not found then raise exception 'Club unavailable' using errcode = 'P0002'; end if;
  select * into prior from public.pcs_club_sites where club_id = p_club_id for update;
  if coalesce(prior.revision, 0) <> p_revision then
    raise exception 'Draft changed; reload before retrying' using errcode = '40001';
  end if;
  if p_action = 'save' then
    if p_document is null or jsonb_typeof(p_document) <> 'object'
      or octet_length(p_document::text) > 2500000
      or coalesce(length(trim(p_document->>'name')), 0) not between 1 and 120
      or coalesce(p_document->>'visibility', '') not in ('listed', 'unlisted') then
      raise exception 'Invalid website document' using errcode = '22023';
    end if;
    insert into public.pcs_club_sites(club_id, draft) values(p_club_id, p_document)
      on conflict (club_id) do update set draft = excluded.draft,
        revision = pcs_club_sites.revision + 1, updated_at = now();
  elsif p_action = 'publish' then
    if prior.club_id is null or not club.is_active then
      raise exception 'Active club and saved draft required' using errcode = '22023';
    end if;
    update public.pcs_club_sites set published = draft, published_at = now(),
      revision = revision + 1, updated_at = now() where club_id = p_club_id;
    update public.clubs set public_site_status = 'published' where id = p_club_id;
  elsif p_action = 'unpublish' then
    if prior.club_id is null then raise exception 'Website unavailable' using errcode = '22023'; end if;
    update public.pcs_club_sites set published = null, published_at = null,
      revision = revision + 1, updated_at = now() where club_id = p_club_id;
    update public.clubs set public_site_status = 'draft' where id = p_club_id;
  elsif p_action = 'discard' then
    if prior.published is null then raise exception 'No published version' using errcode = '22023'; end if;
    update public.pcs_club_sites set draft = published, revision = revision + 1,
      updated_at = now() where club_id = p_club_id;
  end if;
  select * into saved from public.pcs_club_sites where club_id = p_club_id;
  insert into public.club_website_settings_audit(club_id, actor_id, action, before_state, after_state)
    values(p_club_id, p_actor_id, p_action, coalesce(to_jsonb(prior), '{}'::jsonb), to_jsonb(saved));
  return to_jsonb(saved) || jsonb_build_object('slug', club.slug, 'club_active', club.is_active);
end $$;
revoke all on function public.save_club_website_settings(uuid, text, text, integer, text, jsonb)
  from public, anon, authenticated;
grant execute on function public.save_club_website_settings(uuid, text, text, integer, text, jsonb)
  to service_role;
notify pgrst, 'reload schema';
commit;
