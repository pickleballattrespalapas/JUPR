-- Private reusable business profiles. Only the authorized FastAPI service reads
-- or writes these; published tournaments continue to hold independent snapshots.
create table public.club_sponsors (
  id uuid primary key default gen_random_uuid(),
  club_id text not null references public.clubs(id) on delete cascade,
  name text not null check (length(btrim(name)) between 1 and 120),
  name_key text generated always as (lower(regexp_replace(btrim(name), '\s+', ' ', 'g'))) stored,
  website text not null default '' check (length(website) <= 2048),
  public_description text not null default '' check (length(public_description) <= 500),
  logo_path text not null default '',
  logo_tournament_id text,
  contact_name text not null default '' check (length(contact_name) <= 120),
  contact_email text not null default '' check (length(contact_email) <= 254),
  contact_phone text not null default '' check (length(contact_phone) <= 80),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (club_id, name_key),
  constraint club_sponsor_logo_scope check (
    (logo_path = '' and logo_tournament_id is null) or
    (logo_tournament_id is not null and logo_tournament_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
     and split_part(logo_path, '/', 1) = club_id
     and split_part(logo_path, '/', 2) = logo_tournament_id
     and split_part(logo_path, '/', 3) ~ '^[0-9a-f]{32}\.webp$'
     and array_length(string_to_array(logo_path, '/'), 1) = 3)
  )
);
alter table public.club_sponsors enable row level security;
revoke all on public.club_sponsors from public, anon, authenticated;
grant select, insert, update, delete on public.club_sponsors to service_role;
create policy club_sponsors_service_only on public.club_sponsors
  for all to service_role using (true) with check (true);

-- Import the club's existing published and saved draft sponsors once. Match
-- names case-insensitively, normalize whitespace, retain the most recent nonempty
-- details, and never import tournament-specific tiers, labels, or private notes.
with sources as (
  select t.club_id, s.tournament_id, s.updated_at, item.position as sponsor_position, item.value as sponsor
  from public.tournament_registration_settings s
  join public.tournaments t on t.id::text = s.tournament_id::text
  cross join lateral jsonb_array_elements(
    case when jsonb_typeof(s.sponsors_json) = 'array' then s.sponsors_json else '[]'::jsonb end
    || case when jsonb_typeof(s.builder_draft_json #> '{basics,sponsors_json}') = 'array'
       then s.builder_draft_json #> '{basics,sponsors_json}' else '[]'::jsonb end
    || case when jsonb_typeof(s.builder_draft_json #> '{settings,sponsors_json}') = 'array'
       then s.builder_draft_json #> '{settings,sponsors_json}' else '[]'::jsonb end
  ) with ordinality as item(value, position)
), clean as (
  select club_id, tournament_id, updated_at, sponsor_position,
    left(regexp_replace(btrim(sponsor->>'name'), '\s+', ' ', 'g'), 120) as name,
    left(coalesce(sponsor->>'website', ''), 2048) as website,
    left(coalesce(sponsor->>'public_description', ''), 500) as description,
    case when tournament_id::text ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
      and split_part(sponsor->>'logo_path', '/', 1) = club_id
      and split_part(sponsor->>'logo_path', '/', 2) = tournament_id::text
      and split_part(sponsor->>'logo_path', '/', 3) ~ '^[0-9a-f]{32}\.webp$'
      and array_length(string_to_array(sponsor->>'logo_path', '/'), 1) = 3
      then sponsor->>'logo_path' else '' end as logo_path
  from sources where jsonb_typeof(sponsor) = 'object' and btrim(coalesce(sponsor->>'name', '')) <> ''
), grouped as (
  select club_id, lower(name) as name_key,
    (array_agg(name order by updated_at desc nulls last, sponsor_position desc, tournament_id))[1] as name,
    (array_agg(website order by updated_at desc nulls last, sponsor_position desc, tournament_id) filter (where website <> ''))[1] as website,
    (array_agg(description order by updated_at desc nulls last, sponsor_position desc, tournament_id) filter (where description <> ''))[1] as description,
    (array_agg(logo_path order by updated_at desc nulls last, sponsor_position desc, tournament_id) filter (where logo_path <> ''))[1] as logo_path
  from clean group by club_id, lower(name)
)
insert into public.club_sponsors (club_id, name, website, public_description, logo_path, logo_tournament_id)
select club_id, name, coalesce(website, ''), coalesce(description, ''), coalesce(logo_path, ''),
  case when logo_path is not null then split_part(logo_path, '/', 2) end
from grouped on conflict (club_id, name_key) do nothing;

comment on table public.club_sponsors is 'Club-private reusable sponsor details and contacts. Event placement and published content remain tournament snapshots.';
