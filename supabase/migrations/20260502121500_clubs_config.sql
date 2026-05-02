create table if not exists public.clubs (
  id text primary key,
  slug text unique not null,
  name text not null,
  tagline text,
  support_email text,
  public_base_url text,
  logo_url text,
  primary_color text,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

insert into public.clubs (
  id,
  slug,
  name,
  tagline,
  support_email,
  public_base_url,
  is_active
)
values (
  'tres_palapas',
  'tres-palapas',
  'Tres Palapas',
  'Official player ratings and events for Tres Palapas',
  'joe@juprleagues.com',
  'https://juprtrespalapas.streamlit.app',
  true
)
on conflict (id) do update
set
  slug = excluded.slug,
  name = excluded.name,
  tagline = excluded.tagline,
  support_email = excluded.support_email,
  public_base_url = excluded.public_base_url,
  is_active = excluded.is_active,
  updated_at = now();
