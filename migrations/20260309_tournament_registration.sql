-- JUPR Streamlit tournament registration module
-- Adds reusable registration config, public registration intake, and partner-board support
-- keyed to the existing tournaments table.

create table if not exists tournament_registration_settings (
  id text primary key,
  tournament_id text not null unique,
  registration_slug text unique,
  locale text not null default 'en',
  registration_status text not null default 'draft',
  registration_open_at timestamptz,
  registration_close_at timestamptz,
  waitlist_enabled boolean not null default true,
  partner_board_enabled boolean not null default true,
  max_events_per_day integer not null default 1,
  rules_markdown text,
  refund_policy_markdown text,
  sponsor_markdown text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists tournament_registration_days (
  id text primary key,
  tournament_id text not null,
  sort_order integer not null,
  label text not null,
  event_date date,
  created_at timestamptz not null default now()
);

create index if not exists idx_tournament_registration_days_tournament
  on tournament_registration_days (tournament_id, sort_order);

create table if not exists tournament_event_options (
  id text primary key,
  tournament_id text not null,
  registration_day_id text not null,
  sort_order integer not null,
  label text not null,
  event_type text not null,
  gender_restriction text not null default 'ANY',
  skill_label text,
  age_label text,
  partner_required boolean not null default false,
  capacity_teams integer,
  public_partner_board boolean not null default true,
  price_usd numeric(10,2),
  created_at timestamptz not null default now()
);

create index if not exists idx_tournament_event_options_tournament
  on tournament_event_options (tournament_id, registration_day_id, sort_order);

create table if not exists tournament_registrations (
  id text primary key,
  tournament_id text not null,
  submitted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  status text not null default 'confirmed',
  payment_status text not null default 'unpaid',
  first_name text,
  last_name text,
  display_name text not null,
  email text not null,
  phone text,
  dupr_id text,
  doubles_skill numeric(4,2),
  singles_skill numeric(4,2),
  age integer,
  age_bracket text,
  gender text,
  notes text,
  wants_partner_board_contact boolean not null default false
);

create unique index if not exists idx_tournament_registrations_email_unique
  on tournament_registrations (tournament_id, lower(email));

create index if not exists idx_tournament_registrations_tournament
  on tournament_registrations (tournament_id, submitted_at desc);

create table if not exists tournament_registration_selections (
  id text primary key,
  tournament_id text not null,
  registration_id text not null,
  registration_day_id text not null,
  event_option_id text not null,
  partner_mode text not null default 'NONE',
  partner_name text,
  partner_email text,
  partner_phone text,
  partner_dupr_id text,
  partner_skill numeric(4,2),
  partner_age integer,
  partner_note text,
  show_on_partner_board boolean not null default false,
  sort_order integer not null default 0,
  created_at timestamptz not null default now()
);

create index if not exists idx_tournament_registration_selections_tournament
  on tournament_registration_selections (tournament_id, event_option_id, sort_order);

create index if not exists idx_tournament_registration_selections_registration
  on tournament_registration_selections (registration_id, registration_day_id);

-- Optional foreign keys after confirming the tournaments.id type in your database.
-- alter table tournament_registration_settings
--   add constraint fk_tournament_registration_settings_tournament
--   foreign key (tournament_id) references tournaments(id) on delete cascade;
--
-- alter table tournament_registration_days
--   add constraint fk_tournament_registration_days_tournament
--   foreign key (tournament_id) references tournaments(id) on delete cascade;
--
-- alter table tournament_event_options
--   add constraint fk_tournament_event_options_tournament
--   foreign key (tournament_id) references tournaments(id) on delete cascade;
--
-- alter table tournament_event_options
--   add constraint fk_tournament_event_options_day
--   foreign key (registration_day_id) references tournament_registration_days(id) on delete cascade;
--
-- alter table tournament_registrations
--   add constraint fk_tournament_registrations_tournament
--   foreign key (tournament_id) references tournaments(id) on delete cascade;
--
-- alter table tournament_registration_selections
--   add constraint fk_tournament_registration_selections_registration
--   foreign key (registration_id) references tournament_registrations(id) on delete cascade;
--
-- alter table tournament_registration_selections
--   add constraint fk_tournament_registration_selections_day
--   foreign key (registration_day_id) references tournament_registration_days(id) on delete cascade;
--
-- alter table tournament_registration_selections
--   add constraint fk_tournament_registration_selections_event
--   foreign key (event_option_id) references tournament_event_options(id) on delete cascade;
