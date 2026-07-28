-- Fixed-partner team leagues and durable league-award records.
--
-- All tables in this migration are private implementation details. FastAPI is
-- the only supported data-plane client and must use SUPABASE_SERVICE_ROLE_KEY.
-- Public registration, partner confirmation, schedules, standings, results,
-- and public awards are projected through bounded API responses.

create extension if not exists pgcrypto;

do $requirements$
begin
  if to_regclass('public.players') is null
     or to_regclass('public.leagues_metadata') is null
     or to_regclass('public.matches') is null
     or to_regclass('public.admin_activity_log') is null then
    raise exception using
      errcode = '42P01',
      message = 'team leagues require players, leagues_metadata, matches, and admin_activity_log';
  end if;
end
$requirements$;

create table if not exists public.team_league_settings (
  club_id text not null,
  league_name text not null,
  status text not null default 'draft',
  registration_open boolean not null default false,
  allow_substitutes boolean not null default false,
  playoff_format text not null default 'none',
  playoff_team_count integer,
  start_date date,
  weekday smallint not null default 0,
  start_time time not null default '18:00',
  timezone text not null default 'UTC',
  venue text,
  registration_closes_at timestamptz,
  settings_version integer not null default 0,
  schedule_version integer not null default 0,
  standings_version integer not null default 0,
  created_by text,
  updated_by text,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  primary key (club_id, league_name),
  constraint team_league_settings_status_check
    check (status in (
      'draft',
      'registration_open',
      'registration_closed',
      'scheduled',
      'active',
      'playoffs',
      'complete',
      'archived'
    )),
  constraint team_league_settings_playoff_check
    check (playoff_format in (
      'none',
      'top_2_final',
      'top_4_single_elimination',
      'all_team_single_elimination'
    )),
  constraint team_league_settings_playoff_count_check
    check (
      playoff_team_count is null
      or playoff_team_count between 2 and 128
    ),
  constraint team_league_settings_weekday_check check (weekday between 0 and 6),
  constraint team_league_settings_settings_version_check check (settings_version >= 0),
  constraint team_league_settings_schedule_version_check check (schedule_version >= 0),
  constraint team_league_settings_standings_version_check check (standings_version >= 0)
);

create table if not exists public.team_league_teams (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_name text not null,
  team_name text not null,
  status text not null default 'pending_partner',
  captain_player_id bigint not null,
  partner_player_id bigint not null,
  captain_contact_email text not null,
  partner_contact_email text not null,
  created_operation_id uuid,
  partner_invite_token_hash text,
  partner_invite_expires_at timestamptz,
  invitation_delivery_status text not null default 'pending',
  invitation_provider_message_id text,
  invitation_delivered_at timestamptz,
  invitation_delivery_attempts integer not null default 0,
  invitation_claim_token uuid,
  invitation_claimed_at timestamptz,
  invitation_delivery_error text,
  partner_confirmed_at timestamptz,
  withdrawn_at timestamptz,
  playoff_seed integer,
  playoff_seed_standings_version integer,
  playoff_seeded_at timestamptz,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  foreign key (club_id, league_name)
    references public.team_league_settings (club_id, league_name)
    on delete cascade,
  constraint team_league_teams_status_check
    check (status in ('pending_partner', 'confirmed', 'declined', 'withdrawn')),
  constraint team_league_teams_distinct_players_check
    check (captain_player_id <> partner_player_id),
  constraint team_league_teams_playoff_seed_check
    check (playoff_seed is null or playoff_seed between 1 and 128),
  constraint team_league_teams_email_check
    check (
      pg_catalog.char_length(captain_contact_email) between 3 and 320
      and captain_contact_email like '%@%'
      and pg_catalog.char_length(partner_contact_email) between 3 and 320
      and partner_contact_email like '%@%'
    ),
  constraint team_league_teams_invite_hash_check
    check (
      partner_invite_token_hash is null
      or partner_invite_token_hash ~ '^[0-9a-f]{64}$'
    ),
  constraint team_league_teams_delivery_status_check
    check (
      invitation_delivery_status in (
        'pending',
        'claimed',
        'not_required',
        'dry_run',
        'staging_redirect',
        'sent',
        'failed'
      )
    ),
  constraint team_league_teams_delivery_attempts_check
    check (invitation_delivery_attempts >= 0),
  unique (club_id, league_name, id)
);

create unique index if not exists team_league_active_team_name_idx
  on public.team_league_teams (
    club_id,
    league_name,
    pg_catalog.lower(pg_catalog.btrim(team_name))
  )
  where status in ('pending_partner', 'confirmed');

create unique index if not exists team_league_partner_invite_hash_idx
  on public.team_league_teams (partner_invite_token_hash)
  where partner_invite_token_hash is not null;

create index if not exists team_league_teams_club_league_status_idx
  on public.team_league_teams (club_id, league_name, status, created_at);

create table if not exists public.team_league_solo_waitlist (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_name text not null,
  player_id bigint not null,
  contact_email text not null,
  note text,
  status text not null default 'waiting',
  created_operation_id uuid,
  matched_team_id uuid,
  withdrawn_at timestamptz,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  foreign key (club_id, league_name)
    references public.team_league_settings (club_id, league_name)
    on delete cascade,
  foreign key (club_id, league_name, matched_team_id)
    references public.team_league_teams (club_id, league_name, id),
  constraint team_league_waitlist_status_check
    check (status in ('waiting', 'matched', 'withdrawn')),
  constraint team_league_waitlist_email_check
    check (
      pg_catalog.char_length(contact_email) between 3 and 320
      and contact_email like '%@%'
    )
);

create unique index if not exists team_league_waitlist_active_player_idx
  on public.team_league_solo_waitlist (club_id, league_name, player_id)
  where status = 'waiting';

create index if not exists team_league_waitlist_team_fk_idx
  on public.team_league_solo_waitlist (club_id, league_name, matched_team_id)
  where matched_team_id is not null;

create table if not exists public.team_league_fixtures (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_name text not null,
  phase text not null default 'regular',
  round_number integer not null,
  week_number integer,
  bracket_slot integer,
  scheduled_at timestamptz,
  team_a_id uuid,
  team_b_id uuid,
  team_a_source text,
  team_b_source text,
  status text not null default 'scheduled',
  team_a_score integer,
  team_b_score integer,
  winner_team_id uuid,
  resolution text,
  official_match_id bigint,
  team_a_player_1_id bigint,
  team_a_player_2_id bigint,
  team_b_player_1_id bigint,
  team_b_player_2_id bigint,
  substitutions_json jsonb not null default '[]'::jsonb,
  score_note text,
  schedule_operation_id uuid,
  score_operation_id uuid,
  scored_by text,
  scored_at timestamptz,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  foreign key (club_id, league_name)
    references public.team_league_settings (club_id, league_name)
    on delete cascade,
  foreign key (club_id, league_name, team_a_id)
    references public.team_league_teams (club_id, league_name, id),
  foreign key (club_id, league_name, team_b_id)
    references public.team_league_teams (club_id, league_name, id),
  constraint team_league_fixtures_phase_check
    check (phase in ('regular', 'playoff')),
  constraint team_league_fixtures_status_check
    check (status in ('scheduled', 'complete', 'forfeit', 'bye', 'cancelled')),
  constraint team_league_fixtures_resolution_check
    check (resolution is null or resolution in ('played', 'forfeit', 'bye', 'cancelled')),
  constraint team_league_fixtures_round_check check (round_number >= 1),
  constraint team_league_fixtures_week_check check (week_number is null or week_number >= 1),
  constraint team_league_fixtures_distinct_teams_check
    check (team_a_id is null or team_b_id is null or team_a_id <> team_b_id),
  constraint team_league_fixtures_score_pair_check
    check (
      (team_a_score is null and team_b_score is null)
      or (team_a_score is not null and team_b_score is not null
          and team_a_score >= 0 and team_b_score >= 0
          and team_a_score <> team_b_score)
    ),
  constraint team_league_fixtures_substitutions_check
    check (pg_catalog.jsonb_typeof(substitutions_json) = 'array'),
  unique (club_id, league_name, phase, round_number, bracket_slot)
);

create index if not exists team_league_fixtures_public_idx
  on public.team_league_fixtures (
    club_id,
    league_name,
    phase,
    week_number,
    round_number,
    scheduled_at
  );

create index if not exists team_league_fixtures_match_idx
  on public.team_league_fixtures (club_id, official_match_id)
  where official_match_id is not null;

create index if not exists team_league_fixtures_team_a_fk_idx
  on public.team_league_fixtures (club_id, league_name, team_a_id)
  where team_a_id is not null;

create index if not exists team_league_fixtures_team_b_fk_idx
  on public.team_league_fixtures (club_id, league_name, team_b_id)
  where team_b_id is not null;

create unique index if not exists team_league_team_operation_idx
  on public.team_league_teams (created_operation_id)
  where created_operation_id is not null;

create unique index if not exists team_league_waitlist_operation_idx
  on public.team_league_solo_waitlist (created_operation_id)
  where created_operation_id is not null;

create index if not exists team_league_fixture_schedule_operation_idx
  on public.team_league_fixtures (schedule_operation_id)
  where schedule_operation_id is not null;

create unique index if not exists team_league_fixture_score_operation_idx
  on public.team_league_fixtures (score_operation_id)
  where score_operation_id is not null;

create table if not exists public.team_league_operations (
  id uuid primary key,
  club_id text not null,
  league_name text not null,
  idempotency_key text not null,
  request_fingerprint text not null,
  operation_type text not null,
  status text not null default 'started',
  request_json jsonb not null,
  result_json jsonb,
  actor_email text,
  actor_role text,
  source text not null,
  recovery_note text,
  started_at timestamptz not null default pg_catalog.clock_timestamp(),
  completed_at timestamptz,
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  constraint team_league_operations_key_check
    check (
      pg_catalog.char_length(idempotency_key) between 8 and 160
      and idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]+$'
    ),
  constraint team_league_operations_fingerprint_check
    check (request_fingerprint ~ '^[0-9a-f]{64}$'),
  constraint team_league_operations_status_check
    check (status in ('started', 'complete', 'recovery_required', 'compensated')),
  constraint team_league_operations_request_check
    check (pg_catalog.jsonb_typeof(request_json) = 'object'),
  constraint team_league_operations_result_check
    check (result_json is null or pg_catalog.jsonb_typeof(result_json) = 'object'),
  unique (club_id, idempotency_key)
);

alter table public.team_league_teams
  add constraint team_league_team_operation_fk
  foreign key (created_operation_id)
  references public.team_league_operations (id);

alter table public.team_league_solo_waitlist
  add constraint team_league_waitlist_operation_fk
  foreign key (created_operation_id)
  references public.team_league_operations (id);

alter table public.team_league_fixtures
  add constraint team_league_fixture_schedule_operation_fk
  foreign key (schedule_operation_id)
  references public.team_league_operations (id);

alter table public.team_league_fixtures
  add constraint team_league_fixture_score_operation_fk
  foreign key (score_operation_id)
  references public.team_league_operations (id);

create index if not exists team_league_operations_recovery_idx
  on public.team_league_operations (club_id, league_name, status, updated_at desc);

create table if not exists public.league_award_result_sets (
  club_id text not null,
  league_name text not null,
  workflow_revision integer not null,
  preview_fingerprint text not null,
  result_fingerprint text not null,
  record_count integer not null,
  source_snapshot jsonb not null default '{}'::jsonb,
  finalized_at timestamptz,
  created_by text,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  primary key (club_id, league_name, workflow_revision),
  constraint league_award_result_set_revision_check
    check (workflow_revision >= 0),
  constraint league_award_result_set_preview_fingerprint_check
    check (preview_fingerprint ~ '^[0-9a-f]{64}$'),
  constraint league_award_result_set_result_fingerprint_check
    check (result_fingerprint ~ '^[0-9a-f]{64}$'),
  constraint league_award_result_set_count_check check (record_count >= 0),
  constraint league_award_result_set_snapshot_check
    check (pg_catalog.jsonb_typeof(source_snapshot) = 'object')
);

create table if not exists public.league_award_result_records (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_name text not null,
  workflow_revision integer not null,
  preview_fingerprint text not null,
  result_fingerprint text not null,
  award_key text not null,
  category_key text not null,
  category_label text not null,
  recipient_type text not null default 'player',
  player_id bigint,
  team_id uuid,
  recipient_name text not null,
  placement integer not null default 1,
  is_co_winner boolean not null default false,
  metric_value numeric,
  computed_metric_value numeric,
  computed_player_id bigint,
  computed_team_id uuid,
  computed_recipient_name text,
  metric_display text not null,
  manual_label text,
  is_override boolean not null default false,
  override_reason text,
  public_visible boolean not null default true,
  source_snapshot jsonb not null default '{}'::jsonb,
  finalized_at timestamptz,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  foreign key (club_id, league_name, workflow_revision)
    references public.league_award_result_sets (
      club_id,
      league_name,
      workflow_revision
    )
    on delete cascade,
  foreign key (club_id, league_name, team_id)
    references public.team_league_teams (club_id, league_name, id),
  constraint league_award_result_revision_check check (workflow_revision >= 0),
  constraint league_award_result_fingerprint_check
    check (preview_fingerprint ~ '^[0-9a-f]{64}$'),
  constraint league_award_result_result_fingerprint_check
    check (result_fingerprint ~ '^[0-9a-f]{64}$'),
  constraint league_award_result_key_check
    check (pg_catalog.char_length(pg_catalog.btrim(award_key)) between 3 and 240),
  constraint league_award_result_recipient_check
    check (
      (recipient_type = 'player' and player_id is not null and team_id is null)
      or (recipient_type = 'team' and team_id is not null and player_id is null)
    ),
  constraint league_award_result_recipient_name_check
    check (
      pg_catalog.char_length(pg_catalog.btrim(recipient_name))
        between 1 and 160
    ),
  constraint league_award_result_placement_check check (placement between 1 and 3),
  constraint league_award_result_override_check
    check (
      not is_override
      or (
        pg_catalog.char_length(pg_catalog.btrim(coalesce(override_reason, ''))) >= 8
        and (
          metric_value is not null
          or pg_catalog.char_length(pg_catalog.btrim(coalesce(manual_label, ''))) > 0
        )
      )
    ),
  constraint league_award_result_snapshot_check
    check (pg_catalog.jsonb_typeof(source_snapshot) = 'object')
);

create index if not exists league_award_result_public_idx
  on public.league_award_result_records (
    club_id,
    league_name,
    public_visible,
    finalized_at desc,
    category_key,
    placement
  );

create index if not exists league_award_result_sets_public_idx
  on public.league_award_result_sets (
    club_id,
    league_name,
    finalized_at desc,
    workflow_revision desc
  );

create unique index if not exists league_award_result_identity_idx
  on public.league_award_result_records (
    club_id,
    league_name,
    workflow_revision,
    award_key
  );

create index if not exists league_award_result_team_fk_idx
  on public.league_award_result_records (club_id, league_name, team_id)
  where team_id is not null;

-- Prevent one player from occupying two live teams in the same league. This
-- includes pending invitations so a player cannot be overbooked.
create or replace function public.enforce_team_league_unique_active_players_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if new.status not in ('pending_partner', 'confirmed') then
    return new;
  end if;
  if exists (
    select 1
      from public.team_league_teams as existing
     where existing.club_id = new.club_id
       and existing.league_name = new.league_name
       and existing.id <> new.id
       and existing.status in ('pending_partner', 'confirmed')
       and (
         existing.captain_player_id in (new.captain_player_id, new.partner_player_id)
         or existing.partner_player_id in (new.captain_player_id, new.partner_player_id)
       )
  ) then
    raise exception using
      errcode = '23505',
      message = 'TEAM_LEAGUE_PLAYER_ALREADY_REGISTERED';
  end if;
  return new;
end
$function$;

drop trigger if exists trg_team_league_unique_active_players
  on public.team_league_teams;
create trigger trg_team_league_unique_active_players
before insert or update of status, captain_player_id, partner_player_id
on public.team_league_teams
for each row execute function public.enforce_team_league_unique_active_players_v1();

-- Completing a fixture changes standings. The monotonic version gives API
-- clients an inexpensive stale-state signal and supports safe recovery.
create or replace function public.bump_team_league_standings_version_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if new.phase = 'regular'
     and (
       old.status is distinct from new.status
       or old.team_a_score is distinct from new.team_a_score
       or old.team_b_score is distinct from new.team_b_score
       or old.winner_team_id is distinct from new.winner_team_id
     ) then
    update public.team_league_settings
       set standings_version = standings_version + 1,
           updated_at = pg_catalog.clock_timestamp()
     where club_id = new.club_id
       and league_name = new.league_name;
  end if;
  return new;
end
$function$;

drop trigger if exists trg_team_league_fixture_standings_version
  on public.team_league_fixtures;
create trigger trg_team_league_fixture_standings_version
after update of status, team_a_score, team_b_score, winner_team_id
on public.team_league_fixtures
for each row execute function public.bump_team_league_standings_version_v1();

-- Public signup is one transaction: idempotency receipt, registration row, and
-- audit record either all commit or all roll back. The raw partner token never
-- enters Postgres; FastAPI supplies only its SHA-256 hash.
create or replace function public.team_league_register_public_v1(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_signup_type text,
  p_player_id bigint,
  p_partner_player_id bigint,
  p_team_name text,
  p_contact_email text,
  p_partner_email text,
  p_note text,
  p_invite_token_hash text,
  p_invite_expires_at timestamptz,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_league_name text := nullif(pg_catalog.left(pg_catalog.btrim(p_league_name), 120), '');
  v_key text := nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_fingerprint text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_signup_type text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_signup_type, '')));
  v_team_name text := nullif(pg_catalog.left(pg_catalog.btrim(p_team_name), 120), '');
  v_email text := pg_catalog.lower(pg_catalog.left(pg_catalog.btrim(coalesce(p_contact_email, '')), 320));
  v_partner_email text := pg_catalog.lower(pg_catalog.left(pg_catalog.btrim(coalesce(p_partner_email, '')), 320));
  v_note text := nullif(pg_catalog.left(pg_catalog.btrim(p_note), 500), '');
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'public_team_league_registration'
  );
  v_operation public.team_league_operations%rowtype;
  v_settings public.team_league_settings%rowtype;
  v_team_id uuid;
  v_waitlist_id uuid;
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_league_name is null
     or v_key is null
     or v_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$'
     or v_fingerprint !~ '^[0-9a-f]{64}$'
     or v_signup_type not in ('team', 'solo')
     or p_player_id is null
     or pg_catalog.char_length(v_email) not between 3 and 320
     or v_email not like '%@%' then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SIGNUP_INVALID';
  end if;
  if v_signup_type = 'team' and (
    p_partner_player_id is null
    or p_partner_player_id = p_player_id
    or v_team_name is null
    or p_invite_token_hash !~ '^[0-9a-f]{64}$'
    or p_invite_expires_at is null
    or pg_catalog.char_length(v_partner_email) not between 3 and 320
    or v_partner_email not like '%@%'
  ) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_TEAM_SIGNUP_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-public:' || v_club_id || ':' || v_key,
      0
    )
  );

  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.club_id = v_club_id
     and operation.idempotency_key = v_key
   for update;
  if found then
    if v_operation.request_fingerprint <> v_fingerprint then
      raise exception using
        errcode = '23505',
        message = 'TEAM_LEAGUE_IDEMPOTENCY_CONFLICT';
    end if;
    if v_operation.status = 'complete' and v_operation.result_json is not null then
      return v_operation.result_json || '{"idempotent": true}'::jsonb;
    end if;
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_OPERATION_RECOVERY_REQUIRED';
  end if;

  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name
   for update;
  if not found
     or not v_settings.registration_open
     or (
       v_settings.registration_closes_at is not null
       and v_settings.registration_closes_at <= pg_catalog.clock_timestamp()
     ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_REGISTRATION_CLOSED';
  end if;

  if not exists (
    select 1
      from public.players as player
     where player.club_id = v_club_id
       and player.id = p_player_id
       and coalesce(player.active, true)
       and player.inactive_at is null
  ) or (
    v_signup_type = 'team'
    and not exists (
      select 1
        from public.players as player
       where player.club_id = v_club_id
         and player.id = p_partner_player_id
         and coalesce(player.active, true)
         and player.inactive_at is null
    )
  ) then
    raise exception using
      errcode = '23503',
      message = 'TEAM_LEAGUE_PLAYER_UNAVAILABLE';
  end if;

  if exists (
    select 1
      from public.team_league_teams as team
     where team.club_id = v_club_id
       and team.league_name = v_league_name
       and team.status in ('pending_partner', 'confirmed')
       and (
         team.captain_player_id = p_player_id
         or team.partner_player_id = p_player_id
         or (
           p_partner_player_id is not null
           and (
             team.captain_player_id = p_partner_player_id
             or team.partner_player_id = p_partner_player_id
           )
         )
       )
  ) or exists (
    select 1
      from public.team_league_solo_waitlist as waitlist
     where waitlist.club_id = v_club_id
       and waitlist.league_name = v_league_name
       and waitlist.status = 'waiting'
       and waitlist.player_id in (p_player_id, p_partner_player_id)
  ) then
    raise exception using
      errcode = '23505',
      message = 'TEAM_LEAGUE_PLAYER_ALREADY_REGISTERED';
  end if;

  insert into public.team_league_operations (
    id,
    club_id,
    league_name,
    idempotency_key,
    request_fingerprint,
    operation_type,
    status,
    request_json,
    actor_email,
    actor_role,
    source
  ) values (
    p_operation_id,
    v_club_id,
    v_league_name,
    v_key,
    v_fingerprint,
    'public_' || v_signup_type || '_signup',
    'started',
    pg_catalog.jsonb_build_object(
      'signup_type', v_signup_type,
      'player_id', p_player_id,
      'partner_player_id', p_partner_player_id,
      'team_name', v_team_name,
      'contact_email', v_email,
      'partner_email', nullif(v_partner_email, ''),
      'note', v_note
    ),
    v_email,
    'public',
    v_source
  );

  if v_signup_type = 'team' then
    insert into public.team_league_teams (
      club_id,
      league_name,
      team_name,
      status,
      captain_player_id,
      partner_player_id,
      captain_contact_email,
      partner_contact_email,
      created_operation_id,
      partner_invite_token_hash,
      partner_invite_expires_at
    ) values (
      v_club_id,
      v_league_name,
      v_team_name,
      'pending_partner',
      p_player_id,
      p_partner_player_id,
      v_email,
      v_partner_email,
      p_operation_id,
      p_invite_token_hash,
      p_invite_expires_at
    )
    returning id into v_team_id;
    v_result := pg_catalog.jsonb_build_object(
      'ok', true,
      'committed', true,
      'operation_id', p_operation_id,
      'signup_type', 'team',
      'team_id', v_team_id,
      'status', 'pending_partner',
      'message', 'Team saved. Your partner must confirm before the team is entered.',
      'idempotent', false
    );
  else
    insert into public.team_league_solo_waitlist (
      club_id,
      league_name,
      player_id,
      contact_email,
      note,
      status,
      created_operation_id
    ) values (
      v_club_id,
      v_league_name,
      p_player_id,
      v_email,
      v_note,
      'waiting',
      p_operation_id
    )
    returning id into v_waitlist_id;
    v_result := pg_catalog.jsonb_build_object(
      'ok', true,
      'committed', true,
      'operation_id', p_operation_id,
      'signup_type', 'solo',
      'waitlist_id', v_waitlist_id,
      'status', 'waiting',
      'message', 'You are on the partner waitlist.',
      'idempotent', false
    );
  end if;

  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_club_id,
    v_email,
    'public',
    'team_league_' || v_signup_type || '_signup',
    'team_league',
    v_league_name,
    '{}'::jsonb,
    v_result - 'idempotent',
    null,
    v_source,
    false
  );

  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

-- Partner response is likewise atomic. Possession of the one-time token is the
-- confirmation credential; the raw token is never returned by this function.
create or replace function public.team_league_confirm_partner_public_v1(
  p_operation_id uuid,
  p_team_id uuid,
  p_token_hash text,
  p_accept boolean,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_key text := nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_fingerprint text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'public_team_league_partner_confirmation'
  );
  v_team public.team_league_teams%rowtype;
  v_operation public.team_league_operations%rowtype;
  v_next_status text;
  v_result jsonb;
begin
  if p_operation_id is null
     or p_team_id is null
     or p_token_hash !~ '^[0-9a-f]{64}$'
     or v_key is null
     or v_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$'
     or v_fingerprint !~ '^[0-9a-f]{64}$' then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_PARTNER_CONFIRMATION_INVALID';
  end if;

  select team.*
    into v_team
    from public.team_league_teams as team
   where team.id = p_team_id
   for update;
  if not found
     or v_team.partner_invite_token_hash is distinct from p_token_hash then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_PARTNER_INVITATION_NOT_FOUND';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-public:' || v_team.club_id || ':' || v_key,
      0
    )
  );
  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.club_id = v_team.club_id
     and operation.idempotency_key = v_key
   for update;
  if found then
    if v_operation.request_fingerprint <> v_fingerprint then
      raise exception using
        errcode = '23505',
        message = 'TEAM_LEAGUE_IDEMPOTENCY_CONFLICT';
    end if;
    if v_operation.status = 'complete' and v_operation.result_json is not null then
      return v_operation.result_json || '{"idempotent": true}'::jsonb;
    end if;
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_OPERATION_RECOVERY_REQUIRED';
  end if;

  if v_team.status <> 'pending_partner'
     or v_team.partner_invite_expires_at is null
     or v_team.partner_invite_expires_at < pg_catalog.clock_timestamp() then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_PARTNER_INVITATION_EXPIRED_OR_CLOSED';
  end if;

  insert into public.team_league_operations (
    id,
    club_id,
    league_name,
    idempotency_key,
    request_fingerprint,
    operation_type,
    status,
    request_json,
    actor_email,
    actor_role,
    source
  ) values (
    p_operation_id,
    v_team.club_id,
    v_team.league_name,
    v_key,
    v_fingerprint,
    'public_partner_confirmation',
    'started',
    pg_catalog.jsonb_build_object(
      'team_id', p_team_id,
      'accept', p_accept,
      'token_hash', p_token_hash
    ),
    'partner-confirmation',
    'public',
    v_source
  );

  v_next_status := case when p_accept then 'confirmed' else 'declined' end;
  update public.team_league_teams
     set status = v_next_status,
         partner_confirmed_at = case
           when p_accept then pg_catalog.clock_timestamp()
           else null
         end,
         updated_at = pg_catalog.clock_timestamp()
   where id = p_team_id;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'team_id', p_team_id,
    'status', v_next_status,
    'message', case
      when p_accept then 'Team confirmed. You are entered in the league.'
      else 'Invitation declined.'
    end,
    'idempotent', false
  );

  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_team.club_id,
    'partner-confirmation',
    'public',
    'team_league_partner_confirmation',
    'team_league',
    v_team.league_name,
    pg_catalog.jsonb_build_object('status', v_team.status, 'team_id', p_team_id),
    pg_catalog.jsonb_build_object('status', v_next_status, 'team_id', p_team_id),
    null,
    v_source,
    false
  );

  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

-- Invitation delivery uses a durable short lease. FastAPI can safely resume a
-- pending or failed dry-run/redirect/send attempt without exposing the token.
create or replace function public.team_league_claim_partner_invitation_v1(
  p_team_id uuid,
  p_token_hash text,
  p_claim_token uuid
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_team public.team_league_teams%rowtype;
begin
  if p_team_id is null
     or p_claim_token is null
     or p_token_hash !~ '^[0-9a-f]{64}$' then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_INVITATION_CLAIM_INVALID';
  end if;
  select team.*
    into v_team
    from public.team_league_teams as team
   where team.id = p_team_id
     and team.partner_invite_token_hash = p_token_hash
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_PARTNER_INVITATION_NOT_FOUND';
  end if;
  if v_team.status <> 'pending_partner'
     or v_team.partner_invite_expires_at is null
     or v_team.partner_invite_expires_at <= pg_catalog.clock_timestamp() then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_PARTNER_INVITATION_EXPIRED_OR_CLOSED';
  end if;
  if v_team.invitation_delivery_status in (
    'dry_run',
    'staging_redirect',
    'sent'
  ) then
    return pg_catalog.jsonb_build_object(
      'ok', true,
      'send_required', false,
      'status', v_team.invitation_delivery_status,
      'attempt_count', v_team.invitation_delivery_attempts
    );
  end if;
  if v_team.invitation_delivery_status = 'claimed'
     and v_team.invitation_claimed_at >
       pg_catalog.clock_timestamp() - interval '5 minutes' then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_INVITATION_DELIVERY_IN_PROGRESS';
  end if;
  update public.team_league_teams
     set invitation_delivery_status = 'claimed',
         invitation_claim_token = p_claim_token,
         invitation_claimed_at = pg_catalog.clock_timestamp(),
         invitation_delivery_attempts = invitation_delivery_attempts + 1,
         invitation_delivery_error = null,
         updated_at = pg_catalog.clock_timestamp()
   where id = p_team_id
  returning * into v_team;
  return pg_catalog.jsonb_build_object(
    'ok', true,
    'send_required', true,
    'status', 'claimed',
    'team_id', v_team.id,
    'club_id', v_team.club_id,
    'league_name', v_team.league_name,
    'team_name', v_team.team_name,
    'captain_player_id', v_team.captain_player_id,
    'partner_player_id', v_team.partner_player_id,
    'partner_email', v_team.partner_contact_email,
    'created_operation_id', v_team.created_operation_id,
    'attempt_count', v_team.invitation_delivery_attempts
  );
end
$function$;

create or replace function public.team_league_finish_partner_invitation_v1(
  p_team_id uuid,
  p_claim_token uuid,
  p_delivery_status text,
  p_provider_message_id text,
  p_delivery_error text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_team public.team_league_teams%rowtype;
  v_status text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_delivery_status, '')));
  v_provider_id text := nullif(
    pg_catalog.left(pg_catalog.btrim(p_provider_message_id), 240),
    ''
  );
  v_error text := nullif(
    pg_catalog.left(pg_catalog.btrim(p_delivery_error), 500),
    ''
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'public_team_league_partner_invitation'
  );
begin
  if p_team_id is null
     or p_claim_token is null
     or v_status not in ('dry_run', 'staging_redirect', 'sent', 'failed') then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_INVITATION_FINISH_INVALID';
  end if;
  select team.*
    into v_team
    from public.team_league_teams as team
   where team.id = p_team_id
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_PARTNER_INVITATION_NOT_FOUND';
  end if;
  if v_team.invitation_delivery_status in (
    'dry_run',
    'staging_redirect',
    'sent'
  ) then
    return pg_catalog.jsonb_build_object(
      'ok', true,
      'status', v_team.invitation_delivery_status,
      'attempt_count', v_team.invitation_delivery_attempts,
      'idempotent', true
    );
  end if;
  if v_team.invitation_delivery_status <> 'claimed'
     or v_team.invitation_claim_token is distinct from p_claim_token then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_INVITATION_CLAIM_CONFLICT';
  end if;
  update public.team_league_teams
     set invitation_delivery_status = v_status,
         invitation_provider_message_id = v_provider_id,
         invitation_delivered_at = case
           when v_status in ('dry_run', 'staging_redirect', 'sent')
             then pg_catalog.clock_timestamp()
           else null
         end,
         invitation_delivery_error = case
           when v_status = 'failed' then coalesce(v_error, 'delivery failed')
           else null
         end,
         invitation_claim_token = null,
         invitation_claimed_at = null,
         updated_at = pg_catalog.clock_timestamp()
   where id = p_team_id;

  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_team.club_id,
    'team-league-invitation',
    'system',
    'team_league_partner_invitation_' || v_status,
    'team_league_team',
    p_team_id::text,
    pg_catalog.jsonb_build_object(
      'status', v_team.invitation_delivery_status,
      'attempt_count', v_team.invitation_delivery_attempts
    ),
    pg_catalog.jsonb_build_object(
      'status', v_status,
      'attempt_count', v_team.invitation_delivery_attempts
    ),
    v_error,
    v_source,
    v_status = 'failed'
  );
  return pg_catalog.jsonb_build_object(
    'ok', true,
    'status', v_status,
    'attempt_count', v_team.invitation_delivery_attempts,
    'idempotent', false
  );
end
$function$;

-- Staff can pair two waiting players or withdraw selected waitlist entries.
-- Receipt, domain rows, and audit evidence are one transaction.
create or replace function public.team_league_admin_waitlist_action_v1(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_action text,
  p_waitlist_ids uuid[],
  p_team_name text,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_league_name text := nullif(pg_catalog.left(pg_catalog.btrim(p_league_name), 120), '');
  v_key text := nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_fingerprint text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_action text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_action, '')));
  v_team_name text := nullif(pg_catalog.left(pg_catalog.btrim(p_team_name), 120), '');
  v_actor_email text := nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), 320), '');
  v_actor_role text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_role)), 80), ''),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_waitlist'
  );
  v_operation public.team_league_operations%rowtype;
  v_first public.team_league_solo_waitlist%rowtype;
  v_second public.team_league_solo_waitlist%rowtype;
  v_team_id uuid;
  v_affected integer := 0;
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_league_name is null
     or v_key is null
     or v_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$'
     or v_fingerprint !~ '^[0-9a-f]{64}$'
     or v_action not in ('pair', 'withdraw')
     or p_waitlist_ids is null
     or pg_catalog.array_length(p_waitlist_ids, 1) is null
     or pg_catalog.array_length(p_waitlist_ids, 1) > 100
     or (
       v_action = 'pair'
       and (
         pg_catalog.array_length(p_waitlist_ids, 1) <> 2
         or p_waitlist_ids[1] = p_waitlist_ids[2]
         or v_team_name is null
       )
     ) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_WAITLIST_ACTION_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-admin:' || v_club_id || ':' || v_key,
      0
    )
  );
  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.club_id = v_club_id
     and operation.idempotency_key = v_key
   for update;
  if found then
    if v_operation.request_fingerprint <> v_fingerprint then
      raise exception using
        errcode = '23505',
        message = 'TEAM_LEAGUE_IDEMPOTENCY_CONFLICT';
    end if;
    if v_operation.status = 'complete' and v_operation.result_json is not null then
      return v_operation.result_json || '{"idempotent": true}'::jsonb;
    end if;
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_OPERATION_RECOVERY_REQUIRED';
  end if;

  perform 1
    from public.team_league_solo_waitlist as waitlist
   where waitlist.club_id = v_club_id
     and waitlist.league_name = v_league_name
     and waitlist.id = any(p_waitlist_ids)
     and waitlist.status = 'waiting'
   for update;
  get diagnostics v_affected = row_count;
  if v_affected <> pg_catalog.array_length(p_waitlist_ids, 1) then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_WAITLIST_CHANGED';
  end if;

  insert into public.team_league_operations (
    id,
    club_id,
    league_name,
    idempotency_key,
    request_fingerprint,
    operation_type,
    status,
    request_json,
    actor_email,
    actor_role,
    source
  ) values (
    p_operation_id,
    v_club_id,
    v_league_name,
    v_key,
    v_fingerprint,
    'admin_waitlist_' || v_action,
    'started',
    pg_catalog.jsonb_build_object(
      'action', v_action,
      'waitlist_ids', pg_catalog.to_jsonb(p_waitlist_ids),
      'team_name', v_team_name
    ),
    v_actor_email,
    v_actor_role,
    v_source
  );

  if v_action = 'pair' then
    select waitlist.*
      into v_first
      from public.team_league_solo_waitlist as waitlist
     where waitlist.id = p_waitlist_ids[1];
    select waitlist.*
      into v_second
      from public.team_league_solo_waitlist as waitlist
     where waitlist.id = p_waitlist_ids[2];
    if v_first.player_id = v_second.player_id then
      raise exception using
        errcode = '23505',
        message = 'TEAM_LEAGUE_WAITLIST_DUPLICATE_PLAYER';
    end if;
    insert into public.team_league_teams (
      club_id,
      league_name,
      team_name,
      status,
      captain_player_id,
      partner_player_id,
      captain_contact_email,
      partner_contact_email,
      created_operation_id,
      partner_confirmed_at,
      invitation_delivery_status
    ) values (
      v_club_id,
      v_league_name,
      v_team_name,
      'confirmed',
      v_first.player_id,
      v_second.player_id,
      v_first.contact_email,
      v_second.contact_email,
      p_operation_id,
      pg_catalog.clock_timestamp(),
      'not_required'
    )
    returning id into v_team_id;
    update public.team_league_solo_waitlist
       set status = 'matched',
           matched_team_id = v_team_id,
           updated_at = pg_catalog.clock_timestamp()
     where id = any(p_waitlist_ids);
    get diagnostics v_affected = row_count;
  else
    update public.team_league_solo_waitlist
       set status = 'withdrawn',
           withdrawn_at = pg_catalog.clock_timestamp(),
           updated_at = pg_catalog.clock_timestamp()
     where id = any(p_waitlist_ids)
       and status = 'waiting';
    get diagnostics v_affected = row_count;
  end if;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'action', v_action,
    'affected_count', v_affected,
    'team_id', v_team_id,
    'message', case
      when v_action = 'pair' then 'Waitlisted players were paired as a confirmed team.'
      else 'Waitlist entries were withdrawn.'
    end,
    'idempotent', false
  );
  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_club_id,
    coalesce(v_actor_email, 'unknown'),
    v_actor_role,
    'team_league_waitlist_' || v_action,
    'team_league',
    v_league_name,
    pg_catalog.jsonb_build_object(
      'waitlist_ids', pg_catalog.to_jsonb(p_waitlist_ids)
    ),
    v_result - 'idempotent',
    null,
    v_source,
    false
  );
  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

-- A canonical rated match may commit just before this call. Fixture state,
-- team standings version, operation receipt, and audit then finalize together.
create or replace function public.team_league_finalize_fixture_v1(
  p_operation_id uuid,
  p_club_id text,
  p_fixture_id uuid,
  p_status text,
  p_team_a_score integer,
  p_team_b_score integer,
  p_winner_team_id uuid,
  p_official_match_id bigint,
  p_team_a_player_1_id bigint,
  p_team_a_player_2_id bigint,
  p_team_b_player_1_id bigint,
  p_team_b_player_2_id bigint,
  p_substitutions jsonb,
  p_score_note text,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation public.team_league_operations%rowtype;
  v_before public.team_league_fixtures%rowtype;
  v_fixture public.team_league_fixtures%rowtype;
  v_status text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_status, '')));
  v_actor_email text := nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), 320), '');
  v_actor_role text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_role)), 80), ''),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_score'
  );
  v_note text := nullif(pg_catalog.left(pg_catalog.btrim(p_score_note), 500), '');
  v_resolved_count integer := 0;
  v_result jsonb;
begin
  if p_operation_id is null
     or nullif(pg_catalog.btrim(p_club_id), '') is null
     or p_fixture_id is null
     or v_status not in ('complete', 'forfeit')
     or p_winner_team_id is null
     or p_substitutions is null
     or pg_catalog.jsonb_typeof(p_substitutions) <> 'array' then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_FIXTURE_FINALIZE_INVALID';
  end if;
  if v_status = 'complete' and (
    p_team_a_score is null
    or p_team_b_score is null
    or p_team_a_score < 0
    or p_team_b_score < 0
    or p_team_a_score = p_team_b_score
    or p_official_match_id is null
    or p_team_a_player_1_id is null
    or p_team_a_player_2_id is null
    or p_team_b_player_1_id is null
    or p_team_b_player_2_id is null
    or pg_catalog.cardinality(array[
      p_team_a_player_1_id,
      p_team_a_player_2_id,
      p_team_b_player_1_id,
      p_team_b_player_2_id
    ]) <> pg_catalog.cardinality(array(
      select distinct player_id
      from unnest(array[
        p_team_a_player_1_id,
        p_team_a_player_2_id,
        p_team_b_player_1_id,
        p_team_b_player_2_id
      ]) as player_id
    ))
  ) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_PLAYED_FIXTURE_INVALID';
  end if;
  if v_status = 'forfeit' and (
    p_team_a_score is not null
    or p_team_b_score is not null
    or p_official_match_id is not null
    or pg_catalog.jsonb_array_length(p_substitutions) <> 0
  ) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_FORFEIT_FIXTURE_INVALID';
  end if;

  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.id = p_operation_id
     and operation.club_id = pg_catalog.btrim(p_club_id)
   for update;
  if not found or v_operation.operation_type <> 'admin_score_fixture' then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SCORE_OPERATION_NOT_FOUND';
  end if;
  if v_operation.status = 'complete' and v_operation.result_json is not null then
    return v_operation.result_json || '{"idempotent": true}'::jsonb;
  end if;

  select fixture.*
    into v_fixture
    from public.team_league_fixtures as fixture
   where fixture.id = p_fixture_id
     and fixture.club_id = pg_catalog.btrim(p_club_id)
     and fixture.league_name = v_operation.league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_FIXTURE_NOT_FOUND';
  end if;
  v_before := v_fixture;
  if v_fixture.score_operation_id = p_operation_id
     and v_fixture.status in ('complete', 'forfeit') then
    null;
  elsif v_fixture.status <> 'scheduled'
        or v_fixture.score_operation_id is not null then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_FIXTURE_CHANGED';
  else
    if p_winner_team_id not in (v_fixture.team_a_id, v_fixture.team_b_id) then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_FIXTURE_WINNER_INVALID';
    end if;
    update public.team_league_fixtures
       set status = v_status,
           resolution = case
             when v_status = 'complete' then 'played'
             else 'forfeit'
           end,
           team_a_score = p_team_a_score,
           team_b_score = p_team_b_score,
           winner_team_id = p_winner_team_id,
           official_match_id = p_official_match_id,
           team_a_player_1_id = p_team_a_player_1_id,
           team_a_player_2_id = p_team_a_player_2_id,
           team_b_player_1_id = p_team_b_player_1_id,
           team_b_player_2_id = p_team_b_player_2_id,
           substitutions_json = p_substitutions,
           score_note = v_note,
           score_operation_id = p_operation_id,
           scored_by = v_actor_email,
           scored_at = pg_catalog.clock_timestamp(),
           updated_at = pg_catalog.clock_timestamp()
     where id = p_fixture_id
    returning * into v_fixture;
  end if;

  if v_fixture.phase = 'playoff' and v_fixture.winner_team_id is not null then
    loop
      update public.team_league_fixtures as target
         set team_a_id = case
               when target.team_a_id is null
                and target.team_a_source =
                  'winner:' || source.round_number::text || ':' ||
                  source.bracket_slot::text
                 then source.winner_team_id
               else target.team_a_id
             end,
             team_b_id = case
               when target.team_b_id is null
                and target.team_b_source =
                  'winner:' || source.round_number::text || ':' ||
                  source.bracket_slot::text
                 then source.winner_team_id
               else target.team_b_id
             end,
             updated_at = pg_catalog.clock_timestamp()
        from public.team_league_fixtures as source
       where target.club_id = v_fixture.club_id
         and target.league_name = v_fixture.league_name
         and target.phase = 'playoff'
         and source.club_id = target.club_id
         and source.league_name = target.league_name
         and source.phase = 'playoff'
         and source.winner_team_id is not null
         and (
           (
             target.team_a_id is null
             and target.team_a_source =
               'winner:' || source.round_number::text || ':' ||
               source.bracket_slot::text
           )
           or (
             target.team_b_id is null
             and target.team_b_source =
               'winner:' || source.round_number::text || ':' ||
               source.bracket_slot::text
           )
         );
      get diagnostics v_resolved_count = row_count;
      exit when v_resolved_count = 0;
    end loop;
  end if;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'fixture_id', p_fixture_id,
    'status', v_fixture.status,
    'winner_team_id', v_fixture.winner_team_id,
    'official_match_id', v_fixture.official_match_id,
    'rated', v_fixture.status = 'complete',
    'substitution_count', pg_catalog.jsonb_array_length(v_fixture.substitutions_json),
    'message', case
      when v_fixture.status = 'complete'
        then 'Result saved and player ratings updated.'
      else 'Forfeit saved without a rated score.'
    end,
    'idempotent', false
  );
  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_fixture.club_id,
    coalesce(v_actor_email, 'unknown'),
    v_actor_role,
    'team_league_fixture_scored',
    'team_league_fixture',
    p_fixture_id::text,
    pg_catalog.to_jsonb(v_before),
    pg_catalog.to_jsonb(v_fixture),
    case
      when pg_catalog.jsonb_array_length(v_fixture.substitutions_json) > 0
        then pg_catalog.jsonb_array_length(v_fixture.substitutions_json)::text
          || ' documented substitution(s).'
      else null
    end,
    v_source,
    false
  );
  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         recovery_note = null,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

create or replace function public.team_league_reconcile_fixture_v1(
  p_operation_id uuid,
  p_club_id text,
  p_fixture_id uuid,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation public.team_league_operations%rowtype;
  v_before public.team_league_fixtures%rowtype;
  v_updated public.team_league_fixtures%rowtype;
  v_match public.matches%rowtype;
  v_actor_email text := nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), 320), '');
  v_actor_role text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_role)), 80), ''),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_reconcile'
  );
  v_result jsonb;
begin
  if p_operation_id is null
     or nullif(pg_catalog.btrim(p_club_id), '') is null
     or p_fixture_id is null then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_RECONCILE_INVALID';
  end if;
  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.id = p_operation_id
     and operation.club_id = pg_catalog.btrim(p_club_id)
   for update;
  if not found or v_operation.operation_type <> 'admin_reconcile_fixture' then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_RECONCILE_OPERATION_NOT_FOUND';
  end if;
  if v_operation.status = 'complete' and v_operation.result_json is not null then
    return v_operation.result_json || '{"idempotent": true}'::jsonb;
  end if;
  select fixture.*
    into v_before
    from public.team_league_fixtures as fixture
   where fixture.id = p_fixture_id
     and fixture.club_id = pg_catalog.btrim(p_club_id)
     and fixture.league_name = v_operation.league_name
   for update;
  if not found or v_before.official_match_id is null then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_CANONICAL_MATCH_NOT_LINKED';
  end if;
  select match_row.*
    into v_match
    from public.matches as match_row
   where match_row.club_id = pg_catalog.btrim(p_club_id)
     and match_row.id = v_before.official_match_id;
  if not found
     or v_match.deleted_at is not null
     or coalesce(v_match.excluded_from_ratings, false) then
    update public.team_league_fixtures
       set status = 'cancelled',
           resolution = 'cancelled',
           winner_team_id = null,
           team_a_score = null,
           team_b_score = null,
           score_note = 'Canonical match was excluded or removed through Match Log.',
           updated_at = pg_catalog.clock_timestamp()
     where id = p_fixture_id
    returning * into v_updated;
  else
    if v_match.score_t1 is null
       or v_match.score_t2 is null
       or v_match.score_t1 = v_match.score_t2 then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_CANONICAL_MATCH_SCORE_INVALID';
    end if;
    update public.team_league_fixtures
       set status = 'complete',
           resolution = 'played',
           team_a_score = v_match.score_t1,
           team_b_score = v_match.score_t2,
           winner_team_id = case
             when v_match.score_t1 > v_match.score_t2 then team_a_id
             else team_b_id
           end,
           score_note = null,
           updated_at = pg_catalog.clock_timestamp()
     where id = p_fixture_id
    returning * into v_updated;
  end if;
  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'fixture_id', p_fixture_id,
    'status', v_updated.status,
    'official_match_id', v_updated.official_match_id,
    'message', 'Fixture refreshed from the canonical match.',
    'idempotent', false
  );
  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_updated.club_id,
    coalesce(v_actor_email, 'unknown'),
    v_actor_role,
    'team_league_fixture_reconciled',
    'team_league_fixture',
    p_fixture_id::text,
    pg_catalog.to_jsonb(v_before),
    pg_catalog.to_jsonb(v_updated),
    null,
    v_source,
    false
  );
  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         recovery_note = null,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

-- Settings use an explicit version compare-and-swap. The receipt, settings
-- write, and audit record commit in one transaction.
create or replace function public.team_league_save_settings_v1(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_expected_settings_version integer,
  p_settings jsonb,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_league_name text := nullif(pg_catalog.left(pg_catalog.btrim(p_league_name), 120), '');
  v_key text := nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_fingerprint text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_actor_email text := nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), 320), '');
  v_actor_role text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_role)), 80), ''),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_settings'
  );
  v_operation public.team_league_operations%rowtype;
  v_before public.team_league_settings%rowtype;
  v_updated public.team_league_settings%rowtype;
  v_current_version integer := 0;
  v_next_status text;
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_league_name is null
     or v_key is null
     or v_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$'
     or v_fingerprint !~ '^[0-9a-f]{64}$'
     or p_expected_settings_version is null
     or p_expected_settings_version < 0
     or p_settings is null
     or pg_catalog.jsonb_typeof(p_settings) <> 'object' then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SETTINGS_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-admin:' || v_club_id || ':' || v_key,
      0
    )
  );

  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.club_id = v_club_id
     and operation.idempotency_key = v_key
   for update;
  if found then
    if v_operation.request_fingerprint <> v_fingerprint then
      raise exception using
        errcode = '23505',
        message = 'TEAM_LEAGUE_IDEMPOTENCY_CONFLICT';
    end if;
    if v_operation.status = 'complete' and v_operation.result_json is not null then
      return v_operation.result_json || '{"idempotent": true}'::jsonb;
    end if;
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_OPERATION_RECOVERY_REQUIRED';
  end if;

  if not exists (
    select 1
      from public.leagues_metadata as league
     where league.club_id = v_club_id
       and league.league_name = v_league_name
  ) then
    raise exception using
      errcode = '23503',
      message = 'TEAM_LEAGUE_LEAGUE_NOT_FOUND';
  end if;

  select settings.*
    into v_before
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name
   for update;
  if found then
    v_current_version := v_before.settings_version;
  end if;
  if v_current_version <> p_expected_settings_version then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_SETTINGS_VERSION_CONFLICT';
  end if;

  v_next_status := case
    when coalesce((p_settings ->> 'registration_open')::boolean, false)
      then 'registration_open'
    when v_before.status = 'registration_open'
      then 'registration_closed'
    else coalesce(v_before.status, 'draft')
  end;

  insert into public.team_league_operations (
    id,
    club_id,
    league_name,
    idempotency_key,
    request_fingerprint,
    operation_type,
    status,
    request_json,
    actor_email,
    actor_role,
    source
  ) values (
    p_operation_id,
    v_club_id,
    v_league_name,
    v_key,
    v_fingerprint,
    'admin_save_settings',
    'started',
    pg_catalog.jsonb_build_object(
      'settings', p_settings,
      'expected_settings_version', p_expected_settings_version
    ),
    v_actor_email,
    v_actor_role,
    v_source
  );

  insert into public.team_league_settings (
    club_id,
    league_name,
    status,
    registration_open,
    allow_substitutes,
    playoff_format,
    playoff_team_count,
    start_date,
    weekday,
    start_time,
    timezone,
    venue,
    registration_closes_at,
    settings_version,
    created_by,
    updated_by,
    updated_at
  ) values (
    v_club_id,
    v_league_name,
    v_next_status,
    coalesce((p_settings ->> 'registration_open')::boolean, false),
    coalesce((p_settings ->> 'allow_substitutes')::boolean, false),
    coalesce(nullif(p_settings ->> 'playoff_format', ''), 'none'),
    nullif(p_settings ->> 'playoff_team_count', '')::integer,
    nullif(p_settings ->> 'start_date', '')::date,
    coalesce(nullif(p_settings ->> 'weekday', '')::smallint, 0),
    coalesce(nullif(p_settings ->> 'start_time', '')::time, '18:00'::time),
    coalesce(nullif(pg_catalog.left(p_settings ->> 'timezone', 80), ''), 'UTC'),
    nullif(pg_catalog.left(pg_catalog.btrim(p_settings ->> 'venue'), 240), ''),
    nullif(p_settings ->> 'registration_closes_at', '')::timestamptz,
    p_expected_settings_version + 1,
    coalesce(v_before.created_by, v_actor_email),
    v_actor_email,
    pg_catalog.clock_timestamp()
  )
  on conflict (club_id, league_name) do update
    set status = excluded.status,
        registration_open = excluded.registration_open,
        allow_substitutes = excluded.allow_substitutes,
        playoff_format = excluded.playoff_format,
        playoff_team_count = excluded.playoff_team_count,
        start_date = excluded.start_date,
        weekday = excluded.weekday,
        start_time = excluded.start_time,
        timezone = excluded.timezone,
        venue = excluded.venue,
        registration_closes_at = excluded.registration_closes_at,
        settings_version = excluded.settings_version,
        updated_by = excluded.updated_by,
        updated_at = excluded.updated_at
  returning * into v_updated;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'league_name', v_league_name,
    'settings', pg_catalog.to_jsonb(v_updated),
    'settings_version', v_updated.settings_version,
    'message', 'Team-league setup saved.',
    'idempotent', false
  );

  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_club_id,
    coalesce(v_actor_email, 'unknown'),
    v_actor_role,
    'team_league_settings_saved',
    'team_league',
    v_league_name,
    coalesce(pg_catalog.to_jsonb(v_before), '{}'::jsonb),
    pg_catalog.to_jsonb(v_updated),
    null,
    v_source,
    false
  );

  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

-- Schedule and bracket replacement is one version-checked transaction. A
-- scored target phase is immutable here; corrections stay in Match Log.
create or replace function public.team_league_replace_schedule_v1(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_phase text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_expected_schedule_version integer,
  p_expected_standings_version integer,
  p_fixtures jsonb,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_league_name text := nullif(pg_catalog.left(pg_catalog.btrim(p_league_name), 120), '');
  v_phase text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_phase, '')));
  v_key text := nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_fingerprint text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_actor_email text := nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), 320), '');
  v_actor_role text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_role)), 80), ''),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_schedule'
  );
  v_operation public.team_league_operations%rowtype;
  v_settings public.team_league_settings%rowtype;
  v_before_count integer := 0;
  v_inserted_count integer := 0;
  v_resolved_count integer := 0;
  v_next_version integer;
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_league_name is null
     or v_phase not in ('regular', 'playoff')
     or v_key is null
     or v_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$'
     or v_fingerprint !~ '^[0-9a-f]{64}$'
     or p_expected_schedule_version is null
     or p_expected_schedule_version < 0
     or p_expected_standings_version is null
     or p_expected_standings_version < 0
     or p_fixtures is null
     or pg_catalog.jsonb_typeof(p_fixtures) <> 'array'
     or pg_catalog.jsonb_array_length(p_fixtures) = 0 then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SCHEDULE_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-admin:' || v_club_id || ':' || v_key,
      0
    )
  );

  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.club_id = v_club_id
     and operation.idempotency_key = v_key
   for update;
  if found then
    if v_operation.request_fingerprint <> v_fingerprint then
      raise exception using
        errcode = '23505',
        message = 'TEAM_LEAGUE_IDEMPOTENCY_CONFLICT';
    end if;
    if v_operation.status = 'complete' and v_operation.result_json is not null then
      return v_operation.result_json || '{"idempotent": true}'::jsonb;
    end if;
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_OPERATION_RECOVERY_REQUIRED';
  end if;

  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;
  if v_settings.schedule_version <> p_expected_schedule_version then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_SCHEDULE_VERSION_CONFLICT';
  end if;
  if v_phase = 'playoff'
     and v_settings.standings_version <> p_expected_standings_version then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_STANDINGS_VERSION_CONFLICT';
  end if;
  if v_phase = 'regular' and exists (
    select 1
      from public.team_league_fixtures as fixture
     where fixture.club_id = v_club_id
       and fixture.league_name = v_league_name
       and fixture.phase = 'playoff'
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_PLAYOFFS_ALREADY_EXIST';
  end if;
  if exists (
    select 1
      from public.team_league_fixtures as fixture
     where fixture.club_id = v_club_id
       and fixture.league_name = v_league_name
       and fixture.phase = v_phase
       and (
         fixture.status in ('complete', 'forfeit')
         or fixture.official_match_id is not null
       )
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_SCORED_SCHEDULE_LOCKED';
  end if;

  if exists (
    select 1
      from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
        round_number integer,
        week_number integer,
        bracket_slot integer,
        team_a_id uuid,
        team_b_id uuid,
        team_a_source text,
        team_b_source text,
        status text
      )
     where item.round_number is null
        or item.round_number < 1
        or item.bracket_slot is null
        or item.bracket_slot < 1
        or item.status not in ('scheduled', 'bye')
        or (
          v_phase = 'regular'
          and (
            item.team_a_id is null
            or (
              item.status = 'scheduled'
              and item.team_b_id is null
            )
            or (
              item.status = 'bye'
              and item.team_b_id is not null
            )
          )
        )
        or (
          v_phase = 'playoff'
          and item.team_a_id is null
          and nullif(pg_catalog.btrim(item.team_a_source), '') is null
        )
  ) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_FIXTURE_SET_INVALID';
  end if;

  select pg_catalog.count(*)::integer
    into v_before_count
    from public.team_league_fixtures as fixture
   where fixture.club_id = v_club_id
     and fixture.league_name = v_league_name
     and fixture.phase = v_phase;

  insert into public.team_league_operations (
    id,
    club_id,
    league_name,
    idempotency_key,
    request_fingerprint,
    operation_type,
    status,
    request_json,
    actor_email,
    actor_role,
    source
  ) values (
    p_operation_id,
    v_club_id,
    v_league_name,
    v_key,
    v_fingerprint,
    case
      when v_phase = 'playoff' then 'admin_generate_playoffs'
      else 'admin_generate_schedule'
    end,
    'started',
    pg_catalog.jsonb_build_object(
      'phase', v_phase,
      'expected_schedule_version', p_expected_schedule_version,
      'expected_standings_version', p_expected_standings_version,
      'fixtures', p_fixtures
    ),
    v_actor_email,
    v_actor_role,
    v_source
  );

  delete from public.team_league_fixtures
   where club_id = v_club_id
     and league_name = v_league_name
     and phase = v_phase;

  insert into public.team_league_fixtures (
    club_id,
    league_name,
    phase,
    round_number,
    week_number,
    bracket_slot,
    scheduled_at,
    team_a_id,
    team_b_id,
    team_a_source,
    team_b_source,
    status,
    resolution,
    schedule_operation_id
  )
  select
    v_club_id,
    v_league_name,
    v_phase,
    item.round_number,
    item.week_number,
    item.bracket_slot,
    item.scheduled_at,
    item.team_a_id,
    item.team_b_id,
    nullif(pg_catalog.left(pg_catalog.btrim(item.team_a_source), 120), ''),
    nullif(pg_catalog.left(pg_catalog.btrim(item.team_b_source), 120), ''),
    item.status,
    case when item.status = 'bye' then 'bye' else null end,
    p_operation_id
  from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
    round_number integer,
    week_number integer,
    bracket_slot integer,
    scheduled_at timestamptz,
    team_a_id uuid,
    team_b_id uuid,
    team_a_source text,
    team_b_source text,
    status text
  );
  get diagnostics v_inserted_count = row_count;

  if v_phase = 'playoff' then
    update public.team_league_teams
       set playoff_seed = null,
           playoff_seed_standings_version = null,
           playoff_seeded_at = null,
           updated_at = pg_catalog.clock_timestamp()
     where club_id = v_club_id
       and league_name = v_league_name;
    update public.team_league_teams as team
       set playoff_seed = seed.seed_number,
           playoff_seed_standings_version = v_settings.standings_version,
           playoff_seeded_at = pg_catalog.clock_timestamp(),
           updated_at = pg_catalog.clock_timestamp()
      from (
        select item.team_a_id as team_id,
               pg_catalog.split_part(item.team_a_source, ':', 2)::integer
                 as seed_number
          from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
            team_a_id uuid,
            team_b_id uuid,
            team_a_source text,
            team_b_source text
          )
         where item.team_a_id is not null
           and item.team_a_source ~ '^seed:[0-9]+$'
        union all
        select item.team_b_id as team_id,
               pg_catalog.split_part(item.team_b_source, ':', 2)::integer
                 as seed_number
          from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
            team_a_id uuid,
            team_b_id uuid,
            team_a_source text,
            team_b_source text
          )
         where item.team_b_id is not null
           and item.team_b_source ~ '^seed:[0-9]+$'
      ) as seed
     where team.id = seed.team_id
       and team.club_id = v_club_id
       and team.league_name = v_league_name;

    update public.team_league_fixtures
       set winner_team_id = team_a_id,
           updated_at = pg_catalog.clock_timestamp()
     where club_id = v_club_id
       and league_name = v_league_name
       and phase = 'playoff'
       and status = 'bye'
       and team_a_id is not null
       and team_b_id is null;

    loop
      update public.team_league_fixtures as target
         set team_a_id = case
               when target.team_a_id is null
                and target.team_a_source =
                  'winner:' || source.round_number::text || ':' ||
                  source.bracket_slot::text
                 then source.winner_team_id
               else target.team_a_id
             end,
             team_b_id = case
               when target.team_b_id is null
                and target.team_b_source =
                  'winner:' || source.round_number::text || ':' ||
                  source.bracket_slot::text
                 then source.winner_team_id
               else target.team_b_id
             end,
             updated_at = pg_catalog.clock_timestamp()
        from public.team_league_fixtures as source
       where target.club_id = v_club_id
         and target.league_name = v_league_name
         and target.phase = 'playoff'
         and source.club_id = target.club_id
         and source.league_name = target.league_name
         and source.phase = 'playoff'
         and source.winner_team_id is not null
         and (
           (
             target.team_a_id is null
             and target.team_a_source =
               'winner:' || source.round_number::text || ':' ||
               source.bracket_slot::text
           )
           or (
             target.team_b_id is null
             and target.team_b_source =
               'winner:' || source.round_number::text || ':' ||
               source.bracket_slot::text
           )
         );
      get diagnostics v_resolved_count = row_count;
      exit when v_resolved_count = 0;
    end loop;
  end if;

  v_next_version := p_expected_schedule_version + 1;
  update public.team_league_settings
     set status = case
           when v_phase = 'playoff' then 'playoffs'
           else 'scheduled'
         end,
         registration_open = case
           when v_phase = 'regular' then false
           else registration_open
         end,
         schedule_version = v_next_version,
         updated_by = v_actor_email,
         updated_at = pg_catalog.clock_timestamp()
   where club_id = v_club_id
     and league_name = v_league_name;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'league_name', v_league_name,
    'phase', v_phase,
    'schedule_version', v_next_version,
    'fixture_count', v_inserted_count,
    'message', case
      when v_phase = 'playoff' then 'Playoff bracket generated.'
      else 'Single round-robin schedule generated.'
    end,
    'idempotent', false
  );

  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_club_id,
    coalesce(v_actor_email, 'unknown'),
    v_actor_role,
    case
      when v_phase = 'playoff' then 'team_league_playoffs_generated'
      else 'team_league_schedule_generated'
    end,
    'team_league',
    v_league_name,
    pg_catalog.jsonb_build_object(
      'fixture_count', v_before_count,
      'schedule_version', p_expected_schedule_version
    ),
    pg_catalog.jsonb_build_object(
      'fixture_count', v_inserted_count,
      'schedule_version', v_next_version,
      'standings_version', v_settings.standings_version
    ),
    null,
    v_source,
    false
  );

  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

-- Persist one immutable award result-set revision. Retrying the exact revision
-- is an idempotent read; changing its fingerprint is a conflict.
create or replace function public.league_awards_replace_records_v1(
  p_club_id text,
  p_league_name text,
  p_workflow_revision integer,
  p_preview_fingerprint text,
  p_result_fingerprint text,
  p_records jsonb,
  p_source_snapshot jsonb,
  p_finalized boolean,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_league_name text := nullif(
    pg_catalog.left(pg_catalog.btrim(p_league_name), 120),
    ''
  );
  v_preview_fingerprint text := pg_catalog.lower(
    pg_catalog.btrim(coalesce(p_preview_fingerprint, ''))
  );
  v_result_fingerprint text := pg_catalog.lower(
    pg_catalog.btrim(coalesce(p_result_fingerprint, ''))
  );
  v_actor_email text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_email)),
        320
      ),
      ''
    ),
    'unknown'
  );
  v_actor_role text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_role)),
        80
      ),
      ''
    ),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_league_manager_awards'
  );
  v_existing public.league_award_result_sets%rowtype;
  v_record_count integer := 0;
  v_result jsonb;
begin
  if v_club_id is null
     or v_league_name is null
     or p_workflow_revision is null
     or p_workflow_revision < 0
     or v_preview_fingerprint !~ '^[0-9a-f]{64}$'
     or v_result_fingerprint !~ '^[0-9a-f]{64}$'
     or p_records is null
     or pg_catalog.jsonb_typeof(p_records) <> 'array'
     or p_source_snapshot is null
     or pg_catalog.jsonb_typeof(p_source_snapshot) <> 'object'
     or p_finalized is null then
    raise exception using
      errcode = '22023',
      message = 'LEAGUE_AWARD_RECORD_SET_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:league-awards:' || v_club_id || ':' || v_league_name,
      0
    )
  );

  select result_set.*
    into v_existing
    from public.league_award_result_sets as result_set
   where result_set.club_id = v_club_id
     and result_set.league_name = v_league_name
     and result_set.workflow_revision = p_workflow_revision
   for update;
  if found then
    if v_existing.preview_fingerprint = v_preview_fingerprint
       and v_existing.result_fingerprint = v_result_fingerprint
       and v_existing.record_count = pg_catalog.jsonb_array_length(p_records)
       and (v_existing.finalized_at is not null) = p_finalized then
      return pg_catalog.jsonb_build_object(
        'ok', true,
        'committed', true,
        'idempotent', true,
        'workflow_revision', p_workflow_revision,
        'record_count', v_existing.record_count,
        'preview_fingerprint', v_preview_fingerprint,
        'result_fingerprint', v_result_fingerprint,
        'finalized', p_finalized
      );
    end if;
    raise exception using
      errcode = '40001',
      message = 'LEAGUE_AWARD_RECORD_REVISION_CONFLICT';
  end if;

  if exists (
    select 1
      from pg_catalog.jsonb_to_recordset(p_records) as item(
        award_key text,
        category_key text,
        category_label text,
        recipient_type text,
        player_id bigint,
        team_id uuid,
        recipient_name text,
        placement integer,
        metric_display text,
        source_snapshot jsonb
      )
     where nullif(pg_catalog.btrim(item.award_key), '') is null
        or nullif(pg_catalog.btrim(item.category_key), '') is null
        or nullif(pg_catalog.btrim(item.category_label), '') is null
        or item.recipient_type not in ('player', 'team')
        or nullif(pg_catalog.btrim(item.recipient_name), '') is null
        or item.placement not between 1 and 3
        or nullif(pg_catalog.btrim(item.metric_display), '') is null
        or item.source_snapshot is null
        or pg_catalog.jsonb_typeof(item.source_snapshot) <> 'object'
        or (
          item.recipient_type = 'player'
          and (item.player_id is null or item.team_id is not null)
        )
        or (
          item.recipient_type = 'team'
          and (item.team_id is null or item.player_id is not null)
        )
  ) then
    raise exception using
      errcode = '22023',
      message = 'LEAGUE_AWARD_RECORD_INVALID';
  end if;

  insert into public.league_award_result_sets (
    club_id,
    league_name,
    workflow_revision,
    preview_fingerprint,
    result_fingerprint,
    record_count,
    source_snapshot,
    finalized_at,
    created_by
  ) values (
    v_club_id,
    v_league_name,
    p_workflow_revision,
    v_preview_fingerprint,
    v_result_fingerprint,
    pg_catalog.jsonb_array_length(p_records),
    p_source_snapshot,
    case when p_finalized then pg_catalog.clock_timestamp() else null end,
    v_actor_email
  );

  insert into public.league_award_result_records (
    club_id,
    league_name,
    workflow_revision,
    preview_fingerprint,
    result_fingerprint,
    award_key,
    category_key,
    category_label,
    recipient_type,
    player_id,
    team_id,
    recipient_name,
    placement,
    is_co_winner,
    metric_value,
    computed_metric_value,
    computed_player_id,
    computed_team_id,
    computed_recipient_name,
    metric_display,
    manual_label,
    is_override,
    override_reason,
    public_visible,
    source_snapshot,
    finalized_at
  )
  select
    v_club_id,
    v_league_name,
    p_workflow_revision,
    v_preview_fingerprint,
    v_result_fingerprint,
    pg_catalog.left(pg_catalog.btrim(item.award_key), 240),
    pg_catalog.left(pg_catalog.btrim(item.category_key), 80),
    pg_catalog.left(pg_catalog.btrim(item.category_label), 160),
    item.recipient_type,
    item.player_id,
    item.team_id,
    pg_catalog.left(pg_catalog.btrim(item.recipient_name), 160),
    coalesce(item.placement, 1),
    coalesce(item.is_co_winner, false),
    item.metric_value,
    item.computed_metric_value,
    item.computed_player_id,
    item.computed_team_id,
    nullif(
      pg_catalog.left(pg_catalog.btrim(item.computed_recipient_name), 160),
      ''
    ),
    pg_catalog.left(pg_catalog.btrim(item.metric_display), 240),
    nullif(pg_catalog.left(pg_catalog.btrim(item.manual_label), 160), ''),
    coalesce(item.is_override, false),
    nullif(pg_catalog.left(pg_catalog.btrim(item.override_reason), 500), ''),
    coalesce(item.public_visible, true),
    item.source_snapshot,
    case when p_finalized then pg_catalog.clock_timestamp() else null end
  from pg_catalog.jsonb_to_recordset(p_records) as item(
    award_key text,
    category_key text,
    category_label text,
    recipient_type text,
    player_id bigint,
    team_id uuid,
    recipient_name text,
    placement integer,
    is_co_winner boolean,
    metric_value numeric,
    computed_metric_value numeric,
    computed_player_id bigint,
    computed_team_id uuid,
    computed_recipient_name text,
    metric_display text,
    manual_label text,
    is_override boolean,
    override_reason text,
    public_visible boolean,
    source_snapshot jsonb
  );
  get diagnostics v_record_count = row_count;
  if v_record_count <> pg_catalog.jsonb_array_length(p_records) then
    raise exception using
      errcode = '40001',
      message = 'LEAGUE_AWARD_RECORD_COUNT_MISMATCH';
  end if;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'idempotent', false,
    'workflow_revision', p_workflow_revision,
    'record_count', v_record_count,
    'preview_fingerprint', v_preview_fingerprint,
    'result_fingerprint', v_result_fingerprint,
    'finalized', p_finalized
  );
  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_club_id,
    v_actor_email,
    v_actor_role,
    'league_award_result_set_saved',
    'league_awards',
    v_league_name,
    pg_catalog.jsonb_build_object(
      'workflow_revision', p_workflow_revision - 1
    ),
    v_result,
    null,
    v_source,
    false
  );
  return v_result;
end
$function$;

-- Recovery decisions are made from bounded FastAPI evidence, then the receipt
-- transition and recovery audit commit together. A compensated operation may
-- be retried with the exact original key and fingerprint; a finalized one is
-- an immutable idempotent receipt.
create or replace function public.team_league_resolve_operation_v1(
  p_operation_id uuid,
  p_club_id text,
  p_resolution text,
  p_result jsonb,
  p_recovery_note text,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation public.team_league_operations%rowtype;
  v_resolution text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_resolution, '')));
  v_actor_email text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), 320), ''),
    'unknown'
  );
  v_actor_role text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_role)), 80), ''),
    'unknown'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_recovery'
  );
  v_note text := nullif(pg_catalog.left(pg_catalog.btrim(p_recovery_note), 500), '');
  v_result jsonb := p_result;
begin
  if p_operation_id is null
     or nullif(pg_catalog.btrim(p_club_id), '') is null
     or v_resolution not in ('finalize', 'compensate')
     or (
       v_resolution = 'finalize'
       and (v_result is null or pg_catalog.jsonb_typeof(v_result) <> 'object')
     ) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_RECOVERY_RESOLUTION_INVALID';
  end if;

  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.id = p_operation_id
     and operation.club_id = pg_catalog.btrim(p_club_id)
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_OPERATION_NOT_FOUND';
  end if;
  if v_operation.status = 'complete' then
    return coalesce(v_operation.result_json, '{}'::jsonb)
      || '{"idempotent": true, "safe_action": "none"}'::jsonb;
  end if;

  if v_resolution = 'finalize' then
    v_result := v_result
      || pg_catalog.jsonb_build_object(
        'ok', true,
        'committed', true,
        'operation_id', p_operation_id,
        'idempotent', true,
        'recovered', true
      );
    update public.team_league_operations
       set status = 'complete',
           result_json = v_result,
           recovery_note = v_note,
           completed_at = pg_catalog.clock_timestamp(),
           updated_at = pg_catalog.clock_timestamp()
     where id = p_operation_id;
  else
    v_result := pg_catalog.jsonb_build_object(
      'ok', true,
      'committed', false,
      'operation_id', p_operation_id,
      'idempotent', false,
      'recovered', true,
      'safe_action', 'retry_same_request',
      'status', 'compensated'
    );
    update public.team_league_operations
       set status = 'compensated',
           result_json = null,
           recovery_note = v_note,
           completed_at = pg_catalog.clock_timestamp(),
           updated_at = pg_catalog.clock_timestamp()
     where id = p_operation_id;
  end if;

  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_operation.club_id,
    v_actor_email,
    v_actor_role,
    'team_league_operation_' || v_resolution,
    'team_league_operation',
    p_operation_id::text,
    pg_catalog.jsonb_build_object(
      'status', v_operation.status,
      'operation_type', v_operation.operation_type
    ),
    pg_catalog.jsonb_build_object(
      'status', case when v_resolution = 'finalize' then 'complete' else 'compensated' end,
      'operation_type', v_operation.operation_type,
      'result', v_result
    ),
    v_note,
    v_source,
    false
  );
  return v_result;
end
$function$;

alter table public.team_league_settings enable row level security;
alter table public.team_league_settings force row level security;
alter table public.team_league_teams enable row level security;
alter table public.team_league_teams force row level security;
alter table public.team_league_solo_waitlist enable row level security;
alter table public.team_league_solo_waitlist force row level security;
alter table public.team_league_fixtures enable row level security;
alter table public.team_league_fixtures force row level security;
alter table public.team_league_operations enable row level security;
alter table public.team_league_operations force row level security;
alter table public.league_award_result_sets enable row level security;
alter table public.league_award_result_sets force row level security;
alter table public.league_award_result_records enable row level security;
alter table public.league_award_result_records force row level security;

revoke all on table public.team_league_settings from public, anon, authenticated;
revoke all on table public.team_league_teams from public, anon, authenticated;
revoke all on table public.team_league_solo_waitlist from public, anon, authenticated;
revoke all on table public.team_league_fixtures from public, anon, authenticated;
revoke all on table public.team_league_operations from public, anon, authenticated;
revoke all on table public.league_award_result_sets from public, anon, authenticated;
revoke all on table public.league_award_result_records from public, anon, authenticated;

grant usage on schema public to service_role;
grant select, insert, update, delete on table public.team_league_settings to service_role;
grant select, insert, update, delete on table public.team_league_teams to service_role;
grant select, insert, update, delete on table public.team_league_solo_waitlist to service_role;
grant select, insert, update, delete on table public.team_league_fixtures to service_role;
grant select, insert, update on table public.team_league_operations to service_role;
grant select, insert, update, delete on table public.league_award_result_sets to service_role;
grant select, insert, update, delete on table public.league_award_result_records to service_role;

revoke all on function public.enforce_team_league_unique_active_players_v1()
  from public, anon, authenticated;
revoke all on function public.bump_team_league_standings_version_v1()
  from public, anon, authenticated;
grant execute on function public.enforce_team_league_unique_active_players_v1()
  to service_role;
grant execute on function public.bump_team_league_standings_version_v1()
  to service_role;
revoke all on function public.team_league_register_public_v1(
  uuid, text, text, text, text, text, bigint, bigint, text, text, text, text, text,
  timestamptz, text
) from public, anon, authenticated;
revoke all on function public.team_league_confirm_partner_public_v1(
  uuid, uuid, text, boolean, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_register_public_v1(
  uuid, text, text, text, text, text, bigint, bigint, text, text, text, text, text,
  timestamptz, text
) to service_role;
grant execute on function public.team_league_confirm_partner_public_v1(
  uuid, uuid, text, boolean, text, text, text
) to service_role;
revoke all on function public.team_league_claim_partner_invitation_v1(
  uuid, text, uuid
) from public, anon, authenticated;
grant execute on function public.team_league_claim_partner_invitation_v1(
  uuid, text, uuid
) to service_role;
revoke all on function public.team_league_finish_partner_invitation_v1(
  uuid, uuid, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_finish_partner_invitation_v1(
  uuid, uuid, text, text, text, text
) to service_role;
revoke all on function public.team_league_admin_waitlist_action_v1(
  uuid, text, text, text, text, text, uuid[], text, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_admin_waitlist_action_v1(
  uuid, text, text, text, text, text, uuid[], text, text, text, text
) to service_role;
revoke all on function public.team_league_finalize_fixture_v1(
  uuid, text, uuid, text, integer, integer, uuid, bigint,
  bigint, bigint, bigint, bigint, jsonb, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_finalize_fixture_v1(
  uuid, text, uuid, text, integer, integer, uuid, bigint,
  bigint, bigint, bigint, bigint, jsonb, text, text, text, text
) to service_role;
revoke all on function public.team_league_reconcile_fixture_v1(
  uuid, text, uuid, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_reconcile_fixture_v1(
  uuid, text, uuid, text, text, text
) to service_role;
revoke all on function public.team_league_save_settings_v1(
  uuid, text, text, text, text, integer, jsonb, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_save_settings_v1(
  uuid, text, text, text, text, integer, jsonb, text, text, text
) to service_role;
revoke all on function public.team_league_replace_schedule_v1(
  uuid, text, text, text, text, text, integer, integer, jsonb, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_replace_schedule_v1(
  uuid, text, text, text, text, text, integer, integer, jsonb, text, text, text
) to service_role;
revoke all on function public.league_awards_replace_records_v1(
  text, text, integer, text, text, jsonb, jsonb, boolean, text, text, text
) from public, anon, authenticated;
grant execute on function public.league_awards_replace_records_v1(
  text, text, integer, text, text, jsonb, jsonb, boolean, text, text, text
) to service_role;
revoke all on function public.team_league_resolve_operation_v1(
  uuid, text, text, jsonb, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_resolve_operation_v1(
  uuid, text, text, jsonb, text, text, text, text
) to service_role;

comment on table public.team_league_settings is
  'Private fixed-partner league configuration. FastAPI service-role access only.';
comment on table public.team_league_teams is
  'Private confirmed and pending fixed-partner registrations; invite tokens are stored only as SHA-256 hashes.';
comment on table public.team_league_solo_waitlist is
  'Private solo-player pairing waitlist. Public responses never expose contact email.';
comment on table public.team_league_fixtures is
  'Private regular-season and playoff fixtures linked to canonical rated matches when played.';
comment on table public.team_league_operations is
  'Durable idempotency and recovery receipts for every team-league mutation.';
comment on table public.league_award_result_sets is
  'Immutable, idempotent award result-set revisions with exact frozen analytics provenance.';
comment on table public.league_award_result_records is
  'Durable award snapshots including co-winners, actual override metrics, manual labels, and public visibility.';

notify pgrst, 'reload schema';
