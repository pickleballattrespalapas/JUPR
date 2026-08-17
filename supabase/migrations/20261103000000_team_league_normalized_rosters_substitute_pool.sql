-- Normalize Team League season rosters while preserving the existing team,
-- fixture, standings, and canonical doubles-match identities.
--
-- The playing lineup remains exactly two players. A team may carry 2-4
-- primary season players plus configured alternates. One-off replacements must
-- opt in through the league substitute pool and are still recorded on the
-- fixture's existing player columns and substitutions_json audit snapshot.

do $requirements$
begin
  if to_regclass('public.team_league_settings') is null
     or to_regclass('public.team_league_teams') is null
     or to_regclass('public.team_league_fixtures') is null
     or to_regclass('public.team_league_operations') is null
     or to_regclass('public.players') is null
     or to_regclass('public.leagues_metadata') is null
     or to_regclass('public.admin_activity_log') is null
     or to_regclass('public.matches') is null
     or to_regprocedure('public.team_league_finalize_fixture_v1(uuid,text,uuid,text,integer,integer,uuid,bigint,bigint,bigint,bigint,bigint,jsonb,text,text,text,text)') is null
     or to_regprocedure('extensions.digest(text,text)') is null then
    raise exception using
      errcode = '42P01',
      message = 'normalized team rosters require the Team League base schema';
  end if;
end
$requirements$;

alter table public.team_league_settings
  add column if not exists max_alternates smallint not null default 0,
  add column if not exists substitute_pool_enabled boolean not null default false,
  add column if not exists mixed_required_men smallint not null default 1,
  add column if not exists mixed_required_women smallint not null default 1;

-- A normalized forming team may begin with only its captain. Legacy public
-- pair writers still populate both columns, while admin-created 3/4-player
-- teams can fill the second projection after the parent identity exists.
alter table public.team_league_teams
  alter column partner_player_id drop not null,
  alter column partner_contact_email drop not null;
alter table public.team_league_teams
  drop constraint if exists team_league_teams_distinct_players_check,
  drop constraint if exists team_league_teams_email_check;
alter table public.team_league_teams
  add constraint team_league_teams_distinct_players_check
    check (
      partner_player_id is null
      or captain_player_id <> partner_player_id
    ),
  add constraint team_league_teams_email_check
    check (
      pg_catalog.char_length(captain_contact_email) between 3 and 320
      and captain_contact_email like '%@%'
      and (
        partner_contact_email is null
        or (
          pg_catalog.char_length(partner_contact_email) between 3 and 320
          and partner_contact_email like '%@%'
        )
      )
    );

alter table public.team_league_settings
  drop constraint if exists team_league_settings_team_size_check;
alter table public.team_league_settings
  add constraint team_league_settings_team_size_check
  check (team_size between 2 and 4);

do $settings_constraints$
begin
  if not exists (
    select 1 from pg_catalog.pg_constraint
     where conname = 'team_league_settings_max_alternates_check'
       and conrelid = 'public.team_league_settings'::regclass
  ) then
    alter table public.team_league_settings
      add constraint team_league_settings_max_alternates_check
      check (max_alternates between 0 and 4);
  end if;
  if not exists (
    select 1 from pg_catalog.pg_constraint
     where conname = 'team_league_settings_mixed_composition_check'
       and conrelid = 'public.team_league_settings'::regclass
  ) then
    alter table public.team_league_settings
      add constraint team_league_settings_mixed_composition_check
      check (
        team_category <> 'mixed'
        or (
          mixed_required_men >= 1
          and mixed_required_women >= 1
          and mixed_required_men + mixed_required_women = team_size
        )
      );
  end if;
  if not exists (
    select 1 from pg_catalog.pg_constraint
     where conname = 'team_league_settings_pool_requires_substitutes_check'
       and conrelid = 'public.team_league_settings'::regclass
  ) then
    alter table public.team_league_settings
      add constraint team_league_settings_pool_requires_substitutes_check
      check (not substitute_pool_enabled or allow_substitutes);
  end if;
end
$settings_constraints$;

create table if not exists public.team_league_team_members (
  id uuid primary key default gen_random_uuid(),
  team_id uuid not null,
  club_id text not null,
  league_name text not null,
  player_id bigint not null,
  role text not null default 'primary',
  status text not null default 'invited',
  contact_email text,
  invitation_token_hash text,
  invitation_expires_at timestamptz,
  invitation_confirmed_at timestamptz,
  created_operation_id uuid,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  foreign key (club_id, league_name, team_id)
    references public.team_league_teams (club_id, league_name, id)
    on delete cascade,
  foreign key (created_operation_id)
    references public.team_league_operations (id),
  constraint team_league_team_members_role_check
    check (role in ('captain', 'primary', 'alternate')),
  constraint team_league_team_members_status_check
    check (status in ('invited', 'active', 'declined', 'removed')),
  constraint team_league_team_members_email_check
    check (
      contact_email is null
      or (
        pg_catalog.char_length(contact_email) between 3 and 320
        and contact_email like '%@%'
      )
    ),
  constraint team_league_team_members_invite_hash_check
    check (
      invitation_token_hash is null
      or invitation_token_hash ~ '^[0-9a-f]{64}$'
    ),
  unique (club_id, league_name, team_id, id)
);

create unique index if not exists team_league_team_members_team_player_idx
  on public.team_league_team_members (team_id, player_id)
  where status in ('invited', 'active');
create unique index if not exists team_league_team_members_league_player_idx
  on public.team_league_team_members (club_id, league_name, player_id)
  where status in ('invited', 'active');
create unique index if not exists team_league_team_members_captain_idx
  on public.team_league_team_members (team_id)
  where role = 'captain' and status in ('invited', 'active');
create unique index if not exists team_league_team_members_invite_hash_idx
  on public.team_league_team_members (invitation_token_hash)
  where invitation_token_hash is not null;
create index if not exists team_league_team_members_scope_idx
  on public.team_league_team_members (club_id, league_name, team_id, status, role);
create index if not exists team_league_team_members_player_idx
  on public.team_league_team_members (player_id);
create index if not exists team_league_team_members_operation_idx
  on public.team_league_team_members (created_operation_id)
  where created_operation_id is not null;

create table if not exists public.team_league_substitute_pool (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_name text not null,
  player_id bigint not null,
  status text not null default 'available',
  contact_email text,
  note text,
  created_operation_id uuid,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  foreign key (club_id, league_name)
    references public.team_league_settings (club_id, league_name)
    on delete cascade,
  foreign key (created_operation_id)
    references public.team_league_operations (id),
  constraint team_league_substitute_pool_status_check
    check (status in ('available', 'unavailable', 'withdrawn')),
  constraint team_league_substitute_pool_email_check
    check (
      contact_email is null
      or (
        pg_catalog.char_length(contact_email) between 3 and 320
        and contact_email like '%@%'
      )
    )
);

create unique index if not exists team_league_substitute_pool_current_player_idx
  on public.team_league_substitute_pool (club_id, league_name, player_id)
  where status in ('available', 'unavailable');
create index if not exists team_league_substitute_pool_scope_idx
  on public.team_league_substitute_pool (club_id, league_name, status, player_id);
create index if not exists team_league_substitute_pool_operation_idx
  on public.team_league_substitute_pool (created_operation_id)
  where created_operation_id is not null;

-- Existing fixed pairs become normalized two-player rosters without changing
-- any team, fixture, result, or invitation identity.
insert into public.team_league_team_members (
  team_id,
  club_id,
  league_name,
  player_id,
  role,
  status,
  contact_email,
  invitation_confirmed_at,
  created_operation_id,
  created_at,
  updated_at
)
select
  team.id,
  team.club_id,
  team.league_name,
  team.captain_player_id,
  'captain',
  case
    when team.status in ('pending_partner', 'confirmed') then 'active'
    else 'removed'
  end,
  team.captain_contact_email,
  case when team.status = 'confirmed' then team.partner_confirmed_at else null end,
  team.created_operation_id,
  team.created_at,
  team.updated_at
from public.team_league_teams as team
where not exists (
  select 1
    from public.team_league_team_members as member
   where member.team_id = team.id
     and member.player_id = team.captain_player_id
);

insert into public.team_league_team_members (
  team_id,
  club_id,
  league_name,
  player_id,
  role,
  status,
  contact_email,
  invitation_token_hash,
  invitation_expires_at,
  invitation_confirmed_at,
  created_operation_id,
  created_at,
  updated_at
)
select
  team.id,
  team.club_id,
  team.league_name,
  team.partner_player_id,
  'primary',
  case
    when team.status = 'confirmed' then 'active'
    when team.status = 'pending_partner' then 'invited'
    when team.status = 'declined' then 'declined'
    else 'removed'
  end,
  team.partner_contact_email,
  team.partner_invite_token_hash,
  team.partner_invite_expires_at,
  team.partner_confirmed_at,
  team.created_operation_id,
  team.created_at,
  team.updated_at
from public.team_league_teams as team
where team.partner_player_id is not null
  and not exists (
  select 1
    from public.team_league_team_members as member
   where member.team_id = team.id
     and member.player_id = team.partner_player_id
);

alter table public.team_league_team_members enable row level security;
alter table public.team_league_team_members force row level security;
alter table public.team_league_substitute_pool enable row level security;
alter table public.team_league_substitute_pool force row level security;

revoke all on table public.team_league_team_members
  from public, anon, authenticated;
revoke all on table public.team_league_substitute_pool
  from public, anon, authenticated;
grant select, insert, update on table public.team_league_team_members
  to service_role;
grant select, insert, update on table public.team_league_substitute_pool
  to service_role;

-- One validator is shared by settings changes, normalized member actions, and
-- legacy pair projection. It rejects impossible partial rosters and requires
-- exact composition as soon as every primary slot has been assigned.
create or replace function public.team_league_assert_roster_policy_v1(
  p_club_id text,
  p_league_name text,
  p_team_size smallint,
  p_team_category text,
  p_max_alternates smallint,
  p_required_men smallint,
  p_required_women smallint
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if exists (
    select 1
      from public.team_league_teams as team
      join public.team_league_team_members as member
        on member.team_id = team.id
       and member.status in ('invited', 'active')
     where team.club_id = p_club_id
       and team.league_name = p_league_name
     group by team.id
    having count(*) filter (where member.role in ('captain', 'primary')) > p_team_size
        or count(*) filter (where member.role = 'alternate') > p_max_alternates
  ) then
    raise exception using
      errcode = '23514',
      message = 'TEAM_LEAGUE_EXISTING_ROSTER_EXCEEDS_SETTINGS';
  end if;

  if p_team_category = 'open' then
    return;
  end if;
  if p_team_category in ('mens', 'womens') and exists (
    select 1
      from public.team_league_team_members as member
      left join public.players as player
        on player.club_id = member.club_id
       and player.id = member.player_id
     where member.club_id = p_club_id
       and member.league_name = p_league_name
       and member.status in ('invited', 'active')
       and case
         when p_team_category = 'mens' then
           pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
             not in ('m', 'man', 'men', 'male', 'mens', 'men''s')
         else
           pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
             not in ('f', 'w', 'woman', 'women', 'female', 'womens', 'women''s')
       end
  ) then
    raise exception using
      errcode = '23514',
      message = 'TEAM_LEAGUE_EXISTING_ROSTER_CATEGORY_INVALID';
  end if;
  if p_team_category = 'mixed' and exists (
    select 1
      from public.team_league_team_members as member
      left join public.players as player
        on player.club_id = member.club_id
       and player.id = member.player_id
     where member.club_id = p_club_id
       and member.league_name = p_league_name
       and member.status in ('invited', 'active')
       and pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
         not in (
           'm', 'man', 'men', 'male', 'mens', 'men''s',
           'f', 'w', 'woman', 'women', 'female', 'womens', 'women''s'
         )
  ) then
    raise exception using
      errcode = '23514',
      message = 'TEAM_LEAGUE_EXISTING_ROSTER_CATEGORY_INVALID';
  end if;
  if p_team_category = 'mixed' and exists (
    select 1
      from public.team_league_teams as team
      join public.team_league_team_members as member
        on member.team_id = team.id
       and member.status in ('invited', 'active')
       and member.role in ('captain', 'primary')
      left join public.players as player
        on player.club_id = member.club_id
       and player.id = member.player_id
     where team.club_id = p_club_id
       and team.league_name = p_league_name
     group by team.id
    having count(*) filter (
             where pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
               in ('m', 'man', 'men', 'male', 'mens', 'men''s')
           ) > p_required_men
        or count(*) filter (
             where pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
               in ('f', 'w', 'woman', 'women', 'female', 'womens', 'women''s')
           ) > p_required_women
        or count(*) filter (
             where pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
               not in (
                 'm', 'man', 'men', 'male', 'mens', 'men''s',
                 'f', 'w', 'woman', 'women', 'female', 'womens', 'women''s'
               )
           ) > 0
        or (
          count(*) = p_team_size
          and (
            count(*) filter (
              where pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
                in ('m', 'man', 'men', 'male', 'mens', 'men''s')
            ) <> p_required_men
            or count(*) filter (
              where pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
                in ('f', 'w', 'woman', 'women', 'female', 'womens', 'women''s')
            ) <> p_required_women
          )
        )
  ) then
    raise exception using
      errcode = '23514',
      message = 'TEAM_LEAGUE_EXISTING_ROSTER_CATEGORY_INVALID';
  end if;
end
$function$;

-- Fail the migration closed if legacy data cannot satisfy the normalized
-- policy. This runs after deterministic pair backfill and before this migration
-- can commit, so operators get an actionable data-integrity error
-- instead of carrying a latent invalid roster into production.
do $validate_existing_normalized_team_rosters$
declare
  v_settings public.team_league_settings%rowtype;
begin
  for v_settings in
    select settings.*
      from public.team_league_settings as settings
     order by settings.club_id, settings.league_name
  loop
    perform public.team_league_assert_roster_policy_v1(
      v_settings.club_id,
      v_settings.league_name,
      v_settings.team_size,
      v_settings.team_category,
      v_settings.max_alternates,
      v_settings.mixed_required_men,
      v_settings.mixed_required_women
    );
  end loop;
end
$validate_existing_normalized_team_rosters$;

-- The legacy guard treats every pending -> confirmed transition as a public
-- registration. A normalized roster can already be complete before its parent
-- status is reconciled (for example after a team-size change). Permit only
-- that exact status repair when the active normalized primaries satisfy the
-- current policy; legacy partner confirmation still has an invited partner at
-- this BEFORE trigger and therefore remains subject to registration closure.
create or replace function public.team_league_guard_roster_mutation_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := case when tg_op = 'DELETE' then old.club_id else new.club_id end;
  v_league_name text := case when tg_op = 'DELETE' then old.league_name else new.league_name end;
  v_settings public.team_league_settings%rowtype;
  v_becoming_confirmed boolean := (
    tg_op <> 'DELETE'
    and new.status = 'confirmed'
    and (tg_op = 'INSERT' or old.status is distinct from 'confirmed')
  );
  v_new_registration boolean := (
    tg_op = 'INSERT'
    and new.status in ('pending_partner', 'confirmed')
  );
  v_roster_changed boolean := (
    (tg_op = 'INSERT' and new.status = 'confirmed')
    or (tg_op = 'DELETE' and old.status = 'confirmed')
    or (
      tg_op = 'UPDATE'
      and (old.status = 'confirmed' or new.status = 'confirmed')
      and (
        old.status is distinct from new.status
        or old.captain_player_id is distinct from new.captain_player_id
        or old.partner_player_id is distinct from new.partner_player_id
      )
    )
  );
  v_normalized_completion boolean := false;
begin
  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name
   for update;
  if not found then
    raise exception using
      errcode = '23503',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;

  if tg_op = 'UPDATE'
     and old.status = 'pending_partner'
     and new.status = 'confirmed'
     and (
       pg_catalog.to_jsonb(old) - 'status' - 'updated_at'
     ) = (
       pg_catalog.to_jsonb(new) - 'status' - 'updated_at'
     ) then
    select count(*) = v_settings.team_size
      into v_normalized_completion
      from public.team_league_team_members as member
     where member.team_id = new.id
       and member.club_id = new.club_id
       and member.league_name = new.league_name
       and member.status = 'active'
       and member.role in ('captain', 'primary');
    if v_normalized_completion then
      perform public.team_league_assert_roster_policy_v1(
        v_club_id,
        v_league_name,
        v_settings.team_size,
        v_settings.team_category,
        v_settings.max_alternates,
        v_settings.mixed_required_men,
        v_settings.mixed_required_women
      );
    end if;
  end if;

  if (v_becoming_confirmed or v_new_registration)
     and not v_normalized_completion
     and (
       not v_settings.registration_open
       or v_settings.status <> 'registration_open'
       or (
         v_settings.registration_closes_at is not null
         and v_settings.registration_closes_at <= pg_catalog.clock_timestamp()
       )
       or v_settings.schedule_version <> 0
       or exists (
         select 1
           from public.team_league_fixtures as fixture
          where fixture.club_id = v_club_id
            and fixture.league_name = v_league_name
       )
     ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_REGISTRATION_CLOSED';
  end if;

  if v_roster_changed
     and v_settings.schedule_version > 0
     and not (
       tg_op = 'UPDATE'
       and old.status = new.status
       and old.captain_player_id is not distinct from new.captain_player_id
       and old.partner_player_id is not distinct from new.partner_player_id
     ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_ROSTER_LOCKED_AFTER_SCHEDULE';
  end if;
  return case when tg_op = 'DELETE' then old else new end;
end
$function$;

-- Create the normalized parent identity without requiring a legacy fixed pair.
-- The optional first primary is projected into the nullable compatibility
-- column; remaining primaries and alternates use the roster-action RPC.
create or replace function public.team_league_create_team_v1(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_team_name text,
  p_captain_player_id bigint,
  p_captain_contact_email text,
  p_initial_primary_player_id bigint,
  p_initial_primary_contact_email text,
  p_expected_roster_version integer,
  p_idempotency_key text,
  p_request_fingerprint text,
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
  v_team_name text := nullif(pg_catalog.left(pg_catalog.btrim(p_team_name), 120), '');
  v_captain_email text := nullif(
    pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_captain_contact_email)), 320),
    ''
  );
  v_primary_email text := nullif(
    pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_initial_primary_contact_email)), 320),
    ''
  );
  v_key text := nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_fingerprint text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_actor_email text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), 320), ''),
    'unknown'
  );
  v_actor_role text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_role)), 80), ''),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_create_team'
  );
  v_settings public.team_league_settings%rowtype;
  v_operation public.team_league_operations%rowtype;
  v_team public.team_league_teams%rowtype;
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_league_name is null
     or v_team_name is null
     or p_captain_player_id is null
     or v_captain_email is null
     or pg_catalog.char_length(v_captain_email) < 3
     or v_captain_email not like '%@%'
     or (
       p_initial_primary_player_id is not null
       and p_initial_primary_player_id = p_captain_player_id
     )
     or (
       v_primary_email is not null
       and (
         pg_catalog.char_length(v_primary_email) < 3
         or v_primary_email not like '%@%'
       )
     )
     or p_expected_roster_version is null
     or p_expected_roster_version < 0
     or v_key is null
     or v_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$'
     or v_fingerprint !~ '^[0-9a-f]{64}$' then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_CREATE_TEAM_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-roster:' || v_club_id || ':' || v_league_name,
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
    from public.team_league_teams as team
   where team.club_id = v_club_id
     and team.league_name = v_league_name
   order by team.id
   for update;
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
  if v_settings.roster_version <> p_expected_roster_version then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_ROSTER_VERSION_CONFLICT';
  end if;

  insert into public.team_league_operations (
    id, club_id, league_name, idempotency_key, request_fingerprint,
    operation_type, status, request_json, actor_email, actor_role, source
  ) values (
    p_operation_id,
    v_club_id,
    v_league_name,
    v_key,
    v_fingerprint,
    'admin_create_team',
    'started',
    pg_catalog.jsonb_build_object(
      'team_name', v_team_name,
      'captain_player_id', p_captain_player_id,
      'initial_primary_player_id', p_initial_primary_player_id,
      'expected_roster_version', p_expected_roster_version
    ),
    v_actor_email,
    v_actor_role,
    v_source
  );

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
    case
      when v_settings.team_size = 2
        and p_initial_primary_player_id is not null then 'confirmed'
      else 'pending_partner'
    end,
    p_captain_player_id,
    p_initial_primary_player_id,
    v_captain_email,
    v_primary_email,
    p_operation_id,
    case
      when p_initial_primary_player_id is not null
        then pg_catalog.clock_timestamp()
      else null
    end,
    'not_required'
  )
  returning * into v_team;

  if p_initial_primary_player_id is not null then
    update public.team_league_team_members
       set status = 'active',
           invitation_confirmed_at = coalesce(
             invitation_confirmed_at,
             pg_catalog.clock_timestamp()
           ),
           updated_at = pg_catalog.clock_timestamp()
     where team_id = v_team.id
       and player_id = p_initial_primary_player_id
       and status = 'invited';
  end if;
  perform public.team_league_assert_roster_policy_v1(
    v_club_id,
    v_league_name,
    v_settings.team_size,
    v_settings.team_category,
    v_settings.max_alternates,
    v_settings.mixed_required_men,
    v_settings.mixed_required_women
  );
  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name;
  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'team', pg_catalog.to_jsonb(v_team),
    'roster_version', v_settings.roster_version,
    'idempotent', false
  );
  insert into public.admin_activity_log (
    club_id, actor_email, actor_role, action_type, entity_type, entity_id,
    before_json, after_json, note, source_page, flagged_for_review
  ) values (
    v_club_id,
    v_actor_email,
    v_actor_role,
    'team_league_team_created',
    'team_league_team',
    v_team.id::text,
    '{}'::jsonb,
    pg_catalog.to_jsonb(v_team),
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

-- Legacy pair writers may only mark a team confirmed when their projected
-- active roster can satisfy the configured primary size. This makes the old
-- two-player waitlist RPC fail closed for 3/4-player leagues.
create or replace function public.team_league_guard_team_confirmation_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_team_size smallint;
  v_prospective_active integer;
begin
  if new.status <> 'confirmed' then
    return new;
  end if;
  select settings.team_size
    into v_team_size
    from public.team_league_settings as settings
   where settings.club_id = new.club_id
     and settings.league_name = new.league_name
   for update;
  if not found then
    raise exception using
      errcode = '23503',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;
  if tg_op = 'INSERT' then
    v_prospective_active := case
      when new.partner_player_id is null then 1
      else 2
    end;
  else
    select count(distinct member.player_id)
      into v_prospective_active
      from public.team_league_team_members as member
     where member.team_id = new.id
       and member.role in ('captain', 'primary')
       and (
         member.status = 'active'
         or (
           member.status = 'invited'
           and member.player_id in (
             new.captain_player_id,
             new.partner_player_id
           )
         )
       );
  end if;
  if v_prospective_active <> v_team_size then
    raise exception using
      errcode = '23514',
      message = 'TEAM_LEAGUE_CONFIRMED_ROSTER_INCOMPLETE';
  end if;
  return new;
end
$function$;

drop trigger if exists team_league_guard_team_confirmation
  on public.team_league_teams;
create trigger team_league_guard_team_confirmation
before insert or update of status on public.team_league_teams
for each row execute function public.team_league_guard_team_confirmation_v1();

create or replace function public.team_league_validate_member_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_settings public.team_league_settings%rowtype;
  v_team public.team_league_teams%rowtype;
  v_primary_count integer;
  v_alternate_count integer;
  v_gender text;
  v_structural_change boolean := (
    tg_op = 'INSERT'
    or old.team_id is distinct from new.team_id
    or old.club_id is distinct from new.club_id
    or old.league_name is distinct from new.league_name
    or old.player_id is distinct from new.player_id
    or old.role is distinct from new.role
    or old.status is distinct from new.status
  );
begin
  if tg_op = 'UPDATE' and (
    old.team_id is distinct from new.team_id
    or old.club_id is distinct from new.club_id
    or old.league_name is distinct from new.league_name
  ) then
    raise exception using
      errcode = '23514',
      message = 'TEAM_LEAGUE_MEMBER_SCOPE_IMMUTABLE';
  end if;
  select team.*
    into v_team
    from public.team_league_teams as team
   where team.id = new.team_id
     and team.club_id = new.club_id
     and team.league_name = new.league_name
   for update;
  if not found then
    raise exception using
      errcode = '23503',
      message = 'TEAM_LEAGUE_MEMBER_SCOPE_INVALID';
  end if;
  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = new.club_id
     and settings.league_name = new.league_name
   for update;
  if not found then
    raise exception using
      errcode = '23503',
      message = 'TEAM_LEAGUE_MEMBER_SCOPE_INVALID';
  end if;
  if tg_op = 'UPDATE'
     and v_team.status in ('pending_partner', 'confirmed')
     and old.player_id = v_team.captain_player_id
     and old.role = 'captain'
     and old.status in ('invited', 'active')
     and (
       new.player_id is distinct from old.player_id
       or new.role is distinct from 'captain'
       or new.status not in ('invited', 'active')
     ) then
    raise exception using
      errcode = '23514',
      message = 'TEAM_LEAGUE_ACTIVE_CAPTAIN_REQUIRED';
  end if;
  if new.status in ('invited', 'active') and (
    (new.role = 'captain' and new.player_id <> v_team.captain_player_id)
    or (
      new.player_id = v_team.captain_player_id
      and new.role <> 'captain'
    )
  ) then
    raise exception using
      errcode = '23514',
      message = 'TEAM_LEAGUE_CAPTAIN_ROLE_INVALID';
  end if;
  if v_structural_change
     and v_settings.schedule_version > 0
     and (
       new.role in ('captain', 'primary')
       or (
         tg_op = 'UPDATE'
         and old.role in ('captain', 'primary')
       )
     ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_PRIMARY_ROSTER_LOCKED_AFTER_SCHEDULE';
  end if;
  if new.status in ('invited', 'active') then
    select pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
      into v_gender
      from public.players as player
     where player.club_id = new.club_id
       and player.id = new.player_id
       and coalesce(player.active, true)
       and player.inactive_at is null;
    if not found then
      raise exception using
        errcode = '23503',
        message = 'TEAM_LEAGUE_MEMBER_PLAYER_UNAVAILABLE';
    end if;
    if v_settings.team_category = 'mens'
       and v_gender not in ('m', 'man', 'men', 'male', 'mens', 'men''s') then
      raise exception using
        errcode = '23514',
        message = 'TEAM_LEAGUE_MEMBER_CATEGORY_INVALID';
    end if;
    if v_settings.team_category = 'womens'
       and v_gender not in ('f', 'w', 'woman', 'women', 'female', 'womens', 'women''s') then
      raise exception using
        errcode = '23514',
        message = 'TEAM_LEAGUE_MEMBER_CATEGORY_INVALID';
    end if;
    if v_settings.team_category = 'mixed'
       and v_gender not in (
         'm', 'man', 'men', 'male', 'mens', 'men''s',
         'f', 'w', 'woman', 'women', 'female', 'womens', 'women''s'
       ) then
      raise exception using
        errcode = '23514',
        message = 'TEAM_LEAGUE_MEMBER_CATEGORY_INVALID';
    end if;
    if exists (
      select 1
        from public.team_league_substitute_pool as pool
       where pool.club_id = new.club_id
         and pool.league_name = new.league_name
         and pool.player_id = new.player_id
         and pool.status in ('available', 'unavailable')
    ) then
      raise exception using
        errcode = '23505',
        message = 'TEAM_LEAGUE_PLAYER_ALREADY_IN_SUBSTITUTE_POOL';
    end if;
  end if;
  select
    count(*) filter (where member.role in ('captain', 'primary')),
    count(*) filter (where member.role = 'alternate')
    into v_primary_count, v_alternate_count
    from public.team_league_team_members as member
   where member.team_id = new.team_id
     and member.status in ('invited', 'active')
     and (tg_op = 'INSERT' or member.id <> new.id);
  if new.status in ('invited', 'active') and new.role in ('captain', 'primary') then
    v_primary_count := v_primary_count + 1;
  end if;
  if new.status in ('invited', 'active') and new.role = 'alternate' then
    v_alternate_count := v_alternate_count + 1;
  end if;
  if v_primary_count > v_settings.team_size
     or v_alternate_count > v_settings.max_alternates then
    raise exception using
      errcode = '23514',
      message = 'TEAM_LEAGUE_MEMBER_CAPACITY_EXCEEDED';
  end if;
  new.updated_at := pg_catalog.clock_timestamp();
  return new;
end
$function$;

drop trigger if exists team_league_validate_member
  on public.team_league_team_members;
create trigger team_league_validate_member
before insert or update on public.team_league_team_members
for each row execute function public.team_league_validate_member_v1();

create or replace function public.team_league_assert_member_policy_after_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := case when tg_op = 'DELETE' then old.club_id else new.club_id end;
  v_league_name text := case when tg_op = 'DELETE' then old.league_name else new.league_name end;
  v_settings public.team_league_settings%rowtype;
begin
  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name;
  perform public.team_league_assert_roster_policy_v1(
    v_club_id,
    v_league_name,
    v_settings.team_size,
    v_settings.team_category,
    v_settings.max_alternates,
    v_settings.mixed_required_men,
    v_settings.mixed_required_women
  );
  return null;
end
$function$;

drop trigger if exists team_league_assert_member_policy_after
  on public.team_league_team_members;
create trigger team_league_assert_member_policy_after
after insert or update or delete on public.team_league_team_members
for each row execute function public.team_league_assert_member_policy_after_v1();

create or replace function public.team_league_validate_pool_player_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_enabled boolean;
begin
  if new.status in ('available', 'unavailable') then
    select settings.substitute_pool_enabled
      into v_enabled
      from public.team_league_settings as settings
     where settings.club_id = new.club_id
       and settings.league_name = new.league_name
     for update;
    if not found or not v_enabled then
      raise exception using
        errcode = '55000',
        message = 'TEAM_LEAGUE_SUBSTITUTE_POOL_DISABLED';
    end if;
    if not exists (
      select 1
        from public.players as player
       where player.club_id = new.club_id
         and player.id = new.player_id
         and coalesce(player.active, true)
         and player.inactive_at is null
    ) then
      raise exception using
        errcode = '23503',
        message = 'TEAM_LEAGUE_POOL_PLAYER_UNAVAILABLE';
    end if;
    if exists (
      select 1
        from public.team_league_team_members as member
       where member.club_id = new.club_id
         and member.league_name = new.league_name
         and member.player_id = new.player_id
         and member.status in ('invited', 'active')
    ) then
      raise exception using
        errcode = '23505',
        message = 'TEAM_LEAGUE_PLAYER_ALREADY_ASSIGNED_TO_TEAM';
    end if;
  end if;
  new.updated_at := pg_catalog.clock_timestamp();
  return new;
end
$function$;

drop trigger if exists team_league_validate_pool_player
  on public.team_league_substitute_pool;
create trigger team_league_validate_pool_player
before insert or update on public.team_league_substitute_pool
for each row execute function public.team_league_validate_pool_player_v1();

create or replace function public.team_league_validate_settings_rosters_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  perform public.team_league_assert_roster_policy_v1(
    new.club_id,
    new.league_name,
    new.team_size,
    new.team_category,
    new.max_alternates,
    new.mixed_required_men,
    new.mixed_required_women
  );
  return new;
end
$function$;

drop trigger if exists team_league_validate_settings_rosters
  on public.team_league_settings;
create trigger team_league_validate_settings_rosters
before insert or update of
  team_size,
  team_category,
  max_alternates,
  mixed_required_men,
  mixed_required_women
on public.team_league_settings
for each row execute function public.team_league_validate_settings_rosters_v1();

create or replace function public.team_league_bump_normalized_roster_version_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := case when tg_op = 'DELETE' then old.club_id else new.club_id end;
  v_league_name text := case when tg_op = 'DELETE' then old.league_name else new.league_name end;
begin
  update public.team_league_settings
     set roster_version = roster_version + 1,
         updated_at = pg_catalog.clock_timestamp()
   where club_id = v_club_id
     and league_name = v_league_name;
  return null;
end
$function$;

drop trigger if exists team_league_bump_member_roster_version
  on public.team_league_team_members;
create trigger team_league_bump_member_roster_version
after insert or update or delete on public.team_league_team_members
for each row execute function public.team_league_bump_normalized_roster_version_v1();

drop trigger if exists team_league_bump_pool_roster_version
  on public.team_league_substitute_pool;
create trigger team_league_bump_pool_roster_version
after insert or update or delete on public.team_league_substitute_pool
for each row execute function public.team_league_bump_normalized_roster_version_v1();

-- Keep the legacy captain/partner projection synchronized for existing public
-- pair registration and waitlist RPCs. New 3/4-player workflows mutate the
-- normalized table directly; the two legacy identity columns remain stable.
create or replace function public.team_league_project_legacy_pair_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_captain_status text := case
    when new.status in ('pending_partner', 'confirmed') then 'active'
    else 'removed'
  end;
  v_partner_status text := case
    when new.status = 'confirmed' then 'active'
    when new.status = 'declined' then 'declined'
    when new.status = 'withdrawn' then 'removed'
    else 'invited'
  end;
  v_settings public.team_league_settings%rowtype;
  v_identity_or_invite_changed boolean := (
    tg_op = 'INSERT'
    or old.captain_player_id is distinct from new.captain_player_id
    or old.partner_player_id is distinct from new.partner_player_id
    or old.captain_contact_email is distinct from new.captain_contact_email
    or old.partner_contact_email is distinct from new.partner_contact_email
    or old.partner_invite_token_hash is distinct from new.partner_invite_token_hash
    or old.partner_invite_expires_at is distinct from new.partner_invite_expires_at
    or old.partner_confirmed_at is distinct from new.partner_confirmed_at
  );
  v_lifecycle_sync boolean := (
    tg_op = 'INSERT'
    or new.status in ('declined', 'withdrawn')
    or (
      new.status = 'confirmed'
      and new.partner_confirmed_at is not null
      and old.partner_confirmed_at is distinct from new.partner_confirmed_at
    )
  );
begin
  -- Normalized roster/settings reconciliation updates only the parent status.
  -- Never use that status-only transition to resurrect a removed legacy
  -- partner or to displace a normalized replacement primary.
  if not v_identity_or_invite_changed and not v_lifecycle_sync then
    return new;
  end if;
  if tg_op = 'UPDATE'
     and old.captain_player_id is distinct from new.captain_player_id then
    update public.team_league_team_members
       set status = 'removed',
           updated_at = pg_catalog.clock_timestamp()
     where team_id = new.id
       and player_id = old.captain_player_id
       and status in ('invited', 'active');
  end if;
  if tg_op = 'UPDATE'
     and old.partner_player_id is not null
     and old.partner_player_id is distinct from new.partner_player_id then
    update public.team_league_team_members
       set status = 'removed',
           updated_at = pg_catalog.clock_timestamp()
     where team_id = new.id
       and player_id = old.partner_player_id
       and status in ('invited', 'active');
  end if;
  update public.team_league_team_members as member
     set role = 'captain',
         status = v_captain_status,
         contact_email = new.captain_contact_email,
         created_operation_id = coalesce(new.created_operation_id, member.created_operation_id),
         updated_at = pg_catalog.clock_timestamp()
   where member.team_id = new.id
     and member.player_id = new.captain_player_id
     and (
       member.role,
       member.status,
       member.contact_email,
       member.created_operation_id
     ) is distinct from (
       'captain',
       v_captain_status,
       new.captain_contact_email,
       coalesce(new.created_operation_id, member.created_operation_id)
     );
  if not exists (
    select 1
      from public.team_league_team_members as existing
     where existing.team_id = new.id
       and existing.player_id = new.captain_player_id
  ) then
    insert into public.team_league_team_members (
      team_id, club_id, league_name, player_id, role, status,
      contact_email, created_operation_id
    ) values (
      new.id, new.club_id, new.league_name, new.captain_player_id, 'captain',
      v_captain_status, new.captain_contact_email, new.created_operation_id
    );
  end if;

  if new.partner_player_id is not null then
    update public.team_league_team_members as member
     set role = 'primary',
         status = case
           when new.status = 'pending_partner'
             and member.status in ('active', 'removed') then member.status
           else v_partner_status
         end,
         contact_email = new.partner_contact_email,
         invitation_token_hash = new.partner_invite_token_hash,
         invitation_expires_at = new.partner_invite_expires_at,
         invitation_confirmed_at = new.partner_confirmed_at,
         created_operation_id = coalesce(new.created_operation_id, member.created_operation_id),
         updated_at = pg_catalog.clock_timestamp()
   where member.team_id = new.id
     and member.player_id = new.partner_player_id
     and (
       member.role,
       member.status,
       member.contact_email,
       member.invitation_token_hash,
       member.invitation_expires_at,
       member.invitation_confirmed_at,
       member.created_operation_id
     ) is distinct from (
       'primary',
       case
         when new.status = 'pending_partner'
           and member.status in ('active', 'removed') then member.status
         else v_partner_status
       end,
       new.partner_contact_email,
       new.partner_invite_token_hash,
       new.partner_invite_expires_at,
       new.partner_confirmed_at,
       coalesce(new.created_operation_id, member.created_operation_id)
     );
    if not exists (
      select 1
        from public.team_league_team_members as existing
       where existing.team_id = new.id
         and existing.player_id = new.partner_player_id
    ) then
      insert into public.team_league_team_members (
        team_id, club_id, league_name, player_id, role, status, contact_email,
        invitation_token_hash, invitation_expires_at, invitation_confirmed_at,
        created_operation_id
      ) values (
        new.id, new.club_id, new.league_name, new.partner_player_id, 'primary',
        v_partner_status, new.partner_contact_email, new.partner_invite_token_hash,
        new.partner_invite_expires_at, new.partner_confirmed_at,
        new.created_operation_id
      );
    end if;
  end if;

  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = new.club_id
     and settings.league_name = new.league_name;
  perform public.team_league_assert_roster_policy_v1(
    new.club_id,
    new.league_name,
    v_settings.team_size,
    v_settings.team_category,
    v_settings.max_alternates,
    v_settings.mixed_required_men,
    v_settings.mixed_required_women
  );
  return new;
end
$function$;

drop trigger if exists team_league_project_legacy_pair
  on public.team_league_teams;
create trigger team_league_project_legacy_pair
after insert or update of
  status,
  captain_player_id,
  partner_player_id,
  captain_contact_email,
  partner_contact_email,
  partner_invite_token_hash,
  partner_invite_expires_at,
  partner_confirmed_at
on public.team_league_teams
for each row execute function public.team_league_project_legacy_pair_v1();

create or replace function public.team_league_confirmed_roster_fingerprint_v1(
  p_club_id text,
  p_league_name text
)
returns text
language sql
stable
security invoker
set search_path = ''
as $function$
  select pg_catalog.encode(
    extensions.digest(
      coalesce(
        pg_catalog.string_agg(
          team.id::text || ':' || member.player_id::text || ':' || member.role,
          '|' order by team.id::text, member.role, member.player_id
        ),
        ''
      ),
      'sha256'
    ),
    'hex'
  )
  from public.team_league_teams as team
  join public.team_league_settings as settings
    on settings.club_id = team.club_id
   and settings.league_name = team.league_name
  join public.team_league_team_members as member
    on member.team_id = team.id
   and member.club_id = team.club_id
   and member.league_name = team.league_name
   and member.status = 'active'
  where team.club_id = pg_catalog.btrim(p_club_id)
    and team.league_name = pg_catalog.btrim(p_league_name)
    and team.status = 'confirmed'
    and (
      select count(*)
        from public.team_league_team_members as primary_member
       where primary_member.team_id = team.id
         and primary_member.status = 'active'
         and primary_member.role in ('captain', 'primary')
    ) = settings.team_size
$function$;

revoke all on function public.team_league_validate_member_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_validate_pool_player_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_assert_member_policy_after_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_validate_settings_rosters_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_bump_normalized_roster_version_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_assert_roster_policy_v1(
  text, text, smallint, text, smallint, smallint, smallint
) from public, anon, authenticated;
revoke all on function public.team_league_project_legacy_pair_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_guard_team_confirmation_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_confirmed_roster_fingerprint_v1(text, text)
  from public, anon, authenticated;
revoke all on function public.team_league_create_team_v1(
  uuid, text, text, text, bigint, text, bigint, text, integer,
  text, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_validate_member_v1()
  to service_role;
grant execute on function public.team_league_validate_pool_player_v1()
  to service_role;
grant execute on function public.team_league_assert_member_policy_after_v1()
  to service_role;
grant execute on function public.team_league_validate_settings_rosters_v1()
  to service_role;
grant execute on function public.team_league_bump_normalized_roster_version_v1()
  to service_role;
grant execute on function public.team_league_assert_roster_policy_v1(
  text, text, smallint, text, smallint, smallint, smallint
) to service_role;
grant execute on function public.team_league_project_legacy_pair_v1()
  to service_role;
grant execute on function public.team_league_guard_team_confirmation_v1()
  to service_role;
grant execute on function public.team_league_confirmed_roster_fingerprint_v1(text, text)
  to service_role;
grant execute on function public.team_league_create_team_v1(
  uuid, text, text, text, bigint, text, bigint, text, integer,
  text, text, text, text, text
) to service_role;

-- The v2 settings RPC preserves the existing idempotency/version contract and
-- expands only the normalized roster-policy fields. The original v1 remains
-- available for older fixed-pair clients during rollout.
create or replace function public.team_league_save_settings_v2(
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
  v_team_size smallint := coalesce(nullif(p_settings ->> 'team_size', '')::smallint, 2);
  v_max_alternates smallint := coalesce(nullif(p_settings ->> 'max_alternates', '')::smallint, 0);
  v_category text := coalesce(nullif(p_settings ->> 'team_category', ''), 'open');
  v_required_men smallint := coalesce(nullif(p_settings ->> 'mixed_required_men', '')::smallint, 1);
  v_required_women smallint := coalesce(nullif(p_settings ->> 'mixed_required_women', '')::smallint, 1);
  v_allow_substitutes boolean := coalesce((p_settings ->> 'allow_substitutes')::boolean, false);
  v_pool_enabled boolean := coalesce((p_settings ->> 'substitute_pool_enabled')::boolean, false);
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
     or pg_catalog.jsonb_typeof(p_settings) <> 'object'
     or v_team_size not between 2 and 4
     or v_max_alternates not between 0 and 4
     or v_category not in ('open', 'mens', 'womens', 'mixed')
     or (v_category = 'mixed' and (
       v_required_men < 1
       or v_required_women < 1
       or v_required_men + v_required_women <> v_team_size
     ))
     or (v_pool_enabled and not v_allow_substitutes) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SETTINGS_INVALID';
  end if;

  -- Serialize settings with create/member/pool mutations so the deterministic
  -- team-row scan cannot miss a concurrently inserted forming team.
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-roster:' || v_club_id || ':' || v_league_name,
      0
    )
  );
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

  -- Legacy public pair/invitation writers predate the league roster advisory
  -- key. EXCLUSIVE conflicts with both their initial SELECT ... FOR UPDATE
  -- (ROW SHARE) and their later DML, preventing a lock-upgrade deadlock while
  -- this transaction locks/reconciles the complete scoped team set.
  lock table public.team_league_teams in exclusive mode;
  perform 1
    from public.team_league_teams as team
   where team.club_id = v_club_id
     and team.league_name = v_league_name
   order by team.id
   for update;
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
  perform public.team_league_assert_roster_policy_v1(
    v_club_id,
    v_league_name,
    v_team_size,
    v_category,
    v_max_alternates,
    v_required_men,
    v_required_women
  );

  v_next_status := case
    when coalesce((p_settings ->> 'registration_open')::boolean, false)
      then 'registration_open'
    when v_before.status = 'registration_open'
      then 'registration_closed'
    else coalesce(v_before.status, 'draft')
  end;
  insert into public.team_league_operations (
    id, club_id, league_name, idempotency_key, request_fingerprint,
    operation_type, status, request_json, actor_email, actor_role, source
  ) values (
    p_operation_id, v_club_id, v_league_name, v_key, v_fingerprint,
    'admin_save_settings', 'started',
    pg_catalog.jsonb_build_object(
      'settings', p_settings,
      'expected_settings_version', p_expected_settings_version
    ),
    v_actor_email, v_actor_role, v_source
  );

  insert into public.team_league_settings (
    club_id, league_name, status, registration_open,
    team_size, team_category, max_alternates, substitute_pool_enabled,
    mixed_required_men, mixed_required_women, allow_substitutes,
    playoff_format, playoff_team_count, start_date, weekday, start_time,
    timezone, venue, registration_closes_at, settings_version,
    created_by, updated_by, updated_at
  ) values (
    v_club_id,
    v_league_name,
    v_next_status,
    coalesce((p_settings ->> 'registration_open')::boolean, false),
    v_team_size,
    v_category,
    v_max_alternates,
    v_pool_enabled,
    v_required_men,
    v_required_women,
    v_allow_substitutes,
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
        team_size = excluded.team_size,
        team_category = excluded.team_category,
        max_alternates = excluded.max_alternates,
        substitute_pool_enabled = excluded.substitute_pool_enabled,
        mixed_required_men = excluded.mixed_required_men,
        mixed_required_women = excluded.mixed_required_women,
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

  -- A roster-size change may make a formerly complete team partial (or the
  -- reverse). Keep the legacy status aligned with normalized active primaries
  -- so Python previews and the SQL commit fingerprint see the same teams.
  with completion as (
    select
      team.id,
      count(member.id) filter (
        where member.status = 'active'
          and member.role in ('captain', 'primary')
      ) as active_primary_count
    from public.team_league_teams as team
    left join public.team_league_team_members as member
      on member.team_id = team.id
    where team.club_id = v_club_id
      and team.league_name = v_league_name
      and team.status in ('pending_partner', 'confirmed')
    group by team.id
  )
  update public.team_league_teams as team
     set status = case
           when completion.active_primary_count = v_team_size then 'confirmed'
           else 'pending_partner'
         end,
         updated_at = pg_catalog.clock_timestamp()
    from completion
   where team.id = completion.id
     and team.status is distinct from case
           when completion.active_primary_count = v_team_size then 'confirmed'
           else 'pending_partner'
         end;

  select settings.*
    into v_updated
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name;

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
    club_id, actor_email, actor_role, action_type, entity_type, entity_id,
    before_json, after_json, note, source_page, flagged_for_review
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

create or replace function public.team_league_guard_settings_after_schedule_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if old.schedule_version > 0 and (
    (not old.registration_open and new.registration_open)
    or old.start_date is distinct from new.start_date
    or old.weekday is distinct from new.weekday
    or old.start_time is distinct from new.start_time
    or old.timezone is distinct from new.timezone
    or old.venue is distinct from new.venue
    or old.registration_closes_at is distinct from new.registration_closes_at
    or old.team_size is distinct from new.team_size
    or old.team_category is distinct from new.team_category
    or old.max_alternates is distinct from new.max_alternates
    or old.mixed_required_men is distinct from new.mixed_required_men
    or old.mixed_required_women is distinct from new.mixed_required_women
    or old.allow_substitutes is distinct from new.allow_substitutes
    or old.substitute_pool_enabled is distinct from new.substitute_pool_enabled
    or old.playoff_format is distinct from new.playoff_format
    or old.playoff_team_count is distinct from new.playoff_team_count
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_STRUCTURAL_SETTINGS_LOCKED_AFTER_SCHEDULE';
  end if;
  return new;
end
$function$;

-- One CAS/idempotent mutation surface owns assigned-member and pool changes.
create or replace function public.team_league_apply_roster_action_v1(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_action text,
  p_team_id uuid,
  p_player_id bigint,
  p_member_role text,
  p_member_status text,
  p_contact_email text,
  p_note text,
  p_expected_roster_version integer,
  p_idempotency_key text,
  p_request_fingerprint text,
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
  v_action text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_action, '')));
  v_role text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_member_role, '')));
  v_status text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_member_status, '')));
  v_email text := nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_contact_email)), 320), '');
  v_note text := nullif(pg_catalog.left(pg_catalog.btrim(p_note), 500), '');
  v_key text := nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_fingerprint text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_actor_email text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), 320), ''),
    'unknown'
  );
  v_actor_role text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_role)), 80), ''),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_roster'
  );
  v_settings public.team_league_settings%rowtype;
  v_team public.team_league_teams%rowtype;
  v_operation public.team_league_operations%rowtype;
  v_member public.team_league_team_members%rowtype;
  v_pool public.team_league_substitute_pool%rowtype;
  v_primary_count integer;
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_league_name is null
     or v_action not in ('add_member', 'remove_member', 'set_pool')
     or p_player_id is null
     or p_expected_roster_version is null
     or p_expected_roster_version < 0
     or v_key is null
     or v_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$'
     or v_fingerprint !~ '^[0-9a-f]{64}$'
     or (v_action in ('add_member', 'remove_member') and p_team_id is null)
     or (v_action = 'add_member' and (
       v_role not in ('captain', 'primary', 'alternate')
       or v_status not in ('invited', 'active')
     ))
     or (v_action = 'set_pool' and v_status not in ('available', 'unavailable', 'withdrawn')) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_ROSTER_ACTION_INVALID';
  end if;
  if v_email is not null and (pg_catalog.char_length(v_email) < 3 or v_email not like '%@%') then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_ROSTER_EMAIL_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-roster:' || v_club_id || ':' || v_league_name,
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
  if v_action in ('add_member', 'remove_member') then
    select team.*
      into v_team
      from public.team_league_teams as team
     where team.id = p_team_id
       and team.club_id = v_club_id
       and team.league_name = v_league_name
       and team.status in ('pending_partner', 'confirmed')
     for update;
    if not found then
      raise exception using
        errcode = 'P0002',
        message = 'TEAM_LEAGUE_TEAM_NOT_FOUND';
    end if;
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
  if v_settings.roster_version <> p_expected_roster_version then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_ROSTER_VERSION_CONFLICT';
  end if;

  insert into public.team_league_operations (
    id, club_id, league_name, idempotency_key, request_fingerprint,
    operation_type, status, request_json, actor_email, actor_role, source
  ) values (
    p_operation_id,
    v_club_id,
    v_league_name,
    v_key,
    v_fingerprint,
    'admin_roster_' || v_action,
    'started',
    pg_catalog.jsonb_build_object(
      'action', v_action,
      'team_id', p_team_id,
      'player_id', p_player_id,
      'member_role', nullif(v_role, ''),
      'member_status', v_status
    ),
    v_actor_email,
    v_actor_role,
    v_source
  );

  if v_action = 'add_member' then
    select member.*
      into v_member
      from public.team_league_team_members as member
     where member.team_id = p_team_id
       and member.club_id = v_club_id
       and member.league_name = v_league_name
       and member.player_id = p_player_id
     order by member.created_at desc
     limit 1
     for update;
    if found then
      update public.team_league_team_members
         set role = v_role,
             status = v_status,
             contact_email = v_email,
             created_operation_id = p_operation_id,
             updated_at = pg_catalog.clock_timestamp()
       where id = v_member.id
         and club_id = v_club_id
         and league_name = v_league_name
      returning * into v_member;
    else
      insert into public.team_league_team_members (
        team_id, club_id, league_name, player_id, role, status,
        contact_email, created_operation_id
      ) values (
        p_team_id, v_club_id, v_league_name, p_player_id, v_role, v_status,
        v_email, p_operation_id
      )
      returning * into v_member;
    end if;
  elsif v_action = 'remove_member' then
    update public.team_league_team_members
       set status = 'removed',
           created_operation_id = p_operation_id,
           updated_at = pg_catalog.clock_timestamp()
     where team_id = p_team_id
       and club_id = v_club_id
       and league_name = v_league_name
       and player_id = p_player_id
       and status in ('invited', 'active')
       and role <> 'captain'
    returning * into v_member;
    if not found then
      raise exception using
        errcode = 'P0002',
        message = 'TEAM_LEAGUE_REMOVABLE_MEMBER_NOT_FOUND';
    end if;
  else
    select pool.*
      into v_pool
      from public.team_league_substitute_pool as pool
     where pool.club_id = v_club_id
       and pool.league_name = v_league_name
       and pool.player_id = p_player_id
     order by pool.created_at desc
     limit 1
     for update;
    if found then
      update public.team_league_substitute_pool
         set status = v_status,
             contact_email = v_email,
             note = v_note,
             created_operation_id = p_operation_id,
             updated_at = pg_catalog.clock_timestamp()
       where id = v_pool.id
      returning * into v_pool;
    else
      insert into public.team_league_substitute_pool (
        club_id, league_name, player_id, status, contact_email, note,
        created_operation_id
      ) values (
        v_club_id, v_league_name, p_player_id, v_status, v_email, v_note,
        p_operation_id
      )
      returning * into v_pool;
    end if;
  end if;

  if v_action = 'add_member'
     and v_role = 'primary'
     and (
       v_team.partner_player_id is null
       or not exists (
         select 1
           from public.team_league_team_members as legacy_partner
          where legacy_partner.team_id = p_team_id
            and legacy_partner.player_id = v_team.partner_player_id
            and legacy_partner.status in ('invited', 'active')
       )
     ) then
    update public.team_league_teams
       set partner_player_id = p_player_id,
           partner_contact_email = v_email,
           partner_confirmed_at = case
             when v_status = 'active' then pg_catalog.clock_timestamp()
             else null
           end,
           invitation_delivery_status = 'not_required',
           updated_at = pg_catalog.clock_timestamp()
     where id = p_team_id
       and club_id = v_club_id
       and league_name = v_league_name;
  end if;

  if v_action in ('add_member', 'remove_member') then
    perform public.team_league_assert_roster_policy_v1(
      v_club_id,
      v_league_name,
      v_settings.team_size,
      v_settings.team_category,
      v_settings.max_alternates,
      v_settings.mixed_required_men,
      v_settings.mixed_required_women
    );
    select count(*)
      into v_primary_count
      from public.team_league_team_members as member
     where member.team_id = p_team_id
       and member.club_id = v_club_id
       and member.league_name = v_league_name
       and member.role in ('captain', 'primary')
       and member.status = 'active';
    update public.team_league_teams
       set status = case
             when v_primary_count = v_settings.team_size then 'confirmed'
             else 'pending_partner'
           end,
           updated_at = pg_catalog.clock_timestamp()
     where id = p_team_id
       and club_id = v_club_id
       and league_name = v_league_name
       and status in ('pending_partner', 'confirmed');
  end if;

  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name;
  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'action', v_action,
    'team_id', p_team_id,
    'player_id', p_player_id,
    'member', case when v_action in ('add_member', 'remove_member') then pg_catalog.to_jsonb(v_member) else null end,
    'pool_entry', case when v_action = 'set_pool' then pg_catalog.to_jsonb(v_pool) else null end,
    'roster_version', v_settings.roster_version,
    'idempotent', false
  );
  insert into public.admin_activity_log (
    club_id, actor_email, actor_role, action_type, entity_type, entity_id,
    before_json, after_json, note, source_page, flagged_for_review
  ) values (
    v_club_id,
    v_actor_email,
    v_actor_role,
    'team_league_roster_' || v_action,
    'team_league_roster',
    coalesce(p_team_id::text, p_player_id::text),
    '{}'::jsonb,
    v_result - 'idempotent',
    v_note,
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

revoke all on function public.team_league_save_settings_v2(
  uuid, text, text, text, text, integer, jsonb, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_save_settings_v2(
  uuid, text, text, text, text, integer, jsonb, text, text, text
) to service_role;
revoke all on function public.team_league_apply_roster_action_v1(
  uuid, text, text, text, uuid, bigint, text, text, text, text, integer,
  text, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_apply_roster_action_v1(
  uuid, text, text, text, uuid, bigint, text, text, text, text, integer,
  text, text, text, text, text
) to service_role;

-- Replace the reservation-aware finalizer's fixed captain/partner check with
-- normalized roster and opt-in pool membership. The canonical match and the
-- original atomic fixture finalizer remain unchanged.
create or replace function public.team_league_finalize_fixture_v2(
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
  v_status text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_status, '')));
  v_operation public.team_league_operations%rowtype;
  v_fixture public.team_league_fixtures%rowtype;
  v_settings public.team_league_settings%rowtype;
  v_pool_substitution_count integer;
  v_expected_substitutions jsonb := '[]'::jsonb;
  v_result jsonb;
begin
  if v_status not in ('complete', 'forfeit')
     or p_substitutions is null
     or pg_catalog.jsonb_typeof(p_substitutions) <> 'array' then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_FIXTURE_RESULT_INVALID';
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
  if nullif(v_operation.request_json ->> 'fixture_id', '') is distinct from
     p_fixture_id::text then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SCORE_OPERATION_FIXTURE_MISMATCH';
  end if;
  if v_operation.status = 'complete'
     and v_operation.result_json is not null then
    return v_operation.result_json || '{"idempotent": true}'::jsonb;
  end if;

  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_operation.club_id
     and settings.league_name = v_operation.league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;
  select fixture.*
    into v_fixture
    from public.team_league_fixtures as fixture
   where fixture.id = p_fixture_id
     and fixture.club_id = v_operation.club_id
     and fixture.league_name = v_operation.league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_FIXTURE_NOT_FOUND';
  end if;
  if v_fixture.score_operation_id = p_operation_id
     and v_fixture.status in ('complete', 'forfeit') then
    null;
  elsif v_fixture.status <> 'scheduled'
        or v_fixture.score_operation_id is distinct from p_operation_id
        or v_fixture.score_reserved_at is null then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_FIXTURE_SCORE_RESERVATION_CONFLICT';
  else
    if v_status = 'complete' then
      if not exists (
        select 1
          from public.matches as match_row
         where match_row.club_id = v_fixture.club_id
           and match_row.id = p_official_match_id
           and match_row.league = v_fixture.league_name
           and match_row.match_type = 'Team League'
           and match_row.deleted_at is null
           and not coalesce(match_row.excluded_from_ratings, false)
           and match_row.score_t1 = p_team_a_score
           and match_row.score_t2 = p_team_b_score
           and (
             case
               when match_row.score_t1 > match_row.score_t2
                 then v_fixture.team_a_id
               else v_fixture.team_b_id
             end
           ) = p_winner_team_id
           and array[match_row.t1_p1, match_row.t1_p2] @>
               array[p_team_a_player_1_id, p_team_a_player_2_id]
           and array[match_row.t1_p1, match_row.t1_p2] <@
               array[p_team_a_player_1_id, p_team_a_player_2_id]
           and array[match_row.t2_p1, match_row.t2_p2] @>
               array[p_team_b_player_1_id, p_team_b_player_2_id]
           and array[match_row.t2_p1, match_row.t2_p2] <@
               array[p_team_b_player_1_id, p_team_b_player_2_id]
        for share
      ) then
        raise exception using
          errcode = '22023',
          message = 'TEAM_LEAGUE_CANONICAL_MATCH_INVALID';
      end if;

      select coalesce(
               pg_catalog.jsonb_agg(
                 pg_catalog.jsonb_build_object(
                   'incoming_player_id', lineup.player_id,
                   'team_id', lineup.expected_team_id,
                   'source', 'substitute_pool'
                 )
                 order by lineup.expected_team_id::text, lineup.player_id
               ),
               '[]'::jsonb
             )
        into v_expected_substitutions
        from (values
          (p_team_a_player_1_id, v_fixture.team_a_id),
          (p_team_a_player_2_id, v_fixture.team_a_id),
          (p_team_b_player_1_id, v_fixture.team_b_id),
          (p_team_b_player_2_id, v_fixture.team_b_id)
        ) as lineup(player_id, expected_team_id)
       where not exists (
         select 1
           from public.team_league_team_members as member
          where member.player_id = lineup.player_id
            and member.status = 'active'
            and member.team_id = lineup.expected_team_id
       );
      v_pool_substitution_count :=
        pg_catalog.jsonb_array_length(v_expected_substitutions);
      if v_pool_substitution_count > 0 and (
        not v_settings.allow_substitutes
        or not v_settings.substitute_pool_enabled
        or exists (
          select 1
            from (values
              (p_team_a_player_1_id, v_fixture.team_a_id),
              (p_team_a_player_2_id, v_fixture.team_a_id),
              (p_team_b_player_1_id, v_fixture.team_b_id),
              (p_team_b_player_2_id, v_fixture.team_b_id)
            ) as lineup(player_id, expected_team_id)
           where not exists (
             select 1
               from public.team_league_team_members as member
              where member.player_id = lineup.player_id
                and member.status = 'active'
                and member.team_id = lineup.expected_team_id
           )
             and not exists (
               select 1
                 from public.team_league_substitute_pool as pool
                where pool.club_id = v_fixture.club_id
                  and pool.league_name = v_fixture.league_name
                  and pool.player_id = lineup.player_id
                  and pool.status = 'available'
             )
        )
      ) then
        raise exception using
          errcode = '55000',
          message = 'TEAM_LEAGUE_SUBSTITUTE_POOL_INVALID';
      end if;
      if pg_catalog.jsonb_array_length(p_substitutions)
           <> v_pool_substitution_count
         or not (p_substitutions @> v_expected_substitutions)
         or not (p_substitutions <@ v_expected_substitutions) then
        raise exception using
          errcode = '22023',
          message = 'TEAM_LEAGUE_SUBSTITUTION_AUDIT_INVALID';
      end if;

      if v_settings.team_category in ('mens', 'womens', 'mixed') and exists (
        select 1
          from (values
            (p_team_a_player_1_id, 'a'),
            (p_team_a_player_2_id, 'a'),
            (p_team_b_player_1_id, 'b'),
            (p_team_b_player_2_id, 'b')
          ) as lineup(player_id, side)
          left join public.players as player
            on player.id = lineup.player_id
           and player.club_id = v_fixture.club_id
         group by lineup.side
        having count(*) filter (
          where pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
            in ('m', 'man', 'men', 'male', 'mens', 'men''s')
        ) <> case when v_settings.team_category = 'mens' then 2
                  when v_settings.team_category = 'mixed' then 1
                  else 0 end
            or count(*) filter (
          where pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
            in ('f', 'w', 'woman', 'women', 'female', 'womens', 'women''s')
        ) <> case when v_settings.team_category = 'womens' then 2
                  when v_settings.team_category = 'mixed' then 1
                  else 0 end
      ) then
        raise exception using
          errcode = '23514',
          message = 'TEAM_LEAGUE_LINEUP_CATEGORY_INVALID';
      end if;
    end if;

    update public.team_league_fixtures
       set score_operation_id = null
     where id = p_fixture_id;
  end if;

  select public.team_league_finalize_fixture_v1(
    p_operation_id,
    p_club_id,
    p_fixture_id,
    v_status,
    p_team_a_score,
    p_team_b_score,
    p_winner_team_id,
    p_official_match_id,
    p_team_a_player_1_id,
    p_team_a_player_2_id,
    p_team_b_player_1_id,
    p_team_b_player_2_id,
    p_substitutions,
    p_score_note,
    p_actor_email,
    p_actor_role,
    p_source
  ) into v_result;
  return v_result;
end
$function$;

revoke all on function public.team_league_finalize_fixture_v2(
  uuid, text, uuid, text, integer, integer, uuid, bigint,
  bigint, bigint, bigint, bigint, jsonb, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_finalize_fixture_v2(
  uuid, text, uuid, text, integer, integer, uuid, bigint,
  bigint, bigint, bigint, bigint, jsonb, text, text, text, text
) to service_role;

comment on table public.team_league_team_members is
  'Private normalized 2-4 player Team League season rosters and assigned alternates.';
comment on table public.team_league_substitute_pool is
  'Private league-wide pool of opt-in one-off substitutes who are not assigned to a team.';
comment on column public.team_league_settings.team_size is
  'Number of primary season players required to confirm a team; supported values are 2, 3, and 4.';
comment on column public.team_league_settings.max_alternates is
  'Maximum season-long alternates assigned to each team; alternates do not count toward team_size.';

notify pgrst, 'reload schema';
