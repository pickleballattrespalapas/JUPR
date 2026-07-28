-- Combined-rating divisions and four-player team tournaments.
-- Forward-only, service-role-only operational storage. Public actions must pass
-- through token-gated FastAPI services and the staging public-intake write gate.

do $$
begin
  if to_regclass('public.tournaments') is null
     or to_regclass('public.tournament_event_options') is null
     or to_regclass('public.tournament_registrations') is null
     or to_regclass('public.tournament_registration_selections') is null
     or to_regclass('public.tournament_event_draws') is null
     or to_regclass('public.tournament_teams') is null
     or to_regclass('public.tournament_games') is null
     or to_regclass('public.players') is null then
    raise exception 'Combined-rating/team tournament base tables are missing';
  end if;
end
$$;

alter table public.tournament_event_options
  add column if not exists eligibility_mode text not null default 'STANDARD',
  add column if not exists combined_rating_cap numeric(5,2) null,
  add column if not exists rating_source_policy text not null default 'PCS_OR_ORGANIZER_VERIFIED',
  add column if not exists rating_review_timing text not null default 'INITIAL_AND_REGISTRATION_CLOSE',
  add column if not exists competition_format text not null default 'STANDARD',
  add column if not exists team_roster_size smallint not null default 2,
  add column if not exists team_gender_rule text not null default 'NONE',
  add column if not exists team_tiebreak_mode text not null default 'SINGLES',
  add column if not exists team_playoff_format text not null default 'NONE',
  add column if not exists team_allow_substitutes boolean not null default false,
  add column if not exists updated_at timestamptz not null default timezone('utc', now());

do $$
begin
  if not exists (
    select 1 from pg_constraint
     where conrelid = 'public.tournament_event_options'::regclass
       and conname = 'tournament_event_options_eligibility_mode_chk'
  ) then
    alter table public.tournament_event_options
      add constraint tournament_event_options_eligibility_mode_chk
      check (eligibility_mode in ('STANDARD', 'COMBINED_RATING_CAP')) not valid;
  end if;
  if not exists (
    select 1 from pg_constraint
     where conrelid = 'public.tournament_event_options'::regclass
       and conname = 'tournament_event_options_combined_cap_chk'
  ) then
    alter table public.tournament_event_options
      add constraint tournament_event_options_combined_cap_chk
      check (
        (eligibility_mode <> 'COMBINED_RATING_CAP' and combined_rating_cap is null)
        or
        (eligibility_mode = 'COMBINED_RATING_CAP'
          and combined_rating_cap > 0 and combined_rating_cap <= 14)
      ) not valid;
  end if;
  if not exists (
    select 1 from pg_constraint
     where conrelid = 'public.tournament_event_options'::regclass
       and conname = 'tournament_event_options_competition_format_chk'
  ) then
    alter table public.tournament_event_options
      add constraint tournament_event_options_competition_format_chk
      check (competition_format in ('STANDARD', 'FOUR_PLAYER_TEAM')) not valid;
  end if;
  if not exists (
    select 1 from pg_constraint
     where conrelid = 'public.tournament_event_options'::regclass
       and conname = 'tournament_event_options_team_contract_chk'
  ) then
    alter table public.tournament_event_options
      add constraint tournament_event_options_team_contract_chk
      check (
        (competition_format = 'STANDARD'
          and team_roster_size = 2 and team_gender_rule = 'NONE'
          and team_playoff_format = 'NONE')
        or
        (competition_format = 'FOUR_PLAYER_TEAM'
          and eligibility_mode = 'STANDARD'
          and team_roster_size = 4
          and team_gender_rule = 'TWO_MEN_TWO_WOMEN'
          and team_tiebreak_mode in ('SINGLES', 'SKINNY_RELAY')
          and team_playoff_format in (
            'NONE', 'TOP_2_FINAL', 'TOP_4_SEMIFINALS',
            'TOP_4_SEMIFINALS_WITH_BRONZE'
          ))
      ) not valid;
  end if;
end
$$;

alter table public.tournament_event_draws
  add column if not exists draw_kind text not null default 'STANDARD',
  add column if not exists parent_draw_id uuid null references public.tournament_event_draws(id) on delete cascade,
  add column if not exists hidden_from_primary_ops boolean not null default false;

create table if not exists public.tournament_four_player_teams (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text not null references public.tournament_event_options(id) on delete cascade,
  draw_id uuid null references public.tournament_event_draws(id) on delete set null,
  name text not null,
  captain_registration_id text not null references public.tournament_registrations(id) on delete restrict,
  captain_player_id integer null references public.players(id) on delete set null,
  status text not null default 'FORMING',
  eligibility_state text not null default 'REVIEW_REQUIRED',
  creation_fingerprint text null,
  version integer not null default 1,
  created_by text null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint tournament_four_player_teams_status_chk
    check (status in (
      'FORMING', 'CONFIRMED', 'WAITLIST', 'REVIEW_REQUIRED',
      'INELIGIBLE', 'WITHDRAWN', 'CANCELLED'
    )),
  constraint tournament_four_player_teams_eligibility_chk
    check (eligibility_state in (
      'ELIGIBLE', 'INELIGIBLE', 'REVIEW_REQUIRED',
      'PROVISIONAL_NEEDS_PARTNER', 'NOT_REQUIRED'
    ))
);

alter table public.tournament_four_player_teams
  add column if not exists creation_fingerprint text null;

create unique index if not exists uq_tournament_four_player_team_name
  on public.tournament_four_player_teams (
    tournament_id, event_option_id, lower(btrim(name))
  )
  where status not in ('WITHDRAWN', 'CANCELLED');
create unique index if not exists uq_tournament_four_player_team_captain
  on public.tournament_four_player_teams (
    tournament_id, event_option_id, captain_registration_id
  )
  where status not in ('WITHDRAWN', 'CANCELLED');
create index if not exists idx_tournament_four_player_teams_draw
  on public.tournament_four_player_teams (draw_id, status);

create table if not exists public.tournament_four_player_team_members (
  id uuid primary key default gen_random_uuid(),
  team_id uuid not null references public.tournament_four_player_teams(id) on delete cascade,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text not null references public.tournament_event_options(id) on delete cascade,
  slot text not null,
  invited_email text not null,
  registration_id text null references public.tournament_registrations(id) on delete set null,
  player_id integer null references public.players(id) on delete set null,
  display_name_snapshot text null,
  gender_snapshot text null,
  status text not null default 'INVITED',
  invitation_version integer not null default 1,
  invitation_token_hash text null,
  invited_at timestamptz not null default timezone('utc', now()),
  accepted_at timestamptz null,
  declined_at timestamptz null,
  updated_at timestamptz not null default timezone('utc', now()),
  constraint tournament_four_player_team_members_slot_chk
    check (slot in ('MAN_1', 'MAN_2', 'WOMAN_1', 'WOMAN_2')),
  constraint tournament_four_player_team_members_status_chk
    check (status in ('INVITED', 'ACCEPTED', 'DECLINED', 'REMOVED'))
);

create unique index if not exists uq_tournament_four_player_team_member_slot
  on public.tournament_four_player_team_members (team_id, slot)
  where status <> 'REMOVED';
create unique index if not exists uq_tournament_four_player_team_member_player
  on public.tournament_four_player_team_members (team_id, player_id)
  where player_id is not null and status in ('INVITED', 'ACCEPTED');
create unique index if not exists uq_tournament_four_player_team_member_email
  on public.tournament_four_player_team_members (team_id, lower(invited_email))
  where status in ('INVITED', 'ACCEPTED');
create unique index if not exists uq_tournament_four_player_event_player
  on public.tournament_four_player_team_members (
    tournament_id, event_option_id, player_id
  )
  where player_id is not null and status = 'ACCEPTED';

create table if not exists public.tournament_rating_verifications (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text null references public.tournament_event_options(id) on delete cascade,
  registration_id text not null references public.tournament_registrations(id) on delete cascade,
  player_id integer null references public.players(id) on delete set null,
  rating numeric(4,2) not null check (rating >= 0 and rating <= 7),
  source text not null default 'ORGANIZER_VERIFIED',
  status text not null default 'ACTIVE',
  note text null,
  version integer not null default 1,
  verified_by text not null,
  verified_at timestamptz not null default timezone('utc', now()),
  revoked_at timestamptz null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint tournament_rating_verifications_source_chk
    check (source = 'ORGANIZER_VERIFIED'),
  constraint tournament_rating_verifications_status_chk
    check (status in ('ACTIVE', 'REVOKED'))
);

create unique index if not exists uq_tournament_rating_verification_active
  on public.tournament_rating_verifications (
    tournament_id, coalesce(event_option_id, ''), registration_id
  )
  where status = 'ACTIVE';

create table if not exists public.tournament_rating_eligibility_reviews (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text not null references public.tournament_event_options(id) on delete cascade,
  selection_id text null references public.tournament_registration_selections(id) on delete cascade,
  team_id uuid null references public.tournament_four_player_teams(id) on delete cascade,
  registration_id text not null references public.tournament_registrations(id) on delete cascade,
  partner_registration_id text null references public.tournament_registrations(id) on delete set null,
  player_id_snapshot integer null references public.players(id) on delete restrict,
  partner_player_id_snapshot integer null references public.players(id) on delete restrict,
  review_phase text not null,
  state text not null,
  player_rating numeric(4,2) null,
  partner_rating numeric(4,2) null,
  combined_rating numeric(5,2) null,
  combined_rating_cap numeric(5,2) not null,
  player_rating_source text not null,
  partner_rating_source text not null,
  player_verification_id uuid null references public.tournament_rating_verifications(id) on delete set null,
  partner_verification_id uuid null references public.tournament_rating_verifications(id) on delete set null,
  rating_as_of timestamptz not null default timezone('utc', now()),
  finalized_at timestamptz null,
  override_state text null,
  override_reason text null,
  reviewed_by text null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint tournament_rating_eligibility_phase_chk
    check (review_phase in ('INITIAL', 'REGISTRATION_CLOSE', 'ADMIN_REVIEW')),
  constraint tournament_rating_eligibility_state_chk
    check (state in (
      'ELIGIBLE', 'INELIGIBLE', 'REVIEW_REQUIRED', 'PROVISIONAL_NEEDS_PARTNER'
    )),
  constraint tournament_rating_eligibility_source_chk
    check (
      player_rating_source in ('PCS_LINKED', 'ORGANIZER_VERIFIED', 'MISSING')
      and partner_rating_source in ('PCS_LINKED', 'ORGANIZER_VERIFIED', 'MISSING')
    ),
  constraint tournament_rating_eligibility_math_chk
    check (
      (combined_rating is null)
      or
      (player_rating is not null and partner_rating is not null
        and combined_rating = player_rating + partner_rating)
    ),
  constraint tournament_rating_eligibility_strict_state_chk
    check (
      (state = 'ELIGIBLE' and combined_rating < combined_rating_cap)
      or
      (state = 'INELIGIBLE' and combined_rating >= combined_rating_cap)
      or
      (state in ('REVIEW_REQUIRED', 'PROVISIONAL_NEEDS_PARTNER')
        and combined_rating is null)
  )
);

alter table public.tournament_rating_eligibility_reviews
  add column if not exists player_id_snapshot integer null
    references public.players(id) on delete restrict,
  add column if not exists partner_player_id_snapshot integer null
    references public.players(id) on delete restrict;

create unique index if not exists uq_tournament_rating_review_selection_phase
  on public.tournament_rating_eligibility_reviews (
    event_option_id, selection_id, review_phase
  )
  where selection_id is not null;
create unique index if not exists uq_tournament_rating_review_team_phase
  on public.tournament_rating_eligibility_reviews (
    event_option_id, team_id, registration_id, review_phase
  )
  where team_id is not null;

create or replace function public.guard_finalized_combined_rating_review_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  if old.review_phase = 'REGISTRATION_CLOSE'
     and old.finalized_at is not null then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_RATING_REVIEW_FINALIZED_IMMUTABLE';
  end if;
  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

drop trigger if exists guard_finalized_combined_rating_review
  on public.tournament_rating_eligibility_reviews;
create trigger guard_finalized_combined_rating_review
before update or delete on public.tournament_rating_eligibility_reviews
for each row execute function
  public.guard_finalized_combined_rating_review_v1();

create or replace function public.guard_finalized_combined_rating_relationship_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_selection_ids text[];
begin
  if tg_table_name = 'tournament_registration_selections' then
    v_selection_ids := array[old.id::text];
  elsif tg_table_name = 'tournament_registration_team_members' then
    if tg_op = 'INSERT' then
      v_selection_ids := array[new.selection_id::text];
    elsif tg_op = 'DELETE' then
      v_selection_ids := array[old.selection_id::text];
    else
      v_selection_ids := array[
        old.selection_id::text, new.selection_id::text
      ];
    end if;
  elsif tg_table_name = 'tournament_registration_team_links' then
    if tg_op = 'INSERT' then
      v_selection_ids := array[
        new.selection1_id::text, new.selection2_id::text
      ];
    elsif tg_op = 'DELETE' then
      v_selection_ids := array[
        old.selection1_id::text, old.selection2_id::text
      ];
    else
      v_selection_ids := array[
        old.selection1_id::text, old.selection2_id::text,
        new.selection1_id::text, new.selection2_id::text
      ];
    end if;
  else
    raise exception using errcode = '0A000',
      message = 'JUPR_TOURNAMENT_RATING_RELATIONSHIP_GUARD_INVALID';
  end if;

  if exists (
    select 1
      from public.tournament_rating_eligibility_reviews review
     where review.selection_id::text = any(
       pg_catalog.array_remove(v_selection_ids, null)
     )
       and review.review_phase = 'REGISTRATION_CLOSE'
       and review.finalized_at is not null
  ) then
    raise exception using errcode = 'P0001',
      message =
        'JUPR_TOURNAMENT_RATING_FINALIZED_RELATIONSHIP_IMMUTABLE';
  end if;

  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

drop trigger if exists guard_finalized_combined_rating_selection
  on public.tournament_registration_selections;
create trigger guard_finalized_combined_rating_selection
before update or delete on public.tournament_registration_selections
for each row execute function
  public.guard_finalized_combined_rating_relationship_v1();

drop trigger if exists guard_finalized_combined_rating_team_member
  on public.tournament_registration_team_members;
create trigger guard_finalized_combined_rating_team_member
before insert or update or delete
on public.tournament_registration_team_members
for each row execute function
  public.guard_finalized_combined_rating_relationship_v1();

drop trigger if exists guard_finalized_combined_rating_team_link
  on public.tournament_registration_team_links;
create trigger guard_finalized_combined_rating_team_link
before insert or update or delete
on public.tournament_registration_team_links
for each row execute function
  public.guard_finalized_combined_rating_relationship_v1();

create or replace function public.guard_finalized_combined_rating_registration_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_identity_changed boolean;
begin
  if tg_op = 'DELETE' then
    v_identity_changed := true;
  else
    v_identity_changed := new.player_id is distinct from old.player_id;
  end if;
  if v_identity_changed and exists (
       select 1
         from public.tournament_rating_eligibility_reviews review
        where (
          review.registration_id::text = old.id::text
          or review.partner_registration_id::text = old.id::text
        )
          and review.review_phase = 'REGISTRATION_CLOSE'
          and review.finalized_at is not null
     ) then
    raise exception using errcode = 'P0001',
      message =
        'JUPR_TOURNAMENT_RATING_FINALIZED_PLAYER_IDENTITY_IMMUTABLE';
  end if;
  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

drop trigger if exists guard_finalized_combined_rating_registration
  on public.tournament_registrations;
create trigger guard_finalized_combined_rating_registration
before update of player_id or delete on public.tournament_registrations
for each row execute function
  public.guard_finalized_combined_rating_registration_v1();

revoke all on function
  public.guard_finalized_combined_rating_review_v1()
  from public, anon, authenticated;
revoke all on function
  public.guard_finalized_combined_rating_relationship_v1()
  from public, anon, authenticated;
revoke all on function
  public.guard_finalized_combined_rating_registration_v1()
  from public, anon, authenticated;
grant execute on function
  public.guard_finalized_combined_rating_review_v1()
  to service_role;
grant execute on function
  public.guard_finalized_combined_rating_relationship_v1()
  to service_role;
grant execute on function
  public.guard_finalized_combined_rating_registration_v1()
  to service_role;

create table if not exists public.tournament_team_matchups (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text not null references public.tournament_event_options(id) on delete cascade,
  draw_id uuid not null references public.tournament_event_draws(id) on delete cascade,
  stage text not null default 'ROUND_ROBIN',
  round_number integer null,
  slot_number integer null,
  playoff_game_code text null,
  team_a_id uuid null references public.tournament_four_player_teams(id) on delete restrict,
  team_b_id uuid null references public.tournament_four_player_teams(id) on delete restrict,
  team_a_source jsonb null,
  team_b_source jsonb null,
  tiebreak_mode text not null,
  status text not null default 'LINEUPS_PENDING',
  team_a_game_wins integer not null default 0,
  team_b_game_wins integer not null default 0,
  winner_team_id uuid null references public.tournament_four_player_teams(id) on delete restrict,
  loser_team_id uuid null references public.tournament_four_player_teams(id) on delete restrict,
  finalized_at timestamptz null,
  version integer not null default 1,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint tournament_team_matchups_distinct_teams_chk
    check (
      (team_a_id is not null and team_b_id is not null and team_a_id <> team_b_id)
      or
      (stage = 'PLAYOFF' and team_a_id is null and team_b_id is null
        and team_a_source is not null and team_b_source is not null)
    ),
  constraint tournament_team_matchups_stage_chk
    check (stage in ('ROUND_ROBIN', 'PLAYOFF')),
  constraint tournament_team_matchups_tiebreak_chk
    check (tiebreak_mode in ('SINGLES', 'SKINNY_RELAY')),
  constraint tournament_team_matchups_status_chk
    check (status in (
      'LINEUPS_PENDING', 'READY', 'IN_PROGRESS', 'TIEBREAK_REQUIRED',
      'FINAL', 'CORRECTION_REQUIRED', 'VOID'
    ))
);

create unique index if not exists uq_tournament_team_matchups_rr
  on public.tournament_team_matchups (draw_id, round_number, slot_number)
  where stage = 'ROUND_ROBIN';
create unique index if not exists uq_tournament_team_matchups_playoff
  on public.tournament_team_matchups (draw_id, playoff_game_code)
  where stage = 'PLAYOFF' and playoff_game_code is not null;

create table if not exists public.tournament_team_lineup_submissions (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  matchup_id uuid not null references public.tournament_team_matchups(id) on delete cascade,
  team_id uuid not null references public.tournament_four_player_teams(id) on delete cascade,
  mixed_pairing text not null,
  singles_tiebreak_player_id integer null references public.players(id) on delete set null,
  status text not null default 'LOCKED',
  version integer not null default 1,
  locked_by text not null,
  locked_at timestamptz not null default timezone('utc', now()),
  revealed_at timestamptz null,
  updated_at timestamptz not null default timezone('utc', now()),
  constraint tournament_team_lineups_pairing_chk
    check (mixed_pairing in ('STRAIGHT', 'CROSS')),
  constraint tournament_team_lineups_status_chk
    check (status in ('LOCKED', 'REVEALED'))
);

create unique index if not exists uq_tournament_team_lineup_match_team
  on public.tournament_team_lineup_submissions (matchup_id, team_id);

create table if not exists public.tournament_team_match_games (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  matchup_id uuid not null references public.tournament_team_matchups(id) on delete cascade,
  game_code text not null,
  game_order smallint not null,
  match_format text not null,
  counts_for_rating boolean not null,
  team_a_player_ids integer[] not null,
  team_b_player_ids integer[] not null,
  rating_draw_id uuid null references public.tournament_event_draws(id) on delete set null,
  tournament_game_id uuid null,
  score_a integer null,
  score_b integer null,
  winner_team_id uuid null references public.tournament_four_player_teams(id) on delete restrict,
  loser_team_id uuid null references public.tournament_four_player_teams(id) on delete restrict,
  status text not null default 'SCHEDULED',
  finalized_at timestamptz null,
  version integer not null default 1,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint tournament_team_match_games_code_chk
    check (game_code in ('WOMENS', 'MENS', 'MIXED_1', 'MIXED_2', 'TIEBREAK')),
  constraint tournament_team_match_games_format_chk
    check (match_format in ('DOUBLES', 'SINGLES', 'SKINNY_RELAY')),
  constraint tournament_team_match_games_rating_chk
    check (
      (match_format in ('DOUBLES', 'SINGLES') and counts_for_rating)
      or
      (match_format = 'SKINNY_RELAY' and not counts_for_rating)
    ),
  constraint tournament_team_match_games_score_chk
    check (
      (score_a is null and score_b is null)
      or
      (score_a is not null and score_b is not null
        and score_a >= 0 and score_b >= 0 and score_a <> score_b)
    ),
  constraint tournament_team_match_games_status_chk
    check (status in ('SCHEDULED', 'FINAL', 'VOID'))
);

create unique index if not exists uq_tournament_team_match_game_code
  on public.tournament_team_match_games (matchup_id, game_code);
create unique index if not exists uq_tournament_team_match_game_rating_game
  on public.tournament_team_match_games (tournament_game_id)
  where tournament_game_id is not null;

alter table public.tournament_team_match_games
  drop constraint if exists tournament_team_match_games_tournament_game_fk;
alter table public.tournament_team_match_games
  add constraint tournament_team_match_games_tournament_game_fk
  foreign key (tournament_game_id) references public.tournament_games(id) on delete set null
  not valid;

alter table public.tournament_teams
  add column if not exists team_match_game_id uuid null
    references public.tournament_team_match_games(id) on delete cascade,
  add column if not exists team_match_side text null,
  add column if not exists source_selection_id text null
    references public.tournament_registration_selections(id) on delete restrict,
  add column if not exists eligibility_review_id uuid null
    references public.tournament_rating_eligibility_reviews(id) on delete restrict,
  add column if not exists eligibility_reviewed_at timestamptz null,
  add column if not exists eligibility_state_snapshot text null,
  add column if not exists combined_rating_snapshot numeric(5,2) null,
  add column if not exists combined_rating_cap_snapshot numeric(5,2) null;
create unique index if not exists uq_tournament_team_rating_lineup_side
  on public.tournament_teams (team_match_game_id, team_match_side)
  where team_match_game_id is not null;
create index if not exists idx_tournament_teams_source_selection
  on public.tournament_teams (source_selection_id);
create index if not exists idx_tournament_teams_eligibility_review
  on public.tournament_teams (eligibility_review_id);

alter table public.tournament_games
  add column if not exists team_match_game_id uuid null
    references public.tournament_team_match_games(id) on delete set null,
  add column if not exists parent_result_only boolean not null default false;
create unique index if not exists uq_tournament_game_team_match_game
  on public.tournament_games (team_match_game_id)
  where team_match_game_id is not null;

create table if not exists public.tournament_four_player_podium (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  draw_id uuid not null references public.tournament_event_draws(id) on delete cascade,
  placement smallint not null check (placement between 1 and 3),
  team_id uuid not null references public.tournament_four_player_teams(id) on delete restrict,
  source text not null,
  published_at timestamptz null,
  published_by text null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);
create unique index if not exists uq_tournament_four_player_podium_placement
  on public.tournament_four_player_podium (draw_id, placement);
create unique index if not exists uq_tournament_four_player_podium_team
  on public.tournament_four_player_podium (draw_id, team_id);

create table if not exists public.tournament_team_audit_events (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text null,
  entity_type text not null,
  entity_id text not null,
  action text not null,
  actor text not null,
  before_json jsonb null,
  after_json jsonb null,
  request_fingerprint text null,
  created_at timestamptz not null default timezone('utc', now())
);
create index if not exists idx_tournament_team_audit_scope
  on public.tournament_team_audit_events (tournament_id, created_at desc);

create or replace function public.normalize_tournament_combined_rating(
  p_value text
)
returns numeric
language plpgsql
immutable
security invoker
set search_path = ''
as $$
declare
  v_rating numeric;
begin
  if nullif(btrim(coalesce(p_value, '')), '') is null
     or btrim(p_value) !~ '^[0-9]+([.][0-9]+)?$' then
    return null;
  end if;
  v_rating := btrim(p_value)::numeric;
  if v_rating > 10 then
    v_rating := v_rating / 400;
  end if;
  if v_rating < 0 or v_rating > 7 then
    return null;
  end if;
  return round(v_rating, 2);
end;
$$;

create or replace function public.refresh_initial_combined_rating_review_v1(
  p_selection_id text,
  p_actor text
)
returns void
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_selection public.tournament_registration_selections%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_registration public.tournament_registrations%rowtype;
  v_partner_registration public.tournament_registrations%rowtype;
  v_player public.players%rowtype;
  v_partner public.players%rowtype;
  v_player_verification public.tournament_rating_verifications%rowtype;
  v_partner_verification public.tournament_rating_verifications%rowtype;
  v_saved public.tournament_rating_eligibility_reviews%rowtype;
  v_before jsonb;
  v_club_id text;
  v_player_rating numeric(4,2);
  v_partner_rating numeric(4,2);
  v_combined numeric(5,2);
  v_player_source text := 'MISSING';
  v_partner_source text := 'MISSING';
  v_state text;
begin
  select selection.* into v_selection
    from public.tournament_registration_selections selection
   where selection.id::text = p_selection_id
   for share;
  if not found then
    return;
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = v_selection.event_option_id::text
     and event.tournament_id = v_selection.tournament_id
   for share;
  if not found or v_event.eligibility_mode <> 'COMBINED_RATING_CAP'
     or v_event.combined_rating_cap is null then
    return;
  end if;
  select registration.* into v_registration
    from public.tournament_registrations registration
   where registration.id::text = v_selection.registration_id::text
     and registration.tournament_id = v_selection.tournament_id
   for share;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_INITIAL_RATING_REGISTRATION_NOT_FOUND';
  end if;
  select tournament.club_id::text into v_club_id
    from public.tournaments tournament
   where tournament.id = v_selection.tournament_id;

  select partner_registration.* into v_partner_registration
    from public.tournament_registration_team_members selected_member
    join public.tournament_registration_team_members partner_member
      on partner_member.team_link_id = selected_member.team_link_id
     and partner_member.selection_id::text <> selected_member.selection_id::text
     and upper(coalesce(partner_member.status, '')) = 'ACTIVE'
    join public.tournament_registrations partner_registration
      on partner_registration.id::text = partner_member.registration_id::text
     and partner_registration.tournament_id = v_selection.tournament_id
   where selected_member.selection_id::text = v_selection.id::text
     and selected_member.event_option_id::text = v_selection.event_option_id::text
     and upper(coalesce(selected_member.status, '')) = 'ACTIVE'
   order by partner_member.player_order, partner_member.id
   limit 1
     for share of partner_registration;
  if not found and nullif(lower(btrim(v_selection.partner_email)), '') is not null then
    select registration.* into v_partner_registration
      from public.tournament_registrations registration
     where registration.tournament_id = v_selection.tournament_id
       and lower(btrim(registration.email)) = lower(btrim(v_selection.partner_email))
       and upper(coalesce(registration.status, '')) not in ('CANCELLED', 'WITHDRAWN')
     order by registration.created_at, registration.id
     limit 1
       for share of registration;
  end if;

  if v_registration.player_id is not null then
    select player.* into v_player
      from public.players player
     where player.id = v_registration.player_id
       for share of player;
    v_player_rating := public.normalize_tournament_combined_rating(
      coalesce(
        to_jsonb(v_player)->>'doubles_rating',
        to_jsonb(v_player)->>'doubles_skill',
        to_jsonb(v_player)->>'rating'
      )
    );
    if v_player_rating is not null then
      v_player_source := 'PCS_LINKED';
    end if;
  end if;
  if v_player_rating is null then
    select verification.* into v_player_verification
      from public.tournament_rating_verifications verification
     where verification.tournament_id = v_selection.tournament_id
       and verification.event_option_id::text = v_selection.event_option_id::text
       and verification.registration_id::text = v_registration.id::text
       and verification.status = 'ACTIVE'
     order by verification.verified_at desc, verification.id
     limit 1
       for share of verification;
    if found then
      v_player_rating := v_player_verification.rating;
      v_player_source := 'ORGANIZER_VERIFIED';
    end if;
  end if;

  if v_partner_registration.id is not null
     and v_partner_registration.player_id is not null then
    select player.* into v_partner
      from public.players player
     where player.id = v_partner_registration.player_id
       for share of player;
    v_partner_rating := public.normalize_tournament_combined_rating(
      coalesce(
        to_jsonb(v_partner)->>'doubles_rating',
        to_jsonb(v_partner)->>'doubles_skill',
        to_jsonb(v_partner)->>'rating'
      )
    );
    if v_partner_rating is not null then
      v_partner_source := 'PCS_LINKED';
    end if;
  end if;
  if v_partner_rating is null and v_partner_registration.id is not null then
    select verification.* into v_partner_verification
      from public.tournament_rating_verifications verification
     where verification.tournament_id = v_selection.tournament_id
       and verification.event_option_id::text = v_selection.event_option_id::text
       and verification.registration_id::text = v_partner_registration.id::text
       and verification.status = 'ACTIVE'
     order by verification.verified_at desc, verification.id
     limit 1
       for share of verification;
    if found then
      v_partner_rating := v_partner_verification.rating;
      v_partner_source := 'ORGANIZER_VERIFIED';
    end if;
  end if;

  if upper(coalesce(v_selection.partner_mode, '')) = 'NEEDS_PARTNER' then
    v_state := 'PROVISIONAL_NEEDS_PARTNER';
    v_combined := null;
  elsif v_player_rating is null or v_partner_rating is null then
    v_state := 'REVIEW_REQUIRED';
    v_combined := null;
  else
    v_combined := round(v_player_rating + v_partner_rating, 2);
    v_state := case when v_combined < v_event.combined_rating_cap
      then 'ELIGIBLE' else 'INELIGIBLE' end;
  end if;

  select to_jsonb(review) into v_before
    from public.tournament_rating_eligibility_reviews review
   where review.event_option_id::text = v_selection.event_option_id::text
     and review.selection_id::text = v_selection.id::text
     and review.review_phase = 'INITIAL';
  insert into public.tournament_rating_eligibility_reviews (
    tournament_id, event_option_id, selection_id, registration_id,
    partner_registration_id, player_id_snapshot,
    partner_player_id_snapshot, review_phase, state, player_rating,
    partner_rating, combined_rating, combined_rating_cap,
    player_rating_source, partner_rating_source,
    player_verification_id, partner_verification_id, rating_as_of,
    finalized_at, override_state, override_reason, reviewed_by
  ) values (
    v_selection.tournament_id, v_selection.event_option_id, v_selection.id,
    v_registration.id,
    case when v_partner_registration.id is null
      then null else v_partner_registration.id end,
    v_registration.player_id,
    v_partner_registration.player_id,
    'INITIAL', v_state, v_player_rating, v_partner_rating, v_combined,
    v_event.combined_rating_cap, v_player_source, v_partner_source,
    case when v_player_source = 'ORGANIZER_VERIFIED'
      then v_player_verification.id else null end,
    case when v_partner_source = 'ORGANIZER_VERIFIED'
      then v_partner_verification.id else null end,
    clock_timestamp(), null, null, null,
    coalesce(nullif(p_actor, ''), 'registration-trigger')
  )
  on conflict (event_option_id, selection_id, review_phase)
    where selection_id is not null
  do update set
    registration_id = excluded.registration_id,
    partner_registration_id = excluded.partner_registration_id,
    player_id_snapshot = excluded.player_id_snapshot,
    partner_player_id_snapshot = excluded.partner_player_id_snapshot,
    state = excluded.state,
    player_rating = excluded.player_rating,
    partner_rating = excluded.partner_rating,
    combined_rating = excluded.combined_rating,
    combined_rating_cap = excluded.combined_rating_cap,
    player_rating_source = excluded.player_rating_source,
    partner_rating_source = excluded.partner_rating_source,
    player_verification_id = excluded.player_verification_id,
    partner_verification_id = excluded.partner_verification_id,
    rating_as_of = excluded.rating_as_of,
    finalized_at = null,
    override_state = null,
    override_reason = null,
    reviewed_by = excluded.reviewed_by
  returning * into v_saved;

  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json
  ) values (
    v_club_id, v_selection.tournament_id, v_selection.event_option_id,
    'tournament_rating_eligibility_review', v_saved.id::text,
    'rating_eligibility_initial_refreshed',
    coalesce(nullif(p_actor, ''), 'registration-trigger'),
    v_before, to_jsonb(v_saved)
  );
  -- No exception is swallowed: review/audit failure rolls the selection
  -- statement back, while ON CONFLICT makes a retried statement converge.
end;
$$;

create or replace function public.trigger_initial_combined_rating_review_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  perform public.refresh_initial_combined_rating_review_v1(
    new.id::text, 'registration-trigger'
  );
  return new;
end;
$$;

drop trigger if exists trg_tournament_selection_initial_rating_review
  on public.tournament_registration_selections;
create trigger trg_tournament_selection_initial_rating_review
after insert or update of registration_id, event_option_id, partner_mode, partner_email
on public.tournament_registration_selections
for each row execute function public.trigger_initial_combined_rating_review_v1();

revoke all on function public.normalize_tournament_combined_rating(text)
  from public, anon, authenticated;
revoke all on function public.refresh_initial_combined_rating_review_v1(text, text)
  from public, anon, authenticated;
revoke all on function public.trigger_initial_combined_rating_review_v1()
  from public, anon, authenticated;
grant execute on function public.normalize_tournament_combined_rating(text)
  to service_role;
grant execute on function public.refresh_initial_combined_rating_review_v1(text, text)
  to service_role;
grant execute on function public.trigger_initial_combined_rating_review_v1()
  to service_role;

create table if not exists public.tournament_team_operations (
  operation_key text primary key,
  request_fingerprint text not null,
  club_id text not null,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  surface text not null,
  action text not null,
  entity_type text not null,
  entity_id text not null,
  actor text not null,
  status text not null default 'INTENT',
  result_json jsonb null,
  error_text text null,
  attempt_count integer not null default 1,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  completed_at timestamptz null,
  constraint tournament_team_operations_status_chk
    check (status in ('INTENT', 'COMPLETED', 'RECOVERY_REQUIRED'))
);
create index if not exists idx_tournament_team_operations_scope
  on public.tournament_team_operations (tournament_id, updated_at desc);

create table if not exists public.tournament_team_invitation_deliveries (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  team_id uuid not null references public.tournament_four_player_teams(id) on delete cascade,
  member_id uuid not null references public.tournament_four_player_team_members(id) on delete cascade,
  invitation_version integer not null,
  email_mode text not null,
  status text not null,
  provider_message_id text null,
  recipient_email_hash text not null,
  operation_key text not null references public.tournament_team_operations(operation_key) on delete restrict,
  created_at timestamptz not null default timezone('utc', now()),
  constraint tournament_team_invitation_delivery_status_chk
    check (status in (
      'pending', 'dry_run', 'staging_redirect', 'sent', 'skipped', 'failed'
    ))
);
create unique index if not exists uq_tournament_team_invitation_delivery
  on public.tournament_team_invitation_deliveries (
    member_id, invitation_version, email_mode
  )
  where status in ('pending', 'dry_run', 'staging_redirect', 'sent');

-- Keep every foreign-key lookup indexable from its leading column. These are
-- deliberately explicit because delete/correction paths lock parent and child
-- rows in a stable order.
create index if not exists idx_tournament_event_draws_parent_draw
  on public.tournament_event_draws (parent_draw_id);
create index if not exists idx_four_player_teams_event
  on public.tournament_four_player_teams (event_option_id);
create index if not exists idx_four_player_teams_captain_registration
  on public.tournament_four_player_teams (captain_registration_id);
create index if not exists idx_four_player_teams_captain_player
  on public.tournament_four_player_teams (captain_player_id);
create index if not exists idx_four_player_members_tournament
  on public.tournament_four_player_team_members (tournament_id);
create index if not exists idx_four_player_members_event
  on public.tournament_four_player_team_members (event_option_id);
create index if not exists idx_four_player_members_registration
  on public.tournament_four_player_team_members (registration_id);
create index if not exists idx_four_player_members_player
  on public.tournament_four_player_team_members (player_id);
create index if not exists idx_rating_verifications_event
  on public.tournament_rating_verifications (event_option_id);
create index if not exists idx_rating_verifications_registration
  on public.tournament_rating_verifications (registration_id);
create index if not exists idx_rating_verifications_player
  on public.tournament_rating_verifications (player_id);
create index if not exists idx_rating_reviews_tournament
  on public.tournament_rating_eligibility_reviews (tournament_id);
create index if not exists idx_rating_reviews_selection
  on public.tournament_rating_eligibility_reviews (selection_id);
create index if not exists idx_rating_reviews_team
  on public.tournament_rating_eligibility_reviews (team_id);
create index if not exists idx_rating_reviews_registration
  on public.tournament_rating_eligibility_reviews (registration_id);
create index if not exists idx_rating_reviews_partner_registration
  on public.tournament_rating_eligibility_reviews (partner_registration_id);
create index if not exists idx_rating_reviews_player_verification
  on public.tournament_rating_eligibility_reviews (player_verification_id);
create index if not exists idx_rating_reviews_partner_verification
  on public.tournament_rating_eligibility_reviews (partner_verification_id);
create index if not exists idx_team_matchups_tournament
  on public.tournament_team_matchups (tournament_id);
create index if not exists idx_team_matchups_event
  on public.tournament_team_matchups (event_option_id);
create index if not exists idx_team_matchups_team_a
  on public.tournament_team_matchups (team_a_id);
create index if not exists idx_team_matchups_team_b
  on public.tournament_team_matchups (team_b_id);
create index if not exists idx_team_matchups_winner
  on public.tournament_team_matchups (winner_team_id);
create index if not exists idx_team_matchups_loser
  on public.tournament_team_matchups (loser_team_id);
create index if not exists idx_team_lineups_tournament
  on public.tournament_team_lineup_submissions (tournament_id);
create index if not exists idx_team_lineups_team
  on public.tournament_team_lineup_submissions (team_id);
create index if not exists idx_team_lineups_singles_player
  on public.tournament_team_lineup_submissions (singles_tiebreak_player_id);
create index if not exists idx_team_match_games_tournament
  on public.tournament_team_match_games (tournament_id);
create index if not exists idx_team_match_games_rating_draw
  on public.tournament_team_match_games (rating_draw_id);
create index if not exists idx_team_match_games_winner
  on public.tournament_team_match_games (winner_team_id);
create index if not exists idx_team_match_games_loser
  on public.tournament_team_match_games (loser_team_id);
create index if not exists idx_four_player_podium_tournament
  on public.tournament_four_player_podium (tournament_id);
create index if not exists idx_team_invitation_deliveries_tournament
  on public.tournament_team_invitation_deliveries (tournament_id);
create index if not exists idx_team_invitation_deliveries_team
  on public.tournament_team_invitation_deliveries (team_id);
create index if not exists idx_team_invitation_deliveries_operation
  on public.tournament_team_invitation_deliveries (operation_key);

-- New operational tables are backend-only. There are intentionally no anon or
-- authenticated RLS policies; service-role access is explicit for projects
-- created after Supabase's 2026 Data API grant change.
alter table public.tournament_four_player_teams enable row level security;
alter table public.tournament_four_player_teams force row level security;
alter table public.tournament_four_player_team_members enable row level security;
alter table public.tournament_four_player_team_members force row level security;
alter table public.tournament_rating_verifications enable row level security;
alter table public.tournament_rating_verifications force row level security;
alter table public.tournament_rating_eligibility_reviews enable row level security;
alter table public.tournament_rating_eligibility_reviews force row level security;
alter table public.tournament_team_matchups enable row level security;
alter table public.tournament_team_matchups force row level security;
alter table public.tournament_team_lineup_submissions enable row level security;
alter table public.tournament_team_lineup_submissions force row level security;
alter table public.tournament_team_match_games enable row level security;
alter table public.tournament_team_match_games force row level security;
alter table public.tournament_four_player_podium enable row level security;
alter table public.tournament_four_player_podium force row level security;
alter table public.tournament_team_audit_events enable row level security;
alter table public.tournament_team_audit_events force row level security;
alter table public.tournament_team_operations enable row level security;
alter table public.tournament_team_operations force row level security;
alter table public.tournament_team_invitation_deliveries enable row level security;
alter table public.tournament_team_invitation_deliveries force row level security;

revoke all on table public.tournament_four_player_teams from public, anon, authenticated;
revoke all on table public.tournament_four_player_team_members from public, anon, authenticated;
revoke all on table public.tournament_rating_verifications from public, anon, authenticated;
revoke all on table public.tournament_rating_eligibility_reviews from public, anon, authenticated;
revoke all on table public.tournament_team_matchups from public, anon, authenticated;
revoke all on table public.tournament_team_lineup_submissions from public, anon, authenticated;
revoke all on table public.tournament_team_match_games from public, anon, authenticated;
revoke all on table public.tournament_four_player_podium from public, anon, authenticated;
revoke all on table public.tournament_team_audit_events from public, anon, authenticated;
revoke all on table public.tournament_team_operations from public, anon, authenticated;
revoke all on table public.tournament_team_invitation_deliveries from public, anon, authenticated;

grant select, insert, update, delete on table public.tournament_four_player_teams to service_role;
grant select, insert, update, delete on table public.tournament_four_player_team_members to service_role;
grant select, insert, update, delete on table public.tournament_rating_verifications to service_role;
grant select, insert, update, delete on table public.tournament_rating_eligibility_reviews to service_role;
grant select, insert, update, delete on table public.tournament_team_matchups to service_role;
grant select, insert, update, delete on table public.tournament_team_lineup_submissions to service_role;
grant select, insert, update, delete on table public.tournament_team_match_games to service_role;
grant select, insert, update, delete on table public.tournament_four_player_podium to service_role;
grant select, insert on table public.tournament_team_audit_events to service_role;
grant select, insert, update on table public.tournament_team_operations to service_role;
grant select, insert, update on table public.tournament_team_invitation_deliveries to service_role;

create or replace function public.begin_tournament_team_operation(
  p_operation_key text,
  p_request_fingerprint text,
  p_club_id text,
  p_tournament_id text,
  p_surface text,
  p_action text,
  p_entity_type text,
  p_entity_id text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_existing public.tournament_team_operations%rowtype;
begin
  if nullif(btrim(p_operation_key), '') is null
     or nullif(btrim(p_request_fingerprint), '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_OPERATION_ID_REQUIRED';
  end if;
  select operation.* into v_existing
    from public.tournament_team_operations operation
   where operation.operation_key = p_operation_key
   for update;
  if found then
    if v_existing.request_fingerprint <> p_request_fingerprint
       or v_existing.club_id <> p_club_id
       or v_existing.tournament_id::text <> p_tournament_id
       or v_existing.action <> p_action
       or v_existing.entity_type <> p_entity_type
       or v_existing.entity_id <> p_entity_id then
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_TEAM_OPERATION_KEY_REUSED';
    end if;
    update public.tournament_team_operations
       set attempt_count = attempt_count + 1,
           updated_at = clock_timestamp()
     where operation_key = p_operation_key;
    if v_existing.status = 'COMPLETED' and v_existing.result_json is not null then
      return jsonb_build_object(
        'replay', true,
        'result', v_existing.result_json
      );
    end if;
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_OPERATION_RECOVERY_REQUIRED';
  end if;
  insert into public.tournament_team_operations (
    operation_key, request_fingerprint, club_id, tournament_id, surface,
    action, entity_type, entity_id, actor, status
  ) values (
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id::uuid,
    p_surface, p_action, p_entity_type, p_entity_id,
    coalesce(nullif(p_actor, ''), 'unknown'), 'INTENT'
  );
  return jsonb_build_object('replay', false);
end;
$$;

create or replace function public.complete_tournament_team_operation(
  p_operation_key text,
  p_request_fingerprint text,
  p_result jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
begin
  update public.tournament_team_operations
     set status = 'COMPLETED',
         result_json = coalesce(p_result, '{}'::jsonb),
         error_text = null,
         completed_at = clock_timestamp(),
         updated_at = clock_timestamp()
   where operation_key = p_operation_key
     and request_fingerprint = p_request_fingerprint
     and status = 'INTENT';
  if not found then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_OPERATION_COMPLETION_STALE';
  end if;
  return coalesce(p_result, '{}'::jsonb);
end;
$$;

revoke all on function public.begin_tournament_team_operation(
  text, text, text, text, text, text, text, text, text
) from public, anon, authenticated;
revoke all on function public.complete_tournament_team_operation(
  text, text, jsonb
) from public, anon, authenticated;
grant execute on function public.begin_tournament_team_operation(
  text, text, text, text, text, text, text, text, text
) to service_role;
grant execute on function public.complete_tournament_team_operation(
  text, text, jsonb
) to service_role;

create or replace function public.reject_tournament_team_audit_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_AUDIT_IMMUTABLE';
end;
$$;

drop trigger if exists trg_tournament_team_audit_immutable
  on public.tournament_team_audit_events;
create trigger trg_tournament_team_audit_immutable
before update or delete on public.tournament_team_audit_events
for each row execute function public.reject_tournament_team_audit_mutation();

revoke all on function public.reject_tournament_team_audit_mutation()
  from public, anon, authenticated;
grant execute on function public.reject_tournament_team_audit_mutation()
  to service_role;

create or replace function public.touch_team_tournament_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  new.updated_at := clock_timestamp();
  if tg_table_name in (
    'tournament_four_player_teams',
    'tournament_team_matchups',
    'tournament_four_player_podium'
  ) and new.draw_id is not null then
    update public.tournament_event_draws
       set updated_at = clock_timestamp()
     where id = new.draw_id;
  elsif tg_table_name = 'tournament_team_match_games' then
    update public.tournament_event_draws draw
       set updated_at = clock_timestamp()
      from public.tournament_team_matchups matchup
     where matchup.id = new.matchup_id
       and draw.id = matchup.draw_id;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_four_player_teams_touch
  on public.tournament_four_player_teams;
create trigger trg_four_player_teams_touch
before update on public.tournament_four_player_teams
for each row execute function public.touch_team_tournament_updated_at();
drop trigger if exists trg_four_player_members_touch
  on public.tournament_four_player_team_members;
create trigger trg_four_player_members_touch
before update on public.tournament_four_player_team_members
for each row execute function public.touch_team_tournament_updated_at();
drop trigger if exists trg_rating_verifications_touch
  on public.tournament_rating_verifications;
create trigger trg_rating_verifications_touch
before update on public.tournament_rating_verifications
for each row execute function public.touch_team_tournament_updated_at();
drop trigger if exists trg_rating_reviews_touch
  on public.tournament_rating_eligibility_reviews;
create trigger trg_rating_reviews_touch
before update on public.tournament_rating_eligibility_reviews
for each row execute function public.touch_team_tournament_updated_at();
drop trigger if exists trg_team_matchups_touch
  on public.tournament_team_matchups;
create trigger trg_team_matchups_touch
before update on public.tournament_team_matchups
for each row execute function public.touch_team_tournament_updated_at();
drop trigger if exists trg_team_lineups_touch
  on public.tournament_team_lineup_submissions;
create trigger trg_team_lineups_touch
before update on public.tournament_team_lineup_submissions
for each row execute function public.touch_team_tournament_updated_at();
drop trigger if exists trg_team_match_games_touch
  on public.tournament_team_match_games;
create trigger trg_team_match_games_touch
before update on public.tournament_team_match_games
for each row execute function public.touch_team_tournament_updated_at();
drop trigger if exists trg_four_player_podium_touch
  on public.tournament_four_player_podium;
create trigger trg_four_player_podium_touch
before update on public.tournament_four_player_podium
for each row execute function public.touch_team_tournament_updated_at();

revoke all on function public.touch_team_tournament_updated_at()
  from public, anon, authenticated;
grant execute on function public.touch_team_tournament_updated_at()
  to service_role;

create or replace function public.enforce_combined_rating_draw_eligibility()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_event public.tournament_event_options%rowtype;
  v_review public.tournament_rating_eligibility_reviews%rowtype;
begin
  if new.event_option_id is null then
    return new;
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = new.event_option_id::text
     and event.tournament_id = new.tournament_id
   for share;
  if not found or v_event.eligibility_mode <> 'COMBINED_RATING_CAP' then
    return new;
  end if;
  if new.source_selection_id is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_SELECTION_REQUIRED';
  end if;
  select review.* into v_review
    from public.tournament_rating_eligibility_reviews review
   where review.tournament_id = new.tournament_id
     and review.event_option_id::text = new.event_option_id::text
     and review.selection_id = new.source_selection_id
     and review.review_phase = 'REGISTRATION_CLOSE'
   for share;
  if not found or v_review.finalized_at is null
     or coalesce(v_review.override_state, v_review.state) <> 'ELIGIBLE'
     or (
       v_review.override_state is not null
       and nullif(btrim(v_review.override_reason), '') is null
     ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_DRAW_BLOCKED';
  end if;
  new.eligibility_review_id := v_review.id;
  new.eligibility_reviewed_at := v_review.rating_as_of;
  new.eligibility_state_snapshot := coalesce(
    v_review.override_state, v_review.state
  );
  new.combined_rating_snapshot := v_review.combined_rating;
  new.combined_rating_cap_snapshot := v_review.combined_rating_cap;
  return new;
end;
$$;

drop trigger if exists trg_tournament_team_combined_rating_eligibility
  on public.tournament_teams;
create trigger trg_tournament_team_combined_rating_eligibility
before insert or update of event_option_id, source_selection_id
on public.tournament_teams
for each row execute function public.enforce_combined_rating_draw_eligibility();

revoke all on function public.enforce_combined_rating_draw_eligibility()
  from public, anon, authenticated;
grant execute on function public.enforce_combined_rating_draw_eligibility()
  to service_role;

create or replace function public.admin_update_tournament_competition_config_cas(
  p_club_id text,
  p_tournament_id text,
  p_event_option_id text,
  p_expected_updated_at timestamptz,
  p_patch jsonb,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_event public.tournament_event_options%rowtype;
  v_before jsonb;
  v_after jsonb;
  v_eligibility text;
  v_cap numeric(5,2);
  v_format text;
  v_tiebreak text;
  v_playoff_format text;
  v_allow_substitutes boolean;
  v_existing_selections integer;
  v_existing_draws integer;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'setup', 'competition_config_update', 'tournament_event_option',
    p_event_option_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if pg_catalog.jsonb_typeof(coalesce(p_patch, '{}'::jsonb)) <> 'object' then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMPETITION_PATCH_INVALID';
  end if;
  perform 1
    from public.tournaments tournament
   where tournament.id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_NOT_FOUND';
  end if;

  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = p_event_option_id
     and event.tournament_id::text = p_tournament_id
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_EVENT_NOT_FOUND';
  end if;
  if p_expected_updated_at is null
     or v_event.updated_at is distinct from p_expected_updated_at then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_COMPETITION_CONFIG_STALE';
  end if;

  v_eligibility := upper(coalesce(
    nullif(p_patch->>'eligibility_mode', ''),
    v_event.eligibility_mode
  ));
  v_cap := case
    when p_patch ? 'combined_rating_cap'
      then nullif(p_patch->>'combined_rating_cap', '')::numeric
    else v_event.combined_rating_cap
  end;
  v_format := upper(coalesce(
    nullif(p_patch->>'competition_format', ''),
    v_event.competition_format
  ));
  v_tiebreak := upper(coalesce(
    nullif(p_patch->>'team_tiebreak_mode', ''),
    v_event.team_tiebreak_mode
  ));
  v_playoff_format := upper(coalesce(
    nullif(p_patch->>'team_playoff_format', ''),
    v_event.team_playoff_format
  ));
  v_allow_substitutes := case
    when p_patch ? 'team_allow_substitutes'
      then (p_patch->>'team_allow_substitutes')::boolean
    else v_event.team_allow_substitutes
  end;
  if v_eligibility not in ('STANDARD', 'COMBINED_RATING_CAP')
     or (v_eligibility = 'COMBINED_RATING_CAP'
       and (
         v_cap is null or v_cap <= 0 or v_cap > 14
         or upper(coalesce(v_event.event_type, '')) = 'SINGLES'
         or not coalesce(v_event.partner_required, false)
       ))
     or (v_eligibility = 'STANDARD' and v_cap is not null) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_CONFIG_INVALID';
  end if;
  if v_format not in ('STANDARD', 'FOUR_PLAYER_TEAM')
     or (v_format = 'FOUR_PLAYER_TEAM'
       and (
         v_eligibility <> 'STANDARD'
         or
         v_tiebreak not in ('SINGLES', 'SKINNY_RELAY')
         or v_playoff_format not in (
           'NONE', 'TOP_2_FINAL', 'TOP_4_SEMIFINALS',
           'TOP_4_SEMIFINALS_WITH_BRONZE'
         )
       )) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_FORMAT_CONFIG_INVALID';
  end if;

  select count(*) into v_existing_selections
    from public.tournament_registration_selections selection
   where selection.tournament_id::text = p_tournament_id
     and selection.event_option_id::text = p_event_option_id;
  select count(*) into v_existing_draws
    from public.tournament_event_draws draw
   where draw.tournament_id::text = p_tournament_id
     and draw.event_option_id::text = p_event_option_id;
  if (v_existing_selections > 0 or v_existing_draws > 0)
     and (
       v_eligibility is distinct from v_event.eligibility_mode
       or v_cap is distinct from v_event.combined_rating_cap
       or v_format is distinct from v_event.competition_format
       or v_tiebreak is distinct from v_event.team_tiebreak_mode
       or v_playoff_format is distinct from v_event.team_playoff_format
       or v_allow_substitutes is distinct from v_event.team_allow_substitutes
     ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_COMPETITION_CONFIG_IN_USE';
  end if;

  v_before := to_jsonb(v_event);
  update public.tournament_event_options event
     set eligibility_mode = v_eligibility,
         combined_rating_cap = v_cap,
         rating_source_policy = 'PCS_OR_ORGANIZER_VERIFIED',
         rating_review_timing = 'INITIAL_AND_REGISTRATION_CLOSE',
         competition_format = v_format,
         team_roster_size = case when v_format = 'FOUR_PLAYER_TEAM' then 4 else 2 end,
         team_gender_rule = case
           when v_format = 'FOUR_PLAYER_TEAM' then 'TWO_MEN_TWO_WOMEN'
           else 'NONE'
         end,
         team_tiebreak_mode = v_tiebreak,
         team_playoff_format = case
           when v_format = 'FOUR_PLAYER_TEAM' then v_playoff_format
           else 'NONE'
         end,
         team_allow_substitutes = case
           when v_format = 'FOUR_PLAYER_TEAM' then v_allow_substitutes
           else false
         end,
         updated_at = clock_timestamp()
   where event.id = v_event.id
  returning to_jsonb(event.*) into v_after;

  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json
  ) values (
    p_club_id, p_tournament_id::uuid, p_event_option_id,
    'tournament_event_option', p_event_option_id,
    'competition_config_updated', coalesce(nullif(p_actor, ''), 'unknown'),
    v_before, v_after
  );
  v_result := jsonb_build_object(
    'ok', true, 'event_option', v_after, 'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_update_tournament_competition_config_cas(
  text, text, text, timestamptz, jsonb, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_update_tournament_competition_config_cas(
  text, text, text, timestamptz, jsonb, text, text, text
) to service_role;

create or replace function public.admin_upsert_tournament_rating_verification_cas(
  p_club_id text,
  p_tournament_id text,
  p_event_option_id text,
  p_registration_id text,
  p_rating numeric,
  p_note text,
  p_expected_version integer,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_registration public.tournament_registrations%rowtype;
  v_existing public.tournament_rating_verifications%rowtype;
  v_saved public.tournament_rating_verifications%rowtype;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'registration', 'rating_verification_upsert',
    'tournament_registration', p_registration_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if p_rating is null or p_rating < 0 or p_rating > 7 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_VERIFIED_RATING_INVALID';
  end if;
  perform 1
    from public.tournaments tournament
   where tournament.id::text = p_tournament_id
     and tournament.club_id::text = p_club_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_NOT_FOUND';
  end if;
  perform 1
    from public.tournament_event_options event
   where event.id::text = p_event_option_id
     and event.tournament_id::text = p_tournament_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_EVENT_NOT_FOUND';
  end if;
  select registration.* into v_registration
    from public.tournament_registrations registration
   where registration.id::text = p_registration_id
     and registration.tournament_id::text = p_tournament_id
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_REGISTRATION_NOT_FOUND';
  end if;

  select verification.* into v_existing
    from public.tournament_rating_verifications verification
   where verification.tournament_id::text = p_tournament_id
     and verification.event_option_id::text = p_event_option_id
     and verification.registration_id::text = p_registration_id
     and verification.status = 'ACTIVE'
   for update;
  if found and (
    p_expected_version is null
    or v_existing.version is distinct from p_expected_version
  ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_RATING_VERIFICATION_STALE';
  end if;
  if not found and coalesce(p_expected_version, 0) <> 0 then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_RATING_VERIFICATION_STALE';
  end if;

  if v_existing.id is null then
    insert into public.tournament_rating_verifications (
      tournament_id, event_option_id, registration_id, player_id, rating,
      source, status, note, version, verified_by
    ) values (
      p_tournament_id::uuid, p_event_option_id, p_registration_id,
      v_registration.player_id, round(p_rating::numeric, 2),
      'ORGANIZER_VERIFIED', 'ACTIVE', nullif(btrim(p_note), ''), 1,
      coalesce(nullif(p_actor, ''), 'unknown')
    )
    returning * into v_saved;
  else
    update public.tournament_rating_verifications verification
       set player_id = v_registration.player_id,
           rating = round(p_rating::numeric, 2),
           note = nullif(btrim(p_note), ''),
           version = verification.version + 1,
           verified_by = coalesce(nullif(p_actor, ''), 'unknown'),
           verified_at = clock_timestamp(),
           updated_at = clock_timestamp()
     where verification.id = v_existing.id
    returning * into v_saved;
  end if;

  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json
  ) values (
    p_club_id, p_tournament_id::uuid, p_event_option_id,
    'tournament_rating_verification', v_saved.id::text,
    case when v_existing.id is null then 'rating_verified' else 'rating_verification_updated' end,
    coalesce(nullif(p_actor, ''), 'unknown'),
    case when v_existing.id is null then null else to_jsonb(v_existing) end,
    to_jsonb(v_saved)
  );
  perform public.refresh_initial_combined_rating_review_v1(
    selection.id::text,
    coalesce(nullif(p_actor, ''), 'rating-verification')
  )
    from public.tournament_registration_selections selection
   where selection.tournament_id::text = p_tournament_id
     and selection.event_option_id::text = p_event_option_id
     and (
       selection.registration_id::text = p_registration_id
       or exists (
         select 1
           from public.tournament_rating_eligibility_reviews review
          where review.tournament_id = selection.tournament_id
            and review.event_option_id::text = selection.event_option_id::text
            and review.selection_id::text = selection.id::text
            and review.review_phase = 'INITIAL'
            and review.partner_registration_id::text = p_registration_id
       )
     );
  v_result := jsonb_build_object(
    'ok', true, 'verification', to_jsonb(v_saved),
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_upsert_tournament_rating_verification_cas(
  text, text, text, text, numeric, text, integer, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_upsert_tournament_rating_verification_cas(
  text, text, text, text, numeric, text, integer, text, text, text
) to service_role;

create or replace function public.admin_record_tournament_rating_review_cas(
  p_club_id text,
  p_tournament_id text,
  p_event_option_id text,
  p_review jsonb,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_event public.tournament_event_options%rowtype;
  v_selection public.tournament_registration_selections%rowtype;
  v_authoritative public.tournament_rating_eligibility_reviews%rowtype;
  v_existing_final public.tournament_rating_eligibility_reviews%rowtype;
  v_saved public.tournament_rating_eligibility_reviews%rowtype;
  v_phase text := upper(coalesce(p_review->>'review_phase', 'INITIAL'));
  v_state text;
  v_selection_id text := nullif(p_review->>'selection_id', '');
  v_team_id uuid := nullif(p_review->>'team_id', '')::uuid;
  v_registration_id text;
  v_partner_registration_id text;
  v_player_id_snapshot integer;
  v_partner_player_id_snapshot integer;
  v_player_rating numeric(4,2);
  v_partner_rating numeric(4,2);
  v_combined numeric(5,2);
  v_player_source text;
  v_partner_source text;
  v_player_verification uuid;
  v_partner_verification uuid;
  v_override_state text := nullif(upper(p_review->>'override_state'), '');
  v_override_reason text := nullif(btrim(p_review->>'override_reason'), '');
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'registration', 'rating_eligibility_review',
    case when v_selection_id is not null
      then 'tournament_registration_selection'
      else 'tournament_four_player_team' end,
    coalesce(v_selection_id, v_team_id::text), p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  perform 1
    from public.tournaments tournament
   where tournament.id::text = p_tournament_id
     and tournament.club_id::text = p_club_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_NOT_FOUND';
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = p_event_option_id
     and event.tournament_id::text = p_tournament_id
   for share;
  if not found
     or v_event.eligibility_mode <> 'COMBINED_RATING_CAP'
     or v_event.combined_rating_cap is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_EVENT_REQUIRED';
  end if;
  if (v_selection_id is null and v_team_id is null)
     or (v_selection_id is not null and v_team_id is not null) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_RATING_REVIEW_SCOPE_INVALID';
  end if;
  if v_selection_id is not null then
    select selection.* into v_selection
      from public.tournament_registration_selections selection
     where selection.id::text = v_selection_id
       and selection.tournament_id::text = p_tournament_id
       and selection.event_option_id::text = p_event_option_id
     for update;
    if not found then
      raise exception using errcode = 'P0002',
        message = 'JUPR_TOURNAMENT_SELECTION_NOT_FOUND';
    end if;
    if nullif(p_review->>'expected_selection_updated_at', '') is null
       or v_selection.updated_at is distinct from
         (p_review->>'expected_selection_updated_at')::timestamptz then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_RATING_REVIEW_SELECTION_STALE';
    end if;
    -- Recompute linked/verified evidence inside this transaction. The helper
    -- holds share locks on registrations, players, and active verifications,
    -- so REGISTRATION_CLOSE is a point-in-time authoritative snapshot rather
    -- than a copy of browser JSON or a stale INITIAL row.
    perform public.refresh_initial_combined_rating_review_v1(
      v_selection.id::text,
      coalesce(nullif(p_actor, ''), 'rating-review-refresh')
    );
    select review.* into v_authoritative
      from public.tournament_rating_eligibility_reviews review
     where review.tournament_id::text = p_tournament_id
       and review.event_option_id::text = p_event_option_id
       and review.selection_id::text = v_selection.id::text
       and review.review_phase = 'INITIAL'
     for update;
    if not found then
      raise exception using errcode = 'P0002',
        message = 'JUPR_TOURNAMENT_AUTHORITATIVE_RATING_EVIDENCE_NOT_FOUND';
    end if;
    v_registration_id := v_authoritative.registration_id::text;
    v_partner_registration_id :=
      v_authoritative.partner_registration_id::text;
    v_player_id_snapshot := v_authoritative.player_id_snapshot;
    v_partner_player_id_snapshot :=
      v_authoritative.partner_player_id_snapshot;
    v_state := v_authoritative.state;
    v_player_rating := v_authoritative.player_rating;
    v_partner_rating := v_authoritative.partner_rating;
    v_combined := v_authoritative.combined_rating;
    v_player_source := v_authoritative.player_rating_source;
    v_partner_source := v_authoritative.partner_rating_source;
    v_player_verification := v_authoritative.player_verification_id;
    v_partner_verification := v_authoritative.partner_verification_id;
  else
    -- The team-scoped branch is retained for compatibility. Four-player team
    -- events are constrained to STANDARD eligibility, so it cannot finalize a
    -- combined-rating draw entry. Do not let it weaken selection evidence.
    v_registration_id := nullif(p_review->>'registration_id', '');
    v_partner_registration_id :=
      nullif(p_review->>'partner_registration_id', '');
    v_state := upper(coalesce(p_review->>'state', 'REVIEW_REQUIRED'));
    v_player_rating := nullif(p_review->>'player_rating', '')::numeric;
    v_partner_rating := nullif(p_review->>'partner_rating', '')::numeric;
    v_combined := nullif(p_review->>'combined_rating', '')::numeric;
    v_player_source :=
      upper(coalesce(p_review->>'player_rating_source', 'MISSING'));
    v_partner_source :=
      upper(coalesce(p_review->>'partner_rating_source', 'MISSING'));
    v_player_verification :=
      nullif(p_review->>'player_verification_id', '')::uuid;
    v_partner_verification :=
      nullif(p_review->>'partner_verification_id', '')::uuid;
    perform 1 from public.tournament_four_player_teams team
     where team.id = v_team_id
       and team.tournament_id::text = p_tournament_id
       and team.event_option_id::text = p_event_option_id;
    if not found then
      raise exception using errcode = 'P0002',
        message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_NOT_FOUND';
    end if;
  end if;
  if v_registration_id is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_RATING_REVIEW_SCOPE_INVALID';
  end if;
  perform 1 from public.tournament_registrations registration
   where registration.id::text = v_registration_id
     and registration.tournament_id::text = p_tournament_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_REGISTRATION_NOT_FOUND';
  end if;
  if v_phase not in ('INITIAL', 'REGISTRATION_CLOSE', 'ADMIN_REVIEW')
     or v_state not in (
       'ELIGIBLE', 'INELIGIBLE', 'REVIEW_REQUIRED', 'PROVISIONAL_NEEDS_PARTNER'
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_RATING_REVIEW_STATE_INVALID';
  end if;
  if v_override_state is not null
     and (
       v_override_state not in ('ELIGIBLE', 'INELIGIBLE')
       or v_override_reason is null
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_RATING_OVERRIDE_REASON_REQUIRED';
  end if;
  if (v_combined is not null
       and (
         v_player_rating is null
         or v_partner_rating is null
         or v_combined <> round(v_player_rating + v_partner_rating, 2)
       ))
     or (v_state = 'ELIGIBLE' and not (v_combined < v_event.combined_rating_cap))
     or (v_state = 'INELIGIBLE' and not (v_combined >= v_event.combined_rating_cap))
     or (v_state in ('REVIEW_REQUIRED', 'PROVISIONAL_NEEDS_PARTNER')
       and v_combined is not null) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_RATING_REVIEW_MATH_INVALID';
  end if;
  if v_player_source = 'ORGANIZER_VERIFIED' and not exists (
    select 1 from public.tournament_rating_verifications verification
     where verification.id = v_player_verification
       and verification.tournament_id::text = p_tournament_id
       and verification.registration_id::text = v_registration_id
       and verification.status = 'ACTIVE'
       and verification.rating = v_player_rating
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_PLAYER_VERIFICATION_INVALID';
  end if;
  if v_partner_source = 'ORGANIZER_VERIFIED' and not exists (
    select 1 from public.tournament_rating_verifications verification
     where verification.id = v_partner_verification
       and verification.tournament_id::text = p_tournament_id
       and verification.registration_id::text = v_partner_registration_id
       and verification.status = 'ACTIVE'
       and verification.rating = v_partner_rating
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_PARTNER_VERIFICATION_INVALID';
  end if;
  if v_selection_id is not null and v_phase = 'REGISTRATION_CLOSE'
     and exists (
       select 1 from public.tournament_teams imported_team
        where imported_team.tournament_id::text = p_tournament_id
          and imported_team.source_selection_id::text = v_selection_id
     ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_RATING_REVIEW_IMPORTED_TO_DRAW';
  end if;

  if v_selection_id is not null then
    if v_phase = 'REGISTRATION_CLOSE' then
      select review.* into v_existing_final
        from public.tournament_rating_eligibility_reviews review
       where review.tournament_id::text = p_tournament_id
         and review.event_option_id::text = p_event_option_id
         and review.selection_id::text = v_selection_id
         and review.review_phase = 'REGISTRATION_CLOSE'
         and review.finalized_at is not null
       for update;
      if found then
        if v_existing_final.registration_id::text = v_registration_id
           and v_existing_final.partner_registration_id::text
             is not distinct from v_partner_registration_id
           and v_existing_final.player_id_snapshot
             is not distinct from v_player_id_snapshot
           and v_existing_final.partner_player_id_snapshot
             is not distinct from v_partner_player_id_snapshot
           and v_existing_final.state = v_state
           and v_existing_final.player_rating is not distinct from v_player_rating
           and v_existing_final.partner_rating is not distinct from v_partner_rating
           and v_existing_final.combined_rating is not distinct from v_combined
           and v_existing_final.combined_rating_cap
             is not distinct from v_event.combined_rating_cap
           and v_existing_final.player_rating_source = v_player_source
           and v_existing_final.partner_rating_source = v_partner_source
           and v_existing_final.player_verification_id
             is not distinct from v_player_verification
           and v_existing_final.partner_verification_id
             is not distinct from v_partner_verification
           and v_existing_final.override_state
             is not distinct from v_override_state
           and v_existing_final.override_reason
             is not distinct from v_override_reason then
          v_result := jsonb_build_object(
            'ok', true,
            'review', to_jsonb(v_existing_final),
            'immutable_replay', true,
            'operation_key', p_operation_key
          );
          return public.complete_tournament_team_operation(
            p_operation_key, p_request_fingerprint, v_result
          );
        end if;
        raise exception using errcode = 'P0001',
          message =
            'JUPR_TOURNAMENT_RATING_REVIEW_FINALIZED_IMMUTABLE';
      end if;
    end if;
    insert into public.tournament_rating_eligibility_reviews (
      tournament_id, event_option_id, selection_id, registration_id,
      partner_registration_id, player_id_snapshot,
      partner_player_id_snapshot, review_phase, state, player_rating,
      partner_rating, combined_rating, combined_rating_cap,
      player_rating_source, partner_rating_source,
      player_verification_id, partner_verification_id, rating_as_of,
      finalized_at,
      override_state, override_reason, reviewed_by
    ) values (
      p_tournament_id::uuid, p_event_option_id, v_selection_id,
      v_registration_id, v_partner_registration_id,
      v_player_id_snapshot, v_partner_player_id_snapshot, v_phase, v_state,
      v_player_rating, v_partner_rating, v_combined,
      v_event.combined_rating_cap, v_player_source, v_partner_source,
      v_player_verification, v_partner_verification,
      coalesce(v_authoritative.rating_as_of, clock_timestamp()),
      case when v_phase = 'REGISTRATION_CLOSE' then clock_timestamp() else null end,
      v_override_state, v_override_reason,
      coalesce(nullif(p_actor, ''), 'unknown')
    )
    on conflict (event_option_id, selection_id, review_phase)
      where selection_id is not null
    do update set
      partner_registration_id = excluded.partner_registration_id,
      player_id_snapshot = excluded.player_id_snapshot,
      partner_player_id_snapshot = excluded.partner_player_id_snapshot,
      state = excluded.state,
      player_rating = excluded.player_rating,
      partner_rating = excluded.partner_rating,
      combined_rating = excluded.combined_rating,
      combined_rating_cap = excluded.combined_rating_cap,
      player_rating_source = excluded.player_rating_source,
      partner_rating_source = excluded.partner_rating_source,
      player_verification_id = excluded.player_verification_id,
      partner_verification_id = excluded.partner_verification_id,
      rating_as_of = clock_timestamp(),
      finalized_at = excluded.finalized_at,
      override_state = excluded.override_state,
      override_reason = excluded.override_reason,
      reviewed_by = excluded.reviewed_by
    returning * into v_saved;
  else
    insert into public.tournament_rating_eligibility_reviews (
      tournament_id, event_option_id, team_id, registration_id,
      partner_registration_id, player_id_snapshot,
      partner_player_id_snapshot, review_phase, state, player_rating,
      partner_rating, combined_rating, combined_rating_cap,
      player_rating_source, partner_rating_source,
      player_verification_id, partner_verification_id, finalized_at,
      override_state, override_reason, reviewed_by
    ) values (
      p_tournament_id::uuid, p_event_option_id, v_team_id,
      v_registration_id, v_partner_registration_id,
      v_player_id_snapshot, v_partner_player_id_snapshot, v_phase, v_state,
      v_player_rating, v_partner_rating, v_combined,
      v_event.combined_rating_cap, v_player_source, v_partner_source,
      v_player_verification, v_partner_verification,
      case when v_phase = 'REGISTRATION_CLOSE' then clock_timestamp() else null end,
      v_override_state, v_override_reason,
      coalesce(nullif(p_actor, ''), 'unknown')
    )
    on conflict (event_option_id, team_id, registration_id, review_phase)
      where team_id is not null
    do update set
      state = excluded.state,
      player_id_snapshot = excluded.player_id_snapshot,
      partner_player_id_snapshot = excluded.partner_player_id_snapshot,
      player_rating = excluded.player_rating,
      partner_rating = excluded.partner_rating,
      combined_rating = excluded.combined_rating,
      combined_rating_cap = excluded.combined_rating_cap,
      player_rating_source = excluded.player_rating_source,
      partner_rating_source = excluded.partner_rating_source,
      player_verification_id = excluded.player_verification_id,
      partner_verification_id = excluded.partner_verification_id,
      rating_as_of = clock_timestamp(),
      finalized_at = excluded.finalized_at,
      override_state = excluded.override_state,
      override_reason = excluded.override_reason,
      reviewed_by = excluded.reviewed_by
    returning * into v_saved;
  end if;

  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, after_json, request_fingerprint
  ) values (
    p_club_id, p_tournament_id::uuid, p_event_option_id,
    'tournament_rating_eligibility_review', v_saved.id::text,
    case when v_phase = 'REGISTRATION_CLOSE'
      then 'rating_eligibility_finalized'
      else 'rating_eligibility_reviewed' end,
    coalesce(nullif(p_actor, ''), 'unknown'), to_jsonb(v_saved),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'review', to_jsonb(v_saved),
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_record_tournament_rating_review_cas(
  text, text, text, jsonb, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_record_tournament_rating_review_cas(
  text, text, text, jsonb, text, text, text
) to service_role;

create or replace function public.admin_create_tournament_four_player_team(
  p_club_id text,
  p_tournament_id text,
  p_event_option_id text,
  p_team_name text,
  p_captain_registration_id text,
  p_members jsonb,
  p_creation_fingerprint text,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_event public.tournament_event_options%rowtype;
  v_captain public.tournament_registrations%rowtype;
  v_team public.tournament_four_player_teams%rowtype;
  v_member jsonb;
  v_registration public.tournament_registrations%rowtype;
  v_slot text;
  v_email text;
  v_registration_id text;
  v_is_captain boolean;
  v_gender text;
  v_members jsonb;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'registration', 'four_player_team_create', 'tournament_event_option',
    p_event_option_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if nullif(btrim(p_team_name), '') is null
     or pg_catalog.jsonb_typeof(coalesce(p_members, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_members) <> 4
     or coalesce(p_creation_fingerprint, '') !~ '^[0-9a-f]{64}$' then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_PAYLOAD_INVALID';
  end if;
  perform 1
    from public.tournaments tournament
   where tournament.id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_NOT_FOUND';
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = p_event_option_id
     and event.tournament_id::text = p_tournament_id
   for share;
  if not found or v_event.competition_format <> 'FOUR_PLAYER_TEAM' then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_EVENT_REQUIRED';
  end if;
  select registration.* into v_captain
    from public.tournament_registrations registration
   where registration.id::text = p_captain_registration_id
     and registration.tournament_id::text = p_tournament_id
     and upper(coalesce(registration.status, '')) = 'CONFIRMED'
   for share;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_CAPTAIN_REGISTRATION_NOT_FOUND';
  end if;
  perform 1
    from public.tournament_registration_selections selection
   where selection.tournament_id::text = p_tournament_id
     and selection.registration_id::text = p_captain_registration_id
     and selection.event_option_id::text = p_event_option_id
     and upper(coalesce(selection.partner_mode, 'NONE')) = 'NONE'
   for share;
  if not found then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_CAPTAIN_EVENT_SELECTION_REQUIRED';
  end if;
  if v_captain.player_id is not null
     and not exists (
       select 1 from public.players player
        where player.id = v_captain.player_id
          and player.club_id::text = p_club_id
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_CAPTAIN_PLAYER_LINK_INVALID';
  end if;
  if (
    select count(distinct upper(member->>'slot'))
      from pg_catalog.jsonb_array_elements(p_members) member
  ) <> 4 or exists (
    select 1
      from pg_catalog.jsonb_array_elements(p_members) member
     where upper(coalesce(member->>'slot', '')) not in (
       'MAN_1', 'MAN_2', 'WOMAN_1', 'WOMAN_2'
     )
       or nullif(lower(btrim(member->>'email')), '') is null
  ) or (
    select count(distinct lower(btrim(member->>'email')))
      from pg_catalog.jsonb_array_elements(p_members) member
  ) <> 4 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_ROSTER_INVALID';
  end if;
  if (
    select count(*)
      from pg_catalog.jsonb_array_elements(p_members) member
     where nullif(member->>'registration_id', '') = p_captain_registration_id
  ) <> 1 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_CAPTAIN_MUST_BE_ON_TEAM';
  end if;

  select team.* into v_team
    from public.tournament_four_player_teams team
   where team.tournament_id::text = p_tournament_id
     and team.event_option_id::text = p_event_option_id
     and team.captain_registration_id::text = p_captain_registration_id
     and team.status not in ('WITHDRAWN', 'CANCELLED')
   for update;
  if found then
    if v_team.creation_fingerprint is distinct from p_creation_fingerprint then
      raise exception using errcode = '23505',
        message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_CREATE_CONFLICT';
    end if;
    select coalesce(
             jsonb_agg(to_jsonb(member) order by member.slot),
             '[]'::jsonb
           )
      into v_members
      from public.tournament_four_player_team_members member
     where member.team_id = v_team.id
       and member.status <> 'REMOVED';
    v_result := jsonb_build_object(
      'ok', true,
      'team', to_jsonb(v_team),
      'members', v_members,
      'recovered_by_business_identity', true
    );
    return public.complete_tournament_team_operation(
      p_operation_key, p_request_fingerprint, v_result
    );
  end if;

  insert into public.tournament_four_player_teams (
    tournament_id, event_option_id, name, captain_registration_id,
    captain_player_id, status, eligibility_state, creation_fingerprint,
    created_by
  ) values (
    p_tournament_id::uuid, p_event_option_id, btrim(p_team_name),
    p_captain_registration_id, v_captain.player_id,
    'FORMING',
    case when v_event.eligibility_mode = 'COMBINED_RATING_CAP'
      then 'REVIEW_REQUIRED' else 'NOT_REQUIRED' end,
    p_creation_fingerprint,
    coalesce(nullif(p_actor, ''), 'unknown')
  )
  returning * into v_team;

  for v_member in
    select member from pg_catalog.jsonb_array_elements(p_members) member
  loop
    v_slot := upper(v_member->>'slot');
    v_email := lower(btrim(v_member->>'email'));
    v_registration_id := nullif(v_member->>'registration_id', '');
    v_is_captain := v_registration_id = p_captain_registration_id;
    v_registration := null;
    if v_registration_id is not null then
      select registration.* into v_registration
        from public.tournament_registrations registration
       where registration.id::text = v_registration_id
         and registration.tournament_id::text = p_tournament_id
         and lower(btrim(registration.email)) = v_email
         and upper(coalesce(registration.status, 'CONFIRMED')) not in (
           'CANCELLED', 'WITHDRAWN'
         );
      if not found then
        raise exception using errcode = '22023',
          message = 'JUPR_TOURNAMENT_TEAM_MEMBER_REGISTRATION_INVALID';
      end if;
      v_gender := regexp_replace(
        lower(coalesce(v_registration.gender, '')), '[^a-z]', '', 'g'
      );
    else
      v_gender := regexp_replace(
        lower(coalesce(v_member->>'gender', '')), '[^a-z]', '', 'g'
      );
    end if;
    if (v_slot like 'MAN_%' and v_gender not in (
      'm', 'male', 'man', 'men', 'mens', 'boy', 'boys'
    )) or (v_slot like 'WOMAN_%' and v_gender not in (
      'f', 'female', 'woman', 'women', 'womens', 'girl', 'girls'
    )) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_TEAM_MEMBER_GENDER_INVALID';
    end if;
    insert into public.tournament_four_player_team_members (
      team_id, tournament_id, event_option_id, slot, invited_email,
      registration_id, player_id, display_name_snapshot, gender_snapshot,
      status, accepted_at
    ) values (
      v_team.id, v_team.tournament_id, v_team.event_option_id,
      v_slot, v_email, v_registration.id, v_registration.player_id,
      coalesce(
        nullif(v_registration.display_name, ''),
        nullif(btrim(concat_ws(' ', v_registration.first_name, v_registration.last_name)), ''),
        nullif(v_member->>'display_name', '')
      ),
      coalesce(nullif(v_registration.gender, ''), nullif(v_member->>'gender', '')),
      case when v_is_captain then 'ACCEPTED' else 'INVITED' end,
      case when v_is_captain then clock_timestamp() else null end
    );
  end loop;

  select coalesce(jsonb_agg(to_jsonb(member) order by member.slot), '[]'::jsonb)
    into v_members
    from public.tournament_four_player_team_members member
   where member.team_id = v_team.id;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, after_json, request_fingerprint
  ) values (
    p_club_id, v_team.tournament_id, v_team.event_option_id,
    'tournament_four_player_team', v_team.id::text,
    'four_player_team_created', coalesce(nullif(p_actor, ''), 'unknown'),
    jsonb_build_object('team', to_jsonb(v_team), 'members', v_members),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'team', to_jsonb(v_team), 'members', v_members,
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_create_tournament_four_player_team(
  text, text, text, text, text, jsonb, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_create_tournament_four_player_team(
  text, text, text, text, text, jsonb, text, text, text, text
) to service_role;

create or replace function public.server_respond_tournament_four_player_invite(
  p_club_id text,
  p_tournament_id text,
  p_team_id text,
  p_member_id text,
  p_registration_id text,
  p_invitation_version integer,
  p_invitation_token_hash text,
  p_action text,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_member public.tournament_four_player_team_members%rowtype;
  v_registration public.tournament_registrations%rowtype;
  v_team public.tournament_four_player_teams%rowtype;
  v_before jsonb;
  v_action text := upper(coalesce(p_action, ''));
  v_gender text;
  v_accepted integer;
  v_linked integer;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'public_registration', 'four_player_invite_' || lower(v_action),
    'tournament_four_player_team_member', p_member_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if v_action not in ('ACCEPT', 'DECLINE') then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_INVITE_ACTION_INVALID';
  end if;
  perform 1
    from public.tournaments tournament
   where tournament.id::text = p_tournament_id
     and tournament.club_id::text = p_club_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_NOT_FOUND';
  end if;
  select team.* into v_team
    from public.tournament_four_player_teams team
   where team.id::text = p_team_id
     and team.tournament_id::text = p_tournament_id
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_NOT_FOUND';
  end if;
  if v_team.status in ('WITHDRAWN', 'CANCELLED') then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_INACTIVE';
  end if;
  select member.* into v_member
    from public.tournament_four_player_team_members member
   where member.id::text = p_member_id
     and member.team_id = v_team.id
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_NOT_FOUND';
  end if;
  if v_member.invitation_version is distinct from p_invitation_version then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_STALE';
  end if;
  if nullif(btrim(p_invitation_token_hash), '') is null
     or v_member.invitation_token_hash is distinct from p_invitation_token_hash then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_TOKEN_INVALID';
  end if;
  select registration.* into v_registration
    from public.tournament_registrations registration
   where registration.id::text = p_registration_id
     and registration.tournament_id::text = p_tournament_id
     and lower(btrim(registration.email)) = lower(btrim(v_member.invited_email))
     and upper(coalesce(registration.status, '')) = 'CONFIRMED'
   for share;
  if not found then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_IDENTITY_MISMATCH';
  end if;
  if v_member.status = 'ACCEPTED' and v_action = 'ACCEPT'
     and v_member.registration_id::text = p_registration_id then
    v_result := jsonb_build_object(
      'ok', true, 'status', 'ACCEPTED', 'idempotent', true,
      'team', jsonb_build_object(
        'id', v_team.id,
        'name', v_team.name,
        'status', v_team.status,
        'version', v_team.version
      ),
      'invitation', jsonb_build_object(
        'member_id', v_member.id,
        'slot', v_member.slot,
        'status', v_member.status,
        'invitation_version', v_member.invitation_version
      )
    );
    return public.complete_tournament_team_operation(
      p_operation_key, p_request_fingerprint, v_result
    );
  end if;
  if v_member.status <> 'INVITED' then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_ALREADY_RESOLVED';
  end if;
  v_before := to_jsonb(v_member);
  if v_action = 'ACCEPT' then
    v_gender := regexp_replace(
      lower(coalesce(v_registration.gender, '')), '[^a-z]', '', 'g'
    );
    if (v_member.slot like 'MAN_%' and v_gender not in (
      'm', 'male', 'man', 'men', 'mens', 'boy', 'boys'
    )) or (v_member.slot like 'WOMAN_%' and v_gender not in (
      'f', 'female', 'woman', 'women', 'womens', 'girl', 'girls'
    )) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_TEAM_MEMBER_GENDER_INVALID';
    end if;
    if v_registration.player_id is not null
       and not exists (
         select 1 from public.players player
          where player.id = v_registration.player_id
            and player.club_id::text = p_club_id
       ) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_TEAM_MEMBER_PLAYER_LINK_INVALID';
    end if;
  end if;

  update public.tournament_four_player_team_members member
     set registration_id = v_registration.id,
         player_id = v_registration.player_id,
         display_name_snapshot = coalesce(
           nullif(v_registration.display_name, ''),
           nullif(btrim(concat_ws(
             ' ', v_registration.first_name, v_registration.last_name
           )), ''),
           member.display_name_snapshot
         ),
         gender_snapshot = v_registration.gender,
         status = case when v_action = 'ACCEPT' then 'ACCEPTED' else 'DECLINED' end,
         invitation_token_hash = null,
         accepted_at = case when v_action = 'ACCEPT'
           then clock_timestamp() else null end,
         declined_at = case when v_action = 'DECLINE'
           then clock_timestamp() else null end
   where member.id = v_member.id
  returning * into v_member;

  select count(*) filter (where member.status = 'ACCEPTED'),
         count(*) filter (
           where member.status = 'ACCEPTED' and member.player_id is not null
         )
    into v_accepted, v_linked
    from public.tournament_four_player_team_members member
   where member.team_id = v_team.id;
  update public.tournament_four_player_teams team
     set status = case
       when v_action = 'DECLINE' then 'FORMING'
       when v_accepted = 4 and v_linked = 4 then 'CONFIRMED'
       when v_accepted = 4 then 'REVIEW_REQUIRED'
       else 'FORMING'
     end,
         version = team.version + 1
   where team.id = v_team.id
  returning * into v_team;

  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json, request_fingerprint
  ) values (
    p_club_id, v_team.tournament_id, v_team.event_option_id,
    'tournament_four_player_team_member', v_member.id::text,
    'four_player_invite_' || lower(v_action),
    coalesce(nullif(p_actor, ''), lower(v_member.invited_email)),
    v_before,
    jsonb_build_object('team', to_jsonb(v_team), 'member', to_jsonb(v_member)),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'status', v_member.status,
    'team', jsonb_build_object(
      'id', v_team.id,
      'name', v_team.name,
      'status', v_team.status,
      'version', v_team.version
    ),
    'invitation', jsonb_build_object(
      'member_id', v_member.id,
      'slot', v_member.slot,
      'status', v_member.status,
      'invitation_version', v_member.invitation_version
    )
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.server_respond_tournament_four_player_invite(
  text, text, text, text, text, integer, text, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.server_respond_tournament_four_player_invite(
  text, text, text, text, text, integer, text, text, text, text, text
) to service_role;

create or replace function public.admin_reissue_tournament_four_player_invite_cas(
  p_club_id text,
  p_tournament_id text,
  p_team_id text,
  p_member_id text,
  p_expected_invitation_version integer,
  p_invited_email text,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_team public.tournament_four_player_teams%rowtype;
  v_member public.tournament_four_player_team_members%rowtype;
  v_pending record;
  v_before jsonb;
  v_recovered_delivery_ids jsonb := '[]'::jsonb;
  v_recovered_operation_keys jsonb := '[]'::jsonb;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'registration', 'four_player_invite_reissue',
    'tournament_four_player_team_member', p_member_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if nullif(lower(btrim(p_invited_email)), '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_EMAIL_REQUIRED';
  end if;
  select team.* into v_team
    from public.tournament_four_player_teams team
    join public.tournaments tournament on tournament.id = team.tournament_id
   where team.id::text = p_team_id
     and team.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update of team;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_NOT_FOUND';
  end if;
  select member.* into v_member
    from public.tournament_four_player_team_members member
   where member.id::text = p_member_id
     and member.team_id = v_team.id
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_NOT_FOUND';
  end if;
  if v_member.invitation_version is distinct from p_expected_invitation_version then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_STALE';
  end if;
  if v_member.status = 'REMOVED' then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_REMOVED';
  end if;
  if v_member.registration_id::text = v_team.captain_registration_id then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_CAPTAIN_INVITATION_IMMUTABLE';
  end if;
  if exists (
    select 1 from public.tournament_team_matchups matchup
     where matchup.team_a_id = v_team.id or matchup.team_b_id = v_team.id
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_ROSTER_IN_USE';
  end if;
  v_before := to_jsonb(v_member);
  -- An interrupted email attempt must never be silently resent. Reissuing the
  -- invitation explicitly invalidates each prior pending delivery and closes
  -- its durable operation before a new invitation version is created.
  for v_pending in
    select delivery.*
      from public.tournament_team_invitation_deliveries delivery
     where delivery.member_id = v_member.id
       and delivery.status = 'pending'
     order by delivery.created_at, delivery.id
     for update
  loop
    update public.tournament_team_invitation_deliveries delivery
       set status = 'skipped',
           provider_message_id = coalesce(
             nullif(delivery.provider_message_id, ''),
             'superseded_by_invitation_reissue'
           )
     where delivery.id = v_pending.id;
    update public.tournament_team_operations operation
       set status = 'COMPLETED',
           result_json = jsonb_build_object(
             'ok', false,
             'status', 'skipped',
             'delivery_id', v_pending.id,
             'superseded_by_reissue', true
           ),
           error_text = null,
           updated_at = clock_timestamp(),
           completed_at = clock_timestamp()
     where operation.operation_key = v_pending.operation_key
       and operation.status in ('INTENT', 'RECOVERY_REQUIRED');
    v_recovered_delivery_ids := v_recovered_delivery_ids
      || jsonb_build_array(v_pending.id);
    v_recovered_operation_keys := v_recovered_operation_keys
      || jsonb_build_array(v_pending.operation_key);
  end loop;
  update public.tournament_four_player_team_members member
     set invited_email = lower(btrim(p_invited_email)),
         registration_id = null,
         player_id = null,
         display_name_snapshot = null,
         gender_snapshot = null,
         status = 'INVITED',
         invitation_version = member.invitation_version + 1,
         invitation_token_hash = null,
         invited_at = clock_timestamp(),
         accepted_at = null,
         declined_at = null
   where member.id = v_member.id
  returning * into v_member;
  update public.tournament_four_player_teams team
     set status = 'FORMING',
         version = team.version + 1
   where team.id = v_team.id
  returning * into v_team;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json, request_fingerprint
  ) values (
    p_club_id, v_team.tournament_id, v_team.event_option_id,
    'tournament_four_player_team_member', v_member.id::text,
    'four_player_invite_reissued',
    coalesce(nullif(p_actor, ''), 'unknown'),
    v_before,
    jsonb_build_object(
      'member', to_jsonb(v_member),
      'recovered_delivery_ids', v_recovered_delivery_ids,
      'recovered_operation_keys', v_recovered_operation_keys
    ),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'team', to_jsonb(v_team), 'member', to_jsonb(v_member),
    'recovered_delivery_ids', v_recovered_delivery_ids,
    'recovered_operation_keys', v_recovered_operation_keys,
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_reissue_tournament_four_player_invite_cas(
  text, text, text, text, integer, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_reissue_tournament_four_player_invite_cas(
  text, text, text, text, integer, text, text, text, text
) to service_role;

create or replace function public.materialize_tournament_team_rating_asset(
  p_match_game_id uuid
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_child public.tournament_team_match_games%rowtype;
  v_matchup public.tournament_team_matchups%rowtype;
  v_parent_draw public.tournament_event_draws%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_team_a public.tournament_teams%rowtype;
  v_team_b public.tournament_teams%rowtype;
  v_game public.tournament_games%rowtype;
  v_name text;
begin
  select child.* into v_child
    from public.tournament_team_match_games child
   where child.id = p_match_game_id
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_MATCH_GAME_NOT_FOUND';
  end if;
  if not v_child.counts_for_rating then
    return jsonb_build_object(
      'ok', true, 'counts_for_rating', false, 'match_game_id', v_child.id
    );
  end if;
  if v_child.rating_draw_id is not null and v_child.tournament_game_id is not null then
    return jsonb_build_object(
      'ok', true, 'counts_for_rating', true,
      'draw_id', v_child.rating_draw_id,
      'tournament_game_id', v_child.tournament_game_id
    );
  end if;
  if (
    v_child.match_format = 'DOUBLES'
    and (
      cardinality(v_child.team_a_player_ids) <> 2
      or cardinality(v_child.team_b_player_ids) <> 2
    )
  ) or (
    v_child.match_format = 'SINGLES'
    and (
      cardinality(v_child.team_a_player_ids) <> 1
      or cardinality(v_child.team_b_player_ids) <> 1
    )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_RATING_LINEUP_INVALID';
  end if;
  select matchup.* into v_matchup
    from public.tournament_team_matchups matchup
   where matchup.id = v_child.matchup_id
   for share;
  select draw.* into v_parent_draw
    from public.tournament_event_draws draw
   where draw.id = v_matchup.draw_id
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_PARENT_DRAW_NOT_FOUND';
  end if;
  v_name := left(
    v_parent_draw.name || ' · rated ' ||
    left(v_matchup.id::text, 8) || ' ' || lower(v_child.game_code),
    240
  );
  insert into public.tournament_event_draws (
    tournament_id, registration_day_id, event_option_id, name, status,
    draw_kind, parent_draw_id, hidden_from_primary_ops
  ) values (
    v_child.tournament_id, v_parent_draw.registration_day_id,
    v_matchup.event_option_id, v_name, 'draft',
    'TEAM_RATING_CHILD', v_parent_draw.id, true
  )
  returning * into v_draw;

  insert into public.tournament_teams (
    tournament_id, draw_id, registration_day_id, event_option_id,
    team_number, player1_id, player2_id, source, notes, team_match_game_id,
    team_match_side
  ) values (
    v_child.tournament_id, v_draw.id, v_parent_draw.registration_day_id,
    v_matchup.event_option_id, 1, v_child.team_a_player_ids[1],
    case when v_child.match_format = 'DOUBLES'
      then v_child.team_a_player_ids[2] else null end,
    'FOUR_PLAYER_TEAM_CHILD', v_child.game_code, v_child.id, 'A'
  )
  returning * into v_team_a;
  insert into public.tournament_teams (
    tournament_id, draw_id, registration_day_id, event_option_id,
    team_number, player1_id, player2_id, source, notes, team_match_game_id,
    team_match_side
  ) values (
    v_child.tournament_id, v_draw.id, v_parent_draw.registration_day_id,
    v_matchup.event_option_id, 2, v_child.team_b_player_ids[1],
    case when v_child.match_format = 'DOUBLES'
      then v_child.team_b_player_ids[2] else null end,
    'FOUR_PLAYER_TEAM_CHILD', v_child.game_code, v_child.id, 'B'
  )
  returning * into v_team_b;
  insert into public.tournament_games (
    tournament_id, draw_id, registration_day_id, event_option_id,
    stage, rr_round_number, rr_slot_number, playoff_game_code,
    playoff_round, team_a_id, team_b_id, team_match_game_id,
    parent_result_only
  ) values (
    v_child.tournament_id, v_draw.id, v_parent_draw.registration_day_id,
    v_matchup.event_option_id,
    case when v_matchup.stage = 'PLAYOFF' then 'PLAYOFF' else 'ROUND_ROBIN' end,
    case when v_matchup.stage = 'ROUND_ROBIN' then 1 else null end,
    case when v_matchup.stage = 'ROUND_ROBIN' then 1 else null end,
    case when v_matchup.stage = 'PLAYOFF'
      then coalesce(v_matchup.playoff_game_code, 'PLAYOFF') || '-' || v_child.game_code
      else null end,
    case when v_matchup.stage = 'PLAYOFF'
      then v_matchup.playoff_game_code else null end,
    v_team_a.id, v_team_b.id, v_child.id, false
  )
  returning * into v_game;
  update public.tournament_team_match_games child
     set rating_draw_id = v_draw.id,
         tournament_game_id = v_game.id
   where child.id = v_child.id
  returning * into v_child;
  return jsonb_build_object(
    'ok', true, 'counts_for_rating', true,
    'draw_id', v_draw.id, 'tournament_game_id', v_game.id,
    'team_a_id', v_team_a.id, 'team_b_id', v_team_b.id
  );
end;
$$;

revoke all on function public.materialize_tournament_team_rating_asset(uuid)
  from public, anon, authenticated;
grant execute on function public.materialize_tournament_team_rating_asset(uuid)
  to service_role;

create or replace function public.lock_tournament_team_draw(p_draw_id uuid)
returns void
language plpgsql
security invoker
set search_path = ''
as $$
begin
  if p_draw_id is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_DRAW_LOCK_ID_REQUIRED';
  end if;
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'jupr:tournament-team-draw:' || p_draw_id::text,
      0
    )
  );
end;
$$;

revoke all on function public.lock_tournament_team_draw(uuid)
  from public, anon, authenticated;
grant execute on function public.lock_tournament_team_draw(uuid)
  to service_role;

create or replace function public.admin_replace_tournament_team_matchups_cas(
  p_club_id text,
  p_tournament_id text,
  p_event_option_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_matchups jsonb,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_authorized_draw_id uuid;
  v_event public.tournament_event_options%rowtype;
  v_saved jsonb;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'operations', 'team_matchups_replace', 'tournament_event_draw',
    p_draw_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if pg_catalog.jsonb_typeof(coalesce(p_matchups, 'null'::jsonb)) <> 'array' then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_MATCHUPS_INVALID';
  end if;
  select draw.id into v_authorized_draw_id
    from public.tournament_event_draws draw
    join public.tournaments tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_DRAW_NOT_FOUND';
  end if;
  perform public.lock_tournament_team_draw(v_authorized_draw_id);
  select draw.* into v_draw
    from public.tournament_event_draws draw
    join public.tournaments tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update of draw;
  if not found or p_expected_draw_updated_at is null
     or v_draw.updated_at is distinct from p_expected_draw_updated_at then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_DRAW_STALE';
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = p_event_option_id
     and event.tournament_id::text = p_tournament_id
     and event.competition_format = 'FOUR_PLAYER_TEAM'
   for share;
  if not found or v_draw.event_option_id::text is distinct from p_event_option_id then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_DRAW_REQUIRED';
  end if;
  perform team.id
    from public.tournament_four_player_teams team
    join (
      select nullif(x.team_a_id, '') as team_id
        from pg_catalog.jsonb_to_recordset(p_matchups) as x(
          team_a_id text, team_b_id text
        )
      union
      select nullif(x.team_b_id, '') as team_id
        from pg_catalog.jsonb_to_recordset(p_matchups) as x(
          team_a_id text, team_b_id text
        )
    ) selected on selected.team_id = team.id::text
   order by team.id
   for update of team;
  if exists (
    select 1
      from public.tournament_team_matchups matchup
      join public.tournament_team_lineup_submissions lineup
        on lineup.matchup_id = matchup.id
     where matchup.draw_id = v_draw.id
  ) or exists (
    select 1
      from public.tournament_team_matchups matchup
      join public.tournament_team_match_games child
        on child.matchup_id = matchup.id
     where matchup.draw_id = v_draw.id
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_SCHEDULE_IN_USE';
  end if;
  if exists (
    select 1
      from pg_catalog.jsonb_to_recordset(p_matchups) as x(
        stage text, team_a_id text, team_b_id text
      )
     where upper(coalesce(x.stage, '')) not in ('ROUND_ROBIN', 'PLAYOFF')
        or (
          x.team_a_id is not null and not exists (
            select 1 from public.tournament_four_player_teams team
             where team.id::text = x.team_a_id
               and team.tournament_id = v_draw.tournament_id
               and team.event_option_id::text = p_event_option_id
               and team.status = 'CONFIRMED'
               and team.eligibility_state in ('ELIGIBLE', 'NOT_REQUIRED')
               and (team.draw_id is null or team.draw_id = v_draw.id)
               and not exists (
                 select 1
                   from public.tournament_team_matchups assigned
                  where assigned.draw_id <> v_draw.id
                    and team.id in (assigned.team_a_id, assigned.team_b_id)
               )
          )
        )
        or (
          x.team_b_id is not null and not exists (
            select 1 from public.tournament_four_player_teams team
             where team.id::text = x.team_b_id
               and team.tournament_id = v_draw.tournament_id
               and team.event_option_id::text = p_event_option_id
               and team.status = 'CONFIRMED'
               and team.eligibility_state in ('ELIGIBLE', 'NOT_REQUIRED')
               and (team.draw_id is null or team.draw_id = v_draw.id)
               and not exists (
                 select 1
                   from public.tournament_team_matchups assigned
                  where assigned.draw_id <> v_draw.id
                    and team.id in (assigned.team_a_id, assigned.team_b_id)
               )
          )
        )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_SCHEDULE_TEAM_INVALID';
  end if;
  delete from public.tournament_team_matchups matchup
   where matchup.draw_id = v_draw.id;
  with inserted as (
    insert into public.tournament_team_matchups (
      tournament_id, event_option_id, draw_id, stage, round_number,
      slot_number, playoff_game_code, team_a_id, team_b_id,
      team_a_source, team_b_source, tiebreak_mode
    )
    select v_draw.tournament_id, p_event_option_id, v_draw.id,
           upper(x.stage), x.round_number, x.slot_number,
           nullif(upper(x.playoff_game_code), ''),
           nullif(x.team_a_id, '')::uuid,
           nullif(x.team_b_id, '')::uuid,
           x.team_a_source, x.team_b_source, v_event.team_tiebreak_mode
      from pg_catalog.jsonb_to_recordset(p_matchups) as x(
        stage text, round_number integer, slot_number integer,
        playoff_game_code text, team_a_id text, team_b_id text,
        team_a_source jsonb, team_b_source jsonb
      )
     order by upper(x.stage), x.round_number, x.slot_number, x.playoff_game_code
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted)), '[]'::jsonb)
    into v_saved from inserted;
  update public.tournament_event_draws draw
     set draw_kind = 'TEAM_PARENT',
         hidden_from_primary_ops = true,
         updated_at = clock_timestamp()
   where draw.id = v_draw.id;
  update public.tournament_four_player_teams team
     set draw_id = v_draw.id
   where team.tournament_id = v_draw.tournament_id
     and team.event_option_id::text = p_event_option_id
     and exists (
       select 1 from public.tournament_team_matchups matchup
        where matchup.draw_id = v_draw.id
          and team.id in (matchup.team_a_id, matchup.team_b_id)
     );
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, after_json, request_fingerprint
  ) values (
    p_club_id, v_draw.tournament_id, p_event_option_id,
    'tournament_event_draw', v_draw.id::text, 'team_matchups_replaced',
    coalesce(nullif(p_actor, ''), 'unknown'),
    jsonb_build_object('matchups', v_saved), p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'matchups', v_saved, 'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_replace_tournament_team_matchups_cas(
  text, text, text, text, timestamptz, jsonb, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_replace_tournament_team_matchups_cas(
  text, text, text, text, timestamptz, jsonb, text, text, text
) to service_role;

create or replace function public.admin_lock_tournament_team_lineup_cas(
  p_club_id text,
  p_tournament_id text,
  p_matchup_id text,
  p_team_id text,
  p_mixed_pairing text,
  p_singles_tiebreak_player_id integer,
  p_expected_matchup_version integer,
  p_expected_lineup_version integer,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_matchup public.tournament_team_matchups%rowtype;
  v_authorized_draw_id uuid;
  v_team public.tournament_four_player_teams%rowtype;
  v_existing public.tournament_team_lineup_submissions%rowtype;
  v_saved public.tournament_team_lineup_submissions%rowtype;
  v_lineup_a public.tournament_team_lineup_submissions%rowtype;
  v_lineup_b public.tournament_team_lineup_submissions%rowtype;
  v_a_m1 integer;
  v_a_m2 integer;
  v_a_w1 integer;
  v_a_w2 integer;
  v_b_m1 integer;
  v_b_m2 integer;
  v_b_w1 integer;
  v_b_w2 integer;
  v_lineup_count integer;
  v_games jsonb := '[]'::jsonb;
  v_child record;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'operations', 'team_lineup_lock',
    'tournament_team_matchup', p_matchup_id || ':' || p_team_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if upper(coalesce(p_mixed_pairing, '')) not in ('STRAIGHT', 'CROSS') then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_MIXED_PAIRING_INVALID';
  end if;
  select matchup.draw_id into v_authorized_draw_id
    from public.tournament_team_matchups matchup
    join public.tournaments tournament on tournament.id = matchup.tournament_id
   where matchup.id::text = p_matchup_id
     and matchup.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_MATCHUP_NOT_FOUND';
  end if;
  perform public.lock_tournament_team_draw(v_authorized_draw_id);
  select matchup.* into v_matchup
    from public.tournament_team_matchups matchup
    join public.tournaments tournament on tournament.id = matchup.tournament_id
   where matchup.id::text = p_matchup_id
     and matchup.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update of matchup;
  if not found or p_expected_matchup_version is null
     or v_matchup.version is distinct from p_expected_matchup_version then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_MATCHUP_STALE';
  end if;
  if v_matchup.team_a_id is null or v_matchup.team_b_id is null
     or p_team_id::uuid not in (v_matchup.team_a_id, v_matchup.team_b_id)
     or v_matchup.status in ('FINAL', 'VOID', 'CORRECTION_REQUIRED') then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_LINEUP_LOCK_REFUSED';
  end if;
  select team.* into v_team
    from public.tournament_four_player_teams team
   where team.id::text = p_team_id
     and team.tournament_id = v_matchup.tournament_id
     and team.event_option_id = v_matchup.event_option_id
     and team.status = 'CONFIRMED'
     and team.eligibility_state in ('ELIGIBLE', 'NOT_REQUIRED')
   for share;
  if not found or (
    select count(*) from public.tournament_four_player_team_members member
     where member.team_id = v_team.id
       and member.status = 'ACCEPTED'
       and member.player_id is not null
  ) <> 4 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_ROSTER_NOT_CONFIRMED';
  end if;
  if v_matchup.tiebreak_mode = 'SINGLES' and (
    p_singles_tiebreak_player_id is null
    or not exists (
      select 1 from public.tournament_four_player_team_members member
       where member.team_id = v_team.id
         and member.player_id = p_singles_tiebreak_player_id
         and member.status = 'ACCEPTED'
    )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_SINGLES_TIEBREAKER_INVALID';
  end if;
  select lineup.* into v_existing
    from public.tournament_team_lineup_submissions lineup
   where lineup.matchup_id = v_matchup.id
     and lineup.team_id = v_team.id
   for update;
  if found then
    if p_expected_lineup_version is null
       or v_existing.version is distinct from p_expected_lineup_version then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_TEAM_LINEUP_STALE';
    end if;
    if exists (
      select 1 from public.tournament_team_match_games child
       where child.matchup_id = v_matchup.id
    ) then
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_TEAM_LINEUP_ALREADY_REVEALED';
    end if;
    update public.tournament_team_lineup_submissions lineup
       set mixed_pairing = upper(p_mixed_pairing),
           singles_tiebreak_player_id = p_singles_tiebreak_player_id,
           version = lineup.version + 1,
           locked_by = coalesce(nullif(p_actor, ''), 'unknown'),
           locked_at = clock_timestamp()
     where lineup.id = v_existing.id
    returning * into v_saved;
  else
    if coalesce(p_expected_lineup_version, 0) <> 0 then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_TEAM_LINEUP_STALE';
    end if;
    insert into public.tournament_team_lineup_submissions (
      tournament_id, matchup_id, team_id, mixed_pairing,
      singles_tiebreak_player_id, status, locked_by
    ) values (
      v_matchup.tournament_id, v_matchup.id, v_team.id,
      upper(p_mixed_pairing), p_singles_tiebreak_player_id, 'LOCKED',
      coalesce(nullif(p_actor, ''), 'unknown')
    )
    returning * into v_saved;
  end if;
  select count(*) into v_lineup_count
    from public.tournament_team_lineup_submissions lineup
   where lineup.matchup_id = v_matchup.id
     and lineup.status in ('LOCKED', 'REVEALED');
  if v_lineup_count = 2 then
    select lineup.* into v_lineup_a
      from public.tournament_team_lineup_submissions lineup
     where lineup.matchup_id = v_matchup.id
       and lineup.team_id = v_matchup.team_a_id
     for update;
    select lineup.* into v_lineup_b
      from public.tournament_team_lineup_submissions lineup
     where lineup.matchup_id = v_matchup.id
       and lineup.team_id = v_matchup.team_b_id
     for update;
    select
      max(member.player_id) filter (where member.slot = 'MAN_1'),
      max(member.player_id) filter (where member.slot = 'MAN_2'),
      max(member.player_id) filter (where member.slot = 'WOMAN_1'),
      max(member.player_id) filter (where member.slot = 'WOMAN_2')
      into v_a_m1, v_a_m2, v_a_w1, v_a_w2
      from public.tournament_four_player_team_members member
     where member.team_id = v_matchup.team_a_id and member.status = 'ACCEPTED';
    select
      max(member.player_id) filter (where member.slot = 'MAN_1'),
      max(member.player_id) filter (where member.slot = 'MAN_2'),
      max(member.player_id) filter (where member.slot = 'WOMAN_1'),
      max(member.player_id) filter (where member.slot = 'WOMAN_2')
      into v_b_m1, v_b_m2, v_b_w1, v_b_w2
      from public.tournament_four_player_team_members member
     where member.team_id = v_matchup.team_b_id and member.status = 'ACCEPTED';
    if v_a_m1 is null or v_a_m2 is null or v_a_w1 is null or v_a_w2 is null
       or v_b_m1 is null or v_b_m2 is null or v_b_w1 is null or v_b_w2 is null then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_TEAM_ROSTER_NOT_CONFIRMED';
    end if;
    insert into public.tournament_team_match_games (
      tournament_id, matchup_id, game_code, game_order, match_format,
      counts_for_rating, team_a_player_ids, team_b_player_ids
    ) values
      (
        v_matchup.tournament_id, v_matchup.id, 'WOMENS', 1, 'DOUBLES', true,
        array[v_a_w1, v_a_w2], array[v_b_w1, v_b_w2]
      ),
      (
        v_matchup.tournament_id, v_matchup.id, 'MENS', 2, 'DOUBLES', true,
        array[v_a_m1, v_a_m2], array[v_b_m1, v_b_m2]
      ),
      (
        v_matchup.tournament_id, v_matchup.id, 'MIXED_1', 3, 'DOUBLES', true,
        case when v_lineup_a.mixed_pairing = 'STRAIGHT'
          then array[v_a_m1, v_a_w1] else array[v_a_m1, v_a_w2] end,
        case when v_lineup_b.mixed_pairing = 'STRAIGHT'
          then array[v_b_m1, v_b_w1] else array[v_b_m1, v_b_w2] end
      ),
      (
        v_matchup.tournament_id, v_matchup.id, 'MIXED_2', 4, 'DOUBLES', true,
        case when v_lineup_a.mixed_pairing = 'STRAIGHT'
          then array[v_a_m2, v_a_w2] else array[v_a_m2, v_a_w1] end,
        case when v_lineup_b.mixed_pairing = 'STRAIGHT'
          then array[v_b_m2, v_b_w2] else array[v_b_m2, v_b_w1] end
      );
    update public.tournament_team_lineup_submissions lineup
       set status = 'REVEALED', revealed_at = clock_timestamp()
     where lineup.matchup_id = v_matchup.id;
    for v_child in
      select child.id from public.tournament_team_match_games child
       where child.matchup_id = v_matchup.id
       order by child.game_order
    loop
      perform public.materialize_tournament_team_rating_asset(v_child.id);
    end loop;
    update public.tournament_team_matchups matchup
       set status = 'READY', version = matchup.version + 1
     where matchup.id = v_matchup.id
    returning * into v_matchup;
    select coalesce(jsonb_agg(to_jsonb(child) order by child.game_order), '[]'::jsonb)
      into v_games
      from public.tournament_team_match_games child
     where child.matchup_id = v_matchup.id;
  else
    update public.tournament_team_matchups matchup
       set status = 'LINEUPS_PENDING', version = matchup.version + 1
     where matchup.id = v_matchup.id
    returning * into v_matchup;
  end if;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json, request_fingerprint
  ) values (
    p_club_id, v_matchup.tournament_id, v_matchup.event_option_id,
    'tournament_team_lineup_submission', v_saved.id::text,
    case when v_lineup_count = 2
      then 'team_lineups_revealed' else 'team_lineup_locked' end,
    coalesce(nullif(p_actor, ''), 'unknown'),
    case when v_existing.id is null then null else to_jsonb(v_existing) end,
    jsonb_build_object(
      'lineup', to_jsonb(v_saved),
      'lineups_revealed', v_lineup_count = 2,
      'games', case when v_lineup_count = 2 then v_games else '[]'::jsonb end
    ),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'lineup', to_jsonb(v_saved),
    'lineups_revealed', v_lineup_count = 2,
    'matchup', to_jsonb(v_matchup),
    'games', case when v_lineup_count = 2 then v_games else '[]'::jsonb end,
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_lock_tournament_team_lineup_cas(
  text, text, text, text, text, integer, integer, integer,
  text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_lock_tournament_team_lineup_cas(
  text, text, text, text, text, integer, integer, integer,
  text, text, text
) to service_role;

create or replace function public.resolve_tournament_team_playoff_dependencies(
  p_draw_id uuid
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_target record;
  v_source_a public.tournament_team_matchups%rowtype;
  v_source_b public.tournament_team_matchups%rowtype;
  v_team_a uuid;
  v_team_b uuid;
  v_updated jsonb;
  v_updates jsonb := '[]'::jsonb;
begin
  for v_target in
    select matchup.*
      from public.tournament_team_matchups matchup
     where matchup.draw_id = p_draw_id
       and matchup.stage = 'PLAYOFF'
       and matchup.team_a_source->>'type' in ('WINNER', 'LOSER')
       and matchup.team_b_source->>'type' in ('WINNER', 'LOSER')
     order by matchup.playoff_game_code
     for update
  loop
    select source.* into v_source_a
      from public.tournament_team_matchups source
     where source.draw_id = p_draw_id
       and source.playoff_game_code = v_target.team_a_source->>'game_code';
    select source.* into v_source_b
      from public.tournament_team_matchups source
     where source.draw_id = p_draw_id
       and source.playoff_game_code = v_target.team_b_source->>'game_code';
    v_team_a := case v_target.team_a_source->>'type'
      when 'WINNER' then v_source_a.winner_team_id
      when 'LOSER' then v_source_a.loser_team_id
      else null end;
    v_team_b := case v_target.team_b_source->>'type'
      when 'WINNER' then v_source_b.winner_team_id
      when 'LOSER' then v_source_b.loser_team_id
      else null end;
    if exists (
      select 1 from public.tournament_team_lineup_submissions lineup
       where lineup.matchup_id = v_target.id
    ) or exists (
      select 1 from public.tournament_team_match_games child
       where child.matchup_id = v_target.id
    ) then
      if v_target.team_a_id is distinct from v_team_a
         or v_target.team_b_id is distinct from v_team_b then
        raise exception using errcode = 'P0001',
          message = 'JUPR_TOURNAMENT_TEAM_PLAYOFF_DEPENDENCY_LOCKED';
      end if;
      continue;
    end if;
    update public.tournament_team_matchups matchup
       set team_a_id = case
             when v_team_a is not null and v_team_b is not null then v_team_a
             else null
           end,
           team_b_id = case
             when v_team_a is not null and v_team_b is not null then v_team_b
             else null
           end,
           status = 'LINEUPS_PENDING',
           version = matchup.version + 1
     where matchup.id = v_target.id
       and (
         matchup.team_a_id is distinct from case
           when v_team_a is not null and v_team_b is not null then v_team_a
           else null end
         or matchup.team_b_id is distinct from case
           when v_team_a is not null and v_team_b is not null then v_team_b
           else null end
       )
    returning to_jsonb(matchup) into v_updated;
    if v_updated is not null then
      v_updates := v_updates || jsonb_build_array(v_updated);
    end if;
  end loop;
  return v_updates;
end;
$$;

revoke all on function public.resolve_tournament_team_playoff_dependencies(uuid)
  from public, anon, authenticated;
grant execute on function public.resolve_tournament_team_playoff_dependencies(uuid)
  to service_role;

create or replace function public.admin_score_tournament_team_match_game_cas(
  p_club_id text,
  p_tournament_id text,
  p_match_game_id text,
  p_score_a integer,
  p_score_b integer,
  p_expected_game_version integer,
  p_expected_matchup_version integer,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_child public.tournament_team_match_games%rowtype;
  v_authorized_draw_id uuid;
  v_before_child jsonb;
  v_matchup public.tournament_team_matchups%rowtype;
  v_rating_game public.tournament_games%rowtype;
  v_tiebreak public.tournament_team_match_games%rowtype;
  v_lineup_a public.tournament_team_lineup_submissions%rowtype;
  v_lineup_b public.tournament_team_lineup_submissions%rowtype;
  v_reg_complete integer;
  v_reg_a integer;
  v_reg_b integer;
  v_tiebreak_side text;
  v_total_a integer;
  v_total_b integer;
  v_winner uuid;
  v_loser uuid;
  v_status text;
  v_games jsonb;
  v_dependencies jsonb := '[]'::jsonb;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'operations', 'team_match_game_score',
    'tournament_team_match_game', p_match_game_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if p_score_a is null or p_score_b is null
     or p_score_a < 0 or p_score_b < 0 or p_score_a = p_score_b then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_SCORE_INVALID';
  end if;
  select matchup.draw_id into v_authorized_draw_id
    from public.tournament_team_match_games child
    join public.tournament_team_matchups matchup on matchup.id = child.matchup_id
    join public.tournaments tournament on tournament.id = child.tournament_id
   where child.id::text = p_match_game_id
     and child.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_MATCH_GAME_NOT_FOUND';
  end if;
  perform public.lock_tournament_team_draw(v_authorized_draw_id);
  select child.* into v_child
    from public.tournament_team_match_games child
    join public.tournament_team_matchups matchup on matchup.id = child.matchup_id
    join public.tournaments tournament on tournament.id = child.tournament_id
   where child.id::text = p_match_game_id
     and child.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update of child;
  if not found or p_expected_game_version is null
     or v_child.version is distinct from p_expected_game_version then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_MATCH_GAME_STALE';
  end if;
  select matchup.* into v_matchup
    from public.tournament_team_matchups matchup
   where matchup.id = v_child.matchup_id
   for update;
  if p_expected_matchup_version is null
     or v_matchup.version is distinct from p_expected_matchup_version
     or v_matchup.status in ('VOID', 'CORRECTION_REQUIRED') then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_MATCHUP_STALE';
  end if;
  if v_child.status = 'VOID' and v_child.game_code <> 'TIEBREAK' then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_MATCH_GAME_VOID';
  end if;
  if v_matchup.stage = 'ROUND_ROBIN' and exists (
    select 1
      from public.tournament_team_matchups playoff
     where playoff.draw_id = v_matchup.draw_id
       and playoff.stage = 'PLAYOFF'
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_ROUND_ROBIN_LOCKED_BY_PLAYOFFS';
  end if;
  if v_child.tournament_game_id is not null and exists (
    select 1 from public.matches official
     where official.tournament_id::text = p_tournament_id
       and official.tournament_game_id = v_child.tournament_game_id
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_GAME_ALREADY_PUBLISHED';
  end if;
  if v_child.score_a is not null and exists (
    select 1
      from public.tournament_team_matchups target
     where target.draw_id = v_matchup.draw_id
       and target.stage = 'PLAYOFF'
       and (
         target.team_a_source->>'game_code' = v_matchup.playoff_game_code
         or target.team_b_source->>'game_code' = v_matchup.playoff_game_code
       )
       and (
         exists (
           select 1 from public.tournament_team_lineup_submissions lineup
            where lineup.matchup_id = target.id
         )
         or exists (
           select 1 from public.tournament_team_match_games child
            where child.matchup_id = target.id
         )
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_PLAYOFF_DEPENDENCY_LOCKED';
  end if;
  v_before_child := to_jsonb(v_child);
  v_winner := case when p_score_a > p_score_b
    then v_matchup.team_a_id else v_matchup.team_b_id end;
  v_loser := case when p_score_a > p_score_b
    then v_matchup.team_b_id else v_matchup.team_a_id end;
  update public.tournament_team_match_games child
     set score_a = p_score_a,
         score_b = p_score_b,
         winner_team_id = v_winner,
         loser_team_id = v_loser,
         status = 'FINAL',
         finalized_at = clock_timestamp(),
         version = child.version + 1
   where child.id = v_child.id
  returning * into v_child;
  if v_child.tournament_game_id is not null then
    select game.* into v_rating_game
      from public.tournament_games game
     where game.id = v_child.tournament_game_id
     for update;
    update public.tournament_games game
       set score_a = p_score_a,
           score_b = p_score_b,
           winner_team_id = case when p_score_a > p_score_b
             then v_rating_game.team_a_id else v_rating_game.team_b_id end,
           loser_team_id = case when p_score_a > p_score_b
             then v_rating_game.team_b_id else v_rating_game.team_a_id end,
           finalized_at = clock_timestamp(),
           updated_at = clock_timestamp()
     where game.id = v_rating_game.id;
  end if;
  select count(*) filter (
           where child.game_code <> 'TIEBREAK' and child.status = 'FINAL'
         ),
         count(*) filter (
           where child.game_code <> 'TIEBREAK'
             and child.status = 'FINAL'
             and child.winner_team_id = v_matchup.team_a_id
         ),
         count(*) filter (
           where child.game_code <> 'TIEBREAK'
             and child.status = 'FINAL'
             and child.winner_team_id = v_matchup.team_b_id
         )
    into v_reg_complete, v_reg_a, v_reg_b
    from public.tournament_team_match_games child
   where child.matchup_id = v_matchup.id;
  select child.* into v_tiebreak
    from public.tournament_team_match_games child
   where child.matchup_id = v_matchup.id
     and child.game_code = 'TIEBREAK'
   for update;
  if v_reg_complete = 4 and v_reg_a = 2 and v_reg_b = 2
     and v_tiebreak.id is null then
    select lineup.* into v_lineup_a
      from public.tournament_team_lineup_submissions lineup
     where lineup.matchup_id = v_matchup.id
       and lineup.team_id = v_matchup.team_a_id;
    select lineup.* into v_lineup_b
      from public.tournament_team_lineup_submissions lineup
     where lineup.matchup_id = v_matchup.id
       and lineup.team_id = v_matchup.team_b_id;
    if v_matchup.tiebreak_mode = 'SINGLES' then
      insert into public.tournament_team_match_games (
        tournament_id, matchup_id, game_code, game_order, match_format,
        counts_for_rating, team_a_player_ids, team_b_player_ids
      ) values (
        v_matchup.tournament_id, v_matchup.id, 'TIEBREAK', 5, 'SINGLES',
        true, array[v_lineup_a.singles_tiebreak_player_id],
        array[v_lineup_b.singles_tiebreak_player_id]
      )
      returning * into v_tiebreak;
      perform public.materialize_tournament_team_rating_asset(v_tiebreak.id);
    else
      insert into public.tournament_team_match_games (
        tournament_id, matchup_id, game_code, game_order, match_format,
        counts_for_rating, team_a_player_ids, team_b_player_ids
      )
      select v_matchup.tournament_id, v_matchup.id, 'TIEBREAK', 5,
             'SKINNY_RELAY', false,
             array_agg(member.player_id order by
               case member.slot
                 when 'WOMAN_1' then 1 when 'WOMAN_2' then 2
                 when 'MAN_1' then 3 else 4 end
             ) filter (where member.team_id = v_matchup.team_a_id),
             array_agg(member.player_id order by
               case member.slot
                 when 'WOMAN_1' then 1 when 'WOMAN_2' then 2
                 when 'MAN_1' then 3 else 4 end
             ) filter (where member.team_id = v_matchup.team_b_id)
        from public.tournament_four_player_team_members member
       where member.team_id in (v_matchup.team_a_id, v_matchup.team_b_id)
         and member.status = 'ACCEPTED'
      returning * into v_tiebreak;
    end if;
  elsif v_reg_complete = 4 and v_reg_a <> v_reg_b
        and v_tiebreak.id is not null then
    if v_tiebreak.tournament_game_id is not null and exists (
      select 1 from public.matches official
       where official.tournament_game_id = v_tiebreak.tournament_game_id
    ) then
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_TEAM_TIEBREAK_ALREADY_PUBLISHED';
    end if;
    update public.tournament_team_match_games child
       set score_a = null, score_b = null, winner_team_id = null,
           loser_team_id = null, status = 'VOID', finalized_at = null,
           version = child.version + 1
     where child.id = v_tiebreak.id
    returning * into v_tiebreak;
    update public.tournament_games game
       set score_a = null, score_b = null, winner_team_id = null,
           loser_team_id = null, finalized_at = null,
           updated_at = clock_timestamp()
     where game.id = v_tiebreak.tournament_game_id;
  elsif v_reg_complete = 4 and v_reg_a = 2 and v_reg_b = 2
        and v_tiebreak.id is not null and v_tiebreak.status = 'VOID' then
    update public.tournament_team_match_games child
       set status = 'SCHEDULED', version = child.version + 1
     where child.id = v_tiebreak.id
    returning * into v_tiebreak;
  end if;
  v_tiebreak_side := case
    when v_tiebreak.status = 'FINAL'
      and v_tiebreak.winner_team_id = v_matchup.team_a_id then 'A'
    when v_tiebreak.status = 'FINAL'
      and v_tiebreak.winner_team_id = v_matchup.team_b_id then 'B'
    else null end;
  v_total_a := v_reg_a + case when v_tiebreak_side = 'A' then 1 else 0 end;
  v_total_b := v_reg_b + case when v_tiebreak_side = 'B' then 1 else 0 end;
  if v_reg_complete < 4 then
    v_status := 'IN_PROGRESS';
    v_winner := null;
    v_loser := null;
  elsif v_reg_a = v_reg_b and v_tiebreak_side is null then
    v_status := 'TIEBREAK_REQUIRED';
    v_winner := null;
    v_loser := null;
  else
    v_status := 'FINAL';
    v_winner := case when v_total_a > v_total_b
      then v_matchup.team_a_id else v_matchup.team_b_id end;
    v_loser := case when v_total_a > v_total_b
      then v_matchup.team_b_id else v_matchup.team_a_id end;
  end if;
  update public.tournament_team_matchups matchup
     set status = v_status,
         team_a_game_wins = v_total_a,
         team_b_game_wins = v_total_b,
         winner_team_id = v_winner,
         loser_team_id = v_loser,
         finalized_at = case when v_status = 'FINAL'
           then clock_timestamp() else null end,
         version = matchup.version + 1
   where matchup.id = v_matchup.id
  returning * into v_matchup;
  if v_matchup.stage = 'PLAYOFF' then
    v_dependencies := public.resolve_tournament_team_playoff_dependencies(
      v_matchup.draw_id
    );
  end if;
  select coalesce(jsonb_agg(to_jsonb(child) order by child.game_order), '[]'::jsonb)
    into v_games
    from public.tournament_team_match_games child
   where child.matchup_id = v_matchup.id;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json, request_fingerprint
  ) values (
    p_club_id, v_matchup.tournament_id, v_matchup.event_option_id,
    'tournament_team_match_game', v_child.id::text,
    case when (v_before_child->>'score_a') is null
      then 'team_match_game_scored' else 'team_match_game_corrected' end,
    coalesce(nullif(p_actor, ''), 'unknown'), v_before_child,
    jsonb_build_object(
      'game', to_jsonb(v_child), 'matchup', to_jsonb(v_matchup),
      'playoff_dependency_updates', v_dependencies
    ),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'matchup', to_jsonb(v_matchup), 'games', v_games,
    'playoff_dependency_updates', v_dependencies,
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_score_tournament_team_match_game_cas(
  text, text, text, integer, integer, integer, integer, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_score_tournament_team_match_game_cas(
  text, text, text, integer, integer, integer, integer, text, text, text
) to service_role;

create or replace function public.server_claim_tournament_team_invitation_delivery(
  p_club_id text,
  p_tournament_id text,
  p_team_id text,
  p_member_id text,
  p_invitation_version integer,
  p_email_mode text,
  p_recipient_email_hash text,
  p_invitation_token_hash text,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_team public.tournament_four_player_teams%rowtype;
  v_member public.tournament_four_player_team_members%rowtype;
  v_operation public.tournament_team_operations%rowtype;
  v_delivery public.tournament_team_invitation_deliveries%rowtype;
begin
  if lower(coalesce(p_email_mode, '')) not in (
    'dry_run', 'staging_redirect', 'live'
  ) or nullif(btrim(p_recipient_email_hash), '') is null
     or nullif(btrim(p_invitation_token_hash), '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_DELIVERY_INVALID';
  end if;
  select operation.* into v_operation
    from public.tournament_team_operations operation
   where operation.operation_key = p_operation_key
   for update;
  if found then
    if v_operation.request_fingerprint <> p_request_fingerprint
       or v_operation.club_id <> p_club_id
       or v_operation.tournament_id::text <> p_tournament_id
       or v_operation.action <> 'team_invitation_delivery' then
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_TEAM_OPERATION_KEY_REUSED';
    end if;
    if v_operation.status = 'COMPLETED' then
      return v_operation.result_json;
    end if;
    return jsonb_build_object(
      'ok', false, 'send_required', false, 'recovery_required', true,
      'operation_key', p_operation_key,
      'message', 'Invitation delivery outcome must be reconciled; do not resend.'
    );
  end if;
  select team.* into v_team
    from public.tournament_four_player_teams team
    join public.tournaments tournament on tournament.id = team.tournament_id
   where team.id::text = p_team_id
     and team.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update of team;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_NOT_FOUND';
  end if;
  select member.* into v_member
    from public.tournament_four_player_team_members member
   where member.id::text = p_member_id
     and member.team_id = v_team.id
     and member.status = 'INVITED'
   for update;
  if not found or v_member.invitation_version is distinct from p_invitation_version then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_STALE';
  end if;
  insert into public.tournament_team_operations (
    operation_key, request_fingerprint, club_id, tournament_id, surface,
    action, entity_type, entity_id, actor, status
  ) values (
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id::uuid,
    'registration', 'team_invitation_delivery',
    'tournament_four_player_team_member', p_member_id,
    coalesce(nullif(p_actor, ''), 'unknown'), 'INTENT'
  );
  update public.tournament_four_player_team_members member
     set invitation_token_hash = p_invitation_token_hash
   where member.id = v_member.id;
  insert into public.tournament_team_invitation_deliveries (
    tournament_id, team_id, member_id, invitation_version, email_mode,
    status, recipient_email_hash, operation_key
  ) values (
    v_team.tournament_id, v_team.id, v_member.id, p_invitation_version,
    lower(p_email_mode), 'pending', p_recipient_email_hash, p_operation_key
  )
  returning * into v_delivery;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, after_json, request_fingerprint
  ) values (
    p_club_id, v_team.tournament_id, v_team.event_option_id,
    'tournament_team_invitation_delivery', v_delivery.id::text,
    'team_invitation_delivery_claimed',
    coalesce(nullif(p_actor, ''), 'unknown'),
    jsonb_build_object(
      'member_id', v_member.id, 'invitation_version', p_invitation_version,
      'email_mode', lower(p_email_mode), 'status', 'pending'
    ),
    p_request_fingerprint
  );
  return jsonb_build_object(
    'ok', true, 'send_required', true, 'recovery_required', false,
    'delivery_id', v_delivery.id, 'operation_key', p_operation_key
  );
end;
$$;

revoke all on function public.server_claim_tournament_team_invitation_delivery(
  text, text, text, text, integer, text, text, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.server_claim_tournament_team_invitation_delivery(
  text, text, text, text, integer, text, text, text, text, text, text
) to service_role;

create or replace function public.server_complete_tournament_team_invitation_delivery(
  p_club_id text,
  p_tournament_id text,
  p_delivery_id text,
  p_status text,
  p_provider_message_id text,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_delivery public.tournament_team_invitation_deliveries%rowtype;
  v_team public.tournament_four_player_teams%rowtype;
  v_result jsonb;
begin
  if lower(coalesce(p_status, '')) not in (
    'dry_run', 'staging_redirect', 'sent', 'skipped', 'failed'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_DELIVERY_STATUS_INVALID';
  end if;
  select delivery.* into v_delivery
    from public.tournament_team_invitation_deliveries delivery
   where delivery.id::text = p_delivery_id
     and delivery.tournament_id::text = p_tournament_id
     and delivery.operation_key = p_operation_key
   for update;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_DELIVERY_NOT_FOUND';
  end if;
  select team.* into v_team
    from public.tournament_four_player_teams team
   where team.id = v_delivery.team_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_NOT_FOUND';
  end if;
  perform 1 from public.tournaments tournament
   where tournament.id = v_team.tournament_id
     and tournament.club_id::text = p_club_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_NOT_FOUND';
  end if;
  if v_delivery.status <> 'pending' then
    select operation.result_json into v_result
      from public.tournament_team_operations operation
     where operation.operation_key = p_operation_key
       and operation.request_fingerprint = p_request_fingerprint
       and operation.status = 'COMPLETED';
    if v_result is not null then
      return v_result;
    end if;
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_INVITATION_DELIVERY_RECOVERY_REQUIRED';
  end if;
  update public.tournament_team_invitation_deliveries delivery
     set status = lower(p_status),
         provider_message_id = nullif(p_provider_message_id, '')
   where delivery.id = v_delivery.id
  returning * into v_delivery;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, after_json, request_fingerprint
  ) values (
    p_club_id, v_team.tournament_id, v_team.event_option_id,
    'tournament_team_invitation_delivery', v_delivery.id::text,
    'team_invitation_delivery_completed',
    coalesce(nullif(p_actor, ''), 'unknown'),
    jsonb_build_object(
      'member_id', v_delivery.member_id,
      'invitation_version', v_delivery.invitation_version,
      'email_mode', v_delivery.email_mode, 'status', v_delivery.status,
      'provider_message_id', v_delivery.provider_message_id
    ),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', v_delivery.status <> 'failed',
    'status', v_delivery.status,
    'delivery_id', v_delivery.id,
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.server_complete_tournament_team_invitation_delivery(
  text, text, text, text, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.server_complete_tournament_team_invitation_delivery(
  text, text, text, text, text, text, text, text
) to service_role;

create or replace function public.admin_amend_tournament_four_player_roster_cas(
  p_club_id text,
  p_tournament_id text,
  p_team_id text,
  p_expected_team_version integer,
  p_action text,
  p_members jsonb,
  p_reason text,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_team public.tournament_four_player_teams%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_before jsonb;
  v_after_members jsonb;
  v_member jsonb;
  v_registration public.tournament_registrations%rowtype;
  v_slot text;
  v_email text;
  v_registration_id text;
  v_gender text;
  v_action text := upper(coalesce(p_action, ''));
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'registration', 'four_player_roster_' || lower(v_action),
    'tournament_four_player_team', p_team_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if v_action not in ('REPLACE', 'WITHDRAW')
     or nullif(btrim(p_reason), '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_ROSTER_ACTION_INVALID';
  end if;
  select team.* into v_team
    from public.tournament_four_player_teams team
    join public.tournaments tournament on tournament.id = team.tournament_id
   where team.id::text = p_team_id
     and team.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update of team;
  if not found or p_expected_team_version is null
     or v_team.version is distinct from p_expected_team_version then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_STALE';
  end if;
  if v_team.status in ('WITHDRAWN', 'CANCELLED') then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_INACTIVE';
  end if;
  if v_action = 'WITHDRAW' and exists (
    select 1
      from public.tournament_team_matchups matchup
     where v_team.id in (matchup.team_a_id, matchup.team_b_id)
       and (
         matchup.status in ('IN_PROGRESS', 'TIEBREAK_REQUIRED', 'FINAL')
         or exists (
           select 1
             from public.tournament_team_lineup_submissions lineup
            where lineup.matchup_id = matchup.id
         )
         or exists (
           select 1
             from public.tournament_team_match_games child
            where child.matchup_id = matchup.id
         )
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_WITHDRAWAL_LOCKED_BY_MATCH_ACTIVITY';
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id = v_team.event_option_id
     and event.tournament_id = v_team.tournament_id
   for share;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_EVENT_NOT_FOUND';
  end if;
  if v_action = 'REPLACE' and exists (
    select 1
      from public.tournament_team_matchups matchup
     where v_team.id in (matchup.team_a_id, matchup.team_b_id)
       and (
         matchup.status in ('IN_PROGRESS', 'TIEBREAK_REQUIRED', 'FINAL')
         or exists (
           select 1 from public.tournament_team_lineup_submissions lineup
            where lineup.matchup_id = matchup.id
         )
         or exists (
           select 1 from public.tournament_team_match_games child
           where child.matchup_id = matchup.id
         )
       )
  ) and (
    not v_event.team_allow_substitutes
    or exists (
      select 1
        from public.tournament_team_matchups matchup
       where v_team.id in (matchup.team_a_id, matchup.team_b_id)
         and matchup.status not in ('FINAL', 'VOID')
         and (
           exists (
             select 1 from public.tournament_team_lineup_submissions lineup
              where lineup.matchup_id = matchup.id
           )
           or exists (
             select 1 from public.tournament_team_match_games child
              where child.matchup_id = matchup.id
           )
         )
    )
  ) then
    raise exception using errcode = 'P0001',
      message = case
        when v_event.team_allow_substitutes
          then 'JUPR_TOURNAMENT_TEAM_SUBSTITUTE_LOCKED_BY_ACTIVE_MATCH'
        else 'JUPR_TOURNAMENT_TEAM_SUBSTITUTES_DISABLED'
      end;
  end if;
  select jsonb_build_object(
    'team', to_jsonb(v_team),
    'members', coalesce(jsonb_agg(to_jsonb(member) order by member.slot), '[]'::jsonb)
  ) into v_before
    from public.tournament_four_player_team_members member
   where member.team_id = v_team.id
     and member.status <> 'REMOVED';

  if v_action = 'WITHDRAW' then
    update public.tournament_four_player_team_members member
       set status = 'REMOVED',
           invitation_version = member.invitation_version + 1,
           invitation_token_hash = null
     where member.team_id = v_team.id
       and member.status <> 'REMOVED';
    update public.tournament_team_matchups matchup
       set status = 'VOID', version = matchup.version + 1
     where v_team.id in (matchup.team_a_id, matchup.team_b_id)
       and matchup.status in ('LINEUPS_PENDING', 'READY');
    update public.tournament_four_player_teams team
       set status = 'WITHDRAWN', version = team.version + 1
     where team.id = v_team.id
    returning * into v_team;
  else
    if pg_catalog.jsonb_typeof(coalesce(p_members, 'null'::jsonb)) <> 'array'
       or pg_catalog.jsonb_array_length(p_members) <> 4
       or (
         select count(distinct upper(member->>'slot'))
           from pg_catalog.jsonb_array_elements(p_members) member
       ) <> 4
       or exists (
         select 1 from pg_catalog.jsonb_array_elements(p_members) member
          where upper(coalesce(member->>'slot', '')) not in (
            'MAN_1', 'MAN_2', 'WOMAN_1', 'WOMAN_2'
          )
            or nullif(lower(btrim(member->>'email')), '') is null
       )
       or (
         select count(distinct lower(btrim(member->>'email')))
           from pg_catalog.jsonb_array_elements(p_members) member
       ) <> 4
       or (
         select count(*) from pg_catalog.jsonb_array_elements(p_members) member
          where nullif(member->>'registration_id', '') =
            v_team.captain_registration_id
       ) <> 1 then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_FOUR_PLAYER_ROSTER_INVALID';
    end if;
    update public.tournament_four_player_team_members member
       set status = 'REMOVED',
           invitation_version = member.invitation_version + 1,
           invitation_token_hash = null
     where member.team_id = v_team.id
       and member.status <> 'REMOVED';
    for v_member in
      select member from pg_catalog.jsonb_array_elements(p_members) member
    loop
      v_slot := upper(v_member->>'slot');
      v_email := lower(btrim(v_member->>'email'));
      v_registration_id := nullif(v_member->>'registration_id', '');
      v_registration := null;
      if v_registration_id is not null then
        select registration.* into v_registration
          from public.tournament_registrations registration
         where registration.id::text = v_registration_id
           and registration.tournament_id = v_team.tournament_id
           and lower(btrim(registration.email)) = v_email
           and upper(coalesce(registration.status, 'CONFIRMED')) not in (
             'CANCELLED', 'WITHDRAWN'
           );
        if not found then
          raise exception using errcode = '22023',
            message = 'JUPR_TOURNAMENT_TEAM_MEMBER_REGISTRATION_INVALID';
        end if;
        v_gender := regexp_replace(
          lower(coalesce(v_registration.gender, '')), '[^a-z]', '', 'g'
        );
      else
        v_gender := regexp_replace(
          lower(coalesce(v_member->>'gender', '')), '[^a-z]', '', 'g'
        );
      end if;
      if (v_slot like 'MAN_%' and v_gender not in (
        'm', 'male', 'man', 'men', 'mens', 'boy', 'boys'
      )) or (v_slot like 'WOMAN_%' and v_gender not in (
        'f', 'female', 'woman', 'women', 'womens', 'girl', 'girls'
      )) then
        raise exception using errcode = '22023',
          message = 'JUPR_TOURNAMENT_TEAM_MEMBER_GENDER_INVALID';
      end if;
      insert into public.tournament_four_player_team_members (
        team_id, tournament_id, event_option_id, slot, invited_email,
        registration_id, player_id, display_name_snapshot, gender_snapshot,
        status, accepted_at
      ) values (
        v_team.id, v_team.tournament_id, v_team.event_option_id,
        v_slot, v_email, v_registration.id, v_registration.player_id,
        coalesce(
          nullif(v_registration.display_name, ''),
          nullif(btrim(concat_ws(
            ' ', v_registration.first_name, v_registration.last_name
          )), ''),
          nullif(v_member->>'display_name', '')
        ),
        coalesce(
          nullif(v_registration.gender, ''), nullif(v_member->>'gender', '')
        ),
        case when v_registration_id = v_team.captain_registration_id
          then 'ACCEPTED' else 'INVITED' end,
        case when v_registration_id = v_team.captain_registration_id
          then clock_timestamp() else null end
      );
    end loop;
    update public.tournament_four_player_teams team
       set status = 'FORMING', version = team.version + 1
     where team.id = v_team.id
    returning * into v_team;
  end if;
  select coalesce(jsonb_agg(to_jsonb(member) order by member.slot), '[]'::jsonb)
    into v_after_members
    from public.tournament_four_player_team_members member
   where member.team_id = v_team.id
     and member.status <> 'REMOVED';
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json, request_fingerprint
  ) values (
    p_club_id, v_team.tournament_id, v_team.event_option_id,
    'tournament_four_player_team', v_team.id::text,
    case when v_action = 'WITHDRAW'
      then 'four_player_team_withdrawn' else 'four_player_roster_replaced' end,
    coalesce(nullif(p_actor, ''), 'unknown'), v_before,
    jsonb_build_object(
      'team', to_jsonb(v_team), 'members', v_after_members,
      'reason', btrim(p_reason),
      'substitutes_allowed', v_event.team_allow_substitutes
    ),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'team', to_jsonb(v_team), 'members', v_after_members,
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_amend_tournament_four_player_roster_cas(
  text, text, text, integer, text, jsonb, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_amend_tournament_four_player_roster_cas(
  text, text, text, integer, text, jsonb, text, text, text, text
) to service_role;

create or replace function public.calculate_tournament_team_podium(
  p_draw_id uuid
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_final public.tournament_team_matchups%rowtype;
  v_bronze public.tournament_team_matchups%rowtype;
  v_seed_ids uuid[];
  v_third uuid;
begin
  select draw.* into v_draw
    from public.tournament_event_draws draw
   where draw.id = p_draw_id
     and draw.draw_kind = 'TEAM_PARENT';
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_DRAW_NOT_FOUND';
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id = v_draw.event_option_id
     and event.tournament_id = v_draw.tournament_id
     and event.competition_format = 'FOUR_PLAYER_TEAM';
  if not found then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_FOUR_PLAYER_DRAW_REQUIRED';
  end if;
  if exists (
    select 1
      from public.tournament_team_matchups matchup
     where matchup.draw_id = v_draw.id
       and matchup.status not in ('FINAL', 'VOID')
       and (
         matchup.stage = 'ROUND_ROBIN'
         or matchup.playoff_game_code in ('FINAL', 'BRONZE')
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_PODIUM_MATCHUPS_INCOMPLETE';
  end if;
  with team_stats as (
    select team.id,
           team.name,
           count(matchup.id) filter (
             where matchup.winner_team_id = team.id
           )::integer as match_wins,
           count(matchup.id) filter (
             where matchup.loser_team_id = team.id
           )::integer as match_losses,
           coalesce(sum(
             case
               when matchup.team_a_id = team.id then matchup.team_a_game_wins
               when matchup.team_b_id = team.id then matchup.team_b_game_wins
               else 0
             end
           ), 0)::integer as game_wins,
           coalesce(sum(
             case
               when matchup.team_a_id = team.id then matchup.team_b_game_wins
               when matchup.team_b_id = team.id then matchup.team_a_game_wins
               else 0
             end
           ), 0)::integer as game_losses
      from public.tournament_four_player_teams team
      left join public.tournament_team_matchups matchup
        on matchup.draw_id = v_draw.id
       and matchup.stage = 'ROUND_ROBIN'
       and matchup.status = 'FINAL'
       and team.id in (matchup.team_a_id, matchup.team_b_id)
     where team.draw_id = v_draw.id
       and team.status = 'CONFIRMED'
       and team.eligibility_state in ('ELIGIBLE', 'NOT_REQUIRED')
     group by team.id, team.name
  ),
  ranked_stats as (
    select stats.*,
           (
             select count(*)
               from public.tournament_team_matchups head_to_head
               join team_stats opponent
                 on opponent.id = case
                   when head_to_head.team_a_id = stats.id
                     then head_to_head.team_b_id
                   else head_to_head.team_a_id
                 end
                and opponent.match_wins = stats.match_wins
                and opponent.match_losses = stats.match_losses
              where head_to_head.draw_id = v_draw.id
                and head_to_head.stage = 'ROUND_ROBIN'
                and head_to_head.status = 'FINAL'
                and head_to_head.winner_team_id = stats.id
           )::integer as tied_head_to_head_wins
      from team_stats stats
  )
  select array_agg(
           ranked.id
           order by ranked.match_wins desc,
                    ranked.match_losses,
                    ranked.tied_head_to_head_wins desc,
                    (ranked.game_wins - ranked.game_losses) desc,
                    ranked.game_wins desc,
                    lower(ranked.name),
                    ranked.id
         )
    into v_seed_ids
    from ranked_stats ranked;
  if cardinality(v_seed_ids) < 3 then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_PODIUM_SOURCE_INCOMPLETE';
  end if;
  if v_event.team_playoff_format = 'NONE' then
    return jsonb_build_array(
      jsonb_build_object('placement', 1, 'team_id', v_seed_ids[1]::text),
      jsonb_build_object('placement', 2, 'team_id', v_seed_ids[2]::text),
      jsonb_build_object('placement', 3, 'team_id', v_seed_ids[3]::text)
    );
  end if;
  select matchup.* into v_final
    from public.tournament_team_matchups matchup
   where matchup.draw_id = v_draw.id
     and matchup.stage = 'PLAYOFF'
     and matchup.playoff_game_code = 'FINAL';
  if not found or v_final.status <> 'FINAL'
     or v_final.winner_team_id is null or v_final.loser_team_id is null
     or not (v_final.winner_team_id = any(v_seed_ids))
     or not (v_final.loser_team_id = any(v_seed_ids)) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_PODIUM_FINAL_INCOMPLETE';
  end if;
  if v_event.team_playoff_format = 'TOP_4_SEMIFINALS_WITH_BRONZE' then
    select matchup.* into v_bronze
      from public.tournament_team_matchups matchup
     where matchup.draw_id = v_draw.id
       and matchup.stage = 'PLAYOFF'
       and matchup.playoff_game_code = 'BRONZE';
    if not found or v_bronze.status <> 'FINAL'
       or v_bronze.winner_team_id is null
       or not (v_bronze.winner_team_id = any(v_seed_ids)) then
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_TEAM_PODIUM_BRONZE_INCOMPLETE';
    end if;
    v_third := v_bronze.winner_team_id;
  elsif v_event.team_playoff_format = 'TOP_4_SEMIFINALS' then
    select seed_id into v_third
      from unnest(v_seed_ids) with ordinality seed(seed_id, seed_number)
     where seed.seed_id in (
       select matchup.loser_team_id
         from public.tournament_team_matchups matchup
        where matchup.draw_id = v_draw.id
          and matchup.stage = 'PLAYOFF'
          and matchup.playoff_game_code in ('SF1', 'SF2')
          and matchup.status = 'FINAL'
     )
     order by seed.seed_number
     limit 1;
  else
    select seed_id into v_third
      from unnest(v_seed_ids) with ordinality seed(seed_id, seed_number)
     where seed.seed_id not in (
       v_final.winner_team_id, v_final.loser_team_id
     )
     order by seed.seed_number
     limit 1;
  end if;
  if v_third is null then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_PODIUM_THIRD_PLACE_INCOMPLETE';
  end if;
  return jsonb_build_array(
    jsonb_build_object('placement', 1, 'team_id', v_final.winner_team_id::text),
    jsonb_build_object('placement', 2, 'team_id', v_final.loser_team_id::text),
    jsonb_build_object('placement', 3, 'team_id', v_third::text)
  );
end;
$$;

revoke all on function public.calculate_tournament_team_podium(uuid)
  from public, anon, authenticated;
grant execute on function public.calculate_tournament_team_podium(uuid)
  to service_role;

create or replace function public.admin_replace_tournament_team_podium_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_podium jsonb,
  p_publish boolean,
  p_reason text,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_authorized_draw_id uuid;
  v_before jsonb;
  v_before_placements jsonb;
  v_requested_placements jsonb;
  v_derived_placements jsonb;
  v_saved jsonb;
  v_was_public boolean;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'operations', 'team_podium_replace', 'tournament_event_draw',
    p_draw_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if pg_catalog.jsonb_typeof(coalesce(p_podium, 'null'::jsonb)) <> 'array' then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_PODIUM_INVALID';
  end if;
  select draw.id into v_authorized_draw_id
    from public.tournament_event_draws draw
    join public.tournaments tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
     and draw.draw_kind = 'TEAM_PARENT';
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_DRAW_NOT_FOUND';
  end if;
  perform public.lock_tournament_team_draw(v_authorized_draw_id);
  select draw.* into v_draw
    from public.tournament_event_draws draw
    join public.tournaments tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
     and draw.draw_kind = 'TEAM_PARENT'
   for update of draw;
  if not found or p_expected_draw_updated_at is null
     or v_draw.updated_at is distinct from p_expected_draw_updated_at then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_DRAW_STALE';
  end if;
  v_derived_placements := public.calculate_tournament_team_podium(v_draw.id);
  select coalesce(
           jsonb_agg(
             jsonb_build_object(
               'placement', podium.placement, 'team_id', podium.team_id::text
             )
             order by podium.placement
           ),
           '[]'::jsonb
         )
    into v_before_placements
    from public.tournament_four_player_podium podium
   where podium.draw_id = v_draw.id;
  if pg_catalog.jsonb_array_length(p_podium) = 0 then
    v_requested_placements := v_derived_placements;
  else
    select coalesce(
             jsonb_agg(
               jsonb_build_object(
                 'placement', x.placement, 'team_id', x.team_id
               )
               order by x.placement
             ),
             '[]'::jsonb
           )
      into v_requested_placements
      from pg_catalog.jsonb_to_recordset(p_podium) as x(
        placement integer, team_id text
      );
    if v_requested_placements is distinct from v_derived_placements then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_TEAM_PODIUM_CALCULATION_MISMATCH';
    end if;
  end if;
  v_was_public := upper(coalesce(v_draw.status, '')) = 'PUBLISHED';
  if v_was_public
     and (
       not coalesce(p_publish, false)
       or v_before_placements is distinct from v_requested_placements
     )
     and nullif(btrim(p_reason), '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_PUBLIC_CORRECTION_REASON_REQUIRED';
  end if;
  if p_publish and exists (
    select 1 from public.tournament_team_matchups matchup
     where matchup.draw_id = v_draw.id
       and matchup.status not in ('FINAL', 'VOID')
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_PODIUM_MATCHUPS_INCOMPLETE';
  end if;
  if p_publish then
    perform child.id
      from public.tournament_team_match_games child
      join public.tournament_team_matchups matchup
        on matchup.id = child.matchup_id
     where matchup.draw_id = v_draw.id
       and child.counts_for_rating
       and child.status = 'FINAL'
     order by child.id
     for share of child;
    perform canonical.id
      from public.matches canonical
      join public.tournament_team_match_games child
        on child.tournament_game_id = canonical.tournament_game_id
      join public.tournament_team_matchups matchup
        on matchup.id = child.matchup_id
     where matchup.draw_id = v_draw.id
       and child.counts_for_rating
       and child.status = 'FINAL'
       and canonical.deleted_at is null
       and not coalesce(canonical.excluded_from_ratings, false)
     order by canonical.id
     for share of canonical;
  end if;
  if p_publish and exists (
    select 1
      from public.tournament_team_match_games child
      join public.tournament_team_matchups matchup
        on matchup.id = child.matchup_id
     where matchup.draw_id = v_draw.id
       and child.counts_for_rating
       and child.status = 'FINAL'
       and (
         child.tournament_game_id is null
         or (
           select count(*)
             from public.matches canonical
            where canonical.tournament_id = v_draw.tournament_id
              and canonical.tournament_game_id = child.tournament_game_id
              and canonical.deleted_at is null
              and not coalesce(canonical.excluded_from_ratings, false)
         ) <> 1
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_RATED_CHILD_CANONICAL_RESULT_REQUIRED';
  end if;
  if exists (
    select 1
      from pg_catalog.jsonb_to_recordset(v_derived_placements) as x(
        placement integer, team_id text
      )
     where x.placement not between 1 and 3
        or not exists (
          select 1 from public.tournament_four_player_teams team
           where team.id::text = x.team_id
             and team.tournament_id = v_draw.tournament_id
             and team.event_option_id::text = v_draw.event_option_id::text
             and team.draw_id = v_draw.id
             and team.status = 'CONFIRMED'
             and team.eligibility_state in ('ELIGIBLE', 'NOT_REQUIRED')
        )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_PODIUM_TEAM_INVALID';
  end if;
  select coalesce(jsonb_agg(to_jsonb(podium) order by podium.placement), '[]'::jsonb)
    into v_before
    from public.tournament_four_player_podium podium
   where podium.draw_id = v_draw.id;
  delete from public.tournament_four_player_podium podium
   where podium.draw_id = v_draw.id;
  with inserted as (
    insert into public.tournament_four_player_podium (
      tournament_id, draw_id, placement, team_id, source,
      published_at, published_by
    )
    select v_draw.tournament_id, v_draw.id, x.placement, x.team_id::uuid,
           'CALCULATED',
           case when p_publish then clock_timestamp() else null end,
           case when p_publish
             then coalesce(nullif(p_actor, ''), 'unknown') else null end
      from pg_catalog.jsonb_to_recordset(v_derived_placements) as x(
        placement integer, team_id text
      )
     order by x.placement
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted) order by inserted.placement), '[]'::jsonb)
    into v_saved from inserted;
  update public.tournament_event_draws draw
     set status = case when p_publish then 'published' else 'draft' end,
         updated_at = clock_timestamp()
   where draw.id = v_draw.id;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json, request_fingerprint
  ) values (
    p_club_id, v_draw.tournament_id, v_draw.event_option_id,
    'tournament_four_player_podium', v_draw.id::text,
    case
      when p_publish and v_was_public
        and v_before_placements is distinct from v_requested_placements
        then 'four_player_podium_corrected'
      when p_publish then 'four_player_podium_published'
      when v_was_public then 'four_player_podium_unpublished'
      else 'four_player_podium_saved'
    end,
    coalesce(nullif(p_actor, ''), 'unknown'), v_before,
    jsonb_build_object(
      'podium', v_saved, 'published', p_publish,
      'public_draw_status', case when p_publish then 'published' else 'draft' end,
      'correction_reason', nullif(btrim(p_reason), '')
    ),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'podium', v_saved, 'published', p_publish,
    'draw_status', case when p_publish then 'published' else 'draft' end,
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_replace_tournament_team_podium_cas(
  text, text, text, timestamptz, jsonb, boolean, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_replace_tournament_team_podium_cas(
  text, text, text, timestamptz, jsonb, boolean, text, text, text, text
) to service_role;

create or replace function public.admin_reconcile_tournament_team_match_game_cas(
  p_club_id text,
  p_tournament_id text,
  p_match_game_id text,
  p_match_id text,
  p_expected_match_row_version integer,
  p_expected_game_version integer,
  p_expected_matchup_version integer,
  p_reason text,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_child public.tournament_team_match_games%rowtype;
  v_authorized_draw_id uuid;
  v_before_child jsonb;
  v_matchup public.tournament_team_matchups%rowtype;
  v_official public.matches%rowtype;
  v_tiebreak public.tournament_team_match_games%rowtype;
  v_lineup_a public.tournament_team_lineup_submissions%rowtype;
  v_lineup_b public.tournament_team_lineup_submissions%rowtype;
  v_reg_complete integer;
  v_reg_a integer;
  v_reg_b integer;
  v_total_a integer;
  v_total_b integer;
  v_winner uuid;
  v_loser uuid;
  v_status text;
  v_deleted boolean;
  v_excluded boolean;
  v_shape_valid boolean := false;
  v_shape_rejected boolean := false;
  v_side_swapped boolean := false;
  v_score_a integer;
  v_score_b integer;
  v_dependencies jsonb := '[]'::jsonb;
  v_operation jsonb;
  v_result jsonb;
begin
  if nullif(pg_catalog.btrim(p_reason), '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_RECONCILIATION_REASON_REQUIRED';
  end if;
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'operations', 'team_match_game_reconcile',
    'tournament_team_match_game', p_match_game_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  select matchup.draw_id into v_authorized_draw_id
    from public.tournament_team_match_games child
    join public.tournament_team_matchups matchup on matchup.id = child.matchup_id
    join public.tournaments tournament on tournament.id = child.tournament_id
   where child.id::text = p_match_game_id
     and child.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_MATCH_GAME_NOT_FOUND';
  end if;
  perform public.lock_tournament_team_draw(v_authorized_draw_id);
  select child.* into v_child
    from public.tournament_team_match_games child
    join public.tournaments tournament on tournament.id = child.tournament_id
   where child.id::text = p_match_game_id
     and child.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update of child;
  if not found or p_expected_game_version is null
     or v_child.version is distinct from p_expected_game_version
     or v_child.tournament_game_id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_MATCH_GAME_STALE';
  end if;
  select matchup.* into v_matchup
    from public.tournament_team_matchups matchup
   where matchup.id = v_child.matchup_id
   for update;
  if p_expected_matchup_version is null
     or v_matchup.version is distinct from p_expected_matchup_version then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_MATCHUP_STALE';
  end if;
  if v_matchup.stage = 'ROUND_ROBIN' and exists (
    select 1
      from public.tournament_team_matchups playoff
     where playoff.draw_id = v_matchup.draw_id
       and playoff.stage = 'PLAYOFF'
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_ROUND_ROBIN_LOCKED_BY_PLAYOFFS';
  end if;
  select official.* into v_official
    from public.matches official
   where official.id::text = p_match_id
     and official.club_id::text = p_club_id
     and official.tournament_id::text = p_tournament_id
     and official.tournament_game_id = v_child.tournament_game_id
   for share;
  if not found or p_expected_match_row_version is null
     or v_official.row_version is distinct from p_expected_match_row_version then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_OFFICIAL_MATCH_STALE';
  end if;
  v_before_child := to_jsonb(v_child);
  v_excluded := coalesce(v_official.excluded_from_ratings, false);
  v_deleted := v_official.deleted_at is not null or v_excluded;
  if not v_deleted and v_child.match_format = 'SINGLES'
     and lower(coalesce(v_official.match_format, '')) = 'singles'
     and cardinality(v_child.team_a_player_ids) = 1
     and cardinality(v_child.team_b_player_ids) = 1 then
    if array[v_official.t1_p1] = v_child.team_a_player_ids
       and array[v_official.t2_p1] = v_child.team_b_player_ids
       and v_official.t1_p2 is null and v_official.t2_p2 is null then
      v_shape_valid := true;
    elsif array[v_official.t1_p1] = v_child.team_b_player_ids
       and array[v_official.t2_p1] = v_child.team_a_player_ids
       and v_official.t1_p2 is null and v_official.t2_p2 is null then
      v_shape_valid := true;
      v_side_swapped := true;
    end if;
  elsif not v_deleted and v_child.match_format = 'DOUBLES'
     and lower(coalesce(v_official.match_format, '')) = 'doubles'
     and cardinality(v_child.team_a_player_ids) = 2
     and cardinality(v_child.team_b_player_ids) = 2 then
    if v_child.team_a_player_ids @> array[v_official.t1_p1, v_official.t1_p2]
       and v_child.team_a_player_ids <@ array[v_official.t1_p1, v_official.t1_p2]
       and v_child.team_b_player_ids @> array[v_official.t2_p1, v_official.t2_p2]
       and v_child.team_b_player_ids <@ array[v_official.t2_p1, v_official.t2_p2] then
      v_shape_valid := true;
    elsif v_child.team_a_player_ids @> array[v_official.t2_p1, v_official.t2_p2]
       and v_child.team_a_player_ids <@ array[v_official.t2_p1, v_official.t2_p2]
       and v_child.team_b_player_ids @> array[v_official.t1_p1, v_official.t1_p2]
       and v_child.team_b_player_ids <@ array[v_official.t1_p1, v_official.t1_p2] then
      v_shape_valid := true;
      v_side_swapped := true;
    end if;
  end if;
  v_shape_rejected := not v_deleted and not v_shape_valid;
  v_deleted := v_deleted or v_shape_rejected;
  v_score_a := case when v_side_swapped
    then v_official.score_t2 else v_official.score_t1 end;
  v_score_b := case when v_side_swapped
    then v_official.score_t1 else v_official.score_t2 end;
  if v_deleted or v_score_a is null or v_score_b is null
     or v_score_a = v_score_b then
    update public.tournament_team_match_games child
       set score_a = null, score_b = null, winner_team_id = null,
           loser_team_id = null, status = 'VOID', finalized_at = null,
           version = child.version + 1
     where child.id = v_child.id
    returning * into v_child;
    update public.tournament_games game
       set score_a = null, score_b = null, winner_team_id = null,
           loser_team_id = null, finalized_at = null,
           updated_at = clock_timestamp()
     where game.id = v_child.tournament_game_id;
    update public.tournament_team_matchups matchup
       set status = 'CORRECTION_REQUIRED',
           winner_team_id = null, loser_team_id = null, finalized_at = null,
           version = matchup.version + 1
     where matchup.id = v_matchup.id
    returning * into v_matchup;
  else
    v_winner := case when v_score_a > v_score_b
      then v_matchup.team_a_id else v_matchup.team_b_id end;
    v_loser := case when v_score_a > v_score_b
      then v_matchup.team_b_id else v_matchup.team_a_id end;
    update public.tournament_team_match_games child
       set score_a = v_score_a,
           score_b = v_score_b,
           winner_team_id = v_winner, loser_team_id = v_loser,
           status = 'FINAL', finalized_at = coalesce(
             v_official.updated_at, clock_timestamp()
           ),
           version = child.version + 1
     where child.id = v_child.id
    returning * into v_child;
    update public.tournament_games game
       set score_a = v_score_a,
           score_b = v_score_b,
           winner_team_id = case when v_score_a > v_score_b
             then game.team_a_id else game.team_b_id end,
           loser_team_id = case when v_score_a > v_score_b
             then game.team_b_id else game.team_a_id end,
           finalized_at = coalesce(v_official.updated_at, clock_timestamp()),
           updated_at = clock_timestamp()
     where game.id = v_child.tournament_game_id;
    select count(*) filter (
             where child.game_code <> 'TIEBREAK' and child.status = 'FINAL'
           ),
           count(*) filter (
             where child.game_code <> 'TIEBREAK' and child.status = 'FINAL'
               and child.winner_team_id = v_matchup.team_a_id
           ),
           count(*) filter (
             where child.game_code <> 'TIEBREAK' and child.status = 'FINAL'
               and child.winner_team_id = v_matchup.team_b_id
           )
      into v_reg_complete, v_reg_a, v_reg_b
      from public.tournament_team_match_games child
     where child.matchup_id = v_matchup.id;
    select child.* into v_tiebreak
      from public.tournament_team_match_games child
     where child.matchup_id = v_matchup.id
       and child.game_code = 'TIEBREAK'
     for update;
    if v_reg_complete = 4 and v_reg_a = 2 and v_reg_b = 2
       and v_tiebreak.id is null then
      select lineup.* into v_lineup_a
        from public.tournament_team_lineup_submissions lineup
       where lineup.matchup_id = v_matchup.id
         and lineup.team_id = v_matchup.team_a_id;
      select lineup.* into v_lineup_b
        from public.tournament_team_lineup_submissions lineup
       where lineup.matchup_id = v_matchup.id
         and lineup.team_id = v_matchup.team_b_id;
      if v_matchup.tiebreak_mode = 'SINGLES' then
        insert into public.tournament_team_match_games (
          tournament_id, matchup_id, game_code, game_order, match_format,
          counts_for_rating, team_a_player_ids, team_b_player_ids
        ) values (
          v_matchup.tournament_id, v_matchup.id, 'TIEBREAK', 5, 'SINGLES',
          true, array[v_lineup_a.singles_tiebreak_player_id],
          array[v_lineup_b.singles_tiebreak_player_id]
        )
        returning * into v_tiebreak;
        perform public.materialize_tournament_team_rating_asset(v_tiebreak.id);
      else
        insert into public.tournament_team_match_games (
          tournament_id, matchup_id, game_code, game_order, match_format,
          counts_for_rating, team_a_player_ids, team_b_player_ids
        )
        select v_matchup.tournament_id, v_matchup.id, 'TIEBREAK', 5,
               'SKINNY_RELAY', false,
               array_agg(member.player_id order by
                 case member.slot
                   when 'WOMAN_1' then 1 when 'WOMAN_2' then 2
                   when 'MAN_1' then 3 else 4 end
               ) filter (where member.team_id = v_matchup.team_a_id),
               array_agg(member.player_id order by
                 case member.slot
                   when 'WOMAN_1' then 1 when 'WOMAN_2' then 2
                   when 'MAN_1' then 3 else 4 end
               ) filter (where member.team_id = v_matchup.team_b_id)
          from public.tournament_four_player_team_members member
         where member.team_id in (v_matchup.team_a_id, v_matchup.team_b_id)
           and member.status = 'ACCEPTED'
        returning * into v_tiebreak;
      end if;
    elsif v_reg_complete = 4 and v_reg_a <> v_reg_b
          and v_tiebreak.id is not null then
      if v_tiebreak.tournament_game_id is not null and exists (
        select 1 from public.matches official
         where official.tournament_game_id = v_tiebreak.tournament_game_id
           and official.deleted_at is null
           and not coalesce(official.excluded_from_ratings, false)
      ) then
        update public.tournament_team_matchups matchup
           set status = 'CORRECTION_REQUIRED', winner_team_id = null,
               loser_team_id = null, finalized_at = null,
               version = matchup.version + 1
         where matchup.id = v_matchup.id
        returning * into v_matchup;
        v_status := 'CORRECTION_REQUIRED';
      else
        update public.tournament_team_match_games child
           set score_a = null, score_b = null, winner_team_id = null,
               loser_team_id = null, status = 'VOID', finalized_at = null,
               version = child.version + 1
         where child.id = v_tiebreak.id
        returning * into v_tiebreak;
        update public.tournament_games game
           set score_a = null, score_b = null, winner_team_id = null,
               loser_team_id = null, finalized_at = null,
               updated_at = clock_timestamp()
         where game.id = v_tiebreak.tournament_game_id;
      end if;
    end if;
    if v_status is distinct from 'CORRECTION_REQUIRED' then
      v_total_a := v_reg_a + case
        when v_tiebreak.status = 'FINAL'
          and v_tiebreak.winner_team_id = v_matchup.team_a_id then 1 else 0 end;
      v_total_b := v_reg_b + case
        when v_tiebreak.status = 'FINAL'
          and v_tiebreak.winner_team_id = v_matchup.team_b_id then 1 else 0 end;
      if v_reg_complete < 4 then
        v_status := 'IN_PROGRESS';
        v_winner := null;
        v_loser := null;
      elsif v_reg_a = v_reg_b and coalesce(v_tiebreak.status, '') <> 'FINAL' then
        v_status := 'TIEBREAK_REQUIRED';
        v_winner := null;
        v_loser := null;
      else
        v_status := 'FINAL';
        v_winner := case when v_total_a > v_total_b
          then v_matchup.team_a_id else v_matchup.team_b_id end;
        v_loser := case when v_total_a > v_total_b
          then v_matchup.team_b_id else v_matchup.team_a_id end;
      end if;
      update public.tournament_team_matchups matchup
         set status = v_status, team_a_game_wins = v_total_a,
             team_b_game_wins = v_total_b, winner_team_id = v_winner,
             loser_team_id = v_loser,
             finalized_at = case when v_status = 'FINAL'
               then clock_timestamp() else null end,
             version = matchup.version + 1
       where matchup.id = v_matchup.id
      returning * into v_matchup;
    end if;
  end if;
  if v_matchup.stage = 'PLAYOFF' then
    v_dependencies := public.resolve_tournament_team_playoff_dependencies(
      v_matchup.draw_id
    );
  end if;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json, request_fingerprint
  ) values (
    p_club_id, v_matchup.tournament_id, v_matchup.event_option_id,
    'tournament_team_match_game', v_child.id::text,
    case when v_deleted
      then case when v_shape_rejected
        then 'team_match_game_official_shape_rejected'
        when v_excluded
        then 'team_match_game_official_exclusion_reconciled'
        else 'team_match_game_official_deletion_reconciled' end
      else 'team_match_game_official_correction_reconciled' end,
    coalesce(nullif(p_actor, ''), 'unknown'), v_before_child,
    jsonb_build_object(
      'game', to_jsonb(v_child), 'matchup', to_jsonb(v_matchup),
      'official_match_id', v_official.id,
      'official_match_row_version', v_official.row_version,
      'official_excluded_from_ratings', v_excluded,
      'official_shape_rejected', v_shape_rejected,
      'official_side_swapped', v_side_swapped,
      'reason', pg_catalog.btrim(p_reason),
      'playoff_dependency_updates', v_dependencies
    ),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'game', to_jsonb(v_child), 'matchup', to_jsonb(v_matchup),
    'playoff_dependency_updates', v_dependencies,
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_reconcile_tournament_team_match_game_cas(
  text, text, text, text, integer, integer, integer, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_reconcile_tournament_team_match_game_cas(
  text, text, text, text, integer, integer, integer, text, text, text, text
) to service_role;

create or replace function public.admin_append_tournament_team_playoffs_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_matchups jsonb,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_authorized_draw_id uuid;
  v_event public.tournament_event_options%rowtype;
  v_saved jsonb;
  v_topology_valid boolean;
  v_seed_ids uuid[];
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'operations', 'team_playoffs_append', 'tournament_event_draw',
    p_draw_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if pg_catalog.jsonb_typeof(coalesce(p_matchups, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_matchups) = 0 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_PLAYOFFS_INVALID';
  end if;
  select draw.id into v_authorized_draw_id
    from public.tournament_event_draws draw
    join public.tournaments tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
     and draw.draw_kind = 'TEAM_PARENT';
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_TEAM_DRAW_NOT_FOUND';
  end if;
  perform public.lock_tournament_team_draw(v_authorized_draw_id);
  select draw.* into v_draw
    from public.tournament_event_draws draw
    join public.tournaments tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
     and draw.draw_kind = 'TEAM_PARENT'
   for update of draw;
  if not found or p_expected_draw_updated_at is null
     or v_draw.updated_at is distinct from p_expected_draw_updated_at then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_DRAW_STALE';
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = v_draw.event_option_id::text
     and event.tournament_id = v_draw.tournament_id
     and event.competition_format = 'FOUR_PLAYER_TEAM';
  if not found or v_event.team_playoff_format = 'NONE' then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_PLAYOFF_FORMAT_REQUIRED';
  end if;
  perform matchup.id
    from public.tournament_team_matchups matchup
   where matchup.draw_id = v_draw.id
     and matchup.stage = 'ROUND_ROBIN'
   order by matchup.id
   for update of matchup;
  if not found or exists (
    select 1
      from public.tournament_team_matchups matchup
     where matchup.draw_id = v_draw.id
       and matchup.stage = 'ROUND_ROBIN'
       and matchup.status not in ('FINAL', 'VOID')
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_PLAYOFF_SOURCE_INCOMPLETE';
  end if;
  with team_stats as (
    select team.id,
           team.name,
           count(matchup.id) filter (
             where matchup.winner_team_id = team.id
           )::integer as match_wins,
           count(matchup.id) filter (
             where matchup.loser_team_id = team.id
           )::integer as match_losses,
           coalesce(sum(
             case
               when matchup.team_a_id = team.id
                 then matchup.team_a_game_wins
               when matchup.team_b_id = team.id
                 then matchup.team_b_game_wins
               else 0
             end
           ), 0)::integer as game_wins,
           coalesce(sum(
             case
               when matchup.team_a_id = team.id
                 then matchup.team_b_game_wins
               when matchup.team_b_id = team.id
                 then matchup.team_a_game_wins
               else 0
             end
           ), 0)::integer as game_losses
      from public.tournament_four_player_teams team
      left join public.tournament_team_matchups matchup
        on matchup.draw_id = v_draw.id
       and matchup.stage = 'ROUND_ROBIN'
       and matchup.status = 'FINAL'
       and team.id in (matchup.team_a_id, matchup.team_b_id)
     where team.draw_id = v_draw.id
       and team.status = 'CONFIRMED'
       and team.eligibility_state in ('ELIGIBLE', 'NOT_REQUIRED')
     group by team.id, team.name
  ),
  ranked_stats as (
    select stats.*,
           (
             select count(*)
               from public.tournament_team_matchups head_to_head
               join team_stats opponent
                 on opponent.id = case
                   when head_to_head.team_a_id = stats.id
                     then head_to_head.team_b_id
                   else head_to_head.team_a_id
                 end
                and opponent.match_wins = stats.match_wins
                and opponent.match_losses = stats.match_losses
              where head_to_head.draw_id = v_draw.id
                and head_to_head.stage = 'ROUND_ROBIN'
                and head_to_head.status = 'FINAL'
                and head_to_head.winner_team_id = stats.id
                and stats.id in (
                  head_to_head.team_a_id, head_to_head.team_b_id
                )
           )::integer as tied_head_to_head_wins
      from team_stats stats
  )
  select array_agg(
           ranked.id
           order by ranked.match_wins desc,
                    ranked.match_losses,
                    ranked.tied_head_to_head_wins desc,
                    (ranked.game_wins - ranked.game_losses) desc,
                    ranked.game_wins desc,
                    lower(ranked.name),
                    ranked.id
         )
    into v_seed_ids
    from ranked_stats ranked;
  if cardinality(v_seed_ids) < (
       case
         when v_event.team_playoff_format = 'TOP_2_FINAL' then 2
         else 4
       end
     ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_PLAYOFF_SOURCE_INCOMPLETE';
  end if;
  if v_event.team_playoff_format = 'TOP_2_FINAL' then
    select count(*) = 1
       and count(*) filter (
         where upper(x.playoff_game_code) = 'FINAL'
           and x.team_a_id is not null
           and x.team_b_id is not null
           and x.team_a_id = v_seed_ids[1]::text
           and x.team_b_id = v_seed_ids[2]::text
           and x.team_a_source->>'type' = 'SEED'
           and x.team_a_source->>'seed' = '1'
           and x.team_b_source->>'type' = 'SEED'
           and x.team_b_source->>'seed' = '2'
       ) = 1
      into v_topology_valid
      from pg_catalog.jsonb_to_recordset(p_matchups) as x(
        playoff_game_code text, team_a_id text, team_b_id text,
        team_a_source jsonb, team_b_source jsonb
      );
  else
    select count(*) = case
             when v_event.team_playoff_format =
               'TOP_4_SEMIFINALS_WITH_BRONZE' then 4 else 3 end
       and count(*) filter (
         where upper(x.playoff_game_code) = 'SF1'
           and x.team_a_id is not null and x.team_b_id is not null
           and x.team_a_id = v_seed_ids[1]::text
           and x.team_b_id = v_seed_ids[4]::text
           and x.team_a_source->>'type' = 'SEED'
           and x.team_a_source->>'seed' = '1'
           and x.team_b_source->>'type' = 'SEED'
           and x.team_b_source->>'seed' = '4'
       ) = 1
       and count(*) filter (
         where upper(x.playoff_game_code) = 'SF2'
           and x.team_a_id is not null and x.team_b_id is not null
           and x.team_a_id = v_seed_ids[2]::text
           and x.team_b_id = v_seed_ids[3]::text
           and x.team_a_source->>'type' = 'SEED'
           and x.team_a_source->>'seed' = '2'
           and x.team_b_source->>'type' = 'SEED'
           and x.team_b_source->>'seed' = '3'
       ) = 1
       and count(*) filter (
         where upper(x.playoff_game_code) = 'FINAL'
           and x.team_a_id is null and x.team_b_id is null
           and x.team_a_source = '{"type":"WINNER","game_code":"SF1"}'::jsonb
           and x.team_b_source = '{"type":"WINNER","game_code":"SF2"}'::jsonb
       ) = 1
       and (
         (
           v_event.team_playoff_format = 'TOP_4_SEMIFINALS'
           and count(*) filter (
             where upper(x.playoff_game_code) = 'BRONZE'
           ) = 0
         )
         or
         (
           v_event.team_playoff_format =
             'TOP_4_SEMIFINALS_WITH_BRONZE'
           and count(*) filter (
             where upper(x.playoff_game_code) = 'BRONZE'
               and x.team_a_id is null and x.team_b_id is null
               and x.team_a_source =
                 '{"type":"LOSER","game_code":"SF1"}'::jsonb
               and x.team_b_source =
                 '{"type":"LOSER","game_code":"SF2"}'::jsonb
           ) = 1
         )
       )
      into v_topology_valid
      from pg_catalog.jsonb_to_recordset(p_matchups) as x(
        playoff_game_code text, team_a_id text, team_b_id text,
        team_a_source jsonb, team_b_source jsonb
      );
  end if;
  if not coalesce(v_topology_valid, false) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_PLAYOFF_TOPOLOGY_INVALID';
  end if;
  if exists (
    select 1 from public.tournament_team_matchups matchup
     where matchup.draw_id = v_draw.id and matchup.stage = 'PLAYOFF'
  ) or not exists (
    select 1 from public.tournament_team_matchups matchup
     where matchup.draw_id = v_draw.id and matchup.stage = 'ROUND_ROBIN'
  ) or exists (
    select 1 from public.tournament_team_matchups matchup
     where matchup.draw_id = v_draw.id
       and matchup.stage = 'ROUND_ROBIN'
       and matchup.status not in ('FINAL', 'VOID')
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_PLAYOFF_SOURCE_INCOMPLETE';
  end if;
  if exists (
    select 1
      from pg_catalog.jsonb_to_recordset(p_matchups) as x(
        playoff_game_code text, team_a_id text, team_b_id text,
        team_a_source jsonb, team_b_source jsonb
      )
     where nullif(upper(x.playoff_game_code), '') is null
        or (
          x.team_a_id is null and (
            x.team_a_source->>'type' not in ('WINNER', 'LOSER')
            or nullif(x.team_a_source->>'game_code', '') is null
          )
        )
        or (
          x.team_b_id is null and (
            x.team_b_source->>'type' not in ('WINNER', 'LOSER')
            or nullif(x.team_b_source->>'game_code', '') is null
          )
        )
        or (
          x.team_a_id is not null and not exists (
            select 1 from public.tournament_four_player_teams team
             where team.id::text = x.team_a_id
               and team.tournament_id = v_draw.tournament_id
               and team.event_option_id::text = v_draw.event_option_id::text
               and team.status = 'CONFIRMED'
               and team.eligibility_state in ('ELIGIBLE', 'NOT_REQUIRED')
               and team.draw_id = v_draw.id
          )
        )
        or (
          x.team_b_id is not null and not exists (
            select 1 from public.tournament_four_player_teams team
             where team.id::text = x.team_b_id
               and team.tournament_id = v_draw.tournament_id
               and team.event_option_id::text = v_draw.event_option_id::text
               and team.status = 'CONFIRMED'
               and team.eligibility_state in ('ELIGIBLE', 'NOT_REQUIRED')
               and team.draw_id = v_draw.id
          )
        )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_PLAYOFFS_INVALID';
  end if;
  with inserted as (
    insert into public.tournament_team_matchups (
      tournament_id, event_option_id, draw_id, stage, playoff_game_code,
      team_a_id, team_b_id, team_a_source, team_b_source, tiebreak_mode
    )
    select v_draw.tournament_id, v_draw.event_option_id, v_draw.id, 'PLAYOFF',
           upper(x.playoff_game_code), nullif(x.team_a_id, '')::uuid,
           nullif(x.team_b_id, '')::uuid, x.team_a_source, x.team_b_source,
           v_event.team_tiebreak_mode
      from pg_catalog.jsonb_to_recordset(p_matchups) as x(
        playoff_game_code text, team_a_id text, team_b_id text,
        team_a_source jsonb, team_b_source jsonb
      )
     order by case upper(x.playoff_game_code)
       when 'SF1' then 1 when 'SF2' then 2
       when 'BRONZE' then 3 when 'FINAL' then 4 else 5 end
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted)), '[]'::jsonb)
    into v_saved from inserted;
  update public.tournament_event_draws draw
     set updated_at = clock_timestamp()
   where draw.id = v_draw.id;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, after_json, request_fingerprint
  ) values (
    p_club_id, v_draw.tournament_id, v_draw.event_option_id,
    'tournament_event_draw', v_draw.id::text,
    'team_playoffs_created', coalesce(nullif(p_actor, ''), 'unknown'),
    jsonb_build_object(
      'playoff_format', v_event.team_playoff_format, 'matchups', v_saved
    ),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'matchups', v_saved,
    'playoff_format', v_event.team_playoff_format,
    'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_append_tournament_team_playoffs_cas(
  text, text, text, timestamptz, jsonb, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_append_tournament_team_playoffs_cas(
  text, text, text, timestamptz, jsonb, text, text, text
) to service_role;

create or replace function public.admin_write_combined_rating_draw_teams_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_replace boolean,
  p_teams jsonb,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_saved jsonb;
  v_operation jsonb;
  v_result jsonb;
begin
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'import_handoff', 'combined_rating_draw_team_write',
    'tournament_event_draw', p_draw_id, p_actor
  );
  if coalesce((v_operation->>'replay')::boolean, false) then
    return v_operation->'result';
  end if;
  if pg_catalog.jsonb_typeof(coalesce(p_teams, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_teams) = 0 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_DRAW_TEAMS_INVALID';
  end if;
  select draw.* into v_draw
    from public.tournament_event_draws draw
    join public.tournaments tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for update of draw;
  if not found or p_expected_draw_updated_at is null
     or v_draw.updated_at is distinct from p_expected_draw_updated_at then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_DRAW_STALE';
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = v_draw.event_option_id::text
     and event.tournament_id = v_draw.tournament_id
     and event.eligibility_mode = 'COMBINED_RATING_CAP'
   for share;
  if not found then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_DRAW_REQUIRED';
  end if;
  perform team.id
    from public.tournament_teams team
   where team.tournament_id = v_draw.tournament_id
     and team.draw_id = v_draw.id
   order by team.id
   for update;
  if exists (
    select 1 from public.tournament_games game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
  ) or exists (
    select 1 from public.tournament_podium podium
     where podium.tournament_id = v_draw.tournament_id
       and podium.draw_id = v_draw.id
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_DRAW_IN_USE';
  end if;
  if (
    select count(distinct x.team_number)
      from pg_catalog.jsonb_to_recordset(p_teams) as x(
        team_number integer, player1_id integer, player2_id integer,
        source_selection_id text
      )
  ) <> pg_catalog.jsonb_array_length(p_teams)
  or exists (
    select 1
      from pg_catalog.jsonb_to_recordset(p_teams) as x(
        team_number integer, player1_id integer, player2_id integer,
        source_selection_id text
      )
     where x.team_number is null or x.team_number < 1
        or x.player1_id is null or x.player2_id is null
        or x.player1_id = x.player2_id
        or nullif(x.source_selection_id, '') is null
        or not exists (
          select 1
            from public.tournament_rating_eligibility_reviews review
           where review.tournament_id = v_draw.tournament_id
             and review.event_option_id::text = v_draw.event_option_id::text
             and review.selection_id::text = x.source_selection_id
             and review.review_phase = 'REGISTRATION_CLOSE'
             and review.finalized_at is not null
             and coalesce(review.override_state, review.state) = 'ELIGIBLE'
             and (
               review.override_state is null
               or nullif(btrim(review.override_reason), '') is not null
             )
             and review.partner_registration_id is not null
             and review.player_id_snapshot = x.player1_id
             and review.partner_player_id_snapshot = x.player2_id
             and exists (
               select 1
                 from public.tournament_registrations registration
                where registration.id = review.registration_id
                  and registration.tournament_id = v_draw.tournament_id
                  and registration.player_id = x.player1_id
                  and upper(coalesce(registration.status, '')) in (
                    'CONFIRMED', 'ADMIN_CONFIRMED'
                  )
             )
             and exists (
               select 1
                 from public.tournament_registrations partner_registration
                where partner_registration.id = review.partner_registration_id
                  and partner_registration.tournament_id = v_draw.tournament_id
                  and partner_registration.player_id = x.player2_id
                  and upper(coalesce(partner_registration.status, '')) in (
                    'CONFIRMED', 'ADMIN_CONFIRMED'
                  )
             )
        )
  ) or exists (
    select player_id
      from (
        select x.player1_id as player_id
          from pg_catalog.jsonb_to_recordset(p_teams) as x(
            player1_id integer, player2_id integer
          )
        union all
        select x.player2_id as player_id
          from pg_catalog.jsonb_to_recordset(p_teams) as x(
            player1_id integer, player2_id integer
          )
      ) players
     group by player_id
    having count(*) > 1
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_DRAW_BLOCKED';
  end if;
  if coalesce(p_replace, false) then
    delete from public.tournament_teams team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id;
  elsif exists (
    select 1
      from pg_catalog.jsonb_to_recordset(p_teams) as x(
        team_number integer, player1_id integer, player2_id integer
      )
      join public.tournament_teams existing
        on existing.tournament_id = v_draw.tournament_id
       and existing.draw_id = v_draw.id
       and (
         existing.team_number = x.team_number
         or existing.player1_id in (x.player1_id, x.player2_id)
         or existing.player2_id in (x.player1_id, x.player2_id)
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_DRAW_APPEND_CONFLICT';
  end if;
  with inserted as (
    insert into public.tournament_teams (
      id, tournament_id, draw_id, registration_day_id, event_option_id,
      team_number, player1_id, player2_id, seed, source, notes,
      source_selection_id, created_at, updated_at
    )
    select coalesce(nullif(x.id, '')::uuid, gen_random_uuid()),
           v_draw.tournament_id, v_draw.id, v_draw.registration_day_id,
           v_draw.event_option_id, x.team_number, x.player1_id, x.player2_id,
           x.seed, 'REGISTRATION_COMBINED_RATING',
           nullif(x.notes, ''), x.source_selection_id,
           coalesce(x.created_at, clock_timestamp()), clock_timestamp()
      from pg_catalog.jsonb_to_recordset(p_teams) as x(
        id text, team_number integer, player1_id integer, player2_id integer,
        seed integer, notes text, source_selection_id text,
        created_at timestamptz
      )
     order by x.team_number
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted) order by inserted.team_number), '[]'::jsonb)
    into v_saved from inserted;
  update public.tournament_event_draws draw
     set updated_at = clock_timestamp()
   where draw.id = v_draw.id;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, after_json, request_fingerprint
  ) values (
    p_club_id, v_draw.tournament_id, v_draw.event_option_id,
    'tournament_event_draw', v_draw.id::text,
    'combined_rating_draw_teams_written',
    coalesce(nullif(p_actor, ''), 'unknown'),
    jsonb_build_object('replace', p_replace, 'teams', v_saved),
    p_request_fingerprint
  );
  v_result := jsonb_build_object(
    'ok', true, 'teams', v_saved, 'operation_key', p_operation_key
  );
  return public.complete_tournament_team_operation(
    p_operation_key, p_request_fingerprint, v_result
  );
end;
$$;

revoke all on function public.admin_write_combined_rating_draw_teams_cas(
  text, text, text, timestamptz, boolean, jsonb, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_write_combined_rating_draw_teams_cas(
  text, text, text, timestamptz, boolean, jsonb, text, text, text
) to service_role;

-- Keep protected four-player parent/result assets out of canonical ratings,
-- even if a future caller bypasses the application-level publish preflight.
create or replace function public.enforce_tournament_team_official_match_source()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_game public.tournament_games%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_child public.tournament_team_match_games%rowtype;
  v_team_a public.tournament_teams%rowtype;
  v_team_b public.tournament_teams%rowtype;
  v_club_id text;
  v_new_association boolean;
begin
  if new.tournament_game_id is null then
    return new;
  end if;

  select game.* into v_game
    from public.tournament_games game
   where game.id = new.tournament_game_id;
  if not found then
    return new;
  end if;
  select draw.* into v_draw
    from public.tournament_event_draws draw
   where draw.id = v_game.draw_id;
  if not found or v_draw.draw_kind not in ('TEAM_PARENT', 'TEAM_RATING_CHILD') then
    return new;
  end if;
  if v_draw.draw_kind = 'TEAM_PARENT' then
    raise exception using errcode = '42501',
      message = 'JUPR_TOURNAMENT_TEAM_PARENT_MATCH_PUBLISH_FORBIDDEN';
  end if;

  select tournament.club_id::text into v_club_id
    from public.tournaments tournament
   where tournament.id = v_game.tournament_id;
  select child.* into v_child
    from public.tournament_team_match_games child
   where child.id = v_game.team_match_game_id
     and child.tournament_game_id = v_game.id
     and child.rating_draw_id = v_draw.id;
  select team.* into v_team_a
    from public.tournament_teams team
   where team.id = v_game.team_a_id
     and team.draw_id = v_draw.id
     and team.team_match_game_id = v_game.team_match_game_id
     and team.team_match_side = 'A'
     and team.source = 'FOUR_PLAYER_TEAM_CHILD';
  select team.* into v_team_b
    from public.tournament_teams team
   where team.id = v_game.team_b_id
     and team.draw_id = v_draw.id
     and team.team_match_game_id = v_game.team_match_game_id
     and team.team_match_side = 'B'
     and team.source = 'FOUR_PLAYER_TEAM_CHILD';

  if v_child.id is null or v_team_a.id is null or v_team_b.id is null
     or not v_child.counts_for_rating
     or v_game.parent_result_only
     or new.tournament_id is distinct from v_game.tournament_id
     or new.club_id::text is distinct from v_club_id
     or new.context_type is distinct from 'tournament_game'
     or new.context_id is distinct from v_game.id::text
     or coalesce(new.rating_bonus_elo, 0) <> 0
     or nullif(pg_catalog.btrim(coalesce(new.rating_bonus_reason, '')), '') is not null
     or new.t1_p1 is distinct from v_team_a.player1_id
     or new.t1_p2 is distinct from v_team_a.player2_id
     or new.t2_p1 is distinct from v_team_b.player1_id
     or new.t2_p2 is distinct from v_team_b.player2_id
     or lower(coalesce(new.match_format, '')) is distinct from
        lower(v_child.match_format) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_RATING_MATCH_SOURCE_INVALID';
  end if;

  v_new_association := tg_op = 'INSERT';
  if tg_op = 'UPDATE' then
    v_new_association :=
      old.tournament_game_id is distinct from new.tournament_game_id;
  end if;
  if v_new_association and (
    v_child.status <> 'FINAL'
    or new.score_t1 is distinct from v_child.score_a
    or new.score_t2 is distinct from v_child.score_b
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_TEAM_RATING_MATCH_NOT_FINAL';
  end if;
  return new;
end;
$$;

revoke all on function public.enforce_tournament_team_official_match_source()
  from public, anon, authenticated;

drop trigger if exists trg_enforce_tournament_team_official_match_source
  on public.matches;
create trigger trg_enforce_tournament_team_official_match_source
before insert or update on public.matches
for each row execute function public.enforce_tournament_team_official_match_source();
