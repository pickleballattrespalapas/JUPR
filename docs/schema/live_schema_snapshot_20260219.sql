-- Live schema snapshot generated from provided export on 2026-02-19.
-- Nullability was not provided in the source export; columns are marked NULL in this baseline.

CREATE TABLE badge_eval_queue (
  id uuid NULL,
  created_at timestamp with time zone NULL,
  club_id text NULL,
  context_id text NULL,
  event_type text NULL,
  player_ids ARRAY NULL,
  match_id text NULL,
  payload_json jsonb NULL,
  status text NULL,
  attempts integer NULL,
  last_error text NULL,
  processed_at timestamp with time zone NULL
);

CREATE TABLE badge_eval_runs (
  id uuid NULL,
  created_at timestamp with time zone NULL,
  created_by text NULL,
  mode text NULL,
  scope_json jsonb NULL,
  status text NULL,
  started_at timestamp with time zone NULL,
  finished_at timestamp with time zone NULL,
  summary_json jsonb NULL,
  error text NULL
);

CREATE TABLE badge_events_v2 (
  id uuid NULL,
  user_id uuid NULL,
  event_type text NULL,
  event_ts timestamp with time zone NULL,
  data jsonb NULL,
  idempotency_key text NULL,
  created_at timestamp with time zone NULL
);

CREATE TABLE badge_fact_registry (
  fact_key text NULL,
  description text NULL,
  data_type text NULL,
  allowed_scope text NULL
);

CREATE TABLE badge_rule_conditions (
  id uuid NULL,
  badge_id uuid NULL,
  fact_key text NULL,
  operator text NULL,
  value_numeric numeric NULL,
  created_at timestamp with time zone NULL
);

CREATE TABLE badge_rules_v2 (
  id uuid NULL,
  badge_id uuid NULL,
  rule_type text NULL,
  params jsonb NULL,
  trigger_event text NULL,
  evaluation_mode text NULL,
  enabled boolean NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL
);

CREATE TABLE badges (
  badge_id text NULL,
  name text NULL,
  prestige integer NULL,
  category text NULL,
  is_stackable boolean NULL,
  is_active boolean NULL,
  created_at timestamp with time zone NULL,
  rarity text NULL,
  tier integer NULL,
  icon_key text NULL,
  lore text NULL,
  hint text NULL,
  scope text NULL,
  status text NULL,
  is_locked boolean NULL,
  award_count integer NULL,
  published_at timestamp with time zone NULL,
  archived_at timestamp with time zone NULL,
  club_id text NULL,
  is_system_badge boolean NULL,
  created_by_admin_id text NULL
);

CREATE TABLE badges_v2 (
  id uuid NULL,
  slug text NULL,
  name text NULL,
  display_text text NULL,
  description text NULL,
  icon_url text NULL,
  icon_emoji text NULL,
  category text NULL,
  rarity text NULL,
  sort_order integer NULL,
  is_active boolean NULL,
  is_hidden boolean NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL,
  status text NULL,
  is_locked boolean NULL,
  award_count integer NULL,
  published_at timestamp with time zone NULL,
  archived_at timestamp with time zone NULL,
  club_id text NULL,
  is_system_badge boolean NULL,
  created_by_admin_id text NULL
);

CREATE TABLE club_user_roles (
  club_id text NULL,
  user_id uuid NULL,
  role text NULL,
  assigned_by uuid NULL,
  assigned_at timestamp with time zone NULL
);

CREATE TABLE clubs (
  id text NULL,
  name text NULL,
  status text NULL,
  logo_url text NULL,
  primary_color text NULL,
  created_at timestamp with time zone NULL
);

CREATE TABLE events (
  id uuid NULL,
  club_id text NULL,
  name text NULL,
  event_type text NULL,
  is_active boolean NULL,
  created_at timestamp with time zone NULL,
  starts_at timestamp with time zone NULL,
  ends_at timestamp with time zone NULL,
  notes text NULL
);

CREATE TABLE ladder_audit_log (
  id bigint NULL,
  club_id text NULL,
  actor text NULL,
  action_type text NULL,
  entity_type text NULL,
  entity_id text NULL,
  before jsonb NULL,
  after jsonb NULL,
  created_at timestamp with time zone NULL
);

CREATE TABLE ladder_challenge_matches (
  id bigint NULL,
  club_id text NULL,
  challenge_id bigint NULL,
  match_no integer NULL,
  def_partner_id bigint NULL,
  chal_partner_id bigint NULL,
  g1_def integer NULL,
  g1_chal integer NULL,
  g2_def integer NULL,
  g2_chal integer NULL,
  g3_def integer NULL,
  g3_chal integer NULL,
  verified boolean NULL,
  verified_at timestamp with time zone NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL
);

CREATE TABLE ladder_challenges (
  id bigint NULL,
  club_id text NULL,
  challenger_id bigint NULL,
  defender_id bigint NULL,
  challenger_rank_at_create integer NULL,
  defender_rank_at_create integer NULL,
  status text NULL,
  ledger_ref text NULL,
  created_by text NULL,
  created_at timestamp with time zone NULL,
  accept_by timestamp with time zone NULL,
  accepted_at timestamp with time zone NULL,
  play_by timestamp with time zone NULL,
  scheduled_for timestamp with time zone NULL,
  pass_used_by bigint NULL,
  pass_used_at timestamp with time zone NULL,
  forfeit_by bigint NULL,
  forfeit_reason text NULL,
  winner_id bigint NULL,
  completed_at timestamp with time zone NULL,
  resolution_notes text NULL,
  tier_id text NULL,
  notice_sent_at timestamp with time zone NULL
);

CREATE TABLE ladder_pass_usage (
  id bigint NULL,
  club_id text NULL,
  player_id bigint NULL,
  month_key text NULL,
  used_at timestamp with time zone NULL,
  challenge_id bigint NULL,
  created_at timestamp with time zone NULL
);

CREATE TABLE ladder_player_flags (
  club_id text NULL,
  player_id bigint NULL,
  vacation_until timestamp with time zone NULL,
  reinstate_required boolean NULL,
  reinstate_notes text NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL,
  tier_move_flag boolean NULL,
  tier_move_dest_tier text NULL,
  tier_move_count integer NULL,
  tier_move_triggered_at timestamp with time zone NULL,
  tier_move_last_eval_at timestamp with time zone NULL
);

CREATE TABLE ladder_roster (
  id bigint NULL,
  club_id text NULL,
  player_id bigint NULL,
  rank integer NULL,
  is_active boolean NULL,
  joined_at timestamp with time zone NULL,
  left_at timestamp with time zone NULL,
  notes text NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL,
  tier_id text NULL
);

CREATE TABLE ladder_settings (
  club_id text NULL,
  challenge_range integer NULL,
  accept_window_hours integer NULL,
  play_window_days integer NULL,
  cooldown_hours integer NULL,
  protected_hours integer NULL,
  pass_hold_hours integer NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL
);

CREATE TABLE league_ratings (
  id bigint NULL,
  player_id bigint NULL,
  club_id text NULL,
  league_name text NULL,
  rating numeric NULL,
  matches_played integer NULL,
  wins integer NULL,
  losses integer NULL,
  starting_rating numeric NULL,
  is_active boolean NULL,
  inactive_at timestamp with time zone NULL
);

CREATE TABLE leagues_metadata (
  id bigint NULL,
  club_id text NULL,
  league_name text NULL,
  league_type text NULL,
  min_weeks integer NULL,
  is_active boolean NULL,
  created_at timestamp with time zone NULL,
  min_games integer NULL,
  description text NULL,
  k_factor integer NULL,
  ended_at timestamp with time zone NULL,
  ended_by text NULL,
  status text NULL,
  end_awards jsonb NULL,
  started_at timestamp with time zone NULL,
  schedule_config jsonb NULL,
  court_board_defaults jsonb NULL,
  rules_config jsonb NULL,
  awards_config jsonb NULL
);

CREATE TABLE matches (
  id bigint NULL,
  club_id text NULL,
  date timestamp with time zone NULL,
  league text NULL,
  t1_p1 bigint NULL,
  t1_p2 bigint NULL,
  t2_p1 bigint NULL,
  t2_p2 bigint NULL,
  score_t1 integer NULL,
  score_t2 integer NULL,
  elo_delta numeric NULL,
  match_type text NULL,
  week_tag text NULL,
  t1_p1_r numeric NULL,
  t1_p2_r numeric NULL,
  t2_p1_r numeric NULL,
  t2_p2_r numeric NULL,
  t1_p1_r_end numeric NULL,
  t1_p2_r_end numeric NULL,
  t2_p1_r_end numeric NULL,
  t2_p2_r_end numeric NULL,
  elo_delta_t1 double precision NULL,
  elo_delta_t2 double precision NULL,
  context_type text NULL,
  context_id uuid NULL,
  tournament_id uuid NULL,
  tournament_game_id uuid NULL,
  idempotency_key text NULL
);

CREATE TABLE player_badge_facts (
  club_id text NULL,
  player_id bigint NULL,
  context_id text NULL,
  fact_key text NULL,
  fact_value_num double precision NULL,
  updated_at timestamp with time zone NULL
);

CREATE TABLE player_badges (
  id uuid NULL,
  club_id text NULL,
  player_id bigint NULL,
  badge_id text NULL,
  earned_at timestamp with time zone NULL,
  context_type text NULL,
  context_id text NULL,
  match_id text NULL,
  value_num numeric NULL,
  value_json jsonb NULL,
  awarded_by uuid NULL,
  rule_version text NULL,
  eval_run_id uuid NULL,
  revoked_at timestamp with time zone NULL,
  revoked_by uuid NULL,
  revoked_reason text NULL,
  revoke_reason text NULL
);

CREATE TABLE player_profiles (
  club_id text NULL,
  player_id bigint NULL,
  preferred_channel text NULL,
  email text NULL,
  phone text NULL,
  whatsapp text NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL
);

CREATE TABLE players (
  id bigint NULL,
  club_id text NULL,
  name text NULL,
  rating numeric NULL,
  starting_rating numeric NULL,
  matches_played integer NULL,
  wins integer NULL,
  losses integer NULL,
  created_at timestamp with time zone NULL,
  active boolean NULL,
  last_game_at timestamp with time zone NULL,
  inactive_at timestamp with time zone NULL,
  normalized_name text NULL
);

CREATE TABLE processed_match_facts (
  club_id text NULL,
  match_id text NULL,
  player_id bigint NULL
);

CREATE TABLE tournament_games (
  id uuid NULL,
  tournament_id uuid NULL,
  stage text NULL,
  rr_round_number integer NULL,
  rr_slot_number integer NULL,
  playoff_game_code text NULL,
  playoff_round text NULL,
  team_a_id uuid NULL,
  team_b_id uuid NULL,
  team_a_source jsonb NULL,
  team_b_source jsonb NULL,
  score_a integer NULL,
  score_b integer NULL,
  winner_team_id uuid NULL,
  loser_team_id uuid NULL,
  finalized_at timestamp with time zone NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL,
  series_game_number integer NULL
);

CREATE TABLE tournament_podium (
  id uuid NULL,
  tournament_id uuid NULL,
  placement integer NULL,
  team_id uuid NULL,
  source text NULL,
  created_at timestamp with time zone NULL
);

CREATE TABLE tournament_teams (
  id uuid NULL,
  tournament_id uuid NULL,
  team_number integer NULL,
  player1_id integer NULL,
  player2_id integer NULL,
  seed integer NULL
);

CREATE TABLE tournaments (
  id uuid NULL,
  club_id text NULL,
  name text NULL,
  status text NULL,
  team_count integer NULL,
  playoff_advance_count integer NULL,
  created_by_admin_id text NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL,
  playoff_best_of integer NULL
);

CREATE TABLE user_badges_v2 (
  user_id uuid NULL,
  badge_id uuid NULL,
  awarded_at timestamp with time zone NULL,
  awarded_by uuid NULL,
  reason text NULL,
  metadata jsonb NULL
);

CREATE TABLE weekly_recaps (
  id uuid NULL,
  club_id text NULL,
  week_start date NULL,
  week_end date NULL,
  status text NULL,
  generated_json jsonb NULL,
  edits_json jsonb NULL,
  final_json jsonb NULL,
  created_at timestamp with time zone NULL,
  updated_at timestamp with time zone NULL,
  published_at timestamp with time zone NULL,
  published_by text NULL
);
