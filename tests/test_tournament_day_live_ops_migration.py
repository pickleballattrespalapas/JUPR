from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase/migrations/20261102000000_tournament_day_live_ops.sql"

DAY_LIVE_TABLES = (
    "tournament_day_live_runs",
    "tournament_day_live_draws",
    "tournament_day_live_courts",
    "tournament_day_live_queue",
    "tournament_day_live_participant_claims",
)

DAY_LIVE_RPCS = (
    "admin_activate_tournament_day_live_cas",
    "admin_transition_tournament_day_draw_cas",
    "admin_fill_tournament_day_courts_cas",
    "admin_close_tournament_day_live_cas",
    "admin_score_release_tournament_day_game_cas",
    "admin_correct_completed_tournament_day_game_cas",
    "admin_generate_tournament_day_playoffs_cas",
)

DAY_LIVE_HELPERS = (
    "assert_tournament_day_live_operation",
    "guard_tournament_game_day_live_mutation",
    "guard_tournament_check_in_during_player_claim",
    "guard_tournament_team_during_player_claim",
    "tournament_day_live_game_player_ids",
    "assert_tournament_day_live_draw_ready",
    "tournament_day_live_players_ready",
    "seed_tournament_day_live_draw",
    "fill_tournament_day_live_courts",
)


def _sql() -> str:
    assert MIGRATION.exists(), "add the durable Tournament Day Live migration"
    return MIGRATION.read_text(encoding="utf-8").lower()


def _function_body(sql: str, function_name: str) -> str:
    match = re.search(
        rf"create\s+or\s+replace\s+function\s+public\.{function_name}\b"
        rf"(?P<header>.*?)\bas\s+\$(?P<tag>[a-z_]*)\$"
        rf"(?P<body>.*?)\$(?P=tag)\$;",
        sql,
        flags=re.DOTALL,
    )
    assert match, f"missing RPC {function_name}"
    return f"{match.group('header')}\n{match.group('body')}"


def test_day_live_schema_is_relational_day_scoped_and_conflict_safe() -> None:
    sql = _sql()

    assert "to_regclass('public.tournament_commerce_orders')" in sql
    assert "to_regclass('public.matches')" in sql

    for table in DAY_LIVE_TABLES:
        assert f"create table if not exists public.{table}" in sql

    # A run is the durable authority for one tournament day. Draws, courts,
    # queued games and player claims must all attach to that run rather than
    # treating the existing JSON court/day configuration as live truth.
    assert "unique (tournament_id, registration_day_id)" in sql
    assert "references public.tournament_registration_days(id)" in sql
    assert "references public.tournament_event_draws(id)" in sql
    assert "references public.tournament_games(id)" in sql
    assert "references public.tournament_teams(id)" in sql
    assert "references public.players(id)" in sql

    # These exact resource constraints are the database-level race authority:
    # one draw membership, one queue row per game, one occupied game per court,
    # and one active assignment per player across every draw on the day.
    assert "unique (run_id, draw_id)" in sql
    assert "unique (run_id, game_id)" in sql
    assert "unique (run_id, court_key)" in sql
    assert re.search(
        r"create\s+unique\s+index\s+(?:if\s+not\s+exists\s+)?"
        r"\S+\s+on\s+"
        r"public\.tournament_day_live_participant_claims\s*\(run_id,\s*player_id\)\s*"
        r"where\s+released_at\s+is\s+null",
        sql,
    )
    assert "where completed_at is null" in sql or "where released_at is null" in sql

    assert "check (state in" in sql
    for state in ("draft", "active", "paused", "closed"):
        assert f"'{state}'" in sql
    for state in ("waiting", "held", "called", "on_court", "completed", "blocked", "withdrawn"):
        assert f"'{state}'" in sql


def test_day_live_tables_are_force_rls_service_only() -> None:
    sql = _sql()

    for table in DAY_LIVE_TABLES:
        assert f"alter table public.{table} enable row level security" in sql
        assert f"alter table public.{table} force row level security" in sql
        assert (
            f"revoke all on table public.{table} from public, anon, authenticated, service_role"
            in sql
        )
        assert re.search(
            rf"grant\s+(?:select|select,\s*insert,\s*update,\s*delete)[^;]*"
            rf"on\s+table\s+public\.{table}\s+to\s+service_role",
            sql,
        )

    # The API must remain the only browser-facing write authority.
    assert "grant" not in "\n".join(
        line
        for line in sql.splitlines()
        if "on table public.tournament_day_live_" in line
        and (" to anon" in line or " to authenticated" in line)
    )


def test_day_live_rpcs_bind_scope_operation_identity_cas_and_service_role() -> None:
    sql = _sql()

    for rpc in DAY_LIVE_RPCS:
        body = _function_body(sql, rpc)
        assert "security invoker" in body
        assert "set search_path = ''" in body
        assert "p_club_id" in body
        assert "p_tournament_id" in body
        assert "p_registration_day_id" in body
        assert "p_operation_key" in body
        assert "p_request_fingerprint" in body
        assert "for update" in body
        assert (
            f"revoke execute on function public.{rpc}" in sql
        ), f"{rpc} must revoke PostgreSQL's default PUBLIC execute grant"
        assert re.search(
            rf"revoke\s+execute\s+on\s+function\s+public\.{rpc}\([^;]*\)\s+"
            rf"from\s+public,\s*anon,\s*authenticated",
            sql,
            flags=re.DOTALL,
        )
        assert re.search(
            rf"grant\s+execute\s+on\s+function\s+public\.{rpc}\([^;]*\)\s+"
            rf"to\s+service_role",
            sql,
            flags=re.DOTALL,
        )

    # Operation identity is checked under lock by SQL, not trusted merely
    # because FastAPI created a similarly named ledger row.
    assert "from public.tournament_admin_operations" in sql
    assert "operation_key" in sql
    assert "request_fingerprint" in sql
    assert "jupr_tournament_day_live_operation" in sql
    assert "jupr_tournament_day_live_stale" in sql

    activation = _function_body(sql, "admin_activate_tournament_day_live_cas")
    assert "p_expected_run_version" in activation
    assert "p_activation_evidence jsonb" in activation
    assert "{payload,activation_evidence}" in activation
    assert "v_current_activation_evidence" in activation
    assert "is distinct from p_activation_evidence" in activation
    assert "jsonb_agg" in activation
    assert "court_key" in activation and "position" in activation
    for rpc in (
        "admin_transition_tournament_day_draw_cas",
        "admin_fill_tournament_day_courts_cas",
        "admin_score_release_tournament_day_game_cas",
        "admin_generate_tournament_day_playoffs_cas",
    ):
        body = _function_body(sql, rpc)
        assert "p_expected_run_version" in body
        assert "p_expected_queue_version" in body
        assert "run.queue_version = p_expected_queue_version" in body
    score = _function_body(sql, "admin_score_release_tournament_day_game_cas")
    assert "p_expected_court_version" in score
    assert "court.version = p_expected_court_version" in score


def test_public_day_live_helpers_do_not_keep_postgres_default_execute() -> None:
    sql = _sql()

    for helper in DAY_LIVE_HELPERS:
        body = _function_body(sql, helper)
        assert "security invoker" in body
        assert "set search_path = ''" in body
        assert re.search(
            rf"revoke\s+execute\s+on\s+function\s+public\.{helper}\([^;]*\)\s+"
            rf"from\s+public,\s*anon,\s*authenticated",
            sql,
            flags=re.DOTALL,
        ), f"{helper} keeps PostgreSQL's default PUBLIC execute grant"
        assert re.search(
            rf"grant\s+execute\s+on\s+function\s+public\.{helper}\([^;]*\)\s+"
            rf"to\s+service_role",
            sql,
            flags=re.DOTALL,
        )


def test_day_live_function_ddl_has_basic_static_syntax_sanity() -> None:
    sql = _sql()
    functions = (*DAY_LIVE_HELPERS, *DAY_LIVE_RPCS)

    for function_name in functions:
        function = _function_body(sql, function_name)
        assert function.count("language ") == 1, (
            f"{function_name} repeats or omits its LANGUAGE option"
        )
        assert not re.search(r"(?m)^\s*end;\s*\n\s*end;\s*$", function)

    activation = _function_body(sql, "admin_activate_tournament_day_live_cas")
    assert "p_draw_plan" not in activation
    assert "individual_draw_activation" in activation
    assert "seed_tournament_day_live_draw" not in activation

    fill = _function_body(sql, "fill_tournament_day_live_courts")
    # A PL/pgSQL composite/%%ROWTYPE target cannot share a multi-item INTO
    # list with a scalar target (PostgreSQL 42601). Select the queue row into
    # its rowtype alone; derive player ids only after the candidate is locked.
    assert "into v_queue, v_player_ids" not in fill
    assert "select queue.* into v_queue" in fill
    assert fill.index("select queue.* into v_queue") < fill.index(
        "v_player_ids := public.tournament_day_live_game_player_ids"
    )


def test_draw_transitions_lock_and_revalidate_the_reviewed_source_scope() -> None:
    sql = _sql()
    transition = _function_body(sql, "admin_transition_tournament_day_draw_cas")
    readiness = _function_body(sql, "assert_tournament_day_live_draw_ready")
    seed = _function_body(sql, "seed_tournament_day_live_draw")

    # Activation must use the child-before-parent order used by existing game
    # and team version triggers.  Holding the source draw and then waiting for
    # a child row can deadlock an updater that already holds the child and is
    # waiting to touch the draw version.
    assert readiness.index("select day.* into v_day") < (
        readiness.index("select event.* into v_event")
    )
    assert readiness.index("select event.* into v_event") < (
        readiness.index("perform team.id")
    )
    assert readiness.index("perform team.id") < (
        readiness.index("perform game.id")
    )
    assert readiness.index("perform game.id") < (
        readiness.index("select draw.*")
    )
    assert "for share" in readiness
    assert "day.enabled is true" in readiness
    assert "v_event.enabled is not true" in readiness
    assert "v_draw.updated_at is distinct from p_expected_draw_updated_at" in readiness
    assert "hidden_from_primary_ops" in readiness
    assert "draw_kind" in readiness

    # Seed delegates lock ownership to the readiness boundary rather than
    # taking a source-draw lock before child locks in a second order.
    assert seed.index("assert_tournament_day_live_draw_ready") < seed.index(
        "insert into public.tournament_day_live_draws"
    )
    assert "perform team.id" not in seed
    assert "perform game.id" not in seed

    # Resume rechecks the runnable day/event/draw scope after locking it.
    # Pause intentionally skips runnable-state membership so it remains an
    # emergency stop, but every action must CAS the exact reviewed draw row.
    assert "v_action = 'resume'" in transition
    assert "from public.tournament_registration_days as day" in transition
    assert "from public.tournament_event_options as event" in transition
    assert "from public.tournament_event_draws as draw" in transition
    assert transition.index("select day.* into v_day") < transition.index(
        "select event.* into v_event"
    )
    assert transition.index("select event.* into v_event") < transition.index(
        "perform team.id"
    )
    assert transition.index("perform team.id") < transition.index(
        "perform game.id"
    )
    assert transition.index("perform game.id") < transition.index(
        "select draw.* into v_draw"
    )
    assert "v_draw.updated_at is distinct from p_expected_draw_updated_at" in transition
    assert "v_draw.event_option_id is distinct from v_event.id" in transition
    assert "jupr_tournament_day_live_draw_scope" in transition


def test_draw_activation_rejects_unsupported_or_stranded_game_shapes() -> None:
    sql = _sql()
    readiness = _function_body(sql, "assert_tournament_day_live_draw_ready")

    # The day runner can progress only the two engine stages it understands.
    # A null-sided RR game can never be resolved by bracket propagation, so it
    # must fail before durable draw membership is created.
    assert "not in ('round_robin', 'playoff')" in readiness
    assert "jupr_tournament_day_live_game_stage" in readiness
    assert "jupr_tournament_day_live_round_robin" in readiness
    assert "game.team_a_id is null" in readiness
    assert "game.team_b_id is null" in readiness
    assert "participant_counts.player_count not in (2, 4)" in readiness
    assert "team_a.registration_day_id = p_registration_day_id" in readiness
    assert "team_b.registration_day_id = p_registration_day_id" in readiness
    assert "team_a.event_option_id = v_draw.event_option_id" in readiness
    assert "team_b.event_option_id = v_draw.event_option_id" in readiness
    assert "team.registration_day_id is distinct from p_registration_day_id" in readiness
    assert "team.event_option_id is distinct from v_draw.event_option_id" in readiness
    assert "jupr_tournament_day_live_team_day_scope" in readiness
    assert "jupr_tournament_day_live_roster_player_duplicate" in readiness
    assert "array_agg(distinct side.team_id order by side.team_id)" in readiness
    assert "jupr_tournament_day_live_round_robin_roster" in readiness

    # Initial activation is RR-only.  Existing playoff rows—including a valid
    # looking but cyclic unresolved graph—must enter only through the guarded
    # generate-playoffs operation after RR completion.
    assert "= 'playoff'" in readiness
    assert "jupr_tournament_day_live_playoffs_already_generated" in readiness

    # Closeout requires a supported 4/5/6-team playoff plan, so a draw with no
    # possible advance count must not become an uncloseable durable day draw.
    assert "from public.tournament_teams as team" in readiness
    assert ") < 4 then" in readiness
    assert "jupr_tournament_day_live_playoff_format" in readiness

    players = _function_body(sql, "tournament_day_live_game_player_ids")
    assert "team_a.registration_day_id = game.registration_day_id" in players
    assert "team_b.registration_day_id = game.registration_day_id" in players
    assert "team_a.event_option_id = game.event_option_id" in players
    assert "team_b.event_option_id = game.event_option_id" in players


def test_fill_and_score_release_are_atomic_across_draws_courts_and_players() -> None:
    sql = _sql()
    fill = "\n".join(
        (
            _function_body(sql, "admin_fill_tournament_day_courts_cas"),
            _function_body(sql, "fill_tournament_day_live_courts"),
        )
    )
    score = _function_body(sql, "admin_score_release_tournament_day_game_cas")

    # The shared scheduler must lock candidates deterministically and claim
    # every participant before publishing an assignment.
    assert "tournament_day_live_queue" in fill
    assert "tournament_day_live_courts" in fill
    assert "tournament_day_live_participant_claims" in fill
    assert "order by" in fill
    assert "for update" in fill
    assert "on conflict" in fill
    assert "player_id" in fill
    assert "state = 'on_court'" in fill
    assert "'on_court', 1, p_operation_key" in fill

    # Every assignment is fenced against setup drift, including the refill
    # performed by score+release.  The candidate filter avoids stale draws and
    # the post-lock check closes the filter/use race with setup edits.
    assert "join public.tournament_registration_days as source_day" in fill
    assert "join public.tournament_event_options as source_event" in fill
    assert "join public.tournament_event_draws as source_draw" in fill
    assert "source_draw.updated_at = day_draw.source_draw_updated_at" in fill
    assert "source_day.enabled is true" in fill
    assert "source_event.enabled is true" in fill
    assert "source_draw.hidden_from_primary_ops" in fill
    assert "select locked_game.* into v_game" in fill
    assert fill.index("perform locked_team.id") < fill.index(
        "select locked_game.* into v_game"
    )
    assert fill.index("select locked_game.* into v_game") < fill.index(
        "select source_day.* into v_day"
    )
    assert fill.index("select source_day.* into v_day") < fill.index(
        "select source_event.* into v_event"
    )
    assert fill.index("select source_event.* into v_event") < fill.index(
        "select source_draw.* into v_draw"
    )
    assert "v_draw.updated_at is distinct from v_day_draw.source_draw_updated_at" in fill
    assert "locked_team.registration_day_id = v_queue.registration_day_id" in fill
    assert "locked_team.event_option_id" in fill

    # Score finalization, dependency propagation, court release, participant
    # release, and filling the next eligible game are one SQL transaction.
    assert "tournament_games" in score
    assert "score_a" in score and "score_b" in score
    assert "winner_team_id" in score and "loser_team_id" in score
    assert "tournament_day_live_courts" in score
    assert "tournament_day_live_participant_claims" in score
    assert "released_at" in score
    assert "array_agg" in score and "claim.player_id" in score
    assert "admin_fill_tournament_day_courts_cas" in score or "tournament_day_live_queue" in score
    assert "dependency" in score or "team_a_source" in score
    assert "v_locked_game_ids" in score
    assert "array_append(v_dependency_ids, v_queue.game_id)" in score
    assert "order by game.id" in score
    assert "dependency_queue.state = 'blocked'" in score
    assert "dependency_queue.court_id is null" in score
    assert "downstream.team_a_source" in score
    assert "downstream.team_b_source" in score
    assert "downstream updates may resolve teams only" in score
    assert "dependency.value->field.key is distinct from 'null'::jsonb" in score
    assert "queue.game_id = any(v_dependency_ids)" in score
    assert "v_expected_dependency_ids" in score
    assert "downstream.team_a_source->>'winnerof' = v_game.playoff_game_code" in score
    assert "downstream.team_b_source->>'loserof' = v_game.playoff_game_code" in score
    assert "v_dependency_ids is distinct from v_expected_dependency_ids" in score


def test_playoff_generation_stays_inside_the_active_day_fence() -> None:
    sql = _sql()
    playoff = _function_body(sql, "admin_generate_tournament_day_playoffs_cas")

    assert "tournament_day_live_runs" in playoff
    assert "tournament_day_live_draws" in playoff
    assert "tournament_day_live_queue" in playoff
    assert "tournament_games" in playoff
    assert "round_robin" in playoff
    assert "finalized_at" in playoff
    assert "score_a" in playoff and "score_b" in playoff
    assert "admin_insert_tournament_draw_games_cas" in playoff
    assert "fill_tournament_day_live_courts" in playoff
    assert "p_expected_team_versions" in playoff
    assert "p_expected_source_game_versions" in playoff
    assert "p_expected_draw_version" in playoff
    assert "p_expected_run_version" in playoff
    assert "p_advance_count" in playoff
    assert "p_advance_count not in (4, 5, 6)" in playoff
    assert "v_requested_game_ids" in playoff
    assert "v_inserted_game_ids is distinct from v_requested_game_ids" in playoff
    assert "v_queue_insert_count" in playoff
    assert "v_queued_game_ids is distinct from v_requested_game_ids" in playoff
    assert "planned.registration_day_id is distinct from p_registration_day_id" in playoff
    assert "planned.event_option_id is distinct from v_draw.event_option_id::text" in playoff
    assert "held" in playoff and "called" in playoff and "on_court" in playoff
    assert "jupr_tournament_day_live_playoff" in playoff
    assert "select source_day.* into v_day" in playoff
    assert "select source_event.* into v_event" in playoff
    assert "perform locked_team.id" in playoff
    assert "perform locked_game.id" in playoff
    assert "select source_draw.* into v_draw" in playoff
    assert playoff.index("select source_day.* into v_day") < playoff.index(
        "select source_event.* into v_event"
    )
    assert playoff.index("select source_event.* into v_event") < playoff.index(
        "perform locked_team.id"
    )
    assert playoff.index("perform locked_team.id") < playoff.index(
        "perform locked_game.id"
    )
    assert playoff.index("perform locked_game.id") < playoff.index(
        "select source_draw.* into v_draw"
    )
    assert "source_day.enabled is true" in playoff
    assert "v_event.enabled is not true" in playoff
    assert "v_draw.hidden_from_primary_ops" in playoff
    assert "v_draw.updated_at is distinct from p_expected_draw_version" in playoff
    assert "jupr_tournament_day_live_playoff_team_scope" in playoff
    assert "jupr_tournament_day_live_playoff_roster_player_duplicate" in playoff
    assert "jupr_tournament_day_live_playoff_roster" in playoff

    activation = _function_body(sql, "admin_activate_tournament_day_live_cas")
    assert "from public.tournament_event_draws as source_draw" in activation
    assert "join public.tournament_event_options as source_event" in activation
    assert "jupr_tournament_day_live_draws" in activation


def test_completed_score_correction_is_day_owned_and_fail_closed() -> None:
    sql = _sql()
    correction = _function_body(
        sql, "admin_correct_completed_tournament_day_game_cas"
    )

    assert "tournament_day_live_correct_completed_score" in correction
    assert "state = 'completed'" in correction
    assert "court_id is null" in correction
    assert "released_at is not null" in correction
    assert "tournament_day_live_participant_claims" in correction
    assert "released_at is null" in correction
    assert "stage = 'round_robin'" in correction
    assert "playoff_reset_required" in correction
    assert "tournament_podium" in correction
    assert "public.matches" in correction
    assert "admin_score_tournament_game_cas" in correction
    assert "p_dependency_updates is distinct from '[]'::jsonb" in correction
    assert "source_draw_updated_at" in correction
    assert "queue.version + 1" in correction
    assert "day_draw.version + 1" in correction
    assert "queue_version = run.queue_version + 1" in correction


def test_close_day_requires_reviewed_playoffs_podium_awards_and_closes_courts() -> None:
    sql = _sql()
    close = _function_body(sql, "admin_close_tournament_day_live_cas")

    assert "game.stage = 'playoff'" in close
    assert "tournament_podium" in close
    assert "podium_review_evidence" in close
    assert "review_fingerprint" in close
    assert "draw_updated_at" not in close
    assert "player_badges" in close
    assert "expected_awards" in close and "actual_awards" in close
    assert "jupr_tournament_day_live_close_draws" in close
    assert "source_draw.hidden_from_primary_ops" in close
    assert "day_draw.state <> 'removed'" in close
    assert "first_place.team_id = final_game.winner_team_id" in close
    assert "second_place.team_id = final_game.loser_team_id" in close
    assert "third_place.team_id = bronze_game.winner_team_id" in close
    assert "jupr_tournament_day_live_close_podium_result" in close
    assert "set state = 'closed'" in close
    assert "tournament_day_live_courts" in close
    setup_lock = (
        "lock table public.tournament_registration_days, "
        "public.tournament_event_options, public.tournament_event_draws "
        "in share mode"
    )
    assert setup_lock in " ".join(close.split())
    assert close.index("lock table") < close.index(
        "from public.tournament_registration_days as source_day"
    )
    assert close.index(
        "from public.tournament_registration_days as source_day"
    ) < close.index("from public.tournament_podium")
    assert close.index("from public.tournament_event_draws as source_draw") > close.index(
        "from public.player_badges"
    )
    assert close.index("from public.tournament_podium") < close.index(
        "from public.tournament_teams"
    )
    assert close.index("from public.tournament_teams") < close.index(
        "from public.tournament_games"
    )
    assert close.index("from public.tournament_games") < close.index(
        "from public.player_badges"
    )


def test_legacy_score_rpc_cannot_bypass_an_active_day_queue() -> None:
    sql = _sql()

    # Existing callers may still use the draw-scoped score RPC for games that
    # are not day-managed. Once a game is assigned/active on the day board it
    # must be scored only by the score+release transaction so resource claims
    # cannot be orphaned.
    assert "admin_score_tournament_game_cas" in sql
    assert "tournament_day_live_queue" in sql
    assert "jupr_tournament_day_live_score_path_required" in sql
    assert "score_and_release" in sql
    assert "guard_tournament_game_day_live_mutation" in sql
    assert "before insert or update or delete on public.tournament_games" in sql
    assert "current_setting('jupr.day_live_operation_key', true)" in sql
    assert "operation.status = 'intent'" in sql
    assert "tournament_day_live_generate_playoffs" in sql

    # Check-in identity/attendance cannot change while the same canonical
    # player is claimed by an active assignment.
    assert "guard_tournament_check_in_during_player_claim" in sql
    assert "before update on public.tournament_registration_check_ins" in sql
    assert "jupr_tournament_day_live_player_claim" in sql
