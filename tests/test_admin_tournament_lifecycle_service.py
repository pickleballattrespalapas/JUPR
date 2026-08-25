from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
)
from jupr_app.domain.tournament_podium import PODIUM_BADGE_MAP
from jupr_app.services.admin_tournament_lifecycle_service import (
    build_admin_tournament_lifecycle,
    require_admin_tournament_official_publish_readiness,
)
from jupr_app.services.admin_tournament_match_publish_service import (
    build_admin_tournament_official_publish_plan,
)
from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_state_fingerprint
from jupr_app.services.admin_tournament_podium_review_service import review_admin_tournament_draw_podium
from tests.test_admin_match_log_service import FakeQuery, FakeSupabase
from tests.test_admin_tournament_podium_review_service import podium_review_tables


def _versions(rows: list[dict]) -> list[dict[str, str]]:
    return sorted(
        [{"id": str(row["id"]), "updated_at": str(row["updated_at"])} for row in rows],
        key=lambda row: row["id"],
    )


def _review_draw(supabase: FakeSupabase, tables: dict[str, list[dict]], draw_id: str) -> None:
    draw = next(row for row in tables["tournament_event_draws"] if row["id"] == draw_id)
    teams = [row for row in tables["tournament_teams"] if row["draw_id"] == draw_id]
    games = [row for row in tables["tournament_games"] if row["draw_id"] == draw_id]
    review_admin_tournament_draw_podium(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id=draw_id,
        expected_state_fingerprint=get_admin_tournament_ops_state_fingerprint(
            supabase,
            club_id="club",
            tournament_id="tour-1",
        ),
        expected_draw_updated_at=draw["updated_at"],
        expected_team_versions=_versions(teams),
        expected_source_game_versions=_versions(games),
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="REVIEW PODIUM",
    )


def _add_awards(tables: dict[str, list[dict]], draw_id: str) -> None:
    teams = {
        str(row["id"]): row
        for row in tables["tournament_teams"]
        if row["draw_id"] == draw_id
    }
    for podium in [row for row in tables["tournament_podium"] if row["draw_id"] == draw_id]:
        placement = int(podium["placement"])
        team = teams[str(podium["team_id"])]
        for player_id in (team["player1_id"], team["player2_id"]):
            tables["player_badges"].append(
                {
                    "id": f"badge-{draw_id}-{placement}-{player_id}",
                    "club_id": "club",
                    "player_id": player_id,
                    "badge_id": PODIUM_BADGE_MAP[placement],
                    "context_type": "tournament",
                    "context_id": f"tour-1:draw:{draw_id}:podium:{placement}",
                    "revoked_at": None,
                }
            )


def _add_second_draw(tables: dict[str, list[dict]]) -> None:
    updated = "2026-08-15T12:00:00Z"
    tables["tournament_event_draws"].append(
        {
            "id": "draw-2",
            "tournament_id": "tour-1",
            "event_option_id": "event-2",
            "name": "Mixed Doubles",
            "status": "draft",
            "updated_at": updated,
        }
    )
    for number in (1, 2, 3):
        tables["tournament_teams"].append(
            {
                "id": f"draw2-team-{number}",
                "tournament_id": "tour-1",
                "draw_id": "draw-2",
                "team_number": number,
                "player1_id": number * 2 + 5,
                "player2_id": number * 2 + 6,
                "updated_at": updated,
            }
        )
    pairs = ((1, 2, 11, 7), (1, 3, 11, 8), (2, 3, 11, 9))
    for index, (a, b, score_a, score_b) in enumerate(pairs, start=1):
        tables["tournament_games"].append(
            {
                "id": f"draw2-game-{index}",
                "tournament_id": "tour-1",
                "draw_id": "draw-2",
                "stage": "ROUND_ROBIN",
                "rr_round_number": index,
                "rr_slot_number": 1,
                "team_a_id": f"draw2-team-{a}",
                "team_b_id": f"draw2-team-{b}",
                "score_a": score_a,
                "score_b": score_b,
                "winner_team_id": f"draw2-team-{a}",
                "loser_team_id": f"draw2-team-{b}",
                "finalized_at": updated,
                "updated_at": updated,
            }
        )
    for placement in (1, 2, 3):
        tables["tournament_podium"].append(
            {
                "id": f"draw2-podium-{placement}",
                "tournament_id": "tour-1",
                "draw_id": "draw-2",
                "placement": placement,
                "team_id": f"draw2-team-{placement}",
                "source": "ROUND_ROBIN",
                "updated_at": updated,
            }
        )
    tables["players"].extend(
        {"club_id": "club", "id": player_id, "name": f"Player {player_id}"}
        for player_id in range(7, 13)
    )


def _ready_tables(monkeypatch) -> tuple[dict[str, list[dict]], FakeSupabase]:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    supabase = FakeSupabase(tables)
    _review_draw(supabase, tables, "draw-1")
    _add_awards(tables, "draw-1")
    return tables, supabase


class _EventOptionsReadFailureSupabase(FakeSupabase):
    def table(self, name):
        if name == "tournament_event_options":
            raise RuntimeError("simulated event-option evidence outage")
        return super().table(name)


class _ShortRangePageQuery(FakeQuery):
    def __init__(self, *args, page_cap: int, range_calls: list[tuple[int, int]], **kwargs):
        super().__init__(*args, **kwargs)
        self.page_cap = int(page_cap)
        self.range_calls = range_calls
        self.range_start: int | None = None
        self.range_end: int | None = None

    def limit(self, value):
        # This also makes the regression fail against the former one-shot
        # implementation, which could not distinguish a server-capped page.
        self.limit_value = min(int(value), self.page_cap)
        return self

    def range(self, start, end):
        self.range_start = int(start)
        self.range_end = int(end)
        self.range_calls.append((self.range_start, self.range_end))
        return self

    def execute(self):
        response = super().execute()
        if self.range_start is None or self.range_end is None:
            return response
        stop = min(self.range_end + 1, self.range_start + self.page_cap)
        return SimpleNamespace(data=list(response.data)[self.range_start:stop])


class _ShortRangePageSupabase(FakeSupabase):
    def __init__(self, tables, *, page_cap: int):
        super().__init__(tables)
        self.page_cap = int(page_cap)
        self.event_option_range_calls: list[tuple[int, int]] = []

    def table(self, name):
        if name == "tournament_event_options":
            return _ShortRangePageQuery(
                self.tables,
                name,
                operations=self.operations,
                page_cap=self.page_cap,
                range_calls=self.event_option_range_calls,
            )
        return super().table(name)


def _official_match_for_game(
    tables: dict[str, list[dict]],
    game: dict,
    *,
    match_id: str | None = None,
) -> dict:
    teams = {str(row["id"]): row for row in tables["tournament_teams"]}
    team_a = teams[str(game["team_a_id"])]
    team_b = teams[str(game["team_b_id"])]
    singles = team_a.get("player2_id") is None and team_b.get("player2_id") is None
    return {
        "id": match_id or f"match-{game['id']}",
        "club_id": "club",
        "tournament_id": str(game["tournament_id"]),
        "tournament_game_id": str(game["id"]),
        "context_type": "tournament_game",
        "context_id": str(game["id"]),
        "match_format": "singles" if singles else "doubles",
        "t1_p1": team_a.get("player1_id"),
        "t1_p2": team_a.get("player2_id"),
        "t2_p1": team_b.get("player1_id"),
        "t2_p2": team_b.get("player2_id"),
        "score_t1": game.get("score_a"),
        "score_t2": game.get("score_b"),
        "row_version": 1,
        "deleted_at": None,
    }


def _publish_draw_with_immutable_evidence(
    tables: dict[str, list[dict]],
    supabase: FakeSupabase,
    draw_id: str,
) -> dict:
    plan = build_admin_tournament_official_publish_plan(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id=draw_id,
    )
    request = build_tournament_admin_operation_request(
        club_id="club",
        surface="tournament_live",
        action="tournament_live_official_publish",
        entity_type="tournament_event_draw",
        entity_id=draw_id,
        lock_scope=f"tournament:tour-1:draw:{draw_id}",
        expected_state="reviewed-state",
        payload={
            "command": "publish_official_matches",
            "publish_plan": plan,
        },
        idempotency_key="123e4567-e89b-42d3-a456-426614174000",
    )
    tables["tournament_admin_operations"].append(
        {
            **{
                key: request[key]
                for key in (
                    "operation_key",
                    "request_fingerprint",
                    "club_id",
                    "surface",
                    "action",
                    "entity_type",
                    "entity_id",
                    "lock_scope",
                    "expected_state",
                )
            },
            "client_idempotency_key": request["idempotency_key"],
            "status": "completed",
            "request_json": request,
            "result_json": {"ok": True},
            "attempt_count": 1,
            "created_by": "director@example.com",
            "updated_by": "director@example.com",
            "created_at": "2026-08-15T12:01:00Z",
            "updated_at": "2026-08-15T12:01:00Z",
        }
    )
    tables["matches"].extend(
        {
            **projection,
            "id": f"match-{projection['tournament_game_id']}",
            "row_version": 1,
            "deleted_at": None,
        }
        for projection in plan["match_payload_projections"]
    )
    return plan


def test_lifecycle_reports_authoritative_open_game_and_exact_blockers(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    tables["tournament_games"][1].update(
        {
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        }
    )
    tables["tournament_podium"] = []
    lifecycle = build_admin_tournament_lifecycle(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert lifecycle["contract"] == "jupr:tournament-lifecycle:v1"
    assert lifecycle["authority"] == "python_fastapi"
    assert lifecycle["phase"] == "live_in_progress"
    assert lifecycle["draw_id"] == "draw-1"
    assert lifecycle["counts"]["finalized_games"] == 2
    assert lifecycle["counts"]["open_games"] == 1
    assert lifecycle["counts"]["tied_games"] == 0
    assert lifecycle["counts"]["podium_entries"] == 0
    assert lifecycle["counts"]["unexpected_awards"] == 0
    assert lifecycle["counts"]["unpublished_games"] == 3
    assert lifecycle["counts"]["duplicate_publications"] == 0
    assert lifecycle["counts"]["uncertain_operations"] == 0
    assert lifecycle["states"]["live_operations"] == "in_progress"
    assert lifecycle["draws"][0]["status"] == "DRAFT"
    assert lifecycle["draws"][0]["protected"] is False
    blockers = lifecycle["domain_readiness"]["official_publish"]["blockers"]
    assert {row["code"] for row in blockers} >= {
        "GAMES_NOT_FINALIZED",
        "PODIUM_INCOMPLETE",
        "PODIUM_REVIEW_REQUIRED",
        "AWARDS_NOT_DERIVABLE",
    }


def test_nonprotected_draw_with_teams_and_zero_games_is_a_blocker(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    tables["tournament_games"] = []
    tables["tournament_podium"] = []
    lifecycle = build_admin_tournament_lifecycle(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )
    blockers = lifecycle["domain_readiness"]["official_publish"]["blockers"]
    missing = next(row for row in blockers if row["code"] == "DRAW_GAMES_MISSING")
    assert missing["draw_id"] == "draw-1"
    assert missing["count"] == 3


def test_empty_active_primary_draw_blocks_tournament_wide_closeout(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    tables["tournament_event_draws"].append(
        {
            "id": "empty-active-draw",
            "tournament_id": "tour-1",
            "event_option_id": "event-empty",
            "name": "Uninitialized active division",
            "status": "active",
            "updated_at": "2026-08-15T12:00:00Z",
        }
    )

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert {row["draw_id"] for row in lifecycle["draws"]} == {
        "draw-1",
        "empty-active-draw",
    }
    blockers = lifecycle["domain_readiness"]["official_publish"]["blockers"]
    missing = next(
        row
        for row in blockers
        if row["code"] == "DRAW_GAMES_MISSING"
        and row.get("draw_id") == "empty-active-draw"
    )
    assert missing["count"] == 0
    assert lifecycle["domain_readiness"]["official_publish"]["ready"] is False
    assert lifecycle["domain_readiness"]["archive"]["ready"] is False


def test_enabled_event_without_draw_blocks_completion_until_explicitly_disabled(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    tables["tournament_event_options"].append(
        {
            "id": "event-empty",
            "tournament_id": "tour-1",
            "division_name": "Women's 3.0",
            "enabled": True,
            "status": "active",
        }
    )

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )
    blocker = next(
        row
        for row in lifecycle["domain_readiness"]["completion"]["blockers"]
        if row["code"] == "EVENT_DRAW_MISSING"
    )
    assert blocker["entity_id"] == "event-empty"

    tables["tournament_event_options"][0].update(
        {"enabled": False, "status": "cancelled"}
    )
    after_cancel = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )
    assert not any(
        row["code"] == "EVENT_DRAW_MISSING"
        for row in after_cancel["domain_readiness"]["completion"]["blockers"]
    )


def test_event_option_evidence_failure_blocks_completion(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    tables["tournaments"][0]["status"] = "ACTIVE"
    tables["tournament_event_options"].append(
        {
            "id": "event-missing",
            "tournament_id": "tour-1",
            "division_name": "Missing division",
            "enabled": True,
            "status": "active",
        }
    )

    lifecycle = build_admin_tournament_lifecycle(
        _EventOptionsReadFailureSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
    )

    assert lifecycle["evidence"]["event_options_available"] is False
    assert lifecycle["domain_readiness"]["completion"]["ready"] is False
    assert any(
        blocker["code"] == "EVENT_OPTION_EVIDENCE_UNAVAILABLE"
        for blocker in lifecycle["domain_readiness"]["completion"]["blockers"]
    )


def test_short_server_capped_pages_are_exhausted_before_completion(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    tables["tournaments"][0]["status"] = "ACTIVE"
    tables["tournament_event_options"] = [
        {
            "id": "event-1",
            "tournament_id": "tour-1",
            "division_name": "Existing division",
            "enabled": True,
            "status": "active",
        },
        {
            "id": "event-missing",
            "tournament_id": "tour-1",
            "division_name": "Missing division",
            "enabled": True,
            "status": "active",
        },
    ]
    capped = _ShortRangePageSupabase(tables, page_cap=1)

    lifecycle = build_admin_tournament_lifecycle(
        capped,
        club_id="club",
        tournament_id="tour-1",
    )

    assert [start for start, _ in capped.event_option_range_calls] == [0, 1, 2]
    assert lifecycle["evidence"]["event_options_available"] is True
    assert lifecycle["domain_readiness"]["completion"]["ready"] is False
    assert any(
        blocker["code"] == "EVENT_DRAW_MISSING"
        and blocker.get("entity_id") == "event-missing"
        for blocker in lifecycle["domain_readiness"]["completion"]["blockers"]
    )


def test_active_team_parent_draw_blocks_closeout_until_canonical_team_review_exists(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    tables["tournament_event_draws"].append(
        {
            "id": "team-parent",
            "tournament_id": "tour-1",
            "event_option_id": "event-team",
            "name": "Four-player team division",
            "draw_kind": "TEAM_PARENT",
            "status": "active",
            "updated_at": "2026-08-15T13:00:00Z",
        }
    )

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert lifecycle["counts"]["active_team_parent_draws"] == 1
    assert lifecycle["evidence"]["active_team_parent_draw_ids"] == ["team-parent"]
    for action in ("official_publish", "archive"):
        blockers = lifecycle["domain_readiness"][action]["blockers"]
        assert any(
            row["code"] == "TEAM_COMPETITION_CLOSEOUT_UNSUPPORTED"
            and row.get("draw_id") == "team-parent"
            for row in blockers
        )


def test_orphan_game_evidence_blocks_publish_and_archive(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    tables["tournament_games"].append(
        {
            "id": "orphan-game",
            "tournament_id": "tour-1",
            "draw_id": "missing-draw",
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": "2026-08-15T12:00:00Z",
            "updated_at": "2026-08-15T12:00:00Z",
        }
    )

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert lifecycle["counts"]["orphan_games"] == 1
    for action in ("official_publish", "archive"):
        readiness = lifecycle["domain_readiness"][action]
        assert readiness["ready"] is False
        blocker = next(
            row
            for row in readiness["blockers"]
            if row["code"] == "ORPHAN_TOURNAMENT_GAMES"
        )
        assert blocker["count"] == 1


def test_current_review_and_exact_awards_make_unpublished_draw_domain_ready(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert lifecycle["domain_readiness"]["official_publish"]["ready"] is True
    assert lifecycle["domain_readiness"]["archive"]["ready"] is False
    assert lifecycle["draws"][0]["review_evidence"]["current"] is True
    assert lifecycle["draws"][0]["counts"]["awards_complete"] is True
    assert lifecycle["runtime_capability"] is not lifecycle["domain_readiness"]
    assert tables["matches"] == []


def test_already_published_draw_does_not_block_another_ready_draw(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    _add_second_draw(tables)
    supabase = FakeSupabase(tables)
    _review_draw(supabase, tables, "draw-1")
    _review_draw(supabase, tables, "draw-2")
    _add_awards(tables, "draw-1")
    _add_awards(tables, "draw-2")
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-2")

    lifecycle = require_admin_tournament_official_publish_readiness(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )
    assert lifecycle["domain_readiness"]["official_publish"]["ready"] is True
    published = next(row for row in lifecycle["draws"] if row["draw_id"] == "draw-2")
    assert published["publication_evidence"]["complete"] is True


def test_completion_requires_one_official_link_per_game_and_rejects_duplicates(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )
    assert lifecycle["domain_readiness"]["completion"]["ready"] is True

    tables["matches"].append(deepcopy(tables["matches"][0]))
    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )
    blockers = lifecycle["domain_readiness"]["completion"]["blockers"]
    assert any(row["code"] == "OFFICIAL_LINKS_DUPLICATE" for row in blockers)


def test_completion_counts_only_played_games_as_rating_publication_eligible(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    tables["tournament_games"][0].update(
        {"result_type": "FORFEIT", "rating_publish_eligible": False}
    )
    supabase = FakeSupabase(tables)
    _review_draw(supabase, tables, "draw-1")
    _add_awards(tables, "draw-1")

    plan = _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )

    assert len(plan["tournament_game_ids"]) == 2
    assert lifecycle["counts"]["games"] == 3
    assert lifecycle["counts"]["rating_publish_eligible_games"] == 2
    assert lifecycle["counts"]["published_games"] == 2
    assert lifecycle["counts"]["unpublished_games"] == 0
    assert lifecycle["domain_readiness"]["completion"]["ready"] is True


def test_all_non_played_draw_can_complete_without_rating_publications(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    for game in tables["tournament_games"]:
        game.update({"result_type": "NO_SHOW", "rating_publish_eligible": False})
    supabase = FakeSupabase(tables)
    _review_draw(supabase, tables, "draw-1")
    _add_awards(tables, "draw-1")

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )

    assert lifecycle["counts"]["games"] == 3
    assert lifecycle["counts"]["rating_publish_eligible_games"] == 0
    assert lifecycle["counts"]["published_games"] == 0
    assert lifecycle["counts"]["unpublished_games"] == 0
    assert lifecycle["draws"][0]["publication_evidence"]["complete"] is True
    assert lifecycle["domain_readiness"]["completion"]["ready"] is True


def test_completion_capability_reports_atomic_support_without_contradiction(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        runtime_capability={
            "operations_mutations_enabled": True,
            "tournament_mutations_enabled": True,
        },
    )

    assert lifecycle["domain_readiness"]["completion"]["ready"] is True
    assert lifecycle["runtime_capability"]["completion_available"] is True
    assert lifecycle["runtime_capability"]["completion_atomic_commit_enabled"] is True
    assert lifecycle["runtime_capability"]["archive_available"] is False
    assert lifecycle["runtime_capability"]["archive_atomic_commit_enabled"] is True
    assert "archive_blocker" not in lifecycle["runtime_capability"]
    assert lifecycle["next_action"]["key"] == "complete_tournament"


def test_soft_deleted_match_is_not_official_publication_evidence(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    tables["matches"][0]["deleted_at"] = "2026-08-15T13:00:00Z"

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )

    assert lifecycle["counts"]["published_games"] == 2
    assert lifecycle["counts"]["unpublished_games"] == 1
    assert lifecycle["counts"]["soft_deleted_official_matches"] == 1
    assert lifecycle["domain_readiness"]["archive"]["ready"] is False
    assert any(
        row["code"] == "OFFICIAL_MATCH_HISTORY_EXCLUDED"
        for row in lifecycle["domain_readiness"]["official_publish"]["blockers"]
    )
    assert any(
        row["code"] == "OFFICIAL_LINKS_PARTIAL"
        for row in lifecycle["domain_readiness"]["archive"]["blockers"]
    )
    assert lifecycle["evidence"]["soft_deleted_official_match_ids"] == ["match-game-1"]


def test_edited_official_match_payload_requires_reconciliation(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    tables["matches"][0]["score_t1"] = 10
    tables["matches"][0]["row_version"] = 2

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )

    assert lifecycle["counts"]["mismatched_official_matches"] == 1
    assert lifecycle["counts"]["published_games"] == 2
    assert lifecycle["domain_readiness"]["archive"]["ready"] is False
    blocker = next(
        row
        for row in lifecycle["domain_readiness"]["archive"]["blockers"]
        if row["code"] == "OFFICIAL_MATCH_PAYLOAD_MISMATCH"
    )
    assert blocker["draw_id"] == "draw-1"
    assert lifecycle["draws"][0]["publication_evidence"]["mismatched_matches"] == [
        {
            "match_id": "match-game-1",
            "tournament_game_id": "game-1",
            "fields": ["score_t1"],
        }
    ]

    # A completed rating replay can bump the row version without changing the
    # immutable tournament-link projection.
    tables["matches"][0]["score_t1"] = 11
    replayed = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )
    assert replayed["counts"]["mismatched_official_matches"] == 0
    assert replayed["domain_readiness"]["completion"]["ready"] is True
    assert replayed["domain_readiness"]["archive"]["ready"] is False


def test_match_links_without_completed_publish_plan_fail_closed(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    tables["matches"] = [
        _official_match_for_game(tables, game)
        for game in tables["tournament_games"]
    ]

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )

    assert lifecycle["counts"]["official_matches_without_publication_evidence"] == 3
    assert lifecycle["counts"]["published_games"] == 0
    assert lifecycle["draws"][0]["publication_evidence"]["state"] == "evidence_unavailable"
    assert lifecycle["draws"][0]["publication_evidence"]["immutable_plan_available"] is False
    blocker = next(
        row
        for row in lifecycle["domain_readiness"]["archive"]["blockers"]
        if row["code"] == "OFFICIAL_PUBLICATION_EVIDENCE_UNAVAILABLE"
    )
    assert blocker["draw_id"] == "draw-1"
    assert blocker["count"] == 3


@pytest.mark.parametrize(
    ("field", "edited_value"),
    (
        ("match_type", "League"),
        ("league", "Renamed classification"),
        ("date", "2026-09-03T00:00:00+00:00"),
        ("week_tag", "Edited week"),
    ),
)
def test_edited_official_match_classification_is_compared_to_immutable_publish_plan(
    monkeypatch,
    field: str,
    edited_value: str,
) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    tables["matches"][0][field] = edited_value

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )

    assert lifecycle["counts"]["mismatched_official_matches"] == 1
    assert lifecycle["counts"]["published_games"] == 2
    assert lifecycle["domain_readiness"]["archive"]["ready"] is False
    assert lifecycle["draws"][0]["publication_evidence"]["mismatched_matches"] == [
        {
            "match_id": "match-game-1",
            "tournament_game_id": "game-1",
            "fields": [field],
        }
    ]


def test_tournament_linked_match_log_recovery_blocks_publish_and_completion(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    tables["match_exclusion_operations"] = [
        {
            "id": "exclude-1",
            "club_id": "club",
            "status": "pending_replay",
            "replay_job_id": "replay-1",
            "targets_json": [{"match_id": tables["matches"][0]["id"]}],
        }
    ]
    tables["replay_jobs"] = [
        {
            "id": "replay-1",
            "club_id": "club",
            "status": "pending",
        }
    ]

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert lifecycle["counts"]["unsettled_match_exclusions"] == 1
    assert lifecycle["counts"]["unsettled_replay_jobs"] == 1
    for action in ("official_publish", "completion"):
        codes = {
            row["code"]
            for row in lifecycle["domain_readiness"][action]["blockers"]
        }
        assert "MATCH_EXCLUSION_RECOVERY_UNSETTLED" in codes
        assert "MATCH_REPLAY_UNSETTLED" in codes


def test_unrelated_club_match_log_recovery_does_not_block_tournament(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    tables["match_exclusion_operations"] = [
        {
            "id": "exclude-other",
            "club_id": "club",
            "status": "pending_replay",
            "replay_job_id": "replay-other",
            "targets_json": [{"match_id": "unrelated-match"}],
        }
    ]
    tables["replay_jobs"] = [
        {
            "id": "replay-other",
            "club_id": "club",
            "status": "pending",
        }
    ]

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )

    assert lifecycle["counts"]["unsettled_match_exclusions"] == 0
    assert lifecycle["counts"]["unsettled_replay_jobs"] == 0
    assert lifecycle["domain_readiness"]["completion"]["ready"] is True


def test_historical_exclusion_and_replay_rows_are_filtered_before_read_bound(
    monkeypatch,
) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    linked_match_id = tables["matches"][0]["id"]
    tables["match_exclusion_operations"] = [
        {
            "id": f"settled-{index}",
            "club_id": "club",
            "status": "succeeded",
            "replay_job_id": f"settled-replay-{index}",
            "targets_json": [{"match_id": f"historical-{index}"}],
        }
        for index in range(5001)
    ] + [
        {
            "id": "linked-pending",
            "club_id": "club",
            "status": "pending_replay",
            "replay_job_id": "linked-replay",
            "targets_json": [{"match_id": linked_match_id}],
        }
    ]
    tables["replay_jobs"] = [
        {
            "id": f"settled-replay-{index}",
            "club_id": "club",
            "status": "completed",
        }
        for index in range(5001)
    ] + [
        {"id": "linked-replay", "club_id": "club", "status": "pending"}
    ]

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
    )

    assert lifecycle["counts"]["unsettled_match_exclusions"] == 1
    assert lifecycle["counts"]["unsettled_replay_jobs"] == 1
    assert not any("safe lifecycle read bound" in warning for warning in lifecycle["warnings"])


def test_orphan_team_and_podium_evidence_blocks_closeout(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    tables["tournament_teams"].append(
        {
            "id": "orphan-team",
            "tournament_id": "tour-1",
            "draw_id": "missing-draw",
            "player1_id": 1,
            "player2_id": 2,
            "updated_at": "2026-08-15T13:00:00Z",
        }
    )
    tables["tournament_podium"].append(
        {
            "id": "orphan-podium",
            "tournament_id": "tour-1",
            "draw_id": "missing-draw",
            "placement": 1,
            "team_id": "orphan-team",
            "updated_at": "2026-08-15T13:00:00Z",
        }
    )

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert lifecycle["counts"]["orphan_teams"] == 1
    assert lifecycle["counts"]["orphan_podium_entries"] == 1
    codes = {
        row["code"]
        for row in lifecycle["domain_readiness"]["official_publish"]["blockers"]
    }
    assert {"ORPHAN_TOURNAMENT_TEAMS", "ORPHAN_TOURNAMENT_PODIUM"} <= codes


def test_live_prefix_operation_blocks_and_only_exact_own_key_can_be_ignored(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    tables["tournament_admin_operations"] = [
        {
            "operation_key": "a" * 64,
            "club_id": "club",
            "surface": "tournament_live",
            "action": "tournament_live_official_publish",
            "entity_id": "draw-1",
            "lock_scope": "tournament:tour-1:draw:draw-1",
            "status": "intent",
        }
    ]
    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )
    blockers = lifecycle["domain_readiness"]["official_publish"]["blockers"]
    assert any(row["code"] == "ACTIVE_OR_UNCERTAIN_OPERATIONS" for row in blockers)

    lifecycle = require_admin_tournament_official_publish_readiness(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        ignore_operation_key="a" * 64,
    )
    assert lifecycle["domain_readiness"]["official_publish"]["ready"] is True

    tables["tournament_admin_operations"].append(
        {
            "operation_key": "b" * 64,
            "club_id": "club",
            "surface": "operations",
            "action": "ops_game_score",
            "entity_id": "draw-1",
            "lock_scope": "tour-1",
            "status": "recovery_required",
        }
    )
    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
        ignore_operation_keys={"a" * 64},
    )
    assert lifecycle["domain_readiness"]["official_publish"]["ready"] is False
    assert lifecycle["counts"]["active_operations"] == 1
    assert lifecycle["counts"]["uncertain_operations"] == 1


def test_lifecycle_operation_counts_dedupe_by_operation_key(monkeypatch) -> None:
    tables, supabase = _ready_tables(monkeypatch)
    operation = {
        "operation_key": "c" * 64,
        "club_id": "club",
        "surface": "tournament_live",
        "action": "tournament_live_score",
        "entity_id": "draw-1",
        "lock_scope": "tournament:tour-1:draw:draw-1",
        "status": "mutated",
    }
    tables["tournament_admin_operations"] = [deepcopy(operation), deepcopy(operation)]

    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert lifecycle["counts"]["active_operations"] == 1
    assert lifecycle["counts"]["uncertain_operations"] == 1
    assert len(lifecycle["evidence"]["active_operations"]) == 1
