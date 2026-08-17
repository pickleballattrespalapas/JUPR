from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event,
    create_generator_preview,
    generator_event_standings,
    generator_match_play_format,
    history_before_round,
    mark_generator_round_played,
    mutate_generator_roster,
    save_generator_round,
    schedule_export_rows,
    start_generator_event,
)

ROOT = Path(__file__).resolve().parents[1]


def _names(count: int) -> list[str]:
    return [f"Player {index}" for index in range(1, count + 1)]


def _matches(round_row: dict) -> list[dict]:
    if round_row.get("matches"):
        return list(round_row.get("matches") or [])
    return [
        match
        for court in round_row.get("courts") or []
        for match in court.get("matches") or []
    ]


def _score_round(event: dict, round_number: int) -> dict:
    round_row = next(row for row in event["rounds"] if row["number"] == round_number)
    return save_generator_round(
        event,
        round_number=round_number,
        scores=[
            {
                "match_id": match["id"],
                "score_a": 11,
                "score_b": 7,
            }
            for match in _matches(round_row)
        ],
    )


def _mixed_preview(
    *,
    count: int = 9,
    doubles_courts: int = 1,
    singles_courts: int = 2,
    rounds: int = 9,
    scoring_mode: str = "scored",
) -> dict:
    return create_generator_preview(
        generator_kind="round_robin",
        play_format="doubles_singles",
        title="Mixed Round Robin",
        participant_names=_names(count),
        total_rounds=rounds,
        doubles_court_count=doubles_courts,
        singles_court_count=singles_courts,
        scoring_mode=scoring_mode,
    )


def test_mixed_format_validation_and_fingerprint_contract() -> None:
    with pytest.raises(ValueError, match="only for Round-Robin"):
        create_generator_preview(
            generator_kind="ladder",
            play_format="doubles_singles",
            title="Bad mixed ladder",
            participant_names=_names(8),
            total_rounds=4,
            doubles_court_count=1,
            singles_court_count=2,
        )
    with pytest.raises(ValueError, match="at least 6"):
        _mixed_preview(count=5, doubles_courts=1, singles_courts=1, rounds=2)
    with pytest.raises(ValueError, match="one doubles court and one singles court"):
        _mixed_preview(count=8, doubles_courts=0, singles_courts=2, rounds=2)
    with pytest.raises(ValueError, match="requires 10 players"):
        _mixed_preview(count=8, doubles_courts=2, singles_courts=1, rounds=2)

    one_mix = _mixed_preview(count=8, doubles_courts=1, singles_courts=2, rounds=4)
    repeated_mix = _mixed_preview(count=8, doubles_courts=1, singles_courts=2, rounds=4)
    another_mix = _mixed_preview(count=8, doubles_courts=1, singles_courts=1, rounds=4)
    assert one_mix["playFormat"] == "doubles_singles"
    assert one_mix["previewFingerprint"] == repeated_mix["previewFingerprint"]
    assert one_mix["courtCount"] == 3
    assert one_mix["doublesCourtCount"] == 1
    assert one_mix["singlesCourtCount"] == 2
    assert one_mix["previewFingerprint"] != another_mix["previewFingerprint"]


def test_mixed_rounds_use_each_player_once_and_preserve_court_formats() -> None:
    event = _mixed_preview(count=9, doubles_courts=1, singles_courts=2, rounds=9)

    for round_row in event["rounds"]:
        matches = _matches(round_row)
        actual_formats = [
            generator_match_play_format(match, event["playFormat"])
            for match in matches
        ]
        assert actual_formats.count("doubles") == 1
        assert actual_formats.count("singles") == 2
        assert round_row["formatCounts"] == {"doubles": 1, "singles": 2}
        assert sorted(int(match["court"]) for match in matches) == [1, 2, 3]

        playing = [
            str(participant_id)
            for match in matches
            for participant_id in [*(match.get("sideA") or []), *(match.get("sideB") or [])]
        ]
        assert len(playing) == 8
        assert len(set(playing)) == 8
        assert len(round_row["byeParticipantIds"]) == 1
        assert set(playing).isdisjoint(round_row["byeParticipantIds"])
        assert len(set(playing).union(round_row["byeParticipantIds"])) == 9

    export_rows = schedule_export_rows(event)
    assert {row["play_format"] for row in export_rows} == {"singles", "doubles"}


def test_mixed_schedule_balances_roles_byes_and_avoids_avoidable_repeats() -> None:
    event = _mixed_preview(count=9, doubles_courts=1, singles_courts=2, rounds=9)
    history = history_before_round(event, 10, include_preview=True)

    assert set(history["games"].values()) == {8}
    assert set(history["singles_games"].values()) == {4}
    assert set(history["doubles_games"].values()) == {4}
    assert set(history["byes"].values()) == {1}
    assert max(history["partners"].values(), default=0) <= 1
    assert max(history["singles_opponents"].values(), default=0) <= 1


def test_mixed_schedule_balances_multiple_byes_when_organizer_uses_fewer_courts() -> None:
    event = _mixed_preview(
        count=10,
        doubles_courts=1,
        singles_courts=1,
        rounds=5,
    )
    history = history_before_round(event, 6, include_preview=True)

    assert set(history["games"].values()) == {3}
    assert set(history["singles_games"].values()) == {1}
    assert set(history["doubles_games"].values()) == {2}
    assert set(history["byes"].values()) == {2}
    assert all(len(round_row["byeParticipantIds"]) == 4 for round_row in event["rounds"])


def test_larger_mixed_schedule_avoids_repeat_partners_and_singles_opponents_when_available() -> None:
    event = _mixed_preview(
        count=18,
        doubles_courts=3,
        singles_courts=3,
        rounds=18,
    )
    history = history_before_round(event, 19, include_preview=True)

    assert set(history["games"].values()) == {18}
    assert set(history["singles_games"].values()) == {6}
    assert set(history["doubles_games"].values()) == {12}
    assert max(history["partners"].values(), default=0) <= 1
    assert max(history["singles_opponents"].values(), default=0) <= 1


def test_mixed_scored_and_unscored_lifecycles_follow_existing_round_robin_rules() -> None:
    scored = start_generator_event(
        _mixed_preview(count=8, doubles_courts=1, singles_courts=2, rounds=2)
    )
    scored = _score_round(scored, 1)
    standings = generator_event_standings(scored)
    assert len(standings) == 8
    assert sum(row["matches"] for row in standings) == 8
    scored = advance_generator_event(scored)
    scored = _score_round(scored, 2)
    scored = advance_generator_event(scored)
    assert scored["status"] == "completed"

    unscored = start_generator_event(
        _mixed_preview(
            count=8,
            doubles_courts=1,
            singles_courts=2,
            rounds=2,
            scoring_mode="unscored",
        )
    )
    unscored = mark_generator_round_played(unscored, round_number=1)
    assert unscored["rounds"][0]["status"] == "played"
    unscored = advance_generator_event(unscored)
    assert unscored["currentRoundNumber"] == 2
    unscored = mark_generator_round_played(unscored, round_number=2)
    unscored = advance_generator_event(unscored)
    assert unscored["status"] == "completed"
    assert generator_event_standings(unscored) == []


def test_adaptive_roster_change_preserves_completed_mixed_round_and_rebalances_future_rounds() -> None:
    event = start_generator_event(
        _mixed_preview(count=8, doubles_courts=1, singles_courts=2, rounds=4)
    )
    event = _score_round(event, 1)
    completed_round = deepcopy(event["rounds"][0])

    event = mutate_generator_roster(
        event,
        action="remove",
        participant_id="p-8",
    )

    assert event["rounds"][0] == completed_round
    for round_row in event["rounds"][1:]:
        assert round_row["formatCounts"] == {"doubles": 1, "singles": 1}
        assert len(_matches(round_row)) == 2
        assert len(round_row["byeParticipantIds"]) == 1
        assert any("active roster supports" in warning for warning in round_row["warnings"])
        playing = [
            participant_id
            for match in _matches(round_row)
            for participant_id in [*(match.get("sideA") or []), *(match.get("sideB") or [])]
        ]
        assert "p-8" not in playing


def test_public_admin_api_and_browser_draft_include_mixed_configuration() -> None:
    admin_routes = (ROOT / "services/api/admin_play_generator_routes.py").read_text()
    public_routes = (ROOT / "services/api/public_play_generator_routes.py").read_text()
    admin_workspace = (
        ROOT / "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx"
    ).read_text()
    public_workspace = (
        ROOT
        / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx"
    ).read_text()
    roster_setup = (ROOT / "apps/web/components/GeneratorRosterSetup.tsx").read_text()
    draft = (ROOT / "apps/web/lib/playGeneratorDraft.ts").read_text()

    for routes in (admin_routes, public_routes):
        assert "doubles_singles" in routes
        assert "doubles_court_count" in routes
        assert "singles_court_count" in routes

    for workspace in (admin_workspace, public_workspace):
        assert "Doubles + Singles Mix" in workspace
        assert "doubles_court_count" in workspace
        assert "singles_court_count" in workspace
        assert "Format" in workspace

    assert "Doubles courts" in roster_setup
    assert "Singles courts" in roster_setup
    assert "recommendedMixedCourtSetup" in roster_setup
    assert "doubles_singles" in draft
    assert "doublesCourtCount" in draft
    assert "singlesCourtCount" in draft


def test_mixed_official_publish_splits_singles_and_doubles(monkeypatch) -> None:
    from types import SimpleNamespace

    from tests.conftest import require_api_dependency

    require_api_dependency("postgrest")
    require_api_dependency("supabase")
    import jupr_app.services.admin_play_generator_service as service

    class FakeQuery:
        def __init__(self, storage, table_name):
            self.storage = storage
            self.table_name = table_name
            self.filters = []
            self.in_filters = []
            self.limit_count = None
            self.insert_payload = None
            self.update_payload = None

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, key, value):
            self.filters.append((key, value))
            return self

        def in_(self, key, values):
            self.in_filters.append((key, {str(value) for value in values}))
            return self

        def limit(self, value):
            self.limit_count = int(value)
            return self

        def insert(self, payload):
            self.insert_payload = payload
            return self

        def update(self, payload):
            self.update_payload = dict(payload)
            return self

        def _matches(self, row):
            return (
                all(str(row.get(key)) == str(value) for key, value in self.filters)
                and all(
                    str(row.get(key)) in values
                    for key, values in self.in_filters
                )
            )

        def execute(self):
            rows = self.storage.setdefault(self.table_name, [])
            if self.insert_payload is not None:
                inserted = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
                rows.extend(dict(row) for row in inserted)
                return SimpleNamespace(data=inserted)
            selected = [row for row in rows if self._matches(row)]
            if self.update_payload is not None:
                for row in selected:
                    row.update(self.update_payload)
                return SimpleNamespace(data=selected)
            if self.limit_count is not None:
                selected = selected[: self.limit_count]
            return SimpleNamespace(data=selected)

    class FakeSupabase:
        def __init__(self):
            self.storage = {
                "live_sessions": [],
                "players": [
                    {
                        "club_id": "club",
                        "id": player_id,
                        "name": f"Player {index}",
                    }
                    for index, player_id in enumerate(range(101, 109), start=1)
                ],
                "admin_activity_log": [],
            }

        def table(self, name):
            return FakeQuery(self.storage, name)

    supabase = FakeSupabase()
    created = service.create_play_generator_session(
        supabase,
        club_id="club",
        generator_kind="round_robin",
        play_format="doubles_singles",
        title="Mixed official",
        participant_names=_names(8),
        player_ids=list(range(101, 109)),
        total_rounds=1,
        court_count=0,
        doubles_court_count=1,
        singles_court_count=2,
        preview_fingerprint=None,
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )
    session = created["session"]
    matches = _matches(session["event"]["rounds"][0])
    saved = service.save_play_generator_round(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        round_number=1,
        scores=[
            {"match_id": match["id"], "score_a": 11, "score_b": 7}
            for match in matches
        ],
        expected_version=session["version"],
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )["session"]

    league_rows = object()
    monkeypatch.setattr(
        service,
        "load_data",
        lambda *_args, **_kwargs: (
            None,
            None,
            league_rows,
            None,
            None,
            None,
            None,
            {},
            {},
            False,
            None,
        ),
    )
    calls = []

    def fake_submit(_supabase, **kwargs):
        calls.append(kwargs)
        return {"inserted": len(kwargs["matches"]), "format": kwargs["match_format"]}

    monkeypatch.setattr(service, "submit_atomic_direct_matches", fake_submit)

    published = service.publish_play_generator_matches(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        match_date="2026-08-03",
        expected_version=saved["version"],
        idempotency_key="mixed-publish-123",
        operation_key="a" * 64,
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )

    assert published["published_count"] == 3
    assert published["result"]["mixed_formats"] is True
    assert [call["match_format"] for call in calls] == ["doubles", "singles"]
    assert [len(call["matches"]) for call in calls] == [1, 2]
    doubles_payload = calls[0]["matches"][0]
    singles_payload = calls[1]["matches"][0]
    assert {"t1_p1", "t1_p2", "t2_p1", "t2_p2"} <= doubles_payload.keys()
    assert {"t1_p1", "t2_p1"} <= singles_payload.keys()
    assert "t1_p2" not in singles_payload
    assert calls[0]["df_leagues"] is league_rows
    assert calls[1]["df_leagues"] is league_rows
    assert calls[0]["idempotency_key"] != calls[1]["idempotency_key"]
    assert len(published["session"]["official_publish"]["published_match_ids"]) == 3
