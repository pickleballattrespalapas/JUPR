from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from jupr_app.services.public_challenge_ladder_service import build_public_challenge_ladder


class FakeQuery:
    def __init__(self, rows, *, fail_public_result_select: bool = False):
        self._rows = list(rows)
        self._filters: dict[str, object] = {}
        self._in_filters: dict[str, set[object]] = {}
        self._limit: int | None = None
        self._fail_public_result_select = fail_public_result_select

    def select(self, *args, **_kwargs):
        if (
            self._fail_public_result_select
            and args
            and "public_result_json" in str(args[0])
        ):
            raise RuntimeError("projection column temporarily unavailable")
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def in_(self, key, values):
        self._in_filters[key] = set(values)
        return self

    def limit(self, value):
        self._limit = int(value)
        return self

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
        for key, expected in self._in_filters.items():
            rows = [row for row in rows if row.get(key) in expected]
        if self._limit is not None:
            rows = rows[: self._limit]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, tables, *, fail_challenge_projection: bool = False):
        self._tables = tables
        self.fail_challenge_projection = fail_challenge_projection
        self.table_calls: Counter[str] = Counter()

    def table(self, name):
        self.table_calls[name] += 1
        return FakeQuery(
            self._tables.get(name, []),
            fail_public_result_select=(
                self.fail_challenge_projection and name == "ladder_challenges"
            ),
        )


def fake_supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "ladder_settings": [
                {
                    "club_id": "club",
                    "challenge_range": 7,
                    "accept_window_hours": 48,
                    "play_window_days": 7,
                    "cooldown_hours": 72,
                    "protected_hours": 72,
                    "pass_hold_hours": 72,
                    "internal_notes": "private",
                }
            ],
            "players": [
                {"id": 1, "club_id": "club", "name": "Alex", "rating": 1700, "active": True, "private_email": "hidden"},
                {"id": 2, "club_id": "club", "name": "Blair", "rating": 1600, "active": True},
                {"id": 3, "club_id": "club", "name": "Casey", "rating": 1500, "active": True},
                {"id": 4, "club_id": "club", "name": "Devon", "rating": 1400, "active": False},
            ],
            "ladder_roster": [
                {"id": 10, "club_id": "club", "player_id": 1, "tier_id": "PREM", "rank": 1, "is_active": True, "notes": "private"},
                {"id": 11, "club_id": "club", "player_id": 2, "tier_id": "PREM", "rank": 2, "is_active": True},
                {"id": 12, "club_id": "club", "player_id": 3, "tier_id": "ADV", "rank": 1, "is_active": True},
                {"id": 13, "club_id": "club", "player_id": 4, "tier_id": "ADV", "rank": 2, "is_active": True},
            ],
            "ladder_player_flags": [
                {"club_id": "club", "player_id": 3, "reinstate_required": True, "reinstate_notes": "Needs admin review", "private_reason": "hidden"},
            ],
            "ladder_challenges": [
                {
                    "id": 20,
                    "club_id": "club",
                    "challenger_id": 2,
                    "defender_id": 1,
                    "challenger_rank_at_create": 2,
                    "defender_rank_at_create": 1,
                    "tier_id": "PREM",
                    "status": "PENDING_ACCEPTANCE",
                    "created_at": "2099-01-01T00:00:00Z",
                    "accept_by": "2099-01-03T00:00:00Z",
                    "accepted_at": None,
                    "play_by": None,
                    "completed_at": None,
                    "winner_id": None,
                    "public_result_json": None,
                    "ledger_ref": "private ledger",
                    "challenger_contact": "private@example.com",
                },
                {
                    "id": 21,
                    "club_id": "club",
                    "challenger_id": 3,
                    "defender_id": 1,
                    "challenger_rank_at_create": 2,
                    "defender_rank_at_create": 1,
                    "tier_id": "ADV",
                    "status": "COMPLETED",
                    "created_at": "2026-01-01T00:00:00Z",
                    "accept_by": None,
                    "accepted_at": None,
                    "play_by": None,
                    "completed_at": "2026-01-05T00:00:00Z",
                    "winner_id": 3,
                    "public_result_json": None,
                    "resolution_notes": "private",
                },
            ],
            "ladder_pass_usage": [],
        }
    )


def test_public_challenge_ladder_builds_tiers_status_and_challenge_buckets() -> None:
    payload = build_public_challenge_ladder(fake_supabase(), club_id="club")

    assert payload["settings"]["challenge_range"] == 7
    assert payload["summary"]["active_player_count"] == 3
    assert payload["summary"]["active_challenge_count"] == 1

    prem = next(tier for tier in payload["tiers"] if tier["tier_id"] == "PREM")
    assert [player["player_name"] for player in prem["players"]] == ["Alex", "Blair"]
    locked = next(player for player in prem["players"] if player["player_name"] == "Alex")
    assert locked["status"] == "Locked"
    assert locked["challenge_id"] == 20
    assert "private" not in str(locked).lower()

    adv = next(tier for tier in payload["tiers"] if tier["tier_id"] == "ADV")
    assert [player["player_name"] for player in adv["players"]] == ["Casey"]
    assert adv["players"][0]["status"] == "Reinstate Required"
    assert adv["players"][0]["detail"] == "Staff review required before ladder activity."
    assert "Needs admin review" not in str(payload)

    pending = next(section for section in payload["challenge_sections"] if section["name"] == "Pending Acceptance")
    assert pending["challenges"][0]["challenger"] == {
        "player_id": 2,
        "player_name": "Blair",
        "rank_at_create": 2,
        "current_rank": 2,
        "current_rating_jupr": 4.0,
    }
    assert "ledger_ref" not in pending["challenges"][0]
    assert "challenger_contact" not in pending["challenges"][0]

    recent = next(section for section in payload["challenge_sections"] if section["name"] == "Recently Completed")
    assert recent["challenges"][0]["challenger"] == {
        "player_id": 3,
        "player_name": "Casey",
        "rank_at_create": 2,
        "current_rank": 1,
        "current_rating_jupr": 3.75,
    }
    assert recent["challenges"][0]["winner"] == recent["challenges"][0]["challenger"]
    assert recent["challenges"][0]["completed_at"] == "2026-01-05T00:00:00Z"
    assert "scores" not in recent["challenges"][0]
    assert "matches" not in recent["challenges"][0]
    assert "resolution_notes" not in recent["challenges"][0]
    assert payload["eligibility_authority"] == "python"
    assert len(payload["status_legend"]) == 8
    assert {item["status"] for item in payload["status_legend"]} >= {"Ready to Defend", "Protected", "Cooldown", "Inactive"}
    assert {section["title"] for section in payload["rulebook"]} == {
        "How to make a challenge",
        "Eligibility and timing",
        "Swing Partner Swap format",
        "Staff-managed exceptions",
    }
    assert "private" not in str(payload).lower()


def test_public_completed_result_resolves_only_persisted_match_ids() -> None:
    supabase = fake_supabase()
    completed = supabase._tables["ladder_challenges"][1]
    completed["public_result_json"] = {
        "version": 1,
        "match_ids": {"a": 501, "b": 502},
        "rank_change": {
            "swapped": True,
            "challenger": {"player_id": 3, "before": 2, "after": 1},
            "defender": {"player_id": 1, "before": 1, "after": 2},
        },
    }
    supabase._tables["matches"] = [
        {
            "id": 501,
            "club_id": "club",
            "date": "2026-01-05",
            "context_type": "challenge_ladder",
            "context_id": "private-operation-context-a",
            "deleted_at": None,
            "t1_p1": 3,
            "t1_p2": 2,
            "t2_p1": 1,
            "t2_p2": 4,
            "score_t1": 22,
            "score_t2": 11,
            "t1_p1_r": 1500,
            "t1_p1_r_end": 1510,
            "t1_p2_r": 1600,
            "t1_p2_r_end": 1610,
            "t2_p1_r": 1700,
            "t2_p1_r_end": 1690,
            "t2_p2_r": None,
            "t2_p2_r_end": None,
        },
        {
            "id": 502,
            "club_id": "club",
            "date": "2026-01-05",
            "context_type": "challenge_ladder",
            "context_id": "private-operation-context-b",
            "deleted_at": None,
            "t1_p1": 3,
            "t1_p2": 4,
            "t2_p1": 1,
            "t2_p2": 2,
            "score_t1": 22,
            "score_t2": 10,
            "t1_p1_r": 1510,
            "t1_p1_r_end": 1520,
            "t1_p2_r": 1400,
            "t1_p2_r_end": 1410,
            "t2_p1_r": 1690,
            "t2_p1_r_end": 1680,
            "t2_p2_r": 1610,
            "t2_p2_r_end": 1600,
        },
    ]

    payload = build_public_challenge_ladder(supabase, club_id="club")
    recent = next(
        section
        for section in payload["challenge_sections"]
        if section["name"] == "Recently Completed"
    )["challenges"][0]
    details = recent["result_details"]

    assert details["rank_change"]["challenger"] == {
        "player_id": 3,
        "player_name": "Casey",
        "before": 2,
        "after": 1,
        "delta": -1,
    }
    assert [(row["slot"], row["match_id"]) for row in details["matches"]] == [
        ("a", 501),
        ("b", 502),
    ]
    assert details["matches"][0]["score_challenger_team"] == 22
    assert details["matches"][0]["score_defender_team"] == 11
    assert details["matches"][0]["challenger_partner"] == {
        "player_id": 2,
        "player_name": "Blair",
    }
    assert details["matches"][1]["defender_partner"] == {
        "player_id": 2,
        "player_name": "Blair",
    }
    first_ratings = {
        row["player_id"]: row for row in details["matches"][0]["rating_changes"]
    }
    assert first_ratings[3]["before_jupr"] == 3.75
    assert first_ratings[3]["after_jupr"] == 3.775
    assert first_ratings[3]["delta_jupr"] == 0.025
    assert first_ratings[4]["before_jupr"] is None
    assert first_ratings[4]["delta_jupr"] is None
    assert "context_id" not in str(details)
    assert "private-operation" not in str(details)
    assert supabase.table_calls["matches"] == 1
    assert supabase.table_calls["players"] == 1


def test_public_completed_v1_result_never_falls_back_to_legacy_rows() -> None:
    supabase = fake_supabase()
    supabase._tables["ladder_challenges"][1]["public_result_json"] = {
        "version": 1,
        "match_ids": {"a": 501, "b": 502},
        "rank_change": {
            "swapped": False,
            "challenger": {"player_id": 3, "before": 2, "after": 2},
            "defender": {"player_id": 1, "before": 1, "after": 1},
        },
    }
    supabase._tables["matches"] = [
        {
            "id": match_id,
            "club_id": "club",
            "context_type": "challenge_ladder",
            "deleted_at": "2026-01-06T00:00:00Z" if match_id == 502 else None,
            "t1_p1": 3,
            "t1_p2": 2,
            "t2_p1": 1,
            "t2_p2": 4,
            "score_t1": 20,
            "score_t2": 22,
        }
        for match_id in (501, 502)
    ]
    supabase._tables["ladder_challenge_matches"] = [
        {
            "club_id": "club",
            "challenge_id": 21,
            "match_no": 1,
            "chal_partner_id": 2,
            "def_partner_id": 4,
            "g1_chal": 11,
            "g1_def": 8,
            "g2_chal": 11,
            "g2_def": 9,
            "g3_chal": None,
            "g3_def": None,
            "verified": True,
            "verified_at": "2026-01-05T00:00:00Z",
        }
    ]

    payload = build_public_challenge_ladder(supabase, club_id="club")
    recent = next(
        section
        for section in payload["challenge_sections"]
        if section["name"] == "Recently Completed"
    )["challenges"][0]

    assert "result_details" not in recent


def test_forward_result_rejects_partner_outside_the_club_player_map() -> None:
    supabase = fake_supabase()
    supabase._tables["ladder_challenges"][1]["public_result_json"] = {
        "version": 1,
        "match_ids": {"a": 501, "b": 502},
        "rank_change": {
            "swapped": False,
            "challenger": {"player_id": 3, "before": 2, "after": 2},
            "defender": {"player_id": 1, "before": 1, "after": 1},
        },
    }
    supabase._tables["matches"] = [
        {
            "id": 501,
            "club_id": "club",
            "context_type": "challenge_ladder",
            "deleted_at": None,
            "t1_p1": 3,
            "t1_p2": 999,
            "t2_p1": 1,
            "t2_p2": 2,
            "score_t1": 22,
            "score_t2": 17,
        },
        {
            "id": 502,
            "club_id": "club",
            "context_type": "challenge_ladder",
            "deleted_at": None,
            "t1_p1": 3,
            "t1_p2": 2,
            "t2_p1": 1,
            "t2_p2": 999,
            "score_t1": 22,
            "score_t2": 15,
        },
    ]

    payload = build_public_challenge_ladder(supabase, club_id="club")
    recent = next(
        section
        for section in payload["challenge_sections"]
        if section["name"] == "Recently Completed"
    )["challenges"][0]

    assert "result_details" not in recent
    assert "Player 999" not in str(payload)


def test_verified_legacy_result_projects_partial_score_without_inferred_links() -> None:
    supabase = fake_supabase()
    supabase._tables["players"] = [
        {"id": 990002, "club_id": "club", "name": "Blake Baseline", "rating": 1200, "active": True},
        {"id": 990003, "club_id": "club", "name": "Casey Court", "rating": 1242, "active": True},
        {"id": 990004, "club_id": "club", "name": "Devon Dink", "rating": 1221, "active": True},
    ]
    supabase._tables["ladder_roster"] = [
        {"id": 1, "club_id": "club", "player_id": 990003, "tier_id": "ADV", "rank": 1, "is_active": True},
        {"id": 2, "club_id": "club", "player_id": 990004, "tier_id": "ADV", "rank": 2, "is_active": True},
    ]
    supabase._tables["ladder_player_flags"] = []
    supabase._tables["ladder_challenges"] = [
        {
            "id": 990001,
            "club_id": "club",
            "challenger_id": 990004,
            "defender_id": 990003,
            "challenger_rank_at_create": 2,
            "defender_rank_at_create": 1,
            "tier_id": "ADV",
            "status": "COMPLETED",
            "created_at": "2026-07-01T00:00:00Z",
            "completed_at": "2026-07-05T19:00:00Z",
            "winner_id": 990004,
            "public_result_json": None,
        }
    ]
    supabase._tables["ladder_challenge_matches"] = [
        {
            "club_id": "club",
            "challenge_id": 990001,
            "match_no": 1,
            "chal_partner_id": 990002,
            "def_partner_id": 990004,
            "g1_chal": 11,
            "g1_def": 8,
            "g2_chal": 11,
            "g2_def": 9,
            "g3_chal": None,
            "g3_def": None,
            "verified": True,
            "verified_at": "2026-07-05T19:00:00Z",
            "private_note": "must not project",
        }
    ]

    payload = build_public_challenge_ladder(supabase, club_id="club")
    recent = next(
        section
        for section in payload["challenge_sections"]
        if section["name"] == "Recently Completed"
    )["challenges"][0]
    details = recent["result_details"]

    assert details["completeness"] == "partial"
    assert details["rank_change"] is None
    assert details["matches"] == [
        {
            "slot": "a",
            "match_id": None,
            "date": None,
            "score_challenger_team": 22,
            "score_defender_team": 17,
            "games": [
                {"game": 1, "challenger": 11, "defender": 8},
                {"game": 2, "challenger": 11, "defender": 9},
            ],
            "challenger_partner": {
                "player_id": 990002,
                "player_name": "Blake Baseline",
            },
            "defender_partner": None,
            "rating_changes": [],
        }
    ]
    assert "1 of 2 verified legacy match records" in details["notice"]
    assert "conflicts with the ranked participants" in details["warnings"][0]
    assert "verified_at" not in str(details)
    assert "private" not in str(details)


def test_verified_legacy_result_hides_partner_outside_the_club_player_map() -> None:
    supabase = fake_supabase()
    supabase._tables["ladder_challenge_matches"] = [
        {
            "club_id": "club",
            "challenge_id": 21,
            "match_no": 1,
            "chal_partner_id": 999,
            "def_partner_id": 2,
            "g1_chal": 11,
            "g1_def": 8,
            "g2_chal": 11,
            "g2_def": 9,
            "g3_chal": None,
            "g3_def": None,
            "verified": True,
            "verified_at": "2026-01-05T00:00:00Z",
        }
    ]

    payload = build_public_challenge_ladder(supabase, club_id="club")
    recent = next(
        section
        for section in payload["challenge_sections"]
        if section["name"] == "Recently Completed"
    )["challenges"][0]
    details = recent["result_details"]

    assert details["matches"][0]["challenger_partner"] is None
    assert details["matches"][0]["defender_partner"] == {
        "player_id": 2,
        "player_name": "Blair",
    }
    assert "not a current club player" in details["warnings"][0]
    assert "999" not in str(details)


def test_recently_completed_is_ordered_by_completion_and_bounded_to_twelve() -> None:
    supabase = fake_supabase()
    supabase._tables["ladder_challenges"] = [
        {
            "id": 1000 + index,
            "club_id": "club",
            "challenger_id": 2,
            "defender_id": 1,
            "challenger_rank_at_create": 2,
            "defender_rank_at_create": 1,
            "tier_id": "PREM",
            "status": "COMPLETED",
            "created_at": f"2026-02-{15 - index:02d}T00:00:00Z",
            "completed_at": f"2026-03-{index:02d}T00:00:00Z",
            "winner_id": 1,
            "public_result_json": None,
        }
        for index in range(1, 15)
    ]

    payload = build_public_challenge_ladder(supabase, club_id="club")
    recent = next(
        section
        for section in payload["challenge_sections"]
        if section["name"] == "Recently Completed"
    )

    assert [row["id"] for row in recent["challenges"]] == list(
        range(1014, 1002, -1)
    )
    assert [row["completed_at"] for row in recent["challenges"]] == sorted(
        [row["completed_at"] for row in recent["challenges"]],
        reverse=True,
    )


def test_forward_completed_results_batch_all_exact_match_reads_once() -> None:
    supabase = fake_supabase()
    challenges = []
    matches = []
    for index in range(12):
        match_a_id = 7000 + index * 2
        match_b_id = match_a_id + 1
        challenges.append(
            {
                "id": 8000 + index,
                "club_id": "club",
                "challenger_id": 2,
                "defender_id": 1,
                "challenger_rank_at_create": 2,
                "defender_rank_at_create": 1,
                "tier_id": "PREM",
                "status": "COMPLETED",
                "created_at": f"2026-04-{index + 1:02d}T00:00:00Z",
                "completed_at": f"2026-05-{index + 1:02d}T00:00:00Z",
                "winner_id": 1,
                "public_result_json": {
                    "version": 1,
                    "match_ids": {"a": match_a_id, "b": match_b_id},
                    "rank_change": {
                        "swapped": False,
                        "challenger": {
                            "player_id": 2,
                            "before": 2,
                            "after": 2,
                        },
                        "defender": {
                            "player_id": 1,
                            "before": 1,
                            "after": 1,
                        },
                    },
                },
            }
        )
        for match_id, challenger_partner, defender_partner in (
            (match_a_id, 3, 4),
            (match_b_id, 4, 3),
        ):
            matches.append(
                {
                    "id": match_id,
                    "club_id": "club",
                    "date": "2026-05-01",
                    "context_type": "challenge_ladder",
                    "context_id": f"private-{match_id}",
                    "deleted_at": None,
                    "t1_p1": 2,
                    "t1_p2": challenger_partner,
                    "t2_p1": 1,
                    "t2_p2": defender_partner,
                    "score_t1": 18,
                    "score_t2": 22,
                    "t1_p1_r": 1600,
                    "t1_p1_r_end": 1595,
                    "t1_p2_r": 1500,
                    "t1_p2_r_end": 1495,
                    "t2_p1_r": 1700,
                    "t2_p1_r_end": 1705,
                    "t2_p2_r": 1400,
                    "t2_p2_r_end": 1405,
                }
            )
    supabase._tables["ladder_challenges"] = challenges
    supabase._tables["matches"] = matches

    payload = build_public_challenge_ladder(supabase, club_id="club")
    recent = next(
        section
        for section in payload["challenge_sections"]
        if section["name"] == "Recently Completed"
    )

    assert len(recent["challenges"]) == 12
    assert all(len(row["result_details"]["matches"]) == 2 for row in recent["challenges"])
    assert supabase.table_calls["matches"] == 1
    assert supabase.table_calls["players"] == 1
    assert supabase.table_calls["ladder_challenge_matches"] == 0


def test_legacy_completed_results_batch_all_verified_rows_once() -> None:
    supabase = fake_supabase()
    challenges = []
    legacy_rows = []
    for index in range(12):
        challenge_id = 9000 + index
        challenges.append(
            {
                "id": challenge_id,
                "club_id": "club",
                "challenger_id": 2,
                "defender_id": 1,
                "challenger_rank_at_create": 2,
                "defender_rank_at_create": 1,
                "tier_id": "PREM",
                "status": "COMPLETED",
                "created_at": f"2026-05-{index + 1:02d}T00:00:00Z",
                "completed_at": f"2026-06-{index + 1:02d}T00:00:00Z",
                "winner_id": 1,
                "public_result_json": None,
            }
        )
        legacy_rows.append(
            {
                "club_id": "club",
                "challenge_id": challenge_id,
                "match_no": 1,
                "chal_partner_id": 3,
                "def_partner_id": 4,
                "g1_chal": 11,
                "g1_def": 8,
                "g2_chal": 11,
                "g2_def": 9,
                "g3_chal": None,
                "g3_def": None,
                "verified": True,
                "verified_at": "2026-06-30T00:00:00Z",
            }
        )
    supabase._tables["ladder_challenges"] = challenges
    supabase._tables["ladder_challenge_matches"] = legacy_rows

    payload = build_public_challenge_ladder(supabase, club_id="club")
    recent = next(
        section
        for section in payload["challenge_sections"]
        if section["name"] == "Recently Completed"
    )

    assert len(recent["challenges"]) == 12
    assert all(
        row["result_details"]["completeness"] == "partial"
        for row in recent["challenges"]
    )
    assert supabase.table_calls["ladder_challenge_matches"] == 1
    assert supabase.table_calls["matches"] == 0
    assert supabase.table_calls["players"] == 1


def test_projection_read_failure_never_substitutes_legacy_result_evidence() -> None:
    supabase = fake_supabase()
    supabase.fail_challenge_projection = True
    supabase._tables["ladder_challenge_matches"] = [
        {
            "club_id": "club",
            "challenge_id": 21,
            "match_no": 1,
            "g1_chal": 11,
            "g1_def": 8,
            "g2_chal": 11,
            "g2_def": 9,
            "verified": True,
        }
    ]

    payload = build_public_challenge_ladder(supabase, club_id="club")
    recent = next(
        section
        for section in payload["challenge_sections"]
        if section["name"] == "Recently Completed"
    )

    assert recent["challenges"] == []
    assert supabase.table_calls["ladder_challenges"] == 1
    assert supabase.table_calls["ladder_challenge_matches"] == 0


def test_public_challenge_ladder_degrades_to_empty_when_tables_missing() -> None:
    payload = build_public_challenge_ladder(FakeSupabase({}), club_id="club")

    assert payload["summary"]["active_player_count"] == 0
    assert payload["tiers"][0]["players"] == []
    assert payload["quick_rules"]
    assert payload["rulebook"]
    assert payload["status_legend"]


def test_inactive_roster_players_are_not_projected_when_no_active_players_exist() -> None:
    supabase = fake_supabase()
    for player in supabase._tables["players"]:
        player["active"] = False

    payload = build_public_challenge_ladder(supabase, club_id="club")

    assert payload["summary"]["active_player_count"] == 0
    assert all(not tier["players"] for tier in payload["tiers"])


def test_public_challenge_eligibility_is_computed_by_python_policy() -> None:
    supabase = fake_supabase()
    supabase._tables["ladder_challenges"] = []
    supabase._tables["ladder_player_flags"] = []
    supabase._tables["ladder_roster"] = [
        {"id": 10, "club_id": "club", "player_id": 1, "tier_id": "PREM", "rank": 1, "is_active": True},
        {"id": 11, "club_id": "club", "player_id": 2, "tier_id": "PREM", "rank": 2, "is_active": True},
        {"id": 12, "club_id": "club", "player_id": 3, "tier_id": "PREM", "rank": 8, "is_active": True},
    ]

    payload = build_public_challenge_ladder(supabase, club_id="club")
    premier = next(tier for tier in payload["tiers"] if tier["tier_id"] == "PREM")
    by_name = {player["player_name"]: player for player in premier["players"]}

    assert by_name["Alex"]["eligibility"]["eligible_opponents"] == []
    assert by_name["Blair"]["eligibility"]["eligible_opponents"] == [
        {
            "player_id": 1,
            "player_name": "Alex",
            "rank": 1,
            "status": "Ready to Defend",
            "status_short": "Ready",
            "rank_gap": 1,
        }
    ]
    assert {row["player_name"] for row in by_name["Casey"]["eligibility"]["eligible_opponents"]} == {"Alex", "Blair"}
    assert payload["summary"]["eligible_pair_count"] == 3


def test_protected_player_can_initiate_against_cooldown_player() -> None:
    supabase = fake_supabase()
    supabase._tables["ladder_player_flags"] = []
    supabase._tables["ladder_roster"] = [
        {"id": 10, "club_id": "club", "player_id": 1, "tier_id": "PREM", "rank": 1, "is_active": True},
        {"id": 11, "club_id": "club", "player_id": 2, "tier_id": "PREM", "rank": 2, "is_active": True},
    ]
    supabase._tables["ladder_challenges"] = [
        {
            "id": 99,
            "club_id": "club",
            "challenger_id": 1,
            "defender_id": 2,
            "tier_id": "PREM",
            "status": "COMPLETED",
            "created_at": "2099-01-01T00:00:00Z",
            "completed_at": "2099-01-02T00:00:00Z",
            "winner_id": 2,
        }
    ]

    payload = build_public_challenge_ladder(supabase, club_id="club")
    premier = next(tier for tier in payload["tiers"] if tier["tier_id"] == "PREM")
    by_name = {player["player_name"]: player for player in premier["players"]}

    assert by_name["Blair"]["status"] == "Protected"
    assert by_name["Blair"]["eligibility"]["can_initiate"] is True
    assert by_name["Blair"]["eligibility"]["eligible_opponents"][0]["player_name"] == "Alex"
    assert by_name["Alex"]["status"] == "Cooldown"
    assert by_name["Alex"]["eligibility"]["can_receive"] is True
