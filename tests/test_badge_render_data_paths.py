import pandas as pd

from jupr_app.domain.gamification.profile import build_gamification_summary
from jupr_app.ui.badge_data import normalize_player_badges_frame
from jupr_app.ui.pages.leaderboards import _build_badge_map, _fetch_leaderboard_badges
from jupr_app.ui.pages.players import get_player_trophy_case, resolve_player_badges_for_profile, select_featured_cuts


class _SupabaseQueryStub:
    def __init__(self, payload, calls):
        self.payload = payload
        self.calls = calls
        self.filters = {}

    def select(self, _value):
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def in_(self, column, value):
        self.filters[column] = list(value)
        return self

    def execute(self):
        self.calls.append(dict(self.filters))
        in_players = self.filters.get("player_id")
        if in_players is None:
            data = self.payload
        else:
            in_set = {int(v) for v in in_players}
            data = [row for row in self.payload if int(row.get("player_id")) in in_set]

        return type("_Resp", (), {"data": data})()


class _SupabaseStub:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def table(self, _name):
        return _SupabaseQueryStub(self.payload, self.calls)


class _Ctx:
    def __init__(self, df_player_badges, df_badges, supabase, club_id="club-1"):
        self.df_player_badges = df_player_badges
        self.df_badges = df_badges
        self.supabase = supabase
        self.club_id = club_id


def _badge_defs():
    return pd.DataFrame(
        [
            {"badge_id": "participant", "name": "Participant", "prestige": 5, "is_active": True, "category": "Participation"},
            {"badge_id": "giant_slayer", "name": "Giant Slayer", "prestige": 75, "is_active": True, "category": "Rivalries"},
        ]
    )


def test_leaderboard_badges_handles_string_player_id_from_ctx():
    ctx = _Ctx(
        df_player_badges=pd.DataFrame(
            [{"club_id": "club-1", "player_id": "12", "badge_id": "participant", "earned_at": "2026-02-08T00:00:00Z"}]
        ),
        df_badges=_badge_defs(),
        supabase=_SupabaseStub(payload=[]),
    )

    out = _fetch_leaderboard_badges(ctx, ["12"])
    assert len(out) == 1
    assert int(out.iloc[0]["player_id"]) == 12
    assert out.iloc[0]["name"] == "Participant"


def test_leaderboard_badges_falls_back_when_ctx_misses_visible_players():
    fallback_payload = [
        {
            "player_id": 22,
            "badge_id": "participant",
            "earned_at": "2026-02-08T00:00:00Z",
            "badges": {"badge_id": "participant", "name": "Participant", "prestige": 5, "category": "Participation"},
        }
    ]
    ctx = _Ctx(
        df_player_badges=pd.DataFrame(
            [{"club_id": "club-1", "player_id": 99, "badge_id": "participant", "earned_at": "2026-02-08T00:00:00Z"}]
        ),
        df_badges=_badge_defs(),
        supabase=_SupabaseStub(payload=fallback_payload),
    )

    out = _fetch_leaderboard_badges(ctx, [22])
    assert len(out) == 1
    assert int(out.iloc[0]["player_id"]) == 22
    assert out.iloc[0]["badge_id"] == "participant"


def test_leaderboard_badges_complete_ctx_coverage_skips_fallback_query():
    supabase = _SupabaseStub(payload=[])
    ctx = _Ctx(
        df_player_badges=pd.DataFrame(
            [
                {"club_id": "club-1", "player_id": 12, "badge_id": "participant", "earned_at": "2026-02-08T00:00:00Z"},
                {"club_id": "club-1", "player_id": 22, "badge_id": "giant_slayer", "earned_at": "2026-02-09T00:00:00Z"},
            ]
        ),
        df_badges=_badge_defs(),
        supabase=supabase,
    )

    out = _fetch_leaderboard_badges(ctx, [12, 22])
    assert set(out["player_id"].tolist()) == {12, 22}
    assert supabase.calls == []


def test_leaderboard_badges_partial_ctx_coverage_fetches_missing_only():
    supabase = _SupabaseStub(
        payload=[
            {
                "player_id": 22,
                "badge_id": "participant",
                "earned_at": "2026-02-08T00:00:00Z",
                "badges": {"badge_id": "participant", "name": "Participant", "prestige": 5, "category": "Participation"},
            },
            {
                "player_id": 33,
                "badge_id": "giant_slayer",
                "earned_at": "2026-02-08T00:00:00Z",
                "badges": {"badge_id": "giant_slayer", "name": "Giant Slayer", "prestige": 75, "category": "Rivalries"},
            },
        ]
    )
    ctx = _Ctx(
        df_player_badges=pd.DataFrame(
            [{"club_id": "club-1", "player_id": 12, "badge_id": "participant", "earned_at": "2026-02-07T00:00:00Z"}]
        ),
        df_badges=_badge_defs(),
        supabase=supabase,
    )

    out = _fetch_leaderboard_badges(ctx, [12, 22, 33])
    assert set(out["player_id"].tolist()) == {12, 22, 33}
    assert len(supabase.calls) == 1
    assert set(supabase.calls[0]["player_id"]) == {22, 33}


def test_leaderboard_badge_map_includes_single_badge_players_and_dedupes_duplicates():
    merged = pd.DataFrame(
        [
            {"player_id": 101, "badge_id": "participant", "name": "Participant", "prestige": 5, "earned_at": "2026-02-08T00:00:00Z"},
            {"player_id": 101, "badge_id": "participant", "name": "Participant", "prestige": 5, "earned_at": "2026-02-08T00:00:00Z"},
            {"player_id": 102, "badge_id": "giant_slayer", "name": "Giant Slayer", "prestige": 75, "earned_at": "2026-02-09T00:00:00Z"},
        ]
    )

    badge_map = _build_badge_map(merged)
    assert 101 in badge_map
    assert 102 in badge_map
    assert len(badge_map[101]) == 1
    assert badge_map[101][0].badge_id == "participant"
    assert len(badge_map[102]) == 1


def test_leaderboard_badge_map_missing_prestige_defaults_to_zero_without_crashing():
    merged = pd.DataFrame(
        [
            {"player_id": 301, "badge_id": "participant", "name": "Participant", "earned_at": "2026-02-08T00:00:00Z"},
        ]
    )

    badge_map = _build_badge_map(merged)
    assert 301 in badge_map
    assert len(badge_map[301]) == 1
    assert badge_map[301][0].prestige == 0


def test_leaderboard_badge_map_uses_normal_prestige_column_when_present():
    merged = pd.DataFrame(
        [
            {"player_id": 302, "badge_id": "participant", "name": "Participant", "prestige": 5, "earned_at": "2026-02-08T00:00:00Z"},
            {"player_id": 302, "badge_id": "giant_slayer", "name": "Giant Slayer", "prestige": 75, "earned_at": "2026-02-09T00:00:00Z"},
        ]
    )

    badge_map = _build_badge_map(merged)
    assert [badge.badge_id for badge in badge_map[302]] == ["giant_slayer", "participant"]
    assert [badge.prestige for badge in badge_map[302]] == [75, 5]


def test_leaderboard_badge_map_uses_suffix_style_prestige_column_when_available():
    merged = pd.DataFrame(
        [
            {
                "player_id": 303,
                "badge_id": "participant",
                "badges.name": "Participant",
                "prestige_def": 11,
                "created_at": "2026-02-08T00:00:00Z",
            },
            {
                "player_id": 303,
                "badge_id": "giant_slayer",
                "badges.name": "Giant Slayer",
                "prestige_def": 99,
                "created_at": "2026-02-09T00:00:00Z",
            },
        ]
    )

    badge_map = _build_badge_map(merged)
    assert [badge.badge_id for badge in badge_map[303]] == ["giant_slayer", "participant"]
    assert [badge.prestige for badge in badge_map[303]] == [99, 11]
    assert [badge.name for badge in badge_map[303]] == ["Giant Slayer", "Participant"]


def test_leaderboard_badge_map_builds_from_minimal_row_shape():
    merged = pd.DataFrame(
        [
            {"player_id": 304, "badge_id": "participant", "name": "Participant"},
        ]
    )

    badge_map = _build_badge_map(merged)
    assert 304 in badge_map
    assert len(badge_map[304]) == 1
    assert badge_map[304][0].badge_id == "participant"
    assert badge_map[304][0].name == "Participant"
    assert badge_map[304][0].prestige == 0


def test_leaderboard_badge_map_prefers_plain_name_column():
    merged = pd.DataFrame(
        [
            {
                "player_id": 401,
                "badge_id": "participant",
                "name": "Participant",
                "name_x": "Wrong X",
                "name_y": "Wrong Y",
                "prestige": 5,
            }
        ]
    )

    badge_map = _build_badge_map(merged)
    assert badge_map[401][0].name == "Participant"


def test_leaderboard_badge_map_reads_suffix_name_columns():
    merged = pd.DataFrame(
        [
            {
                "player_id": 402,
                "badge_id": "participant",
                "name_y": "Participant",
                "prestige_def": 5,
                "created_at": "2026-02-08T00:00:00Z",
            },
            {
                "player_id": 402,
                "badge_id": "giant_slayer",
                "name_y": "Giant Slayer",
                "prestige_def": 75,
                "created_at": "2026-02-09T00:00:00Z",
            },
        ]
    )

    badge_map = _build_badge_map(merged)
    assert [badge.name for badge in badge_map[402]] == ["Giant Slayer", "Participant"]


def test_leaderboard_badge_map_reads_nested_name_and_fallback_prestige():
    merged = pd.DataFrame(
        [
            {
                "player_id": 403,
                "badges.badge_id": "participant",
                "badges.name": "Participant",
                "badge_id_def": "participant",
                "prestige_def": 5,
            },
            {
                "player_id": 403,
                "badges.badge_id": "giant_slayer",
                "badges.name": "Giant Slayer",
                "badge_id_def": "giant_slayer",
                "prestige_def": 75,
            },
        ]
    )

    badge_map = _build_badge_map(merged)
    assert [badge.name for badge in badge_map[403]] == ["Giant Slayer", "Participant"]
    assert [badge.prestige for badge in badge_map[403]] == [75, 5]
    assert all(badge.name != "Badge" for badge in badge_map[403])


def test_player_profile_summary_unlocked_badges_from_selected_player_rows():
    class _PlayerCtx:
        df_player_badges = pd.DataFrame(
            [
                {"club_id": "club-1", "player_id": "7", "badge_id": "participant", "earned_at": "2026-02-08T00:00:00Z"},
                {"club_id": "club-1", "player_id": "8", "badge_id": "giant_slayer", "earned_at": "2026-02-09T00:00:00Z"},
            ]
        )

    resolved = resolve_player_badges_for_profile(_PlayerCtx(), _SupabaseStub([]), "club-1", 7)
    summary = build_gamification_summary(7, _badge_defs(), resolved)
    assert summary["collected_unique_count"] == 1
    assert summary["unlocked_badges"][0]["badge_id"] == "participant"


def test_trophy_case_can_be_empty_while_badge_collection_is_nonzero():
    player_badges = pd.DataFrame(
        [{"club_id": "club-1", "player_id": "7", "badge_id": "participant", "earned_at": "2026-02-08T00:00:00Z"}]
    )
    normalized = normalize_player_badges_frame(player_badges)
    summary = build_gamification_summary(7, _badge_defs(), normalized)
    trophy_case = get_player_trophy_case(normalized, 7, completed_league_ids=set())
    assert summary["collected_unique_count"] == 1
    assert trophy_case.empty


def test_grace_like_single_participant_populates_collection_and_featured():
    player_badges = pd.DataFrame(
        [{"club_id": "club-1", "player_id": "101", "badge_id": "participant", "earned_at": "2026-02-08T00:00:00Z"}]
    )
    summary = build_gamification_summary(101, _badge_defs(), normalize_player_badges_frame(player_badges))
    featured = select_featured_cuts(summary["unlocked_badges"], limit=3)
    assert summary["collected_unique_count"] == 1
    assert len(featured) == 1
    assert featured[0]["badge_id"] == "participant"
