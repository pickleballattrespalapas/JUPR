from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from jupr_app.domain.notifications.player_update_charts import render_player_digest_chart_png
from jupr_app.domain.notifications.player_update_email_template import (
    build_player_update_email_html,
    build_player_update_email_text,
)
from jupr_app.domain.recaps import player_weekly_digest as digest_mod


class _FakeExec:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, table_name: str, data_map: dict[str, list[dict]]):
        self._table_name = table_name
        self._data_map = data_map

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def gte(self, *_args, **_kwargs):
        return self

    def lte(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def in_(self, *_args, **_kwargs):
        return self

    def execute(self):
        return _FakeExec(self._data_map.get(self._table_name, []))


class _FakeSupabase:
    def __init__(self, data_map: dict[str, list[dict]]):
        self._data_map = data_map

    def table(self, table_name: str):
        return _FakeQuery(table_name, self._data_map)


class _Ctx:
    def __init__(self):
        self.club_id = "club-1"
        self.supabase = None
        self.id_to_name = {1: "Ada", 2: "Joel", 3: "Keith", 4: "Natasha"}


def test_display_range_uses_month_day_ordinals():
    assert digest_mod._display_range(date(2026, 3, 15), date(2026, 4, 22)) == "March 15th – April 22nd"


def test_load_badges_uses_canonical_description(monkeypatch):
    class _Badge:
        title = "Blowout Artist"

        class display:
            flavor = "Win by a big margin."
            requirements = "Win by 6+"

    monkeypatch.setattr(digest_mod, "badge_schema_by_id", lambda: {"blowout_artist": _Badge()})

    supabase = _FakeSupabase(
        {
            "player_badges": [
                {
                    "badge_id": "blowout_artist",
                    "earned_at": datetime(2026, 4, 10, tzinfo=timezone.utc).isoformat(),
                    "context_type": "match",
                    "context_id": "m1",
                },
                {
                    "badge_id": "blowout_artist",
                    "earned_at": datetime(2026, 4, 11, tzinfo=timezone.utc).isoformat(),
                    "context_type": "match",
                    "context_id": "m2",
                },
            ],
            "badges": [],
        }
    )

    rows = digest_mod._load_badges_earned(
        supabase,
        "club-1",
        1,
        datetime(2026, 4, 1, tzinfo=timezone.utc),
        datetime(2026, 4, 30, tzinfo=timezone.utc),
    )
    grouped = digest_mod._group_badges_for_display(rows)

    assert grouped[0]["name"] == "Blowout Artist"
    assert grouped[0]["count"] == 2
    assert grouped[0]["description"] == "Win by a big margin."


def test_digest_has_chart_points_and_match_rows_when_window_active(monkeypatch):
    base_df = pd.DataFrame(
        [
            {
                "id": 10,
                "Date": pd.Timestamp("2026-04-01T16:00:00Z"),
                "League": "L1",
                "Score": "11-8",
                "Result": "WIN",
                "Overall Δ": 0.01,
                "Overall After": 3.2,
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 8,
            },
            {
                "id": 11,
                "Date": pd.Timestamp("2026-04-03T16:00:00Z"),
                "League": "L1",
                "Score": "9-11",
                "Result": "LOSS",
                "Overall Δ": -0.005,
                "Overall After": 3.195,
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 9,
                "score_t2": 11,
            },
        ]
    )

    monkeypatch.setattr(digest_mod, "build_player_overall_rating_series", lambda *_args, **_kwargs: base_df)
    monkeypatch.setattr(digest_mod, "filter_rating_series_for_window", lambda *_args, **_kwargs: base_df)
    monkeypatch.setattr(digest_mod, "_load_badges_earned", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(digest_mod, "_load_trophies_earned", lambda *_args, **_kwargs: [])

    digest = digest_mod.compute_player_weekly_digest(_Ctx(), 1, date(2026, 4, 1), date(2026, 4, 10))

    assert digest["chart"]["points"], "Expected chart points for active span"
    assert digest["chart"]["points"][0]["result"] == "ANCHOR"
    assert len([p for p in digest["chart"]["points"] if not p.get("is_anchor")]) == 2
    assert digest["matches_played_rows"], "Expected matches list rows"
    assert digest["matches_played_rows"][0]["summary"].startswith("L 9-11 — April 3rd — with Joel vs Keith / Natasha")


def test_digest_summary_formats_multi_day_with_result_score_first(monkeypatch):
    base_df = pd.DataFrame(
        [
            {
                "id": 11,
                "Date": pd.Timestamp("2026-04-24T16:00:00Z"),
                "League": "L1",
                "Score": "10-12",
                "Result": "LOSS",
                "Overall Δ": -0.01,
                "Overall After": 3.19,
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 10,
                "score_t2": 12,
            },
        ]
    )

    monkeypatch.setattr(digest_mod, "build_player_overall_rating_series", lambda *_args, **_kwargs: base_df)
    monkeypatch.setattr(digest_mod, "filter_rating_series_for_window", lambda *_args, **_kwargs: base_df)
    monkeypatch.setattr(digest_mod, "_load_badges_earned", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(digest_mod, "_load_trophies_earned", lambda *_args, **_kwargs: [])

    digest = digest_mod.compute_player_weekly_digest(_Ctx(), 1, date(2026, 4, 20), date(2026, 4, 25))
    assert digest["matches_played_rows"][0]["summary"] == "L 10-12 — April 24th — with Joel vs Keith / Natasha"


def test_chart_renderer_accepts_date_points_without_match_numbers():
    digest = {
        "chart": {
            "points": [
                {"date": "2026-04-22T00:00:00+00:00", "overall_after": 3.2},
            ]
        }
    }

    png = render_player_digest_chart_png(digest)
    assert isinstance(png, bytes)
    assert len(png) > 100


def test_email_html_renders_matches_and_quiet_empty_trophies():
    digest = {
        "player_name": "Ada",
        "display_range": "March 15th – April 22nd",
        "summary": {"overall_jupr_after": 3.2, "overall_delta": 0.02, "record": "2-1", "matches_played": 3},
        "chart": {"points": [{"date": "2026-04-22T00:00:00+00:00", "overall_after": 3.2}]},
        "badges_grouped": [{"name": "Blowout Artist", "count": 3, "description": "Win by a big margin.", "prestige": 50}],
        "matches_played_rows": [
            {"summary": "April 3rd — with Joel vs Keith / Natasha — W 15-11"},
        ],
        "people": {"top_partners": [], "top_opponents": []},
        "links": {"player_profile": "https://example.com/p", "unsubscribe": "https://example.com/u"},
        "trophies_earned": [],
    }

    html = build_player_update_email_html(digest, chart_cid="chart-1")
    assert "Chart unavailable for this date range" not in html
    assert "Matches" in html
    assert "Blowout Artist x3" in html
    assert "None this period" not in html


def test_email_templates_render_all_recent_matches_without_cap():
    digest = {
        "player_name": "Ada",
        "display_range": "April 24th – April 24th",
        "summary": {"overall_jupr_after": 3.2, "overall_delta": 0.02, "record": "6-0", "matches_played": 6},
        "chart": {"points": [{"date": "2026-04-24T00:00:00+00:00", "overall_after": 3.2}]},
        "matches_played_rows": [{"summary": f"W 11-{i} — with Joel vs Keith / Natasha"} for i in range(1, 7)],
        "links": {"player_profile": "https://example.com/p", "unsubscribe": "https://example.com/u"},
    }

    html = build_player_update_email_html(digest, chart_cid="chart-1")
    text = build_player_update_email_text(digest)

    assert html.count("with Joel vs Keith / Natasha") == 6
    assert text.count("with Joel vs Keith / Natasha") == 6
