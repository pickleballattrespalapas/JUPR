from types import SimpleNamespace

import pandas as pd

from jupr_app.data import load
from jupr_app.ui.page_registry import HIDDEN_PAGE_KEYS, PUBLIC_NAV_KEYS
from jupr_app.ui.pages import _fetch_player_badges_resilient
from jupr_app.ui.pages import badge_codex
from jupr_app.ui.pages.players import get_player_trophy_case, resolve_player_badges_for_profile


class EmptyQuery:
    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def is_(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def range(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=[], count=0)


class NoBadgeReads:
    def __init__(self):
        self.tables = []

    def table(self, name):
        self.tables.append(name)
        assert name not in {"badges", "player_badges", "badge_eval_queue"}
        return EmptyQuery()


def test_legacy_load_avoids_all_badge_reads_and_preflight(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Retired badge preflight must not run")

    monkeypatch.setattr(load, "ensure_badge_schema_preflight", forbidden)
    db = NoBadgeReads()
    data = load.load_data(db, "tres_palapas", match_limit=5)
    assert data[5].empty and data[6].empty
    assert set(db.tables) == {"players", "league_ratings", "matches", "leagues_metadata"}
    assert data[9] is False


def test_legacy_badge_route_hidden_and_has_no_context_reads(monkeypatch):
    class NoContext:
        def __getattr__(self, name):
            raise AssertionError("Retired catalog must not read context")

    messages = []
    monkeypatch.setattr(badge_codex.st, "info", messages.append)
    badge_codex.render(NoContext())
    assert messages == ["This page is no longer available."]
    assert "badge_codex" not in PUBLIC_NAV_KEYS
    assert {"badge_codex", "badge_debug", "badge_audit"}.issubset(HIDDEN_PAGE_KEYS)


def test_legacy_profile_keeps_formal_trophies_from_cached_rows_only():
    rows = pd.DataFrame([
        {"club_id": "tres_palapas", "player_id": 12, "badge_id": "participant"},
        {"club_id": "tres_palapas", "player_id": 12, "badge_id": "league_champion", "context_type": "league", "context_id": "Fall:podium:1", "value_json": {"league_id": "Fall", "rank": 1}},
        {"club_id": "tres_palapas", "player_id": 99, "badge_id": "league_runner_up"},
    ])
    db = NoBadgeReads()
    selected = resolve_player_badges_for_profile(SimpleNamespace(df_player_badges=rows, df_badges=pd.DataFrame([{ "badge_id": "league_champion", "name": "League Champion"}])), db, "tres_palapas", 12)
    assert selected["badge_id"].tolist() == ["league_champion"]
    assert db.tables == []
    trophies = get_player_trophy_case(selected, 12, {"Fall"})
    assert trophies["badge_id"].tolist() == ["league_champion"]


def test_legacy_trophy_query_filters_before_loading_definitions():
    queries = []

    class Query:
        def __init__(self, table):
            self.table = table
            self.filters = {}

        def select(self, columns):
            return self

        def eq(self, column, value):
            self.filters[column] = value
            return self

        def in_(self, column, value):
            self.filters[column] = value
            return self

        def execute(self):
            queries.append((self.table, self.filters))
            if self.table == "player_badges":
                return SimpleNamespace(data=[
                    {"player_id": 12, "badge_id": "league_champion", "context_type": "league"},
                    {"player_id": 12, "badge_id": "participant"},
                ])
            return SimpleNamespace(data=[{"badge_id": "league_champion", "name": "League Champion"}])

    result = _fetch_player_badges_resilient(SimpleNamespace(table=Query), "tres_palapas", 12)
    assert result["badge_id"].tolist() == ["league_champion"]
    assert queries[0][1]["player_id"] == 12
    assert queries[0][1]["club_id"] == "tres_palapas"
    assert "participant" not in queries[0][1]["badge_id"]
    assert queries[1][1]["badge_id"] == ["league_champion"]
