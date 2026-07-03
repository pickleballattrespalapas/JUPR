from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.admin_match_log_service import build_admin_match_log


class FakeQuery:
    def __init__(self, rows):
        self.rows = list(rows)
        self.filters: list[tuple[str, object]] = []
        self.order_key: str | None = None
        self.order_desc = False
        self.limit_value: int | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def execute(self):
        result = list(self.rows)
        for key, expected in self.filters:
            result = [row for row in result if str(row.get(key)) == str(expected)]
        if self.order_key:
            result = sorted(result, key=lambda row: str(row.get(self.order_key) or ""), reverse=self.order_desc)
        if self.limit_value is not None:
            result = result[: self.limit_value]
        return SimpleNamespace(data=result)


class FakeSupabase:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return FakeQuery(self.tables.get(name, []))


def fake_supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "players": [
                {"club_id": "club", "id": 1, "name": "Alex"},
                {"club_id": "club", "id": 2, "name": "Blair"},
                {"club_id": "club", "id": 3, "name": "Casey"},
                {"club_id": "club", "id": 4, "name": "Devon"},
            ],
            "matches": [
                {
                    "club_id": "club",
                    "id": 1,
                    "date": "2026-03-01T10:00:00Z",
                    "league": "Open",
                    "week_tag": "Week 1",
                    "match_type": "Live Match",
                    "t1_p1": 1,
                    "t1_p2": 2,
                    "t2_p1": 3,
                    "t2_p2": 4,
                    "score_t1": 11,
                    "score_t2": 8,
                    "notes": "private note should not appear",
                },
                {
                    "club_id": "club",
                    "id": 2,
                    "date": "2026-03-01T10:02:00Z",
                    "league": "Open",
                    "week_tag": "Week 1",
                    "match_type": "Live Match",
                    "t1_p1": 4,
                    "t1_p2": 3,
                    "t2_p1": 2,
                    "t2_p2": 1,
                    "score_t1": 8,
                    "score_t2": 11,
                },
                {
                    "club_id": "club",
                    "id": 3,
                    "date": "2026-03-02T10:00:00Z",
                    "league": "POPUP",
                    "week_tag": "Event",
                    "match_type": "PopUp",
                    "t1_p1": 1,
                    "t1_p2": 3,
                    "t2_p1": 2,
                    "t2_p2": 4,
                    "score_t1": 9,
                    "score_t2": 11,
                },
            ],
        }
    )


def test_admin_match_log_disabled_is_db_free(monkeypatch) -> None:
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", raising=False)

    payload = build_admin_match_log(None, club_id="club")

    assert payload["enabled"] is False
    assert payload["status"] == "streamlit_fallback"
    assert payload["matches"] == []
    assert payload["correction_plan"]["mode"] == "planning_only"


def test_admin_match_log_duplicate_scan(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")

    payload = build_admin_match_log(fake_supabase(), club_id="club", filter_type="League", limit=20)

    assert payload["enabled"] is True
    assert payload["summary"]["duplicate_groups"] == 1
    assert payload["duplicate_groups"][0]["keep_id"] == 1
    assert payload["duplicate_groups"][0]["delete_ids"] == [2]
    assert payload["duplicate_delete_preview"]["recommended_replay_scope"] == "ALL"
    assert payload["duplicate_delete_preview"]["recompute_scope"] == {"standings": True, "ratings": True}
    assert payload["matches"][0]["team1"][0]["name"] in {"Alex", "Devon"}
    assert "notes" not in payload["matches"][0]


def test_admin_match_log_filters_popup(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")

    payload = build_admin_match_log(fake_supabase(), club_id="club", filter_type="Pop-Up", limit=20)

    assert payload["summary"]["returned_matches"] == 1
    assert payload["matches"][0]["match_type"] == "PopUp"
    assert payload["duplicate_groups"] == []
