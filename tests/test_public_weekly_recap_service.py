from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.public_weekly_recap_service import (
    build_public_weekly_recaps,
    build_weekly_recap_pdf_bytes,
)


class FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters: dict[str, object] = {}
        self._order_key: str | None = None
        self._order_desc = False

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def order(self, key, desc=False):
        self._order_key = key
        self._order_desc = bool(desc)
        return self

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
        if self._order_key:
            rows = sorted(rows, key=lambda row: str(row.get(self._order_key) or ""), reverse=self._order_desc)
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return FakeQuery(self._tables.get(name, []))


def fake_supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "weekly_recaps": [
                {
                    "club_id": "club",
                    "week_start": "2026-02-01",
                    "week_end": "2026-02-07",
                    "status": "draft",
                    "final_json": {"highlights": ["draft should not be public"]},
                    "edits_json": {"private": True},
                },
                {
                    "club_id": "club",
                    "week_start": "2026-02-08",
                    "week_end": "2026-02-14",
                    "status": "published",
                    "updated_at": "2026-02-15T00:00:00Z",
                    "final_json": {
                        "week_start": "2026-02-08",
                        "week_end": "2026-02-14",
                        "numbers": {"matches": 12, "players": 18, "private_metric": {"hidden": True}, "internal_count": 99},
                        "numbers_cards": [{"key": "internal_count", "label": "Private", "value": 99}],
                        "spotlight": [
                            {
                                "key": "TOP_PERFORMER_WEEK",
                                "label": "Top Performer",
                                "players": ["Alex", "Blair"],
                                "description": "Best week",
                                "order": 1,
                                "include": True,
                                "candidate_ids": [1, 2],
                            }
                        ],
                        "around_club": {
                            "leagues": [{"league_name": "Open", "highlights": [{"display": "Open had 8 matches", "private": "hidden"}]}],
                        },
                        "looking_ahead": ["More matches next week"],
                        "internal_notes": "hidden",
                    },
                },
            ]
        }
    )


def test_public_weekly_recaps_returns_only_published_sanitized_rows() -> None:
    payload = build_public_weekly_recaps(fake_supabase(), club_id="club")

    assert len(payload["recaps"]) == 1
    assert payload["recaps"][0]["week_start"] == "2026-02-08"
    selected = payload["selected_recap"]
    assert selected["summary"]["headline"] == "Top Performer: Alex, Blair"
    recap = selected["recap"]
    assert recap["numbers"] == {"matches": 12, "players": 18}
    assert [card["key"] for card in recap["numbers_cards"]] == [
        "matches",
        "players",
        "leagues",
        "round_robins",
        "community_events",
        "new_faces",
    ]
    assert recap["spotlight"][0]["players"] == ["Alex", "Blair"]
    assert "candidate_ids" not in recap["spotlight"][0]
    assert "internal_notes" not in recap
    assert "private" not in recap["around_club"]["leagues"][0]["highlights"][0]


def test_public_weekly_recaps_can_select_specific_published_week() -> None:
    payload = build_public_weekly_recaps(fake_supabase(), club_id="club", week_start="2026-02-08")

    assert payload["selected_recap"]["week_start"] == "2026-02-08"


def test_weekly_recap_pdf_builder_returns_pdf_bytes() -> None:
    payload = build_public_weekly_recaps(fake_supabase(), club_id="club")
    selected = payload["selected_recap"]

    pdf = build_weekly_recap_pdf_bytes(selected["recap"], week_start=selected["week_start"], week_end=selected["week_end"])

    assert pdf.startswith(b"%PDF-1.4")
    assert b"JUPR Weekly Recap" in pdf
    assert b"Weekly Numbers" in pdf
    assert b"Spotlight Reel" in pdf
    assert b"Around the Club" in pdf
    assert b"Tournaments" in pdf
    assert b"Looking Ahead" in pdf


def test_public_weekly_recaps_pages_published_rows_and_keeps_deep_link_private_safe() -> None:
    rows = [
        {
            "club_id": "club",
            "week_start": f"2026-03-{day:02d}",
            "week_end": f"2026-03-{day:02d}",
            "status": "published",
            "final_json": {"numbers": {"matches": day}, "internal_notes": f"private-{day}"},
        }
        for day in range(1, 8)
    ]
    rows.append(
        {
            "club_id": "club",
            "week_start": "2026-04-01",
            "week_end": "2026-04-07",
            "status": "draft",
            "final_json": {"highlights": ["draft must stay private"]},
        }
    )
    supabase = FakeSupabase({"weekly_recaps": rows})

    first = build_public_weekly_recaps(supabase, club_id="club", page=1, page_size=3)
    second = build_public_weekly_recaps(supabase, club_id="club", page=2, page_size=3)
    deep_link = build_public_weekly_recaps(
        supabase,
        club_id="club",
        week_start="2026-03-01",
        page=1,
        page_size=3,
    )

    assert [row["week_start"] for row in first["recaps"]] == ["2026-03-07", "2026-03-06", "2026-03-05"]
    assert first["pagination"] == {"page": 1, "page_size": 3, "has_previous": False, "has_next": True}
    assert [row["week_start"] for row in second["recaps"]] == ["2026-03-04", "2026-03-03", "2026-03-02"]
    assert second["pagination"] == {"page": 2, "page_size": 3, "has_previous": True, "has_next": True}
    assert deep_link["selected_recap"]["week_start"] == "2026-03-01"
    assert "internal_notes" not in deep_link["selected_recap"]["recap"]
    assert all(row["week_start"] != "2026-04-01" for row in first["recaps"] + second["recaps"])
