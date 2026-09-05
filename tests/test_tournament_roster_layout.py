from pathlib import Path

from jupr_app.ui.pages.tournament_roster import _group_partner_rows


def test_tournament_roster_keeps_partner_discovery_on_dedicated_page():
    source = Path("jupr_app/ui/pages/tournament_roster.py").read_text(encoding="utf-8")

    assert "st.tabs" not in source
    assert 'page_key = "tournament_partner_board" if focus_partners else "tournament_roster"' in source
    assert 'state.get("partner_board_entries") or []' in source
    assert 'state.get("players_needing_partners")' not in source
    assert '"Registered entries"' in source
    assert '"Players needing partners"' in source
    assert '"Public players"' not in source


def test_tournament_roster_uses_compact_text_rows_for_mobile():
    source = Path("jupr_app/ui/pages/tournament_roster.py").read_text(encoding="utf-8")

    assert "st.dataframe" not in source
    assert "_compact_roster_line" in source
    assert "_compact_partner_division_line" in source
    assert "st.container(border=True)" in source
    assert "[request partner]" in source


def test_tournament_roster_links_partner_requests_through_existing_partner_route():
    source = Path("jupr_app/ui/pages/tournament_roster.py").read_text(encoding="utf-8")

    assert "request partner" in source
    assert 'page="tournament_partner_board"' in source
    assert '"target_selection_id"' in source
    assert 'row.get("board_entry_key")' in source
    assert "_request_player_name" in source


def test_partner_rows_group_by_opaque_player_key_and_keep_divisions_separate():
    rows = [
        {
            "player_entry_key": "player-a",
            "board_entry_key": "entry-a1",
            "player_name": "Alex Smith",
            "division": "Mixed Doubles 3.5",
            "event_day_label": "Day 2",
        },
        {
            "player_entry_key": "player-a",
            "board_entry_key": "entry-a2",
            "player_name": "Alex Smith",
            "division": "Women's Doubles 3.5",
            "event_day_label": "Day 10",
        },
        {
            "player_entry_key": "player-b",
            "board_entry_key": "entry-b1",
            "player_name": "Alex Smith",
            "division": "Men's Doubles 4.0",
        },
    ]

    groups = _group_partner_rows(rows)

    assert [group["player_entry_key"] for group in groups] == ["player-a", "player-b"]
    assert [row["division"] for row in groups[0]["entries"]] == [
        "Mixed Doubles 3.5",
        "Women's Doubles 3.5",
    ]
    assert [row["event_day_label"] for row in groups[0]["entries"]] == [
        "Day 2",
        "Day 10",
    ]
    assert [row["division"] for row in groups[1]["entries"]] == ["Men's Doubles 4.0"]
