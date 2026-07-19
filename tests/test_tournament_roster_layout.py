from pathlib import Path


def test_tournament_roster_layout_uses_two_job_tabs_and_clean_metrics():
    source = Path("jupr_app/ui/pages/tournament_roster.py").read_text(encoding="utf-8")

    assert '"Roster", "Looking for Partners"' in source
    assert '"Registered entries"' in source
    assert '"Looking for partners"' in source
    assert '"Public players"' not in source


def test_tournament_roster_uses_compact_text_rows_for_mobile():
    source = Path("jupr_app/ui/pages/tournament_roster.py").read_text(encoding="utf-8")

    assert "st.dataframe" not in source
    assert "_compact_roster_line" in source
    assert "_compact_partner_line" in source
    assert "[request partner]" in source


def test_tournament_roster_links_partner_requests_through_existing_partner_route():
    source = Path("jupr_app/ui/pages/tournament_roster.py").read_text(encoding="utf-8")

    assert "request partner" in source
    assert 'page="tournament_partner_board"' in source
    assert '"target_selection_id"' in source
    assert "_request_player_name" in source
