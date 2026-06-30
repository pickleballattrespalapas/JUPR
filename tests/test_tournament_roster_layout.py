from pathlib import Path


def test_tournament_roster_layout_uses_two_job_tabs_and_clean_metrics():
    source = Path("jupr_app/ui/pages/tournament_roster.py").read_text(encoding="utf-8")

    assert '"Roster", "Looking for Partners"' in source
    assert '"Registered entries"' in source
    assert '"Looking for partners"' in source
    assert '"Public players"' not in source


def test_tournament_roster_uses_tables_instead_of_bullet_lists():
    source = Path("jupr_app/ui/pages/tournament_roster.py").read_text(encoding="utf-8")

    assert "st.dataframe" in source
    assert "- **" not in source
    assert "Status" in source


def test_tournament_roster_links_partner_requests_through_existing_partner_route():
    source = Path("jupr_app/ui/pages/tournament_roster.py").read_text(encoding="utf-8")

    assert "Request {player_name} as partner" in source
    assert 'page="tournament_partner_board"' in source
    assert '"target_selection_id"' in source
    assert "requested player's email stays hidden" in source
