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
