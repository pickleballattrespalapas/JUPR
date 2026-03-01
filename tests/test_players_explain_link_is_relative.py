from pathlib import Path


def test_players_page_uses_linkcolumn_and_no_localhost_explain_links():
    content = Path("jupr_app/ui/pages/players.py").read_text(encoding="utf-8")

    assert "LinkColumn" in content
    assert "localhost:8501" not in content
