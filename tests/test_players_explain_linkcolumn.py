from pathlib import Path


PLAYERS_PAGE = Path("jupr_app/ui/pages/players.py")


def test_players_page_uses_linkcolumn_for_explain() -> None:
    contents = PLAYERS_PAGE.read_text(encoding="utf-8")
    assert "LinkColumn" in contents
    assert '"EXPLAIN": st.column_config.LinkColumn("Explain", display_text="Explain")' in contents


def test_players_page_does_not_use_markdown_explain_links() -> None:
    contents = PLAYERS_PAGE.read_text(encoding="utf-8")
    assert "[Explain](" not in contents
