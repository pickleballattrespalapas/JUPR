from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOURNAMENTS_PAGE = REPO_ROOT / "jupr_app" / "ui" / "pages" / "tournaments.py"


def test_no_raw_tournament_games_insert() -> None:
    text = TOURNAMENTS_PAGE.read_text(encoding="utf-8")
    assert 'table("tournament_games").insert(' not in text
    assert "table('tournament_games').insert(" not in text


def test_tournament_games_upsert_uses_on_conflict() -> None:
    text = TOURNAMENTS_PAGE.read_text(encoding="utf-8")
    segments = text.split('table("tournament_games").upsert(')
    for segment in segments[1:]:
        window = segment[:240]
        assert "on_conflict=" in window
