from pathlib import Path


GUIDE_PATH = Path("apps/web/app/admin/guide/page.tsx")
MATRIX_PATH = Path("docs/next_streamlit_parity_matrix.md")


def test_next_admin_guide_has_route_linked_day_of_runbooks() -> None:
    contents = GUIDE_PATH.read_text(encoding="utf-8")

    assert "Day-of operations runbooks" in contents
    assert "League night" in contents
    assert "Quick, paper-sheet, or pop-up scoring" in contents
    assert "Match correction and replay" in contents
    assert "Player maintenance or merge" in contents
    assert "Challenge Ladder administration" in contents
    assert "Tournament setup through publish" in contents
    assert "Player communications" in contents
    for route in (
        "/admin/league-manager",
        "/admin/match-uploader",
        "/admin/match-log",
        "/admin/players",
        "/admin/challenge-ladder",
        "/admin/tournaments",
        "/admin/player-updates",
    ):
        assert f'href: "{route}"' in contents


def test_next_admin_guide_has_stop_and_recovery_contracts() -> None:
    contents = GUIDE_PATH.read_text(encoding="utf-8")

    assert "Complete when" in contents
    assert "Stop when" in contents
    assert "Global stop conditions" in contents
    assert "Recovery sequence" in contents
    assert "Review Admin Tools activity" in contents
    assert "Run Replay History using the smallest safe scope" in contents
    assert "do not patch ratings or snapshots directly" in contents
    assert "do not copy staging-only data into production" in contents


def test_admin_guide_parity_row_no_longer_claims_guide_is_unported() -> None:
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    row = next(line for line in matrix.splitlines() if "`admin_guide`" in line)

    assert "full guide not ported" not in row
    assert "route-linked day-of runbooks" in row
    assert "recovery sequencing" in row
