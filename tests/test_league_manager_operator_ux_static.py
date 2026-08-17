from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_roster_rating_input_is_an_add_player_value_not_a_filter() -> None:
    source = _read("apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx")

    assert "Starting JUPR or Elo for newly added players" in source
    assert "this is not a roster filter" in source
    assert 'rosterMutable && action === "activate"' in source
    assert "This closed league roster is available for review only." in source


def test_awards_and_live_panels_do_not_expose_deployment_instructions() -> None:
    awards = _read("apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx")
    live = _read("apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx")

    assert "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE" not in awards
    assert "server-only service-role key" not in awards
    assert "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN" not in live
    assert "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT" not in live
    assert "Award changes are unavailable in this build." in awards
    assert "League Live is not available yet" in live


def test_empty_award_evidence_cannot_advance_the_workflow() -> None:
    source = _read("apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx")

    assert "const hasMeasurableResults" in source
    assert "disabled={!writeReady || !hasMeasurableResults}" in source
    assert "disabled={busy || !writeReady || !hasMeasurableResults}" in source
