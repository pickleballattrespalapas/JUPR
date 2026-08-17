from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CREATE_PANEL = ROOT / "apps/web/app/admin/league-manager/create/LeagueCreatePanel.tsx"


def test_team_create_ui_forces_doubles_and_hides_singles_option() -> None:
    source = CREATE_PANEL.read_text(encoding="utf-8")

    assert 'const createMatchFormat = leagueType === "Team" ? "doubles" : matchFormat' in source
    assert 'if (nextLeagueType === "Team") setMatchFormat("doubles")' in source
    assert 'disabled={leagueType === "Team"}' in source
    assert 'leagueType === "Individual" ? <option value="singles">Singles</option> : null' in source
    assert 'match_format: createMatchFormat' in source
    assert "Team leagues use Doubles." in source
