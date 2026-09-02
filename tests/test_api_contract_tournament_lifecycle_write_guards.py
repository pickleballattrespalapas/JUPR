from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def async_body(source: str, function_name: str) -> str:
    marker = f"async function {function_name}"
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated async function: {function_name}")


def test_dedicated_registration_editor_scopes_each_write_to_current_token() -> None:
    panel = read(
        "app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx"
    )

    assert "const actionRequest = useLatestRequestGuard(accessToken);" in panel
    for function_name in ("saveRegistration", "saveSelection"):
        body = async_body(panel, function_name)
        assert "const generation = actionRequest.begin();" in body
        assert "actionRequest.isCurrent(generation)" in body


def test_tournament_home_is_read_only_and_load_remains_token_scoped() -> None:
    panel = read("app/admin/tournaments/tournament/TournamentHomePanel.tsx")
    body = async_body(panel, "loadDetail")

    assert "const detailRequest = useLatestRequestGuard(" in panel
    assert "const generation = detailRequest.begin();" in body
    assert "detailRequest.isCurrent(generation)" in body
    assert "saveTournament" not in panel
    for method in ('method: "PATCH"', 'method: "POST"', 'method: "DELETE"'):
        assert method not in panel
