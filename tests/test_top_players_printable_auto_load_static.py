from pathlib import Path


PANEL = Path(
    "apps/web/app/admin/top-players-printable/TopPlayersPrintablePanel.tsx"
).read_text(encoding="utf-8")


def test_rankings_refresh_clears_stale_print_payload_and_disables_print() -> None:
    load = PANEL.split("async function loadRankings()", 1)[1].split(
        "useAuthenticatedAutoLoad", 1
    )[0]
    assert "setBusy(true);" in load
    assert "setPayload(null);" in load
    assert load.index("setPayload(null);") < load.index("await fetch(")
    assert "disabled={busy || !payload}" in PANEL


def test_rankings_auto_load_waits_for_auth_and_is_token_scoped() -> None:
    assert "useAuthenticatedAutoLoad(status.enabled ? accessToken : \"\", loadRankings)" in PANEL
    assert "useLatestRequestGuard(accessToken" in PANEL
    assert "if (!rankingsRequest.isCurrent(generation)) return;" in PANEL
    assert "Refresh rankings" in PANEL
