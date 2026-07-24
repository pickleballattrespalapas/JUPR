from pathlib import Path


PANEL = Path(
    "apps/web/app/admin/support-requests/SupportRequestsPanel.tsx"
).read_text(encoding="utf-8")


def test_support_filters_auto_load_and_cannot_race_a_status_write() -> None:
    assert "useAuthenticatedAutoLoad(" in PANEL
    assert "`${statusFilter}:${typeFilter}`" in PANEL
    filters = PANEL.split("<h2 style={{ marginTop: 0 }}>Filters</h2>", 1)[1].split(
        "{summary ?", 1
    )[0]
    assert filters.count("disabled={busy}") == 2
    assert "disabled={busy || !accessToken}" in filters


def test_support_selection_and_editor_are_disabled_while_writing() -> None:
    assert "disabled: boolean" in PANEL
    assert "selected={request.id === selectedId} disabled={busy}" in PANEL
    assert "disabled={busy}" in PANEL
    assert "if (!requestsRequest.isCurrent(generation)) return;" in PANEL
