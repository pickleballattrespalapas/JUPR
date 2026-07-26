from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "apps" / "web"
LEGACY_PAGE = WEB_ROOT / "app" / "admin" / "clubs" / "[clubId]" / "score-entry" / "page.tsx"
LEGACY_PROXY = (
    WEB_ROOT
    / "app"
    / "api"
    / "admin"
    / "clubs"
    / "[clubId]"
    / "matches"
    / "batch"
    / "route.ts"
)
CANONICAL_FORM = (
    WEB_ROOT
    / "app"
    / "clubs"
    / "[clubSlug]"
    / "admin"
    / "score-entry"
    / "ScoreEntryForm.tsx"
)
CANONICAL_PAGE = CANONICAL_FORM.with_name("page.tsx")
SCORE_ENTRY_FLAG = WEB_ROOT / "lib" / "scoreEntry.ts"
DIRECT_MATCH_IDEMPOTENCY = WEB_ROOT / "lib" / "directMatchIdempotency.ts"


def _web_source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in WEB_ROOT.rglob("*")
        if path.is_file()
        and ".next" not in path.parts
        and "node_modules" not in path.parts
        and path.suffix in {".ts", ".tsx", ".js", ".mjs"}
    )


def test_legacy_score_entry_route_only_resolves_to_auth_aware_route() -> None:
    source = LEGACY_PAGE.read_text(encoding="utf-8")

    assert "getClub(params.clubId)" in source
    assert "/admin/score-entry" in source
    assert "redirect(" in source
    assert "fetch(" not in source
    assert "<form" not in source


def test_shared_token_score_entry_proxy_is_removed() -> None:
    assert not LEGACY_PROXY.exists()

    source = _web_source()
    assert "JUPR_ADMIN_API_TOKEN" not in source
    assert "x-admin-token" not in source
    assert "x-admin-permission" not in source


def test_canonical_score_entry_uses_supabase_bearer_session() -> None:
    source = CANONICAL_FORM.read_text(encoding="utf-8")

    assert "useAdminSession" in source
    assert "Authorization: `Bearer ${accessToken}`" in source
    assert "/admin/clubs/${clubId}/matches/batch" in source
    assert "JUPR_ADMIN_API_TOKEN" not in source
    assert source.index("if (!accessToken)") < source.index("await fetch(")


def test_score_entry_ui_is_hidden_unless_browser_flag_is_enabled() -> None:
    helper_source = SCORE_ENTRY_FLAG.read_text(encoding="utf-8")

    assert "NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY" in helper_source
    assert "[\"1\", \"true\", \"yes\", \"on\"]" in helper_source
    for page in (LEGACY_PAGE, CANONICAL_PAGE):
        source = page.read_text(encoding="utf-8")
        assert "isNextAdminScoreEntryEnabled" in source
        assert "Score entry is disabled" in source


def test_score_entry_requires_backend_readiness_and_keeps_recovery_paths_visible() -> None:
    page_source = CANONICAL_PAGE.read_text(encoding="utf-8")
    form_source = CANONICAL_FORM.read_text(encoding="utf-8")

    assert "getAdminScoreEntryStatus" in page_source
    assert "readiness.data?.ready" in page_source
    assert "Score entry is in fallback mode" in page_source
    assert "/admin/match-uploader" in page_source
    assert "Streamlit fallback" in page_source
    assert "directMatchIdempotencyKey" in form_source
    assert "idempotency_key" in form_source
    assert "duplicate protection is active" in form_source
    assert "match_write_committed" not in form_source  # server decides commit state


def test_direct_match_retry_key_survives_blocked_browser_storage() -> None:
    source = DIRECT_MATCH_IDEMPOTENCY.read_text(encoding="utf-8")

    assert "new Map<string, PendingDirectMatchWrite>()" in source
    assert "pendingDirectMatchWrites.get(key)" in source
    assert "sessionStorage" in source
    assert "pendingDirectMatchWrites.delete(key)" in source
