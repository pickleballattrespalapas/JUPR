from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_STAGING_SPEC = ROOT / "apps" / "web" / "e2e" / "admin-auth.real.staging.spec.ts"


def test_real_staging_admin_auth_uses_minted_session_without_password_dependency() -> None:
    source = REAL_STAGING_SPEC.read_text(encoding="utf-8")

    assert "STAGING_ADMIN_BEARER_TOKEN" in source
    assert "STAGING_ADMIN_EMAIL" in source
    assert "STAGING_ADMIN_PASSWORD" not in source
    assert '"jupr_admin_session_v1"' in source
    assert "context.addInitScript" in source
    assert "window.location.origin !== allowedOrigin" in source
    assert "allowedOrigin: expectedWebOrigin" in source


def test_real_staging_admin_auth_uses_live_capabilities_and_real_sign_out() -> None:
    source = REAL_STAGING_SPEC.read_text(encoding="utf-8")

    assert "page.route(" not in source
    assert "url.origin === expectedApiOrigin" in source
    assert 'url.pathname === "/admin/auth/capabilities"' in source
    assert "liveCapabilities?.authorized" in source
    assert "url.origin === expectedAuthOrigin" in source
    assert 'url.pathname === "/auth/v1/logout"' in source
    assert 'page.getByRole("button", { name: "Sign out" }).click()' in source
    assert "localStorage.getItem(\"jupr_admin_session_v1\")" in source


def test_real_staging_admin_auth_does_not_log_or_snapshot_the_bearer_token() -> None:
    source = REAL_STAGING_SPEC.read_text(encoding="utf-8")

    assert "console." not in source
    assert ".toBe(adminToken)" not in source
    assert ".toEqual(adminToken)" not in source
