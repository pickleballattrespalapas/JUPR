from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_next_admin_auth_requires_fastapi_capabilities_and_safe_redirects():
    client = _source("apps/web/lib/adminAuthClient.ts")
    login = _source("apps/web/app/admin/login/AdminLoginForm.tsx")

    assert "/admin/auth/capabilities" in client
    assert "authorizeAndSaveAdminSession" in login
    assert "safeAdminNextPath" in login
    assert 'requested.startsWith("//")' in client
    assert 'requested.includes("\\\\")' in client
    assert "authLoop" in client
    assert "Sign-in failed. Check your email and password" in client
    assert "parseAuthError" not in client
    assert "SUPABASE_SERVICE_ROLE_KEY" not in client


def test_recovery_is_pkce_scoped_policy_checked_and_cleaned_up():
    client = _source("apps/web/lib/adminAuthClient.ts")
    reset = _source("apps/web/app/admin/reset-password/AdminResetPasswordForm.tsx")
    streamlit_reset = _source("jupr_app/ui/pages/reset_password.py")

    assert 'code_challenge_method: "s256"' in client
    assert "grant_type=pkce" in client
    assert "auth_code: code" in client
    assert "!session.recovery" in client
    assert "scope=local" in client
    assert "clearRecoveryArtifacts" in client
    assert "If this is an eligible admin account" in reset
    assert "Resend recovery email" in reset
    assert "ADMIN_PASSWORD_MIN_LENGTH = 8" in client
    assert "_MIN_PASSWORD_LENGTH = 8" in streamlit_reset


def test_streamlit_admin_auth_fallback_remains_present():
    assert (ROOT / "jupr_app/ui/pages/admin_login.py").is_file()
    assert (ROOT / "jupr_app/ui/pages/reset_password.py").is_file()
    design = _source("docs/next_admin_auth_design.md")
    assert "Streamlit login/reset remains the fallback" in design
