from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_rating_rules_and_faq_are_cross_linked_and_regression_guarded():
    rules = _read("apps/web/app/how-ratings-work/page.tsx")
    faq = _read("apps/web/app/faq/page.tsx")

    for expected in (
        "How ratings move",
        "What counts as rated",
        "What stays unrated",
        "Corrections, exclusions, and replay",
        "How badges are awarded",
        "Why JUPR can differ",
        'href="/faq"',
    ):
        assert expected in rules
    assert 'href="/how-ratings-work"' in faq
    assert "Can my rating go up after a loss?" in faq
    assert "Can my rating go down after a win?" in faq


def test_policy_pages_expose_direct_request_and_preference_routes():
    privacy = _read("apps/web/app/privacy/page.tsx")
    terms = _read("apps/web/app/terms/page.tsx")

    for href in ("/data-corrections", "/profile-privacy", "/email-preferences", "/support#general-support-form"):
        assert href in privacy
    for section in ("Use of the service", "Ratings, standings, and records", "Tournament registration", "Corrections and disputes", "Availability and changes"):
        assert section in terms
    assert "SERVICE_OPERATOR" in privacy
    assert "SERVICE_LOCATION" in privacy


def test_general_support_is_a_durable_intake_not_just_a_mail_link():
    page = _read("apps/web/app/support/page.tsx")
    form = _read("apps/web/app/support/SupportRequestForm.tsx")

    assert "SupportRequestForm" in page
    assert 'id="general-support-form"' in page
    assert 'request_type: "general_support"' in form
    assert "submitPublicSupportRequest" in form
    assert "consent_to_contact" in form
    assert "website:" in form
    assert "supabase" not in form.lower()


def test_support_guardrail_migration_is_service_role_only_and_tracks_privacy_fulfillment():
    migration = _read("supabase/migrations/20260719171000_public_support_intake_guardrails.sql").lower()

    for column in ("request_fingerprint", "request_dedupe_key", "identity_status", "fulfillment_status", "resolution_action", "resolution_evidence"):
        assert column in migration
    assert "revoke all on table public.public_support_requests from public, anon, authenticated" in migration
    assert "grant all privileges on table public.public_support_requests to service_role" in migration
    assert "enable row level security" in migration


def test_admin_support_request_status_is_never_served_from_a_stale_next_cache():
    page = _read("apps/web/app/admin/support-requests/page.tsx")
    api = _read("apps/web/lib/adminSupportRequestsApi.ts")

    assert 'export const dynamic = "force-dynamic"' in page
    assert 'cache: "no-store"' in api
    assert "revalidate" not in api


def test_admin_support_queue_auto_loads_and_keeps_notes_optional():
    panel = _read("apps/web/app/admin/support-requests/SupportRequestsPanel.tsx")

    assert 'import { useAuthenticatedAutoLoad } from "@/lib/useAuthenticatedAutoLoad";' in panel
    assert 'useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadRequests);' in panel
    assert "Refresh requests" in panel
    assert ">Load requests<" not in panel
    assert "Admin note (optional)" in panel
    assert 'admin_note: edit.adminNote.trim() || null' in panel
