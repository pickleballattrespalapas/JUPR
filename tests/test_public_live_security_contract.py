from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_live_migration_hashes_legacy_tokens_and_locks_private_tables() -> None:
    sql = _read("supabase/migrations/20260719220000_public_live_durability.sql").lower()

    assert "digest(coalesce(state #>> '{private,edit_token}', ''), 'sha256')" in sql
    assert "state = state #- '{private,edit_token}'" in sql
    assert "alter table public.live_sessions force row level security" in sql
    assert "alter table public.public_live_operations force row level security" in sql
    assert "revoke all on table public.live_sessions from public, anon, authenticated" in sql
    assert "revoke all on table public.public_live_operations from public, anon, authenticated" in sql
    assert "grant select, insert, update, delete on table public.live_sessions to service_role" in sql
    assert "public_live_operation_limits_before_insert" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "public live rate limit exceeded" in sql
    assert "public_live_operations_club_rate_idx" in sql
    assert "guard_live_session_version_and_completion_before_write" in sql
    assert "live session completion is reserved" in sql
    assert "claim_public_live_completion_executor" in sql
    assert "lease_expires_at" in sql


def test_public_live_edit_credentials_stay_fragment_and_session_only() -> None:
    page = _read("apps/web/app/clubs/[clubSlug]/live/[sessionKey]/page.tsx")
    runner = _read("apps/web/app/clubs/[clubSlug]/live/[sessionKey]/LiveSessionRunner.tsx")
    creator = _read("apps/web/app/clubs/[clubSlug]/live/PublicLiveCreator.tsx")

    assert "searchParams" not in page
    assert 'window.location.hash' in runner
    assert 'sessionStorage.setItem(storageKey, discovered)' in runner
    assert 'query.get("edit")' not in runner
    assert '#edit=${encodeURIComponent(editToken)}' in runner
    assert '#edit=${encodeURIComponent(editToken)}' in creator
    assert "jupr-live-operation:" in runner
    assert "jupr-live-create-operation:" in creator
    assert "JSON.stringify(pending)" in runner
    assert "JSON.stringify(requestPayload)" in creator
    assert "exact preserved payload" in runner
    assert "exact preserved request" in creator


def test_public_live_old_league_rounds_are_read_only_and_writes_are_staging_gated() -> None:
    runner = _read("apps/web/app/clubs/[clubSlug]/live/[sessionKey]/LiveSessionRunner.tsx")
    api = _read("services/api/main.py")
    staging = _read("fly.staging.toml")
    production = _read("fly.toml")

    assert "match.round_number === (session.current_round || 1)" in runner
    assert "editableMatches.map" in runner
    assert "JUPR_ENABLE_PUBLIC_LIVE_WRITES" in api
    assert "JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION" in api
    assert 'JUPR_ENABLE_PUBLIC_LIVE_WRITES = "1"' in staging
    assert 'JUPR_ENABLE_PUBLIC_LIVE_WRITES = "0"' in production
    assert 'JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION = "0"' in production
    assert 'address_scope = f"{fly_address}\\x1f{visitor_address}"' in api


def test_public_live_completion_uses_recoverable_reservation_and_safe_exports() -> None:
    service = _read("jupr_app/services/public_live_write_service.py")

    assert '"pending_operation_key": operation_key' in service
    assert '"pending_operation_action": "complete"' in service
    assert '"pending_operation_key": None' in service
    assert "This live session has an unfinished completion" in service
    assert "_formula_safe" in service
    assert "edit_token_hash" in service
    row_payload = service.split("row_payload = {", 1)[1].split("update_public_live_operation(", 1)[0]
    assert '"edit_token_hash": hash_edit_token(edit_token)' in row_payload
    assert '"edit_token":' not in row_payload
