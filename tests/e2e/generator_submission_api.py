"""Local-only API for generator-submission-browser.cjs.

The production route installers and services use the existing in-memory database
fixture. No sessions are seeded: the browser must preview, start, score, advance,
finish, submit, and approve each one through the real HTTP routes.
Authentication, database access, and post-commit external side effects are faked.
Run this only on loopback; it intentionally supplies a test administrator.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ["JUPR_ENV"] = "test"
os.environ["JUPR_PUBLIC_LIVE_TOKEN_SECRET"] = "local-browser-fixture-secret-00000000000000000000"

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from test_generator_submissions import db as fixture_db
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from jupr_app.services.public_play_generator_service import get_public_play_generator_session
from services.api import admin_play_generator_routes as admin
from services.api.club_site_models import default_site
from services.api.public_play_generator_routes import install_public_play_generator_routes

patches = pytest.MonkeyPatch()
database = fixture_db.__wrapped__(patches)
app = FastAPI()
web_port = int(os.environ.get("GENERATOR_BROWSER_WEB_PORT", "3038"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{web_port}", f"http://127.0.0.1:{web_port}"],
    allow_methods=["*"], allow_headers=["*"],
)


def raise_error(exc):
    status = 403 if isinstance(exc, PermissionError) else 409 if isinstance(exc, PublicLiveConflictError) else 400
    raise HTTPException(status, str(exc))


install_public_play_generator_routes(
    app,
    get_club=lambda slug: {"id": "club", "name": "Test Pickleball Club"},
    get_supabase_client=lambda: database,
    public_club_payload=lambda club, slug: club,
    require_public_writes=lambda: None,
    require_service_role=lambda: None,
    requester_hash=lambda request: "a" * 64,
    raise_public_error=raise_error,
    public_writes_enabled=lambda: True,
    service_role_configured=lambda: True,
)


def authenticate(authorization):
    if authorization != "Bearer local-reviewer":
        raise HTTPException(401, "Local test session required")
    return SimpleNamespace(email="reviewer@example.invalid", user_id="local-reviewer")


patches.setattr(admin, "_require_write_gate", lambda: None)
patches.setattr(admin, "authenticate_bearer", authenticate)
patches.setattr(admin, "resolve_admin_role", lambda **kw: SimpleNamespace(role="administrator"))
admin.install_admin_play_generator_routes(app, get_supabase_client=lambda: database)


@app.get("/public/clubs/{slug}/site")
def site(slug: str):
    return {"club_id": "club", "slug": slug,
            "document": default_site({"name": "Test Pickleball Club"}), "published_at": "2026-09-20"}


@app.get("/admin/auth/capabilities")
def capabilities():
    return {"authorized": True, "user": {"email": "reviewer@example.invalid"},
            "assignments": [{"club_id": "club", "role": "administrator", "permissions": ["enter_scores", "manage_players"]}]}


@app.get("/admin/auth/workspaces")
def workspaces():
    return {"workspaces": [{"club_id": "club", "club_slug": "test", "club_name": "Test Pickleball Club", "roles": ["administrator"]}]}


@app.get("/clubs/{slug}/players")
def players(slug: str):
    return {"players": database.db["players"], "total": len(database.db["players"])}


@app.get("/fixture-result")
def result():
    # Public serializers omit organizer tokens and internal recovery data.
    sessions = [get_public_play_generator_session(database, club_id="club", session_key=row["session_key"])["session"]
                for row in database.db["live_sessions"]]
    return {"matches": database.db["matches"], "players": database.db["players"], "sessions": sessions}
