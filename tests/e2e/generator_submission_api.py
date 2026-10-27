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
os.environ["JUPR_PUBLIC_LIVE_CREATE_LIMIT_PER_HOUR"] = "30"  # Many isolated browser scenarios share one fixture visitor.
os.environ["JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR"] = "1"

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from test_generator_submissions import db as fixture_db, Query as FixtureQuery
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from jupr_app.services.public_play_generator_service import get_public_play_generator_session
from services.api import admin_play_generator_routes as admin
from services.api import admin_player_editor_routes as editor
from services.api.club_site_models import default_site
from services.api.public_play_generator_routes import install_public_play_generator_routes
from services.api.shared_play_generator_routes import install_shared_play_generator_routes

patches = pytest.MonkeyPatch()
database = fixture_db.__wrapped__(patches)
database.db["clubs"] = [{"id": "club", "slug": "test"}]


class CreationQuery(FixtureQuery):
    """Mirror database-generated IDs for real player-creation service calls."""
    def execute(self):
        if self.name == "live_sessions" and self.payload is not None:
            for row in self.payload if isinstance(self.payload, list) else [self.payload]:
                row.setdefault("version", 1)
        if self.name in {"players", "admin_guarded_operations"} and self.payload is not None:
            rows = self.payload if isinstance(self.payload, list) else [self.payload]
            next_id = max((int(row.get("id") or 0) for row in self.db.get(self.name, [])), default=0) + 1
            for row in rows:
                row.setdefault("id", next_id)
                next_id += 1
        return super().execute()


database.table = lambda name: CreationQuery(database.db, name)
_original_generator_rpc = database.rpc


def receipt_rpc(name, params):
    if name in {"commit_admin_play_generator_session", "commit_public_play_generator_session"}:
        from generator_receipt_fake import commit_generator_rpc
        return SimpleNamespace(execute=lambda: commit_generator_rpc(database.db, name, params))
    return _original_generator_rpc(name, params)


database.rpc = receipt_rpc
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
install_shared_play_generator_routes(app, get_club=lambda slug: {"id": "club"},
    get_supabase_client=lambda: database, require_public_writes=lambda: None,
    require_service_role=lambda: None, require_admin_writes=lambda club_id: None,
    requester_hash=lambda request: "a" * 64, raise_public_error=raise_error)


def authenticate(authorization):
    if authorization != "Bearer local-reviewer":
        raise HTTPException(401, "Local test session required")
    return SimpleNamespace(email="reviewer@example.invalid", user_id="local-reviewer")


patches.setattr(admin, "_require_write_gate", lambda *args: None)
patches.setattr(admin, "authenticate_bearer", authenticate)
patches.setattr(admin, "resolve_admin_role", lambda **kw: SimpleNamespace(role="administrator"))
admin.install_admin_play_generator_routes(app, get_supabase_client=lambda: database)
patches.setattr(editor, "authenticate_bearer", authenticate)
patches.setattr(editor, "resolve_admin_role", lambda **kw: SimpleNamespace(role="administrator"))
editor.install_admin_player_editor_routes(app, get_supabase_client=lambda: database)


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
    sessions = [(admin.get_play_generator_session if row["state"].get("mode") == "admin_play_generator" else get_public_play_generator_session)(database, club_id="club", session_key=row["session_key"])["session"]
                for row in database.db["live_sessions"]]
    return {"matches": database.db["matches"], "players": database.db["players"], "sessions": sessions}
