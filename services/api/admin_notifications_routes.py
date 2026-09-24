"""Verified, club-bound access to personal notification preferences and state."""
from typing import Annotated, Literal

from fastapi import HTTPException, Path, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from jupr_app.services.admin_notifications_service import (
    BULK_CLEAR_LIMIT, NotificationConflict, NotificationUnavailable, clear_admin_notifications, get_admin_notifications,
    update_admin_notification_preferences, update_admin_notification_state,
)
from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import auth_header


class NotificationPreferencesUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    categories: dict[str, StrictBool] = Field(max_length=64)


class NotificationStateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["new", "flagged", "cleared"]


class NotificationBulkClear(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keys: list[Annotated[str, Field(strict=True, pattern=r"^[a-f0-9]{64}$")]] = Field(min_length=1, max_length=BULK_CLEAR_LIMIT)


def install_admin_notifications_routes(app, *, get_supabase_client):
    def context(club_id, authorization, response):
        db = get_supabase_client()
        user, assignments = require_admin_assignments(get_supabase_client=lambda: db,
            authorization=authorization, requested_club_id=club_id)
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Vary"] = "Authorization"
        return db, {"club_id": club_id, "user_id": user.user_id, "assignments": assignments}

    def call(function, db, **kwargs):
        try:
            return function(db, **kwargs)
        except NotificationConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except NotificationUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, "This notification change is invalid. Refresh and try again.") from exc

    @app.get("/admin/clubs/{club_id}/notifications")
    def get_notifications(response: Response, club_id: str = Path(pattern=r"^[a-zA-Z0-9_-]{1,100}$"), authorization: str | None = auth_header()):
        db, args = context(club_id, authorization, response)
        return call(get_admin_notifications, db, **args)

    @app.put("/admin/clubs/{club_id}/notifications/preferences")
    def put_preferences(body: NotificationPreferencesUpdate, response: Response, club_id: str = Path(pattern=r"^[a-zA-Z0-9_-]{1,100}$"), authorization: str | None = auth_header()):
        db, args = context(club_id, authorization, response)
        return call(update_admin_notification_preferences, db, **args, categories=body.categories)

    @app.put("/admin/clubs/{club_id}/notifications/items/{key}")
    def put_item(body: NotificationStateUpdate, response: Response, club_id: str = Path(pattern=r"^[a-zA-Z0-9_-]{1,100}$"), key: str = Path(pattern=r"^[a-f0-9]{64}$"), authorization: str | None = auth_header()):
        db, args = context(club_id, authorization, response)
        return call(update_admin_notification_state, db, **args, key=key, state=body.state)

    @app.put("/admin/clubs/{club_id}/notifications/bulk-clear")
    def put_bulk_clear(body: NotificationBulkClear, response: Response, club_id: str = Path(pattern=r"^[a-zA-Z0-9_-]{1,100}$"), authorization: str | None = auth_header()):
        db, args = context(club_id, authorization, response)
        return call(clear_admin_notifications, db, **args, keys=body.keys)
