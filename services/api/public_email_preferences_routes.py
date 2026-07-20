from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel

from jupr_app.services.public_email_preferences_service import (
    apply_public_email_unsubscribe,
    build_public_email_preferences,
)
from services.api.staging_write_guard import require_public_intake_or_403


class PublicEmailUnsubscribeRequest(BaseModel):
    token: str | None = None
    ut: str | None = None
    sid: str | None = None
    subscription_id: str | None = None
    scope: str = "player_updates"


def install_public_email_preferences_routes(app, *, get_supabase_client) -> None:
    """Register public email preference/unsubscribe routes."""

    @app.get("/email-preferences")
    def get_public_email_preferences(
        token: str | None = Query(default=None),
        ut: str | None = Query(default=None),
        sid: str | None = Query(default=None),
        subscription_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        supabase = get_supabase_client()
        try:
            return build_public_email_preferences(
                supabase,
                token=token,
                ut=ut,
                sid=sid,
                subscription_id=subscription_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/email-preferences/unsubscribe")
    def post_public_email_unsubscribe(payload: PublicEmailUnsubscribeRequest) -> dict[str, Any]:
        require_public_intake_or_403()
        supabase = get_supabase_client()
        try:
            return apply_public_email_unsubscribe(
                supabase,
                token=payload.token,
                ut=payload.ut,
                sid=payload.sid,
                subscription_id=payload.subscription_id,
                scope=payload.scope,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
