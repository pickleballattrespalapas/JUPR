"""Shareable, club-owned meet signup with transactional lineup and substitute queues."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import time
from typing import Literal
from uuid import UUID

from fastapi import HTTPException, Query, Request, Response
from pydantic import AwareDatetime, Field, field_validator, model_validator

from jupr_app.config import get_next_web_base_url
from services.api.auth import auth_header
from services.api.interclub_registration_routes import StrictModel
from services.api.interclub_registration_phase import require_season_phase, registration_state
from services.api.staging_write_guard import require_public_intake_or_403
from services.api.interclub_player_pool_routes import (
    MEMBERS, MemberDetails, _b64, _choices, _date, _one, _private_headers, _public_season,
    _query, _requester, _rows, _season_club, _season_end, _secret, pool_admin_context,
)

SETTINGS = "pcs_interclub_meet_signup_settings"
SIGNUPS = "pcs_interclub_meet_signups"
PURPOSE = "pcs-interclub-meet-signup:v1"
BASE = "/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/signup"
PUBLIC = "/public/interclub-meet-signups"


class MeetJoin(StrictModel):
    player_id: int = Field(gt=0)
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(default="", max_length=254)
    division: str = Field(min_length=1, max_length=20)
    request_id: UUID
    confirm_self: Literal[True]
    website: str = Field(default="", max_length=200)

    @field_validator("email")
    @classmethod
    def email_address(cls, value):
        return MemberDetails.email_address(value) if value else ""


class MeetSignupSettings(StrictModel):
    expected_revision: int = Field(ge=0)
    expected_meet_revision: int = Field(ge=1)
    open: bool
    deadline: AwareDatetime


class MeetSignupAction(StrictModel):
    action: Literal["refresh", "promote", "remove", "add"]
    id: UUID | None = None
    expected_revision: int | None = Field(default=None, ge=1)
    player_id: int | None = Field(default=None, gt=0)
    division: str | None = Field(default=None, min_length=1, max_length=20)
    request_id: UUID | None = None

    @model_validator(mode="after")
    def action_fields(self):
        if self.action in {"promote", "remove"} and (self.id is None or self.expected_revision is None):
            raise ValueError("Choose a signup and its current revision.")
        if self.action == "add" and None in (self.player_id, self.division, self.request_id):
            raise ValueError("Choose a player and division.")
        return self


class MeetSignupReview(StrictModel):
    token: str = Field(min_length=24, max_length=2000)


class MeetSignupWithdraw(MeetSignupReview):
    expected_revision: int = Field(ge=1)


def _rpc(db, action, payload, user=None, requester=None):
    try:
        return db.rpc("pcs_interclub_meet_signup_action", {
            "p_action": action, "p_payload": payload,
            "p_actor_id": user.user_id if user else None,
            "p_actor_email": user.email if user else None, "p_requester_hash": requester,
        }).execute().data
    except Exception as exc:
        code = getattr(exc, "code", "")
        # Only explicitly authored errors may expose their message. Driver and
        # database constraint details can contain private request fields.
        if code in {"PT409", "PT422"}:
            raise HTTPException(int(code[2:]), getattr(exc, "message", "Reload this meet and try again.")) from exc
        status, message = {
            "42501": (403, "Your club administrator must reopen signup to confirm access."),
            "P0002": (404, "This meet signup is unavailable."),
            "40001": (409, "The meet or lineup changed. Reload before continuing."),
            "23505": (409, "This player or request is already registered. Reload before continuing."),
            "22023": (422, "Check the player’s season approval, league rating and division."),
            "PT423": (423, "Meet signups open after season registration closes."),
            "54000": (429, "Too many registrations. Please wait and try again."),
        }.get(code, (503, "Could not confirm the result. Retry with the same details."))
        raise HTTPException(status, message) from exc


def _meet(db, club_id, season_id, meet_id):
    row = _one(db, "pcs_interclub_meets", "id,season_id,host_club_id,club_ids,starts_at,roster_deadline,revision,competition_phase", id=str(meet_id), season_id=str(season_id))
    if not row or club_id not in row["club_ids"]:
        raise HTTPException(404, "This meet is unavailable for your club.")
    host = _one(db, "clubs", "name", id=row["host_club_id"])
    return {**row, "host_club_name": host["name"] if host else row["host_club_id"]}


def _url(share_id):
    return f"{get_next_web_base_url().rstrip('/')}/interclub/meet-signup/{share_id}"


def _private_url(row, season):
    claims = {"purpose": PURPOSE, **{key: row[key] for key in ("id", "season_id", "club_id", "meet_id", "token_nonce")}, "exp": int(_season_end(season).timestamp())}
    encoded = _b64(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
    signature = _b64(hmac.new(_secret(), (PURPOSE + ":" + encoded).encode(), hashlib.sha256).digest())
    return f"{get_next_web_base_url().rstrip('/')}/interclub/meet-signup/manage#token={encoded}.{signature}"


def _claims(token):
    secret = _secret()
    try:
        encoded, signature = token.split(".")
        expected = _b64(hmac.new(secret, (PURPOSE + ":" + encoded).encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError()
        claims = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if claims.get("purpose") != PURPOSE or int(claims["exp"]) <= time.time():
            raise ValueError()
        for key in ("id", "season_id", "meet_id", "token_nonce"):
            UUID(claims[key])
        if not isinstance(claims["club_id"], str) or not 1 <= len(claims["club_id"]) <= 100:
            raise ValueError()
        return claims
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, AttributeError) as exc:
        raise HTTPException(404, "This private signup link is invalid or expired.") from exc


def _entry(row, private=False, season=None):
    keys = ("id", "name", "division", "gender", "rating", "status", "placement", "priority", "reason", "registered_at")
    value = {key: row.get(key) for key in keys}
    if private:
        value.update(email=row["email"], player_id=str(row["player_id"]), revision=row["revision"], manage_url=_private_url(row, season))
    return value


def _board(db, club, season, meet, private=False):
    scope = {"season_id": season["id"], "club_id": club["id"], "meet_id": meet["id"]}
    cfg = _one(db, SETTINGS, **scope)
    stale = bool(cfg and cfg["meet_revision"] != meet["revision"])
    deadline = cfg["deadline"] if cfg else meet["roster_deadline"]
    is_open = bool(cfg and cfg["open"] and not stale and registration_state(season)["meet_planning_open"] and
                   datetime.now(timezone.utc) < min(_date(deadline), _date(meet["roster_deadline"]), _date(meet["starts_at"])))
    rows = _rows(_query(db, SIGNUPS, **scope).order("registration_order").limit(1000))
    entries = [_entry(row, private, season) for row in rows if private or row["status"] == "active"]
    # Separate ordered queues: in-band substitutes always precede play-up requests.
    positions = {}
    for entry in sorted(entries, key=lambda item: item["priority"] != "in_band"):
        if entry["status"] == "active" and entry["placement"] == "waitlist":
            key = (entry["division"], entry["gender"])
            positions[key] = positions.get(key, 0) + 1
            entry["queue_position"] = positions[key]
    return {"club": club, "season": _public_season(season), "meet": meet,
            "signup": {"open": is_open, "configured": bool(cfg), "revision": cfg["revision"] if cfg else 0,
                       "deadline": deadline, "schedule_changed": stale, "url": _url(cfg["share_id"]) if cfg else None}, "entries": entries}


def _public_context(db, share_id):
    cfg = _one(db, SETTINGS, share_id=str(share_id))
    if not cfg:
        raise HTTPException(404, "This meet signup link is unavailable.")
    club, season = _season_club(db, cfg["club_id"], cfg["season_id"])
    return club, season, _meet(db, club["id"], season["id"], cfg["meet_id"])


def _review(db, token):
    claims = _claims(token)
    row = _one(db, SIGNUPS, **{key: claims[key] for key in ("id", "season_id", "club_id", "meet_id", "token_nonce")})
    if not row:
        raise HTTPException(404, "This private signup link is no longer available.")
    club, season = _season_club(db, row["club_id"], row["season_id"])
    board = _board(db, club, season, _meet(db, club["id"], season["id"], row["meet_id"]))
    own = _entry(row, True, season)
    own.update(next(({"queue_position": entry.get("queue_position")} for entry in board["entries"] if entry["id"] == row["id"]), {}))
    return {**board, "entry": own}


def _fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def install_interclub_meet_signup_routes(app, *, get_supabase_client):
    @app.middleware("http")
    async def private_signup_responses(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(PUBLIC) or "/meets/" in request.url.path and request.url.path.endswith(("/signup", "/signup/actions")):
            _private_headers(response)
        return response

    def admin_context(club_id, season_id, meet_id, authorization):
        db, user, club, season = pool_admin_context(get_supabase_client, authorization, club_id, season_id)
        require_season_phase(season)
        return db, user, club, season, _meet(db, club_id, season_id, meet_id)

    @app.get(BASE)
    def admin_board(club_id: str, season_id: UUID, meet_id: UUID, authorization: str | None = auth_header()):
        db, _, club, season, meet = admin_context(club_id, season_id, meet_id, authorization)
        return _board(db, club, season, meet, True)

    @app.put("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/signup")
    def settings(club_id: str, season_id: UUID, meet_id: UUID, body: MeetSignupSettings, authorization: str | None = auth_header()):
        db, user, club, season, meet = admin_context(club_id, season_id, meet_id, authorization)
        _secret()  # Fail before opening a page whose private links cannot be signed.
        _rpc(db, "settings", {**body.model_dump(mode="json"), "club_id": club_id, "season_id": str(season_id), "meet_id": str(meet_id)}, user)
        return _board(db, club, season, meet, True)

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/signup/actions")
    def action(club_id: str, season_id: UUID, meet_id: UUID, body: MeetSignupAction, authorization: str | None = auth_header()):
        db, user, club, season, meet = admin_context(club_id, season_id, meet_id, authorization)
        payload = {**body.model_dump(mode="json", exclude_none=True), "club_id": club_id, "season_id": str(season_id), "meet_id": str(meet_id)}
        payload["fingerprint"] = _fingerprint(payload)
        _rpc(db, body.action, payload, user)
        return _board(db, club, season, meet, True)

    # Static private routes must precede the dynamic share-id routes.
    @app.post("/public/interclub-meet-signups/review")
    def review(body: MeetSignupReview):
        return _review(get_supabase_client(), body.token)

    @app.post("/public/interclub-meet-signups/withdraw")
    def withdraw(body: MeetSignupWithdraw):
        require_public_intake_or_403()
        db = get_supabase_client()
        claims = _claims(body.token)
        _rpc(db, "withdraw", {"id": claims["id"], "nonce": claims["token_nonce"], "expected_revision": body.expected_revision})
        return _review(db, body.token)

    @app.get(PUBLIC + "/{share_id}")
    def public_board(share_id: UUID):
        db = get_supabase_client()
        return _board(db, *_public_context(db, share_id))

    @app.get(PUBLIC + "/{share_id}/players")
    def find_players(share_id: UUID, q: str = Query(min_length=2, max_length=120)):
        db = get_supabase_client()
        club, season, meet = _public_context(db, share_id)
        if not _board(db, club, season, meet)["signup"]["open"]:
            raise HTTPException(409, "Meet signup is closed.")
        clean = q.strip().replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        members = _rows(_query(db, MEMBERS, "player_id", season_id=season["id"], club_id=club["id"], status="active", approval_status="approved").ilike("name", f"%{clean}%").limit(25))
        ids = [row["player_id"] for row in members if row.get("player_id")]
        players = _rows(_query(db, "players", "id,name,rating,gender", club_id=club["id"], active=True).in_("id", ids).order("name")) if ids else []
        return {"players": _choices(db, club["id"], season, players)}

    @app.post("/public/interclub-meet-signups/{share_id}")
    def join(share_id: UUID, body: MeetJoin, request: Request):
        require_public_intake_or_403()
        if body.website:
            raise HTTPException(422, "Please leave the website field blank.")
        _secret()
        db = get_supabase_client()
        club, season, meet = _public_context(db, share_id)
        payload = {**body.model_dump(mode="json", exclude={"website"}), "share_id": str(share_id)}
        payload["fingerprint"] = _fingerprint(payload)
        result = _rpc(db, "join", payload, requester=_requester(request))
        if result.get("duplicate"):
            return {"duplicate": True, "message": "You already have a signup for this meet. Use your saved private link, or ask your club to update it."}
        return _review(db, _private_url(result["entry"], season).split("#token=", 1)[1])
