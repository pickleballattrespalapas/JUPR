"""Club-owned season interest pools and personal, account-free meet responses."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import math
import re
import time
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import HTTPException, Query, Request, Response
from pydantic import AwareDatetime, Field, field_validator, model_validator

from jupr_app.config import get_email_mode, get_explicit_registration_edit_token_secret, get_next_web_base_url
from jupr_app.domain.admin.staff_policy import ADMIN_ROLES
from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import authenticate_bearer, auth_header
from services.api.interclub_registration_routes import StrictModel
from jupr_app.services.interclub_registration_phase import REGISTRATION_FIELDS
from services.api.interclub_registration_phase import registration_state, require_season_phase
from services.api.staging_write_guard import require_public_intake_or_403

POOL = "pcs_interclub_pool_settings"
MEMBERS = "pcs_interclub_pool_members"
SETTINGS = "pcs_interclub_availability_settings"
RESPONSES = "pcs_interclub_availability_responses"
MEMBER_FIELDS = "id,season_id,club_id,name,email,divisions,notes,status,player_id,revision,created_at,updated_at,approval_status,late_join,approval_reason"
APPROVAL_FIELDS = "id,club_id,name,player_id,revision,approval_status,late_join,approval_reason"
MEET_FIELDS = "id,season_id,host_club_id,club_ids,starts_at,roster_deadline"
PURPOSE = "pcs-interclub-player:v1"
PLAYER_FIELDS = "id,name,rating,gender"


class PoolSettingsUpdate(StrictModel):
    expected_revision: int = Field(ge=0)
    rotate_link: Literal[True]


class PoolMemberUpdate(StrictModel):
    expected_revision: int = Field(ge=1)
    player_id: int | None = Field(default=None, gt=0)
    status: Literal["active", "withdrawn"]


class PoolApproval(StrictModel):
    member_id: UUID
    expected_revision: int = Field(ge=1)
    approve: bool
    reason: str = Field(min_length=1, max_length=500)


class AvailabilitySettingsUpdate(StrictModel):
    expected_revision: int = Field(ge=0)
    open: bool
    deadline: AwareDatetime


class MemberDetails(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=254)
    divisions: list[str] = Field(default_factory=list, max_length=8)
    notes: str = Field(default="", max_length=1000)

    @field_validator("email")
    @classmethod
    def email_address(cls, value):
        if not re.fullmatch(r"[^\s@,<>]+@[^\s@,<>]+\.[^\s@,<>]+", value):
            raise ValueError("Enter a valid email address.")
        return value.lower()

    @field_validator("divisions")
    @classmethod
    def distinct_divisions(cls, value):
        if len(set(value)) != len(value) or any(not d or len(d) > 20 for d in value):
            raise ValueError("Choose each division once.")
        return value


class PoolSignup(MemberDetails):
    request_id: UUID
    email_consent: Literal[True]
    website: str = Field(default="", max_length=200)
    player_id: int | None = Field(default=None, gt=0)


class BulkPoolMember(StrictModel):
    name: str = Field(default="", max_length=120)
    email: str | None = Field(default=None, max_length=254)
    player_id: int | None = Field(default=None, gt=0)
    divisions: list[str] = Field(default_factory=list, max_length=8)
    notes: str = Field(default="", max_length=1000)

    @field_validator("email")
    @classmethod
    def optional_email(cls, value):
        return MemberDetails.email_address(value) if value else ""

    @field_validator("divisions")
    @classmethod
    def distinct_divisions(cls, value):
        return MemberDetails.distinct_divisions(value)

    @model_validator(mode="after")
    def details(self):
        if self.player_id is None and not self.name.strip():
            raise ValueError("Enter a player name or choose a club player.")
        return self


class BulkPoolMembers(StrictModel):
    members: list[BulkPoolMember] = Field(min_length=1, max_length=200)


class ResponseReview(StrictModel):
    token: str = Field(min_length=24, max_length=2000)


class PlayerResponse(ResponseReview):
    expected_revision: int = Field(ge=1)
    action: Literal["update_season", "respond_meet"]
    status: Literal["active", "withdrawn", "available", "maybe", "unavailable"]
    name: str | None = Field(default=None, min_length=1, max_length=120)
    email: str | None = Field(default=None, max_length=254)
    divisions: list[str] | None = Field(default=None, max_length=8)
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def action_fields(self):
        if self.action == "update_season":
            if self.status not in {"active", "withdrawn"} or None in (self.name, self.email, self.divisions, self.notes):
                raise ValueError("Include your season signup details and status.")
            without_email = self.status == "withdrawn" and not self.email
            details = MemberDetails(name=self.name, email="withdrawal@example.invalid" if without_email else self.email,
                                    divisions=self.divisions, notes=self.notes)
            self.email = "" if without_email else details.email
        elif self.status not in {"available", "maybe", "unavailable"} or any(v is not None for v in (self.name, self.email, self.divisions, self.notes)):
            raise ValueError("Choose available, maybe, or unavailable for this meet.")
        return self


def _query(db, table, fields="*", **filters):
    query = db.table(table).select(fields)
    for key, value in filters.items():
        query = query.eq(key, value)
    return query


def _rows(query):
    try:
        return query.execute().data or []
    except Exception as exc:
        raise HTTPException(503, "Could not load player signups. Please reload and try again.") from exc


def _one(db, table, fields="*", **filters):
    rows = _rows(_query(db, table, fields, **filters).limit(1))
    return rows[0] if rows else None


def pool_rpc(db, name, params):
    try:
        return db.rpc(name, params).execute().data
    except Exception as exc:
        code = getattr(exc, "code", "")
        if code == "PT409":
            code = "40001"
        status, message = {
            "42501": (403, "Your club cannot perform this action. Check administrator access and accept its season invitation first."),
            "P0002": (404, "This signup or invitation is unavailable."),
            "40001": (409, "This signup changed or has closed. Reload before continuing."),
            "23505": (409, "This player already has a season signup. Reload the player pool."),
            "22023": (422, "Check the player, divisions and response deadline for this club."),
            "PT422": (422, "More than one club profile matches this name. Choose your profile or select that none of the matches is you."),
            "PT423": (423, "This action is locked by the season registration window. Reload the league workspace for its current dates."),
            "54000": (429, "Too many requests. Please wait and try again."),
        }.get(code, (503, "Could not confirm the update. Reload before retrying."))
        raise HTTPException(status, message) from exc


def pool_actor(user, club_id, season_id):
    return {"p_actor_id": user.user_id, "p_actor_email": user.email, "p_club_id": club_id, "p_season_id": str(season_id)}


def _season_club(db, club_id, season_id):
    season = _one(db, "pcs_interclub_seasons", "id,organizer_club_id,details,rules," + REGISTRATION_FIELDS, id=str(season_id))
    participation = _one(db, "pcs_interclub_participations", "status", season_id=str(season_id), club_id=club_id)
    club = _one(db, "clubs", "id,name", id=club_id)
    if not season or not club or not participation or participation["status"] != "accepted":
        raise HTTPException(404, "This club's season signup is unavailable.")
    return club, season


def pool_admin_context(get_supabase_client, authorization, club_id, season_id):
    db = get_supabase_client()
    user, assignments = require_admin_assignments(get_supabase_client=lambda: db, authorization=authorization, requested_club_id=club_id)
    if not any(row["role"] in ADMIN_ROLES for row in assignments):
        raise HTTPException(403, "Club administrator access required.")
    club, season = _season_club(db, club_id, season_id)
    return db, user, club, season


def _secret():
    try:
        return get_explicit_registration_edit_token_secret().encode("utf-8")
    except ValueError as exc:
        raise HTTPException(503, "Player signup links are temporarily unavailable.") from exc


def _b64(value):
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _season_end(season):
    details = season["details"]
    return datetime.fromisoformat(details["end_date"]).replace(tzinfo=ZoneInfo(details["timezone"])) + timedelta(days=1)


def _token(row, kind, expires):
    payload = {"purpose": PURPOSE, "kind": kind, "id": row["id"], "season_id": row["season_id"],
               "club_id": row["club_id"], "nonce": row["token_nonce"], "exp": int(expires.timestamp())}
    encoded = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    signature = _b64(hmac.new(_secret(), (PURPOSE + ":" + encoded).encode(), hashlib.sha256).digest())
    return encoded + "." + signature


def _verify_token(token):
    secret = _secret()
    try:
        encoded, signature = token.split(".")
        expected = _b64(hmac.new(secret, (PURPOSE + ":" + encoded).encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError()
        claims = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if claims.get("purpose") != PURPOSE or claims.get("kind") not in {"season", "meet"} or int(claims["exp"]) <= time.time():
            raise ValueError()
        for key in ("id", "season_id", "nonce"):
            UUID(claims[key])
        if not isinstance(claims["club_id"], str) or not 1 <= len(claims["club_id"]) <= 100:
            raise ValueError()
        return claims
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, AttributeError) as exc:
        raise HTTPException(404, "This private link is invalid or has expired. Ask your club for a new invitation.") from exc


def pool_signup_url(share_id):
    return f"{get_next_web_base_url().rstrip('/')}/interclub/signup/{share_id}"


def pool_member_url(member, season):
    return f"{get_next_web_base_url().rstrip('/')}/interclub/respond#token={_token(member, 'season', _season_end(season))}"


def pool_response_url(response, season, meet):
    return f"{get_next_web_base_url().rstrip('/')}/interclub/respond#token={_token(response, 'meet', _date(meet['starts_at']))}"


def _date(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _public_season(season):
    return {"id": season["id"], **{key: season["details"][key] for key in ("name", "start_date", "end_date", "timezone", "divisions")},
            "registration": registration_state(season)}


def _member(row):
    result = {key: row.get(key) for key in MEMBER_FIELDS.split(",")}
    result["player_id"] = str(row["player_id"]) if row.get("player_id") is not None else None
    return result


def _name(value):
    return " ".join(str(value or "").split()).lower()


def _player(row, season):
    rating = float(row["rating"]) / 400 if row.get("rating") is not None else None
    if rating is not None and (not math.isfinite(rating) or rating <= 0):
        rating = None
    gender = str(row.get("gender") or "").strip().lower()
    gender = "female" if gender in {"f", "female", "woman", "women"} else "male" if gender in {"m", "male", "man", "men"} else None
    return {"id": str(row["id"]), "name": row["name"], "rating": rating, "gender": gender,
            "league_rating": row.get("league_rating"), "eligible_divisions": row.get("eligible_divisions", [])}


def _choices(db, club_id, season, players):
    if not players:
        return []
    details = pool_rpc(db, "pcs_interclub_pool_player_details", {"p_season_id": season["id"], "p_club_id": club_id,
        "p_player_ids": [int(row["id"]) for row in players]})
    by_id = {str(row["player_id"]): row for row in details}
    return [_player({**row, **by_id.get(str(row["id"]), {})}, season) for row in players]


def _player_query(db, club_id, q=""):
    query = _query(db, "players", PLAYER_FIELDS, club_id=club_id, active=True)
    if q.strip():
        term = " ".join(q.split()).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = query.ilike("name", f"%{term}%")
    return query.order("name").order("id")


def _account_player(db, club_id, season, authorization):
    if not authorization:
        return None
    try:
        user = authenticate_bearer(authorization)
    except HTTPException as exc:
        if exc.status_code == 401:
            return None
        raise
    # A previously verified club contact can prefill signup. This never creates
    # an account claim, changes a directory profile, or exposes contact data.
    contacts = _rows(_query(db, "player_profile_update_subscriptions", "player_id,verified_at,unsubscribed_at",
        club_id=club_id, email_normalized=user.email.lower(), request_status="active").limit(100))
    ids = {row["player_id"] for row in contacts if row.get("verified_at") and not row.get("unsubscribed_at")}
    if len(ids) != 1:
        return None
    player = _one(db, "players", PLAYER_FIELDS, club_id=club_id, active=True, id=next(iter(ids)))
    return _choices(db, club_id, season, [player])[0] if player else None


def _profile_summaries(db, club_id, season, members):
    ids = sorted({int(m["player_id"]) for m in members if m.get("player_id") is not None})
    players = _rows(_query(db, "players", PLAYER_FIELDS, club_id=club_id).in_("id", ids)) if ids else []
    by_id = {str(row["id"]): row for row in _choices(db, club_id, season, players)}
    return [{**_member(row), **{key: by_id.get(str(row.get("player_id")), {}).get(key, [] if key == "eligible_divisions" else None)
             for key in ("rating", "league_rating", "gender", "eligible_divisions")}} for row in members]


def _bulk_preview(db, club, season, body):
    players = []
    for offset in range(0, 100000, 500):
        page = _rows(_player_query(db, club["id"]).range(offset, offset + 499))
        players.extend(page)
        if len(page) < 500:
            break
    else:
        raise HTTPException(422, "This directory is too large to review in one batch. Contact support.")
    by_id = {str(p["id"]): p for p in players}
    by_name = {}
    for player in players:
        by_name.setdefault(_name(player["name"]), []).append(player)
    relevant = {str(p["id"]): p for item in body.members for p in
                ([by_id[str(item.player_id)]] if item.player_id is not None and str(item.player_id) in by_id else by_name.get(_name(item.name), []))}
    choices = {p["id"]: p for p in _choices(db, club["id"], season, list(relevant.values()))}
    contacts = _rows(_query(db, "player_profile_update_subscriptions",
        "player_id,email,email_normalized,verified_at,unsubscribed_at,preferences_json", club_id=club["id"], request_status="active")
        .in_("player_id", [int(pid) for pid in relevant])) if relevant else []
    contact_emails = {}
    for contact in contacts:
        preferences = contact.get("preferences_json") or {}
        if not contact.get("verified_at") or contact.get("unsubscribed_at") or preferences.get("optional_emails_enabled") is False or preferences.get("unsubscribe_scope") == "global":
            continue
        try:
            email = MemberDetails.email_address(str(contact.get("email_normalized") or contact.get("email") or ""))
        except ValueError:
            continue
        contact_emails.setdefault(str(contact["player_id"]), set()).add(email)
    existing = _rows(_query(db, MEMBERS, "name,email,player_id", season_id=season["id"], club_id=club["id"]).limit(1000))
    seen = list(existing)
    rows = []
    for index, item in enumerate(body.members):
        if not set(item.divisions).issubset(season["details"]["divisions"]):
            raise HTTPException(422, "Choose divisions from this season.")
        explicit_unlinked = "player_id" in item.model_fields_set and item.player_id is None
        matches = [] if explicit_unlinked else by_name.get(_name(item.name), [])
        player = by_id.get(str(item.player_id)) if item.player_id is not None else matches[0] if len(matches) == 1 else None
        if item.player_id is not None and not player:
            raise HTTPException(422, "Choose an active player from this club.")
        choice = choices[str(player["id"])] if player else {"rating": None, "league_rating": None, "gender": None, "eligible_divisions": []}
        name = player["name"] if player else " ".join(item.name.split())
        pid = str(player["id"]) if player else None
        saved_emails = contact_emails.get(pid, set())
        email = item.email or (next(iter(saved_emails)) if len(saved_emails) == 1 and "email" not in item.model_fields_set else "")
        duplicate = any((pid is not None and str(old.get("player_id")) == pid) or
            ((pid is None or old.get("player_id") is None) and _name(old["name"]) == _name(name) and
             (not email or not old.get("email") or old["email"].lower() == email)) for old in seen)
        status = "duplicate" if duplicate else "ambiguous" if player is None and len(matches) > 1 else "matched" if player else "new"
        row = {"index": index, "name": name, "email": email, "player_id": pid,
               "divisions": item.divisions or choice["eligible_divisions"], "notes": item.notes, "status": status,
               "candidates": [choices[str(p["id"])] for p in matches if str(p["id"]) in choices],
               **{key: choice[key] for key in ("rating", "league_rating", "gender", "eligible_divisions")}}
        rows.append(row)
        if status in {"matched", "new"}:
            seen.append(row)
    return {"rows": rows, "ready_count": sum(r["status"] in {"matched", "new"} for r in rows),
            "duplicate_count": sum(r["status"] == "duplicate" for r in rows),
            "ambiguous_count": sum(r["status"] == "ambiguous" for r in rows)}


def _settings(row, season):
    return {"share_id": row["share_id"], "revision": row["revision"], "open": registration_state(season)["can_register"], "url": pool_signup_url(row["share_id"])} if row else {"share_id": None, "revision": 0, "open": False, "url": None}


def _meet(db, club_id, season_id, meet_id):
    row = _one(db, "pcs_interclub_meets", MEET_FIELDS, id=str(meet_id), season_id=str(season_id))
    if not row or club_id not in row["club_ids"]:
        raise HTTPException(404, "This meet is unavailable for your club.")
    host = _one(db, "clubs", "id,name", id=row["host_club_id"])
    return {**row, "host_club_name": host["name"] if host else row["host_club_id"]}


def _pool_payload(db, club, season):
    filters = {"club_id": club["id"], "season_id": season["id"]}
    members = _rows(_query(db, MEMBERS, MEMBER_FIELDS + ",token_nonce", **filters).order("name").order("id").limit(1000))
    summaries = _profile_summaries(db, club["id"], season, members)
    return {"club": club, "season": _public_season(season), "registration": registration_state(season), "signup": _settings(_one(db, POOL, **filters), season),
            "members": [{**summary, "manage_url": pool_member_url(row, season)} for row, summary in zip(members, summaries)], "email_mode": get_email_mode()}


def _availability_payload(db, club, season, meet):
    filters = {"club_id": club["id"], "season_id": season["id"]}
    settings = _one(db, SETTINGS, **filters, meet_id=meet["id"])
    responses = _rows(_query(db, RESPONSES, **filters, meet_id=meet["id"]).order("invited_at").limit(1000))
    members = _rows(_query(db, MEMBERS, MEMBER_FIELDS, **filters).in_("id", [r["member_id"] for r in responses])) if responses else []
    by_id = {m["id"]: m for m in members}
    enriched = []
    for row in responses:
        member = by_id.get(row["member_id"])
        if not member:
            continue
        enriched.append({**{key: row.get(key) for key in ("id", "member_id", "status", "revision", "invited_at", "responded_at")},
                         **{key: _member(member)[key] for key in ("name", "email", "player_id", "divisions", "notes")},
                         "member_status": member["status"], "response_url": pool_response_url(row, season, meet)})
    return {"meet": meet, "settings": {key: settings[key] for key in ("revision", "open", "deadline")} if settings else {"revision": 0, "open": False, "deadline": None},
            "responses": enriched, "email_mode": get_email_mode()}


def prepare_meet_invitations(db, user, club_id, season_id, meet_id, member_ids):
    _, season = _season_club(db, club_id, season_id)
    require_season_phase(season)
    _secret()  # Fail before creating invitations if the signing configuration is unavailable.
    return pool_rpc(db, "pcs_interclub_pool_action", {**pool_actor(user, club_id, season_id), "p_action": "invite",
        "p_payload": {"meet_id": str(meet_id), "member_ids": [str(value) for value in member_ids]}})


def _private_headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    response.headers["Referrer-Policy"] = "no-referrer"


def _requester(request):
    peer = str(request.client.host if request.client else "unknown")
    fly = str(request.headers.get("fly-client-ip") or peer)[:128]
    forwarded = str(request.headers.get("x-vercel-forwarded-for") or request.headers.get("x-forwarded-for") or "").split(",")[-1].strip()[:128]
    return hmac.new(_secret(), f"{PURPOSE}:rate:{fly}:{forwarded}".encode(), hashlib.sha256).hexdigest()


def _review(db, claims):
    club, season = _season_club(db, claims["club_id"], claims["season_id"])
    table = MEMBERS if claims["kind"] == "season" else RESPONSES
    row = _one(db, table, id=claims["id"], season_id=claims["season_id"], club_id=claims["club_id"])
    if not row or not hmac.compare_digest(str(row.get("token_nonce", "")), claims["nonce"]):
        raise HTTPException(404, "This private link is no longer available.")
    member = row if claims["kind"] == "season" else _one(db, MEMBERS, id=row["member_id"], season_id=claims["season_id"], club_id=claims["club_id"])
    if not member:
        raise HTTPException(404, "This player signup is unavailable.")
    result = {"kind": claims["kind"], "club": club, "season": _public_season(season), "member": _member(member), "can_respond": False}
    now = datetime.now(timezone.utc)
    if claims["kind"] == "season":
        result["can_respond"] = bool(registration_state(season)["can_register"] and _season_end(season) > now)
        result["can_withdraw"] = bool(member["status"] == "active" and _season_end(season) > now)
    else:
        require_season_phase(season)
        meet = _meet(db, club["id"], season["id"], row["meet_id"])
        settings = _one(db, SETTINGS, season_id=season["id"], club_id=club["id"], meet_id=meet["id"])
        if not settings:
            raise HTTPException(404, "This meet invitation is unavailable.")
        result["meet"] = meet
        result["availability"] = {"status": row["status"], "revision": row["revision"], "deadline": settings["deadline"], "open": settings["open"]}
        result["can_respond"] = bool(member["status"] == "active" and settings["open"] and _date(settings["deadline"]) > now and _date(meet["starts_at"]) > now)
    return result


def install_interclub_player_pool_routes(app, *, get_supabase_client):
    base = "/admin/clubs/{club_id}/interclub/registrations/{season_id}"

    @app.middleware("http")
    async def private_signup_headers(request, call_next):
        result = await call_next(request)
        path = request.url.path
        if path.startswith(("/public/interclub-signups/", "/public/interclub-player-response/")) or re.fullmatch(
            r"/admin/clubs/[^/]+/interclub/(?:registrations/[^/]+/(?:pool(?:/.*)?|meets/[^/]+/availability)|player-pools/[^/]+/emails(?:/.*)?)", path
        ):
            _private_headers(result)
        return result

    def organizer_context(club_id, season_id, authorization):
        db = get_supabase_client()
        user, assignments = require_admin_assignments(get_supabase_client=lambda: db, authorization=authorization, requested_club_id=club_id)
        if not any(row["role"] in ADMIN_ROLES for row in assignments):
            raise HTTPException(403, "Club administrator access required.")
        season = _one(db, "pcs_interclub_seasons", "id,organizer_club_id", id=str(season_id))
        if not season or season["organizer_club_id"] != club_id:
            raise HTTPException(403, "Only the league organizer approves late season additions.")
        return db, user

    @app.get(base + "/pool/approvals")
    def get_approvals(club_id: str, season_id: UUID, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        db, _ = organizer_context(club_id, season_id, authorization)
        rows = _rows(_query(db, MEMBERS, APPROVAL_FIELDS, season_id=str(season_id), status="active").order("name").limit(1000))
        return {"members": [{key: row.get(key) for key in APPROVAL_FIELDS.split(",")} for row in rows]}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/pool/approvals")
    def review_approval(club_id: str, season_id: UUID, body: PoolApproval, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        db, user = organizer_context(club_id, season_id, authorization)
        result = pool_rpc(db, "pcs_review_interclub_pool_member", {**pool_actor(user, club_id, season_id),
            "p_member_id": str(body.member_id), "p_revision": body.expected_revision, "p_approve": body.approve, "p_reason": body.reason})
        return {"member": {key: result.get(key) for key in APPROVAL_FIELDS.split(",")}}

    @app.get(base + "/pool")
    def get_pool(club_id: str, season_id: UUID, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        db, _, club, season = pool_admin_context(get_supabase_client, authorization, club_id, season_id)
        return _pool_payload(db, club, season)

    @app.put("/admin/clubs/{club_id}/interclub/registrations/{season_id}/pool")
    def update_pool(club_id: str, season_id: UUID, body: PoolSettingsUpdate, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        db, user, club, season = pool_admin_context(get_supabase_client, authorization, club_id, season_id)
        _secret()
        pool_rpc(db, "pcs_interclub_pool_action", {**pool_actor(user, club_id, season_id), "p_action": "settings", "p_payload": body.model_dump()})
        return _pool_payload(db, club, season)

    @app.get(base + "/pool/players")
    def pool_players(club_id: str, season_id: UUID, response: Response, q: str = Query(default="", max_length=120),
                     offset: int = Query(default=0, ge=0, le=100000), authorization: str | None = auth_header()):
        _private_headers(response)
        db, _, club, season = pool_admin_context(get_supabase_client, authorization, club_id, season_id)
        rows = _rows(_player_query(db, club["id"], q).range(offset, offset + 100))
        return {"players": _choices(db, club["id"], season, rows[:100]), "next_offset": offset + 100 if len(rows) > 100 else None}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/pool/bulk-preview")
    def preview_members(club_id: str, season_id: UUID, body: BulkPoolMembers, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        db, _, club, season = pool_admin_context(get_supabase_client, authorization, club_id, season_id)
        require_season_phase(season, intake=True)
        return _bulk_preview(db, club, season, body)

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/pool/bulk-add")
    def add_members(club_id: str, season_id: UUID, body: BulkPoolMembers, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        db, user, club, season = pool_admin_context(get_supabase_client, authorization, club_id, season_id)
        require_season_phase(season, intake=True)
        _secret()  # Personal links must work before retaining any new entries.
        preview = _bulk_preview(db, club, season, body)
        if preview["ambiguous_count"]:
            raise HTTPException(422, "Choose the correct club profile for each name with more than one match.")
        members = [{key: row[key] for key in ("name", "email", "player_id", "divisions", "notes")} for row in preview["rows"]]
        result = pool_rpc(db, "pcs_interclub_pool_bulk_add", {**pool_actor(user, club_id, season_id), "p_members": members})
        return {"added_count": result["added_count"], "skipped_count": result["skipped_count"], "pool": _pool_payload(db, club, season)}

    @app.patch("/admin/clubs/{club_id}/interclub/registrations/{season_id}/pool/members/{member_id}")
    def update_member(club_id: str, season_id: UUID, member_id: UUID, body: PoolMemberUpdate, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        db, user, _, season = pool_admin_context(get_supabase_client, authorization, club_id, season_id)
        if body.status == "active":
            current = _one(db, MEMBERS, "status", id=str(member_id), season_id=str(season_id), club_id=club_id)
            if current and current["status"] == "withdrawn":
                require_season_phase(season, intake=True)
        row = pool_rpc(db, "pcs_interclub_pool_action", {**pool_actor(user, club_id, season_id), "p_action": "member", "p_payload": {**body.model_dump(), "member_id": str(member_id)}})
        return {"member": {**_profile_summaries(db, club_id, season, [row])[0], "manage_url": pool_member_url(row, season)}}

    @app.get(base + "/meets/{meet_id}/availability")
    def get_availability(club_id: str, season_id: UUID, meet_id: UUID, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        db, _, club, season = pool_admin_context(get_supabase_client, authorization, club_id, season_id)
        require_season_phase(season)
        return _availability_payload(db, club, season, _meet(db, club_id, season_id, meet_id))

    @app.put("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/availability")
    def update_availability(club_id: str, season_id: UUID, meet_id: UUID, body: AvailabilitySettingsUpdate, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        db, user, club, season = pool_admin_context(get_supabase_client, authorization, club_id, season_id)
        require_season_phase(season)
        meet = _meet(db, club_id, season_id, meet_id)
        pool_rpc(db, "pcs_interclub_pool_action", {**pool_actor(user, club_id, season_id), "p_action": "availability", "p_payload": {**body.model_dump(mode="json"), "meet_id": str(meet_id)}})
        return _availability_payload(db, club, season, meet)

    @app.get("/public/interclub-signups/{share_id}")
    def public_signup(share_id: UUID, response: Response):
        _private_headers(response)
        db = get_supabase_client()
        settings = _one(db, POOL, share_id=str(share_id))
        if not settings:
            raise HTTPException(404, "This season signup link is unavailable.")
        club, season = _season_club(db, settings["club_id"], settings["season_id"])
        meets = _rows(_query(db, "pcs_interclub_meets", MEET_FIELDS, season_id=season["id"]).contains("club_ids", json.dumps([club["id"]])).order("starts_at").limit(100))
        host_ids = list({m["host_club_id"] for m in meets})
        hosts = _rows(_query(db, "clubs", "id,name").in_("id", host_ids)) if host_ids else []
        host_names = {host["id"]: host["name"] for host in hosts}
        return {"club": club, "season": _public_season(season), "signup": {"open": registration_state(season)["can_register"] and _season_end(season) > datetime.now(timezone.utc)},
                "meets": [{**{key: row[key] for key in ("id", "host_club_id", "starts_at")}, "host_club_name": host_names.get(row["host_club_id"], row["host_club_id"])} for row in meets]}

    @app.post("/public/interclub-signups/{share_id}")
    def submit_signup(share_id: UUID, body: PoolSignup, request: Request, response: Response, authorization: str | None = auth_header()):
        _private_headers(response)
        require_public_intake_or_403()
        if body.website:
            raise HTTPException(422, "Please leave the website field blank.")
        requester = _requester(request)
        db = get_supabase_client()
        settings = _one(db, POOL, share_id=str(share_id))
        if not settings:
            raise HTTPException(404, "This season signup link is unavailable.")
        club, season = _season_club(db, settings["club_id"], settings["season_id"])
        require_season_phase(season, intake=True)
        # Fingerprint the submitted details, not mutable member data. An exact
        # network retry recovers its private link; another signup never can.
        payload = body.model_dump(mode="json", exclude={"website"} | ({"player_id"} if "player_id" not in body.model_fields_set else set()))
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if authorization and "player_id" not in body.model_fields_set:
            linked = _account_player(db, club["id"], season, authorization)
            if linked and _name(linked["name"]) == _name(body.name):
                payload["player_id"] = int(linked["id"])
        result = pool_rpc(db, "pcs_interclub_pool_public_action", {"p_action": "signup", "p_payload": {**payload, "share_id": str(share_id), "request_fingerprint": fingerprint}, "p_requester_hash": requester})
        if result["status"] == "already_registered":
            return {"status": "already_registered", "message": "If you have already joined, use your saved personal link or contact your club administrator. Your existing signup has not changed."}
        member = result["member"]
        _, season = _season_club(db, member["club_id"], member["season_id"])
        return {"status": "registered", "message": "Your season signup is saved. Late additions need league organizer approval before playing." if member.get("late_join") else "Your season signup is saved. Your club will invite you to individual meets.", "manage_url": pool_member_url(member, season)}

    @app.get("/public/interclub-signups/{share_id}/players")
    def signup_players(share_id: UUID, response: Response, q: str = Query(default="", max_length=120), authorization: str | None = auth_header()):
        _private_headers(response)
        db = get_supabase_client()
        settings = _one(db, POOL, share_id=str(share_id))
        if not settings:
            raise HTTPException(404, "This season signup link is unavailable.")
        club, season = _season_club(db, settings["club_id"], settings["season_id"])
        require_season_phase(season, intake=True)
        if _season_end(season) <= datetime.now(timezone.utc):
            raise HTTPException(409, "This season signup is closed.")
        linked = _account_player(db, club["id"], season, authorization)
        rows = pool_rpc(db, "pcs_interclub_pool_search_players", {"p_season_id": season["id"], "p_club_id": club["id"],
            "p_query": q}) if len(q.strip()) >= 2 else []
        return {"players": _choices(db, club["id"], season, rows), "linked_player": linked}

    @app.post("/public/interclub-player-response/review")
    def review_response(body: ResponseReview, response: Response):
        _private_headers(response)
        return _review(get_supabase_client(), _verify_token(body.token))

    @app.post("/public/interclub-player-response/respond")
    def respond(body: PlayerResponse, request: Request, response: Response):
        _private_headers(response)
        require_public_intake_or_403()
        claims = _verify_token(body.token)
        if (claims["kind"] == "season") != (body.action == "update_season"):
            raise HTTPException(404, "This private link cannot make that response.")
        db = get_supabase_client()
        review = _review(db, claims)
        _, season = _season_club(db, claims["club_id"], claims["season_id"])
        if body.action == "respond_meet":
            require_season_phase(season)
            if not review["can_respond"]:
                raise HTTPException(409, "This meet is no longer accepting responses.")
        elif body.status != "withdrawn":
            require_season_phase(season, intake=True)
        elif not review.get("can_withdraw"):
            raise HTTPException(409, "This player is no longer in the active season pool.")
        payload = {**body.model_dump(exclude={"token", "action"}, exclude_none=True), **{k: claims[k] for k in ("id", "season_id", "club_id", "nonce")}}
        if body.action == "update_season" and body.status == "withdrawn" and not registration_state(season)["can_register"]:
            payload.update({key: review["member"][key] for key in ("name", "email", "divisions", "notes")})
        pool_rpc(db, "pcs_interclub_pool_public_action", {"p_action": body.action, "p_payload": payload, "p_requester_hash": _requester(request)})
        return _review(db, claims)
