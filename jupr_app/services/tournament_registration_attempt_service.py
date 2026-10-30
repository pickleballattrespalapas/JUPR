"""Small, private support breadcrumbs. Never a draft registration or identity proof."""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from jupr_app.domain.tournament_registration_repo import get_public_tournament_bundle

LOGGER = logging.getLogger(__name__)
TABLE = "tournament_registration_attempt_events"
CLIENT_EVENTS = {"PROGRESS", "CLIENT_REJECTED", "LEFT_FORM"}
STAGES = {"contact", "profile", "events", "review", "submit", "edit"}


def clean_text(value: Any, limit: int = 160) -> str:
    return re.sub(r"[\x00-\x1f<>]", "", str(value or "")).strip()[:limit]


def support_message(value: Any) -> str:
    text = clean_text(value, 500)
    text = re.sub(r"https?://\S+", "[link omitted]", text)
    text = re.sub(r"eyJ[A-Za-z0-9_.-]+", "[token omitted]", text)
    return re.sub(r"(?i)(token|secret|password|authorization)\s*[:=]\s*\S+", r"\1=[omitted]", text)


def error_code(message: str) -> str:
    for phrase, code in (("own partner", "SELF_PARTNER"), ("already", "DUPLICATE_OR_CONFLICT"),
                         ("rating", "RATING_ELIGIBILITY"), ("skill", "RATING_ELIGIBILITY"),
                         ("gender", "GENDER_ELIGIBILITY"), ("age", "AGE_OR_REQUIRED_FIELD"),
                         ("required", "REQUIRED_FIELD"), ("connection", "CONNECTION")):
        if phrase in message.lower():
            return code
    return "FORM_VALIDATION"


def _scope(supabase: Any, club_id: str, payload: dict[str, Any]) -> tuple[str, set[str]]:
    tournament, settings, _, events = get_public_tournament_bundle(
        supabase, club_id=club_id, tournament_id=payload.get("tournament_id"),
        registration_slug=payload.get("registration_slug"),
    )
    if not tournament or not settings:
        raise ValueError("Tournament registration not found.")
    return str(tournament["id"]), {str(row["id"]) for row in events}


def record_registration_attempt(supabase: Any, *, club_id: str, payload: dict[str, Any],
                                source: str = "BROWSER", event: str | None = None,
                                stage: str | None = None, message: str | None = None,
                                registration_id: str | None = None) -> None:
    """Fail open: support storage must never prevent a real registration."""
    try:
        if payload.get("website"):
            return
        attempt_id = str(uuid.UUID(str(payload.get("attempt_id"))))
        browser = source == "BROWSER"
        event = event or str(payload.get("event") or "PROGRESS")
        stage = stage or str(payload.get("stage") or "contact")
        if (browser and event not in CLIENT_EVENTS) or stage not in STAGES:
            return
        tournament_id, event_ids = _scope(supabase, club_id, payload)
        raw_selections = payload.get("selections") or []
        selections = []
        for row in raw_selections[:8]:
            if not isinstance(row, dict) or str(row.get("event_option_id")) not in event_ids:
                continue
            # Partner email/phone, notes, payment data and tokens never enter support storage.
            selections.append({
                "event_option_id": str(row["event_option_id"]),
                "partner_mode": clean_text(row.get("partner_mode"), 40),
                "partner_name": clean_text(row.get("partner_name")),
                "same_email_as_player": bool(row.get("same_email_as_player")) if browser else bool(
                    row.get("partner_email") and str(row["partner_email"]).strip().lower() == str(payload.get("email") or "").strip().lower()),
            })
        msg = support_message(message if message is not None else payload.get("message"))
        row = {
            "id": str(uuid.uuid4()), "attempt_id": attempt_id, "club_id": club_id,
            "tournament_id": tournament_id, "source": source, "event": event, "stage": stage,
            "flow": "edit" if payload.get("flow") == "edit" or stage == "edit" else "new",
            "name": clean_text(payload.get("display_name") or " ".join(str(payload.get(k) or "") for k in ("first_name", "last_name"))),
            "email": clean_text(payload.get("email"), 320).lower(),
            "message": msg, "error_code": error_code(msg) if "REJECTED" in event else "",
            "selections": selections, "registration_id": registration_id if not browser else None,
        }
        supabase.rpc("record_tournament_registration_attempt", {"p_event": row}).execute()
    except Exception:
        # Do not print payloads, contact details, errors containing secrets or recovery tokens.
        LOGGER.warning("Registration support breadcrumb could not be stored")


def audited_registration_call(supabase: Any, *, club_id: str, payload: dict[str, Any],
                              stage: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    audit_payload = {**payload, "attempt_id": payload.get("attempt_id") or str(uuid.uuid4())}
    record_registration_attempt(supabase, club_id=club_id, payload=audit_payload,
                                source="SERVER", event="SUBMITTING", stage=stage)
    try:
        result = operation()
    except Exception as exc:
        msg = str(exc) if isinstance(exc, (ValueError, PermissionError)) else "Registration service could not complete this request."
        record_registration_attempt(supabase, club_id=club_id, payload=audit_payload,
                                    source="SERVER", event="SERVER_REJECTED", stage=stage, message=msg)
        raise
    record_registration_attempt(supabase, club_id=club_id, payload=audit_payload, source="SERVER",
                                event="SAVED" if result.get("registration_id") else "PROFILE_READY", stage=stage,
                                message="Registration withdrawn." if result.get("withdrawn") else None,
                                registration_id=result.get("registration_id"))
    return result


def list_registration_attempts(supabase: Any, *, club_id: str, tournament_id: str,
                               search: str = "") -> dict[str, Any]:
    tournament = supabase.table("tournaments").select("id").eq("id", tournament_id).eq("club_id", club_id).limit(1).execute()
    if not tournament.data:
        raise ValueError("Tournament not found.")
    now = datetime.now(timezone.utc)
    query = (supabase.table(TABLE).select("*").eq("club_id", club_id).eq("tournament_id", tournament_id)
             .gte("created_at", (now - timedelta(days=30)).isoformat()))
    term = clean_text(search, 160)
    if term:
        if "@" in term:
            query = query.eq("email", term.lower())
        else:
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            query = query.ilike("name", f"%{escaped}%")
    matches = query.order("created_at", desc=True).limit(501).execute().data or []
    ids = list(dict.fromkeys(str(row["attempt_id"]) for row in matches[:500]))[:50]
    # Load each matching attempt's whole timeline; older anonymous steps must not
    # disappear merely because the person entered their name later in the form.
    rows = []
    if ids:
        for start in range(0, 5500, 500):
            batch = (supabase.table(TABLE).select("*").eq("club_id", club_id).eq("tournament_id", tournament_id)
                .gte("created_at", (now - timedelta(days=30)).isoformat()).in_("attempt_id", ids)
                .order("created_at").order("id").range(start, start + 499).execute().data or [])
            rows.extend(batch)
            if len(batch) < 500:
                break
    options = supabase.table("tournament_event_options").select("id,division_name,label").eq("tournament_id", tournament_id).limit(500).execute().data or []
    labels = {str(e["id"]): e.get("division_name") or e.get("label") or str(e["id"]) for e in options}
    for row in rows:
        row["selections"] = [{**s, "event_label": labels.get(str(s.get("event_option_id")), "Selected event")} for s in row.get("selections") or []]
    attempts = []
    for attempt_id in ids:
        timeline = [row for row in rows if str(row["attempt_id"]) == attempt_id]
        if not timeline:
            continue
        latest = timeline[-1]
        saved = next((r for r in reversed(timeline) if r["source"] == "SERVER" and r["event"] == "SAVED"), None)
        identity = next((r for r in reversed(timeline) if r.get("email") or r.get("name")), latest)
        last_at = datetime.fromisoformat(latest["created_at"].replace("Z", "+00:00"))
        outcome = "Saved" if saved else "Incomplete — no recent activity" if now - last_at > timedelta(minutes=30) else "In progress / incomplete"
        if saved:
            later_errors = any("REJECTED" in row["event"] for row in timeline[timeline.index(saved) + 1:])
            outcome = "Change not saved — earlier save succeeded" if later_errors else "Withdrawn" if saved.get("message") == "Registration withdrawn." else "Saved"
        elif all(row["source"] == "SERVER" and row["message"] == "Registration edit page loaded." for row in timeline):
            outcome = "Editor loaded — no save recorded"
        attempts.append({"attempt_id": attempt_id, "name": identity.get("name"), "email": identity.get("email"),
                         "outcome": outcome, "last_seen_at": latest["created_at"], "stage": latest["stage"],
                         "registration_id": saved.get("registration_id") if saved else None, "events": timeline})
    return {"attempts": attempts, "retention_days": 30, "truncated": len(matches) > 500 or len(set(r["attempt_id"] for r in matches)) > 50}
