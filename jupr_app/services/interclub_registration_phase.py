"""One commissioner-controlled registration window for every club in a season."""
from datetime import datetime, timezone


REGISTRATION_FIELDS = "registration_opens_at,registration_closes_at,registration_revision"


class RegistrationPhaseError(ValueError):
    """The current season phase does not permit this operation."""


def _instant(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def registration_state(season, *, now=None):
    opens = _instant(season.get("registration_opens_at"))
    closes = _instant(season.get("registration_closes_at"))
    instant = now or datetime.now(timezone.utc)
    status = "unconfigured" if opens is None or closes is None or opens >= closes else (
        "scheduled" if instant < opens else "open" if instant < closes else "closed")
    return {"opens_at": opens.isoformat() if opens else None, "closes_at": closes.isoformat() if closes else None,
            "revision": int(season.get("registration_revision") or 0), "status": status,
            "can_register": status == "open", "meet_planning_open": status == "closed"}


def registration_season(season):
    return {**season, "registration": registration_state(season)}


def require_registration_phase(season, *, intake=False):
    state = registration_state(season)
    if not state["can_register" if intake else "meet_planning_open"]:
        if state["status"] == "unconfigured":
            message = "The league commissioner must set the season registration dates before registration or meet planning can begin."
        elif intake:
            message = "Season registration is not open. The league commissioner sets the registration dates for all clubs."
        else:
            message = "Meet planning opens after season registration closes for all clubs."
        raise RegistrationPhaseError(message)
    return state
