from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from datetime import date, datetime
from typing import Any
import math

LIVE_SESSION_STATE_VERSION = 1

RECOVERABLE_PAGE_STATE_KEYS = (
    "event",
    "type_label",
    "event_name",
    "participant_count",
    "participant_text",
    "selected_existing_players",
    "league_rounds",
    "official_league",
    "official_week_tag",
    "rating_mode",
    "last_saved_rounds",
    "editing_substitution_id",
    "parsed_roster_lines",
    "roster_candidates",
    "confirmed_roster_rows",
    "roster_confirmed",
    "resolved_roster_ids",
    "admin_roster_rows",
    "default_new_player_rating",
    "quick_paste_nonce",
    "live_session_key",
)


def _is_nan_like(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return math.isnan(value)

    value_type = type(value)
    type_name = value_type.__name__
    type_module = value_type.__module__
    if type_module.startswith("pandas") and type_name in {"NAType", "NaTType"}:
        return True
    if type_module.startswith("pandas") and str(value) == "NaT":
        return True
    return False


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if _is_nan_like(value):
        return None

    value_module = type(value).__module__
    if value_module.startswith("numpy") and hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except Exception:
            return None

    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]

    return str(value)


def _is_recoverable_widget_value(value: Any, *, depth: int = 0) -> bool:
    if depth > 6:
        return False
    if value is None or isinstance(value, (str, bool, int, float, datetime, date)):
        return True
    if _is_nan_like(value):
        return True
    value_module = type(value).__module__
    if value_module.startswith("numpy") and hasattr(value, "item"):
        return True
    if value_module.startswith("pandas"):
        # DataFrames/Series and Streamlit editor internals can be large or unstable.
        return False
    if isinstance(value, Mapping):
        return all(
            isinstance(key, (str, int, float, bool))
            and _is_recoverable_widget_value(val, depth=depth + 1)
            for key, val in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return all(_is_recoverable_widget_value(item, depth=depth + 1) for item in value)
    return False


def extract_live_widget_state(
    st_session_state: Mapping[str, Any] | None,
    *,
    config_state_key: str,
) -> dict[str, Any]:
    """Extract only JUPR Live widget keys, not the full Streamlit session."""
    if st_session_state is None:
        return {}
    prefix = f"{config_state_key}_"
    widget_state: dict[str, Any] = {}
    for raw_key, value in st_session_state.items():
        key = str(raw_key)
        if not key.startswith(prefix):
            continue
        if not _is_recoverable_widget_value(value):
            continue
        widget_state[key] = _json_safe_value(value)
    return widget_state


def build_live_state_payload(
    page_state: Mapping[str, Any] | None,
    *,
    club_id: str,
    session_key: str,
    config_state_key: str,
    st_session_state: Mapping[str, Any] | None = None,
    source: str = "jupr_live_admin",
) -> dict[str, Any]:
    """Build the JSONB payload persisted for one recoverable JUPR Live page state."""
    page_state = page_state or {}
    page_payload = {
        key: _json_safe_value(page_state.get(key))
        for key in RECOVERABLE_PAGE_STATE_KEYS
        if key in page_state
    }
    event = page_state.get("event") if isinstance(page_state, Mapping) else None
    event = event if isinstance(event, Mapping) else {}
    event_name = str(page_state.get("event_name") or event.get("name") or "JUPR Live Session").strip()
    event_type = str(event.get("type") or page_state.get("type_label") or "").strip()
    payload = {
        "version": LIVE_SESSION_STATE_VERSION,
        "mode": "quick_session",
        "club_id": str(club_id),
        "session_key": str(session_key),
        "config_state_key": str(config_state_key),
        "source": str(source or "jupr_live_admin"),
        "event_name": event_name,
        "event_type": event_type,
        "page_state": page_payload,
        "widget_state": extract_live_widget_state(
            st_session_state,
            config_state_key=config_state_key,
        ),
    }
    return _json_safe_value(payload)


def live_state_title(live_state: Mapping[str, Any] | None, *, fallback: str = "JUPR Live Session") -> str:
    if not isinstance(live_state, Mapping):
        return fallback
    title = str(live_state.get("event_name") or "").strip()
    if title:
        return title
    page_state = live_state.get("page_state")
    if isinstance(page_state, Mapping):
        title = str(page_state.get("event_name") or "").strip()
        if title:
            return title
    return fallback


def hydrate_page_state_from_live_state(
    target_state: MutableMapping[str, Any],
    live_state: Mapping[str, Any] | None,
) -> MutableMapping[str, Any]:
    """Hydrate the saved page-level state dict into Streamlit page state."""
    if not isinstance(live_state, Mapping):
        return target_state
    page_state = live_state.get("page_state")
    if not isinstance(page_state, Mapping):
        page_state = live_state
    for key in RECOVERABLE_PAGE_STATE_KEYS:
        if key in page_state:
            target_state[key] = deepcopy(page_state[key])
    return target_state


def hydrate_widget_state_from_live_state(
    st_session_state: MutableMapping[str, Any],
    live_state: Mapping[str, Any] | None,
    *,
    config_state_key: str,
) -> None:
    """Restore only persisted JUPR Live widget keys into Streamlit session state."""
    if not isinstance(live_state, Mapping):
        return
    widget_state = live_state.get("widget_state")
    if not isinstance(widget_state, Mapping):
        return
    prefix = f"{config_state_key}_"
    for raw_key, value in widget_state.items():
        key = str(raw_key)
        if key.startswith(prefix):
            st_session_state[key] = deepcopy(value)
