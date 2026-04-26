# jupr_app/ui/public_links.py
from __future__ import annotations

import logging
import urllib.parse
from collections.abc import Mapping
from datetime import datetime, timezone

import streamlit as st

logger = logging.getLogger(__name__)
_NAV_DEBUG_EVENTS_KEY = "jupr_nav_debug_events"
_NAV_DEBUG_EVENTS_MAX = 50


def get_base_url() -> str:
    return (st.session_state.get("base_url", "") or "").rstrip("/")


def build_public_url(*, page: str, params: dict[str, str] | None = None) -> str:
    """
    Builds a canonical public URL using the configured Streamlit Cloud base.
    NOTE: public=1 is retained for backward compatibility with older links.
    TODO: remove the public=1 requirement from generated links once all callers
    have migrated to the public-first base URL behavior.
    """
    base = get_base_url()
    q = {"page": page, "public": "1"}

    if params:
        for k, v in params.items():
            if v is None:
                continue
            q[str(k)] = str(v)

    return f"{base}/?{urllib.parse.urlencode(q)}"


def _clean_param_values(params: Mapping[str, object] | None = None) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            cleaned[str(key)] = text_value
    return cleaned


def build_public_params(page: str, params: dict[str, str] | None = None) -> dict[str, str]:
    public_params = {"page": str(page).strip(), "public": "1"}
    public_params.update(_clean_param_values(params))
    return public_params


def build_route_params(
    *,
    page: str,
    params: Mapping[str, object] | None = None,
    public_mode: bool = True,
) -> dict[str, str]:
    route_params = {"page": str(page).strip()}
    route_params.update(_clean_param_values(params))
    if public_mode:
        route_params["public"] = "1"
        route_params.pop("admin", None)
    else:
        route_params["admin"] = "1"
        route_params.pop("public", None)
    return route_params


def _query_params_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in st.query_params.keys():
        value = st.query_params.get(key, "")
        if isinstance(value, list):
            value = value[0] if value else ""
        text_value = str(value or "").strip()
        if text_value:
            snapshot[str(key)] = text_value
    return snapshot


def _append_nav_debug_event(event: Mapping[str, object]) -> None:
    events = list(st.session_state.get(_NAV_DEBUG_EVENTS_KEY, []))
    events.append(dict(event))
    st.session_state[_NAV_DEBUG_EVENTS_KEY] = events[-_NAV_DEBUG_EVENTS_MAX:]


def navigate_same_tab(
    page: str,
    params: dict[str, str] | None = None,
    public_mode: bool = True,
    clear_existing: bool = True,
    source: str | None = None,
) -> None:
    previous_query_params = _query_params_snapshot()
    next_params = build_route_params(page=page, params=params, public_mode=public_mode)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(source or "").strip() or "navigate_same_tab",
        "target_page": str(page).strip(),
        "params": _clean_param_values(params),
        "public_mode": bool(public_mode),
        "clear_existing": bool(clear_existing),
        "previous_query_params": previous_query_params,
        "next_query_params": next_params,
    }
    _append_nav_debug_event(event)
    logger.info(
        "Navigation event source=%s target=%s mode=%s clear_existing=%s prev=%s next=%s",
        event["source"],
        event["target_page"],
        "public" if event["public_mode"] else "admin",
        event["clear_existing"],
        previous_query_params,
        next_params,
    )
    if clear_existing:
        st.query_params.clear()
    st.query_params.update(next_params)
    st.rerun()


def public_nav_button(
    label: str,
    page: str,
    params: dict[str, str] | None = None,
    key: str | None = None,
    use_container_width: bool = False,
) -> bool:
    clicked = st.button(label, key=key, use_container_width=use_container_width)
    if clicked:
        navigate_same_tab(
            page=page,
            params=params,
            public_mode=True,
            source=f"public_nav_button:{label}",
        )
    return clicked


def _same_app_query_params_from_url(url: str) -> dict[str, str] | None:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if not parsed.scheme and not parsed.netloc:
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)
        return {k: v[-1] for k, v in query.items() if v and str(v[-1]).strip()}

    base = urllib.parse.urlparse(get_base_url())
    if parsed.scheme in {"http", "https"} and base.netloc and parsed.netloc == base.netloc:
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)
        return {k: v[-1] for k, v in query.items() if v and str(v[-1]).strip()}
    return None


def external_link_button(
    label: str,
    url: str,
    use_container_width: bool = False,
    source: str | None = None,
) -> None:
    logger.info(
        "External link via st.link_button source=%s label=%s url=%s",
        str(source or "").strip() or "external_link_button",
        label,
        url,
    )
    st.link_button(label, url, use_container_width=use_container_width)


def public_link_button(
    label: str,
    url: str,
    *,
    key: str | None = None,
    use_container_width: bool = False,
) -> None:
    """Backward compatible helper: same-app URLs navigate in-tab; external URLs use link_button."""
    route_params = _same_app_query_params_from_url(url)
    if route_params and route_params.get("page"):
        clicked = st.button(label, key=key, use_container_width=use_container_width)
        if clicked:
            next_page = route_params.get("page", "home")
            raw_public = str(route_params.get("public", "")).strip().lower()
            public_mode = raw_public in {"1", "true", "yes", "y", "on"}
            nav_params = {k: v for k, v in route_params.items() if k != "page"}
            navigate_same_tab(
                page=next_page,
                params=nav_params,
                public_mode=public_mode,
                clear_existing=True,
                source=f"public_link_button:{label}:same_app",
            )
        return
    logger.info("public_link_button label=%s classified=external url=%s", label, url)
    external_link_button(
        label,
        url,
        use_container_width=use_container_width,
        source=f"public_link_button:{label}:external",
    )
