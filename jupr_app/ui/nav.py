from __future__ import annotations

import logging

import streamlit as st


logger = logging.getLogger(__name__)


def navigate_to_player_profile(
    pid: int,
    public_mode: bool,
    extra_params: dict | None = None,
) -> None:
    """Navigate internally to the players page using Streamlit query params + rerun."""
    try:
        player_id = int(pid)
    except Exception:
        logger.warning("Navigation skipped: invalid player id pid=%r", pid)
        return

    next_params: dict[str, str] = {"page": "players", "pid": str(player_id)}
    if public_mode:
        next_params["public"] = "1"

    for key, value in (extra_params or {}).items():
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            next_params[str(key)] = text_value

    st.query_params.clear()
    st.query_params.update(next_params)
    logger.info(
        "Internal navigation to player profile via query-param mutation (full_reload=False): pid=%s public_mode=%s extra_keys=%s",
        player_id,
        public_mode,
        sorted((extra_params or {}).keys()),
    )
    st.rerun()
