"""Lightweight page package helpers."""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any


_CONTROL_EXCEPTION_NAMES = {"RerunException", "StopException"}


def _is_streamlit_control_exception(exc: Exception) -> bool:
    return exc.__class__.__name__ in _CONTROL_EXCEPTION_NAMES


def _is_ended_status(value: object) -> bool:
    return str(value or "").strip().lower() in {"ended", "archived", "completed", "complete", "done"}


def _maybe_advance_closed_league_wizard(ctx: Any) -> None:
    import pandas as pd
    import streamlit as st

    if not st.session_state.get("end_league_wizard_open", False):
        return
    try:
        current_step = int(st.session_state.get("end_league_step", 1) or 1)
    except Exception:
        current_step = 1
    if current_step > 1:
        return

    selected_league = str(st.session_state.get("league_editor_select", "") or "").strip()
    if not selected_league:
        return

    df_meta = getattr(ctx, "df_meta", None)
    if df_meta is None or getattr(df_meta, "empty", True) or "league_name" not in df_meta.columns:
        return

    meta = df_meta.copy()
    meta["league_name"] = meta["league_name"].fillna("").astype(str).str.strip()
    rows = meta[meta["league_name"] == selected_league]
    if rows.empty:
        return

    row = rows.iloc[0]
    ended_at = row.get("ended_at") if "ended_at" in rows.columns else None
    status = row.get("status") if "status" in rows.columns else ""
    is_active = row.get("is_active") if "is_active" in rows.columns else None
    has_ended_at = ended_at is not None and not pd.isna(ended_at) and str(ended_at).strip() != ""
    inactive = is_active is not None and not bool(is_active)

    if _is_ended_status(status) or has_ended_at:
        if has_ended_at:
            st.session_state["end_league_frozen_at"] = str(ended_at)
        st.session_state["end_league_step"] = 2
        st.session_state["end_league_wizard_open"] = True
    elif inactive and _is_ended_status(status):
        st.session_state["end_league_step"] = 2
        st.session_state["end_league_wizard_open"] = True


def _render_league_manager(ctx: Any) -> None:
    import streamlit as st

    _maybe_advance_closed_league_wizard(ctx)
    module = importlib.import_module(f"{__name__}.league_manager")
    globals()["league_manager"] = _LEAGUE_MANAGER_PROXY
    try:
        module.render(ctx)
    except Exception as exc:
        if _is_streamlit_control_exception(exc):
            if st.session_state.get("end_league_wizard_open", False):
                try:
                    current_step = int(st.session_state.get("end_league_step", 1) or 1)
                except Exception:
                    current_step = 1
                if current_step <= 1:
                    st.session_state["end_league_step"] = 2
                    st.session_state["end_league_wizard_open"] = True
                    st.session_state["force_data_refresh"] = True
            return
        raise


_LEAGUE_MANAGER_PROXY = SimpleNamespace(render=_render_league_manager)
league_manager = _LEAGUE_MANAGER_PROXY
