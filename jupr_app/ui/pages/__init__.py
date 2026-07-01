"""Lightweight page package helpers."""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any


_CONTROL_EXCEPTION_NAMES = {"RerunException", "StopException"}
_OPTIONAL_LEAGUE_METADATA_COLUMNS = {
    "status",
    "started_at",
    "ended_at",
    "ended_by",
    "end_awards",
    "schedule_config",
    "court_board_defaults",
    "rules_config",
    "awards_config",
    "event_tags",
}
_LEGACY_LEAGUE_METADATA_COLUMNS = {"is_active", "min_games", "description", "k_factor"}


def _is_streamlit_control_exception(exc: BaseException) -> bool:
    return exc.__class__.__name__ in _CONTROL_EXCEPTION_NAMES


def _status_text(value: object) -> str:
    return str(value or "").strip().lower()


def _is_archived_status(value: object) -> bool:
    return _status_text(value) == "archived"


def _is_ended_status(value: object) -> bool:
    return _status_text(value) in {"ended", "completed", "complete", "done"}


def _is_active_status(value: object) -> bool:
    return _status_text(value) in {"active", "running", "live"}


def _prime_end_league_wizard_state(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    status = payload.get("status")
    if _is_archived_status(status):
        import streamlit as st

        st.session_state["end_league_wizard_open"] = False
        st.session_state["end_league_step"] = 1
        return
    if _is_ended_status(status) or payload.get("ended_at"):
        import streamlit as st

        if payload.get("ended_at"):
            st.session_state["end_league_frozen_at"] = str(payload.get("ended_at"))
        st.session_state["end_league_step"] = 2
        st.session_state["end_league_wizard_open"] = True


def _looks_like_optional_league_schema_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if not text:
        return False
    mentions_optional_column = any(col in text for col in _OPTIONAL_LEAGUE_METADATA_COLUMNS)
    mentions_schema = any(
        marker in text
        for marker in [
            "schema cache",
            "could not find",
            "does not exist",
            "column",
            "42703",
            "pgrst204",
        ]
    )
    return mentions_optional_column and mentions_schema


def _minimal_lifecycle_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    allowed: dict[str, Any] = {}
    for key in ["status", "is_active", "started_at", "ended_at", "ended_by", "end_awards"]:
        if key in payload:
            allowed[key] = payload[key]
    return allowed


def _legacy_league_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    status = payload.get("status")
    if _is_archived_status(status) or _is_ended_status(status):
        return {"is_active": False}
    if _is_active_status(status):
        allowed = {"is_active": True}
    else:
        allowed = {}
    for key in _LEGACY_LEAGUE_METADATA_COLUMNS:
        if key in payload:
            allowed[key] = payload[key]
    if payload.get("is_active") is False:
        allowed["is_active"] = False
    return allowed


class _LeagueMetadataQueryGuard:
    def __init__(self, supabase: Any, inner: Any, payload: Any):
        self._supabase = supabase
        self._inner = inner
        self._payload = payload
        self._filters: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def eq(self, *args: Any, **kwargs: Any) -> "_LeagueMetadataQueryGuard":
        self._filters.append(("eq", args, kwargs))
        self._inner = self._inner.eq(*args, **kwargs)
        return self

    def _apply_filters(self, query: Any) -> Any:
        for method_name, method_args, method_kwargs in self._filters:
            query = getattr(query, method_name)(*method_args, **method_kwargs)
        return query

    def _retry_update(self, payload: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        query = self._supabase.table("leagues_metadata").update(payload)
        query = self._apply_filters(query)
        return query.execute(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._inner.execute(*args, **kwargs)
        except BaseException as exc:
            if _is_streamlit_control_exception(exc) or not _looks_like_optional_league_schema_error(exc):
                raise
            lifecycle_payload = _minimal_lifecycle_payload(self._payload)
            if lifecycle_payload:
                try:
                    return self._retry_update(lifecycle_payload, *args, **kwargs)
                except BaseException as lifecycle_exc:
                    if _is_streamlit_control_exception(lifecycle_exc) or not _looks_like_optional_league_schema_error(lifecycle_exc):
                        raise
            legacy_payload = _legacy_league_payload(self._payload)
            if not legacy_payload:
                raise
            return self._retry_update(legacy_payload, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def _call(*args: Any, **kwargs: Any) -> Any:
            self._filters.append((name, args, kwargs))
            result = attr(*args, **kwargs)
            if result is self._inner:
                return self
            self._inner = result
            return self

        return _call


class _LeagueMetadataTableGuard:
    def __init__(self, supabase: Any, inner: Any):
        self._supabase = supabase
        self._inner = inner

    def update(self, payload: Any, *args: Any, **kwargs: Any) -> _LeagueMetadataQueryGuard:
        _prime_end_league_wizard_state(payload)
        return _LeagueMetadataQueryGuard(
            self._supabase,
            self._inner.update(payload, *args, **kwargs),
            payload,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _LeagueManagerSupabaseGuard:
    def __init__(self, inner: Any):
        self._inner = inner

    def table(self, table_name: str) -> Any:
        table = self._inner.table(table_name)
        if str(table_name) == "leagues_metadata":
            return _LeagueMetadataTableGuard(self._inner, table)
        return table

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


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
    has_ended_at = ended_at is not None and not pd.isna(ended_at) and str(ended_at).strip() != ""

    if _is_archived_status(status):
        st.session_state["end_league_wizard_open"] = False
        st.session_state["end_league_step"] = 1
    elif _is_ended_status(status) or has_ended_at:
        if has_ended_at:
            st.session_state["end_league_frozen_at"] = str(ended_at)
        st.session_state["end_league_step"] = 2
        st.session_state["end_league_wizard_open"] = True


def _render_league_manager(ctx: Any) -> None:
    _maybe_advance_closed_league_wizard(ctx)
    module = importlib.import_module(f"{__name__}.league_manager")
    globals()["league_manager"] = _LEAGUE_MANAGER_PROXY
    original_supabase = getattr(ctx, "supabase", None)
    if original_supabase is not None:
        ctx.supabase = _LeagueManagerSupabaseGuard(original_supabase)
    try:
        module.render(ctx)
    finally:
        if original_supabase is not None:
            ctx.supabase = original_supabase


_LEAGUE_MANAGER_PROXY = SimpleNamespace(render=_render_league_manager)
league_manager = _LEAGUE_MANAGER_PROXY
