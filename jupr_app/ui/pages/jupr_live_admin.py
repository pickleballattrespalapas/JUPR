from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import streamlit as st

from jupr_app.domain.live_beta_engine import clear_expired_substitutions
from jupr_app.domain.live_session_repo import (
    LIVE_SESSIONS_INSTALL_MESSAGE,
    abandon_expired_live_sessions,
    get_live_session,
    is_missing_live_sessions_table_error,
    is_restorable_live_session,
    mark_live_session_abandoned,
    upsert_live_session,
)
from jupr_app.services import ServiceContext, submit_match_batch
from jupr_app.services.live_session_state import (
    build_live_state_payload,
    hydrate_page_state_from_live_state,
    hydrate_widget_state_from_live_state,
    live_state_title,
)
from jupr_app.ui.layout import page_shell
from jupr_app.ui.live.shared import (
    LivePageConfig,
    build_league_round_official_payloads,
    build_rr_official_payloads,
    build_tournament_official_payloads,
    mark_tournament_payloads_saved,
    render_live_page,
)


LIVE_SESSION_QUERY_PARAM = "live_session"
LIVE_SESSION_TTL_HOURS = 18

ADMIN_CONFIG = LivePageConfig(
    state_key="jupr_live_admin_state",
    intro_markdown=(
        "Run JUPR Live events as rated overall matches or unrated recorded games. "
        "No league/session setup required."
    ),
    event_types=("Round Robin", "League / Ladder", "Tournament"),
    mode_pill_label="Official",
    allow_official=True,
    allow_tournament=True,
    show_rating_mode=True,
    requires_roster_resolution=True,
    use_admin_roster_builder=True,
)


def _query_param_text(key: str) -> str:
    value = st.query_params.get(key, "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _set_query_param_if_needed(key: str, value: str) -> None:
    value = str(value or "").strip()
    if value and _query_param_text(key) != value:
        st.query_params[key] = value


def _resolve_live_session_key(config: LivePageConfig) -> str:
    state_key = f"{config.state_key}_live_session_key"
    session_key = _query_param_text(LIVE_SESSION_QUERY_PARAM) or str(
        st.session_state.get(state_key) or ""
    ).strip()
    if not session_key:
        session_key = uuid4().hex
    st.session_state[state_key] = session_key
    _set_query_param_if_needed(LIVE_SESSION_QUERY_PARAM, session_key)
    return session_key


def _valid_uuid(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(UUID(text))
    except (TypeError, ValueError):
        return None


def _current_admin_identity(ctx) -> tuple[str | None, str | None]:
    user = st.session_state.get("admin_auth_user")
    user_id = _valid_uuid(getattr(user, "id", None))
    email = (
        str(getattr(user, "email", "") or "").strip().lower()
        or str(getattr(ctx, "admin_email", "") or "").strip().lower()
        or str(st.session_state.get("admin_email") or "").strip().lower()
    )
    return user_id, (email or None)


def _live_session_expires_at_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=LIVE_SESSION_TTL_HOURS)).isoformat()


def _record_live_session_persistence_error(config: LivePageConfig, exc: Exception) -> None:
    message = (
        LIVE_SESSIONS_INSTALL_MESSAGE
        if is_missing_live_sessions_table_error(exc)
        else "Durable JUPR Live recovery is temporarily unavailable; the current browser session can continue."
    )
    st.session_state[f"{config.state_key}_durable_persistence_error"] = message


def _clear_live_session_persistence_error(config: LivePageConfig) -> None:
    st.session_state.pop(f"{config.state_key}_durable_persistence_error", None)


def _restore_live_session_if_needed(ctx, config: LivePageConfig, session_key: str) -> None:
    restore_marker_key = f"{config.state_key}_durable_restored_session_key"
    if st.session_state.get(restore_marker_key) == session_key:
        return

    try:
        row = get_live_session(
            ctx.supabase,
            club_id=str(ctx.club_id),
            session_key=session_key,
        )
    except Exception as exc:
        _record_live_session_persistence_error(config, exc)
        st.session_state[restore_marker_key] = session_key
        return

    st.session_state[restore_marker_key] = session_key
    if not row:
        return

    if not is_restorable_live_session(row):
        if str(row.get("status") or "").strip().lower() == "active":
            try:
                mark_live_session_abandoned(
                    ctx.supabase,
                    club_id=str(ctx.club_id),
                    session_key=session_key,
                )
            except Exception:
                pass
        return

    persisted_state = row.get("state") if isinstance(row.get("state"), dict) else {}
    page_state = st.session_state.setdefault(config.state_key, {})
    if not isinstance(page_state, dict):
        page_state = {}
        st.session_state[config.state_key] = page_state
    hydrate_page_state_from_live_state(page_state, persisted_state)
    hydrate_widget_state_from_live_state(
        st.session_state,
        persisted_state,
        config_state_key=config.state_key,
    )
    st.session_state[f"{config.state_key}_durable_restored"] = True


def _persist_live_session_state(ctx, config: LivePageConfig, session_key: str) -> None:
    page_state = st.session_state.get(config.state_key)
    if not isinstance(page_state, dict):
        return

    page_state["live_session_key"] = session_key
    payload = build_live_state_payload(
        page_state,
        club_id=str(ctx.club_id),
        session_key=session_key,
        config_state_key=config.state_key,
        st_session_state=st.session_state,
        source="jupr_live_admin",
    )
    created_by, created_by_email = _current_admin_identity(ctx)
    upsert_live_session(
        ctx.supabase,
        club_id=str(ctx.club_id),
        session_key=session_key,
        state=payload,
        title=live_state_title(payload),
        created_by=created_by,
        created_by_email=created_by_email,
        expires_at=_live_session_expires_at_iso(),
        source="jupr_live_admin",
    )
    _clear_live_session_persistence_error(config)


def _maybe_cleanup_expired_live_sessions(ctx, config: LivePageConfig) -> None:
    cleanup_key = f"{config.state_key}_durable_cleanup_done"
    if st.session_state.get(cleanup_key):
        return
    try:
        abandon_expired_live_sessions(ctx.supabase, club_id=str(ctx.club_id))
    except Exception:
        pass
    st.session_state[cleanup_key] = True


def _render_persistence_notice(config: LivePageConfig) -> None:
    message = str(st.session_state.get(f"{config.state_key}_durable_persistence_error") or "").strip()
    if message:
        st.caption(f"Recovery note: {message}")


def _process_payloads(ctx, payloads: list[dict]) -> dict:
    service_ctx = ServiceContext(
        supabase=ctx.supabase,
        club_id=str(ctx.club_id),
        actor_email=getattr(ctx, "admin_email", None),
        actor_role=st.session_state.get("admin_role"),
        source="jupr_live_admin",
        public_base_url=st.session_state.get("public_base_url"),
    )
    result = submit_match_batch(
        service_ctx,
        payloads,
        name_to_id=ctx.name_to_id,
        df_players_all=ctx.df_players_all,
        df_leagues=ctx.df_leagues,
        df_meta=ctx.df_meta,
    )
    if not result.ok:
        raise ValueError("; ".join(result.errors) or "Unable to save matches.")
    return result.data


def _save_rr_official(ctx, state: dict, event: dict) -> None:
    if "rr" in set(state.get("last_saved_rounds") or []):
        st.info("Official round robin results were already saved in this session.")
        return
    payloads = build_rr_official_payloads(state, event)
    if not payloads:
        st.warning("Enter at least one scored match before saving officially.")
        return
    res = _process_payloads(ctx, payloads)
    state["last_saved_rounds"] = ["rr"]
    event["saved_rounds"] = ["rr"]
    clear_expired_substitutions(event)
    st.session_state["force_data_refresh"] = True
    st.success(f"Official results saved ({res['inserted']} matches).")


def _save_league_round_official(ctx, state: dict, event: dict) -> bool:
    current_round_number = int(event.get("currentRoundNumber") or 1)
    saved_rounds = set(state.get("last_saved_rounds") or [])
    if current_round_number in saved_rounds:
        return True
    payloads = build_league_round_official_payloads(state, event)
    if not payloads:
        st.warning("Enter at least one scored match before finalizing this round.")
        return False
    res = _process_payloads(ctx, payloads)
    saved_rounds.add(current_round_number)
    state["last_saved_rounds"] = sorted(saved_rounds)
    event["saved_rounds"] = sorted(saved_rounds)
    clear_expired_substitutions(event)
    st.session_state["force_data_refresh"] = True
    st.success(
        f"Official round {current_round_number} saved ({res['inserted']} matches)."
    )
    return True


def _save_tournament_official(ctx, state: dict, event: dict) -> None:
    payloads = build_tournament_official_payloads(state, event)
    if not payloads:
        st.info("No newly completed tournament matches to save yet.")
        return
    res = _process_payloads(ctx, payloads)
    mark_tournament_payloads_saved(event, payloads)
    st.session_state["force_data_refresh"] = True
    st.success(f"Official tournament results saved ({res['inserted']} matches).")


def render(ctx):
    page_shell(
        "🔴 JUPR Live Admin",
        "Run JUPR Live events as rated overall matches or unrated recorded games. No league/session setup required.",
        mode_label="Admin",
    )
    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.warning("Admin login required.")
        return

    live_session_key = _resolve_live_session_key(ADMIN_CONFIG)
    _restore_live_session_if_needed(ctx, ADMIN_CONFIG, live_session_key)
    render_live_page(
        ctx,
        ADMIN_CONFIG,
        on_save_rr=_save_rr_official,
        on_save_league=_save_league_round_official,
        on_save_tournament=_save_tournament_official,
    )
    try:
        _persist_live_session_state(ctx, ADMIN_CONFIG, live_session_key)
        _maybe_cleanup_expired_live_sessions(ctx, ADMIN_CONFIG)
    except Exception as exc:
        _record_live_session_persistence_error(ADMIN_CONFIG, exc)
    _render_persistence_notice(ADMIN_CONFIG)
